from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time
import cv2
import numpy as np

from real_robot_data_collection import red_ring_and_blue_tip_tracker as red_tracker

# Exact constants from collect_learm_demonstrations.py
BLUE_H_LOW = 105
BLUE_H_HIGH = 116
BLUE_S_MIN = 80
BLUE_V_MIN = 45
BLUE_CLOSE_KERNEL = 5
BLUE_OPEN_KERNEL = 3
BLUE_MIN_AREA = 60
BLUE_MAX_AREA = 30000
BLUE_SEARCH_RADIUS_PX = 180
BLUE_POSITION_ALPHA = 0.70
BLUE_AREA_ALPHA = 0.35
BLUE_PREDICT_HOLD_S = 0.20
BLUE_EPISODE_GATE_RADIUS_PX = 45.0
BLUE_EPISODE_AREA_RATIO_MIN = 0.25
BLUE_EPISODE_AREA_RATIO_MAX = 2.00
PHASE_ARM_VISIBLE_S = 0.20
PHASE_PIERCE_LOST_S = 0.25


def build_red_mask_from_hsv(hsv, cfg):
    mask1 = cv2.inRange(
        hsv,
        np.array([0, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([cfg.h_low_end, 255, 255], dtype=np.uint8),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([cfg.h_high_start, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(mask1, mask2)
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if cfg.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def build_blue_mask_from_hsv(hsv):
    mask = cv2.inRange(
        hsv,
        np.array([BLUE_H_LOW, BLUE_S_MIN, BLUE_V_MIN], dtype=np.uint8),
        np.array([BLUE_H_HIGH, 255, 255], dtype=np.uint8),
    )
    if BLUE_CLOSE_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLUE_CLOSE_KERNEL, BLUE_CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if BLUE_OPEN_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLUE_OPEN_KERNEL, BLUE_OPEN_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def detect_blue_marker(mask, previous_center=None, fixed_anchor_center=None, fixed_anchor_area=None):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    spatial_rejects = 0
    area_ratio_rejects = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < BLUE_MIN_AREA or area > BLUE_MAX_AREA:
            continue
        center = np.asarray(centroids[label], dtype=np.float32)
        distance = None
        area_ratio = math.nan
        if fixed_anchor_center is not None:
            distance = float(np.linalg.norm(center - fixed_anchor_center))
            if distance > BLUE_EPISODE_GATE_RADIUS_PX:
                spatial_rejects += 1
                continue
            if fixed_anchor_area is not None and fixed_anchor_area > 0:
                area_ratio = float(area) / float(fixed_anchor_area)
                if not (BLUE_EPISODE_AREA_RATIO_MIN <= area_ratio <= BLUE_EPISODE_AREA_RATIO_MAX):
                    area_ratio_rejects += 1
                    continue
        elif previous_center is not None:
            distance = float(np.linalg.norm(center - previous_center))
            if distance > BLUE_SEARCH_RADIUS_PX:
                spatial_rejects += 1
                continue
        candidates.append({
            'center': center,
            'area': area,
            'bbox': (
                int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]),
            ),
            'distance': distance,
            'area_ratio': area_ratio,
        })
    if not candidates:
        reject = 'NO_BLOB'
        if fixed_anchor_center is not None:
            if area_ratio_rejects:
                reject = 'ANCHOR_AREA_GATE'
            elif spatial_rejects:
                reject = 'ANCHOR_POSITION_GATE'
            else:
                reject = 'ANCHOR_NO_BLUE'
        return None, {
            'candidates': 0, 'reject': reject,
            'spatial_rejects': spatial_rejects, 'area_ratio_rejects': area_ratio_rejects,
            'distance_px': math.nan, 'area_ratio': math.nan,
        }
    best = max(candidates, key=lambda c: c['area'])
    return best, {
        'candidates': len(candidates), 'reject': '', 'raw_area': best['area'],
        'spatial_rejects': spatial_rejects, 'area_ratio_rejects': area_ratio_rejects,
        'distance_px': math.nan if best['distance'] is None else float(best['distance']),
        'area_ratio': math.nan if not math.isfinite(best['area_ratio']) else float(best['area_ratio']),
    }


class BlueMarkerTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.center = None
        self.area_smooth = None
        self.last_measure_t = None

    def is_locked(self, now):
        return self.center is not None and self.last_measure_t is not None and now - self.last_measure_t <= BLUE_PREDICT_HOLD_S

    def update(self, detection, now):
        measured = detection is not None
        stale = self.last_measure_t is None or now - self.last_measure_t > BLUE_PREDICT_HOLD_S
        if measured:
            z = detection['center']
            raw_area = float(detection['area'])
            if self.center is None or stale:
                self.center = z.copy()
                self.area_smooth = raw_area
            else:
                self.center = (1.0 - BLUE_POSITION_ALPHA) * self.center + BLUE_POSITION_ALPHA * z
                self.area_smooth = (1.0 - BLUE_AREA_ALPHA) * self.area_smooth + BLUE_AREA_ALPHA * raw_area
            self.last_measure_t = now
            return {
                'valid': True, 'measured': True, 'center': self.center.copy(),
                'raw_center': z.copy(), 'raw_area': raw_area, 'area_smooth': float(self.area_smooth),
            }
        return {
            'valid': False, 'measured': False, 'center': None, 'raw_center': None,
            'raw_area': 0.0, 'area_smooth': 0.0 if self.area_smooth is None else float(self.area_smooth),
        }


@dataclass
class RealVisionResult:
    red_valid: bool
    red_measured: bool
    red_u_px: float
    red_v_px: float
    red_du_px: float
    red_dv_px: float
    red_major_px: float
    red_minor_px: float
    red_angle_deg: float
    red_source: str
    red_reject: str
    blue_valid: bool
    blue_measured: bool
    blue_u_px: float
    blue_v_px: float
    blue_raw_area: float
    blue_reject: str
    blue_anchor_locked: bool
    blue_lost_s: float
    phase_armed: bool
    phase_pierced: bool
    red_mask: np.ndarray
    blue_mask: np.ndarray


class RealLeArmVisionPipeline:
    """Run the same red/blue semantics used during LeArm data collection on sim RGB."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = red_tracker.ConfigStore(str(self.config_path)).load()
        # Match the frozen values in the collection script even if another JSON is supplied.
        self.cfg.h_low_end = 7
        self.cfg.h_high_start = 173
        self.cfg.s_min = 160
        self.cfg.v_min = 90
        self.red_tracker = red_tracker.Tracker()
        self.blue_tracker = BlueMarkerTracker()
        self.blue_anchor_center = None
        self.blue_anchor_area = None
        self.phase_last_t = None
        self.blue_visible_run_s = 0.0
        self.blue_lost_s = 0.0
        self.phase_armed = False
        self.phase_pierced = False

    def reset(self):
        self.red_tracker.reset()
        self.blue_tracker.reset()
        self.blue_anchor_center = None
        self.blue_anchor_area = None
        self.phase_last_t = None
        self.blue_visible_run_s = 0.0
        self.blue_lost_s = 0.0
        self.phase_armed = False
        self.phase_pierced = False

    def _update_phase(self, blue_measured: bool, now: float):
        if self.phase_last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, min(now - self.phase_last_t, 0.10))
        self.phase_last_t = now
        if self.phase_pierced:
            if blue_measured:
                self.blue_lost_s = 0.0
            else:
                self.blue_lost_s += dt
            return
        if blue_measured:
            self.blue_visible_run_s += dt
            self.blue_lost_s = 0.0
            if self.blue_visible_run_s >= PHASE_ARM_VISIBLE_S:
                self.phase_armed = True
        else:
            self.blue_visible_run_s = 0.0
            if self.phase_armed:
                self.blue_lost_s += dt
                if self.blue_lost_s >= PHASE_PIERCE_LOST_S:
                    self.phase_pierced = True
            else:
                self.blue_lost_s = 0.0

    def process(self, rgb: np.ndarray, now: float | None = None) -> RealVisionResult:
        now = time.perf_counter() if now is None else float(now)
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        red_mask = build_red_mask_from_hsv(hsv, self.cfg)
        locked = self.red_tracker.is_locked(now, self.cfg)
        predicted = self.red_tracker.predicted_center(now) if locked else None
        previous_major = self.red_tracker.previous_major() if locked else None
        detection, red_diag = red_tracker.detect_ring(
            red_mask, self.cfg, predicted_center=predicted, previous_major=previous_major
        )
        center, red_valid, red_measured = self.red_tracker.update(detection, now, self.cfg)
        if center is None:
            ru = rv = rdu = rdv = rmajor = rminor = rangle = math.nan
        else:
            ru, rv = map(float, center)
            rdu, rdv = ru - 0.5 * w, rv - 0.5 * h
            ellipse = detection['outer_ellipse'] if detection is not None else self.red_tracker.outer_ellipse
            if ellipse is None:
                rmajor = rminor = rangle = math.nan
            else:
                (_, _), (rmajor, rminor), rangle = red_tracker.canonical_ellipse(ellipse)
        red_source = '' if detection is None else str(detection.get('source', ''))

        blue_mask = build_blue_mask_from_hsv(hsv)
        previous_blue = self.blue_tracker.center if self.blue_tracker.is_locked(now) else None
        blue_det, blue_diag = detect_blue_marker(
            blue_mask,
            previous_center=previous_blue if self.blue_anchor_center is None else None,
            fixed_anchor_center=self.blue_anchor_center,
            fixed_anchor_area=self.blue_anchor_area,
        )
        blue_res = self.blue_tracker.update(blue_det, now)
        if self.blue_anchor_center is None and blue_res['measured']:
            # Equivalent to freezing the rigid marker's episode-local image anchor.
            self.blue_anchor_center = blue_res['raw_center'].copy()
            self.blue_anchor_area = float(blue_res['raw_area'])
        if blue_res['measured']:
            bu, bv = map(float, blue_res['center'])
        else:
            bu = bv = math.nan
        self._update_phase(bool(blue_res['measured']), now)

        return RealVisionResult(
            red_valid=bool(red_valid), red_measured=bool(red_measured),
            red_u_px=ru, red_v_px=rv, red_du_px=rdu, red_dv_px=rdv,
            red_major_px=float(rmajor), red_minor_px=float(rminor), red_angle_deg=float(rangle),
            red_source=red_source, red_reject=str(red_diag.get('reject', '')),
            blue_valid=bool(blue_res['valid']), blue_measured=bool(blue_res['measured']),
            blue_u_px=bu, blue_v_px=bv, blue_raw_area=float(blue_res['raw_area']),
            blue_reject=str(blue_diag.get('reject', '')),
            blue_anchor_locked=self.blue_anchor_center is not None,
            blue_lost_s=float(self.blue_lost_s), phase_armed=bool(self.phase_armed), phase_pierced=bool(self.phase_pierced),
            red_mask=red_mask, blue_mask=blue_mask,
        )

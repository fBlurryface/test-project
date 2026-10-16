from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class VisionObservation:
    target_valid: bool
    target_measured: bool
    target_u_px: float
    target_v_px: float
    target_du_px: float
    target_dv_px: float
    target_major_px: float
    target_minor_px: float
    target_axis_ratio: float
    blue_measured: bool
    blue_u_px: float
    blue_v_px: float


def _largest_ellipse(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= 30 and len(c) >= 5]
    if not contours:
        return None
    pts = np.vstack(contours)
    if len(pts) < 5:
        return None
    e = cv2.fitEllipse(pts)
    (cx, cy), (w, h), angle = e
    major, minor = (max(w, h), min(w, h))
    return float(cx), float(cy), float(major), float(minor), float(angle)


def _center_nearest_blue(mask: np.ndarray):
    """Pick the blue component nearest the image center, not the largest one.

    Menagerie Panda contains a stock light-blue visual material, so 'largest blue blob'
    is not a safe marker detector even in simulation.
    """
    h, w = mask.shape[:2]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return False, float("nan"), float("nan")
    center = np.array([0.5 * w, 0.5 * h], dtype=float)
    candidates = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 15 or area > 20000:
            continue
        c = np.asarray(centroids[idx], dtype=float)
        dist = float(np.linalg.norm(c - center))
        candidates.append((dist, -area, c))
    if not candidates:
        return False, float("nan"), float("nan")
    _, _, c = min(candidates, key=lambda x: (x[0], x[1]))
    return True, float(c[0]), float(c[1])



def debug_blue_components(rgb: np.ndarray):
    """Return the rendered blue mask and component diagnostics for acceptance tests."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([105, 120, 40]), np.array([135, 255, 255]))
    h, w = mask.shape[:2]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    center = np.array([0.5 * w, 0.5 * h], dtype=float)
    comps = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        c = np.asarray(centroids[idx], dtype=float)
        comps.append({
            "area_px": area,
            "u_px": float(c[0]),
            "v_px": float(c[1]),
            "center_error_px": float(np.linalg.norm(c - center)),
        })
    comps.sort(key=lambda x: x["center_error_px"])
    return mask, comps

def extract_sim_observation(rgb: np.ndarray) -> VisionObservation:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ellipse = _largest_ellipse(red)

    if ellipse is None:
        target_valid = False
        u = v = du = dv = major = minor = axis = float("nan")
    else:
        u, v, major, minor, _ = ellipse
        du = u - 0.5 * w
        dv = v - 0.5 * h
        axis = minor / max(major, 1e-9)
        target_valid = True

    # The marker is deliberately dark/saturated blue. The narrower hue band rejects
    # the stock Panda light-blue cosmetic material, and center-nearest selection adds
    # another geometric safeguard consistent with the real rigid marker setup.
    blue = cv2.inRange(hsv, np.array([105, 120, 40]), np.array([135, 255, 255]))
    blue_measured, bu, bv = _center_nearest_blue(blue)

    return VisionObservation(
        target_valid=target_valid,
        target_measured=target_valid,
        target_u_px=u,
        target_v_px=v,
        target_du_px=du,
        target_dv_px=dv,
        target_major_px=major,
        target_minor_px=minor,
        target_axis_ratio=axis,
        blue_measured=blue_measured,
        blue_u_px=bu,
        blue_v_px=bv,
    )

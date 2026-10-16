import argparse
import cv2
import numpy as np
import os
import json
import time
import traceback
import faulthandler
from dataclasses import dataclass, asdict, fields


# ============================================================
# Red-ring and blue-tip visual tracker
#
# Design:
#   - only ONE ring exists in the scene
#   - complete and occluded ring use the SAME acquisition path
#   - all meaningful red pieces are merged
#   - convex hull + ellipse gives the ring outer center
#   - if a closed inner hole exists, use it only to REFINE center
#     (it is NOT required for recognition)
#
# No fill-ratio / coverage hard rejection.
#
# Tune:
#   python red_ring_and_blue_tip_tracker.py --tune
#
# Normal:
#   python red_ring_and_blue_tip_tracker.py
# ============================================================


@dataclass
class Config:
    camera_index: int = 1
    width: int = 1280
    height: int = 720
    fps: int = 60

    # Your latest tuned HSV values.
    h_low_end: int = 5
    h_high_start: int = 90
    s_min: int = 168
    v_min: int = 143

    # Morphology.
    close_kernel: int = 7
    open_kernel: int = 9

    # Each visible red piece smaller than this is ignored.
    min_piece_area: int = 120

    # Total visible red area required for target acquisition.
    min_total_area: int = 1000

    # Allow fairly strong perspective.
    min_axis_ratio: float = 0.20

    # When tracking, ignore red pieces very far from the predicted target.
    # Radius = previous outer major axis * roi_scale.
    roi_scale: float = 1.80

    # Closed-hole refinement.
    use_hole_refine: int = 1
    min_hole_area: int = 120
    hole_center_gate: float = 0.40

    # Temporal smoothing.
    position_alpha: float = 0.72
    velocity_alpha: float = 0.20

    # Brief no-measurement prediction.
    predict_hold_s: float = 0.60


def sanitize(cfg):
    cfg.camera_index = max(0, int(cfg.camera_index))
    cfg.width = max(160, int(cfg.width))
    cfg.height = max(120, int(cfg.height))
    cfg.fps = max(1, int(cfg.fps))

    cfg.h_low_end = int(np.clip(cfg.h_low_end, 0, 30))
    cfg.h_high_start = int(np.clip(cfg.h_high_start, 60, 179))
    cfg.s_min = int(np.clip(cfg.s_min, 0, 255))
    cfg.v_min = int(np.clip(cfg.v_min, 0, 255))

    for name in ("close_kernel", "open_kernel"):
        k = max(1, int(getattr(cfg, name)))
        if k > 1 and k % 2 == 0:
            k += 1
        setattr(cfg, name, min(k, 31))

    cfg.min_piece_area = max(1, int(cfg.min_piece_area))
    cfg.min_total_area = max(20, int(cfg.min_total_area))
    cfg.min_axis_ratio = float(np.clip(cfg.min_axis_ratio, 0.05, 1.0))
    cfg.roi_scale = float(np.clip(cfg.roi_scale, 0.5, 5.0))

    cfg.use_hole_refine = int(bool(cfg.use_hole_refine))
    cfg.min_hole_area = max(20, int(cfg.min_hole_area))
    cfg.hole_center_gate = float(np.clip(cfg.hole_center_gate, 0.05, 1.5))

    cfg.position_alpha = float(np.clip(cfg.position_alpha, 0.01, 1.0))
    cfg.velocity_alpha = float(np.clip(cfg.velocity_alpha, 0.0, 1.0))
    cfg.predict_hold_s = float(np.clip(cfg.predict_hold_s, 0.0, 5.0))


class ConfigStore:
    def __init__(self, path):
        self.path = os.path.abspath(path)

    def load(self):
        cfg = Config()

        if not os.path.exists(self.path):
            return cfg

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            valid = {f.name for f in fields(Config)}

            for k, v in raw.items():
                if k in valid:
                    setattr(cfg, k, v)

            sanitize(cfg)
            print(f"[CONFIG] loaded: {self.path}")
            return cfg

        except Exception as exc:
            print(f"[CONFIG] load failed: {exc}")
            return cfg

    def save(self, cfg):
        sanitize(cfg)

        folder = os.path.dirname(self.path) or "."
        os.makedirs(folder, exist_ok=True)

        tmp = self.path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                asdict(cfg),
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True
            )

        os.replace(tmp, self.path)


class TuneUI:
    WINDOW = "Tune Parameters"

    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.initializing = True

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 640, 650)

        self.add("H low end", cfg.h_low_end, 30,
                 lambda v: setattr(cfg, "h_low_end", v))

        self.add("H high start", cfg.h_high_start, 179,
                 lambda v: setattr(cfg, "h_high_start", v))

        self.add("S min", cfg.s_min, 255,
                 lambda v: setattr(cfg, "s_min", v))

        self.add("V min", cfg.v_min, 255,
                 lambda v: setattr(cfg, "v_min", v))

        self.add(
            "Close level",
            0 if cfg.close_kernel <= 1 else (cfg.close_kernel - 1) // 2,
            15,
            lambda v: setattr(cfg, "close_kernel", 1 if v == 0 else 2 * v + 1)
        )

        self.add(
            "Open level",
            0 if cfg.open_kernel <= 1 else (cfg.open_kernel - 1) // 2,
            15,
            lambda v: setattr(cfg, "open_kernel", 1 if v == 0 else 2 * v + 1)
        )

        self.add(
            "Min piece area",
            cfg.min_piece_area,
            3000,
            lambda v: setattr(cfg, "min_piece_area", max(1, v))
        )

        self.add(
            "Min total x100",
            max(1, cfg.min_total_area // 100),
            300,
            lambda v: setattr(cfg, "min_total_area", max(20, v * 100))
        )

        self.add(
            "Min axis ratio %",
            int(round(cfg.min_axis_ratio * 100)),
            100,
            lambda v: setattr(cfg, "min_axis_ratio", max(0.05, v / 100.0))
        )

        self.add(
            "ROI scale %",
            int(round(cfg.roi_scale * 100)),
            500,
            lambda v: setattr(cfg, "roi_scale", max(0.5, v / 100.0))
        )

        self.add(
            "Hole refine",
            cfg.use_hole_refine,
            1,
            lambda v: setattr(cfg, "use_hole_refine", int(v))
        )

        self.add(
            "Min hole area",
            cfg.min_hole_area,
            5000,
            lambda v: setattr(cfg, "min_hole_area", max(20, v))
        )

        self.add(
            "Hole center gate %",
            int(round(cfg.hole_center_gate * 100)),
            150,
            lambda v: setattr(cfg, "hole_center_gate", max(0.05, v / 100.0))
        )

        self.add(
            "Position alpha %",
            int(round(cfg.position_alpha * 100)),
            100,
            lambda v: setattr(cfg, "position_alpha", max(0.01, v / 100.0))
        )

        self.add(
            "Predict hold ms",
            int(round(cfg.predict_hold_s * 1000)),
            5000,
            lambda v: setattr(cfg, "predict_hold_s", v / 1000.0)
        )

        self.initializing = False
        self.store.save(cfg)

    def add(self, name, initial, maximum, setter):
        initial = int(np.clip(initial, 0, maximum))

        def callback(v):
            setter(v)
            sanitize(self.cfg)

            if not self.initializing:
                self.store.save(self.cfg)
                print(f"[CONFIG] saved: {name}={v}")

        cv2.createTrackbar(
            name,
            self.WINDOW,
            initial,
            maximum,
            callback
        )


def build_mask(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([0, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([cfg.h_low_end, 255, 255], dtype=np.uint8)
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([cfg.h_high_start, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    mask = cv2.bitwise_or(mask1, mask2)

    if cfg.close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.close_kernel, cfg.close_kernel)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

    if cfg.open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.open_kernel, cfg.open_kernel)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

    return mask


def canonical_ellipse(e):
    (cx, cy), (w, h), angle = e

    w = float(w)
    h = float(h)
    angle = float(angle)

    if w < h:
        w, h = h, w
        angle += 90.0

    return (
        (float(cx), float(cy)),
        (w, h),
        angle % 180.0
    )


def ellipse_is_sane(ellipse, image_shape=None):
    if ellipse is None:
        return False

    try:
        (cx, cy), (major, minor), angle = canonical_ellipse(ellipse)

        values = np.array(
            [cx, cy, major, minor, angle],
            dtype=np.float64
        )

        if not np.all(np.isfinite(values)):
            return False

        if major <= 2.0 or minor <= 2.0:
            return False

        if major > 10000.0 or minor > 10000.0:
            return False

        if image_shape is not None:
            h, w = image_shape[:2]
            limit = float(max(w, h)) * 4.0

            if major > limit or minor > limit:
                return False

            # Allow an ellipse center to sit somewhat outside the image under
            # heavy occlusion, but reject obviously corrupted coordinates.
            if not (-2 * w <= cx <= 3 * w):
                return False

            if not (-2 * h <= cy <= 3 * h):
                return False

        return True

    except Exception:
        return False


def fit_ellipse_safe(points, image_shape=None):
    arr = np.asarray(points, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr.reshape(-1, 1, 2)

    if len(arr) < 5:
        return None

    # AMS is usually good, but can occasionally become numerically unstable
    # on nearly-degenerate point sets. Validate it before accepting it.
    try:
        candidate = canonical_ellipse(
            cv2.fitEllipseAMS(arr)
        )

        if ellipse_is_sane(candidate, image_shape):
            return candidate

    except cv2.error:
        pass

    # Fallback to the standard direct least-squares ellipse.
    try:
        candidate = canonical_ellipse(
            cv2.fitEllipse(arr)
        )

        if ellipse_is_sane(candidate, image_shape):
            return candidate

    except cv2.error:
        pass

    return None


def collect_valid_components(mask, cfg, predicted_center=None, previous_major=None):
    """
    Find visible red pieces without allocating a full-frame temporary mask
    for every component.

    v6 used connectedComponentsWithStats() and then created one 1280x720
    piece_mask per component per frame. With many red fragments that caused
    unnecessary memory churn. External contours give us the same information
    needed by this tracker much more cheaply.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    pieces = []

    roi_radius = None

    if (
        predicted_center is not None
        and previous_major is not None
    ):
        roi_radius = max(
            80.0,
            previous_major * cfg.roi_scale
        )

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < cfg.min_piece_area:
            continue

        if len(contour) < 2:
            continue

        moments = cv2.moments(contour)

        if abs(moments["m00"]) > 1e-6:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + 0.5 * w
            cy = y + 0.5 * h

        if roi_radius is not None:
            dist = float(
                np.linalg.norm(
                    np.array(
                        [cx, cy],
                        dtype=np.float32
                    )
                    - predicted_center
                )
            )

            if dist > roi_radius:
                continue

        pieces.append(
            {
                "area": area,
                "contour": contour,
            }
        )

    # Largest pieces first. This also bounds worst-case work if the mask
    # contains many red speckles.
    pieces.sort(
        key=lambda p: p["area"],
        reverse=True
    )

    return pieces[:32]


def find_closed_hole(mask, outer_center, outer_major, cfg):
    if not cfg.use_hole_refine:
        return None

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:
        return None

    best = None

    for i, contour in enumerate(contours):
        parent = hierarchy[0][i][3]

        if parent == -1:
            continue

        area = cv2.contourArea(contour)

        if area < cfg.min_hole_area:
            continue

        ellipse = fit_ellipse_safe(contour, mask.shape)

        if ellipse is None:
            continue

        hc = np.array(
            ellipse[0],
            dtype=np.float32
        )

        dist = float(
            np.linalg.norm(
                hc - outer_center
            )
        )

        gate = max(
            20.0,
            outer_major * cfg.hole_center_gate
        )

        if dist > gate:
            continue

        if best is None or area > best["area"]:
            best = {
                "area": area,
                "ellipse": ellipse,
                "center": hc,
                "contour": contour,
            }

    return best


def detect_ring(mask, cfg, predicted_center=None, previous_major=None):
    pieces = collect_valid_components(
        mask,
        cfg,
        predicted_center,
        previous_major
    )

    total_area = sum(
        p["area"]
        for p in pieces
    )

    diagnostics = {
        "pieces": len(pieces),
        "total_area": total_area,
        "reject": "",
    }

    if total_area < cfg.min_total_area:
        diagnostics["reject"] = "AREA"
        return None, diagnostics

    all_points = []

    for piece in pieces:
        pts = piece["contour"].reshape(-1, 2)
        all_points.append(pts)

    if not all_points:
        diagnostics["reject"] = "NO_POINTS"
        return None, diagnostics

    points = np.vstack(all_points).astype(np.float32)

    if len(points) < 5:
        diagnostics["reject"] = "FEW_POINTS"
        return None, diagnostics

    # Convex hull naturally bridges the missing sector caused by an occluder.
    hull = cv2.convexHull(
        points.reshape(-1, 1, 2)
    )

    outer_ellipse = fit_ellipse_safe(hull, mask.shape)

    if outer_ellipse is None:
        diagnostics["reject"] = "ELLIPSE"
        return None, diagnostics

    (cx, cy), (major, minor), angle = outer_ellipse

    axis_ratio = (
        minor / major
        if major > 0
        else 0.0
    )

    diagnostics["axis_ratio"] = axis_ratio
    diagnostics["major"] = major
    diagnostics["minor"] = minor

    if axis_ratio < cfg.min_axis_ratio:
        diagnostics["reject"] = "AXIS"
        return None, diagnostics

    outer_center = np.array(
        [cx, cy],
        dtype=np.float32
    )

    # Recognition succeeds here. Closed hole only improves center precision.
    center = outer_center.copy()
    source = "OUTER"
    hole = find_closed_hole(
        mask,
        outer_center,
        major,
        cfg
    )

    if hole is not None:
        center = hole["center"].copy()
        source = "HOLE"

    return {
        "center": center,
        "outer_center": outer_center,
        "outer_ellipse": outer_ellipse,
        "hull": hull,
        "pieces": pieces,
        "total_area": total_area,
        "axis_ratio": axis_ratio,
        "source": source,
        "hole": hole,
    }, diagnostics


class Tracker:
    def __init__(self):
        self.center = None
        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )
        self.outer_ellipse = None
        self.last_t = None
        self.last_measure_t = None

    def reset(self):
        self.__init__()

    def is_locked(self, now, cfg):
        # The previous target location is trustworthy only for a short time.
        # After predict_hold_s expires, release all spatial gating and do a
        # full-frame reacquisition search.
        return (
            self.center is not None
            and self.last_measure_t is not None
            and (
                now - self.last_measure_t
                <= cfg.predict_hold_s
            )
        )

    def predicted_center(self, now):
        if self.center is None:
            return None

        if self.last_t is None:
            return self.center.copy()

        dt = float(
            np.clip(
                now - self.last_t,
                0.0,
                0.20
            )
        )

        return (
            self.center
            + self.velocity * dt
        )

    def previous_major(self):
        if self.outer_ellipse is None:
            return None

        return float(
            canonical_ellipse(
                self.outer_ellipse
            )[1][0]
        )

    def update(self, detection, now, cfg):
        """
        Reacquisition-aware temporal update.

        Once the target has been absent longer than predict_hold_s, an old
        center/velocity must not constrain a newly found target. The next
        detection is treated as a fresh acquisition.
        """
        stale = (
            self.last_measure_t is None
            or (
                now - self.last_measure_t
                > cfg.predict_hold_s
            )
        )

        # First acquisition or reacquisition after a genuine loss.
        if detection is not None and (
            self.center is None
            or stale
        ):
            self.center = detection[
                "center"
            ].copy()

            self.outer_ellipse = detection[
                "outer_ellipse"
            ]

            self.velocity[:] = 0.0
            self.last_t = now
            self.last_measure_t = now

            return (
                self.center.copy(),
                True,
                True
            )

        if self.center is None:
            return None, False, False

        dt = float(
            np.clip(
                now - self.last_t,
                1e-3,
                0.20
            )
        )

        predicted = (
            self.center
            + self.velocity * dt
        )

        old_center = self.center.copy()

        measured = (
            detection is not None
        )

        if measured:
            z = detection[
                "center"
            ]

            self.center = (
                (1.0 - cfg.position_alpha)
                * predicted
                + cfg.position_alpha
                * z
            )

            v_measured = (
                self.center
                - old_center
            ) / dt

            self.velocity = (
                (1.0 - cfg.velocity_alpha)
                * self.velocity
                + cfg.velocity_alpha
                * v_measured
            )

            self.outer_ellipse = detection[
                "outer_ellipse"
            ]

            self.last_measure_t = now

        else:
            if not stale:
                # Only predict while the lock is still considered recent.
                self.center = predicted
                self.velocity *= 0.94
            else:
                # After a true loss, stop stale motion completely.
                self.velocity[:] = 0.0

        self.last_t = now

        valid = (
            self.last_measure_t is not None
            and (
                now - self.last_measure_t
                <= cfg.predict_hold_s
            )
        )

        return (
            self.center.copy(),
            valid,
            measured
        )


def open_camera(cfg):
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("AUTO", cv2.CAP_ANY),
    ]

    for name, backend in backends:
        print(
            f"Trying camera "
            f"{cfg.camera_index} "
            f"with {name}..."
        )

        cap = cv2.VideoCapture(
            cfg.camera_index,
            backend
        )

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()

        if ok and frame is not None:
            print(
                f"[OK] {frame.shape[1]}x"
                f"{frame.shape[0]}"
            )
            return cap

        cap.release()

    return None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stable single red-ring tracker with persistent tuning."
        )
    )

    p.add_argument(
        "--tune",
        action="store_true"
    )

    p.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_tracker_config.json")
    )

    p.add_argument(
        "--camera",
        type=int,
        default=None
    )

    p.add_argument(
        "--reset-config",
        action="store_true"
    )

    p.add_argument(
        "--log",
        default="red_ring_tracker_crash.log",
        help="Crash/error log file."
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Keep a persistent log handle alive for faulthandler. On Windows this
    # can also capture useful information for native access violations.
    fault_log = open(
        args.log,
        "a",
        encoding="utf-8",
        buffering=1
    )

    faulthandler.enable(
        file=fault_log,
        all_threads=True
    )

    print(f"[LOG] crash/error log: {os.path.abspath(args.log)}")

    store = ConfigStore(
        args.config
    )

    if args.reset_config:
        cfg = Config()
        store.save(cfg)
        print(
            f"[CONFIG] reset: {store.path}"
        )
    else:
        cfg = store.load()

    if args.camera is not None:
        cfg.camera_index = args.camera
        store.save(cfg)

    cap = open_camera(cfg)

    if cap is None:
        raise SystemExit(
            f"Could not open camera "
            f"{cfg.camera_index}"
        )

    if args.tune:
        TuneUI(cfg, store)

    tracker = Tracker()

    last_time = time.perf_counter()
    fps_smooth = 0.0

    print("")
    print("Red-ring and blue-tip tracker")
    print("TRACK = measured this frame")
    print("PREDICT = brief measurement loss")
    print("R = reset tracker")
    print("P = print config")
    print("Q / Esc = quit")
    print("")

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            now = time.perf_counter()

            try:
                mask = build_mask(
                    frame,
                    cfg
                )

                # While recently tracked, use a local ROI for stability.
                # Once the target is genuinely lost, drop the ROI entirely
                # and reacquire from the whole frame.
                locked = tracker.is_locked(
                    now,
                    cfg
                )

                if locked:
                    predicted = tracker.predicted_center(
                        now
                    )
                    previous_major = tracker.previous_major()
                else:
                    predicted = None
                    previous_major = None

                detection, diag = detect_ring(
                    mask,
                    cfg,
                    predicted_center=predicted,
                    previous_major=previous_major
                )

                center, valid, measured = tracker.update(
                    detection,
                    now,
                    cfg
                )

            except Exception as exc:
                message = (
                    f"[FRAME ERROR] {type(exc).__name__}: {exc}"
                )

                print(message)

                fault_log.write(
                    "\n"
                    + "=" * 70
                    + "\n"
                    + time.strftime("%Y-%m-%d %H:%M:%S")
                    + " "
                    + message
                    + "\n"
                    + traceback.format_exc()
                    + "\n"
                )

                # A single malformed frame should not terminate the program.
                tracker.reset()
                time.sleep(0.02)
                continue

            display = frame.copy()

            if detection is not None:
                # Draw visible red pieces.
                for piece in detection["pieces"]:
                    piece_contour = piece.get(
                        "contour"
                    )

                    if (
                        piece_contour is not None
                        and len(piece_contour) > 0
                    ):
                        piece_draw = np.asarray(
                            piece_contour,
                            dtype=np.int32
                        )

                        cv2.drawContours(
                            display,
                            [piece_draw],
                            -1,
                            (0, 255, 0),
                            1
                        )

                # Reconstructed outer ellipse.
                cv2.ellipse(
                    display,
                    detection["outer_ellipse"],
                    (255, 0, 255),
                    2
                )

                # Hull used to bridge occluded/missing sectors.
                #
                # OpenCV 5 drawContours() expects integer contour points.
                # The fitting hull is float32 because it is also used by
                # fitEllipse(). Convert a COPY only for drawing.
                hull_draw = np.rint(
                    detection["hull"]
                ).astype(
                    np.int32
                )

                if (
                    hull_draw.ndim >= 2
                    and len(hull_draw) > 0
                ):
                    cv2.drawContours(
                        display,
                        [hull_draw],
                        -1,
                        (255, 255, 0),
                        1
                    )

                if detection["hole"] is not None:
                    cv2.ellipse(
                        display,
                        detection["hole"]["ellipse"],
                        (0, 0, 255),
                        2
                    )

            if valid and center is not None:
                x = int(round(center[0]))
                y = int(round(center[1]))

                color = (
                    (0, 255, 0)
                    if measured
                    else (255, 255, 0)
                )

                cv2.drawMarker(
                    display,
                    (x, y),
                    color,
                    cv2.MARKER_CROSS,
                    34,
                    2
                )

                state = (
                    "TRACK"
                    if measured
                    else "PREDICT"
                )

                source = (
                    detection["source"]
                    if detection is not None
                    else "MEM"
                )

                cv2.putText(
                    display,
                    (
                        f"{state} source={source} "
                        f"center=({center[0]:.1f},"
                        f"{center[1]:.1f})"
                    ),
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    color,
                    2,
                    cv2.LINE_AA
                )

            else:
                cv2.putText(
                    display,
                    "SEARCH / LOST",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Always show rejection diagnostics, even when no detection.
            reject = diag.get(
                "reject",
                ""
            )

            cv2.putText(
                display,
                (
                    f"search={'ROI' if locked else 'GLOBAL'} "
                    f"pieces={diag.get('pieces', 0)} "
                    f"area={diag.get('total_area', 0)} "
                    f"axis={diag.get('axis_ratio', 0.0):.2f} "
                    f"reject={reject or '-'}"
                ),
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            dt = now - last_time
            last_time = now

            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = (
                    fps
                    if fps_smooth == 0
                    else (
                        0.90 * fps_smooth
                        + 0.10 * fps
                    )
                )

            cv2.putText(
                display,
                f"FPS={fps_smooth:.1f} tune={'on' if args.tune else 'off'}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                "Red-ring and blue-tip tracker",
                display
            )

            cv2.imshow(
                "Red Mask",
                mask
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                tracker.reset()
                print("[TRACKER] reset")

            if key == ord("p"):
                print(
                    json.dumps(
                        asdict(cfg),
                        ensure_ascii=False,
                        indent=2
                    )
                )

    finally:
        store.save(cfg)

        print(
            f"[CONFIG] saved: "
            f"{store.path}"
        )

        cap.release()
        cv2.destroyAllWindows()

        try:
            fault_log.flush()
            fault_log.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

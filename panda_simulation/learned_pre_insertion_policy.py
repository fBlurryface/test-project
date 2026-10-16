from __future__ import annotations

from pathlib import Path
import numpy as np


FEATURE_NAMES = (
    "ring_blue_du_norm",
    "ring_blue_dv_norm",
    "ring_major_ratio",
    "ring_minor_ratio",
    "ring_axis_ratio",
)
ACTION_NAMES = ("d_left_m", "d_up_m", "d_forward_m")


class BlueRelativeFeatureBuilder:
    """Build the same 5-D observation used to train the real-data pre-insertion policy.

    The policy is intentionally limited to the pre-blue-loss phase.  Both the red
    ring and blue terminal rod must be actually measured.  Absolute ring pixel size
    is normalized by the first valid frame of the episode to reduce camera/ring-size
    domain shift between LeArm and Panda.
    """

    def __init__(self, image_width: int, image_height: int):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.major0: float | None = None
        self.minor0: float | None = None

    def reset(self) -> None:
        self.major0 = None
        self.minor0 = None

    def build_raw(self, vision) -> np.ndarray | None:
        if vision is None or not (vision.red_measured and vision.blue_measured):
            return None
        values = np.array([
            vision.red_u_px, vision.red_v_px,
            vision.blue_u_px, vision.blue_v_px,
            vision.red_major_px, vision.red_minor_px,
        ], dtype=float)
        if not np.all(np.isfinite(values)):
            return None
        major = float(vision.red_major_px)
        minor = float(vision.red_minor_px)
        if major <= 1.0 or minor <= 1.0:
            return None
        if self.major0 is None or self.minor0 is None:
            self.major0 = major
            self.minor0 = minor
        du = (float(vision.red_u_px) - float(vision.blue_u_px)) / max(0.5 * self.image_width, 1.0)
        dv = (float(vision.red_v_px) - float(vision.blue_v_px)) / max(0.5 * self.image_height, 1.0)
        major_ratio = major / max(self.major0, 1.0)
        minor_ratio = minor / max(self.minor0, 1.0)
        axis_ratio = minor / max(major, 1e-9)
        out = np.array([du, dv, major_ratio, minor_ratio, axis_ratio], dtype=np.float32)
        return out if np.all(np.isfinite(out)) else None


class BlueRelativePreInsertionPolicy:
    """NumPy-only 5->64->64->32->3 MLP exported by behavior_cloning_training/train_blue_relative_pre_insertion_policy.py."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        data = np.load(self.model_path, allow_pickle=False)
        self.obs_mean = data["obs_mean"].astype(np.float32)
        self.obs_std = data["obs_std"].astype(np.float32)
        self.action_mean = data["action_mean"].astype(np.float32)
        self.action_std = data["action_std"].astype(np.float32)
        self.feature_names = tuple(str(x) for x in data["feature_names"].tolist())
        self.action_names = tuple(str(x) for x in data["action_names"].tolist())
        self.policy_dt_s = float(data["policy_dt_s"])
        self.weights = [data[f"w{i}"].astype(np.float32) for i in range(4)]
        self.biases = [data[f"b{i}"].astype(np.float32) for i in range(4)]
        if self.feature_names != FEATURE_NAMES:
            raise ValueError(f"Unexpected feature list: {self.feature_names}")
        if self.action_names != ACTION_NAMES:
            raise ValueError(f"Unexpected action list: {self.action_names}")
        if self.obs_mean.shape != (5,) or self.action_mean.shape != (3,):
            raise ValueError("Unexpected model dimensions")

    def zscore(self, raw_features: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_features, dtype=np.float32).reshape(5)
        return (x - self.obs_mean) / self.obs_std

    def predict_delta(self, raw_features: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(raw_features, dtype=np.float32).reshape(5)
        if not np.all(np.isfinite(x)):
            raise ValueError("Approach observation contains NaN/Inf")
        z = self.zscore(x)
        h = z.copy()
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.T + b
            if i < 3:
                h = np.maximum(h, 0.0)
        action = h * self.action_std + self.action_mean
        return action.astype(np.float32), float(np.max(np.abs(z)))

    def delta_to_twist(self, delta3_m: np.ndarray) -> np.ndarray:
        d = np.asarray(delta3_m, dtype=float).reshape(3)
        twist = np.zeros(6, dtype=float)
        twist[:3] = d / max(self.policy_dt_s, 1e-6)
        return twist

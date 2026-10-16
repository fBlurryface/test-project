from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .task_frames import relative_local_delta
from .simulation_vision import VisionObservation


POLICY_OBS_V1_FEATURES = (
    "ring_du_norm",
    "ring_dv_norm",
    "ring_major_norm",
    "ring_minor_norm",
    "ring_valid",
    "ring_measured",
    "blue_measured",
    "rel_left_m",
    "rel_up_m",
    "rel_forward_m",
    "rel_rot_x_rad",
    "rel_rot_y_rad",
    "rel_rot_z_rad",
)


@dataclass(frozen=True)
class PolicyObservation:
    vector: np.ndarray
    feature_names: tuple[str, ...]
    vision: VisionObservation
    relative_pose6: np.ndarray


class ObservationBuilder:
    """Build the robot-independent policy observation from vision + tool pose.

    The builder deliberately has no access to Panda qpos, Jacobians, actuator forces,
    joint limits or safety state.  Those remain execution-layer diagnostics only.
    """

    def __init__(self, image_width: int, image_height: int):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self._start_p: np.ndarray | None = None
        self._start_R: np.ndarray | None = None

    def reset(self, position_world: np.ndarray, rotation_world_from_tool: np.ndarray) -> None:
        self._start_p = np.asarray(position_world, dtype=float).reshape(3).copy()
        self._start_R = np.asarray(rotation_world_from_tool, dtype=float).reshape(3, 3).copy()

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        v = float(value)
        return v if np.isfinite(v) else 0.0

    def build(
        self,
        vision: VisionObservation,
        position_world: np.ndarray,
        rotation_world_from_tool: np.ndarray,
    ) -> PolicyObservation:
        if self._start_p is None or self._start_R is None:
            raise RuntimeError("ObservationBuilder.reset() must be called at episode start")

        rel6 = relative_local_delta(
            self._start_p,
            self._start_R,
            position_world,
            rotation_world_from_tool,
        )

        # Match the real recorder's image-centre semantics while making the scale
        # independent of the exact image resolution.
        du_norm = self._finite_or_zero(vision.target_du_px) / max(0.5 * self.image_width, 1.0)
        dv_norm = self._finite_or_zero(vision.target_dv_px) / max(0.5 * self.image_height, 1.0)
        major_norm = self._finite_or_zero(vision.target_major_px) / max(float(self.image_width), 1.0)
        minor_norm = self._finite_or_zero(vision.target_minor_px) / max(float(self.image_height), 1.0)

        vec = np.array(
            [
                du_norm,
                dv_norm,
                major_norm,
                minor_norm,
                float(bool(vision.target_valid)),
                float(bool(vision.target_measured)),
                float(bool(vision.blue_measured)),
                *rel6.tolist(),
            ],
            dtype=np.float32,
        )
        if vec.shape != (len(POLICY_OBS_V1_FEATURES),):
            raise AssertionError("Policy observation shape mismatch")
        if not np.all(np.isfinite(vec)):
            raise RuntimeError("Policy observation contains non-finite values")

        return PolicyObservation(
            vector=vec,
            feature_names=POLICY_OBS_V1_FEATURES,
            vision=vision,
            relative_pose6=rel6.copy(),
        )

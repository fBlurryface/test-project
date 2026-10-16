from __future__ import annotations

import math
import numpy as np

from .geometry import matrix_to_rotvec, rotvec_to_matrix


# Canonical shared tool frame used by BOTH LeArm dataset conversion and Panda runtime:
#   +X : tool-left / lateral
#   +Y : tool-up (at the nominal horizontal task pose this is world-up)
#   +Z : tool-forward, collinear with the rod
#
# Local rotation-vector components therefore have the useful nominal interpretation:
#   +rotX : positive LeArm-style pitch (forward axis pitches downward from horizontal)
#   +rotY : positive yaw (forward axis turns left)
#   +rotZ : roll about the rod axis
CANONICAL_ACTION_NAMES = (
    "d_left_m",
    "d_up_m",
    "d_forward_m",
    "d_rot_x_local_rad",
    "d_rot_y_local_rad",
    "d_rot_z_local_rad",
)


def compose_local_delta(
    position_world: np.ndarray,
    rotation_world_from_tool: np.ndarray,
    action6: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a local 6D body-frame delta onto a world-frame tool pose.

    action6 = [dx, dy, dz, drx, dry, drz]
      - translation is expressed in the CURRENT tool frame;
      - rotation is a local/body rotation vector, also expressed in the tool frame.

    This is exactly the runtime meaning expected for LeArm labels generated with
    :func:`relative_local_delta`.
    """
    p = np.asarray(position_world, dtype=float).reshape(3)
    R = np.asarray(rotation_world_from_tool, dtype=float).reshape(3, 3)
    a = np.asarray(action6, dtype=float).reshape(6)
    dp_local = a[:3]
    dw_local = a[3:]
    return p + R @ dp_local, R @ rotvec_to_matrix(dw_local)


def relative_local_delta(
    position_from_world: np.ndarray,
    rotation_world_from_from: np.ndarray,
    position_to_world: np.ndarray,
    rotation_world_from_to: np.ndarray,
) -> np.ndarray:
    """Return the local/body-frame delta taking pose ``from`` to pose ``to``.

    This is the shared label definition for training and Panda policy execution:

        dp_local = R_from.T @ (p_to - p_from)
        dw_local = log(R_from.T @ R_to)

    The returned 6-vector can be passed directly to ``compose_local_delta``.
    """
    p0 = np.asarray(position_from_world, dtype=float).reshape(3)
    R0 = np.asarray(rotation_world_from_from, dtype=float).reshape(3, 3)
    p1 = np.asarray(position_to_world, dtype=float).reshape(3)
    R1 = np.asarray(rotation_world_from_to, dtype=float).reshape(3, 3)
    dp = R0.T @ (p1 - p0)
    dw = matrix_to_rotvec(R0.T @ R1)
    return np.concatenate([dp, dw])


def learm_pose_mm_deg_to_canonical(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a recorded LeArm tool pose into the canonical tool frame.

    This follows the kinematics in ``collect_learm_demonstrations.py``:
    yaw is the base angle, pitch is shoulder+elbow+wrist_pitch, and roll is the
    wrist-roll angle.  The canonical axes are constructed geometrically rather than
    by assuming a generic XYZ Euler convention.

    At zero roll:
      +Z_tool = rod/tool-forward
      +X_tool = lateral-left around the LeArm base
      +Y_tool = Z_tool x X_tool

    Positive roll is applied as a right-handed rotation about +Z_tool.
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    forward = np.array(
        [
            math.sin(pitch) * math.cos(yaw),
            math.sin(pitch) * math.sin(yaw),
            math.cos(pitch),
        ],
        dtype=float,
    )
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    up = np.cross(forward, left)
    up /= max(float(np.linalg.norm(up)), 1e-12)

    R0 = np.column_stack([left, up, forward])
    R = R0 @ rotvec_to_matrix(np.array([0.0, 0.0, roll], dtype=float))
    p = np.array([x_mm, y_mm, z_mm], dtype=float) * 1e-3
    return p, R

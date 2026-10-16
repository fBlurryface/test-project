from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .task_frames import learm_pose_mm_deg_to_canonical, relative_local_delta
from .geometry import matrix_to_rotvec


@dataclass
class LeArmEpisode:
    episode_id: str
    success: bool
    time_s: np.ndarray              # (N,)
    position_m: np.ndarray          # (N,3)
    rotation_world_from_tool: np.ndarray  # (N,3,3)
    source_hz: float

    @property
    def n(self) -> int:
        return int(self.time_s.shape[0])


@dataclass
class ReplayWaypoints:
    episode_id: str
    time_s: np.ndarray
    position_m: np.ndarray
    rotation_world_from_tool: np.ndarray
    actions6: np.ndarray  # action i takes waypoint i -> i+1; shape (M-1,6)
    stride_samples: int
    policy_dt_s: float


def _row_to_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return learm_pose_mm_deg_to_canonical(
        float(row["tool_x_mm"]),
        float(row["tool_y_mm"]),
        float(row["tool_z_mm"]),
        float(row["tool_yaw_deg"]),
        float(row["tool_pitch_deg"]),
        float(row["tool_roll_deg"]),
    )


def _episode_from_bytes(meta_bytes: bytes, traj_bytes: bytes) -> LeArmEpisode:
    meta = json.loads(meta_bytes.decode("utf-8"))
    text = traj_bytes.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{meta.get('episode_id','episode')} has an empty trajectory.csv")

    positions = []
    rotations = []
    times = []
    for row in rows:
        p, R = _row_to_pose(row)
        positions.append(p)
        rotations.append(R)
        times.append(float(row["episode_time_s"]))

    t = np.asarray(times, dtype=float)
    # Episode 1 can start a tiny amount below zero due to host timestamp timing.
    t = t - t[0]
    hz = float(meta.get("actual_sample_hz", meta.get("requested_sample_hz", 50.0)))
    return LeArmEpisode(
        episode_id=str(meta.get("episode_id", "unknown")),
        success=bool(meta.get("success", meta.get("outcome") == "success")),
        time_s=t,
        position_m=np.asarray(positions, dtype=float),
        rotation_world_from_tool=np.asarray(rotations, dtype=float),
        source_hz=hz,
    )


def load_episodes(source: str | Path, *, success_only: bool = True) -> list[LeArmEpisode]:
    """Load LeArm episodes from the original zip, a success-only zip, or an extracted folder."""
    source = Path(source)
    episodes: list[LeArmEpisode] = []

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = set(zf.namelist())
            roots = sorted({n.split("/")[0] for n in names if n.startswith("episode_") and "/" in n})
            for root in roots:
                meta_name = f"{root}/meta.json"
                traj_name = f"{root}/trajectory.csv"
                if meta_name not in names or traj_name not in names:
                    continue
                ep = _episode_from_bytes(zf.read(meta_name), zf.read(traj_name))
                if success_only and not ep.success:
                    continue
                episodes.append(ep)
        return episodes

    if source.is_dir():
        for ep_dir in sorted(source.glob("episode_*")):
            meta_path = ep_dir / "meta.json"
            traj_path = ep_dir / "trajectory.csv"
            if not meta_path.exists() or not traj_path.exists():
                continue
            ep = _episode_from_bytes(meta_path.read_bytes(), traj_path.read_bytes())
            if success_only and not ep.success:
                continue
            episodes.append(ep)
        return episodes

    raise FileNotFoundError(f"Episode source not found or unsupported: {source}")


def make_waypoints(ep: LeArmEpisode, *, stride_samples: int = 5) -> ReplayWaypoints:
    """Downsample a ~50 Hz trajectory to the exact recorded sample stride.

    stride=5 is the intended 100 ms policy horizon for the current dataset.  The final
    source sample is appended if needed so the replay includes the whole demonstration.
    """
    stride = int(stride_samples)
    if stride < 1:
        raise ValueError("stride_samples must be >= 1")
    ids = list(range(0, ep.n, stride))
    if ids[-1] != ep.n - 1:
        ids.append(ep.n - 1)
    ids_arr = np.asarray(ids, dtype=int)

    p = ep.position_m[ids_arr].copy()
    R = ep.rotation_world_from_tool[ids_arr].copy()
    t = ep.time_s[ids_arr].copy()
    actions = []
    for i in range(len(ids) - 1):
        actions.append(relative_local_delta(p[i], R[i], p[i + 1], R[i + 1]))
    action_arr = np.asarray(actions, dtype=float).reshape(-1, 6)
    policy_dt = float(stride / ep.source_hz)
    return ReplayWaypoints(
        episode_id=ep.episode_id,
        time_s=t,
        position_m=p,
        rotation_world_from_tool=R,
        actions6=action_arr,
        stride_samples=stride,
        policy_dt_s=policy_dt,
    )


def map_waypoints_to_panda(
    wp: ReplayWaypoints,
    panda_start_position_m: np.ndarray,
    panda_start_rotation_world_from_tool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map LeArm relative tool motion onto a Panda start pose.

    T_P(t) = T_P(0) * inv(T_L(0)) * T_L(t)

    This preserves the full LeArm relative EE trajectory while leaving Panda joint
    realization entirely to the Panda controller.
    """
    pP0 = np.asarray(panda_start_position_m, dtype=float).reshape(3)
    RP0 = np.asarray(panda_start_rotation_world_from_tool, dtype=float).reshape(3, 3)
    pL0 = wp.position_m[0]
    RL0 = wp.rotation_world_from_tool[0]

    p_out = np.empty_like(wp.position_m)
    R_out = np.empty_like(wp.rotation_world_from_tool)
    for i, (pL, RL) in enumerate(zip(wp.position_m, wp.rotation_world_from_tool)):
        rel_p = RL0.T @ (pL - pL0)
        rel_R = RL0.T @ RL
        p_out[i] = pP0 + RP0 @ rel_p
        R_out[i] = RP0 @ rel_R
    return p_out, R_out


def action_distribution(episodes: list[LeArmEpisode], *, stride_samples: int = 5) -> dict:
    actions = []
    for ep in episodes:
        wp = make_waypoints(ep, stride_samples=stride_samples)
        if len(wp.actions6):
            actions.append(wp.actions6)
    if not actions:
        raise ValueError("No actions found")
    A = np.concatenate(actions, axis=0)
    tn = np.linalg.norm(A[:, :3], axis=1)
    rn = np.linalg.norm(A[:, 3:], axis=1)

    def stats(x: np.ndarray) -> dict:
        return {
            "max": float(np.max(x)),
            "p999": float(np.quantile(x, 0.999)),
            "p99": float(np.quantile(x, 0.99)),
            "p95": float(np.quantile(x, 0.95)),
            "median": float(np.median(x)),
        }

    return {
        "episode_count": len(episodes),
        "action_count": int(A.shape[0]),
        "stride_samples": int(stride_samples),
        "translation_norm_m": stats(tn),
        "rotation_norm_rad": stats(rn),
        "rotation_norm_deg": stats(np.degrees(rn)),
    }


def pose_error(
    target_p: np.ndarray,
    target_R: np.ndarray,
    actual_p: np.ndarray,
    actual_R: np.ndarray,
) -> tuple[float, float]:
    ep = float(np.linalg.norm(np.asarray(target_p) - np.asarray(actual_p)))
    er = float(np.linalg.norm(matrix_to_rotvec(np.asarray(target_R) @ np.asarray(actual_R).T)))
    return ep, er

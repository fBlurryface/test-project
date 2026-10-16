#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import numpy as np
import cv2

from panda_simulation import PandaRingEnv, POLICY_OBS_V1_FEATURES
from panda_simulation.geometry import matrix_to_rotvec
from panda_simulation.task_frames import compose_local_delta, relative_local_delta


def result(name, passed, **metrics):
    return {"name": name, "pass": bool(passed), **metrics}


def main():
    env = PandaRingEnv()
    out = []

    # 0) Pure-math policy action contract must be self-consistent.
    p0 = np.array([0.2, -0.1, 0.4], dtype=float)
    R0 = np.eye(3)
    a0 = np.array([0.004, -0.003, 0.006, 0.03, -0.02, 0.01], dtype=float)
    p1, R1 = compose_local_delta(p0, R0, a0)
    recovered = relative_local_delta(p0, R0, p1, R1)
    contract_err = float(np.max(np.abs(recovered - a0)))
    out.append(result("canonical_action_roundtrip", contract_err < 1e-9,
                      max_abs_error=contract_err))

    # 1) Home must remain stiff/stable under gravity.
    q0 = env.controller._arm_q().copy()
    p0, _ = env.tip_pose()
    env.run_for(2.0)
    q1 = env.controller._arm_q().copy()
    p1, _ = env.tip_pose()
    q_drift = float(np.max(np.abs(q1 - q0)))
    tip_drift = float(np.linalg.norm(p1 - p0))
    out.append(result("home_stability", q_drift < 0.03 and tip_drift < 0.01,
                      max_joint_drift_rad=q_drift, tip_drift_m=tip_drift,
                      actuator_utilization=env.controller.diag.max_actuator_utilization))

    # 2) Policy-facing local translations must move in the requested canonical axis.
    # Use measured-pose composition: this is the exact BC/MLP runtime contract.
    for axis in range(3):
        env.reset()
        p0, R0 = env.tip_pose()
        a = np.zeros(6)
        a[axis] = 0.008
        env.execute_policy_delta(a, 0.30)
        p1, _ = env.tip_pose()
        local_motion = R0.T @ (p1 - p0)
        commanded = float(local_motion[axis])
        motion_norm = float(np.linalg.norm(local_motion))
        alignment_cos = commanded / max(motion_norm, 1e-12)
        out.append(result(
            f"policy_translation_axis_{axis}",
            commanded > 0.004 and alignment_cos > 0.90,
            local_motion_m=local_motion.tolist(), alignment_cos=alignment_cos,
            controller_state=env.controller.diag.state,
            sigma_min=env.controller.diag.sigma_min,
            sigma_min_orientation=env.controller.diag.sigma_min_orientation,
            actuator_utilization=env.controller.diag.max_actuator_utilization,
            position_error_m=env.controller.diag.position_error_m,
            orientation_error_rad=env.controller.diag.orientation_error_rad,
        ))

    # 3) Full orientation contract: local/body rotation-vector axes must actually
    # produce the requested tool-frame rotation.  At nominal task pose these are:
    # X=pitch, Y=yaw, Z=roll.
    for axis in range(3):
        env.reset()
        p0, R0 = env.tip_pose()
        a = np.zeros(6)
        a[3 + axis] = 0.06
        env.execute_policy_delta(a, 0.45)
        p1, R1 = env.tip_pose()
        local_rotvec = matrix_to_rotvec(R0.T @ R1)
        commanded = float(local_rotvec[axis])
        rot_norm = float(np.linalg.norm(local_rotvec))
        alignment_cos = commanded / max(rot_norm, 1e-12)
        translation_drift = float(np.linalg.norm(p1 - p0))
        out.append(result(
            f"policy_rotation_axis_{axis}",
            commanded > 0.025 and alignment_cos > 0.82 and translation_drift < 0.012,
            local_rotvec_rad=local_rotvec.tolist(),
            alignment_cos=alignment_cos,
            translation_drift_m=translation_drift,
            controller_state=env.controller.diag.state,
            sigma_min=env.controller.diag.sigma_min,
            sigma_min_orientation=env.controller.diag.sigma_min_orientation,
            actuator_utilization=env.controller.diag.max_actuator_utilization,
            orientation_error_rad=env.controller.diag.orientation_error_rad,
        ))

    # 4) Blue tip should project near camera center and the red ring should be visible.
    env.reset()
    wrist_rgb = env.render_wrist()
    debug_dir = PROJECT_ROOT / "results" / "acceptance"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_image = debug_dir / "acceptance_wrist.png"
    cv2.imwrite(str(debug_image), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
    from panda_simulation.simulation_vision import extract_sim_observation, debug_blue_components
    obs = extract_sim_observation(wrist_rgb)
    blue_mask, blue_components = debug_blue_components(wrist_rgb)
    blue_mask_path = debug_dir / "acceptance_blue_mask.png"
    cv2.imwrite(str(blue_mask_path), blue_mask)
    cx = env.cfg.task.camera_width * 0.5
    cy = env.cfg.task.camera_height * 0.5
    if obs.blue_measured:
        err = math.hypot(obs.blue_u_px - cx, obs.blue_v_px - cy)
    else:
        err = float("inf")
    cam_geom = env.camera_tip_geometry()
    out.append(result("blue_tip_centering", obs.blue_measured and err < 12.0,
                      blue_measured=obs.blue_measured, center_error_px=err,
                      blue_u_px=obs.blue_u_px, blue_v_px=obs.blue_v_px,
                      camera_axis_error_deg=cam_geom["camera_axis_error_deg"],
                      marker_axis_error_deg=cam_geom["marker_axis_error_deg"],
                      tip_in_camera_m=cam_geom["tip_in_camera_m"].tolist(),
                      marker_in_camera_m=cam_geom["marker_in_camera_m"].tolist(),
                      blue_components=blue_components[:5],
                      debug_image=str(debug_image),
                      debug_blue_mask=str(blue_mask_path)))

    out.append(result("red_ring_visible", obs.target_valid,
                      du_px=obs.target_du_px, dv_px=obs.target_dv_px,
                      major_px=obs.target_major_px, minor_px=obs.target_minor_px))

    # 5) ObservationBuilder: finite, robot-independent, correct size, reset-relative
    # pose approximately zero.  Panda joints must not appear in this interface.
    policy_obs = env.policy_observation()
    rel_norm = float(np.linalg.norm(policy_obs.relative_pose6))
    finite = bool(np.all(np.isfinite(policy_obs.vector)))
    out.append(result(
        "policy_observation_contract",
        finite
        and policy_obs.vector.shape == (len(POLICY_OBS_V1_FEATURES),)
        and rel_norm < 0.01,
        observation_dim=int(policy_obs.vector.shape[0]),
        feature_names=list(policy_obs.feature_names),
        vector=policy_obs.vector.tolist(),
        relative_pose_norm=rel_norm,
    ))

    # 6) Scripted forward insertion: target-integrated mode intentionally locks the
    # side/up target while repeatedly advancing canonical +Z = rod forward.
    env.reset()
    before = env.insertion_metrics()
    max_radial = before["radial_error_m"]
    states = []
    crossed_plane = False
    for _ in range(36):
        a = np.zeros(6)
        a[2] = 0.008
        env.execute_delta(a, 0.14)
        m = env.insertion_metrics()
        max_radial = max(max_radial, m["radial_error_m"])
        states.append(env.controller.diag.state)
        if m["ring_plane_depth_m"] > 0.012:
            crossed_plane = True
            break
    after = env.insertion_metrics()
    saturated_steps = sum(1 for st in states if st == "ACTUATOR_SATURATED")
    insertion_pass = (
        crossed_plane
        and after["inside_hole"]
        and max_radial < 0.015
        and saturated_steps == 0
    )
    out.append(result("scripted_insertion", insertion_pass,
                      before=before, after=after, states=states,
                      steps=len(states), max_radial_error_m=max_radial,
                      saturated_steps=saturated_steps,
                      max_actuator_utilization=env.controller.diag.max_actuator_utilization,
                      servo_lead_max_rad=env.controller.diag.servo_lead_max_rad))

    passed = all(r["pass"] for r in out)
    print(json.dumps({"all_pass": passed, "tests": out}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_simulation import PandaRingEnv
from panda_simulation.learm_trajectory_replay import load_episodes, make_waypoints, map_waypoints_to_panda, pose_error


def overlay(frame, lines):
    img = frame.copy()
    y = 28
    for text in lines:
        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
    return img


def main():
    ap = argparse.ArgumentParser(description="Watch one real LeArm trajectory replayed on Panda in real time.")
    ap.add_argument("--source", type=Path, default=PROJECT_ROOT / "real_robot_dataset" / "learm_demonstrations_structured.zip")
    ap.add_argument("--episode", default="episode_000001")
    ap.add_argument("--mode", choices=("pose", "delta_pose", "delta_stream", "delta_velocity"), default="pose")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--keep-task-geometry", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="Wall-clock playback speed; 1.0 = real time")
    args = ap.parse_args()

    eps = {e.episode_id: e for e in load_episodes(args.source, success_only=True)}
    if args.episode not in eps:
        raise SystemExit(f"Unknown successful episode {args.episode}. Available: {', '.join(sorted(eps))}")
    ep = eps[args.episode]

    env = PandaRingEnv()
    if not args.keep_task_geometry:
        env.clear_task_obstacles_for_replay()
    p0, R0 = env.tip_pose()
    wp = make_waypoints(ep, stride_samples=args.stride)
    target_p, target_R = map_waypoints_to_panda(wp, p0, R0)

    print(f"Replaying {ep.episode_id} | mode={args.mode} | {len(wp.actions6)} policy steps | ~{ep.source_hz/args.stride:.2f} Hz")
    print("ESC in wrist-camera window or close MuJoCo viewer to stop.")

    cv2.namedWindow("Panda wrist camera — LeArm ground-truth replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Panda wrist camera — LeArm ground-truth replay", 960, 540)

    with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=True) as viewer:
        for i in range(len(wp.actions6)):
            if not viewer.is_running():
                break
            dt = float(max(wp.time_s[i + 1] - wp.time_s[i], env.physics_dt))
            requested = wp.actions6[i]
            if args.mode == "pose":
                env.command_pose_target(target_p[i + 1], target_R[i + 1])
                applied = requested
            elif args.mode == "delta_pose":
                applied = env.command_policy_delta(requested)
            else:
                # delta_stream is the firmware-style reference-integrated path. delta_velocity is
                # kept as a CLI alias for old commands.
                applied = env.command_policy_delta_stream(
                    requested,
                    horizon_s=dt,
                    watchdog_s=0.30,
                )

            n = max(1, int(round(dt / env.physics_dt)))
            wall_step = dt / max(args.speed, 1e-6) / n
            for _ in range(n):
                if not viewer.is_running():
                    break
                start = time.perf_counter()
                diag = env.step_physics(1)
                viewer.sync(state_only=True)

                # Render wrist view at a lower rate than MuJoCo physics.
                if int(env.data.time / env.physics_dt) % max(1, int(round(1.0 / (30.0 * env.physics_dt)))) == 0:
                    p, R = env.tip_pose()
                    pe, oe = pose_error(target_p[i + 1], target_R[i + 1], p, R)
                    frame = env.render_wrist()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = overlay(frame, [
                        f"{ep.episode_id}  {args.mode}  step {i+1}/{len(wp.actions6)}",
                        f"target error: {pe*1000:.1f} mm | {math.degrees(oe):.2f} deg",
                        f"controller: {diag.control_mode}/{diag.state} | actuator {diag.max_actuator_utilization:.2f}",
                        f"action: dP={np.linalg.norm(requested[:3])*1000:.1f} mm dR={math.degrees(np.linalg.norm(requested[3:])):.2f} deg",
                    ])
                    cv2.imshow("Panda wrist camera — LeArm ground-truth replay", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        return

                elapsed = time.perf_counter() - start
                sleep_s = wall_step - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

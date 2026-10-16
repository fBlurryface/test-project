#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_simulation import PandaRingEnv, SimConfig
from panda_simulation.learned_pre_insertion_policy import BlueRelativeFeatureBuilder, BlueRelativePreInsertionPolicy
from panda_simulation.real_tracker_vision_pipeline import RealLeArmVisionPipeline

CAMERA_FPS = 30.0
CONTROL_HZ = 50.0
POLICY_HZ = 10.0


def draw_text(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


class VisionWorker:
    """Latest-frame-only tracker so the 50 Hz control stream is never blocked by CV."""
    def __init__(self, config_path: Path):
        self.pipe = RealLeArmVisionPipeline(config_path)
        self.cv = threading.Condition()
        self.pending = None
        self.latest = None
        self.latest_sim_t = None
        self.worker_ms = 0.0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, rgb, sim_t):
        with self.cv:
            self.pending = (rgb.copy(), float(sim_t))
            self.cv.notify()

    def get(self):
        with self.cv:
            return self.latest, self.latest_sim_t, self.worker_ms

    def close(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()
        self.thread.join(timeout=1.0)

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.stop_flag:
                    self.cv.wait(timeout=0.1)
                if self.stop_flag:
                    return
                rgb, sim_t = self.pending
                self.pending = None
            t0 = time.perf_counter()
            out = self.pipe.process(rgb, now=sim_t)
            ms = (time.perf_counter() - t0) * 1000.0
            with self.cv:
                self.latest = out
                self.latest_sim_t = sim_t
                self.worker_ms = ms


def main():
    ap = argparse.ArgumentParser(description='Run the real-data blue-relative BC only through alignment/approach.')
    ap.add_argument('--seconds', type=float, default=18.0)
    ap.add_argument('--model', type=Path, default=PROJECT_ROOT / 'trained_policy' / 'blue_relative_pre_insertion_policy.npz')
    ap.add_argument('--max-left-mm-s', type=float, default=120.0)
    ap.add_argument('--max-up-mm-s', type=float, default=120.0)
    ap.add_argument('--max-forward-mm-s', type=float, default=100.0)
    ap.add_argument('--max-backward-mm-s', type=float, default=40.0)
    ap.add_argument('--stable-blue-loss-s', type=float, default=0.25)
    ap.add_argument('--ood-warn-z', type=float, default=3.0)
    ap.add_argument('--ood-hold-z', type=float, default=6.0)
    ap.add_argument('--shadow', action='store_true', help='show predictions but command zero velocity')
    ap.add_argument('--preset', choices=[
        'baseline', 'high25', 'high45', 'left35', 'right35',
        'high_left', 'high_right', 'random'
    ], default='baseline',
        help='initial ring placement relative to the validated baseline task frame')
    ap.add_argument('--ring-lateral-mm', type=float, default=None,
        help='override preset lateral offset in canonical +left direction (mm)')
    ap.add_argument('--ring-vertical-mm', type=float, default=None,
        help='override preset vertical offset in canonical +up direction (mm)')
    ap.add_argument('--seed', type=int, default=None,
        help='seed for --preset random; random uses lateral +/-40 mm and vertical 0..50 mm')
    args = ap.parse_args()

    presets = {
        'baseline':   (0.0,  0.0),
        'high25':     (0.0, 25.0),
        'high45':     (0.0, 45.0),
        'left35':    (35.0,  0.0),
        'right35':  (-35.0,  0.0),
        'high_left': (35.0, 35.0),
        'high_right':(-35.0, 35.0),
    }
    if args.preset == 'random':
        rng = np.random.default_rng(args.seed)
        lateral_mm = float(rng.uniform(-40.0, 40.0))
        vertical_mm = float(rng.uniform(0.0, 50.0))
    else:
        lateral_mm, vertical_mm = presets[args.preset]
    if args.ring_lateral_mm is not None:
        lateral_mm = float(args.ring_lateral_mm)
    if args.ring_vertical_mm is not None:
        vertical_mm = float(args.ring_vertical_mm)

    cfg = SimConfig()
    cfg.task.ring_lateral_offset_m = lateral_mm * 1e-3
    cfg.task.ring_vertical_offset_m = vertical_mm * 1e-3
    env = PandaRingEnv(cfg)
    policy = BlueRelativePreInsertionPolicy(args.model)
    feat = BlueRelativeFeatureBuilder(env.cfg.task.camera_width, env.cfg.task.camera_height)
    worker = VisionWorker(PROJECT_ROOT / 'real_robot_data_collection' / 'vision_tracker_config.json')

    out_dir = PROJECT_ROOT / 'results' / 'rollouts'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / time.strftime('learned_pre_insertion_%Y%m%d_%H%M%S.csv')
    fields = [
        'sim_time_s','preset','ring_lateral_mm','ring_vertical_mm','state','red_measured','blue_measured','blue_lost_s','phase_pierced',
        'red_u_px','red_v_px','blue_u_px','blue_v_px','error_u_px','error_v_px','error_norm_px',
        'red_major_px','red_minor_px','major_ratio','minor_ratio','axis_ratio','ood_max_abs_z',
        'pred_d_left_mm','pred_d_up_mm','pred_d_forward_mm',
        'cmd_left_mm_s','cmd_up_mm_s','cmd_forward_mm_s','controller_state',
        'inside_hole','radial_error_m','ring_plane_depth_m','ring_lift_m','vision_worker_ms','vision_age_ms',
    ]
    f = log_path.open('w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()

    print('\nLearned blue-relative pre-insertion policy')
    print('  learned from existing REAL successful LeArm data')
    print('  observation = ring center relative to measured blue terminal rod + relative ring scale')
    print('  output = [left, up, forward] actual tool delta over 100 ms')
    print('  rotations are forced to zero in this first experiment')
    print('  stable blue loss ends the test; there is NO learned or handcrafted lift here')
    print('  controller: reference-integrated Panda task-space motion layer')
    print(f'  ring placement: preset={args.preset}  lateral={lateral_mm:+.1f} mm  vertical={vertical_mm:+.1f} mm')
    print('  model:', args.model)
    print('  log:', log_path)
    if args.shadow:
        print('  SHADOW MODE: predictions are displayed but not executed')

    cv2.namedWindow('Learned pre-insertion policy', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Learned pre-insertion policy', 1100, 825)

    run_start = float(env.data.time)
    state = 'WAIT_VISION'
    latest_rgb = None
    command = np.zeros(6, dtype=float)
    pred_delta = np.zeros(3, dtype=float)
    current_features = None
    ood_z = float('nan')
    start_error = None
    min_error = float('inf')
    start_major = None
    max_major_ratio = 1.0
    stable_loss_start = None

    now_wall = time.perf_counter()
    next_control = now_wall
    next_policy = now_wall
    next_camera = now_wall
    next_display = now_wall
    done = False

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=True) as viewer:
            while viewer.is_running() and not done:
                wall = time.perf_counter()
                sim_t = float(env.data.time)
                if sim_t - run_start >= args.seconds:
                    state = 'TIMEOUT'; done = True; command[:] = 0.0

                if wall >= next_camera:
                    latest_rgb = env.render_wrist()
                    worker.submit(latest_rgb, sim_t)
                    next_camera = wall + 1.0 / CAMERA_FPS

                rv, rv_t, worker_ms = worker.get()
                vision_age_ms = float('nan') if rv_t is None else max(0.0, (sim_t - rv_t) * 1000.0)

                # Policy updates at 10 Hz.  The 50 Hz control loop below repeatedly
                # refreshes the latest learned velocity target into the reference-integrated control stream.
                if wall >= next_policy and not done:
                    next_policy = wall + 1.0 / POLICY_HZ
                    if rv is None or not rv.red_measured:
                        state = 'RED_LOST_HOLD'; command[:] = 0.0; current_features = None
                    elif not rv.blue_measured:
                        state = 'BLUE_LOSS_HOLD'; command[:] = 0.0; current_features = None
                        if stable_loss_start is None:
                            stable_loss_start = sim_t
                        if sim_t - stable_loss_start >= args.stable_blue_loss_s:
                            state = 'BLUE_LOSS_CONFIRMED'; done = True
                    else:
                        stable_loss_start = None
                        raw = feat.build_raw(rv)
                        if raw is None:
                            state = 'INVALID_HOLD'; command[:] = 0.0; current_features = None
                        else:
                            current_features = raw.copy()
                            pred_delta, ood_z = policy.predict_delta(raw)
                            twist = policy.delta_to_twist(pred_delta)
                            twist[0] = np.clip(twist[0], -args.max_left_mm_s * 1e-3, args.max_left_mm_s * 1e-3)
                            twist[1] = np.clip(twist[1], -args.max_up_mm_s * 1e-3, args.max_up_mm_s * 1e-3)
                            twist[2] = np.clip(twist[2], -args.max_backward_mm_s * 1e-3, args.max_forward_mm_s * 1e-3)
                            twist[3:] = 0.0
                            if ood_z >= args.ood_hold_z:
                                state = 'OOD_HOLD'; command[:] = 0.0
                            else:
                                state = 'BC_APPROACH' if ood_z < args.ood_warn_z else 'BC_APPROACH_OOD_WARN'
                                command[:] = 0.0 if args.shadow else twist

                did_control = False
                control_lateness_ms = 0.0
                if wall >= next_control:
                    control_lateness_ms = max(0.0, (wall - next_control) * 1000.0)
                    env.command_local_velocity(command, watchdog_s=0.30)
                    next_control += 1.0 / CONTROL_HZ
                    if wall - next_control > 0.2:
                        next_control = wall + 1.0 / CONTROL_HZ
                    did_control = True

                # Use one 2 ms physics step per
                # real-time loop.  Vision never determines the stepping cadence.
                env.step_physics(1)

                if rv is not None and rv.red_measured and rv.blue_measured:
                    eu = float(rv.red_u_px - rv.blue_u_px); ev = float(rv.red_v_px - rv.blue_v_px)
                    en = float(np.hypot(eu, ev))
                    if start_error is None: start_error = en
                    min_error = min(min_error, en)
                    if start_major is None and np.isfinite(rv.red_major_px): start_major = float(rv.red_major_px)
                    if start_major and np.isfinite(rv.red_major_px): max_major_ratio = max(max_major_ratio, float(rv.red_major_px) / start_major)
                else:
                    eu = ev = en = float('nan')

                diag = env.insertion_metrics()
                dyn = env.ring_dynamics_metrics()
                ctrl_state = env.controller.diag.state
                row = {
                    'sim_time_s': sim_t, 'preset': args.preset, 'ring_lateral_mm': lateral_mm, 'ring_vertical_mm': vertical_mm, 'state': state,
                    'red_measured': int(bool(rv and rv.red_measured)), 'blue_measured': int(bool(rv and rv.blue_measured)),
                    'blue_lost_s': float('nan') if rv is None else rv.blue_lost_s,
                    'phase_pierced': int(bool(rv and rv.phase_pierced)),
                    'red_u_px': float('nan') if rv is None else rv.red_u_px, 'red_v_px': float('nan') if rv is None else rv.red_v_px,
                    'blue_u_px': float('nan') if rv is None else rv.blue_u_px, 'blue_v_px': float('nan') if rv is None else rv.blue_v_px,
                    'error_u_px': eu, 'error_v_px': ev, 'error_norm_px': en,
                    'red_major_px': float('nan') if rv is None else rv.red_major_px, 'red_minor_px': float('nan') if rv is None else rv.red_minor_px,
                    'major_ratio': float('nan') if current_features is None else current_features[2],
                    'minor_ratio': float('nan') if current_features is None else current_features[3],
                    'axis_ratio': float('nan') if current_features is None else current_features[4],
                    'ood_max_abs_z': ood_z,
                    'pred_d_left_mm': pred_delta[0] * 1000.0, 'pred_d_up_mm': pred_delta[1] * 1000.0, 'pred_d_forward_mm': pred_delta[2] * 1000.0,
                    'cmd_left_mm_s': command[0] * 1000.0, 'cmd_up_mm_s': command[1] * 1000.0, 'cmd_forward_mm_s': command[2] * 1000.0,
                    'controller_state': ctrl_state,
                    'inside_hole': int(bool(diag.get('inside_hole', False))),
                    'radial_error_m': diag.get('radial_error_m', float('nan')),
                    'ring_plane_depth_m': diag.get('ring_plane_depth_m', float('nan')),
                    'ring_lift_m': dyn.get('lift_m', float('nan')),
                    'vision_worker_ms': worker_ms, 'vision_age_ms': vision_age_ms,
                }
                if did_control:
                    writer.writerow(row)
                    f.flush()

                if latest_rgb is not None and wall >= next_display:
                    img = cv2.cvtColor(latest_rgb, cv2.COLOR_RGB2BGR)
                    if rv is not None:
                        if rv.red_measured:
                            cv2.drawMarker(img, (int(round(rv.red_u_px)), int(round(rv.red_v_px))), (0,255,0), cv2.MARKER_CROSS, 24, 2)
                        if rv.blue_measured:
                            cv2.circle(img, (int(round(rv.blue_u_px)), int(round(rv.blue_v_px))), 8, (0,165,255), 2)
                        if rv.red_measured and rv.blue_measured:
                            cv2.arrowedLine(img, (int(round(rv.blue_u_px)), int(round(rv.blue_v_px))),
                                           (int(round(rv.red_u_px)), int(round(rv.red_v_px))), (255,255,255), 2, tipLength=0.12)
                    draw_text(img, f'state: {state}', 28, (0,255,255))
                    draw_text(img, f'error ring-blue: [{eu:+.1f}, {ev:+.1f}] px  norm={en:.1f}', 52)
                    draw_text(img, f'pred /100ms [L,U,F] = [{pred_delta[0]*1000:+.2f}, {pred_delta[1]*1000:+.2f}, {pred_delta[2]*1000:+.2f}] mm', 76)
                    draw_text(img, f'cmd [L,U,F] = [{command[0]*1000:+.1f}, {command[1]*1000:+.1f}, {command[2]*1000:+.1f}] mm/s', 100)
                    draw_text(img, f'OOD max |z|={ood_z:.2f}   vision worker={worker_ms:.1f} ms  age={vision_age_ms:.1f} ms', 124)
                    if current_features is not None:
                        draw_text(img, f'scale major/minor={current_features[2]:.2f}/{current_features[3]:.2f} axis={current_features[4]:.2f}', 148)
                    draw_text(img, f'placement: {args.preset}  left={lateral_mm:+.0f} mm  up={vertical_mm:+.0f} mm', 172, (200,220,255))
                    draw_text(img, 'ESC/Q quit   use --shadow for prediction-only mode', img.shape[0]-18, (200,200,200))
                    cv2.imshow('Learned pre-insertion policy', img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        done = True
                    next_display = wall + 1.0 / 20.0

                viewer.sync(state_only=True)
                sleep_s = env.physics_dt - (time.perf_counter() - wall)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            command[:] = 0.0
            env.command_local_velocity(command, watchdog_s=0.30)
    finally:
        worker.close(); f.close(); cv2.destroyAllWindows()

    print('\n=== RESULT ===')
    print('placement:', f'{args.preset}  left={lateral_mm:+.1f} mm  up={vertical_mm:+.1f} mm')
    print('final state:', state)
    print('start ring-blue error:', 'n/a' if start_error is None else f'{start_error:.1f} px')
    print('minimum ring-blue error:', 'n/a' if not np.isfinite(min_error) else f'{min_error:.1f} px')
    if start_error is not None and np.isfinite(min_error):
        print('alignment reduction:', f'{100.0*(1.0-min_error/max(start_error,1e-6)):.1f}%')
    print('max ring apparent-size ratio:', f'{max_major_ratio:.2f}x')
    print('stable blue loss reached:', state == 'BLUE_LOSS_CONFIRMED')
    print('log:', log_path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Canonical trainer for the deployed pre-insertion BC policy.

Input:  real_robot_dataset/learm_demonstrations_structured.zip (24 personally collected structured LeArm episodes).
        The loader selects successful episodes and valid measured pre-blue-loss windows.
Obs:    5-D blue-relative ring state.
Target: 100 ms tool-local translation [left, up, forward].
Output: trained_policy/blue_relative_pre_insertion_policy.npz, directly consumed by the Panda runtime.
"""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    "ring_blue_du_norm",
    "ring_blue_dv_norm",
    "ring_major_ratio",
    "ring_minor_ratio",
    "ring_axis_ratio",
)
ACTIONS = ("d_left_m", "d_up_m", "d_forward_m")
HORIZON_STEPS = 5  # 5 * 20 ms = 100 ms
POLICY_DT_S = 0.1


def rv_to_R(v):
    v = np.asarray(v, dtype=float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    x, y, z = v / a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def R_to_rv(R):
    R = np.asarray(R, dtype=float)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    s = 2.0 * math.sin(a)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return axis * a


def pose(row):
    yaw = math.radians(float(row["tool_yaw_deg"]))
    pitch = math.radians(float(row["tool_pitch_deg"]))
    roll = math.radians(float(row["tool_roll_deg"]))
    forward = np.array([
        math.sin(pitch) * math.cos(yaw),
        math.sin(pitch) * math.sin(yaw),
        math.cos(pitch),
    ], dtype=float)
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    forward /= np.linalg.norm(forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    R = np.column_stack([left, up, forward]) @ rv_to_R([0.0, 0.0, roll])
    p = np.array([float(row["tool_x_mm"]), float(row["tool_y_mm"]), float(row["tool_z_mm"])]) * 1e-3
    return p, R


def local_delta(a, b):
    p0, R0 = a
    p1, R1 = b
    return np.r_[R0.T @ (p1 - p0), R_to_rv(R0.T @ R1)]


def as_bool(row, key):
    try:
        return bool(int(float(row.get(key, 0) or 0)))
    except Exception:
        return False


def finite(*values):
    try:
        a = np.asarray([float(v) for v in values], dtype=float)
        return bool(np.all(np.isfinite(a)))
    except Exception:
        return False


def load_success_episodes(zip_path: Path):
    episodes = {}
    with zipfile.ZipFile(zip_path) as zf:
        roots = sorted({n.split('/')[0] for n in zf.namelist() if n.startswith('episode_') and '/' in n})
        for root in roots:
            meta = json.loads(zf.read(root + '/meta.json'))
            success = bool(meta.get('success', meta.get('outcome') == 'success'))
            if not success:
                continue
            rows = list(csv.DictReader(io.StringIO(zf.read(root + '/trajectory.csv').decode('utf-8-sig'))))
            poses = [pose(r) for r in rows]
            W = int(meta.get('vision', {}).get('camera_width', 1280))
            H = int(meta.get('vision', {}).get('camera_height', 960))

            # First valid blue+ring sample defines per-episode apparent-size reference.
            major0 = minor0 = None
            for r in rows:
                if as_bool(r, 'target_measured') and as_bool(r, 'blue_measured') and finite(r.get('target_major_px'), r.get('target_minor_px')):
                    ma = float(r['target_major_px']); mi = float(r['target_minor_px'])
                    if ma > 1.0 and mi > 1.0:
                        major0, minor0 = ma, mi
                        break
            if major0 is None:
                continue

            X, Y = [], []
            for i in range(len(rows) - HORIZON_STEPS):
                # Keep the entire 100 ms label horizon inside the measured pre-blue-loss phase.
                win = rows[i:i + HORIZON_STEPS + 1]
                if any(as_bool(q, 'pierced') for q in win):
                    continue
                if not all(as_bool(q, 'target_measured') and as_bool(q, 'blue_measured') for q in win):
                    continue
                r = rows[i]
                fields = (r.get('target_u_px'), r.get('target_v_px'), r.get('blue_u_px'), r.get('blue_v_px'),
                          r.get('target_major_px'), r.get('target_minor_px'))
                if not finite(*fields):
                    continue
                ru, rv, bu, bv, major, minor = map(float, fields)
                if major <= 1.0 or minor <= 1.0:
                    continue
                x = np.array([
                    (ru - bu) / max(0.5 * W, 1.0),
                    (rv - bv) / max(0.5 * H, 1.0),
                    major / major0,
                    minor / minor0,
                    minor / max(major, 1e-9),
                ], dtype=np.float32)
                y = local_delta(poses[i], poses[i + HORIZON_STEPS])[:3].astype(np.float32)
                if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                    X.append(x); Y.append(y)
            if X:
                eid = str(meta.get('episode_id', root))
                episodes[eid] = (np.stack(X), np.stack(Y))
    return episodes


def mirror_horizontal(X, Y):
    X = X.copy(); Y = Y.copy()
    X[:, 0] *= -1.0
    Y[:, 0] *= -1.0
    return X, Y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


def fit_model(X, Y, X_val=None, Y_val=None, seed=42, max_epochs=150):
    Xm, Ym = mirror_horizontal(X, Y)
    X = np.concatenate([X, Xm], axis=0)
    Y = np.concatenate([Y, Ym], axis=0)
    obs_mean = X.mean(0)
    obs_std = X.std(0)
    obs_std[obs_std < 1e-5] = 1.0
    action_mean = Y.mean(0)
    action_std = Y.std(0)
    action_std[action_std < 1e-6] = 1.0

    # Keep pauses in the data, but make decisive manipulation samples count more.
    mag = np.linalg.norm(Y, axis=1)
    weights = (0.5 + 3.0 * np.clip(mag / 0.006, 0.0, 2.0)).astype(np.float32)

    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x_t = torch.tensor((X - obs_mean) / obs_std, dtype=torch.float32)
    y_t = torch.tensor((Y - action_mean) / action_std, dtype=torch.float32)
    w_t = torch.tensor(weights, dtype=torch.float32)

    if X_val is not None:
        xv = torch.tensor((X_val - obs_mean) / obs_std, dtype=torch.float32)
        yv = torch.tensor((Y_val - action_mean) / action_std, dtype=torch.float32)
    else:
        xv = yv = None

    best_state = None
    best_val = float('inf')
    stale = 0
    epochs_done = 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(x_t), generator=g)
        for start in range(0, len(x_t), 512):
            ix = perm[start:start + 512]
            pred = model(x_t[ix])
            per = torch.nn.functional.smooth_l1_loss(pred, y_t[ix], reduction='none', beta=0.5).mean(1)
            loss = (per * w_t[ix]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        epochs_done = epoch + 1
        if xv is not None:
            model.eval()
            with torch.no_grad():
                val = float(torch.mean((model(xv) - yv) ** 2))
            if val < best_val - 1e-5:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, obs_mean, obs_std, action_mean, action_std, epochs_done, best_val


def predict(model, stats, X):
    obs_mean, obs_std, action_mean, action_std = stats
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor((X - obs_mean) / obs_std, dtype=torch.float32)).numpy()
    return p * action_std + action_mean


def metrics(P, Y):
    E = P - Y
    rmse_axis = np.sqrt(np.mean(E ** 2, axis=0)) * 1000.0
    active_dir = []
    corr = []
    for j in range(3):
        mask = np.abs(Y[:, j]) > 0.001
        active_dir.append(float(np.mean(np.sign(P[mask, j]) == np.sign(Y[mask, j]))) if np.any(mask) else None)
        corr.append(float(np.corrcoef(P[:, j], Y[:, j])[0, 1]))
    return {
        'n': int(len(Y)),
        'vector_rmse_mm_per_100ms': float(np.sqrt(np.mean(np.sum(E ** 2, axis=1))) * 1000.0),
        'axis_rmse_mm_per_100ms': {k: float(v) for k, v in zip(ACTIONS, rmse_axis)},
        'axis_correlation': {k: float(v) for k, v in zip(ACTIONS, corr)},
        'active_direction_accuracy': {k: v for k, v in zip(ACTIONS, active_dir)},
        'true_mean_speed_equiv_mm_s': float(np.mean(np.linalg.norm(Y, axis=1)) * 10000.0),
        'pred_mean_speed_equiv_mm_s': float(np.mean(np.linalg.norm(P, axis=1)) * 10000.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=PROJECT_ROOT / 'real_robot_dataset' / 'learm_demonstrations_structured.zip')
    ap.add_argument('--out', type=Path, default=PROJECT_ROOT / 'trained_policy' / 'blue_relative_pre_insertion_policy.npz')
    ap.add_argument('--report', type=Path, default=PROJECT_ROOT / 'results' / 'training_metrics.json')
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    episodes = load_success_episodes(args.data)
    ids = sorted(episodes)
    if len(ids) < 8:
        raise RuntimeError(f'Need several successful episodes, got {len(ids)}')

    shuffled = ids.copy(); random.Random(42).shuffle(shuffled)
    folds = [shuffled[i::4] for i in range(4)]
    fold_reports = []
    for fi in range(4):
        test_ids = folds[fi]
        train_ids = [e for j, f in enumerate(folds) if j != fi for e in f]
        X = np.concatenate([episodes[e][0] for e in train_ids])
        Y = np.concatenate([episodes[e][1] for e in train_ids])
        XT = np.concatenate([episodes[e][0] for e in test_ids])
        YT = np.concatenate([episodes[e][1] for e in test_ids])
        model, om, osd, am, asd, ne, bv = fit_model(X, Y, XT, YT, seed=100 + fi, max_epochs=140)
        P = predict(model, (om, osd, am, asd), XT)
        fold_reports.append({
            'fold': fi,
            'train_episodes': train_ids,
            'test_episodes': test_ids,
            'epochs': ne,
            'best_normalized_val_mse': bv,
            'metrics': metrics(P, YT),
        })

    # Final model sees all successful episodes.  No validation early stop here; use a
    # conservative fixed epoch count near the grouped-CV median.
    Xall = np.concatenate([episodes[e][0] for e in ids])
    Yall = np.concatenate([episodes[e][1] for e in ids])
    final_model, om, osd, am, asd, ne, _ = fit_model(Xall, Yall, None, None, seed=42, max_epochs=120)
    sd = final_model.state_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs_mean=om.astype(np.float32), obs_std=osd.astype(np.float32),
        action_mean=am.astype(np.float32), action_std=asd.astype(np.float32),
        w0=sd['net.0.weight'].cpu().numpy(), b0=sd['net.0.bias'].cpu().numpy(),
        w1=sd['net.2.weight'].cpu().numpy(), b1=sd['net.2.bias'].cpu().numpy(),
        w2=sd['net.4.weight'].cpu().numpy(), b2=sd['net.4.bias'].cpu().numpy(),
        w3=sd['net.6.weight'].cpu().numpy(), b3=sd['net.6.bias'].cpu().numpy(),
        feature_names=np.asarray(FEATURES), action_names=np.asarray(ACTIONS),
        policy_dt_s=np.asarray(POLICY_DT_S, dtype=np.float32),
    )

    # Human-readable response sweep, in mm / 100 ms.
    def q(eu, ev, scale=1.0, axis=0.87):
        return np.asarray([[eu, ev, scale, scale, axis]], dtype=np.float32)
    sweep = []
    stats = (om, osd, am, asd)
    for scale in (1.0, 1.5, 2.0):
        for eu, ev in ((0, 0), (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0.1, 0.1), (-0.1, -0.1)):
            a = predict(final_model, stats, q(eu, ev, scale))[0] * 1000.0
            sweep.append({'scale': scale, 'e_u': eu, 'e_v': ev,
                          'delta_left_mm': float(a[0]), 'delta_up_mm': float(a[1]), 'delta_forward_mm': float(a[2])})

    cv_vec = [f['metrics']['vector_rmse_mm_per_100ms'] for f in fold_reports]
    axis_keys = ACTIONS
    report = {
        'model': 'blue-relative pre-insertion behavior-cloning policy',
        'architecture': '5 -> 64 -> 64 -> 32 -> 3 ReLU MLP',
        'policy_horizon_s': POLICY_DT_S,
        'training_source': str(args.data.relative_to(PROJECT_ROOT)) if args.data.is_relative_to(PROJECT_ROOT) else str(args.data),
        'success_episodes': ids,
        'sample_count_all_success_pre_blue_loss': int(len(Xall)),
        'features': FEATURES,
        'actions': ACTIONS,
        'design': [
            'success episodes only',
            'entire 100 ms label horizon must keep real ring and blue measurements',
            'no pierced/post-blue-loss samples',
            'ring center is expressed relative to the measured blue terminal marker',
            'ring apparent size is normalized by first valid frame in each episode',
            'actions are actual observed local tool translation, not teleop commands',
            'no LeArm or Panda joint angles',
            'no rotation output in the learned pre-insertion policy',
            'horizontal mirror augmentation',
            'idle samples retained but active samples receive larger loss weight',
        ],
        'cross_validation': {
            'type': '4-fold grouped by whole episode',
            'folds': fold_reports,
            'mean_vector_rmse_mm_per_100ms': float(np.mean(cv_vec)),
            'std_vector_rmse_mm_per_100ms': float(np.std(cv_vec)),
            'mean_axis_rmse_mm_per_100ms': {
                k: float(np.mean([f['metrics']['axis_rmse_mm_per_100ms'][k] for f in fold_reports])) for k in axis_keys
            },
            'mean_active_direction_accuracy': {
                k: float(np.mean([f['metrics']['active_direction_accuracy'][k] for f in fold_reports])) for k in axis_keys
            },
        },
        'final_training_epochs': ne,
        'feature_mean': {k: float(v) for k, v in zip(FEATURES, om)},
        'feature_std': {k: float(v) for k, v in zip(FEATURES, osd)},
        'response_sweep_mm_per_100ms': sweep,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['cross_validation'], indent=2))
    print('model:', args.out)
    print('report:', args.report)


if __name__ == '__main__':
    main()

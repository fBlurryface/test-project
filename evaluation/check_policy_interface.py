#!/usr/bin/env python3
"""Validate that a trained NPZ satisfies the Panda blue-relative policy contract.

This test only needs NumPy; it does not import MuJoCo.
"""
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURES = (
    "ring_blue_du_norm",
    "ring_blue_dv_norm",
    "ring_major_ratio",
    "ring_minor_ratio",
    "ring_axis_ratio",
)
EXPECTED_ACTIONS = ("d_left_m", "d_up_m", "d_forward_m")
EXPECTED_SHAPES = {
    "obs_mean": (5,), "obs_std": (5,),
    "action_mean": (3,), "action_std": (3,),
    "w0": (64, 5), "b0": (64,),
    "w1": (64, 64), "b1": (64,),
    "w2": (32, 64), "b2": (32,),
    "w3": (3, 32), "b3": (3,),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "trained_policy" / "blue_relative_pre_insertion_policy.npz")
    args = ap.parse_args()
    d = np.load(args.model, allow_pickle=False)
    for key, shape in EXPECTED_SHAPES.items():
        if key not in d or d[key].shape != shape:
            raise SystemExit(f"contract failure: {key} has {None if key not in d else d[key].shape}, expected {shape}")
    features = tuple(str(x) for x in d["feature_names"].tolist())
    actions = tuple(str(x) for x in d["action_names"].tolist())
    dt = float(d["policy_dt_s"])
    if features != EXPECTED_FEATURES:
        raise SystemExit(f"contract failure: features={features}")
    if actions != EXPECTED_ACTIONS:
        raise SystemExit(f"contract failure: actions={actions}")
    if not (0.099 <= dt <= 0.101):
        raise SystemExit(f"contract failure: policy_dt_s={dt}")
    print("model contract OK")
    print("model:", args.model)
    print("features:", features)
    print("actions:", actions)
    print("policy_dt_s:", dt)


if __name__ == "__main__":
    main()

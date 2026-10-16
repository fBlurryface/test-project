# Behavior-cloning training

This directory contains the single canonical trainer used to produce the deployed pre-insertion policy.

## Input

`../real_robot_dataset/learm_demonstrations_structured.zip`

The archive contains all 24 structured real-LeArm episodes. The trainer reads episode metadata, selects successful demonstrations internally, and then keeps only 100 ms windows where:

- the red ring is actually measured,
- the blue terminal marker is actually measured,
- `pierced == 0` throughout the prediction horizon.

## Policy definition

Observation (5D):

1. `ring_blue_du_norm`
2. `ring_blue_dv_norm`
3. `ring_major_ratio`
4. `ring_minor_ratio`
5. `ring_axis_ratio`

Target (3D): tool-local displacement over 100 ms:

1. `d_left_m`
2. `d_up_m`
3. `d_forward_m`

Network: `5 -> 64 -> 64 -> 32 -> 3`, with ReLU hidden layers.

## Run

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
```

Outputs:

- `trained_policy/blue_relative_pre_insertion_policy.npz`
- `results/training_metrics.json`

The trained-policy path is also the default model path used by the Panda runtime.

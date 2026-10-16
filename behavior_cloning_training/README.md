# Behavior-cloning training

This directory contains the **single canonical trainer** for the deployed pre-insertion policy:

```text
train_blue_relative_pre_insertion_policy.py
```

## Input

```text
real_robot_dataset/learm_demonstrations_structured.zip
```

The script reads all 24 episodes, selects successful demonstrations internally, and retains a 100 ms training window only when the complete horizon has:

- a real red-ring measurement,
- a real blue-tip measurement,
- `pierced == 0`.

The current dataset yields **8,252 valid samples from 16 successful demonstrations**.

## Observation → action

```text
5D observation
[ring_blue_du_norm,
 ring_blue_dv_norm,
 ring_major_ratio,
 ring_minor_ratio,
 ring_axis_ratio]
        ↓
MLP: 5 → 64 → 64 → 32 → 3
        ↓
100 ms tool-local displacement
[d_left_m, d_up_m, d_forward_m]
```

Training uses observation/action normalization, horizontal mirror augmentation, SmoothL1 loss, and action-magnitude weighting.

## Run

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
```

Output:

```text
trained_policy/blue_relative_pre_insertion_policy.npz
```

The Panda runtime reads that exact path, so retraining automatically replaces the policy used by simulation.

To check the exported model contract:

```bash
python evaluation/check_policy_interface.py
```

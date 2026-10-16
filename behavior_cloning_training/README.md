# Behavior-cloning training

This directory contains the single canonical trainer for the deployed pre-insertion policy:

```text
train_blue_relative_pre_insertion_policy.py
```

## Input

```text
real_robot_dataset/learm_demonstrations_structured.zip
```

The trainer reads all 24 structured real-robot episodes, selects successful demonstrations internally, and retains only 100 ms windows that remain fully measured and pre-insertion.

Current dataset:

```text
16 successful demonstrations
8,252 valid pre-insertion samples
```

## Model contract

```text
5D visual observation
    ↓
MLP: 5 → 64 → 64 → 32 → 3
    ↓
[Δleft, Δup, Δforward] over ~100 ms
```

The action labels come from the recorded tool reference trajectory, not from LeArm teleoperation commands. Human-readable values are naturally interpreted in **mm / 100 ms**; the exported checkpoint stores meters plus `policy_dt_s = 0.1`.

Training uses normalization, horizontal mirror augmentation, SmoothL1 loss, and action-magnitude weighting.

Exact feature equations and label construction are documented in [`../docs/technical_appendix.md`](../docs/technical_appendix.md).

## Run

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
```

Output:

```text
trained_policy/blue_relative_pre_insertion_policy.npz
```

The Panda runtime reads this exact path, so retraining directly replaces the policy used in simulation.

Optional interface check:

```bash
python evaluation/check_policy_interface.py
```

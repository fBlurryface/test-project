# Real LeArm demonstration dataset

This directory contains the structured logs used by the canonical behavior-cloning trainer:

```text
learm_demonstrations_structured.zip
```

The archive contains **24 personally collected physical LeArm demonstrations**:

```text
16 success
8 failure
```

Each episode contains structured robot/reference-state and vision logs, including task-space tool pose, red-ring observations, blue-tip observations, timing, phase fields, and outcome metadata.

The original capture also includes raw eye-in-hand videos (about 3.27 GB). They are not bundled here because the deployed policy trains directly from the structured logs.

The trainer reads the full archive and performs its own success/sample filtering:

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
```

For collection code and visual tracking parameters, see [`../real_robot_data_collection/`](../real_robot_data_collection/).

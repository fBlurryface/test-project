# Real LeArm demonstration dataset

`learm_demonstrations_structured.zip` contains the structured logs from **24 personally collected physical LeArm demonstrations**:

```text
16 success
8 failure
```

Each episode retains the synchronized information needed to reconstruct training samples, including tool reference pose, red-ring measurements, blue-tip measurements, timing, phase fields, and outcome metadata.

The original collection also contains about **3.27 GB of raw eye-in-hand video**. The videos are not bundled here because the deployed policy trains from the structured logs directly.

The canonical trainer reads the full archive and performs its own success/sample filtering:

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
```

Collection code and tracker configuration are under [`../real_robot_data_collection/`](../real_robot_data_collection/).

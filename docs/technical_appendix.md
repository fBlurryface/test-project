# Technical appendix

This page keeps implementation/provenance details out of the main challenge presentation while preserving the information needed to audit the real-data → policy → Panda path.

## 1. Real LeArm reference state

The physical LeArm has no actuator position encoders. During demonstration collection, `real_robot_data_collection/collect_learm_demonstrations.py` maintains a host-side `ShadowLeArm` reference state synchronized with the firmware's fixed startup reference, geometry, command semantics, and reference-motion logic.

The relevant source-arm model uses:

| Quantity | Value |
| --- | ---: |
| Base-to-shoulder vertical offset | 28.9 mm |
| Upper arm | 104.3 mm |
| Forearm | 89.0 mm |
| Wrist-to-tool reference | 177.0 mm |
| Startup rotary reference | `[0, -60, 70, 70, 0]°` |

The challenge-relevant firmware snapshot is under `learm_firmware/`; the complete platform documentation is maintained in [ROS2vision](https://github.com/fBlurryface/ROS2vision).

The recorded task-space trajectory is therefore a deterministic **reference trajectory**, not encoder-measured physical state. For this project, the task-space reconstruction was additionally checked by replaying recorded poses on Panda and comparing the resulting manipulation motion with the original demonstrations.

## 2. Action/frame semantics

The learned action is not a LeArm joint command. It is a canonical local tool-space translation over 100 ms:

```text
+X = tool-left
+Y = tool-up
+Z = rod-forward
```

```text
[d_left_m, d_up_m, d_forward_m]
```

The simulated Panda resolves this canonical command with its own 7-DoF controller. Source and target joint allocations are therefore intentionally different.

## 3. Observation contract

The policy input is exactly:

```text
ring_blue_du_norm
ring_blue_dv_norm
ring_major_ratio
ring_minor_ratio
ring_axis_ratio
```

The first two values are red-ring center error relative to the measured blue rod-tip marker. The remaining values describe normalized apparent ring geometry.

The exported `.npz` stores feature names, action names, normalization statistics, MLP parameters, and the 0.1 s policy horizon.

## 4. Real-data → checkpoint provenance

```text
real_robot_dataset/learm_demonstrations_structured.zip
        ↓
behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
        ↓
trained_policy/blue_relative_pre_insertion_policy.npz
        ↓
run/run_panda_hybrid_ring_pick.py
```

Current bundled artifact SHA256 values:

```text
dataset:
f78de188fe4a1597b67ad5f26172ae2781b0ff6c4c5a8baf090fd53d6d90b495

canonical trainer:
fa7c8aa80f7a15e2e0dd15becb710ec1b23106045d0206aeaffba96e98e1d4df

bundled simulation-validated policy:
5eddf8684168f66ae5fa4b5e5515c45e795267b0adbb2fdb177a4665997bd3c2
```

The structured dataset contains all 24 collected episodes. The trainer performs success filtering and valid pre-insertion-window selection internally.

Running the trainer writes directly to the same checkpoint path consumed by simulation.

Fixed random seeds make data construction and evaluation repeatable, but bitwise-identical trained weights are not claimed across arbitrary PyTorch/CPU numerical environments.

## 5. Hybrid boundary

The MLP controls the pre-insertion phase. Persistent loss of the **actual measured** blue marker for 0.25 s is used as the insertion event. Once triggered, the runtime switches to a deterministic upward translation plus mild pitch primitive.

The event is an engineered visual phase signal, not a contact sensor.

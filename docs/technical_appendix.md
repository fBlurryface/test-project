# Technical appendix

This appendix keeps exact implementation details out of the main challenge presentation while preserving the information needed to audit the real-data → learned policy → Panda path.

## 1. Source and target embodiments

The physical source arm has five rotary arm joints plus a separate claw. For this task its approximate joint-axis sequence is:

```text
LeArm: Z Y Y Y Z + gripper
```

The simulated target is a Franka Panda:

```text
Panda: Z Y Z Y Z Y Z + gripper
```

No source-to-target joint mapping is used. Transfer is performed through task-centric visual observations and local tool-space motion.

For the rod-in-ring task, roll about the approximately axisymmetric rod is not a primary task variable. The task is dominated by tool-tip position and rod direction, allowing the more redundant Panda to realize source demonstrations without reproducing LeArm joint configurations.

## 2. Real LeArm reference trajectory

The physical LeArm has no actuator position encoders. During demonstration collection, `real_robot_data_collection/collect_learm_demonstrations.py` maintains a host-side `ShadowLeArm` reference state synchronized with the firmware's startup reference, geometry, command semantics, and reference-motion logic.

Key geometry/reference values are:

| Quantity | Value |
| --- | ---: |
| base-to-shoulder vertical offset | 28.9 mm |
| upper arm | 104.3 mm |
| forearm | 89.0 mm |
| wrist-to-tool reference | 177.0 mm |
| startup rotary reference | `[0, -60, 70, 70, 0]°` |

The recorded tool trajectory is therefore a deterministic **firmware-synchronized reference trajectory**, not encoder-measured physical state. For this project, recorded tool poses were also replayed on the Panda and checked against the manipulation performed on the physical LeArm.

The challenge-relevant firmware snapshot is in `learm_firmware/`; the wider platform is documented in [ROS2vision](https://github.com/fBlurryface/ROS2vision).

## 3. Why the visual state looks the way it does

The task was intentionally instrumented so that important manipulation state remains observable from an eye-in-hand camera.

- A rod is fixed to the end effector.
- Its blue-marked tip projects near a stable visual reference.
- The target is an upright red ring.
- Ring-tip relative position provides alignment error.
- Apparent ring size provides a proximity cue.
- Persistent blue-tip disappearance provides an insertion event.

The policy input is exactly:

```text
ring_blue_du_norm
ring_blue_dv_norm
ring_major_ratio
ring_minor_ratio
ring_axis_ratio
```

For image width $W$, height $H$, red-ring center $(u_r, v_r)$, blue-tip center $(u_b, v_b)$, current ring axes $(\mathrm{major}, \mathrm{minor})$, and the first valid episode axes $(\mathrm{major}_0, \mathrm{minor}_0)$:

$$
\begin{aligned}
e_u &= \frac{u_r-u_b}{W/2}, &
e_v &= \frac{v_r-v_b}{H/2}, \\
r_{\mathrm{major}} &= \frac{\mathrm{major}}{\mathrm{major}_0}, &
r_{\mathrm{minor}} &= \frac{\mathrm{minor}}{\mathrm{minor}_0}, \\
\rho &= \frac{\mathrm{minor}}{\mathrm{major}}.
\end{aligned}
$$

The continuous MLP observation is therefore

$$
o_t = [e_u, e_v, r_{\mathrm{major}}, r_{\mathrm{minor}}, \rho].
$$

Blue visibility is also used outside the MLP as the phase-transition signal.

## 4. Action-label construction

The learned target is not the source teleoperation input (`L / Z / P / F`) and not a source joint command.

For a recorded tool pose $T_t$, labels use a five-sample horizon at approximately 50 Hz:

$$
\Delta T_t = T_t^{-1}T_{t+5},
\qquad
5 \times 20\,\mathrm{ms} \approx 100\,\mathrm{ms}.
$$

The translational component is expressed in the canonical local tool frame:

```text
+X = tool-left
+Y = tool-up
+Z = rod-forward
```

The human-readable target is:

$$
y_t = [\Delta L, \Delta U, \Delta F]
$$

with values naturally interpreted as **mm / 100 ms**. The exported model stores the numerical action in meters (`d_left_m`, `d_up_m`, `d_forward_m`) and `policy_dt_s = 0.1`; deployment divides by 0.1 s to obtain the commanded task-space velocity.

This action representation is independent of LeArm joint allocation and can be consumed directly by the Panda task-space controller.

## 5. Training data selection

The structured dataset contains all 24 collected episodes:

```text
16 success
8 failure
```

The canonical trainer selects success episodes internally. A sample is used only when the entire 100 ms label horizon remains in the measured pre-insertion phase:

- `target_measured == 1` for every sample in the horizon;
- `blue_measured == 1` for every sample in the horizon;
- `pierced == 0` throughout the horizon.

The current dataset yields 8,252 valid pre-insertion samples.

The recorder stores substantially more information than the final 5D policy needs, including tool reference pose, red-ring measurements, blue-tip measurements, phase state and timing. The policy representation was selected after collection rather than restricting the raw structured record to the final five features.

## 6. Model and training

The deployed network is:

```text
5 → 64 → 64 → 32 → 3
    ReLU  ReLU  ReLU
```

Training uses:

- observation/action normalization;
- horizontal mirror augmentation ($e_u$ and $\Delta L$ change sign);
- SmoothL1 loss (`beta = 0.5`);
- larger loss weights for more active motion samples;
- grouped cross-validation by complete episode.

The model export contains network weights, normalization statistics, feature/action names and the 0.1 s policy horizon.

## 7. Hybrid phase boundary

During the learned phase, the blue marker contributes to the relative visual state. After the marker has first been reliably visible, continuous loss of the **actual measured marker** for 0.25 s is treated as the insertion event.

```text
learned pre-insertion policy
        ↓
persistent blue-tip loss
        ↓
deterministic upward translation + pitch
```

This is an engineered visual phase signal, not a contact sensor.

The post-insertion stage was initially explored as a learned behavior but did not transfer reliably. The physical ring was made from high-friction plasticine and tended to remain on the rod despite inconsistent human lift motions. In simulation the ring can slide, creating both a dynamics mismatch and a different post-insertion observation distribution. The final system therefore learns the geometry-dominated stage and scripts the contact-dynamics-sensitive terminal stage.

## 8. Data → checkpoint → runtime provenance

```text
real_robot_dataset/learm_demonstrations_structured.zip
        ↓
behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
        ↓
trained_policy/blue_relative_pre_insertion_policy.npz
        ↓
run/run_panda_hybrid_ring_pick.py
```

Bundled artifact SHA256 values:

```text
dataset:
f78de188fe4a1597b67ad5f26172ae2781b0ff6c4c5a8baf090fd53d6d90b495

canonical trainer:
fa7c8aa80f7a15e2e0dd15becb710ec1b23106045d0206aeaffba96e98e1d4df

bundled simulation-validated policy:
5eddf8684168f66ae5fa4b5e5515c45e795267b0adbb2fdb177a4665997bd3c2
```

Running the trainer writes directly to the checkpoint path consumed by simulation. Fixed random seeds make data construction and evaluation repeatable, but bitwise-identical learned weights are not claimed across arbitrary PyTorch/CPU numerical environments.

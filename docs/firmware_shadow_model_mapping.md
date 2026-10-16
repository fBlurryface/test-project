# LeArm firmware and shadow-state mapping

The physical LeArm firmware source used during demonstration collection is bundled under `learm_firmware/`.

Platform documentation and the complete ESP-IDF project are maintained in [ROS2vision](https://github.com/fBlurryface/ROS2vision). See its [hardware](https://github.com/fBlurryface/ROS2vision/blob/main/docs/hardware.md#arm-hardware-and-calibration), [architecture](https://github.com/fBlurryface/ROS2vision/blob/main/docs/architecture.md#esp32-arm-controller), and [interfaces](https://github.com/fBlurryface/ROS2vision/blob/main/docs/interfaces.md#esp32-arm-serial-protocol) pages for the wider LeArm context. The challenge data-collection path itself uses direct USB serial rather than the planned ROS 2 arm integration.

`real_robot_data_collection/collect_learm_demonstrations.py` maintains a host-side `ShadowLeArm` reference state. It starts from the same fixed startup pose as the firmware and consumes the same `teleop L Z P F` commands sent to the physical robot.

The shadow implementation mirrors the challenge-relevant firmware behavior:

- 20 ms / 50 Hz reference-motion update,
- 300 ms velocity-command watchdog,
- matching LeArm geometry and startup reference,
- matching joint velocity and acceleration limits,
- shared planar acceleration scaling,
- the same left / vertical / forward teleoperation semantics,
- the same persistent tool-pitch target behavior.

The recorded task-space trajectory is therefore a deterministic reconstructed/reference state rather than encoder-measured state. For this project, the reconstructed trajectory was additionally checked by replaying the recorded tool poses on the Panda and comparing the resulting manipulation motion with the original physical demonstration.

## Cross-embodiment use on Panda

The Panda does **not** copy LeArm joint allocation. The learned action is expressed in a canonical local tool frame and Panda resolves that tool motion using its own 7-DoF Jacobian controller:

```text
canonical tool-space command
        ↓
Panda damped-least-squares Jacobian
        ↓
7D joint-velocity target
        ↓
reference-integrated acceleration-limited motion
        ↓
Panda position actuators
```

This preserves the task-space and temporal semantics needed for transfer without forcing the source robot's kinematic structure onto Panda.

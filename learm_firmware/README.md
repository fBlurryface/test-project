# LeArm firmware snapshot used for data collection

This directory contains the challenge-relevant firmware source used by the physical LeArm during demonstration collection.

The broader LeArm platform, full ESP-IDF project, hardware/calibration documentation, and ROS 2 work are maintained in **[fBlurryface/ROS2vision](https://github.com/fBlurryface/ROS2vision)**. Useful references are:

- [LeArm hardware, calibration, geometry and startup pose](https://github.com/fBlurryface/ROS2vision/blob/main/docs/hardware.md#arm-hardware-and-calibration)
- [ESP32 arm architecture](https://github.com/fBlurryface/ROS2vision/blob/main/docs/architecture.md#esp32-arm-controller)
- [ESP32 command/state semantics](https://github.com/fBlurryface/ROS2vision/blob/main/docs/interfaces.md#esp32-arm-serial-protocol)

This local snapshot is intentionally compact: it includes the application entry point and the four core components required to audit the controller used during challenge data collection, rather than duplicating the complete ESP-IDF workspace and the wider ROS2vision project.

## Included source

```text
learm_firmware/
├── main/
│   ├── CMakeLists.txt
│   └── main.cpp
└── components/
    ├── arm_joint/
    ├── arm_kinematics/
    ├── arm_motion/
    └── servo_pwm/
```

The files most relevant to challenge data provenance are:

- `main/main.cpp` — parses `teleop L Z P F`, maintains the persistent pitch target, combines the planar hierarchical solver with lateral/base motion, and sends reference joint velocities to the motion layer.
- `components/arm_kinematics/` — LeArm geometry, forward kinematics, resolved-rate IK, and the hierarchical planar teleoperation solver.
- `components/arm_motion/` — 20 ms reference-motion integration, velocity/acceleration limits, shared planar acceleration scaling, joint-limit behavior, and the 300 ms command watchdog.
- `components/arm_joint/` — physical joint calibration and ±85° joint limits.
- `components/servo_pwm/` — low-level actuator backend required by the firmware dependency graph.

## Relationship to the data collector

`real_robot_data_collection/collect_learm_demonstrations.py` contains a host-side `ShadowLeArm` implementation of the firmware reference kinematics and motion logic. It starts from the same fixed reference pose and consumes the same `teleop L Z P F` commands sent to the physical robot.

Key parameters match the firmware snapshot, including:

- initial joint reference: `[0, -60, 70, 70, 0]` deg
- geometry: `28.9 / 104.3 / 89.0 / 177.0` mm
- kinematic velocity limits: `100 / 70 / 70 / 85 / 60` deg/s
- motion acceleration limits: `520 / 360 / 360 / 450 / 300` deg/s²
- teleop maxima: left `240`, vertical `180`, forward `100` mm/s
- pitch trim `60` deg/s, servo limit `75` deg/s, target lead `12` deg
- motion update period `20 ms`
- velocity-command watchdog `300 ms`

The arm has no actuator position feedback, so the recorded task-space pose is a deterministic reconstructed/reference state rather than encoder-measured state. For this project, the reconstructed trajectory was additionally checked by replaying recorded task-space trajectories on Panda and comparing the resulting manipulation motion with the original physical demonstrations.

## Scope of the ROS2vision link

ROS2vision is the authoritative home for the wider platform and complete firmware project. Its planned ROS 2 arm integration is **not** a runtime dependency of this challenge repository: challenge data collection uses the existing ESP32 USB-serial path directly.

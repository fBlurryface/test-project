# LeArm firmware snapshot

This directory contains the challenge-relevant ESP32 firmware source used during real demonstration collection.

The broader LeArm platform, complete ESP-IDF project, hardware documentation, calibration notes, and ROS 2 work live in my separate repository:

**https://github.com/fBlurryface/ROS2vision**

Useful references there:

- [LeArm hardware / geometry / calibration](https://github.com/fBlurryface/ROS2vision/blob/main/docs/hardware.md#arm-hardware-and-calibration)
- [ESP32 arm architecture](https://github.com/fBlurryface/ROS2vision/blob/main/docs/architecture.md#esp32-arm-controller)
- [ESP32 command and state semantics](https://github.com/fBlurryface/ROS2vision/blob/main/docs/interfaces.md#esp32-arm-serial-protocol)

This local snapshot is included only so the challenge data path is auditable. It contains the application entry point plus the motion, kinematics, joint-calibration, and PWM components used by the collector's host-side reference-state reconstruction.

The physical LeArm has no actuator position feedback. The collector therefore records a deterministic firmware-synchronized **reference trajectory**, not encoder-measured joint state. The exact correspondence is summarized in [`../docs/technical_appendix.md`](../docs/technical_appendix.md).

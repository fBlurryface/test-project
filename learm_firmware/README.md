# LeArm firmware snapshot

This directory contains the ESP32 firmware subset relevant to the real demonstration-collection path used in this challenge submission.

The wider LeArm project, full platform documentation, calibration notes and ROS 2 work are maintained separately in:

**https://github.com/fBlurryface/ROS2vision**

Useful references:

- [LeArm hardware, geometry and calibration](https://github.com/fBlurryface/ROS2vision/blob/main/docs/hardware.md#arm-hardware-and-calibration)
- [ESP32 arm architecture](https://github.com/fBlurryface/ROS2vision/blob/main/docs/architecture.md#esp32-arm-controller)
- [ESP32 command/state semantics](https://github.com/fBlurryface/ROS2vision/blob/main/docs/interfaces.md#esp32-arm-serial-protocol)

This local snapshot is included so the demonstration-state reconstruction can be audited without cloning the larger project. The physical LeArm has no actuator position encoders; the collector therefore records a deterministic firmware-synchronized **reference trajectory**, not encoder-measured joint state.

The exact firmware/shadow relationship is summarized in [`../docs/technical_appendix.md`](../docs/technical_appendix.md).

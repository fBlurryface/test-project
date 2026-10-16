#pragma once

#include <stdint.h>

#include "esp_err.h"
#include "servo_pwm.hpp"

namespace learm {

enum class ArmJoint : uint8_t {
    Claw = 0,
    WristRoll,
    WristPitch,
    Elbow,
    Shoulder,
    Base,
};

inline constexpr uint8_t kArmJointCount = 6;

struct JointCalibration {
    ArmJoint joint;
    ServoChannel channel;
    float zero_deg;
    uint16_t zero_us;
    float us_per_degree;
    int8_t direction;
    float min_deg;
    float max_deg;
};

struct JointRuntimeState {
    bool configured;
    bool enabled;
    ArmJoint joint;
    ServoChannel channel;
    float command_deg;
    uint16_t command_us;
    uint16_t output_us;
    float zero_deg;
    uint16_t zero_us;
    float us_per_degree;
    int8_t direction;
    float min_deg;
    float max_deg;
};

struct ClawGapPoint {
    uint16_t pulse_us;
    float gap_cm;
};

class ArmJointController {
public:
    esp_err_t init(
        ServoPwm* servo_pwm,
        const JointCalibration* calibrations,
        uint8_t count
    );

    bool is_initialized() const;
    bool is_configured(ArmJoint joint) const;

    esp_err_t enable_joint(ArmJoint joint, float start_deg);
    esp_err_t enable_claw_at_gap(float start_gap_cm);
    esp_err_t disable_joint(ArmJoint joint);
    esp_err_t disable_all();

    esp_err_t write_deg_now(ArmJoint joint, float target_deg);
    esp_err_t write_us_now(ArmJoint joint, uint16_t pulse_us);

    uint16_t claw_gap_cm_to_us(float gap_cm) const;
    float claw_us_to_gap_cm(uint16_t pulse_us) const;
    float read_claw_gap_cm() const;
    float read_command_deg(ArmJoint joint) const;
    uint16_t read_command_us(ArmJoint joint) const;
    uint16_t read_output_us(ArmJoint joint) const;
    uint16_t deg_to_us(ArmJoint joint, float deg) const;
    float us_to_deg(ArmJoint joint, uint16_t pulse_us) const;
    JointCalibration get_calibration(ArmJoint joint) const;
    esp_err_t set_calibration(ArmJoint joint, const JointCalibration& calibration);
    JointRuntimeState get_state(ArmJoint joint) const;

private:
    struct JointState {
        bool configured = false;
        JointCalibration calibration = {};
    };

    static int joint_to_index(ArmJoint joint);
    bool is_valid_joint_index(int index) const;
    esp_err_t normalize_calibration(JointCalibration* calibration) const;
    float clamp_deg_by_index(uint8_t index, float deg) const;
    uint16_t deg_to_us_by_index(uint8_t index, float deg) const;
    float us_to_deg_by_index(uint8_t index, uint16_t pulse_us) const;
    uint16_t claw_gap_cm_to_us_from_table(float gap_cm) const;
    float claw_us_to_gap_cm_from_table(uint16_t pulse_us) const;

private:
    bool initialized_ = false;
    ServoPwm* servo_ = nullptr;
    JointState joints_[kArmJointCount] = {};
};

namespace arm_joint_defaults {
extern const ServoPwmConfig kServoPwmConfigs[];
extern const uint8_t kServoPwmConfigCount;
extern const JointCalibration kJointCalibrations[];
extern const uint8_t kJointCalibrationCount;
extern const ClawGapPoint kClawGapTable[];
extern const uint8_t kClawGapTableCount;
}  // namespace arm_joint_defaults

}  // namespace learm

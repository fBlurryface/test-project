#include "arm_joint.hpp"

#include <algorithm>
#include <cmath>

#include "driver/gpio.h"
#include "esp_log.h"

namespace learm {

static const char* TAG = "arm_joint";

namespace arm_joint_defaults {

static constexpr float kDefaultUsPerDegree = 2000.0f / 180.0f;

const ServoPwmConfig kServoPwmConfigs[] = {
    {ServoChannel::S0, GPIO_NUM_19,   500,   500, 1600, 0},
    {ServoChannel::S1, GPIO_NUM_18,  1480,   500, 2500, 0},
    {ServoChannel::S2, GPIO_NUM_5,   1580,   500, 2500, 0},
    {ServoChannel::S3, GPIO_NUM_4,   1490,   500, 2500, 0},
    {ServoChannel::S4, GPIO_NUM_0,   1530,   500, 2500, 0},
    {ServoChannel::S5, GPIO_NUM_15,  1450,   500, 2500, 0},
};

const uint8_t kServoPwmConfigCount =
    sizeof(kServoPwmConfigs) / sizeof(kServoPwmConfigs[0]);

const JointCalibration kJointCalibrations[] = {
    {ArmJoint::Claw,      ServoChannel::S0, 0.0f, 500,  11.0f,              +1,  0.0f, 100.0f},
    {ArmJoint::WristRoll, ServoChannel::S1, 0.0f, 1480, kDefaultUsPerDegree, -1, -85.0f, 85.0f},
    {ArmJoint::WristPitch,ServoChannel::S2, 0.0f, 1580, kDefaultUsPerDegree, -1, -85.0f, 85.0f},
    {ArmJoint::Elbow,     ServoChannel::S3, 0.0f, 1490, kDefaultUsPerDegree, -1, -85.0f, 85.0f},
    {ArmJoint::Shoulder,  ServoChannel::S4, 0.0f, 1530, kDefaultUsPerDegree, -1, -85.0f, 85.0f},
    {ArmJoint::Base,      ServoChannel::S5, 0.0f, 1450, 10.89f,              +1, -85.0f, 85.0f},
};

const uint8_t kJointCalibrationCount =
    sizeof(kJointCalibrations) / sizeof(kJointCalibrations[0]);

const ClawGapPoint kClawGapTable[] = {
    {500,  5.8f},
    {700,  5.6f},
    {900,  5.0f},
    {1050, 4.4f},
    {1150, 3.6f},
    {1200, 2.9f},
    {1250, 2.8f},
    {1300, 2.4f},
    {1400, 1.6f},
    {1500, 0.6f},
    {1600, 0.0f},
};

const uint8_t kClawGapTableCount =
    sizeof(kClawGapTable) / sizeof(kClawGapTable[0]);

}  // namespace arm_joint_defaults

int ArmJointController::joint_to_index(ArmJoint joint)
{
    const uint8_t index = static_cast<uint8_t>(joint);
    if (index >= kArmJointCount) {
        return -1;
    }
    return static_cast<int>(index);
}

bool ArmJointController::is_valid_joint_index(int index) const
{
    return index >= 0 &&
           index < static_cast<int>(kArmJointCount) &&
           joints_[index].configured;
}

esp_err_t ArmJointController::normalize_calibration(
    JointCalibration* calibration
) const
{
    if (calibration == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    const int joint_index = joint_to_index(calibration->joint);
    if (joint_index < 0) {
        return ESP_ERR_INVALID_ARG;
    }

    const uint8_t channel_index = static_cast<uint8_t>(calibration->channel);
    if (channel_index >= kServoCount ||
        calibration->us_per_degree <= 0.0f ||
        calibration->min_deg > calibration->max_deg) {
        return ESP_ERR_INVALID_ARG;
    }

    calibration->direction = (calibration->direction >= 0) ? 1 : -1;
    calibration->zero_deg = std::clamp<float>(
        calibration->zero_deg,
        calibration->min_deg,
        calibration->max_deg
    );

    if (servo_ != nullptr) {
        const ServoRuntimeState servo_state = servo_->get_state(calibration->channel);
        if (!servo_state.configured) {
            ESP_LOGE(TAG, "calibration rejected: servo channel S%u is not configured",
                     static_cast<unsigned>(channel_index));
            return ESP_ERR_INVALID_ARG;
        }
        if (calibration->zero_us < servo_state.min_us ||
            calibration->zero_us > servo_state.max_us) {
            ESP_LOGE(TAG, "calibration rejected: zero_us=%u outside servo range [%u,%u]",
                     calibration->zero_us, servo_state.min_us, servo_state.max_us);
            return ESP_ERR_INVALID_ARG;
        }
    }

    return ESP_OK;
}

esp_err_t ArmJointController::init(
    ServoPwm* servo_pwm,
    const JointCalibration* calibrations,
    uint8_t count
)
{
    if (servo_pwm == nullptr || calibrations == nullptr ||
        count == 0 || count > kArmJointCount) {
        return ESP_ERR_INVALID_ARG;
    }

    servo_ = servo_pwm;
    for (uint8_t i = 0; i < kArmJointCount; ++i) {
        joints_[i] = JointState{};
    }

    for (uint8_t i = 0; i < count; ++i) {
        JointCalibration cal = calibrations[i];
        esp_err_t ret = normalize_calibration(&cal);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "invalid calibration at index=%u", static_cast<unsigned>(i));
            return ret;
        }

        const int joint_index = joint_to_index(cal.joint);
        if (joint_index < 0 || joints_[joint_index].configured) {
            return ESP_ERR_INVALID_ARG;
        }

        joints_[joint_index].configured = true;
        joints_[joint_index].calibration = cal;

        ESP_LOGI(TAG,
                 "configured joint=%u -> S%u zero_deg=%.2f zero_us=%u us_per_deg=%.4f dir=%d range=[%.2f,%.2f]",
                 static_cast<unsigned>(cal.joint),
                 static_cast<unsigned>(cal.channel),
                 static_cast<double>(cal.zero_deg),
                 cal.zero_us,
                 static_cast<double>(cal.us_per_degree),
                 static_cast<int>(cal.direction),
                 static_cast<double>(cal.min_deg),
                 static_cast<double>(cal.max_deg));
    }

    initialized_ = true;
    ESP_LOGI(TAG, "arm_joint initialized");
    return ESP_OK;
}

bool ArmJointController::is_initialized() const
{
    return initialized_;
}

bool ArmJointController::is_configured(ArmJoint joint) const
{
    return is_valid_joint_index(joint_to_index(joint));
}

float ArmJointController::clamp_deg_by_index(uint8_t index, float deg) const
{
    if (index >= kArmJointCount || !joints_[index].configured) {
        return deg;
    }
    const JointCalibration& cal = joints_[index].calibration;
    return std::clamp<float>(deg, cal.min_deg, cal.max_deg);
}

uint16_t ArmJointController::deg_to_us_by_index(uint8_t index, float deg) const
{
    if (index >= kArmJointCount || !joints_[index].configured) {
        return 0;
    }

    const JointCalibration& cal = joints_[index].calibration;
    deg = clamp_deg_by_index(index, deg);
    const float delta_deg = deg - cal.zero_deg;
    const float pulse = static_cast<float>(cal.zero_us) +
                        static_cast<float>(cal.direction) *
                        delta_deg * cal.us_per_degree;
    int pulse_i = static_cast<int>(std::lround(pulse));

    if (servo_ != nullptr) {
        const ServoRuntimeState servo_state = servo_->get_state(cal.channel);
        if (servo_state.configured) {
            pulse_i = std::clamp<int>(pulse_i, servo_state.min_us, servo_state.max_us);
        }
    }
    pulse_i = std::clamp<int>(pulse_i, kServoMinUs, kServoMaxUs);
    return static_cast<uint16_t>(pulse_i);
}

float ArmJointController::us_to_deg_by_index(uint8_t index, uint16_t pulse_us) const
{
    if (index >= kArmJointCount || !joints_[index].configured) {
        return 0.0f;
    }

    const JointCalibration& cal = joints_[index].calibration;
    const float denom = static_cast<float>(cal.direction) * cal.us_per_degree;
    if (std::fabs(denom) < 0.0001f) {
        return cal.zero_deg;
    }

    const float deg = cal.zero_deg +
        (static_cast<float>(pulse_us) - static_cast<float>(cal.zero_us)) / denom;
    return std::clamp<float>(deg, cal.min_deg, cal.max_deg);
}

uint16_t ArmJointController::deg_to_us(ArmJoint joint, float deg) const
{
    const int index = joint_to_index(joint);
    return is_valid_joint_index(index)
        ? deg_to_us_by_index(static_cast<uint8_t>(index), deg)
        : 0;
}

float ArmJointController::us_to_deg(ArmJoint joint, uint16_t pulse_us) const
{
    const int index = joint_to_index(joint);
    return is_valid_joint_index(index)
        ? us_to_deg_by_index(static_cast<uint8_t>(index), pulse_us)
        : 0.0f;
}

uint16_t ArmJointController::claw_gap_cm_to_us(float gap_cm) const
{
    return claw_gap_cm_to_us_from_table(gap_cm);
}

float ArmJointController::claw_us_to_gap_cm(uint16_t pulse_us) const
{
    return claw_us_to_gap_cm_from_table(pulse_us);
}

uint16_t ArmJointController::claw_gap_cm_to_us_from_table(float gap_cm) const
{
    const ClawGapPoint* table = arm_joint_defaults::kClawGapTable;
    const uint8_t count = arm_joint_defaults::kClawGapTableCount;
    if (table == nullptr || count == 0) {
        return 0;
    }
    if (count == 1) {
        return table[0].pulse_us;
    }

    const float max_gap = table[0].gap_cm;
    const float min_gap = table[count - 1].gap_cm;
    gap_cm = std::clamp<float>(gap_cm, min_gap, max_gap);

    if (gap_cm >= table[0].gap_cm) return table[0].pulse_us;
    if (gap_cm <= table[count - 1].gap_cm) return table[count - 1].pulse_us;

    for (uint8_t i = 0; i + 1 < count; ++i) {
        const ClawGapPoint& a = table[i];
        const ClawGapPoint& b = table[i + 1];
        if (gap_cm <= a.gap_cm && gap_cm >= b.gap_cm) {
            const float denom = b.gap_cm - a.gap_cm;
            if (std::fabs(denom) < 0.0001f) return a.pulse_us;
            const float ratio = (gap_cm - a.gap_cm) / denom;
            const float pulse = static_cast<float>(a.pulse_us) + ratio *
                static_cast<float>(static_cast<int>(b.pulse_us) - static_cast<int>(a.pulse_us));
            int pulse_i = static_cast<int>(std::lround(pulse));
            pulse_i = std::clamp<int>(pulse_i, table[0].pulse_us, table[count - 1].pulse_us);
            return static_cast<uint16_t>(pulse_i);
        }
    }
    return table[count - 1].pulse_us;
}

float ArmJointController::claw_us_to_gap_cm_from_table(uint16_t pulse_us) const
{
    const ClawGapPoint* table = arm_joint_defaults::kClawGapTable;
    const uint8_t count = arm_joint_defaults::kClawGapTableCount;
    if (table == nullptr || count == 0) return 0.0f;
    if (count == 1) return table[0].gap_cm;

    pulse_us = std::clamp<uint16_t>(pulse_us, table[0].pulse_us, table[count - 1].pulse_us);
    if (pulse_us <= table[0].pulse_us) return table[0].gap_cm;
    if (pulse_us >= table[count - 1].pulse_us) return table[count - 1].gap_cm;

    for (uint8_t i = 0; i + 1 < count; ++i) {
        const ClawGapPoint& a = table[i];
        const ClawGapPoint& b = table[i + 1];
        if (pulse_us >= a.pulse_us && pulse_us <= b.pulse_us) {
            const float denom = static_cast<float>(static_cast<int>(b.pulse_us) - static_cast<int>(a.pulse_us));
            if (std::fabs(denom) < 0.0001f) return a.gap_cm;
            const float ratio = static_cast<float>(static_cast<int>(pulse_us) - static_cast<int>(a.pulse_us)) / denom;
            return a.gap_cm + ratio * (b.gap_cm - a.gap_cm);
        }
    }
    return table[count - 1].gap_cm;
}

float ArmJointController::read_claw_gap_cm() const
{
    return claw_us_to_gap_cm(read_command_us(ArmJoint::Claw));
}

esp_err_t ArmJointController::enable_joint(ArmJoint joint, float start_deg)
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return ESP_ERR_INVALID_ARG;
    const JointCalibration& cal = joints_[index].calibration;
    return servo_->enable(cal.channel, deg_to_us_by_index(static_cast<uint8_t>(index), start_deg));
}

esp_err_t ArmJointController::enable_claw_at_gap(float start_gap_cm)
{
    const int index = joint_to_index(ArmJoint::Claw);
    if (!initialized_ || !is_valid_joint_index(index) || servo_ == nullptr ||
        !std::isfinite(start_gap_cm)) return ESP_ERR_INVALID_ARG;
    const JointCalibration& cal = joints_[index].calibration;
    return servo_->enable(cal.channel, claw_gap_cm_to_us(start_gap_cm));
}

esp_err_t ArmJointController::disable_joint(ArmJoint joint)
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return ESP_ERR_INVALID_ARG;
    return servo_->disable(joints_[index].calibration.channel);
}

esp_err_t ArmJointController::disable_all()
{
    if (!initialized_ || servo_ == nullptr) return ESP_ERR_INVALID_STATE;
    esp_err_t first_error = ESP_OK;
    for (uint8_t i = 0; i < kArmJointCount; ++i) {
        if (!joints_[i].configured) continue;
        const esp_err_t ret = servo_->disable(joints_[i].calibration.channel);
        if (ret != ESP_OK && first_error == ESP_OK) first_error = ret;
    }
    return first_error;
}

esp_err_t ArmJointController::write_deg_now(ArmJoint joint, float target_deg)
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return ESP_ERR_INVALID_ARG;
    return write_us_now(joint, deg_to_us_by_index(static_cast<uint8_t>(index), target_deg));
}

esp_err_t ArmJointController::write_us_now(ArmJoint joint, uint16_t pulse_us)
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index) || servo_ == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    return servo_->write_us_now(joints_[index].calibration.channel, pulse_us);
}

float ArmJointController::read_command_deg(ArmJoint joint) const
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return 0.0f;
    const JointCalibration& cal = joints_[index].calibration;
    return us_to_deg_by_index(static_cast<uint8_t>(index), servo_->read_command_us(cal.channel));
}

uint16_t ArmJointController::read_command_us(ArmJoint joint) const
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return 0;
    return servo_->read_command_us(joints_[index].calibration.channel);
}

uint16_t ArmJointController::read_output_us(ArmJoint joint) const
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return 0;
    return servo_->read_output_us(joints_[index].calibration.channel);
}

JointCalibration ArmJointController::get_calibration(ArmJoint joint) const
{
    const int index = joint_to_index(joint);
    return is_valid_joint_index(index) ? joints_[index].calibration : JointCalibration{};
}

esp_err_t ArmJointController::set_calibration(
    ArmJoint joint,
    const JointCalibration& calibration
)
{
    const int index = joint_to_index(joint);
    if (index < 0) return ESP_ERR_INVALID_ARG;
    JointCalibration cal = calibration;
    cal.joint = joint;
    const esp_err_t ret = normalize_calibration(&cal);
    if (ret != ESP_OK) return ret;
    joints_[index].configured = true;
    joints_[index].calibration = cal;
    ESP_LOGW(TAG,
             "calibration updated joint=%u -> S%u zero_deg=%.2f zero_us=%u us_per_deg=%.4f dir=%d range=[%.2f,%.2f]",
             static_cast<unsigned>(cal.joint), static_cast<unsigned>(cal.channel),
             static_cast<double>(cal.zero_deg), cal.zero_us,
             static_cast<double>(cal.us_per_degree), static_cast<int>(cal.direction),
             static_cast<double>(cal.min_deg), static_cast<double>(cal.max_deg));
    return ESP_OK;
}

JointRuntimeState ArmJointController::get_state(ArmJoint joint) const
{
    const int index = joint_to_index(joint);
    if (!initialized_ || !is_valid_joint_index(index)) return {};
    const JointCalibration& cal = joints_[index].calibration;
    const ServoRuntimeState servo_state = servo_->get_state(cal.channel);

    JointRuntimeState state = {};
    state.configured = joints_[index].configured && servo_state.configured;
    state.enabled = servo_state.enabled;
    state.joint = joint;
    state.channel = cal.channel;
    state.command_us = servo_state.command_us;
    state.output_us = servo_state.output_us;
    state.command_deg = us_to_deg_by_index(static_cast<uint8_t>(index), servo_state.command_us);
    state.zero_deg = cal.zero_deg;
    state.zero_us = cal.zero_us;
    state.us_per_degree = cal.us_per_degree;
    state.direction = cal.direction;
    state.min_deg = cal.min_deg;
    state.max_deg = cal.max_deg;
    return state;
}

}  // namespace learm

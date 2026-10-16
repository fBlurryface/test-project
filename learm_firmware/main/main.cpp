#include <algorithm>
#include <cmath>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "arm_joint.hpp"
#include "arm_kinematics.hpp"
#include "arm_motion.hpp"
#include "servo_pwm.hpp"

using learm::ArmCartesianPose;
using learm::ArmJoint;
using learm::ArmJointController;
using learm::ArmKinematics;
using learm::ArmKinematicsJointMask;
using learm::ArmKinematicsJointState;
using learm::ArmKinematicsJointVelocity;
using learm::ArmMotion;
using learm::ArmMotionCommand;
using learm::ArmMotionJointVelocity;
using learm::ArmMotionMode;
using learm::ArmMotionPositionCommand;
using learm::ArmMotionState;
using learm::ArmOutputState;
using learm::ArmToolVelocity;
using learm::ArmTaskWeights;
using learm::ArmTeleopPlanarTask;
using learm::ArmJointVelocityWeights;
using learm::JointCalibration;
using learm::ServoPwm;

static const char* TAG = "hierarchical_teleop";
static constexpr const char* kTuningProfile = "hierarchical_tool_pose_v1";

static ServoPwm g_servo;
static ArmJointController g_joints;
static ArmMotion g_motion;
static ArmKinematics g_kinematics;

// Application-level command ownership. Track is the default source. A valid
// pos command temporarily owns ArmMotion until an explicit stop returns
// control to Track. ArmMotionMode still describes command execution itself.
enum class AppControlMode : uint8_t {
    Track = 0,
    Pos,
};

static AppControlMode g_app_control_mode = AppControlMode::Track;
static bool g_track_ignore_reported = false;

// Fast-tracking v1: reach a moderately higher Cartesian speed earlier while
// keeping the existing solver, joint limits, control period and PWM rate.
// jerr inputs are dimensionless normalized errors in [-1, 1].
struct VisualVelocityParameters {
    float forward_gain_mm_per_s = 90.0f;
    float left_gain_mm_per_s = 500.0f;
    float up_gain_mm_per_s = 350.0f;
    float pitch_gain_deg_per_s = 60.0f;

    float max_forward_mm_per_s = 90.0f;
    float max_left_mm_per_s = 240.0f;
    float max_up_mm_per_s = 240.0f;
    float max_pitch_deg_per_s = 60.0f;
};


static constexpr VisualVelocityParameters kVisualVelocity = {};

// New operator-intent protocol: "teleop L Z P F". All fields are normalized
// to [-1, 1]. L keeps the legacy tool-left Cartesian semantics (which this
// mechanism realizes through base rotation). Z is world-vertical motion. P
// trims a persistent target tool pitch. F is motion along the current
// tool-forward axis. Zero Z/F means that Cartesian direction is free, not
// position-held.
struct TeleopVelocityParameters {
    float max_left_mm_per_s = 240.0f;
    float pitch_trim_deg_per_s = 60.0f;
    float pitch_servo_kp_per_s = 8.0f;
    float max_pitch_servo_deg_per_s = 75.0f;
    float max_pitch_target_lead_deg = 12.0f;

    float max_vertical_mm_per_s = 180.0f;
    float max_forward_mm_per_s = 100.0f;

    // Pitch target integration follows command arrival time, but a delayed or
    // restarted sender must not integrate a stale command over a long gap.
    float nominal_command_period_s = 0.020f;
    float max_integrated_period_s = 0.050f;
    float resync_gap_s = 0.150f;

    float active_input_epsilon = 0.0001f;
};

struct TeleopControllerState {
    bool pitch_target_initialized = false;
    float target_pitch_deg = 0.0f;
    int64_t last_command_us = 0;
};

static constexpr TeleopVelocityParameters kTeleopVelocity = {};
static TeleopControllerState g_teleop = {};

// AIM: mouse pointing has priority. Forward/up zero commands are deliberately
// weak so the solver does not contort the whole arm just to keep the tool tip
// nailed to one Cartesian point while pitch changes.
static constexpr ArmTaskWeights kAimTaskWeights = {
    0.02f,  // forward
    1.50f,  // left: mouse horizontal sweep
    0.02f,  // up
    6.00f,  // pitch
    0.00f,  // roll unused
};

// Strongly prefer wrist pitch, then elbow, then shoulder while aiming.
static constexpr ArmJointVelocityWeights kAimJointWeights = {
    1.0f,    // base
    100.0f,  // shoulder: expensive
    25.0f,   // elbow: moderately expensive
    1.0f,    // wrist pitch: preferred
    1.0f,    // wrist roll
};

// MOVE: forward progress dominates. Pitch is still requested, but it may give
// way when translation and orientation cannot be achieved simultaneously.
static constexpr ArmTaskWeights kMoveTaskWeights = {
    8.00f,  // forward
    2.00f,  // left / steering
    0.05f,  // up=0 should not over-constrain the chain
    0.75f,  // pitch is secondary while moving
    0.00f,  // roll unused
};

// Relax joint preference during translation so shoulder/elbow can contribute
// naturally to forward motion.
static constexpr ArmJointVelocityWeights kMoveJointWeights = {
    1.0f,  // base
    2.0f,  // shoulder
    1.5f,  // elbow
    1.0f,  // wrist pitch
    1.0f,  // wrist roll
};

static float smoothstep01(float value)
{
    const float t = std::clamp(value, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static ArmTaskWeights blend_task_weights(
    const ArmTaskWeights& a,
    const ArmTaskWeights& b,
    float t
)
{
    const float u = std::clamp(t, 0.0f, 1.0f);
    ArmTaskWeights result;
    result.forward = a.forward + (b.forward - a.forward) * u;
    result.left = a.left + (b.left - a.left) * u;
    result.up = a.up + (b.up - a.up) * u;
    result.pitch = a.pitch + (b.pitch - a.pitch) * u;
    result.roll = a.roll + (b.roll - a.roll) * u;
    return result;
}

static ArmJointVelocityWeights blend_joint_weights(
    const ArmJointVelocityWeights& a,
    const ArmJointVelocityWeights& b,
    float t
)
{
    const float u = std::clamp(t, 0.0f, 1.0f);
    ArmJointVelocityWeights result;
    result.base = a.base + (b.base - a.base) * u;
    result.shoulder = a.shoulder + (b.shoulder - a.shoulder) * u;
    result.elbow = a.elbow + (b.elbow - a.elbow) * u;
    result.wrist_pitch =
        a.wrist_pitch + (b.wrist_pitch - a.wrist_pitch) * u;
    result.wrist_roll =
        a.wrist_roll + (b.wrist_roll - a.wrist_roll) * u;
    return result;
}

// Infer interaction mode from forward intent without changing the serial
// protocol. F near zero => AIM. Meaningful W/S command => MOVE.
static float move_priority_blend(float forward_error)
{
    const float magnitude = std::fabs(
        std::isfinite(forward_error)
            ? std::clamp(forward_error, -1.0f, 1.0f)
            : 0.0f
    );
    constexpr float kMoveStart = 0.05f;
    constexpr float kMoveFull = 0.25f;
    return smoothstep01(
        (magnitude - kMoveStart) / (kMoveFull - kMoveStart)
    );
}

// The physical arm must be placed near this pose before power-on. This first
// command becomes the motion layer's initial software reference.
static constexpr ArmMotionPositionCommand kInitialPose = {
    {0.0f, -60.0f, 70.0f, 70.0f, 0.0f},
    1.5f,  // recording setup: fixed initial claw opening
};

static void delay_ms(uint32_t ms)
{
    vTaskDelay(pdMS_TO_TICKS(ms));
}

static void trim_line(char* line)
{
    if (line == nullptr) {
        return;
    }

    size_t len = strlen(line);
    while (len > 0 && isspace(static_cast<unsigned char>(line[len - 1]))) {
        line[--len] = '\0';
    }

    char* first = line;
    while (*first != '\0' && isspace(static_cast<unsigned char>(*first))) {
        ++first;
    }
    if (first != line) {
        memmove(line, first, strlen(first) + 1);
    }
}

static bool has_command_prefix(const char* line, const char* command)
{
    if (line == nullptr || command == nullptr) {
        return false;
    }

    const size_t length = strlen(command);
    return strncmp(line, command, length) == 0 &&
           (line[length] == '\0' ||
            isspace(static_cast<unsigned char>(line[length])));
}

static const char* app_control_mode_name(AppControlMode mode)
{
    switch (mode) {
        case AppControlMode::Track: return "track";
        case AppControlMode::Pos:   return "pos";
        default:                    return "unknown";
    }
}

static void reset_teleop_controller()
{
    g_teleop = {};
}

static void enter_track_mode()
{
    g_app_control_mode = AppControlMode::Track;
    g_track_ignore_reported = false;
    reset_teleop_controller();
}

static void enter_pos_mode()
{
    g_app_control_mode = AppControlMode::Pos;
    g_track_ignore_reported = false;
    reset_teleop_controller();
}

static bool parse_position_command(
    const char* line,
    ArmMotionPositionCommand* command
)
{
    if (line == nullptr || command == nullptr) {
        return false;
    }

    int consumed = -1;
    const int matched = sscanf(
        line,
        "pos %f %f %f %f %f %f%n",
        &command->joint_position.base_deg,
        &command->joint_position.shoulder_deg,
        &command->joint_position.elbow_deg,
        &command->joint_position.wrist_pitch_deg,
        &command->joint_position.wrist_roll_deg,
        &command->claw_gap_cm,
        &consumed
    );

    return matched == 6 && consumed >= 0 && line[consumed] == '\0';
}

static bool parse_claw_command(
    const char* line,
    float* gap_cm
)
{
    if (line == nullptr || gap_cm == nullptr) {
        return false;
    }

    int consumed = -1;
    const int matched = sscanf(
        line,
        "claw %f%n",
        gap_cm,
        &consumed
    );

    return matched == 1 &&
           consumed >= 0 &&
           line[consumed] == '\0' &&
           std::isfinite(*gap_cm) &&
           *gap_cm >= 0.0f &&
           *gap_cm <= 5.8f;
}

static bool parse_teleop_command(
    const char* line,
    float* left_intent,
    float* vertical_intent,
    float* pitch_intent,
    float* forward_intent
)
{
    if (line == nullptr || left_intent == nullptr || vertical_intent == nullptr ||
        pitch_intent == nullptr || forward_intent == nullptr) {
        return false;
    }

    int consumed = -1;
    const int matched = sscanf(
        line,
        "teleop %f %f %f %f%n",
        left_intent,
        vertical_intent,
        pitch_intent,
        forward_intent,
        &consumed
    );

    return matched == 4 && consumed >= 0 && line[consumed] == '\0' &&
           std::isfinite(*left_intent) &&
           std::isfinite(*vertical_intent) &&
           std::isfinite(*pitch_intent) &&
           std::isfinite(*forward_intent);
}

static bool parse_normalized_error_command(
    const char* line,
    float* forward_error,
    float* left_error,
    float* up_error,
    float* pitch_error
)
{
    if (line == nullptr || forward_error == nullptr || left_error == nullptr ||
        up_error == nullptr || pitch_error == nullptr) {
        return false;
    }

    int consumed = -1;
    const int matched = sscanf(
        line,
        "jerr %f %f %f %f%n",
        forward_error,
        left_error,
        up_error,
        pitch_error,
        &consumed
    );

    return matched == 4 && consumed >= 0 && line[consumed] == '\0' &&
           std::isfinite(*forward_error) &&
           std::isfinite(*left_error) &&
           std::isfinite(*up_error) &&
           std::isfinite(*pitch_error);
}

static const char* mode_name(ArmMotionMode mode)
{
    switch (mode) {
        case ArmMotionMode::Hold:     return "hold";
        case ArmMotionMode::Position: return "position";
        case ArmMotionMode::Velocity: return "velocity";
        case ArmMotionMode::Stopping: return "stopping";
        default:                      return "unknown";
    }
}

static const char* output_state_name(ArmOutputState state)
{
    switch (state) {
        case ArmOutputState::Disabled:    return "disabled";
        case ArmOutputState::Enabled:     return "enabled";
        case ArmOutputState::DriverError: return "driver_error";
        default:                          return "unknown";
    }
}

static const char* joint_name(ArmJoint joint)
{
    switch (joint) {
        case ArmJoint::Claw:       return "claw";
        case ArmJoint::WristRoll:  return "wrist_roll";
        case ArmJoint::WristPitch: return "wrist_pitch";
        case ArmJoint::Elbow:      return "elbow";
        case ArmJoint::Shoulder:   return "shoulder";
        case ArmJoint::Base:       return "base";
        default:                   return "unknown";
    }
}

static ArmKinematicsJointState kinematics_state_from_motion(const ArmMotionState& state)
{
    ArmKinematicsJointState result;
    result.base_deg = state.base.reference_deg;
    result.shoulder_deg = state.shoulder.reference_deg;
    result.elbow_deg = state.elbow.reference_deg;
    result.wrist_pitch_deg = state.wrist_pitch.reference_deg;
    result.wrist_roll_deg = state.wrist_roll.reference_deg;
    return result;
}

static ArmMotionJointVelocity motion_velocity_from_kinematics(
    const ArmKinematicsJointVelocity& velocity
)
{
    ArmMotionJointVelocity result;
    result.base_deg_per_s = velocity.base_deg_per_s;
    result.shoulder_deg_per_s = velocity.shoulder_deg_per_s;
    result.elbow_deg_per_s = velocity.elbow_deg_per_s;
    result.wrist_pitch_deg_per_s = velocity.wrist_pitch_deg_per_s;
    result.wrist_roll_deg_per_s = velocity.wrist_roll_deg_per_s;
    return result;
}

static float normalized_error(float value)
{
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return std::clamp(value, -1.0f, 1.0f);
}

static ArmToolVelocity tool_velocity_from_error(
    float forward_error,
    float left_error,
    float up_error,
    float pitch_error
)
{
    const float ef = normalized_error(forward_error);
    const float el = normalized_error(left_error);
    const float eu = normalized_error(up_error);
    const float ep = normalized_error(pitch_error);

    ArmToolVelocity velocity;
    velocity.forward_mm_per_s = std::clamp(
        kVisualVelocity.forward_gain_mm_per_s * ef,
        -kVisualVelocity.max_forward_mm_per_s,
         kVisualVelocity.max_forward_mm_per_s
    );
    velocity.left_mm_per_s = std::clamp(
        kVisualVelocity.left_gain_mm_per_s * el,
        -kVisualVelocity.max_left_mm_per_s,
         kVisualVelocity.max_left_mm_per_s
    );
    velocity.up_mm_per_s = std::clamp(
        kVisualVelocity.up_gain_mm_per_s * eu,
        -kVisualVelocity.max_up_mm_per_s,
         kVisualVelocity.max_up_mm_per_s
    );
    velocity.pitch_deg_per_s = std::clamp(
        kVisualVelocity.pitch_gain_deg_per_s * ep,
        -kVisualVelocity.max_pitch_deg_per_s,
         kVisualVelocity.max_pitch_deg_per_s
    );
    velocity.roll_deg_per_s = 0.0f;
    return velocity;
}

static constexpr float kActiveSetLimitEpsilonDeg = 0.001f;
static constexpr float kActiveSetVelocityEpsilonDegPerS = 0.001f;

static bool disable_if_moving_outward_at_limit(
    float reference_deg,
    float velocity_deg_per_s,
    learm::ArmJoint joint,
    bool* enabled
)
{
    if (enabled == nullptr || !*enabled) {
        return false;
    }

    const JointCalibration calibration = g_joints.get_calibration(joint);
    const bool outward_at_min =
        reference_deg <= calibration.min_deg + kActiveSetLimitEpsilonDeg &&
        velocity_deg_per_s < -kActiveSetVelocityEpsilonDegPerS;
    const bool outward_at_max =
        reference_deg >= calibration.max_deg - kActiveSetLimitEpsilonDeg &&
        velocity_deg_per_s > kActiveSetVelocityEpsilonDegPerS;

    if (!outward_at_min && !outward_at_max) {
        return false;
    }

    *enabled = false;
    return true;
}

static bool remove_outward_limited_joints(
    const ArmMotionState& state,
    const ArmKinematicsJointVelocity& velocity,
    ArmKinematicsJointMask* enabled_joints
)
{
    if (enabled_joints == nullptr) {
        return false;
    }

    bool changed = false;
    changed |= disable_if_moving_outward_at_limit(
        state.base.reference_deg,
        velocity.base_deg_per_s,
        learm::ArmJoint::Base,
        &enabled_joints->base
    );
    changed |= disable_if_moving_outward_at_limit(
        state.shoulder.reference_deg,
        velocity.shoulder_deg_per_s,
        learm::ArmJoint::Shoulder,
        &enabled_joints->shoulder
    );
    changed |= disable_if_moving_outward_at_limit(
        state.elbow.reference_deg,
        velocity.elbow_deg_per_s,
        learm::ArmJoint::Elbow,
        &enabled_joints->elbow
    );
    changed |= disable_if_moving_outward_at_limit(
        state.wrist_pitch.reference_deg,
        velocity.wrist_pitch_deg_per_s,
        learm::ArmJoint::WristPitch,
        &enabled_joints->wrist_pitch
    );
    changed |= disable_if_moving_outward_at_limit(
        state.wrist_roll.reference_deg,
        velocity.wrist_roll_deg_per_s,
        learm::ArmJoint::WristRoll,
        &enabled_joints->wrist_roll
    );
    return changed;
}

static void tool_pitch_limits_deg(float* out_min_deg, float* out_max_deg)
{
    if (out_min_deg == nullptr || out_max_deg == nullptr) {
        return;
    }

    const auto kp = g_kinematics.get_parameters();
    const JointCalibration shoulder =
        g_joints.get_calibration(learm::ArmJoint::Shoulder);
    const JointCalibration elbow =
        g_joints.get_calibration(learm::ArmJoint::Elbow);
    const JointCalibration wrist =
        g_joints.get_calibration(learm::ArmJoint::WristPitch);

    auto contribution_bounds = [](float sign, const JointCalibration& cal,
                                  float* lo, float* hi) {
        if (sign >= 0.0f) {
            *lo = sign * cal.min_deg;
            *hi = sign * cal.max_deg;
        } else {
            *lo = sign * cal.max_deg;
            *hi = sign * cal.min_deg;
        }
    };

    float s_lo = 0.0f, s_hi = 0.0f;
    float e_lo = 0.0f, e_hi = 0.0f;
    float w_lo = 0.0f, w_hi = 0.0f;
    contribution_bounds(kp.shoulder_sign, shoulder, &s_lo, &s_hi);
    contribution_bounds(kp.elbow_sign, elbow, &e_lo, &e_hi);
    contribution_bounds(kp.wrist_pitch_sign, wrist, &w_lo, &w_hi);

    const float offset = kp.shoulder_offset_deg +
                         kp.elbow_offset_deg +
                         kp.wrist_pitch_offset_deg;
    *out_min_deg = offset + s_lo + e_lo + w_lo;
    *out_max_deg = offset + s_hi + e_hi + w_hi;
}

static esp_err_t apply_teleop_intent(
    float left_intent,
    float vertical_intent,
    float pitch_intent,
    float forward_intent
)
{
    const ArmMotionState motion_state = g_motion.get_state();
    if (!motion_state.initialized ||
        motion_state.output_state != ArmOutputState::Enabled) {
        return ESP_ERR_INVALID_STATE;
    }

    const ArmKinematicsJointState joints =
        kinematics_state_from_motion(motion_state);
    ArmCartesianPose pose;
    const esp_err_t fk_ret = g_kinematics.forward_pose(joints, &pose);
    if (fk_ret != ESP_OK) {
        return fk_ret;
    }

    const float l = normalized_error(left_intent);
    const float z = normalized_error(vertical_intent);
    const float p = normalized_error(pitch_intent);
    const float f = normalized_error(forward_intent);

    const int64_t now_us = esp_timer_get_time();
    float dt_s = kTeleopVelocity.nominal_command_period_s;
    bool resync_pitch_target = !g_teleop.pitch_target_initialized;
    if (g_teleop.pitch_target_initialized && g_teleop.last_command_us > 0) {
        const float elapsed_s = static_cast<float>(
            now_us - g_teleop.last_command_us) / 1000000.0f;
        if (!std::isfinite(elapsed_s) ||
            elapsed_s < 0.0f ||
            elapsed_s > kTeleopVelocity.resync_gap_s) {
            resync_pitch_target = true;
        } else {
            dt_s = std::clamp(
                elapsed_s,
                0.0f,
                kTeleopVelocity.max_integrated_period_s
            );
        }
    }

    if (resync_pitch_target) {
        g_teleop.target_pitch_deg = pose.tool_pitch_deg;
        g_teleop.pitch_target_initialized = true;
        dt_s = kTeleopVelocity.nominal_command_period_s;
    }
    g_teleop.last_command_us = now_us;

    const float requested_pitch_target_rate =
        p * kTeleopVelocity.pitch_trim_deg_per_s;
    g_teleop.target_pitch_deg += requested_pitch_target_rate * dt_s;

    float pitch_min_deg = -255.0f;
    float pitch_max_deg = 255.0f;
    tool_pitch_limits_deg(&pitch_min_deg, &pitch_max_deg);
    const float unclamped_target = g_teleop.target_pitch_deg;
    g_teleop.target_pitch_deg = std::clamp(
        g_teleop.target_pitch_deg,
        pitch_min_deg,
        pitch_max_deg
    );

    // Do not allow an unreachable target to wind up far away from the actual
    // reference pose. A bounded lead still lets the user continuously sweep
    // pitch while the arm follows at its physical rate.
    g_teleop.target_pitch_deg = std::clamp(
        g_teleop.target_pitch_deg,
        pose.tool_pitch_deg - kTeleopVelocity.max_pitch_target_lead_deg,
        pose.tool_pitch_deg + kTeleopVelocity.max_pitch_target_lead_deg
    );

    const bool absolute_pitch_bound_hit =
        std::fabs(g_teleop.target_pitch_deg - unclamped_target) > 0.0001f &&
        (g_teleop.target_pitch_deg <= pitch_min_deg + 0.0001f ||
         g_teleop.target_pitch_deg >= pitch_max_deg - 0.0001f);
    const float pitch_feedforward = absolute_pitch_bound_hit
        ? 0.0f
        : requested_pitch_target_rate;
    const float pitch_error_deg =
        g_teleop.target_pitch_deg - pose.tool_pitch_deg;
    const float pitch_rate_cmd_deg_per_s = std::clamp(
        pitch_feedforward +
            kTeleopVelocity.pitch_servo_kp_per_s * pitch_error_deg,
        -kTeleopVelocity.max_pitch_servo_deg_per_s,
         kTeleopVelocity.max_pitch_servo_deg_per_s
    );

    ArmTeleopPlanarTask task;
    task.pitch_deg_per_s = pitch_rate_cmd_deg_per_s;
    task.vertical_mm_per_s = z * kTeleopVelocity.max_vertical_mm_per_s;
    task.forward_mm_per_s = f * kTeleopVelocity.max_forward_mm_per_s;
    task.vertical_active =
        std::fabs(z) > kTeleopVelocity.active_input_epsilon;
    task.forward_active =
        std::fabs(f) > kTeleopVelocity.active_input_epsilon;

    const float left_velocity_mm_per_s =
        l * kTeleopVelocity.max_left_mm_per_s;

    ArmKinematicsJointMask enabled_joints;
    ArmKinematicsJointVelocity solved_velocity;
    bool solved = false;

    // Re-use the existing active-set idea at hard joint limits. The planar
    // solver owns shoulder/elbow/wrist-pitch. Lateral motion deliberately
    // keeps the legacy Cartesian-left semantics, solved with only the base
    // column enabled; for this mechanism the left Jacobian row has no
    // shoulder/elbow/wrist-pitch contribution.
    for (int pass = 0; pass < 6; ++pass) {
        const esp_err_t solve_ret = g_kinematics.solve_teleop_velocity(
            joints,
            task,
            enabled_joints,
            &solved_velocity
        );
        if (solve_ret != ESP_OK) {
            g_motion.stop();
            return solve_ret;
        }

        ArmToolVelocity left_task;
        left_task.left_mm_per_s = left_velocity_mm_per_s;

        ArmKinematicsJointMask left_mask;
        left_mask.base = enabled_joints.base;
        left_mask.shoulder = false;
        left_mask.elbow = false;
        left_mask.wrist_pitch = false;
        left_mask.wrist_roll = false;

        ArmTaskWeights left_task_weights;
        left_task_weights.forward = 0.0f;
        left_task_weights.left = 1.5f;
        left_task_weights.up = 0.0f;
        left_task_weights.pitch = 0.0f;
        left_task_weights.roll = 0.0f;

        ArmJointVelocityWeights left_joint_weights;
        ArmKinematicsJointVelocity left_velocity;
        const esp_err_t left_ret = g_kinematics.solve_tool_velocity(
            joints,
            left_task,
            left_mask,
            left_task_weights,
            left_joint_weights,
            &left_velocity
        );
        if (left_ret != ESP_OK) {
            g_motion.stop();
            return left_ret;
        }

        solved_velocity.base_deg_per_s = left_velocity.base_deg_per_s;
        solved_velocity.wrist_roll_deg_per_s = 0.0f;

        if (!remove_outward_limited_joints(
                motion_state,
                solved_velocity,
                &enabled_joints)) {
            solved = true;
            break;
        }
    }

    if (!solved) {
        g_motion.stop();
        return ESP_ERR_INVALID_STATE;
    }

    ArmMotionCommand motion_command;
    motion_command.joint_velocity =
        motion_velocity_from_kinematics(solved_velocity);
    motion_command.claw_gap_cm = motion_state.claw_target_gap_cm;
    return g_motion.command(motion_command);
}

static esp_err_t apply_normalized_error(
    float forward_error,
    float left_error,
    float up_error,
    float pitch_error
)
{
    reset_teleop_controller();
    const ArmMotionState motion_state = g_motion.get_state();
    if (!motion_state.initialized ||
        motion_state.output_state != ArmOutputState::Enabled) {
        return ESP_ERR_INVALID_STATE;
    }

    const ArmKinematicsJointState joints =
        kinematics_state_from_motion(motion_state);
    const ArmToolVelocity tool_velocity = tool_velocity_from_error(
        forward_error,
        left_error,
        up_error,
        pitch_error
    );

    const float move_blend = move_priority_blend(forward_error);
    const ArmTaskWeights task_weights = blend_task_weights(
        kAimTaskWeights,
        kMoveTaskWeights,
        move_blend
    );
    const ArmJointVelocityWeights joint_weights = blend_joint_weights(
        kAimJointWeights,
        kMoveJointWeights,
        move_blend
    );

    ArmKinematicsJointMask enabled_joints;
    ArmKinematicsJointVelocity solved_velocity;
    bool solved = false;

    // Small active-set loop: if the current solution asks a joint already at
    // its limit to move farther outward, constrain that joint to zero and run
    // the same Jacobian solver again with the remaining joints. Multiple
    // joints may be removed in one pass. Six solves are sufficient for five
    // rotary joints, including the final all-constrained solution.
    for (int pass = 0; pass < 6; ++pass) {
        const esp_err_t solve_ret = g_kinematics.solve_tool_velocity(
            joints,
            tool_velocity,
            enabled_joints,
            task_weights,
            joint_weights,
            &solved_velocity
        );
        if (solve_ret != ESP_OK) {
            g_motion.stop();
            return solve_ret;
        }

        if (!remove_outward_limited_joints(
                motion_state,
                solved_velocity,
                &enabled_joints)) {
            solved = true;
            break;
        }
    }

    if (!solved) {
        g_motion.stop();
        return ESP_ERR_INVALID_STATE;
    }

    ArmMotionCommand motion_command;
    motion_command.joint_velocity =
        motion_velocity_from_kinematics(solved_velocity);
    motion_command.claw_gap_cm = motion_state.claw_target_gap_cm;
    return g_motion.command(motion_command);
}

static void print_help()
{
    printf("\n=== hierarchical tool-pose teleoperation firmware ===\n");
    printf("Tuning profile: %s\n", kTuningProfile);
    printf("Primary teleoperation protocol:\n");
    printf("  teleop L Z P F\n");
    printf("    L/Z/P/F are normalized intents in [-1, 1].\n");
    printf("    L: legacy Cartesian-left velocity; realized by base rotation.\n");
    printf("    Z: world-vertical velocity; zero makes vertical translation free.\n");
    printf("    P: trim persistent target tool pitch; zero keeps last target pitch.\n");
    printf("    F: velocity along current tool-forward axis; zero makes it free.\n");
    printf("    Priority: safety > pitch > requested translation > drift suppression.\n");
    printf("    Z-only softly minimizes radial drift; F-only softly minimizes tool-normal drift.\n");
    printf("    Z+F share the translation level while pitch remains hard priority.\n");
    printf("    Accepted in Track mode; refreshes the velocity watchdog.\n\n");

    printf("Legacy compatibility protocol:\n");
    printf("  jerr F L U P\n");
    printf("    Legacy weighted-DLS normalized-error controller; kept for A/B testing.\n\n");

    printf("Claw teleoperation:\n");
    printf("  claw GAP\n");
    printf("    Set only the claw target gap in cm [0.0, 5.8].\n");
    printf("    Does not interrupt rotary-joint velocity tracking.\n\n");

    printf("Position override:\n");
    printf("  pos B S E WP WR GAP\n");
    printf("    Move to absolute rotary joint angles in degrees and claw gap in cm.\n");
    printf("    A successful pos enters Pos mode; teleop/jerr are ignored until stop.\n");
    printf("  stop       stop current motion and return to Track mode\n");
    printf("  state      print references, targets, watchdog and teleop pitch target\n");
    printf("  fk         print forward pose from reference positions\n");
    printf("  enable 0   when disabled, enable at calibration zero with claw closed\n");
    printf("  disable    immediately disable PWM outputs\n");
    printf("  help       print this help\n\n");

    printf("Teleop limits: L=%.1f mm/s, Z=%.1f mm/s, pitch-trim=%.1f deg/s, F=%.1f mm/s.\n",
           static_cast<double>(kTeleopVelocity.max_left_mm_per_s),
           static_cast<double>(kTeleopVelocity.max_vertical_mm_per_s),
           static_cast<double>(kTeleopVelocity.pitch_trim_deg_per_s),
           static_cast<double>(kTeleopVelocity.max_forward_mm_per_s));
}

static void print_state()
{
    const ArmMotionState s = g_motion.get_state();
    const char* error_joint = s.driver_error_joint_valid
        ? joint_name(s.driver_error_joint)
        : "none";
    printf("initialized=%d app=%s pwm=%s mode=%s reference_moving=%d fresh=%d timeout=%d age=%lums claw_ref=%.2f claw_target=%.2f driver_error=0x%x error_joint=%s\n",
           s.initialized,
           app_control_mode_name(g_app_control_mode),
           output_state_name(s.output_state),
           mode_name(s.mode),
           s.reference_moving,
           s.command_fresh,
           s.command_timed_out,
           static_cast<unsigned long>(s.command_age_ms),
           static_cast<double>(s.claw_reference_gap_cm),
           static_cast<double>(s.claw_target_gap_cm),
           static_cast<unsigned>(s.last_driver_error),
           error_joint);
    if (g_teleop.pitch_target_initialized) {
        printf("teleop target_pitch=%.3f deg last_rx=%lld us\n",
               static_cast<double>(g_teleop.target_pitch_deg),
               static_cast<long long>(g_teleop.last_command_us));
    } else {
        printf("teleop target_pitch=unsynced\n");
    }

    auto print_joint = [](const char* name, const learm::ArmMotionReferenceState& joint) {
        printf("  %-12s q_ref=%8.3f  q_target=%8.3f  v_target=%8.3f  v_applied=%8.3f ref_reached=%d limit=%d\n",
               name,
               static_cast<double>(joint.reference_deg),
               static_cast<double>(joint.target_position_deg),
               static_cast<double>(joint.target_velocity_deg_per_s),
               static_cast<double>(joint.applied_velocity_deg_per_s),
               joint.reference_reached,
               joint.limit_blocked);
    };

    print_joint("base", s.base);
    print_joint("shoulder", s.shoulder);
    print_joint("elbow", s.elbow);
    print_joint("wrist_pitch", s.wrist_pitch);
    print_joint("wrist_roll", s.wrist_roll);
}

static void print_fk()
{
    const ArmMotionState motion_state = g_motion.get_state();
    const ArmKinematicsJointState joints =
        kinematics_state_from_motion(motion_state);

    ArmCartesianPose pose;
    const esp_err_t ret = g_kinematics.forward_pose(joints, &pose);
    if (ret != ESP_OK) {
        printf("forward_pose ret=0x%x\n", static_cast<unsigned>(ret));
        return;
    }

    printf("pose x=%.2f y=%.2f z=%.2f pitch=%.2f roll=%.2f\n",
           static_cast<double>(pose.x_mm),
           static_cast<double>(pose.y_mm),
           static_cast<double>(pose.z_mm),
           static_cast<double>(pose.tool_pitch_deg),
           static_cast<double>(pose.tool_roll_deg));
}

static void process_command(char* line)
{
    trim_line(line);
    if (line[0] == '\0') {
        return;
    }

    if (strcmp(line, "help") == 0) {
        print_help();
        return;
    }
    if (strcmp(line, "state") == 0) {
        print_state();
        return;
    }
    if (strcmp(line, "fk") == 0) {
        print_fk();
        return;
    }
    if (strcmp(line, "stop") == 0) {
        const esp_err_t ret = g_motion.stop();
        enter_track_mode();
        printf("stop ret=0x%x app=%s\n",
               static_cast<unsigned>(ret),
               app_control_mode_name(g_app_control_mode));
        return;
    }
    if (strcmp(line, "enable 0") == 0) {
        const esp_err_t ret = g_motion.enable_all_at_zero_closed();
        if (ret == ESP_OK) {
            enter_track_mode();
        }
        printf(
            "enable 0 ret=0x%x app=%s\n",
            static_cast<unsigned>(ret),
            app_control_mode_name(g_app_control_mode)
        );
        return;
    }
    if (strcmp(line, "disable") == 0) {
        const esp_err_t ret = g_motion.disable_all();
        enter_track_mode();
        printf("disable ret=0x%x app=%s\n",
               static_cast<unsigned>(ret),
               app_control_mode_name(g_app_control_mode));
        return;
    }

    float claw_gap_cm = 0.0f;
    if (parse_claw_command(line, &claw_gap_cm)) {
        const esp_err_t ret = g_motion.set_claw_gap(claw_gap_cm);
        printf(
            "claw gap=%.2f ret=0x%x app=%s\n",
            static_cast<double>(claw_gap_cm),
            static_cast<unsigned>(ret),
            app_control_mode_name(g_app_control_mode)
        );
        return;
    }
    if (has_command_prefix(line, "claw")) {
        printf("invalid claw; expected: claw GAP where GAP is 0.0..5.8 cm\n");
        return;
    }

    ArmMotionPositionCommand position_command;
    if (parse_position_command(line, &position_command)) {
        const esp_err_t ret = g_motion.move_to(position_command);
        if (ret == ESP_OK) {
            enter_pos_mode();
        }
        printf("pos ret=0x%x app=%s\n",
               static_cast<unsigned>(ret),
               app_control_mode_name(g_app_control_mode));
        return;
    }
    if (has_command_prefix(line, "pos")) {
        printf("invalid pos; expected: pos B S E WP WR GAP\n");
        return;
    }

    float left_intent = 0.0f;
    float vertical_intent = 0.0f;
    float pitch_intent = 0.0f;
    float forward_intent = 0.0f;
    if (parse_teleop_command(
            line,
            &left_intent,
            &vertical_intent,
            &pitch_intent,
            &forward_intent)) {
        if (g_app_control_mode != AppControlMode::Track) {
            if (!g_track_ignore_reported) {
                printf("teleop ignored: app=pos; send stop to resume tracking\n");
                g_track_ignore_reported = true;
            }
            return;
        }

        const esp_err_t ret = apply_teleop_intent(
            left_intent,
            vertical_intent,
            pitch_intent,
            forward_intent
        );
        if (ret != ESP_OK) {
            printf("teleop failed ret=0x%x; app=%s\n",
                   static_cast<unsigned>(ret),
                   app_control_mode_name(g_app_control_mode));
        }
        return;
    }
    if (has_command_prefix(line, "teleop")) {
        printf("invalid teleop; expected: teleop L Z P F\n");
        return;
    }

    float forward_error = 0.0f;
    float left_error = 0.0f;
    float up_error = 0.0f;
    float pitch_error = 0.0f;
    if (parse_normalized_error_command(
            line,
            &forward_error,
            &left_error,
            &up_error,
            &pitch_error)) {
        if (g_app_control_mode != AppControlMode::Track) {
            if (!g_track_ignore_reported) {
                printf("jerr ignored: app=pos; send stop to resume tracking\n");
                g_track_ignore_reported = true;
            }
            return;
        }

        const esp_err_t ret = apply_normalized_error(
            forward_error,
            left_error,
            up_error,
            pitch_error
        );
        if (ret != ESP_OK) {
            printf("jerr failed ret=0x%x; app=%s\n",
                   static_cast<unsigned>(ret),
                   app_control_mode_name(g_app_control_mode));
        }
        return;
    }
    if (has_command_prefix(line, "jerr")) {
        printf("invalid jerr; expected: jerr F L U P\n");
        return;
    }

    printf("unknown command; type help\n");
}


// Streaming serial input framing.
//
// The ESP-IDF console stdin can behave as a byte stream / non-blocking stream.
// Under sustained traffic a stdio read may therefore expose a partial command.
// Keep bytes here until an actual '\n' arrives, then call process_command()
// exactly once with one complete command.
//
// '\r' is ignored, so both LF and CRLF senders are accepted.
// If a frame grows beyond the command buffer, discard it until the next '\n'
// to re-synchronize cleanly.
static constexpr size_t kCommandLineCapacity = 160;
static constexpr size_t kSerialReadChunkSize = 64;

struct CommandLineAccumulator {
    char line[kCommandLineCapacity] = {};
    size_t length = 0;
    bool discarding = false;
};

static void reset_command_line(CommandLineAccumulator* rx)
{
    if (rx == nullptr) {
        return;
    }

    rx->length = 0;
    rx->discarding = false;
    rx->line[0] = '\0';
}

static void feed_command_bytes(
    CommandLineAccumulator* rx,
    const char* data,
    size_t size
)
{
    if (rx == nullptr || data == nullptr) {
        return;
    }

    for (size_t i = 0; i < size; ++i) {
        const char c = data[i];

        if (c == '\r') {
            continue;
        }

        if (c == '\n') {
            if (!rx->discarding && rx->length > 0) {
                rx->line[rx->length] = '\0';
                process_command(rx->line);
            } else if (rx->discarding) {
                ESP_LOGW(TAG, "discarded overlong serial command");
            }

            reset_command_line(rx);
            continue;
        }

        if (rx->discarding) {
            continue;
        }

        if (rx->length + 1 >= kCommandLineCapacity) {
            rx->discarding = true;
            continue;
        }

        rx->line[rx->length++] = c;
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "starting hierarchical tool-pose teleoperation firmware");
    ESP_LOGI(TAG, "tuning profile=%s", kTuningProfile);

    ESP_ERROR_CHECK(g_servo.init(
        learm::arm_joint_defaults::kServoPwmConfigs,
        learm::arm_joint_defaults::kServoPwmConfigCount
    ));

    ESP_ERROR_CHECK(g_joints.init(
        &g_servo,
        learm::arm_joint_defaults::kJointCalibrations,
        learm::arm_joint_defaults::kJointCalibrationCount
    ));

    ESP_ERROR_CHECK(g_motion.init(&g_joints));
    ESP_ERROR_CHECK(g_kinematics.init());

    ESP_LOGW(TAG, "servo outputs remain disabled for 3 seconds");
    delay_ms(3000);
    ESP_ERROR_CHECK(g_motion.enable_all_at(kInitialPose));
    enter_track_mode();

    print_help();
    print_state();

    CommandLineAccumulator command_rx;
    char rx_chunk[kSerialReadChunkSize];

    while (true) {
        const ssize_t received =
            read(STDIN_FILENO, rx_chunk, sizeof(rx_chunk));

        if (received > 0) {
            feed_command_bytes(
                &command_rx,
                rx_chunk,
                static_cast<size_t>(received)
            );
            continue;
        }

        // With a non-blocking console, EAGAIN/EWOULDBLOCK simply means that
        // there are currently no bytes available.
        if (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            ESP_LOGE(TAG, "stdin read failed errno=%d", errno);
            delay_ms(20);
            continue;
        }

        // Avoid a busy loop while keeping teleoperation latency low.
        delay_ms(2);
    }
}

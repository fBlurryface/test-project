#pragma once

#include "esp_err.h"

namespace learm {

struct ArmKinematicsParameters {
    // Geometry in millimeters.
    float base_to_shoulder_z_mm = 28.9f;
    float upper_arm_mm = 104.3f;
    float forearm_mm = 89.0f;
    float wrist_to_tool_mm = 177.0f;

    // model_angle = semantic_angle * sign + offset
    float base_sign = 1.0f;
    float shoulder_sign = 1.0f;
    float elbow_sign = 1.0f;
    float wrist_pitch_sign = 1.0f;
    float wrist_roll_sign = 1.0f;

    float base_offset_deg = 0.0f;
    float shoulder_offset_deg = 0.0f;
    float elbow_offset_deg = 0.0f;
    float wrist_pitch_offset_deg = 0.0f;
    float wrist_roll_offset_deg = 0.0f;

    // Damped least-squares parameters. Angular task rows are converted to an
    // equivalent linear scale using orientation_weight_mm.
    float damping_mm = 10.0f;
    float orientation_weight_mm = 150.0f;

    // Output reference-velocity clamps.
    float max_base_deg_per_s = 100.0f;
    float max_shoulder_deg_per_s = 70.0f;
    float max_elbow_deg_per_s = 70.0f;
    float max_wrist_pitch_deg_per_s = 85.0f;
    float max_wrist_roll_deg_per_s = 60.0f;

    // Small damping used for the user-requested translation inside the
    // pitch-preserving null space. It only becomes noticeable close to a
    // singular direction, where demanding exact Cartesian speed would require
    // impractically large joint velocity.
    float teleop_primary_damping_mm = 2.0f;

    // Teleoperation secondary-objective damping. The primary teleop tasks
    // (tool pitch and the user-requested translation) are solved first. This
    // damping only affects the final "minimize uncommanded drift" step so the
    // controller will accept some drift instead of demanding extreme joint
    // rates near a singular configuration.
    float teleop_secondary_damping_mm = 12.0f;
};

struct ArmKinematicsJointState {
    float base_deg = 0.0f;
    float shoulder_deg = 0.0f;
    float elbow_deg = 0.0f;
    float wrist_pitch_deg = 0.0f;
    float wrist_roll_deg = 0.0f;
};

struct ArmCartesianPose {
    float x_mm = 0.0f;
    float y_mm = 0.0f;
    float z_mm = 0.0f;
    float tool_pitch_deg = 0.0f;
    float tool_roll_deg = 0.0f;
};

// Desired instantaneous tool twist expressed in the current tool frame.
struct ArmToolVelocity {
    float forward_mm_per_s = 0.0f;
    float left_mm_per_s = 0.0f;
    float up_mm_per_s = 0.0f;
    float pitch_deg_per_s = 0.0f;
    float roll_deg_per_s = 0.0f;
};

// Relative task importance for weighted resolved-rate IK.
// Zero means "do not care about this task row"; larger values make task error
// more expensive. These are dimensionless because orientation rows have
// already been converted to equivalent millimeters.
struct ArmTaskWeights {
    float forward = 1.0f;
    float left = 1.0f;
    float up = 1.0f;
    float pitch = 1.0f;
    float roll = 1.0f;
};

// Relative joint-motion regularization cost.
// Larger values discourage motion of that joint without making it a hard
// constraint. Hard joint disabling is still handled by ArmKinematicsJointMask.
struct ArmJointVelocityWeights {
    float base = 1.0f;
    float shoulder = 1.0f;
    float elbow = 1.0f;
    float wrist_pitch = 1.0f;
    float wrist_roll = 1.0f;
};


// Planar teleoperation task for shoulder/elbow/wrist-pitch. Lateral motion
// is intentionally handled outside this solver using the legacy Cartesian-left
// task with only the base joint enabled. This keeps left/right control
// independent from the planar pitch/Z/forward hierarchy.
//
// Tool pitch is always a primary task. Vertical and forward are optional: a
// false *_active flag means that direction is FREE, not "hold zero velocity".
struct ArmTeleopPlanarTask {
    float pitch_deg_per_s = 0.0f;
    float vertical_mm_per_s = 0.0f;
    float forward_mm_per_s = 0.0f;
    bool vertical_active = false;
    bool forward_active = false;
};

struct ArmKinematicsJointVelocity {
    float base_deg_per_s = 0.0f;
    float shoulder_deg_per_s = 0.0f;
    float elbow_deg_per_s = 0.0f;
    float wrist_pitch_deg_per_s = 0.0f;
    float wrist_roll_deg_per_s = 0.0f;
};

// Selects which rotary joints may participate in the resolved-rate solution.
// Disabled joints are constrained to zero reference velocity while the same
// solver redistributes the task among the remaining joints.
struct ArmKinematicsJointMask {
    bool base = true;
    bool shoulder = true;
    bool elbow = true;
    bool wrist_pitch = true;
    bool wrist_roll = true;
};

class ArmKinematics {
public:
    esp_err_t init();

    bool is_initialized() const;
    ArmKinematicsParameters get_parameters() const;

    esp_err_t forward_pose(
        const ArmKinematicsJointState& joints,
        ArmCartesianPose* out_pose
    ) const;

    // Default weighted DLS with all task and joint weights equal to one:
    //
    // q_dot = argmin ||J q_dot - x_dot||^2 + lambda^2 ||q_dot||^2
    //
    // q_dot is a commanded/reference velocity, not measured servo velocity.
    esp_err_t solve_tool_velocity(
        const ArmKinematicsJointState& joints,
        const ArmToolVelocity& tool_velocity,
        ArmKinematicsJointVelocity* out_velocity
    ) const;

    // Same calculation with selected joints constrained to zero.
    esp_err_t solve_tool_velocity(
        const ArmKinematicsJointState& joints,
        const ArmToolVelocity& tool_velocity,
        const ArmKinematicsJointMask& enabled_joints,
        ArmKinematicsJointVelocity* out_velocity
    ) const;

    // Weighted regularized least-squares resolved-rate controller:
    //
    // q_dot = argmin
    //   (J q_dot - x_dot)^T W_task (J q_dot - x_dot)
    //   + lambda^2 q_dot^T W_joint q_dot
    //
    // W_task controls task priority; W_joint controls which joints are
    // preferred. The joint mask remains a hard zero-velocity constraint.
    esp_err_t solve_tool_velocity(
        const ArmKinematicsJointState& joints,
        const ArmToolVelocity& tool_velocity,
        const ArmKinematicsJointMask& enabled_joints,
        const ArmTaskWeights& task_weights,
        const ArmJointVelocityWeights& joint_weights,
        ArmKinematicsJointVelocity* out_velocity
    ) const;

    // Hierarchical teleoperation solver for the 3-DOF planar chain
    // [shoulder, elbow, wrist_pitch]. Priority is explicit rather than encoded
    // by weighted task rows:
    //
    //   1. Preserve/servo tool pitch.
    //   2. Realize the user-requested vertical and/or tool-forward motion.
    //   3. With one translation task active, softly minimize the remaining
    //      uncommanded drift (radial drift for vertical motion, tool-normal
    //      drift for forward motion).
    //
    // Disabled joints are hard-zero constraints. Base and wrist-roll outputs
    // are always zero here; the application combines the planar result with
    // its independent lateral/base solution.
    esp_err_t solve_teleop_velocity(
        const ArmKinematicsJointState& joints,
        const ArmTeleopPlanarTask& task,
        const ArmKinematicsJointMask& enabled_joints,
        ArmKinematicsJointVelocity* out_velocity
    ) const;

private:
    static bool finite(float value);
    static float deg_to_rad(float value);
    static float rad_to_deg(float value);
    static float dot3(const float a[3], const float b[3]);
    static bool solve_5x5(const float a[5][5], const float b[5], float x[5]);

    void compute_axes(
        const ArmKinematicsJointState& joints,
        float forward[3], float left[3], float up[3]
    ) const;

    void compute_position_jacobian(
        const ArmKinematicsJointState& joints,
        ArmCartesianPose* out_pose,
        float out_position_jacobian[3][5]
    ) const;

private:
    bool initialized_ = false;
    ArmKinematicsParameters parameters_ = {};
};

}  // namespace learm

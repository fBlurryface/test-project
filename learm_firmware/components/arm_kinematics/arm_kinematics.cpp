#include "arm_kinematics.hpp"

#include <algorithm>
#include <cmath>

#include "esp_log.h"

namespace learm {

static const char* TAG = "arm_velocity_kinematics";
static constexpr float kPi = 3.14159265358979323846f;

namespace {

constexpr float kTeleopVectorEpsilon = 1.0e-7f;

float dot_vec3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float norm_sq_vec3(const float v[3])
{
    return dot_vec3(v, v);
}

void add_scaled_vec3(float dst[3], const float src[3], float scale)
{
    for (int i = 0; i < 3; ++i) {
        dst[i] += src[i] * scale;
    }
}

// Build an orthonormal basis for the active-joint null space of the supplied
// hard task rows. Disabled planar joints never appear in the basis.
int build_nullspace_basis(
    const float hard_rows[3][3],
    int hard_row_count,
    const bool active[3],
    float out_basis[3][3]
)
{
    float row_basis[3][3] = {};
    int row_rank = 0;

    for (int row = 0; row < hard_row_count && row < 3; ++row) {
        float v[3] = {
            active[0] ? hard_rows[row][0] : 0.0f,
            active[1] ? hard_rows[row][1] : 0.0f,
            active[2] ? hard_rows[row][2] : 0.0f,
        };

        for (int k = 0; k < row_rank; ++k) {
            const float projection = dot_vec3(v, row_basis[k]);
            add_scaled_vec3(v, row_basis[k], -projection);
        }

        const float n2 = norm_sq_vec3(v);
        if (n2 <= kTeleopVectorEpsilon) {
            continue;
        }

        const float inv_n = 1.0f / std::sqrt(n2);
        for (int i = 0; i < 3; ++i) {
            row_basis[row_rank][i] = v[i] * inv_n;
        }
        ++row_rank;
    }

    int null_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (!active[axis]) {
            continue;
        }

        float v[3] = {};
        v[axis] = 1.0f;

        for (int k = 0; k < row_rank; ++k) {
            const float projection = dot_vec3(v, row_basis[k]);
            add_scaled_vec3(v, row_basis[k], -projection);
        }
        for (int k = 0; k < null_count; ++k) {
            const float projection = dot_vec3(v, out_basis[k]);
            add_scaled_vec3(v, out_basis[k], -projection);
        }

        const float n2 = norm_sq_vec3(v);
        if (n2 <= kTeleopVectorEpsilon) {
            continue;
        }

        const float inv_n = 1.0f / std::sqrt(n2);
        for (int i = 0; i < 3; ++i) {
            out_basis[null_count][i] = v[i] * inv_n;
        }
        ++null_count;
    }

    return null_count;
}

// Solve a damped least-squares correction inside an already-established
// null space. task_count and basis_count are at most two for the teleop
// hierarchy used below.
bool add_nullspace_dls_correction(
    const float task_rows[2][3],
    const float task_targets[2],
    int task_count,
    const float basis[3][3],
    int basis_count,
    float damping,
    float q[3]
)
{
    if (task_count <= 0 || basis_count <= 0) {
        return false;
    }

    float b[2][3] = {};
    float residual[2] = {};
    for (int task = 0; task < task_count; ++task) {
        residual[task] = task_targets[task] - dot_vec3(task_rows[task], q);
        for (int k = 0; k < basis_count; ++k) {
            b[task][k] = dot_vec3(task_rows[task], basis[k]);
        }
    }

    float normal[3][3] = {};
    float rhs[3] = {};
    for (int i = 0; i < basis_count; ++i) {
        for (int task = 0; task < task_count; ++task) {
            rhs[i] += b[task][i] * residual[task];
            for (int k = 0; k < basis_count; ++k) {
                normal[i][k] += b[task][i] * b[task][k];
            }
        }
        normal[i][i] += damping * damping;
    }

    float coefficient[3] = {};
    if (basis_count == 1) {
        if (normal[0][0] <= kTeleopVectorEpsilon) {
            return false;
        }
        coefficient[0] = rhs[0] / normal[0][0];
    } else if (basis_count == 2) {
        const float det = normal[0][0] * normal[1][1] -
                          normal[0][1] * normal[1][0];
        if (std::fabs(det) <= kTeleopVectorEpsilon) {
            return false;
        }
        coefficient[0] =
            (rhs[0] * normal[1][1] - normal[0][1] * rhs[1]) / det;
        coefficient[1] =
            (normal[0][0] * rhs[1] - rhs[0] * normal[1][0]) / det;
    } else {
        return false;
    }

    for (int k = 0; k < basis_count; ++k) {
        add_scaled_vec3(q, basis[k], coefficient[k]);
    }
    return true;
}

}  // namespace

bool ArmKinematics::finite(float value)
{
    return std::isfinite(value);
}

float ArmKinematics::deg_to_rad(float value)
{
    return value * (kPi / 180.0f);
}

float ArmKinematics::rad_to_deg(float value)
{
    return value * (180.0f / kPi);
}

float ArmKinematics::dot3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

bool ArmKinematics::solve_5x5(const float a[5][5], const float b[5], float x[5])
{
    float m[5][6] = {};
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 5; ++col) {
            m[row][col] = a[row][col];
        }
        m[row][5] = b[row];
    }

    for (int col = 0; col < 5; ++col) {
        int pivot = col;
        float pivot_abs = std::fabs(m[col][col]);
        for (int row = col + 1; row < 5; ++row) {
            const float candidate = std::fabs(m[row][col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        if (pivot_abs < 1.0e-9f) {
            return false;
        }

        if (pivot != col) {
            for (int k = col; k < 6; ++k) {
                std::swap(m[col][k], m[pivot][k]);
            }
        }

        const float divisor = m[col][col];
        for (int k = col; k < 6; ++k) {
            m[col][k] /= divisor;
        }

        for (int row = 0; row < 5; ++row) {
            if (row == col) {
                continue;
            }
            const float factor = m[row][col];
            for (int k = col; k < 6; ++k) {
                m[row][k] -= factor * m[col][k];
            }
        }
    }

    for (int i = 0; i < 5; ++i) {
        x[i] = m[i][5];
        if (!finite(x[i])) {
            return false;
        }
    }
    return true;
}

esp_err_t ArmKinematics::init()
{
    if (parameters_.upper_arm_mm <= 0.0f ||
        parameters_.forearm_mm <= 0.0f ||
        parameters_.wrist_to_tool_mm < 0.0f ||
        parameters_.damping_mm <= 0.0f ||
        parameters_.orientation_weight_mm <= 0.0f ||
        parameters_.teleop_primary_damping_mm <= 0.0f ||
        parameters_.teleop_secondary_damping_mm <= 0.0f) {
        return ESP_ERR_INVALID_STATE;
    }

    initialized_ = true;
    ESP_LOGI(TAG,
             "weighted velocity Jacobian initialized links=[%.1f %.1f %.1f] damping=%.1f orientation_weight=%.1f",
             static_cast<double>(parameters_.upper_arm_mm),
             static_cast<double>(parameters_.forearm_mm),
             static_cast<double>(parameters_.wrist_to_tool_mm),
             static_cast<double>(parameters_.damping_mm),
             static_cast<double>(parameters_.orientation_weight_mm));
    return ESP_OK;
}

bool ArmKinematics::is_initialized() const
{
    return initialized_;
}

ArmKinematicsParameters ArmKinematics::get_parameters() const
{
    return parameters_;
}

void ArmKinematics::compute_axes(
    const ArmKinematicsJointState& joints,
    float forward[3], float left[3], float up[3]
) const
{
    const float base = deg_to_rad(
        joints.base_deg * parameters_.base_sign + parameters_.base_offset_deg
    );
    const float shoulder = deg_to_rad(
        joints.shoulder_deg * parameters_.shoulder_sign + parameters_.shoulder_offset_deg
    );
    const float elbow = deg_to_rad(
        joints.elbow_deg * parameters_.elbow_sign + parameters_.elbow_offset_deg
    );
    const float wrist = deg_to_rad(
        joints.wrist_pitch_deg * parameters_.wrist_pitch_sign + parameters_.wrist_pitch_offset_deg
    );

    const float pitch = shoulder + elbow + wrist;
    const float cb = std::cos(base);
    const float sb = std::sin(base);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);

    forward[0] = cb * sp;
    forward[1] = sb * sp;
    forward[2] = cp;

    left[0] = -sb;
    left[1] = cb;
    left[2] = 0.0f;

    up[0] = -cb * cp;
    up[1] = -sb * cp;
    up[2] = sp;
}

void ArmKinematics::compute_position_jacobian(
    const ArmKinematicsJointState& joints,
    ArmCartesianPose* out_pose,
    float out_position_jacobian[3][5]
) const
{
    const float base = deg_to_rad(
        joints.base_deg * parameters_.base_sign + parameters_.base_offset_deg
    );
    const float shoulder = deg_to_rad(
        joints.shoulder_deg * parameters_.shoulder_sign + parameters_.shoulder_offset_deg
    );
    const float elbow = deg_to_rad(
        joints.elbow_deg * parameters_.elbow_sign + parameters_.elbow_offset_deg
    );
    const float wrist = deg_to_rad(
        joints.wrist_pitch_deg * parameters_.wrist_pitch_sign + parameters_.wrist_pitch_offset_deg
    );

    const float a1 = shoulder;
    const float a2 = shoulder + elbow;
    const float a3 = shoulder + elbow + wrist;

    const float cb = std::cos(base);
    const float sb = std::sin(base);
    const float c1 = std::cos(a1);
    const float s1 = std::sin(a1);
    const float c2 = std::cos(a2);
    const float s2 = std::sin(a2);
    const float c3 = std::cos(a3);
    const float s3 = std::sin(a3);

    const float l1 = parameters_.upper_arm_mm;
    const float l2 = parameters_.forearm_mm;
    const float l3 = parameters_.wrist_to_tool_mm;

    const float radius = l1 * s1 + l2 * s2 + l3 * s3;
    const float z = parameters_.base_to_shoulder_z_mm +
                    l1 * c1 + l2 * c2 + l3 * c3;

    if (out_pose != nullptr) {
        out_pose->x_mm = radius * cb;
        out_pose->y_mm = radius * sb;
        out_pose->z_mm = z;
        out_pose->tool_pitch_deg = rad_to_deg(a3);
        out_pose->tool_roll_deg =
            joints.wrist_roll_deg * parameters_.wrist_roll_sign +
            parameters_.wrist_roll_offset_deg;
    }

    if (out_position_jacobian == nullptr) {
        return;
    }

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 5; ++col) {
            out_position_jacobian[row][col] = 0.0f;
        }
    }

    const float dr_ds = l1 * c1 + l2 * c2 + l3 * c3;
    const float dz_ds = -l1 * s1 - l2 * s2 - l3 * s3;
    const float dr_de = l2 * c2 + l3 * c3;
    const float dz_de = -l2 * s2 - l3 * s3;
    const float dr_dw = l3 * c3;
    const float dz_dw = -l3 * s3;

    out_position_jacobian[0][0] = -radius * sb * parameters_.base_sign;
    out_position_jacobian[1][0] =  radius * cb * parameters_.base_sign;

    out_position_jacobian[0][1] = dr_ds * cb * parameters_.shoulder_sign;
    out_position_jacobian[1][1] = dr_ds * sb * parameters_.shoulder_sign;
    out_position_jacobian[2][1] = dz_ds * parameters_.shoulder_sign;

    out_position_jacobian[0][2] = dr_de * cb * parameters_.elbow_sign;
    out_position_jacobian[1][2] = dr_de * sb * parameters_.elbow_sign;
    out_position_jacobian[2][2] = dz_de * parameters_.elbow_sign;

    out_position_jacobian[0][3] = dr_dw * cb * parameters_.wrist_pitch_sign;
    out_position_jacobian[1][3] = dr_dw * sb * parameters_.wrist_pitch_sign;
    out_position_jacobian[2][3] = dz_dw * parameters_.wrist_pitch_sign;
}

esp_err_t ArmKinematics::forward_pose(
    const ArmKinematicsJointState& joints,
    ArmCartesianPose* out_pose
) const
{
    if (!initialized_) {
        return ESP_ERR_INVALID_STATE;
    }
    if (out_pose == nullptr ||
        !finite(joints.base_deg) ||
        !finite(joints.shoulder_deg) ||
        !finite(joints.elbow_deg) ||
        !finite(joints.wrist_pitch_deg) ||
        !finite(joints.wrist_roll_deg)) {
        return ESP_ERR_INVALID_ARG;
    }

    compute_position_jacobian(joints, out_pose, nullptr);
    return ESP_OK;
}

esp_err_t ArmKinematics::solve_tool_velocity(
    const ArmKinematicsJointState& joints,
    const ArmToolVelocity& tool_velocity,
    ArmKinematicsJointVelocity* out_velocity
) const
{
    return solve_tool_velocity(
        joints,
        tool_velocity,
        ArmKinematicsJointMask{},
        ArmTaskWeights{},
        ArmJointVelocityWeights{},
        out_velocity
    );
}

esp_err_t ArmKinematics::solve_tool_velocity(
    const ArmKinematicsJointState& joints,
    const ArmToolVelocity& tool_velocity,
    const ArmKinematicsJointMask& enabled_joints,
    ArmKinematicsJointVelocity* out_velocity
) const
{
    return solve_tool_velocity(
        joints,
        tool_velocity,
        enabled_joints,
        ArmTaskWeights{},
        ArmJointVelocityWeights{},
        out_velocity
    );
}

esp_err_t ArmKinematics::solve_tool_velocity(
    const ArmKinematicsJointState& joints,
    const ArmToolVelocity& tool_velocity,
    const ArmKinematicsJointMask& enabled_joints,
    const ArmTaskWeights& task_weights,
    const ArmJointVelocityWeights& joint_weights,
    ArmKinematicsJointVelocity* out_velocity
) const
{
    if (!initialized_) {
        return ESP_ERR_INVALID_STATE;
    }

    const float task_weight_array[5] = {
        task_weights.forward,
        task_weights.left,
        task_weights.up,
        task_weights.pitch,
        task_weights.roll,
    };
    const float joint_weight_array[5] = {
        joint_weights.base,
        joint_weights.shoulder,
        joint_weights.elbow,
        joint_weights.wrist_pitch,
        joint_weights.wrist_roll,
    };

    if (out_velocity == nullptr ||
        !finite(joints.base_deg) ||
        !finite(joints.shoulder_deg) ||
        !finite(joints.elbow_deg) ||
        !finite(joints.wrist_pitch_deg) ||
        !finite(joints.wrist_roll_deg) ||
        !finite(tool_velocity.forward_mm_per_s) ||
        !finite(tool_velocity.left_mm_per_s) ||
        !finite(tool_velocity.up_mm_per_s) ||
        !finite(tool_velocity.pitch_deg_per_s) ||
        !finite(tool_velocity.roll_deg_per_s)) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < 5; ++i) {
        if (!finite(task_weight_array[i]) || task_weight_array[i] < 0.0f ||
            !finite(joint_weight_array[i]) || joint_weight_array[i] <= 0.0f) {
            return ESP_ERR_INVALID_ARG;
        }
    }

    float j_position[3][5] = {};
    compute_position_jacobian(joints, nullptr, j_position);

    float forward[3] = {};
    float left[3] = {};
    float up[3] = {};
    compute_axes(joints, forward, left, up);

    float j[5][5] = {};
    for (int col = 0; col < 5; ++col) {
        const float column[3] = {
            j_position[0][col],
            j_position[1][col],
            j_position[2][col],
        };
        j[0][col] = dot3(forward, column);
        j[1][col] = dot3(left, column);
        j[2][col] = dot3(up, column);
    }

    const float orientation_weight = parameters_.orientation_weight_mm;
    j[3][1] = orientation_weight * parameters_.shoulder_sign;
    j[3][2] = orientation_weight * parameters_.elbow_sign;
    j[3][3] = orientation_weight * parameters_.wrist_pitch_sign;
    j[4][4] = orientation_weight * parameters_.wrist_roll_sign;

    const bool joint_enabled[5] = {
        enabled_joints.base,
        enabled_joints.shoulder,
        enabled_joints.elbow,
        enabled_joints.wrist_pitch,
        enabled_joints.wrist_roll,
    };

    // Hard-disabled joints get zero Jacobian columns. Their right-hand side is
    // therefore zero and positive regularization keeps qdot exactly zero.
    for (int joint = 0; joint < 5; ++joint) {
        if (joint_enabled[joint]) {
            continue;
        }
        for (int task_row = 0; task_row < 5; ++task_row) {
            j[task_row][joint] = 0.0f;
        }
    }

    const float task[5] = {
        tool_velocity.forward_mm_per_s,
        tool_velocity.left_mm_per_s,
        tool_velocity.up_mm_per_s,
        orientation_weight * deg_to_rad(tool_velocity.pitch_deg_per_s),
        orientation_weight * deg_to_rad(tool_velocity.roll_deg_per_s),
    };

    // Weighted regularized least-squares normal equations:
    //
    //   (J^T Wt J + lambda^2 Wq) qdot = J^T Wt task
    //
    // With Wt = I and Wq = I this is the standard damped least-squares
    // solution, written in joint-space form so task and joint preferences can
    // be represented explicitly.
    float normal[5][5] = {};
    float rhs[5] = {};

    for (int row = 0; row < 5; ++row) {
        const float weight = task_weight_array[row];
        if (weight == 0.0f) {
            continue;
        }

        for (int i = 0; i < 5; ++i) {
            rhs[i] += weight * j[row][i] * task[row];
            for (int k = 0; k < 5; ++k) {
                normal[i][k] += weight * j[row][i] * j[row][k];
            }
        }
    }

    const float lambda2 =
        parameters_.damping_mm * parameters_.damping_mm;
    for (int joint = 0; joint < 5; ++joint) {
        normal[joint][joint] += lambda2 * joint_weight_array[joint];
    }

    float qdot_rad_per_s[5] = {};
    if (!solve_5x5(normal, rhs, qdot_rad_per_s)) {
        *out_velocity = {};
        return ESP_ERR_INVALID_STATE;
    }

    const float limits[5] = {
        parameters_.max_base_deg_per_s,
        parameters_.max_shoulder_deg_per_s,
        parameters_.max_elbow_deg_per_s,
        parameters_.max_wrist_pitch_deg_per_s,
        parameters_.max_wrist_roll_deg_per_s,
    };

    float qdot_deg_per_s[5] = {};
    float velocity_scale = 1.0f;
    for (int i = 0; i < 5; ++i) {
        qdot_deg_per_s[i] = rad_to_deg(qdot_rad_per_s[i]);
        if (!finite(qdot_deg_per_s[i])) {
            *out_velocity = {};
            return ESP_ERR_INVALID_STATE;
        }

        const float magnitude = std::fabs(qdot_deg_per_s[i]);
        const float limit = std::fabs(limits[i]);
        if (magnitude > limit && magnitude > 0.0f) {
            velocity_scale = std::min(velocity_scale, limit / magnitude);
        }
    }

    // Preserve the resolved-rate vector direction when any joint reaches its
    // reference-velocity limit.
    for (float& value : qdot_deg_per_s) {
        value *= velocity_scale;
    }

    out_velocity->base_deg_per_s = qdot_deg_per_s[0];
    out_velocity->shoulder_deg_per_s = qdot_deg_per_s[1];
    out_velocity->elbow_deg_per_s = qdot_deg_per_s[2];
    out_velocity->wrist_pitch_deg_per_s = qdot_deg_per_s[3];
    out_velocity->wrist_roll_deg_per_s = qdot_deg_per_s[4];
    return ESP_OK;
}


esp_err_t ArmKinematics::solve_teleop_velocity(
    const ArmKinematicsJointState& joints,
    const ArmTeleopPlanarTask& task,
    const ArmKinematicsJointMask& enabled_joints,
    ArmKinematicsJointVelocity* out_velocity
) const
{
    if (!initialized_) {
        return ESP_ERR_INVALID_STATE;
    }
    if (out_velocity == nullptr ||
        !finite(joints.base_deg) ||
        !finite(joints.shoulder_deg) ||
        !finite(joints.elbow_deg) ||
        !finite(joints.wrist_pitch_deg) ||
        !finite(joints.wrist_roll_deg) ||
        !finite(task.pitch_deg_per_s) ||
        !finite(task.vertical_mm_per_s) ||
        !finite(task.forward_mm_per_s)) {
        return ESP_ERR_INVALID_ARG;
    }

    *out_velocity = {};

    // The teleop hierarchy deliberately excludes base yaw and wrist roll.
    // Their application-level semantics are direct and independent.
    const bool active[3] = {
        enabled_joints.shoulder,
        enabled_joints.elbow,
        enabled_joints.wrist_pitch,
    };

    float j_position[3][5] = {};
    ArmCartesianPose pose;
    compute_position_jacobian(joints, &pose, j_position);

    const float base = deg_to_rad(
        joints.base_deg * parameters_.base_sign + parameters_.base_offset_deg
    );
    const float cb = std::cos(base);
    const float sb = std::sin(base);
    const float pitch = deg_to_rad(pose.tool_pitch_deg);
    const float sp = std::sin(pitch);
    const float cp = std::cos(pitch);

    // Rows are expressed against semantic joint velocities in rad/s.
    float j_pitch[3] = {
        parameters_.shoulder_sign,
        parameters_.elbow_sign,
        parameters_.wrist_pitch_sign,
    };
    float j_radial[3] = {};
    float j_vertical[3] = {};
    for (int i = 0; i < 3; ++i) {
        const int col = i + 1;  // shoulder, elbow, wrist-pitch
        j_radial[i] = cb * j_position[0][col] + sb * j_position[1][col];
        j_vertical[i] = j_position[2][col];
    }

    float j_forward[3] = {};
    float j_tool_normal[3] = {};
    for (int i = 0; i < 3; ++i) {
        // Tool forward has planar components [sin(pitch), cos(pitch)] in
        // [radial, world-Z]. Tool-normal/up is the perpendicular direction.
        j_forward[i] = sp * j_radial[i] + cp * j_vertical[i];
        j_tool_normal[i] = -cp * j_radial[i] + sp * j_vertical[i];
    }

    float qdot[3] = {};
    float hard_rows[3][3] = {};
    int hard_row_count = 0;

    // Priority 1: tool pitch. Use a weighted minimum-motion particular
    // solution so pitch-only mouse input mostly uses wrist pitch, then elbow,
    // then shoulder. Later hierarchy levels may redistribute motion, but can
    // never change the achieved pitch rate.
    const float pitch_target = deg_to_rad(task.pitch_deg_per_s);
    const float pitch_cost[3] = {100.0f, 25.0f, 1.0f};
    float pitch_denom = 0.0f;
    for (int i = 0; i < 3; ++i) {
        if (!active[i]) {
            continue;
        }
        pitch_denom += (j_pitch[i] * j_pitch[i]) / pitch_cost[i];
    }
    if (pitch_denom > kTeleopVectorEpsilon) {
        for (int i = 0; i < 3; ++i) {
            if (!active[i]) {
                continue;
            }
            qdot[i] =
                (j_pitch[i] / pitch_cost[i]) * (pitch_target / pitch_denom);
        }
        for (int i = 0; i < 3; ++i) {
            hard_rows[hard_row_count][i] = j_pitch[i];
        }
        ++hard_row_count;
    }

    const bool vertical_active = task.vertical_active;
    const bool forward_active = task.forward_active;

    if (vertical_active || forward_active) {
        float basis[3][3] = {};
        const int basis_count = build_nullspace_basis(
            hard_rows,
            hard_row_count,
            active,
            basis
        );

        if (vertical_active && forward_active) {
            // Two simultaneous user translation intents share the same level.
            // Pitch remains hard-priority; a small DLS term prevents explosive
            // joint rates when the two planar directions become singular.
            float rows[2][3] = {};
            float targets[2] = {
                task.vertical_mm_per_s,
                task.forward_mm_per_s,
            };
            for (int i = 0; i < 3; ++i) {
                rows[0][i] = j_vertical[i];
                rows[1][i] = j_forward[i];
            }
            add_nullspace_dls_correction(
                rows,
                targets,
                2,
                basis,
                basis_count,
                parameters_.teleop_primary_damping_mm,
                qdot
            );
        } else {
            // Exactly one user translation axis is active. Solve it first,
            // then reserve its row as a hard direction for the lower-priority
            // drift minimizer.
            const float* primary_row = vertical_active ? j_vertical : j_forward;
            const float primary_target = vertical_active
                ? task.vertical_mm_per_s
                : task.forward_mm_per_s;

            float rows[2][3] = {};
            float targets[2] = {primary_target, 0.0f};
            for (int i = 0; i < 3; ++i) {
                rows[0][i] = primary_row[i];
            }

            const bool corrected = add_nullspace_dls_correction(
                rows,
                targets,
                1,
                basis,
                basis_count,
                parameters_.teleop_primary_damping_mm,
                qdot
            );

            if (corrected && hard_row_count < 3) {
                for (int i = 0; i < 3; ++i) {
                    hard_rows[hard_row_count][i] = primary_row[i];
                }
                ++hard_row_count;

                // Priority 3: suppress motion the user did not ask for, but
                // only softly. Vertical motion minimizes radial drift;
                // forward motion minimizes tool-normal drift. Damping here is
                // intentionally much larger than the primary damping so the
                // arm accepts drift instead of sacrificing usable motion near
                // singular configurations.
                float drift_basis[3][3] = {};
                const int drift_basis_count = build_nullspace_basis(
                    hard_rows,
                    hard_row_count,
                    active,
                    drift_basis
                );
                const float* drift_row = vertical_active
                    ? j_radial
                    : j_tool_normal;
                float drift_rows[2][3] = {};
                float drift_targets[2] = {0.0f, 0.0f};
                for (int i = 0; i < 3; ++i) {
                    drift_rows[0][i] = drift_row[i];
                }
                add_nullspace_dls_correction(
                    drift_rows,
                    drift_targets,
                    1,
                    drift_basis,
                    drift_basis_count,
                    parameters_.teleop_secondary_damping_mm,
                    qdot
                );

            }
        }
    }

    const float limits_deg_per_s[3] = {
        parameters_.max_shoulder_deg_per_s,
        parameters_.max_elbow_deg_per_s,
        parameters_.max_wrist_pitch_deg_per_s,
    };
    float qdot_deg_per_s[3] = {};
    float scale = 1.0f;
    for (int i = 0; i < 3; ++i) {
        if (!active[i]) {
            qdot[i] = 0.0f;
        }
        qdot_deg_per_s[i] = rad_to_deg(qdot[i]);
        if (!finite(qdot_deg_per_s[i])) {
            *out_velocity = {};
            return ESP_ERR_INVALID_STATE;
        }

        const float magnitude = std::fabs(qdot_deg_per_s[i]);
        const float limit = std::fabs(limits_deg_per_s[i]);
        if (limit > 0.0f && magnitude > limit) {
            scale = std::min(scale, limit / magnitude);
        }
    }

    for (float& value : qdot_deg_per_s) {
        value *= scale;
    }

    out_velocity->shoulder_deg_per_s = qdot_deg_per_s[0];
    out_velocity->elbow_deg_per_s = qdot_deg_per_s[1];
    out_velocity->wrist_pitch_deg_per_s = qdot_deg_per_s[2];
    return ESP_OK;
}

}  // namespace learm

from dataclasses import dataclass, field


@dataclass
class ControllerConfig:
    control_hz: float = 50.0
    kp_position: float = 8.0
    kp_orientation: float = 3.0
    damping_min: float = 0.01
    damping_max: float = 0.18
    singular_value_soft: float = 0.06
    orientation_damping: float = 0.08
    nullspace_gain: float = 0.20
    joint_limit_avoid_gain: float = 0.045
    joint_limit_margin_rad: float = 0.08
    max_joint_speed_rad_s: float = 1.5
    max_translation_step_m: float = 0.025
    max_rotation_step_rad: float = 0.15
    max_cartesian_speed_m_s: float = 0.25
    max_angular_speed_rad_s: float = 1.5
    max_servo_lead_rad: float = 0.06
    # Firmware-style STREAM reference generator.
    # Uploaded LeArm firmware: ArmMotion update_period_ms=20, command_timeout_ms=300.
    command_timeout_s: float = 0.300

    # Panda-specific joint envelope.  We copy the LeArm *architecture* (target qdot
    # -> acceleration ramp -> integrated q_reference), not LeArm's 5-joint limits.
    # Values are deliberately conservative and exposed here for tuning.
    stream_max_joint_speed_rad_s: tuple[float, ...] = (
        1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50
    )
    # 5.2--9.1 rad/s^2 is the range in the uploaded LeArm firmware; 7 rad/s^2
    # is a middle-of-range Panda default while preserving a common vector ramp.
    stream_max_joint_accel_rad_s2: tuple[float, ...] = (
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0
    )
    stream_velocity_epsilon_rad_s: float = 0.005


@dataclass
class TaskConfig:
    # Start pose is solved kinematically at reset.  The Panda gripper/tool centreline
    # (+Z_hand in the Menagerie model) points horizontally along world +X.
    start_hand_pos_world: tuple[float, float, float] = (0.42, 0.0, 0.46)
    start_ik_position_tol_m: float = 0.002
    start_ik_orientation_tol_rad: float = 0.035
    start_ik_max_iters: int = 320
    # Must be >= controller joint-limit margin; otherwise reset can place the arm
    # in a pose that the runtime controller immediately considers unsafe.
    start_ik_joint_margin_rad: float = 0.12
    start_ik_posture_gain: float = 0.08

    # Rod is physically COLLINEAR with the gripper centre axis (+Z_hand).
    rod_start_m: float = 0.105
    rod_tip_m: float = 0.245
    rod_radius_m: float = 0.0055
    # The blue cue is not a separate sphere. The distal part of
    # the physical rod itself is painted blue, matching the real setup more closely.
    blue_terminal_length_m: float = 0.025

    # Eye-in-hand camera: mounted slightly above the gripper centreline, aimed at
    # the painted rod tip.  Its optical axis therefore passes through the blue tip,
    # but the camera is not physically hidden behind the rod.
    camera_up_offset_m: float = 0.040
    camera_forward_m: float = 0.070
    camera_width: int = 1280
    camera_height: int = 960
    camera_fovy_deg: float = 55.0

    # Keep the validated robot/camera/controller unchanged, while making
    # the ring smaller and chunkier so post-insertion occlusion is easier to obtain.
    # Occlusion setting: keep outer diameter at 100 mm, but thicken the
    # ring so the learned late-stage blue-above-center configuration is more
    # likely to produce a real image-space occlusion.
    # Inner diameter = 40 mm, tube diameter = 30 mm, outer diameter = 100 mm.
    ring_inner_radius_m: float = 0.020
    ring_tube_radius_m: float = 0.015
    ring_segments: int = 48
    ring_density_kg_m3: float = 550.0
    ring_stabilizer_half_width_m: float = 0.028
    ring_stabilizer_half_height_m: float = 0.004
    ring_stabilizer_half_depth_m: float = 0.022
    ring_stabilizer_density_kg_m3: float = 220.0
    ring_ahead_of_tip_m: float = 0.14
    ring_lateral_offset_m: float = 0.0
    ring_vertical_offset_m: float = 0.0

    table_half_size_xy_m: tuple[float, float] = (0.34, 0.27)
    table_half_thickness_m: float = 0.018
    table_leg_half_width_m: float = 0.022
    table_leg_half_height_m: float = 0.18


@dataclass
class SimConfig:
    physics_timestep_s: float = 0.002
    settle_s: float = 0.8
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

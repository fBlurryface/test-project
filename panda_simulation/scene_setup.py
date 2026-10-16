from __future__ import annotations

import math
import numpy as np
import mujoco
import mujoco_menagerie as mm

from .simulation_config import SimConfig
from .geometry import matrix_to_quat_wxyz


# Task frame expressed in Panda hand coordinates:
#   +X_task = side  = +Y_hand
#   +Y_task = up    = -X_hand
#   +Z_task = fwd   = +Z_hand  (gripper / rod centreline)
TASK_R_HAND = np.array(
    [[0.0, -1.0, 0.0],
     [1.0,  0.0, 0.0],
     [0.0,  0.0, 1.0]],
    dtype=float,
)


# Collision categories for the task scene.
# Menagerie Panda collision geoms and the rod use MuJoCo's default bit 1.
# The dynamic ring uses bit 1 and *accepts* bits 1|2 so it can contact both
# the robot/rod and the table.  The table uses bit 2 and accepts no incoming
# category, so it contacts the ring (because ring conaffinity includes bit 2)
# but does NOT contact Panda/rod geoms.
#
# Diagnostic setting: remove table-induced
# robot motion while preserving ring-table support and robot-ring interaction.
ROBOT_COLLISION_BIT = 1
TABLE_COLLISION_BIT = 2
RING_CONAFFINITY_BITS = ROBOT_COLLISION_BIT | TABLE_COLLISION_BIT


def desired_start_hand_rotation_world() -> np.ndarray:
    """Hand +Z points world +X; jaw-opening +Y is world +Y; -X is world up."""
    x_hand = np.array([0.0, 0.0, -1.0])
    y_hand = np.array([0.0, 1.0, 0.0])
    z_hand = np.array([1.0, 0.0, 0.0])
    return np.column_stack([x_hand, y_hand, z_hand])


def _add_ring_segments(body, inner_radius: float, tube_radius: float, segments: int, density: float | None = None) -> None:
    center_radius = inner_radius + tube_radius
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        p0 = [center_radius * math.cos(a0), center_radius * math.sin(a0), 0.0]
        p1 = [center_radius * math.cos(a1), center_radius * math.sin(a1), 0.0]
        body.add_geom(
            name=f"ring_seg_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*p0, *p1],
            size=[tube_radius, 0.0, 0.0],
            rgba=[0.95, 0.03, 0.03, 1.0],
            contype=ROBOT_COLLISION_BIT,
            conaffinity=RING_CONAFFINITY_BITS,
            friction=[0.9, 0.02, 0.002],
            **({} if density is None else {"density": float(density)}),
        )


def build_model(cfg: SimConfig):
    spec = mm.get("franka_emika_panda").spec()
    spec.option.timestep = cfg.physics_timestep_s
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), int(cfg.task.camera_width))
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), int(cfg.task.camera_height))

    t = cfg.task
    hand = spec.body("hand")
    if hand is None:
        raise RuntimeError("Menagerie Panda body 'hand' not found")

    # ------------------------------------------------------------------
    # Tool: rod is on the actual gripper centreline, Panda hand +Z.
    # ------------------------------------------------------------------
    rod_start = np.array([0.0, 0.0, t.rod_start_m])
    rod_tip = np.array([0.0, 0.0, t.rod_tip_m])
    blue_start = np.array([0.0, 0.0, t.rod_tip_m - t.blue_terminal_length_m])

    # Physical rod, split only for colour.  There is NO separate blue marker ball:
    # the final section of the actual collision rod is painted blue.
    hand.add_geom(
        name="task_rod_dark",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*rod_start.tolist(), *blue_start.tolist()],
        size=[t.rod_radius_m, 0.0, 0.0],
        rgba=[0.10, 0.10, 0.12, 1.0],
        contype=1,
        conaffinity=1,
        friction=[0.8, 0.02, 0.002],
    )
    hand.add_geom(
        name="task_rod_blue_terminal",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*blue_start.tolist(), *rod_tip.tolist()],
        size=[t.rod_radius_m, 0.0, 0.0],
        # RGB hue ~= OpenCV H=112, inside the real detector's H=105..116 band.
        rgba=[0.0, 0.267, 1.0, 1.0],
        contype=1,
        conaffinity=1,
        friction=[0.8, 0.02, 0.002],
    )

    # Task site orientation is policy-friendly, but its +Z is exactly hand +Z.
    hand.add_site(
        name="tool_tip_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=rod_tip.tolist(),
        quat=matrix_to_quat_wxyz(TASK_R_HAND).tolist(),
        size=[0.002, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )

    # ------------------------------------------------------------------
    # Eye-in-hand camera.  At the solved start pose, world-up is -X_hand.
    # Put the camera above the gripper and aim at the real rod tip.
    # ------------------------------------------------------------------
    up_h = np.array([-1.0, 0.0, 0.0])
    camera_pos = np.array([-t.camera_up_offset_m, 0.0, t.camera_forward_m])
    optical_forward_h = rod_tip - camera_pos
    optical_forward_h /= np.linalg.norm(optical_forward_h)
    cam_z_h = -optical_forward_h                 # MuJoCo camera looks along -Z
    cam_x_h = np.cross(up_h, cam_z_h)
    cam_x_h /= np.linalg.norm(cam_x_h)
    cam_y_h = np.cross(cam_z_h, cam_x_h)
    cam_y_h /= np.linalg.norm(cam_y_h)
    cam_R_hand = np.column_stack([cam_x_h, cam_y_h, cam_z_h])
    cam_quat = matrix_to_quat_wxyz(cam_R_hand)

    # Small visible camera housing, purely cosmetic.
    hand.add_geom(
        name="wrist_camera_housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=camera_pos.tolist(),
        quat=cam_quat.tolist(),
        size=[0.022, 0.016, 0.012],
        rgba=[0.08, 0.08, 0.09, 1.0],
        contype=0,
        conaffinity=0,
    )
    hand.add_camera(
        name="wrist_camera",
        pos=camera_pos.tolist(),
        quat=cam_quat.tolist(),
        resolution=[t.camera_width, t.camera_height],
        fovy=t.camera_fovy_deg,
    )

    # Invisible diagnostic site at the true distal rod endpoint.  The blue detector
    # now sees only the painted physical rod segment above, never a synthetic sphere.
    hand.add_site(
        name="blue_marker_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=rod_tip.tolist(),
        size=[0.001, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )

    # ------------------------------------------------------------------
    # Task objects.
    # ------------------------------------------------------------------
    # Robot/camera/controller remain on the validated setup.
    # Only tool paint and ring geometry are simplified for natural occlusion.
    # The ring is a genuine free rigid body. Robot/camera/controller are untouched.
    ring = spec.worldbody.add_body(name="ring_dynamic")
    ring.add_freejoint(name="ring_free")
    _add_ring_segments(
        ring, t.ring_inner_radius_m, t.ring_tube_radius_m, t.ring_segments,
        density=t.ring_density_kg_m3,
    )
    ring.add_site(
        name="ring_center_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.002, 0.0, 0.0],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )
    # Small dark stabilizer rigidly attached to the ring. It is deliberately
    # non-red and only exists to let the free ring stand upright on the table.
    outer = t.ring_inner_radius_m + 2.0 * t.ring_tube_radius_m
    hh = t.ring_stabilizer_half_height_m
    ring.add_geom(
        name="ring_stabilizer_foot",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, -(outer + hh), 0.0],
        size=[t.ring_stabilizer_half_width_m, hh, t.ring_stabilizer_half_depth_m],
        rgba=[0.12, 0.12, 0.12, 1.0],
        contype=ROBOT_COLLISION_BIT,
        conaffinity=RING_CONAFFINITY_BITS,
        friction=[1.1, 0.02, 0.002],
        density=float(t.ring_stabilizer_density_kg_m3),
    )

    table = spec.worldbody.add_body(name="table_mocap", mocap=True)
    table.add_geom(
        name="task_table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[t.table_half_size_xy_m[0], t.table_half_size_xy_m[1], t.table_half_thickness_m],
        rgba=[0.42, 0.28, 0.17, 1.0],
        contype=TABLE_COLLISION_BIT,
        conaffinity=0,
        friction=[1.0, 0.01, 0.001],
    )
    # Cosmetic/physical legs so the table does not visibly float.
    hx, hy = t.table_half_size_xy_m
    lw = t.table_leg_half_width_m
    lh = t.table_leg_half_height_m
    leg_z = -(t.table_half_thickness_m + lh)
    for k, (sx, sy) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):
        table.add_geom(
            name=f"table_leg_{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sx*(hx-0.045), sy*(hy-0.045), leg_z],
            size=[lw, lw, lh],
            rgba=[0.30, 0.20, 0.12, 1.0],
            contype=TABLE_COLLISION_BIT,
            conaffinity=0,
            friction=[1.0, 0.01, 0.001],
        )

    # Neutral visual backdrop behind the task.  This intentionally removes the
    # blue Menagerie/sky background from the wrist-camera field of view so the
    # real blue-marker detector cannot be contaminated by the environment.
    spec.worldbody.add_geom(
        name="vision_backdrop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.30, 0.0, 0.80],
        size=[0.025, 1.50, 1.20],
        rgba=[0.97, 0.97, 0.97, 1.0],
        contype=0,
        conaffinity=0,
    )

    spec.worldbody.add_geom(
        name="task_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[2.0, 2.0, 0.05],
        rgba=[0.94, 0.94, 0.94, 1.0],
        contype=1,
        conaffinity=1,
        friction=[1.0, 0.01, 0.001],
    )

    model = spec.compile()

    # Simulate the gravity-compensated low-level behavior expected from a Franka.
    for name in ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            model.body_gravcomp[bid] = 1.0
    return model

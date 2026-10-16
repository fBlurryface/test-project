from __future__ import annotations

import time
import numpy as np
import mujoco

from .simulation_config import SimConfig
from .task_space_controller import CartesianDLSController
from .geometry import matrix_to_quat_wxyz, matrix_to_rotvec
from .scene_setup import build_model, desired_start_hand_rotation_world
from .simulation_vision import extract_sim_observation
from .visual_observation import ObservationBuilder, PolicyObservation
from .policy_interface import validate_action


class PandaRingEnv:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.cfg.task.camera_height,
            width=self.cfg.task.camera_width,
        )

        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.tip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip_site")
        self.blue_marker_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "blue_marker_site")
        self.ring_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ring_dynamic")
        self.ring_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ring_free")
        self.table_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table_mocap")
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if min(self.tip_site, self.blue_marker_site, self.ring_body, self.ring_joint, self.table_body, self.hand_body) < 0:
            raise RuntimeError("Task elements missing after model compilation")

        self.ring_qpos_adr = int(self.model.jnt_qposadr[self.ring_joint])
        self.ring_dof_adr = int(self.model.jnt_dofadr[self.ring_joint])
        self.table_mocap = int(self.model.body_mocapid[self.table_body])
        if self.table_mocap < 0:
            raise RuntimeError("Task table was expected to be a mocap body")
        self._ring_reset_z = float("nan")

        self.controller = None
        self.observation_builder = ObservationBuilder(
            self.cfg.task.camera_width, self.cfg.task.camera_height
        )
        self._control_accum = 0.0
        self._last_diag = None
        self.reset()

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.cfg.controller.control_hz

    def reset(self):
        if self.home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # Keep the curated Menagerie home configuration as the redundancy/posture
        # reference.  The task-facing IK may rotate the hand substantially, but the
        # spare Panda DOF should prefer this comfortable configuration rather than
        # hug a joint limit.
        joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ], dtype=int)
        nominal_qadr = self.model.jnt_qposadr[joint_ids].astype(int)
        nominal_q = self.data.qpos[nominal_qadr].copy()

        self._solve_visual_start_pose()
        mujoco.mj_forward(self.model, self.data)

        # Create controller at the task-facing start configuration.
        self.controller = CartesianDLSController(self.model, self.data, self.cfg.controller)
        self.controller.reset()
        self.controller.set_posture_reference(nominal_q)
        self._place_task_from_home_tool()
        mujoco.mj_forward(self.model, self.data)

        # Settle while actively holding the home configuration. This catches force-limit
        # issues immediately rather than letting a sagging arm contaminate later tests.
        n = max(1, int(round(self.cfg.settle_s / self.physics_dt)))
        for _ in range(n):
            self.controller.step(self.physics_dt)
            mujoco.mj_step(self.model, self.data)
        self.controller.hold_current_pose()
        self.controller.q_home = self.controller._arm_q().copy()
        self.controller.q_cmd = self.controller.q_home.copy()
        self._ring_reset_z = float(self.data.xpos[self.ring_body][2])
        self.observation_builder.reset(*self.tip_pose())
        self._control_accum = 0.0
        return self.debug_state()

    def _solve_visual_start_pose(self):
        """Kinematically orient the gripper centreline horizontally toward the task.

        This changes the Panda posture; it never rotates the rod away from the gripper.
        The rod remains +Z_hand at all times.
        """
        t = self.cfg.task
        desired_p = np.asarray(t.start_hand_pos_world, dtype=float)
        desired_R = desired_start_hand_rotation_world()

        joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ], dtype=int)
        qadr = self.model.jnt_qposadr[joint_ids].astype(int)
        dadr = self.model.jnt_dofadr[joint_ids].astype(int)
        raw_min = self.model.jnt_range[joint_ids, 0].copy()
        raw_max = self.model.jnt_range[joint_ids, 1].copy()
        safe_margin = max(
            float(t.start_ik_joint_margin_rad),
            float(self.cfg.controller.joint_limit_margin_rad) + 0.02,
        )
        qmin = raw_min + safe_margin
        qmax = raw_max - safe_margin
        q_seed = np.clip(self.data.qpos[qadr].copy(), qmin, qmax)

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        converged = False
        for _ in range(int(t.start_ik_max_iters)):
            mujoco.mj_forward(self.model, self.data)
            p = self.data.xpos[self.hand_body].copy()
            R = self.data.xmat[self.hand_body].reshape(3, 3).copy()
            ep = desired_p - p
            er = matrix_to_rotvec(desired_R @ R.T)
            if (np.linalg.norm(ep) <= t.start_ik_position_tol_m
                    and np.linalg.norm(er) <= t.start_ik_orientation_tol_rad):
                converged = True
                break

            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body)
            Jp = jacp[:, dadr]
            Jr = jacr[:, dadr]
            # Position/orientation weighted DLS kinematic solve.
            wrot = 0.65
            J = np.vstack([Jp, wrot * Jr])
            e = np.concatenate([ep, wrot * er])
            damping = 0.045
            Jsharp = J.T @ np.linalg.solve(J @ J.T + (damping*damping)*np.eye(6), np.eye(6))
            dq = Jsharp @ e

            # Use the Panda redundancy during the start solve instead of allowing
            # an arbitrary near-limit elbow solution.
            N = np.eye(7) - Jsharp @ J
            q_now = self.data.qpos[qadr].copy()
            dq += N @ (float(t.start_ik_posture_gain) * (q_seed - q_now))

            max_step = 0.10
            m = float(np.max(np.abs(dq)))
            if m > max_step:
                dq *= max_step / m
            q = np.clip(self.data.qpos[qadr] + dq, qmin, qmax)
            self.data.qpos[qadr] = q
            self.data.qvel[dadr] = 0.0

        mujoco.mj_forward(self.model, self.data)
        p = self.data.xpos[self.hand_body].copy()
        R = self.data.xmat[self.hand_body].reshape(3, 3).copy()
        ep = float(np.linalg.norm(desired_p - p))
        er = float(np.linalg.norm(matrix_to_rotvec(desired_R @ R.T)))
        q_final = self.data.qpos[qadr].copy()
        final_margin = np.minimum(q_final - raw_min, raw_max - q_final)
        if float(np.min(final_margin)) < float(self.cfg.controller.joint_limit_margin_rad) - 1e-4:
            raise RuntimeError(
                "Start IK produced a pose inside the runtime joint-limit margin: "
                f"min_margin={float(np.min(final_margin)):.4f} rad, "
                f"required={self.cfg.controller.joint_limit_margin_rad:.4f} rad"
            )
        if not converged and (ep > 0.015 or er > 0.15):
            raise RuntimeError(
                f"Could not solve horizontal gripper start pose: "
                f"position_error={ep:.4f} m orientation_error={er:.3f} rad"
            )

    def _set_ring_free_pose(self, position_world, rotation_world):
        qadr = self.ring_qpos_adr
        dadr = self.ring_dof_adr
        self.data.qpos[qadr:qadr+3] = np.asarray(position_world, dtype=float)
        self.data.qpos[qadr+3:qadr+7] = matrix_to_quat_wxyz(np.asarray(rotation_world, dtype=float))
        self.data.qvel[dadr:dadr+6] = 0.0

    def _place_task_from_home_tool(self):
        t = self.cfg.task
        tip_p = self.data.site_xpos[self.tip_site].copy()
        tip_R = self.data.site_xmat[self.tip_site].reshape(3, 3).copy()

        # Explicit task frame: +X=side, +Y=up, +Z=rod forward.
        side = tip_R[:, 0]
        up = tip_R[:, 1]
        forward = tip_R[:, 2]
        forward = forward / np.linalg.norm(forward)

        ring_center = (
            tip_p
            + t.ring_ahead_of_tip_m * forward
            + t.ring_lateral_offset_m * side
            + t.ring_vertical_offset_m * up
        )

        # At reset the task frame was constructed so +Y is world-up and +Z is
        # horizontal forward. Using it directly makes the ring plane XY truly vertical.
        R_ring = tip_R.copy()

        self._set_ring_free_pose(ring_center, R_ring)

        outer = t.ring_inner_radius_m + 2.0 * t.ring_tube_radius_m
        # The stabilizer extends another 2*half-height below the torus bottom.
        table_top_z = ring_center[2] - outer - 2.0 * t.ring_stabilizer_half_height_m
        self.data.mocap_pos[self.table_mocap] = np.array(
            [ring_center[0], ring_center[1], table_top_z - t.table_half_thickness_m]
        )
        self.data.mocap_quat[self.table_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

    def command_local_delta(self, action6):
        """Manual/debug delta, integrated on the persistent desired tool target."""
        return self.controller.set_local_delta_target(action6, reference="target")

    def command_policy_delta(self, action6):
        """Legacy measured-reference pose-target path.

        Kept only for A/B replay. New MLP execution should use
        :meth:`command_policy_delta_velocity` so ΔT is executed over its horizon.
        """
        action = validate_action(action6)
        return self.controller.set_policy_delta_target(action)

    def command_local_velocity(self, twist6_local, *, watchdog_s: float | None = None):
        """Refresh the firmware-style canonical target twist stream.

        Zero is a fresh zero-velocity target; applied joint velocity decelerates
        according to the reference generator rather than snapping q to measured.
        """
        return self.controller.set_local_velocity_command(twist6_local, watchdog_s=watchdog_s)

    def stop_velocity(self):
        """Enter firmware-style STOPPING (target qdot=0, acceleration-limited stop)."""
        self.controller.stop_velocity_command()

    def command_policy_delta_stream(
        self,
        action6,
        *,
        horizon_s: float,
        watchdog_s: float | None = None,
    ):
        """Preferred behavior-cloning contract: ΔT/horizon -> firmware-style velocity stream."""
        action = validate_action(action6)
        return self.controller.set_policy_delta_stream(
            action, horizon_s=horizon_s, watchdog_s=watchdog_s
        )

    def command_policy_delta_velocity(
        self,
        action6,
        *,
        horizon_s: float,
        watchdog_s: float | None = None,
    ):
        """Backward-compatible alias for :meth:`command_policy_delta_stream`."""
        return self.command_policy_delta_stream(
            action6, horizon_s=horizon_s, watchdog_s=watchdog_s
        )

    def command_pose_target(self, position_world, rotation_world_from_tool):
        """Oracle-only absolute tool-pose command used by ground-truth replay."""
        self.controller.set_pose_target(position_world, rotation_world_from_tool)

    def clear_task_obstacles_for_replay(self):
        """Move task mocap geometry away so replay tests EE capability, not accidental contacts.

        The real demonstrations do not contain a calibrated 3-D ring/table pose, so
        aligning only the LeArm tool trajectory to the Panda task scene can create
        meaningless collisions.  Replay therefore defaults to a collision-free scene.
        """
        self._set_ring_free_pose(np.array([5.0, 5.0, 5.0]), np.eye(3))
        self.data.mocap_pos[self.table_mocap] = np.array([5.0, 5.0, 4.0])
        mujoco.mj_forward(self.model, self.data)

    def step_physics(self, n: int = 1):
        for _ in range(n):
            self._last_diag = self.controller.step(self.physics_dt)
            mujoco.mj_step(self.model, self.data)
        return self._last_diag

    def run_for(self, seconds: float):
        n = max(1, int(round(seconds / self.physics_dt)))
        for _ in range(n):
            self.step_physics(1)
        return self._last_diag

    def execute_delta(self, action6, duration_s: float = 0.12):
        """Execute a manual/debug target-integrated delta."""
        applied = self.command_local_delta(action6)
        self.run_for(duration_s)
        return applied, self._last_diag

    def execute_policy_delta(self, action6, duration_s: float = 0.12):
        """Legacy measured-pose target execution retained for comparison."""
        applied = self.command_policy_delta(action6)
        self.run_for(duration_s)
        return applied, self._last_diag

    def execute_policy_delta_stream(self, action6, duration_s: float = 0.10, watchdog_s: float | None = None):
        """Execute ΔT through the reference-integrated velocity stream."""
        applied = self.command_policy_delta_stream(
            action6, horizon_s=duration_s, watchdog_s=watchdog_s
        )
        self.run_for(duration_s)
        return applied, self._last_diag

    def execute_policy_delta_velocity(self, action6, duration_s: float = 0.10, watchdog_s: float | None = None):
        """Compatibility alias for STREAM execution."""
        return self.execute_policy_delta_stream(action6, duration_s, watchdog_s)

    def render_wrist(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="wrist_camera")
        return self.renderer.render().copy()

    def vision_observation(self):
        return extract_sim_observation(self.render_wrist())

    def policy_observation(self) -> PolicyObservation:
        """Return the robot-independent policy observation.

        No Panda qpos/Jacobian/safety fields are exposed here.
        """
        vision = self.vision_observation()
        p, R = self.tip_pose()
        return self.observation_builder.build(vision, p, R)

    def camera_tip_geometry(self):
        """Return compiled camera/tip alignment diagnostics in the actual MuJoCo frame."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        if cam_id < 0:
            raise RuntimeError("wrist_camera missing")
        tip_p, _ = self.tip_pose()
        marker_p = self.data.site_xpos[self.blue_marker_site].copy()
        cam_p = self.data.cam_xpos[cam_id].copy()
        cam_R = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        rel_cam = cam_R.T @ (tip_p - cam_p)
        rel_marker_cam = cam_R.T @ (marker_p - cam_p)
        optical_depth = -float(rel_cam[2])
        lateral = float(np.linalg.norm(rel_cam[:2]))
        axis_error = float(np.arctan2(lateral, max(optical_depth, 1e-12)))
        marker_depth = -float(rel_marker_cam[2])
        marker_lateral = float(np.linalg.norm(rel_marker_cam[:2]))
        marker_axis_error = float(np.arctan2(marker_lateral, max(marker_depth, 1e-12)))
        return {
            "tip_in_camera_m": rel_cam,
            "camera_axis_error_deg": float(np.degrees(axis_error)),
            "marker_in_camera_m": rel_marker_cam,
            "marker_axis_error_deg": float(np.degrees(marker_axis_error)),
        }

    def tip_pose(self):
        p = self.data.site_xpos[self.tip_site].copy()
        R = self.data.site_xmat[self.tip_site].reshape(3, 3).copy()
        return p, R

    def ring_pose(self):
        return (
            self.data.xpos[self.ring_body].copy(),
            self.data.xmat[self.ring_body].reshape(3, 3).copy(),
        )

    def insertion_metrics(self):
        tip_p, _ = self.tip_pose()
        ring_p, ring_R = self.ring_pose()
        rel = ring_R.T @ (tip_p - ring_p)
        radial = float(np.linalg.norm(rel[:2]))
        plane_depth = float(rel[2])
        return {
            "radial_error_m": radial,
            "ring_plane_depth_m": plane_depth,
            "inside_hole": radial < self.cfg.task.ring_inner_radius_m,
        }

    def ring_dynamics_metrics(self):
        p = self.data.xpos[self.ring_body].copy()
        R = self.data.xmat[self.ring_body].reshape(3, 3).copy()
        v = self.data.cvel[self.ring_body].copy()
        up_world = R[:, 1]
        tilt = float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0))))
        lift = 0.0 if not np.isfinite(self._ring_reset_z) else float(p[2] - self._ring_reset_z)
        return {
            "position_world": p,
            "lift_m": lift,
            "tilt_deg": tilt,
            "linear_speed_m_s": float(np.linalg.norm(v[3:])),
            "angular_speed_rad_s": float(np.linalg.norm(v[:3])),
        }

    def debug_state(self):
        p, R = self.tip_pose()
        q = self.controller._arm_q() if self.controller is not None else np.full(7, np.nan)
        return {
            "time": float(self.data.time),
            "q": q,
            "tip_pos": p,
            "tip_R": R,
            "diag": self._last_diag,
            "insertion": self.insertion_metrics() if self.controller is not None else None,
        }

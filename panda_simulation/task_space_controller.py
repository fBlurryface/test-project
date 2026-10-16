from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco

from .simulation_config import ControllerConfig
from .geometry import matrix_to_rotvec
from .task_frames import compose_local_delta


@dataclass
class ControllerDiagnostics:
    state: str = "HOLD"
    control_mode: str = "POSE"
    sigma_min: float = float("nan")
    sigma_min_orientation: float = float("nan")
    condition_number: float = float("nan")
    damping: float = float("nan")
    joint_margin_min: float = float("nan")
    qdot_scale: float = 1.0
    max_actuator_utilization: float = 0.0
    position_error_m: float = 0.0
    orientation_error_rad: float = 0.0
    servo_lead_max_rad: float = 0.0
    closest_joint: int = -1
    outward_joints_suppressed: int = 0
    commanded_linear_speed_m_s: float = 0.0
    commanded_angular_speed_rad_s: float = 0.0
    watchdog_remaining_s: float = 0.0
    watchdog_tripped: bool = False
    command_age_s: float = 0.0
    qdot_target_max_rad_s: float = 0.0
    qdot_applied_max_rad_s: float = 0.0
    reference_tick_count: int = 0
    reference_update_hz: float = 0.0


class CartesianDLSController:
    """Panda task-space controller with a firmware-style velocity reference layer.

    The main STREAM path intentionally mirrors the *structure* of the
    uploaded LeArm firmware, without forcing Panda to copy LeArm joint behavior:

        task-space target twist
            -> Panda DLS IK -> qdot_target
            -> acceleration-rate limiter -> qdot_applied
            -> integrate q_reference at 50 Hz
            -> Panda position actuators follow q_reference

    Important semantics:
      * q_reference is persistent controller state, just like LeArm reference_deg.
      * zero command means qdot_target=0; qdot_applied ramps to zero.
      * watchdog timeout (default 300 ms) enters STOPPING, not an instant snap.
      * q_reference is NOT re-based on measured q every physics step.
      * DLS/Jacobian for STREAM is evaluated at q_reference using shadow MjData,
        analogous to LeArm doing kinematics from its motion-reference state.

    POSE mode is retained only for absolute ground-truth replay / A-B tests.

    Canonical local frame: +X=left, +Y=up, +Z=rod-forward.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: ControllerConfig):
        self.model = model
        self.data = data
        self.cfg = cfg

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip_site")
        if self.site_id < 0:
            raise RuntimeError("tool_tip_site not found")

        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("One or more Panda arm joints not found")

        self.qpos_adr = model.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_adr = model.jnt_dofadr[self.joint_ids].astype(int)
        self.q_min = model.jnt_range[self.joint_ids, 0].copy()
        self.q_max = model.jnt_range[self.joint_ids, 1].copy()

        self.actuator_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)],
            dtype=int,
        )
        if np.any(self.actuator_ids < 0):
            raise RuntimeError("One or more Panda arm actuators not found")

        # Shadow kinematic state for the reference generator.  This is the
        # reference-space IK is not recomputed from
        # lagging measured q at every physics tick.
        self.ref_data = mujoco.MjData(model)

        self.q_home = self._arm_q().copy()
        self.q_posture_ref = self.q_home.copy()

        self.q_reference = self.q_home.copy()
        self.q_cmd = self.q_reference.copy()
        self.qdot_target = np.zeros(7, dtype=float)
        self.qdot_applied = np.zeros(7, dtype=float)

        self.target_pos, self.target_R = self.current_pose()

        self.control_mode = "POSE"
        self.stream_state = "HOLD"  # HOLD / VELOCITY / STOPPING
        self.velocity_command_local = np.zeros(6, dtype=float)
        self.last_velocity_command_sim_s = -np.inf
        self.command_timeout_s = float(self.cfg.command_timeout_s)
        self._watchdog_tripped = False
        self._reference_accum_s = 0.0
        self._reference_tick_count = 0
        self._last_stream_info = self._empty_info("HOLD")
        self.diag = ControllerDiagnostics()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def _sync_shadow_to_reference(self) -> tuple[np.ndarray, np.ndarray]:
        # Copy the complete simulated state so any non-arm kinematic context stays
        # coherent, then overwrite the 7 Panda arm coordinates with q_reference.
        self.ref_data.qpos[:] = self.data.qpos
        self.ref_data.qvel[:] = 0.0
        if self.model.nmocap:
            self.ref_data.mocap_pos[:] = self.data.mocap_pos
            self.ref_data.mocap_quat[:] = self.data.mocap_quat
        self.ref_data.qpos[self.qpos_adr] = self.q_reference
        mujoco.mj_forward(self.model, self.ref_data)
        p = self.ref_data.site_xpos[self.site_id].copy()
        R = self.ref_data.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def reset(self) -> None:
        self.q_home = self._arm_q().copy()
        self.q_posture_ref = self.q_home.copy()
        self.q_reference = self.q_home.copy()
        self.q_cmd = self.q_reference.copy()
        self.qdot_target[:] = 0.0
        self.qdot_applied[:] = 0.0
        self.target_pos, self.target_R = self.current_pose()
        self.control_mode = "POSE"
        self.stream_state = "HOLD"
        self.velocity_command_local[:] = 0.0
        self.last_velocity_command_sim_s = -np.inf
        self.command_timeout_s = float(self.cfg.command_timeout_s)
        self._watchdog_tripped = False
        self._reference_accum_s = 0.0
        self._reference_tick_count = 0
        self._last_stream_info = self._empty_info("HOLD")
        self.diag = ControllerDiagnostics(reference_update_hz=float(self.cfg.control_hz))
        self._write_ctrl()

    def set_posture_reference(self, q_ref) -> None:
        q = np.asarray(q_ref, dtype=float).reshape(7)
        self.q_posture_ref = np.clip(
            q,
            self.q_min + self.cfg.joint_limit_margin_rad,
            self.q_max - self.cfg.joint_limit_margin_rad,
        )

    # ------------------------------------------------------------------
    # POSE semantics: retained for absolute replay / A-B only
    # ------------------------------------------------------------------
    def _enter_pose_mode(self) -> None:
        if self.control_mode != "POSE":
            # Start pose servo from the last stream reference, not an unrelated
            # old q_cmd.  This keeps mode changes continuous.
            self.q_cmd = self.q_reference.copy()
        self.control_mode = "POSE"
        self.stream_state = "HOLD"
        self.velocity_command_local[:] = 0.0
        self.qdot_target[:] = 0.0
        self.qdot_applied[:] = 0.0
        self._watchdog_tripped = False

    def set_local_delta_target(self, action6, *, reference: str = "target") -> np.ndarray:
        self._enter_pose_mode()
        a = self._clip_delta(action6)
        if reference == "target":
            base_p = self.target_pos.copy()
            base_R = self.target_R.copy()
        elif reference == "measured":
            base_p, base_R = self.current_pose()
        else:
            raise ValueError(f"Unknown local-delta reference: {reference!r}")
        self.target_pos, self.target_R = compose_local_delta(base_p, base_R, a)
        return a.copy()

    def set_policy_delta_target(self, action6) -> np.ndarray:
        return self.set_local_delta_target(action6, reference="measured")

    def set_pose_target(self, position_world, rotation_world_from_tool) -> None:
        self._enter_pose_mode()
        p = np.asarray(position_world, dtype=float).reshape(3)
        R = np.asarray(rotation_world_from_tool, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(R)):
            raise ValueError("Pose target contains NaN/Inf")
        self.target_pos = p.copy()
        self.target_R = R.copy()

    def hold_current_pose(self) -> None:
        self._enter_pose_mode()
        self.target_pos, self.target_R = self.current_pose()
        self.q_cmd = self._arm_q().copy()
        self.q_reference = self.q_cmd.copy()
        self._write_ctrl()

    # ------------------------------------------------------------------
    # Firmware-style STREAM semantics
    # ------------------------------------------------------------------
    def _enter_stream_mode_if_needed(self) -> None:
        if self.control_mode != "STREAM":
            # LeArm keeps its motion reference when entering velocity mode.  At
            # reset/hold the actuator command is the analogous reference here.
            self.q_reference = self.q_cmd.copy()
            self.qdot_target[:] = 0.0
            self.qdot_applied[:] = 0.0
            self._reference_accum_s = 0.0
            self.stream_state = "HOLD"
        self.control_mode = "STREAM"

    def set_local_velocity_command(self, twist6_local, *, watchdog_s: float | None = None) -> np.ndarray:
        """Replace the current canonical local target twist.

        A zero command is still a *fresh* command: target velocity becomes zero,
        then qdot_applied decelerates according to the acceleration limiter.
        This mirrors the LeArm firmware reference-motion behavior rather than hard-snapping to measured state.
        """
        cmd = np.asarray(twist6_local, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(cmd)):
            raise ValueError("Velocity command contains NaN/Inf")

        lin = float(np.linalg.norm(cmd[:3]))
        if lin > self.cfg.max_cartesian_speed_m_s:
            cmd[:3] *= self.cfg.max_cartesian_speed_m_s / lin
        ang = float(np.linalg.norm(cmd[3:]))
        if ang > self.cfg.max_angular_speed_rad_s:
            cmd[3:] *= self.cfg.max_angular_speed_rad_s / ang

        self._enter_stream_mode_if_needed()
        self.velocity_command_local = cmd
        self.last_velocity_command_sim_s = float(self.data.time)
        self._watchdog_tripped = False
        self.stream_state = "VELOCITY"

        # Per-command override exists for experiments, but the firmware-mapped
        # default is 300 ms in ControllerConfig.
        self.command_timeout_s = (
            float(self.cfg.command_timeout_s) if watchdog_s is None
            else max(float(watchdog_s), self.model.opt.timestep)
        )
        return cmd.copy()

    def set_policy_delta_stream(
        self,
        action6,
        *,
        horizon_s: float,
        watchdog_s: float | None = None,
    ) -> np.ndarray:
        """Interpret a policy label ΔT over horizon as a target twist stream."""
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be > 0")
        applied = self._clip_delta(action6)
        twist = applied / float(horizon_s)
        self.set_local_velocity_command(twist, watchdog_s=watchdog_s)
        # Cartesian speed clipping can reduce representable horizon delta.
        return (self.velocity_command_local * float(horizon_s)).copy()

    # Compatibility alias for older policy-call sites.
    def set_policy_delta_velocity(self, action6, *, horizon_s: float, watchdog_s: float | None = None) -> np.ndarray:
        return self.set_policy_delta_stream(action6, horizon_s=horizon_s, watchdog_s=watchdog_s)

    def stop_velocity_command(self) -> None:
        """Enter firmware-style STOPPING: zero target, ramp applied velocity down."""
        self._enter_stream_mode_if_needed()
        self.velocity_command_local[:] = 0.0
        self.qdot_target[:] = 0.0
        self.stream_state = "STOPPING"
        # This is an explicit stop request, not a watchdog event.
        self._watchdog_tripped = False

    # ------------------------------------------------------------------
    # DLS machinery
    # ------------------------------------------------------------------
    def _clip_delta(self, action6) -> np.ndarray:
        a = np.asarray(action6, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(a)):
            raise ValueError("Action contains NaN/Inf")
        dp = a[:3]
        dw = a[3:]
        tn = float(np.linalg.norm(dp))
        if tn > self.cfg.max_translation_step_m:
            dp *= self.cfg.max_translation_step_m / tn
        rn = float(np.linalg.norm(dw))
        if rn > self.cfg.max_rotation_step_rad:
            dw *= self.cfg.max_rotation_step_rad / rn
        return np.concatenate([dp, dw])

    def _adaptive_damping(self, sigma_min: float) -> float:
        c = self.cfg
        if sigma_min >= c.singular_value_soft:
            return c.damping_min
        ratio = 1.0 - max(sigma_min, 0.0) / max(c.singular_value_soft, 1e-9)
        return c.damping_min + (c.damping_max - c.damping_min) * ratio * ratio

    @staticmethod
    def _dls_pinv(J: np.ndarray, damping: float) -> np.ndarray:
        m = J.shape[0]
        return J.T @ np.linalg.solve(J @ J.T + (damping * damping) * np.eye(m), np.eye(m))

    def _empty_info(self, state: str) -> dict:
        q = self.q_reference if hasattr(self, "q_reference") else self._arm_q()
        margin = np.minimum(q - self.q_min, self.q_max - q)
        return {
            "state": state,
            "sigma_min": float("nan"),
            "sigma_rot": float("nan"),
            "condition": float("nan"),
            "damping": float("nan"),
            "joint_margin": float(np.min(margin)),
            "closest": int(np.argmin(margin)) + 1,
            "outward_suppressed": 0,
            "total_scale": 1.0,
        }

    def _solve_task_velocity_on_data(
        self,
        kdata: mujoco.MjData,
        q_for_limits: np.ndarray,
        vpos_world: np.ndarray,
        vrot_world: np.ndarray,
        dt: float,
        *,
        enable_nullspace: bool,
    ):
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, kdata, jacp, jacr, self.site_id)
        Jp = jacp[:, self.dof_adr]
        Jr = jacr[:, self.dof_adr]

        sp = np.linalg.svd(Jp, compute_uv=False)
        sigma_min = float(sp[-1]) if len(sp) else 0.0
        sigma_max = float(sp[0]) if len(sp) else 0.0
        condition = sigma_max / max(sigma_min, 1e-12)
        lam = self._adaptive_damping(sigma_min)
        Jp_sharp = self._dls_pinv(Jp, lam)
        qdot_pos = Jp_sharp @ vpos_world
        Np = np.eye(7) - Jp_sharp @ Jp

        residual_rot = vrot_world - Jr @ qdot_pos
        Jr_eff = Jr @ Np
        sr = np.linalg.svd(Jr_eff, compute_uv=False)
        sigma_rot = float(sr[-1]) if len(sr) else 0.0
        Jr_eff_sharp = self._dls_pinv(Jr_eff, self.cfg.orientation_damping)
        qdot_ori = Np @ (Jr_eff_sharp @ residual_rot)
        Nori = Np @ (np.eye(7) - Jr_eff_sharp @ Jr_eff)

        q = np.asarray(q_for_limits, dtype=float)
        if enable_nullspace:
            center = 0.5 * (self.q_min + self.q_max)
            half = np.maximum(0.5 * (self.q_max - self.q_min), 1e-3)
            limit_avoid = (center - q) / (half * half)
            qdot_null = Nori @ (
                self.cfg.nullspace_gain * (self.q_posture_ref - q)
                + self.cfg.joint_limit_avoid_gain * limit_avoid
            )
        else:
            qdot_null = np.zeros(7, dtype=float)

        qdot = qdot_pos + qdot_ori + qdot_null
        if not np.all(np.isfinite(qdot)):
            return None, {
                "state": "IK_NUMERICAL_FAILURE",
                "sigma_min": sigma_min,
                "sigma_rot": sigma_rot,
                "condition": condition,
                "damping": lam,
                "joint_margin": float("nan"),
                "closest": -1,
                "outward_suppressed": 0,
                "total_scale": 0.0,
            }

        lo_safe = self.q_min + self.cfg.joint_limit_margin_rad
        hi_safe = self.q_max - self.cfg.joint_limit_margin_rad
        outward_suppressed = 0
        for i in range(7):
            if q[i] <= lo_safe[i] and qdot[i] < 0.0:
                qdot[i] = 0.0
                outward_suppressed += 1
            elif q[i] >= hi_safe[i] and qdot[i] > 0.0:
                qdot[i] = 0.0
                outward_suppressed += 1

        # Uniform vector scaling preserves the task-space DLS relationship.
        max_speed = np.asarray(self.cfg.stream_max_joint_speed_rad_s, dtype=float)
        ratios = np.abs(qdot) / np.maximum(max_speed, 1e-9)
        speed_scale = 1.0 / max(1.0, float(np.max(ratios)))
        qdot *= speed_scale

        joint_margin = np.minimum(q - self.q_min, self.q_max - q)
        closest = int(np.argmin(joint_margin))
        state = "OK"
        if sigma_min < self.cfg.singular_value_soft:
            state = "SINGULARITY_DAMPING"
        if speed_scale < 0.999:
            state = "VELOCITY_CLIPPED"
        if outward_suppressed > 0:
            state = "JOINT_LIMIT_AVOIDANCE"

        return qdot, {
            "state": state,
            "sigma_min": sigma_min,
            "sigma_rot": sigma_rot,
            "condition": condition,
            "damping": lam,
            "joint_margin": float(np.min(joint_margin)),
            "closest": closest + 1,
            "outward_suppressed": outward_suppressed,
            "total_scale": speed_scale,
        }

    def _solve_task_velocity_measured(self, vpos_world, vrot_world, dt, *, enable_nullspace: bool):
        return self._solve_task_velocity_on_data(
            self.data, self._arm_q(), vpos_world, vrot_world, dt,
            enable_nullspace=enable_nullspace,
        )

    def _solve_task_velocity_reference(self, vpos_world, vrot_world, dt):
        self._sync_shadow_to_reference()
        # STREAM deliberately uses minimum-norm Panda IK.  No nullspace motion is
        # injected when the operator/policy is commanding zero velocity.
        return self._solve_task_velocity_on_data(
            self.ref_data, self.q_reference, vpos_world, vrot_world, dt,
            enable_nullspace=False,
        )

    @staticmethod
    def _vector_rate_limit(current: np.ndarray, target: np.ndarray, accel_limit: np.ndarray, dt: float) -> np.ndarray:
        """Move the whole joint-velocity vector toward target with one fraction.

        This generalizes the LeArm firmware's shared planar acceleration fraction:
        all Panda joints preserve the DLS qdot direction while ramping.
        """
        delta = target - current
        fraction = 1.0
        for i in range(len(delta)):
            mag = abs(float(delta[i]))
            max_delta = abs(float(accel_limit[i])) * dt
            if mag > max_delta and mag > 1e-12:
                fraction = min(fraction, max_delta / mag)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return current + fraction * delta

    def _reference_limit_scale(self, qdot: np.ndarray, dt: float) -> float:
        lo = self.q_min + self.cfg.joint_limit_margin_rad
        hi = self.q_max - self.cfg.joint_limit_margin_rad
        scale = 1.0
        for i in range(7):
            step = float(qdot[i] * dt)
            if step > 0.0:
                remaining = float(hi[i] - self.q_reference[i])
                if remaining <= 0.0:
                    return 0.0
                if step > remaining:
                    scale = min(scale, remaining / step)
            elif step < 0.0:
                remaining = float(self.q_reference[i] - lo[i])
                if remaining <= 0.0:
                    return 0.0
                if -step > remaining:
                    scale = min(scale, remaining / (-step))
        return float(np.clip(scale, 0.0, 1.0))

    def _stream_reference_tick(self, dt_ref: float) -> None:
        now = float(self.data.time)
        command_age = now - float(self.last_velocity_command_sim_s)

        if self.stream_state == "VELOCITY" and command_age > float(self.command_timeout_s):
            self.velocity_command_local[:] = 0.0
            self.qdot_target[:] = 0.0
            self.stream_state = "STOPPING"
            self._watchdog_tripped = True

        if self.stream_state == "VELOCITY":
            _, R_ref = self._sync_shadow_to_reference()
            cmd = self.velocity_command_local.copy()
            vpos_world = R_ref @ cmd[:3]
            vrot_world = R_ref @ cmd[3:]
            qdot_target, info = self._solve_task_velocity_reference(vpos_world, vrot_world, dt_ref)
            if qdot_target is None:
                self.qdot_target[:] = 0.0
                self.stream_state = "STOPPING"
                self._last_stream_info = info
            else:
                self.qdot_target = qdot_target
                self._last_stream_info = info
        elif self.stream_state in ("STOPPING", "HOLD"):
            self.qdot_target[:] = 0.0
            self._last_stream_info = self._empty_info(self.stream_state)
        else:
            raise RuntimeError(f"Unknown stream_state {self.stream_state!r}")

        accel = np.asarray(self.cfg.stream_max_joint_accel_rad_s2, dtype=float)
        self.qdot_applied = self._vector_rate_limit(
            self.qdot_applied, self.qdot_target, accel, dt_ref
        )

        scale = self._reference_limit_scale(self.qdot_applied, dt_ref)
        self.qdot_applied *= scale
        self._last_stream_info["total_scale"] *= scale
        if scale < 0.999:
            self._last_stream_info["state"] = "JOINT_LIMIT_SCALED"

        lo = self.q_min + self.cfg.joint_limit_margin_rad
        hi = self.q_max - self.cfg.joint_limit_margin_rad
        self.q_reference = np.clip(
            self.q_reference + self.qdot_applied * dt_ref,
            lo,
            hi,
        )
        self.q_cmd = self.q_reference.copy()

        if self.stream_state == "STOPPING" and float(np.max(np.abs(self.qdot_applied))) <= float(self.cfg.stream_velocity_epsilon_rad_s):
            self.qdot_applied[:] = 0.0
            self.qdot_target[:] = 0.0
            self.stream_state = "HOLD"
            self._last_stream_info = self._empty_info("HOLD")

        self._reference_tick_count += 1

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, dt: float) -> ControllerDiagnostics:
        p, R = self.current_pose()
        q_measured = self._arm_q()

        if self.control_mode == "POSE":
            pos_err = self.target_pos - p
            rot_err = matrix_to_rotvec(self.target_R @ R.T)
            vpos = self.cfg.kp_position * pos_err
            vrot = self.cfg.kp_orientation * rot_err
            lin = float(np.linalg.norm(vpos))
            if lin > self.cfg.max_cartesian_speed_m_s:
                vpos *= self.cfg.max_cartesian_speed_m_s / lin
            ang = float(np.linalg.norm(vrot))
            if ang > self.cfg.max_angular_speed_rad_s:
                vrot *= self.cfg.max_angular_speed_rad_s / ang

            qdot, info = self._solve_task_velocity_measured(vpos, vrot, dt, enable_nullspace=True)
            if qdot is None:
                self.q_cmd = q_measured.copy()
                self._write_ctrl()
                self.diag = ControllerDiagnostics(state="IK_NUMERICAL_FAILURE", control_mode="POSE")
                return self.diag

            self.q_cmd = self.q_cmd + qdot * dt
            lead = self.q_cmd - q_measured
            max_lead = max(float(self.cfg.max_servo_lead_rad), 1e-6)
            lead = np.clip(lead, -max_lead, max_lead)
            self.q_cmd = np.clip(
                q_measured + lead,
                self.q_min + self.cfg.joint_limit_margin_rad,
                self.q_max - self.cfg.joint_limit_margin_rad,
            )
            self.q_reference = self.q_cmd.copy()
            commanded_lin = float(np.linalg.norm(vpos))
            commanded_ang = float(np.linalg.norm(vrot))
            pos_err_norm = float(np.linalg.norm(pos_err))
            rot_err_norm = float(np.linalg.norm(rot_err))
            watchdog_remaining = 0.0
            command_age = 0.0
            state = info["state"]

        elif self.control_mode == "STREAM":
            # Firmware motion task runs at 50 Hz even though MuJoCo physics is 500 Hz.
            self._reference_accum_s += float(dt)
            ref_dt = 1.0 / float(self.cfg.control_hz)
            # Normally one tick; while-loop keeps deterministic time if caller batches.
            while self._reference_accum_s + 1e-12 >= ref_dt:
                self._reference_accum_s -= ref_dt
                self._stream_reference_tick(ref_dt)

            self.q_cmd = self.q_reference.copy()
            self._write_ctrl()

            cmd = self.velocity_command_local
            commanded_lin = float(np.linalg.norm(cmd[:3]))
            commanded_ang = float(np.linalg.norm(cmd[3:]))
            command_age = max(0.0, float(self.data.time) - float(self.last_velocity_command_sim_s)) if np.isfinite(self.last_velocity_command_sim_s) else float("inf")
            watchdog_remaining = max(0.0, float(self.command_timeout_s) - command_age) if self.stream_state == "VELOCITY" else 0.0
            info = dict(self._last_stream_info)
            state = info["state"]
            if self._watchdog_tripped and self.stream_state in ("STOPPING", "HOLD"):
                state = "WATCHDOG_STOPPING" if self.stream_state == "STOPPING" else "WATCHDOG_HOLD"
            elif self.stream_state == "STOPPING":
                state = "STOPPING"
            elif self.stream_state == "HOLD":
                state = "HOLD"
            elif commanded_lin < 1e-9 and commanded_ang < 1e-9 and state == "OK":
                state = "ZERO_RAMP"
            pos_err_norm = 0.0
            rot_err_norm = 0.0
        else:
            raise RuntimeError(f"Unknown controller mode {self.control_mode!r}")

        self._write_ctrl()
        actuator_util = self._actuator_utilization()
        if np.isfinite(actuator_util) and actuator_util >= 0.98:
            state = "ACTUATOR_SATURATED"

        joint_margin = np.minimum(self.q_reference - self.q_min, self.q_max - self.q_reference)
        servo_lead = float(np.max(np.abs(self.q_cmd - q_measured)))
        self.diag = ControllerDiagnostics(
            state=state,
            control_mode=("STREAM/" + self.stream_state) if self.control_mode == "STREAM" else "POSE",
            sigma_min=info.get("sigma_min", float("nan")),
            sigma_min_orientation=info.get("sigma_rot", float("nan")),
            condition_number=info.get("condition", float("nan")),
            damping=info.get("damping", float("nan")),
            joint_margin_min=float(np.min(joint_margin)),
            qdot_scale=float(info.get("total_scale", 1.0)),
            max_actuator_utilization=actuator_util,
            position_error_m=pos_err_norm,
            orientation_error_rad=rot_err_norm,
            servo_lead_max_rad=servo_lead,
            closest_joint=int(np.argmin(joint_margin)) + 1,
            outward_joints_suppressed=int(info.get("outward_suppressed", 0)),
            commanded_linear_speed_m_s=commanded_lin,
            commanded_angular_speed_rad_s=commanded_ang,
            watchdog_remaining_s=watchdog_remaining,
            watchdog_tripped=bool(self._watchdog_tripped),
            command_age_s=command_age,
            qdot_target_max_rad_s=float(np.max(np.abs(self.qdot_target))),
            qdot_applied_max_rad_s=float(np.max(np.abs(self.qdot_applied))),
            reference_tick_count=int(self._reference_tick_count),
            reference_update_hz=float(self.cfg.control_hz),
        )
        return self.diag

    def _write_ctrl(self) -> None:
        self.data.ctrl[self.actuator_ids] = self.q_cmd
        gripper = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if gripper >= 0:
            self.data.ctrl[gripper] = 255.0

    def _actuator_utilization(self) -> float:
        if not hasattr(self.data, "actuator_force"):
            return float("nan")
        forces = np.abs(self.data.actuator_force[self.actuator_ids])
        ranges = np.max(np.abs(self.model.actuator_forcerange[self.actuator_ids]), axis=1)
        valid = ranges > 1e-9
        if not np.any(valid):
            return float("nan")
        return float(np.max(forces[valid] / ranges[valid]))

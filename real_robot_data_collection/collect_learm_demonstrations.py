#!/usr/bin/env python3
"""
LeArm teleoperation + shadow-state + red-ring/blue-tip demonstration recorder (Windows)

The ESP32 still receives the normal firmware command:
    teleop L Z P F

No firmware telemetry is required for recording. The PC runs a local copy of
LeArm's teleop kinematics / reference-motion logic from the fixed startup pose,
and records that shadow joint state + tool pose.

Controls
--------
Mouse X       Cartesian left / right translation (L)
Mouse Y       Trim persistent tool-pitch target (P)
Up / Down     World vertical translation (Z)
Mouse Left    Tool-forward motion (F > 0)
Mouse Right   Tool-backward motion (F < 0)
R             Start one episode
S             Finish episode as success
F             Finish episode as failure
F1            Release/capture mouse; releasing also zeros teleop motion
ESC           Save active episode as aborted, stop, quit

Important assumption
--------------------
The physical arm and the shadow model must start from the same firmware startup
pose. Do not manually move the arm after startup, and do not reset only one side.
"""

import argparse
import csv
import ctypes
import json
import math
import queue
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import pygame
import serial

import red_ring_and_blue_tip_tracker as ring_tracker


VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_ESCAPE = 0x1B
VK_UP = 0x26
VK_DOWN = 0x28
VK_F1 = 0x70
VK_R = 0x52
VK_S = 0x53
VK_F = 0x46

user32 = ctypes.windll.user32 if sys.platform == "win32" else None


def key_down(vk):
    if user32 is None:
        return False
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def fourcc_to_text(value):
    value = int(value)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm_sq3(v):
    return dot3(v, v)


def add_scaled3(dst, src, scale):
    for i in range(3):
        dst[i] += src[i] * scale


class ShadowLeArm:
    """Local copy of the current hierarchical LeArm teleop reference logic."""

    # Joint order used everywhere in this class:
    # base, shoulder, elbow, wrist_pitch, wrist_roll
    INITIAL_Q_DEG = [0.0, -60.0, 70.0, 70.0, 0.0]
    Q_MIN_DEG = [-85.0, -85.0, -85.0, -85.0, -85.0]
    Q_MAX_DEG = [85.0, 85.0, 85.0, 85.0, 85.0]

    BASE_TO_SHOULDER_Z_MM = 28.9
    UPPER_ARM_MM = 104.3
    FOREARM_MM = 89.0
    CONTROL_WRIST_TO_TOOL_MM = 177.0

    # Kinematics output velocity clamps.
    KIN_MAX_VEL_DEG_S = [100.0, 70.0, 70.0, 85.0, 60.0]

    # Motion-layer limits.
    MOTION_MAX_VEL_DEG_S = [100.0, 70.0, 70.0, 85.0, 60.0]
    MOTION_MAX_ACCEL_DEG_S2 = [520.0, 360.0, 360.0, 450.0, 300.0]

    # Hierarchical teleop parameters copied from firmware.
    MAX_LEFT_MM_S = 240.0
    PITCH_TRIM_DEG_S = 60.0
    PITCH_SERVO_KP_PER_S = 8.0
    MAX_PITCH_SERVO_DEG_S = 75.0
    MAX_PITCH_TARGET_LEAD_DEG = 12.0
    MAX_VERTICAL_MM_S = 180.0
    MAX_FORWARD_MM_S = 100.0
    NOMINAL_COMMAND_PERIOD_S = 0.020
    MAX_INTEGRATED_PERIOD_S = 0.050
    RESYNC_GAP_S = 0.150
    ACTIVE_INPUT_EPS = 0.0001

    TELEOP_PRIMARY_DAMPING_MM = 2.0
    TELEOP_SECONDARY_DAMPING_MM = 12.0
    TOOL_VELOCITY_DAMPING_MM = 10.0
    ACTIVE_SET_LIMIT_EPS_DEG = 0.001
    ACTIVE_SET_VEL_EPS_DEG_S = 0.001

    def __init__(self, record_wrist_to_tip_mm=None):
        self.record_wrist_to_tip_mm = (
            float(record_wrist_to_tip_mm)
            if record_wrist_to_tip_mm is not None
            else self.CONTROL_WRIST_TO_TOOL_MM
        )
        self.reset()

    def reset(self):
        self.q_deg = list(self.INITIAL_Q_DEG)
        self.qdot_target_deg_s = [0.0] * 5
        self.qdot_applied_deg_s = [0.0] * 5
        self.pitch_target_initialized = False
        self.target_pitch_deg = 0.0
        self.last_command_dt_s = self.NOMINAL_COMMAND_PERIOD_S

    @staticmethod
    def _deg2rad(v):
        return v * math.pi / 180.0

    @staticmethod
    def _rad2deg(v):
        return v * 180.0 / math.pi

    def _pose_and_jacobian(self, q_deg=None, wrist_to_tool_mm=None):
        q = self.q_deg if q_deg is None else q_deg
        l3 = (
            self.CONTROL_WRIST_TO_TOOL_MM
            if wrist_to_tool_mm is None
            else float(wrist_to_tool_mm)
        )

        base = self._deg2rad(q[0])
        shoulder = self._deg2rad(q[1])
        elbow = self._deg2rad(q[2])
        wrist = self._deg2rad(q[3])

        a1 = shoulder
        a2 = shoulder + elbow
        a3 = shoulder + elbow + wrist

        cb, sb = math.cos(base), math.sin(base)
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        c3, s3 = math.cos(a3), math.sin(a3)

        l1 = self.UPPER_ARM_MM
        l2 = self.FOREARM_MM

        radius = l1 * s1 + l2 * s2 + l3 * s3
        z = self.BASE_TO_SHOULDER_Z_MM + l1 * c1 + l2 * c2 + l3 * c3

        pose = {
            "x_mm": radius * cb,
            "y_mm": radius * sb,
            "z_mm": z,
            "yaw_deg": q[0],
            "pitch_deg": self._rad2deg(a3),
            "roll_deg": q[4],
            "radius_mm": radius,
        }

        # Position Jacobian in mm / rad, with semantic joint velocities.
        j = [[0.0] * 5 for _ in range(3)]
        dr_ds = l1 * c1 + l2 * c2 + l3 * c3
        dz_ds = -l1 * s1 - l2 * s2 - l3 * s3
        dr_de = l2 * c2 + l3 * c3
        dz_de = -l2 * s2 - l3 * s3
        dr_dw = l3 * c3
        dz_dw = -l3 * s3

        j[0][0] = -radius * sb
        j[1][0] = radius * cb

        j[0][1] = dr_ds * cb
        j[1][1] = dr_ds * sb
        j[2][1] = dz_ds

        j[0][2] = dr_de * cb
        j[1][2] = dr_de * sb
        j[2][2] = dz_de

        j[0][3] = dr_dw * cb
        j[1][3] = dr_dw * sb
        j[2][3] = dz_dw

        return pose, j

    def control_pose(self):
        return self._pose_and_jacobian(
            wrist_to_tool_mm=self.CONTROL_WRIST_TO_TOOL_MM
        )[0]

    def record_pose(self):
        return self._pose_and_jacobian(
            wrist_to_tool_mm=self.record_wrist_to_tip_mm
        )[0]

    @staticmethod
    def _build_nullspace_basis(hard_rows, active):
        row_basis = []
        for row in hard_rows[:3]:
            v = [row[i] if active[i] else 0.0 for i in range(3)]
            for rb in row_basis:
                add_scaled3(v, rb, -dot3(v, rb))
            n2 = norm_sq3(v)
            if n2 <= 1.0e-7:
                continue
            inv = 1.0 / math.sqrt(n2)
            row_basis.append([x * inv for x in v])

        out = []
        for axis in range(3):
            if not active[axis]:
                continue
            v = [0.0, 0.0, 0.0]
            v[axis] = 1.0
            for rb in row_basis:
                add_scaled3(v, rb, -dot3(v, rb))
            for b in out:
                add_scaled3(v, b, -dot3(v, b))
            n2 = norm_sq3(v)
            if n2 <= 1.0e-7:
                continue
            inv = 1.0 / math.sqrt(n2)
            out.append([x * inv for x in v])
        return out

    @staticmethod
    def _add_nullspace_dls(task_rows, task_targets, basis, damping, q):
        task_count = len(task_rows)
        basis_count = len(basis)
        if task_count <= 0 or basis_count <= 0:
            return False

        # B = task_rows * basis^T; residual = target - Jq.
        b = [[0.0] * basis_count for _ in range(task_count)]
        residual = [0.0] * task_count
        for t in range(task_count):
            residual[t] = task_targets[t] - dot3(task_rows[t], q)
            for k in range(basis_count):
                b[t][k] = dot3(task_rows[t], basis[k])

        normal = [[0.0] * basis_count for _ in range(basis_count)]
        rhs = [0.0] * basis_count
        for i in range(basis_count):
            for t in range(task_count):
                rhs[i] += b[t][i] * residual[t]
                for k in range(basis_count):
                    normal[i][k] += b[t][i] * b[t][k]
            normal[i][i] += damping * damping

        # Firmware's teleop hierarchy normally has <= 2 basis vectors. Handle
        # 1/2 exactly and 3 with a tiny generic Gaussian elimination fallback.
        coeff = [0.0] * basis_count
        if basis_count == 1:
            if normal[0][0] <= 1.0e-7:
                return False
            coeff[0] = rhs[0] / normal[0][0]
        elif basis_count == 2:
            det = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
            if abs(det) <= 1.0e-7:
                return False
            coeff[0] = (rhs[0] * normal[1][1] - normal[0][1] * rhs[1]) / det
            coeff[1] = (normal[0][0] * rhs[1] - rhs[0] * normal[1][0]) / det
        else:
            # Small generic elimination for defensive completeness.
            a = [row[:] + [rhs[i]] for i, row in enumerate(normal)]
            n = basis_count
            for col in range(n):
                pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
                if abs(a[pivot][col]) < 1.0e-10:
                    return False
                a[col], a[pivot] = a[pivot], a[col]
                div = a[col][col]
                for k in range(col, n + 1):
                    a[col][k] /= div
                for r in range(n):
                    if r == col:
                        continue
                    factor = a[r][col]
                    for k in range(col, n + 1):
                        a[r][k] -= factor * a[col][k]
            coeff = [a[i][n] for i in range(n)]

        for k, c in enumerate(coeff):
            add_scaled3(q, basis[k], c)
        return True

    def _solve_planar(self, pitch_deg_s, vertical_mm_s, forward_mm_s,
                      vertical_active, forward_active, enabled):
        # enabled is full 5-joint mask; planar slice is shoulder/elbow/wrist.
        active = [enabled[1], enabled[2], enabled[3]]
        pose, j_position = self._pose_and_jacobian(
            wrist_to_tool_mm=self.CONTROL_WRIST_TO_TOOL_MM
        )

        base = self._deg2rad(self.q_deg[0])
        cb, sb = math.cos(base), math.sin(base)
        pitch = self._deg2rad(pose["pitch_deg"])
        sp, cp = math.sin(pitch), math.cos(pitch)

        j_pitch = [1.0, 1.0, 1.0]
        j_radial = []
        j_vertical = []
        for i in range(3):
            col = i + 1
            j_radial.append(cb * j_position[0][col] + sb * j_position[1][col])
            j_vertical.append(j_position[2][col])

        j_forward = [sp * j_radial[i] + cp * j_vertical[i] for i in range(3)]
        j_tool_normal = [-cp * j_radial[i] + sp * j_vertical[i] for i in range(3)]

        qdot = [0.0, 0.0, 0.0]  # rad/s
        hard_rows = []

        # Priority 1: pitch, with same cost preference as firmware.
        pitch_target = self._deg2rad(pitch_deg_s)
        pitch_cost = [100.0, 25.0, 1.0]
        denom = 0.0
        for i in range(3):
            if active[i]:
                denom += (j_pitch[i] * j_pitch[i]) / pitch_cost[i]
        if denom > 1.0e-7:
            for i in range(3):
                if active[i]:
                    qdot[i] = (j_pitch[i] / pitch_cost[i]) * (pitch_target / denom)
            hard_rows.append(j_pitch[:])

        if vertical_active or forward_active:
            basis = self._build_nullspace_basis(hard_rows, active)
            if vertical_active and forward_active:
                self._add_nullspace_dls(
                    [j_vertical, j_forward],
                    [vertical_mm_s, forward_mm_s],
                    basis,
                    self.TELEOP_PRIMARY_DAMPING_MM,
                    qdot,
                )
            else:
                primary_row = j_vertical if vertical_active else j_forward
                primary_target = vertical_mm_s if vertical_active else forward_mm_s
                corrected = self._add_nullspace_dls(
                    [primary_row],
                    [primary_target],
                    basis,
                    self.TELEOP_PRIMARY_DAMPING_MM,
                    qdot,
                )
                if corrected and len(hard_rows) < 3:
                    hard_rows.append(primary_row[:])
                    drift_basis = self._build_nullspace_basis(hard_rows, active)
                    drift_row = j_radial if vertical_active else j_tool_normal
                    self._add_nullspace_dls(
                        [drift_row],
                        [0.0],
                        drift_basis,
                        self.TELEOP_SECONDARY_DAMPING_MM,
                        qdot,
                    )

        limits = self.KIN_MAX_VEL_DEG_S[1:4]
        qdot_deg = [0.0, 0.0, 0.0]
        scale = 1.0
        for i in range(3):
            if not active[i]:
                qdot[i] = 0.0
            qdot_deg[i] = self._rad2deg(qdot[i])
            mag = abs(qdot_deg[i])
            if limits[i] > 0.0 and mag > limits[i]:
                scale = min(scale, limits[i] / mag)
        return [x * scale for x in qdot_deg]

    def _solve_base_left(self, left_mm_s, base_enabled):
        if not base_enabled:
            return 0.0
        pose = self.control_pose()
        radius = pose["radius_mm"]
        # Exact 1-DOF form of solve_tool_velocity() for left task only:
        # (1.5*r^2 + lambda^2) qdot = 1.5*r*v, lambda=10 mm.
        weight = 1.5
        normal = weight * radius * radius + self.TOOL_VELOCITY_DAMPING_MM ** 2
        if normal <= 1.0e-12:
            return 0.0
        qdot_rad_s = weight * radius * left_mm_s / normal
        qdot_deg_s = self._rad2deg(qdot_rad_s)
        return clamp(qdot_deg_s, -self.KIN_MAX_VEL_DEG_S[0], self.KIN_MAX_VEL_DEG_S[0])

    def _remove_outward_limited(self, qdot, enabled):
        changed = False
        for i in range(5):
            if not enabled[i]:
                continue
            outward_min = (
                self.q_deg[i] <= self.Q_MIN_DEG[i] + self.ACTIVE_SET_LIMIT_EPS_DEG
                and qdot[i] < -self.ACTIVE_SET_VEL_EPS_DEG_S
            )
            outward_max = (
                self.q_deg[i] >= self.Q_MAX_DEG[i] - self.ACTIVE_SET_LIMIT_EPS_DEG
                and qdot[i] > self.ACTIVE_SET_VEL_EPS_DEG_S
            )
            if outward_min or outward_max:
                enabled[i] = False
                changed = True
        return changed

    def apply_teleop(self, left_intent, vertical_intent, pitch_intent,
                     forward_intent, command_dt_s):
        l = clamp(float(left_intent), -1.0, 1.0)
        z = clamp(float(vertical_intent), -1.0, 1.0)
        p = clamp(float(pitch_intent), -1.0, 1.0)
        f = clamp(float(forward_intent), -1.0, 1.0)

        pose = self.control_pose()
        elapsed = float(command_dt_s)
        resync = not self.pitch_target_initialized
        if self.pitch_target_initialized:
            if (not math.isfinite(elapsed)) or elapsed < 0.0 or elapsed > self.RESYNC_GAP_S:
                resync = True

        if resync:
            self.target_pitch_deg = pose["pitch_deg"]
            self.pitch_target_initialized = True
            dt_pitch = self.NOMINAL_COMMAND_PERIOD_S
        else:
            dt_pitch = clamp(elapsed, 0.0, self.MAX_INTEGRATED_PERIOD_S)

        requested_pitch_rate = p * self.PITCH_TRIM_DEG_S
        self.target_pitch_deg += requested_pitch_rate * dt_pitch

        # All three planar joints use [-85, 85], signs=+1, offsets=0.
        pitch_min, pitch_max = -255.0, 255.0
        unclamped_target = self.target_pitch_deg
        self.target_pitch_deg = clamp(self.target_pitch_deg, pitch_min, pitch_max)
        self.target_pitch_deg = clamp(
            self.target_pitch_deg,
            pose["pitch_deg"] - self.MAX_PITCH_TARGET_LEAD_DEG,
            pose["pitch_deg"] + self.MAX_PITCH_TARGET_LEAD_DEG,
        )

        absolute_bound_hit = (
            abs(self.target_pitch_deg - unclamped_target) > 0.0001
            and (
                self.target_pitch_deg <= pitch_min + 0.0001
                or self.target_pitch_deg >= pitch_max - 0.0001
            )
        )
        pitch_feedforward = 0.0 if absolute_bound_hit else requested_pitch_rate
        pitch_error = self.target_pitch_deg - pose["pitch_deg"]
        pitch_rate_cmd = clamp(
            pitch_feedforward + self.PITCH_SERVO_KP_PER_S * pitch_error,
            -self.MAX_PITCH_SERVO_DEG_S,
            self.MAX_PITCH_SERVO_DEG_S,
        )

        vertical_mm_s = z * self.MAX_VERTICAL_MM_S
        forward_mm_s = f * self.MAX_FORWARD_MM_S
        left_mm_s = l * self.MAX_LEFT_MM_S
        vertical_active = abs(z) > self.ACTIVE_INPUT_EPS
        forward_active = abs(f) > self.ACTIVE_INPUT_EPS

        enabled = [True, True, True, True, True]
        solved = None
        for _ in range(6):
            planar = self._solve_planar(
                pitch_rate_cmd,
                vertical_mm_s,
                forward_mm_s,
                vertical_active,
                forward_active,
                enabled,
            )
            base = self._solve_base_left(left_mm_s, enabled[0])
            qdot = [base, planar[0], planar[1], planar[2], 0.0]
            if not self._remove_outward_limited(qdot, enabled):
                solved = qdot
                break

        if solved is None:
            solved = [0.0] * 5

        # ArmMotion::command() clamps again to motion-layer limits.
        self.qdot_target_deg_s = [
            clamp(solved[i], -self.MOTION_MAX_VEL_DEG_S[i], self.MOTION_MAX_VEL_DEG_S[i])
            for i in range(5)
        ]
        self.last_command_dt_s = elapsed

    @staticmethod
    def _approach(current, target, max_change):
        if current < target:
            return min(current + max_change, target)
        if current > target:
            return max(current - max_change, target)
        return current

    def _motion_update(self, dt_s):
        dt_s = max(float(dt_s), 0.0)
        if dt_s <= 0.0:
            return

        candidate = [0.0] * 5

        # Base and wrist-roll ramp independently.
        for i in (0, 4):
            accel_step = abs(self.MOTION_MAX_ACCEL_DEG_S2[i]) * dt_s
            candidate[i] = self._approach(
                self.qdot_applied_deg_s[i],
                self.qdot_target_deg_s[i],
                accel_step,
            )

        # Planar joints share a single acceleration fraction.
        planar_fraction = 1.0
        for i in (1, 2, 3):
            delta = self.qdot_target_deg_s[i] - self.qdot_applied_deg_s[i]
            mag = abs(delta)
            max_delta = abs(self.MOTION_MAX_ACCEL_DEG_S2[i]) * dt_s
            if mag > max_delta and mag > 1.0e-6:
                planar_fraction = min(planar_fraction, max_delta / mag)
        planar_fraction = clamp(planar_fraction, 0.0, 1.0)
        for i in (1, 2, 3):
            candidate[i] = self.qdot_applied_deg_s[i] + (
                self.qdot_target_deg_s[i] - self.qdot_applied_deg_s[i]
            ) * planar_fraction

        # Position-limit scale. Planar chain shares one scale.
        joint_scale = [1.0] * 5
        for i in range(5):
            requested_step = candidate[i] * dt_s
            scale = 1.0
            if requested_step > 0.0:
                remaining = self.Q_MAX_DEG[i] - self.q_deg[i]
                if remaining <= 1.0e-4:
                    scale = 0.0
                elif requested_step > remaining:
                    scale = remaining / requested_step
            elif requested_step < 0.0:
                remaining = self.q_deg[i] - self.Q_MIN_DEG[i]
                if remaining <= 1.0e-4:
                    scale = 0.0
                elif -requested_step > remaining:
                    scale = remaining / -requested_step
            joint_scale[i] = clamp(scale, 0.0, 1.0)

        planar_limit_scale = min(joint_scale[1], joint_scale[2], joint_scale[3])

        for i in range(5):
            scale = planar_limit_scale if i in (1, 2, 3) else joint_scale[i]
            applied = candidate[i] * scale
            self.qdot_applied_deg_s[i] = applied
            self.q_deg[i] = clamp(
                self.q_deg[i] + applied * dt_s,
                self.Q_MIN_DEG[i],
                self.Q_MAX_DEG[i],
            )

    def advance(self, elapsed_s):
        """Advance motion using the previous command before the next packet arrives."""
        remaining = max(float(elapsed_s), 0.0)
        if not math.isfinite(remaining):
            return

        # Normal operation is ~20 ms. Substep so UI jitter does not change the
        # acceleration limiter. If a pathological >300 ms gap occurs, emulate
        # the firmware watchdog approximately by commanding zero after 300 ms.
        active_remaining = min(remaining, 0.300)
        while active_remaining > 1.0e-9:
            dt = min(active_remaining, 0.020)
            self._motion_update(dt)
            active_remaining -= dt

        remaining -= min(remaining, 0.300)
        if remaining > 1.0e-9:
            old_target = self.qdot_target_deg_s[:]
            self.qdot_target_deg_s = [0.0] * 5
            while remaining > 1.0e-9:
                dt = min(remaining, 0.020)
                self._motion_update(dt)
                remaining -= dt
            # New packet will overwrite this immediately in apply_teleop().
            # Keeping zero here is closer to firmware timeout semantics.
            _ = old_target

    def snapshot(self):
        pose = self.record_pose()
        return {
            "q_base_deg": self.q_deg[0],
            "q_shoulder_deg": self.q_deg[1],
            "q_elbow_deg": self.q_deg[2],
            "q_wrist_pitch_deg": self.q_deg[3],
            "q_wrist_roll_deg": self.q_deg[4],
            "tool_x_mm": pose["x_mm"],
            "tool_y_mm": pose["y_mm"],
            "tool_z_mm": pose["z_mm"],
            "tool_yaw_deg": pose["yaw_deg"],
            "tool_pitch_deg": pose["pitch_deg"],
            "tool_roll_deg": pose["roll_deg"],
            "target_pitch_deg": self.target_pitch_deg if self.pitch_target_initialized else pose["pitch_deg"],
        }



# ======================================================================
# Blue marker + task phase
# ======================================================================
#
# This is intentionally the SAME lightweight marker path used by the
# smooth red/marker color calibrator:
#     shared HSV -> inRange -> morphology -> connected components
#     -> MarkerTracker
#
# No extra ROI/reacquisition/state-machine tricks are added to vision.
#
# The task-phase state machine is separate from the visual tracker:
#   1) marker must be actually measured continuously for a short time;
#   2) after that, continuous actual measurement loss triggers pierced;
#   3) pierced latches for the rest of the episode.
#
# Prediction/hold from MarkerTracker is NEVER counted as a measurement.

BLUE_H_LOW = 105
BLUE_H_HIGH = 116
BLUE_S_MIN = 80
BLUE_V_MIN = 45

BLUE_CLOSE_KERNEL = 5
BLUE_OPEN_KERNEL = 3
BLUE_MIN_AREA = 60
BLUE_MAX_AREA = 30000
BLUE_SEARCH_RADIUS_PX = 180
BLUE_POSITION_ALPHA = 0.70
BLUE_AREA_ALPHA = 0.35
BLUE_PREDICT_HOLD_S = 0.20

# During an active episode the blue marker is NOT allowed to reacquire
# globally. Camera + tool + marker are rigid, so its image location and
# apparent area should be almost constant.
BLUE_EPISODE_GATE_RADIUS_PX = 45.0
BLUE_EPISODE_AREA_RATIO_MIN = 0.25
BLUE_EPISODE_AREA_RATIO_MAX = 2.00

PHASE_ARM_VISIBLE_S = 0.20
PHASE_PIERCE_LOST_S = 0.25


def build_red_mask_from_hsv(hsv, cfg):
    """Same red HSV/morphology path as the smooth color calibrator."""
    mask1 = cv2.inRange(
        hsv,
        np.array([0, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([cfg.h_low_end, 255, 255], dtype=np.uint8),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([cfg.h_high_start, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(mask1, mask2)

    if cfg.close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.close_kernel, cfg.close_kernel),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

    if cfg.open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.open_kernel, cfg.open_kernel),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

    return mask


def build_blue_mask_from_hsv(hsv):
    """Exact generic marker HSV mask used in the smooth calibrator."""
    mask = cv2.inRange(
        hsv,
        np.array([BLUE_H_LOW, BLUE_S_MIN, BLUE_V_MIN], dtype=np.uint8),
        np.array([BLUE_H_HIGH, 255, 255], dtype=np.uint8),
    )

    if BLUE_CLOSE_KERNEL > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BLUE_CLOSE_KERNEL, BLUE_CLOSE_KERNEL),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

    if BLUE_OPEN_KERNEL > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BLUE_OPEN_KERNEL, BLUE_OPEN_KERNEL),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

    return mask


def detect_blue_marker(
    mask,
    previous_center=None,
    fixed_anchor_center=None,
    fixed_anchor_area=None,
):
    """
    Lightweight connected-component marker detector.

    Outside an episode:
        behaves like the original smooth calibrator (global acquisition,
        then previous-center gating).

    During an episode:
        fixed_anchor_center/fixed_anchor_area are supplied. Candidates MUST
        remain near that frozen marker location and within a generous area
        ratio. There is NO global reacquisition elsewhere in the image.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates = []
    spatial_rejects = 0
    area_ratio_rejects = 0

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < BLUE_MIN_AREA or area > BLUE_MAX_AREA:
            continue

        center = np.asarray(centroids[label], dtype=np.float32)

        distance = None
        area_ratio = math.nan

        if fixed_anchor_center is not None:
            # ACTIVE EPISODE: only the physical marker's frozen image region
            # is legal. Never jump to another blue object.
            distance = float(
                np.linalg.norm(center - fixed_anchor_center)
            )

            if distance > BLUE_EPISODE_GATE_RADIUS_PX:
                spatial_rejects += 1
                continue

            if fixed_anchor_area is not None and fixed_anchor_area > 0:
                area_ratio = float(area) / float(fixed_anchor_area)

                if not (
                    BLUE_EPISODE_AREA_RATIO_MIN
                    <= area_ratio
                    <= BLUE_EPISODE_AREA_RATIO_MAX
                ):
                    area_ratio_rejects += 1
                    continue

        elif previous_center is not None:
            # PRE-EPISODE: preserve original calibrator behavior.
            distance = float(
                np.linalg.norm(center - previous_center)
            )

            if distance > BLUE_SEARCH_RADIUS_PX:
                spatial_rejects += 1
                continue

        candidates.append({
            "center": center,
            "area": area,
            "bbox": (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ),
            "distance": distance,
            "area_ratio": area_ratio,
        })

    if not candidates:
        reject = "NO_BLOB"

        if fixed_anchor_center is not None:
            if area_ratio_rejects:
                reject = "ANCHOR_AREA_GATE"
            elif spatial_rejects:
                reject = "ANCHOR_POSITION_GATE"
            else:
                reject = "ANCHOR_NO_BLUE"

        return None, {
            "candidates": 0,
            "reject": reject,
            "spatial_rejects": spatial_rejects,
            "area_ratio_rejects": area_ratio_rejects,
            "distance_px": math.nan,
            "area_ratio": math.nan,
        }

    # Within the frozen gate, larger is still the most robust choice.
    best = max(candidates, key=lambda c: c["area"])

    return best, {
        "candidates": len(candidates),
        "reject": "",
        "raw_area": best["area"],
        "spatial_rejects": spatial_rejects,
        "area_ratio_rejects": area_ratio_rejects,
        "distance_px": (
            math.nan
            if best["distance"] is None
            else float(best["distance"])
        ),
        "area_ratio": (
            math.nan
            if not math.isfinite(best["area_ratio"])
            else float(best["area_ratio"])
        ),
    }


class BlueMarkerTracker:
    """
    Exact temporal semantics of the smooth calibrator's MarkerTracker.

    measured=True means a real blue blob was detected THIS frame.
    valid=True may briefly remain true during prediction hold.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.center = None
        self.area_smooth = None
        self.last_measure_t = None

    def is_locked(self, now):
        return (
            self.center is not None
            and self.last_measure_t is not None
            and now - self.last_measure_t <= BLUE_PREDICT_HOLD_S
        )

    def update(self, detection, now):
        measured = detection is not None

        stale = (
            self.last_measure_t is None
            or now - self.last_measure_t > BLUE_PREDICT_HOLD_S
        )

        if measured:
            z = detection["center"]
            raw_area = float(detection["area"])

            if self.center is None or stale:
                self.center = z.copy()
                self.area_smooth = raw_area
            else:
                self.center = (
                    (1.0 - BLUE_POSITION_ALPHA) * self.center
                    + BLUE_POSITION_ALPHA * z
                )
                self.area_smooth = (
                    (1.0 - BLUE_AREA_ALPHA) * self.area_smooth
                    + BLUE_AREA_ALPHA * raw_area
                )

            self.last_measure_t = now

            return {
                "valid": True,
                "measured": True,
                "center": self.center.copy(),
                "raw_center": z.copy(),
                "raw_area": raw_area,
                "area_smooth": float(self.area_smooth),
            }

        valid = (
            self.center is not None
            and self.last_measure_t is not None
            and now - self.last_measure_t <= BLUE_PREDICT_HOLD_S
        )

        return {
            # For this task, "valid" means visually present NOW.
            # Internal history is still retained, but we never expose a stale
            # predicted marker as if it were currently visible.
            "valid": False,
            "measured": False,
            "center": None,
            "raw_center": None,
            "raw_area": 0.0,
            "area_smooth": (
                0.0 if self.area_smooth is None else float(self.area_smooth)
            ),
        }



class EpisodeRecorder:
    FIELDNAMES = [
        "sample_index",
        "host_time_ns",
        "episode_time_s",
        "command_dt_s",
        "L", "Z", "P", "F",
        "q_base_deg", "q_shoulder_deg", "q_elbow_deg",
        "q_wrist_pitch_deg", "q_wrist_roll_deg",
        "tool_x_mm", "tool_y_mm", "tool_z_mm",
        "tool_yaw_deg", "tool_pitch_deg", "tool_roll_deg",
        "target_pitch_deg",
        "vision_frame_id", "vision_capture_ns", "vision_age_ms",
        "target_valid", "target_measured", "target_source",
        "target_u_px", "target_v_px", "target_du_px", "target_dv_px",
        "target_raw_u_px", "target_raw_v_px",
        "target_major_px", "target_minor_px", "target_angle_deg",
        "target_total_area_px2", "target_axis_ratio",
        "blue_valid", "blue_measured",
        "blue_u_px", "blue_v_px", "blue_raw_u_px", "blue_raw_v_px",
        "blue_area_px2", "blue_area_smooth_px2",
        "blue_candidates", "blue_reject",
        "blue_anchor_locked", "blue_anchor_u_px", "blue_anchor_v_px",
        "blue_anchor_area_px2", "blue_gate_distance_px", "blue_area_ratio",
        "blue_visible_run_s", "blue_lost_s", "blue_armed",
        "pierced",
    ]

    def __init__(self, root_dir, sample_hz, port, baud, shadow_model, claw_gap_cm):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sample_hz = float(sample_hz)
        self.port = port
        self.baud = int(baud)
        self.shadow = shadow_model
        self.claw_gap_cm = float(claw_gap_cm)
        self._lock = threading.RLock()

        self._active = False
        self._episode_id = None
        self._episode_dir = None
        self._csv_file = None
        self._writer = None
        self._sample_count = 0
        self._start_wall_ns = 0
        self._start_perf_ns = 0
        self._first_sample_ns = None
        self._last_sample_ns = None

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def episode_id(self):
        with self._lock:
            return self._episode_id

    @property
    def sample_count(self):
        with self._lock:
            return self._sample_count

    @property
    def start_perf_ns(self):
        with self._lock:
            return self._start_perf_ns

    def status(self):
        with self._lock:
            return self._active, self._episode_id, self._sample_count

    def _next_episode_id(self):
        max_id = 0
        for child in self.root_dir.glob("episode_*"):
            if not child.is_dir():
                continue
            try:
                max_id = max(max_id, int(child.name.split("_")[-1]))
            except ValueError:
                pass
        return max_id + 1

    def start(self):
        with self._lock:
            if self._active:
                return None
            episode_num = self._next_episode_id()
            self._episode_id = f"episode_{episode_num:06d}"
            self._episode_dir = self.root_dir / self._episode_id
            self._episode_dir.mkdir(parents=False, exist_ok=False)

            self._csv_file = (self._episode_dir / "trajectory.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()
            self._csv_file.flush()

            self._sample_count = 0
            self._start_wall_ns = time.time_ns()
            self._start_perf_ns = time.perf_counter_ns()
            self._first_sample_ns = None
            self._last_sample_ns = None
            self._active = True
            return self._episode_id, self._episode_dir

    def record_sample(self, host_time_ns, command_dt_s, l, z, p, f, snapshot, vision=None):
        with self._lock:
            if not self._active or self._writer is None:
                return
            if self._first_sample_ns is None:
                self._first_sample_ns = int(host_time_ns)
            self._last_sample_ns = int(host_time_ns)
            vision = vision or {}
            capture_ns = int(vision.get("capture_ns", 0) or 0)
            vision_age_ms = (int(host_time_ns) - capture_ns) / 1e6 if capture_ns else math.nan
            row = {
                "sample_index": self._sample_count,
                "host_time_ns": int(host_time_ns),
                "episode_time_s": (int(host_time_ns) - self._start_perf_ns) / 1e9,
                "command_dt_s": float(command_dt_s),
                "L": float(l), "Z": float(z), "P": float(p), "F": float(f),
                **snapshot,
                "vision_frame_id": int(vision.get("frame_id", 0) or 0),
                "vision_capture_ns": capture_ns,
                "vision_age_ms": vision_age_ms,
                "target_valid": int(bool(vision.get("valid", False))),
                "target_measured": int(bool(vision.get("measured", False))),
                "target_source": str(vision.get("source", "")),
                "target_u_px": vision.get("u_px", math.nan),
                "target_v_px": vision.get("v_px", math.nan),
                "target_du_px": vision.get("du_px", math.nan),
                "target_dv_px": vision.get("dv_px", math.nan),
                "target_raw_u_px": vision.get("raw_u_px", math.nan),
                "target_raw_v_px": vision.get("raw_v_px", math.nan),
                "target_major_px": vision.get("major_px", math.nan),
                "target_minor_px": vision.get("minor_px", math.nan),
                "target_angle_deg": vision.get("angle_deg", math.nan),
                "target_total_area_px2": vision.get("total_area", math.nan),
                "target_axis_ratio": vision.get("axis_ratio", math.nan),

                # Blue marker primitive observations + derived phase label.
                # BC can use pierced directly; raw fields are retained so the
                # phase threshold can be re-labeled offline without re-recording.
                "blue_valid": int(bool(vision.get("blue_valid", False))),
                "blue_measured": int(bool(vision.get("blue_measured", False))),
                "blue_u_px": vision.get("blue_u_px", math.nan),
                "blue_v_px": vision.get("blue_v_px", math.nan),
                "blue_raw_u_px": vision.get("blue_raw_u_px", math.nan),
                "blue_raw_v_px": vision.get("blue_raw_v_px", math.nan),
                "blue_area_px2": vision.get("blue_area", 0.0),
                "blue_area_smooth_px2": vision.get("blue_area_smooth", 0.0),
                "blue_candidates": int(vision.get("blue_candidates", 0) or 0),
                "blue_reject": str(vision.get("blue_reject", "")),
                "blue_anchor_locked": int(bool(vision.get("blue_anchor_locked", False))),
                "blue_anchor_u_px": vision.get("blue_anchor_u_px", math.nan),
                "blue_anchor_v_px": vision.get("blue_anchor_v_px", math.nan),
                "blue_anchor_area_px2": vision.get("blue_anchor_area", math.nan),
                "blue_gate_distance_px": vision.get("blue_gate_distance_px", math.nan),
                "blue_area_ratio": vision.get("blue_area_ratio", math.nan),
                "blue_visible_run_s": vision.get("blue_visible_run_s", 0.0),
                "blue_lost_s": vision.get("blue_lost_s", 0.0),
                "blue_armed": int(bool(vision.get("blue_armed", False))),
                "pierced": int(bool(vision.get("pierced", False))),
            }
            self._writer.writerow(row)
            self._sample_count += 1
            if self._sample_count % 25 == 0:
                self._csv_file.flush()

    def finish(self, outcome):
        with self._lock:
            if not self._active:
                return None
            end_wall_ns = time.time_ns()
            end_perf_ns = time.perf_counter_ns()
            duration_s = (end_perf_ns - self._start_perf_ns) / 1e9

            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()

            actual_sample_hz = None
            sample_span_s = None
            if (
                self._sample_count >= 2
                and self._first_sample_ns is not None
                and self._last_sample_ns is not None
                and self._last_sample_ns > self._first_sample_ns
            ):
                sample_span_s = (self._last_sample_ns - self._first_sample_ns) / 1e9
                actual_sample_hz = (self._sample_count - 1) / sample_span_s

            meta = {
                "episode_id": self._episode_id,
                "outcome": outcome,
                "success": True if outcome == "success" else (
                    False if outcome == "failure" else None
                ),
                "sample_count": self._sample_count,
                "duration_s": duration_s,
                "sample_span_s": sample_span_s,
                "actual_sample_hz": actual_sample_hz,
                "started_at_utc": datetime.fromtimestamp(
                    self._start_wall_ns / 1e9, tz=timezone.utc
                ).isoformat(),
                "ended_at_utc": datetime.fromtimestamp(
                    end_wall_ns / 1e9, tz=timezone.utc
                ).isoformat(),
                "requested_sample_hz": self.sample_hz,
                "serial_port": self.port,
                "serial_baud": self.baud,
                "state_source": "local_shadow_solver_v2_2_threaded_control_vision",
                "initial_q_deg": ShadowLeArm.INITIAL_Q_DEG,
                "control_wrist_to_tool_mm": ShadowLeArm.CONTROL_WRIST_TO_TOOL_MM,
                "record_wrist_to_tip_mm": self.shadow.record_wrist_to_tip_mm,
                "claw_gap_cm": self.claw_gap_cm,
                "assumption": (
                    "Physical arm starts at the same fixed firmware pose and is not "
                    "manually displaced independently of the PC shadow model."
                ),
                "files": {"trajectory": "trajectory.csv"},
            }
            (self._episode_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            result = {
                "episode_id": self._episode_id,
                "episode_dir": self._episode_dir,
                "sample_count": self._sample_count,
                "duration_s": duration_s,
                "actual_sample_hz": actual_sample_hz,
                "outcome": outcome,
            }

            self._active = False
            self._episode_id = None
            self._episode_dir = None
            self._csv_file = None
            self._writer = None
            self._sample_count = 0
            self._first_sample_ns = None
            self._last_sample_ns = None
            return result

class SerialReader(threading.Thread):
    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.running = True

    def run(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if raw:
                    text = raw.decode("utf-8", errors="replace").rstrip()
                    if text:
                        print(f"[ARM] {text}")
            except Exception:
                break



class EpisodeMediaRecorder:
    """Asynchronous raw-video + per-frame vision logger for one active episode."""

    VISION_FIELDS = [
        "vision_seq", "camera_frame_id", "capture_time_ns", "episode_time_s",
        "processed_time_ns", "processing_ms", "valid", "measured", "source",
        "u_px", "v_px", "du_px", "dv_px", "raw_u_px", "raw_v_px",
        "outer_u_px", "outer_v_px", "major_px", "minor_px", "angle_deg",
        "total_area_px2", "axis_ratio", "reject",
        "blue_valid", "blue_measured",
        "blue_u_px", "blue_v_px", "blue_raw_u_px", "blue_raw_v_px",
        "blue_area_px2", "blue_area_smooth_px2",
        "blue_candidates", "blue_reject",
        "blue_anchor_locked", "blue_anchor_u_px", "blue_anchor_v_px",
        "blue_anchor_area_px2", "blue_gate_distance_px", "blue_area_ratio",
        "blue_visible_run_s", "blue_lost_s", "blue_armed", "pierced",
        "frame_width", "frame_height",
    ]
    FRAME_FIELDS = [
        "video_frame_index", "camera_frame_id", "capture_time_ns", "episode_time_s"
    ]

    def __init__(self, queue_size=24, save_video=True):
        self.queue_size = int(queue_size)
        self.save_video = bool(save_video)
        self._lock = threading.RLock()
        self._active = False
        self._queue = None
        self._thread = None
        self._episode_dir = None
        self._episode_start_ns = 0
        self._video_fps = 60.0
        self._dropped_frames = 0
        self._dropped_vision = 0
        self._written_frames = 0
        self._written_vision = 0
        self._error = None

    @property
    def active(self):
        with self._lock:
            return self._active

    def start(self, episode_dir, episode_start_ns, video_fps):
        with self._lock:
            if self._active:
                return False
            self._episode_dir = Path(episode_dir)
            self._episode_start_ns = int(episode_start_ns)
            self._video_fps = max(float(video_fps), 1.0)
            self._dropped_frames = 0
            self._dropped_vision = 0
            self._written_frames = 0
            self._written_vision = 0
            self._error = None
            self._queue = queue.Queue(maxsize=max(self.queue_size, 4))
            self._active = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            return True

    def submit_frame(self, frame_id, capture_ns, frame):
        if not self.save_video:
            return
        with self._lock:
            if not self._active or self._queue is None:
                return
            q = self._queue
        try:
            q.put_nowait(("frame", int(frame_id), int(capture_ns), frame))
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1

    def submit_vision(self, result):
        with self._lock:
            if not self._active or self._queue is None:
                return
            q = self._queue
        try:
            q.put_nowait(("vision", dict(result)))
        except queue.Full:
            with self._lock:
                self._dropped_vision += 1

    def stop_accepting(self):
        with self._lock:
            self._active = False

    def finish(self):
        with self._lock:
            q = self._queue
            thread = self._thread
            self._active = False
        if q is not None:
            # Block only the UI thread here; the dedicated 50 Hz control thread keeps running.
            q.put(("stop",))
        if thread is not None and thread.is_alive():
            thread.join()
        with self._lock:
            summary = {
                "video_file": "camera_raw.avi" if self.save_video else None,
                "frame_index_file": "camera_frames.csv" if self.save_video else None,
                "vision_file": "vision.csv",
                "video_fps_nominal": self._video_fps,
                "video_frames_written": self._written_frames,
                "vision_rows_written": self._written_vision,
                "dropped_video_frames": self._dropped_frames,
                "dropped_vision_rows": self._dropped_vision,
                "media_error": self._error,
            }
            self._queue = None
            self._thread = None
            return summary

    def _worker(self):
        video = None
        frames_file = vision_file = None
        frames_writer = vision_writer = None
        video_index = 0
        try:
            if self.save_video:
                frames_file = (self._episode_dir / "camera_frames.csv").open(
                    "w", newline="", encoding="utf-8"
                )
            vision_file = (self._episode_dir / "vision.csv").open(
                "w", newline="", encoding="utf-8"
            )
            frames_writer = (
                csv.DictWriter(frames_file, fieldnames=self.FRAME_FIELDS)
                if frames_file is not None else None
            )
            vision_writer = csv.DictWriter(vision_file, fieldnames=self.VISION_FIELDS)
            if frames_writer is not None:
                frames_writer.writeheader()
            vision_writer.writeheader()

            while True:
                item = self._queue.get()
                kind = item[0]
                if kind == "stop":
                    break
                if kind == "frame":
                    _, frame_id, capture_ns, frame = item
                    if video is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        video = cv2.VideoWriter(
                            str(self._episode_dir / "camera_raw.avi"),
                            fourcc,
                            self._video_fps,
                            (w, h),
                        )
                        if not video.isOpened():
                            raise RuntimeError("Could not open MJPG video writer")
                    video.write(frame)
                    frames_writer.writerow({
                        "video_frame_index": video_index,
                        "camera_frame_id": frame_id,
                        "capture_time_ns": capture_ns,
                        "episode_time_s": (capture_ns - self._episode_start_ns) / 1e9,
                    })
                    video_index += 1
                    self._written_frames += 1
                    if video_index % 60 == 0:
                        frames_file.flush()
                elif kind == "vision":
                    row = item[1]
                    capture_ns = int(row.get("capture_ns", 0) or 0)
                    vision_writer.writerow({
                        "vision_seq": row.get("vision_seq", 0),
                        "camera_frame_id": row.get("frame_id", 0),
                        "capture_time_ns": capture_ns,
                        "episode_time_s": (
                            (capture_ns - self._episode_start_ns) / 1e9
                            if capture_ns else math.nan
                        ),
                        "processed_time_ns": row.get("processed_ns", 0),
                        "processing_ms": row.get("processing_ms", math.nan),
                        "valid": int(bool(row.get("valid", False))),
                        "measured": int(bool(row.get("measured", False))),
                        "source": row.get("source", ""),
                        "u_px": row.get("u_px", math.nan),
                        "v_px": row.get("v_px", math.nan),
                        "du_px": row.get("du_px", math.nan),
                        "dv_px": row.get("dv_px", math.nan),
                        "raw_u_px": row.get("raw_u_px", math.nan),
                        "raw_v_px": row.get("raw_v_px", math.nan),
                        "outer_u_px": row.get("outer_u_px", math.nan),
                        "outer_v_px": row.get("outer_v_px", math.nan),
                        "major_px": row.get("major_px", math.nan),
                        "minor_px": row.get("minor_px", math.nan),
                        "angle_deg": row.get("angle_deg", math.nan),
                        "total_area_px2": row.get("total_area", math.nan),
                        "axis_ratio": row.get("axis_ratio", math.nan),
                        "reject": row.get("reject", ""),

                        "blue_valid": int(bool(row.get("blue_valid", False))),
                        "blue_measured": int(bool(row.get("blue_measured", False))),
                        "blue_u_px": row.get("blue_u_px", math.nan),
                        "blue_v_px": row.get("blue_v_px", math.nan),
                        "blue_raw_u_px": row.get("blue_raw_u_px", math.nan),
                        "blue_raw_v_px": row.get("blue_raw_v_px", math.nan),
                        "blue_area_px2": row.get("blue_area", 0.0),
                        "blue_area_smooth_px2": row.get("blue_area_smooth", 0.0),
                        "blue_candidates": int(row.get("blue_candidates", 0) or 0),
                        "blue_reject": row.get("blue_reject", ""),
                        "blue_anchor_locked": int(bool(row.get("blue_anchor_locked", False))),
                        "blue_anchor_u_px": row.get("blue_anchor_u_px", math.nan),
                        "blue_anchor_v_px": row.get("blue_anchor_v_px", math.nan),
                        "blue_anchor_area_px2": row.get("blue_anchor_area", math.nan),
                        "blue_gate_distance_px": row.get("blue_gate_distance_px", math.nan),
                        "blue_area_ratio": row.get("blue_area_ratio", math.nan),
                        "blue_visible_run_s": row.get("blue_visible_run_s", 0.0),
                        "blue_lost_s": row.get("blue_lost_s", 0.0),
                        "blue_armed": int(bool(row.get("blue_armed", False))),
                        "pierced": int(bool(row.get("pierced", False))),

                        "frame_width": row.get("frame_width", 0),
                        "frame_height": row.get("frame_height", 0),
                    })
                    self._written_vision += 1
                    if self._written_vision % 60 == 0:
                        vision_file.flush()
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            print(f"[MEDIA ERROR] {self._error}")
        finally:
            if video is not None:
                video.release()
            if frames_file is not None:
                frames_file.flush()
                frames_file.close()
            if vision_file is not None:
                vision_file.flush()
                vision_file.close()


class VisionWorker(threading.Thread):
    """
    Red-ring + blue-marker vision worker.

    Red and blue share ONE BGR->HSV conversion per frame, matching the
    already-tested smooth color-calibrator path.
    """

    def __init__(self, camera, cfg, media_recorder=None):
        super().__init__(daemon=True)
        self.camera = camera
        self.cfg = cfg
        self.media_recorder = media_recorder

        self.tracker = ring_tracker.Tracker()
        self.blue_tracker = BlueMarkerTracker()

        self.stop_event = threading.Event()
        self._lock = threading.RLock()

        self._latest = {
            "frame_id": 0,
            "capture_ns": 0,
            "valid": False,
            "measured": False,
            "source": "",
            "u_px": math.nan,
            "v_px": math.nan,
            "du_px": math.nan,
            "dv_px": math.nan,
            "blue_valid": False,
            "blue_measured": False,
            "blue_u_px": math.nan,
            "blue_v_px": math.nan,
            "blue_area": 0.0,
            "blue_area_smooth": 0.0,
            "blue_anchor_locked": False,
            "blue_anchor_u_px": math.nan,
            "blue_anchor_v_px": math.nan,
            "blue_anchor_area": math.nan,
            "blue_gate_distance_px": math.nan,
            "blue_area_ratio": math.nan,
            "blue_visible_run_s": 0.0,
            "blue_lost_s": 0.0,
            "blue_armed": False,
            "pierced": False,
        }

        self._vision_seq = 0
        self._measured_hz = 0.0
        self._error = None

        # Episode-local phase state.
        self._phase_enabled = False
        self._phase_last_capture_s = None
        self._blue_visible_run_s = 0.0
        self._blue_lost_s = 0.0
        self._blue_armed = False
        self._pierced = False

        # Frozen image-space marker reference for an active episode.
        self._blue_episode_anchor_center = None
        self._blue_episode_anchor_area = None

    def stop(self):
        self.stop_event.set()

    def latest(self):
        with self._lock:
            return dict(self._latest)

    def status(self):
        with self._lock:
            return {
                "measured_hz": self._measured_hz,
                "error": self._error,
                "vision_seq": self._vision_seq,
                "phase_enabled": self._phase_enabled,
                "pierced": self._pierced,
                "blue_anchor_locked": self._blue_episode_anchor_center is not None,
            }

    def start_episode_phase(self):
        """
        Freeze the currently visible physical blue marker as this episode's
        image-space anchor.

        Returns True only when a REAL blue measurement is available now.
        This prevents an episode from starting with an ambiguous/stale marker.
        """
        with self._lock:
            latest_measured = bool(
                self._latest.get("blue_measured", False)
            )

            latest_u = float(
                self._latest.get("blue_raw_u_px", math.nan)
            )
            latest_v = float(
                self._latest.get("blue_raw_v_px", math.nan)
            )
            latest_area = float(
                self._latest.get("blue_area", 0.0)
            )

            if (
                not latest_measured
                or not math.isfinite(latest_u)
                or not math.isfinite(latest_v)
                or latest_area <= 0.0
            ):
                return False

            self._blue_episode_anchor_center = np.asarray(
                [latest_u, latest_v],
                dtype=np.float32,
            )
            self._blue_episode_anchor_area = latest_area

            self._phase_enabled = True
            self._phase_last_capture_s = None
            self._blue_visible_run_s = 0.0
            self._blue_lost_s = 0.0
            self._blue_armed = False
            self._pierced = False

            return True

    def end_episode_phase(self):
        with self._lock:
            self._phase_enabled = False
            self._blue_episode_anchor_center = None
            self._blue_episode_anchor_area = None

    def _update_phase(self, blue_measured, capture_s):
        """
        Episode-local task phase.

        IMPORTANT:
        - Uses blue_measured (real blob THIS frame), not blue_valid.
        - Requires a short stable-visible run before arming.
        - Continuous loss after arming latches pierced=True.
        """
        with self._lock:
            if not self._phase_enabled:
                self._phase_last_capture_s = capture_s
                self._blue_visible_run_s = 0.0
                self._blue_lost_s = 0.0
                self._blue_armed = False
                self._pierced = False
                return {
                    "blue_visible_run_s": 0.0,
                    "blue_lost_s": 0.0,
                    "blue_armed": False,
                    "pierced": False,
                }

            if self._phase_last_capture_s is None:
                dt = 0.0
            else:
                dt = max(
                    0.0,
                    min(capture_s - self._phase_last_capture_s, 0.10),
                )
            self._phase_last_capture_s = capture_s

            if self._pierced:
                # Latched for the rest of the episode.
                if blue_measured:
                    self._blue_lost_s = 0.0
                else:
                    self._blue_lost_s += dt
            elif blue_measured:
                self._blue_visible_run_s += dt
                self._blue_lost_s = 0.0

                if self._blue_visible_run_s >= PHASE_ARM_VISIBLE_S:
                    self._blue_armed = True
            else:
                self._blue_visible_run_s = 0.0

                if self._blue_armed:
                    self._blue_lost_s += dt

                    if self._blue_lost_s >= PHASE_PIERCE_LOST_S:
                        self._pierced = True
                else:
                    self._blue_lost_s = 0.0

            return {
                "blue_visible_run_s": float(self._blue_visible_run_s),
                "blue_lost_s": float(self._blue_lost_s),
                "blue_armed": bool(self._blue_armed),
                "pierced": bool(self._pierced),
            }

    def run(self):
        last_frame_id = 0
        hz_t0 = time.perf_counter()
        hz_count = 0

        while not self.stop_event.is_set():
            frame, frame_id, capture_ns, _ = self.camera.latest()

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = frame_id
            t0 = time.perf_counter_ns()
            frame_h, frame_w = frame.shape[:2]
            now_s = capture_ns / 1e9 if capture_ns else time.perf_counter()

            try:
                # --------------------------------------------------
                # ONE HSV conversion shared by red + blue.
                # --------------------------------------------------
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # --------------------------------------------------
                # RED RING
                # --------------------------------------------------
                mask = build_red_mask_from_hsv(hsv, self.cfg)

                locked = self.tracker.is_locked(now_s, self.cfg)

                if locked:
                    predicted = self.tracker.predicted_center(now_s)
                    previous_major = self.tracker.previous_major()
                else:
                    predicted = None
                    previous_major = None

                detection, diag = ring_tracker.detect_ring(
                    mask,
                    self.cfg,
                    predicted_center=predicted,
                    previous_major=previous_major,
                )

                center, valid, measured = self.tracker.update(
                    detection,
                    now_s,
                    self.cfg,
                )

                raw_u = raw_v = outer_u = outer_v = math.nan
                major = minor = angle = math.nan
                source = "PREDICT" if valid and not measured else ""

                if detection is not None:
                    raw_u = float(detection["center"][0])
                    raw_v = float(detection["center"][1])
                    outer_u = float(detection["outer_center"][0])
                    outer_v = float(detection["outer_center"][1])

                    (_, _), (major, minor), angle = ring_tracker.canonical_ellipse(
                        detection["outer_ellipse"]
                    )

                    source = str(
                        detection.get("source", "OUTER")
                    )

                if valid and center is not None:
                    u = float(center[0])
                    v = float(center[1])
                    du = u - frame_w * 0.5
                    dv = v - frame_h * 0.5
                else:
                    u = v = du = dv = math.nan

                # --------------------------------------------------
                # BLUE MARKER
                # --------------------------------------------------
                blue_mask = build_blue_mask_from_hsv(hsv)

                blue_locked = self.blue_tracker.is_locked(now_s)

                with self._lock:
                    phase_enabled_now = bool(self._phase_enabled)
                    episode_anchor_center = (
                        None
                        if self._blue_episode_anchor_center is None
                        else self._blue_episode_anchor_center.copy()
                    )
                    episode_anchor_area = self._blue_episode_anchor_area

                if phase_enabled_now and episode_anchor_center is not None:
                    # ACTIVE EPISODE:
                    # never reacquire globally. Only accept the physical
                    # marker near the frozen image-space anchor.
                    blue_previous = None
                    blue_detection, blue_diag = detect_blue_marker(
                        blue_mask,
                        fixed_anchor_center=episode_anchor_center,
                        fixed_anchor_area=episode_anchor_area,
                    )
                else:
                    # PRE-EPISODE:
                    # retain the already-tested smooth calibrator behavior.
                    blue_previous = (
                        self.blue_tracker.center.copy()
                        if blue_locked
                        and self.blue_tracker.center is not None
                        else None
                    )

                    blue_detection, blue_diag = detect_blue_marker(
                        blue_mask,
                        previous_center=blue_previous,
                    )

                blue_state = self.blue_tracker.update(
                    blue_detection,
                    now_s,
                )

                if blue_state["center"] is not None:
                    blue_u = float(blue_state["center"][0])
                    blue_v = float(blue_state["center"][1])
                else:
                    blue_u = blue_v = math.nan

                if blue_state["raw_center"] is not None:
                    blue_raw_u = float(blue_state["raw_center"][0])
                    blue_raw_v = float(blue_state["raw_center"][1])
                else:
                    blue_raw_u = blue_raw_v = math.nan

                # Task phase uses ACTUAL measurement only.
                phase = self._update_phase(
                    bool(blue_state["measured"]),
                    now_s,
                )

                processed_ns = time.perf_counter_ns()

                result = {
                    "vision_seq": self._vision_seq,
                    "frame_id": int(frame_id),
                    "capture_ns": int(capture_ns),
                    "processed_ns": int(processed_ns),
                    "processing_ms": (processed_ns - t0) / 1e6,

                    # Red ring
                    "valid": bool(valid),
                    "measured": bool(measured),
                    "source": source,
                    "u_px": u,
                    "v_px": v,
                    "du_px": du,
                    "dv_px": dv,
                    "raw_u_px": raw_u,
                    "raw_v_px": raw_v,
                    "outer_u_px": outer_u,
                    "outer_v_px": outer_v,
                    "major_px": float(major),
                    "minor_px": float(minor),
                    "angle_deg": float(angle),
                    "total_area": float(diag.get("total_area", 0.0)),
                    "axis_ratio": float(diag.get("axis_ratio", math.nan)),
                    "reject": str(diag.get("reject", "")),

                    # Blue marker primitives
                    "blue_valid": bool(blue_state["valid"]),
                    "blue_measured": bool(blue_state["measured"]),
                    "blue_u_px": blue_u,
                    "blue_v_px": blue_v,
                    "blue_raw_u_px": blue_raw_u,
                    "blue_raw_v_px": blue_raw_v,
                    "blue_area": float(blue_state["raw_area"]),
                    "blue_area_smooth": float(blue_state["area_smooth"]),
                    "blue_candidates": int(blue_diag.get("candidates", 0)),
                    "blue_reject": str(blue_diag.get("reject", "")),
                    "blue_anchor_locked": bool(
                        phase_enabled_now and episode_anchor_center is not None
                    ),
                    "blue_anchor_u_px": (
                        math.nan
                        if episode_anchor_center is None
                        else float(episode_anchor_center[0])
                    ),
                    "blue_anchor_v_px": (
                        math.nan
                        if episode_anchor_center is None
                        else float(episode_anchor_center[1])
                    ),
                    "blue_anchor_area": (
                        math.nan
                        if episode_anchor_area is None
                        else float(episode_anchor_area)
                    ),
                    "blue_gate_distance_px": float(
                        blue_diag.get("distance_px", math.nan)
                    ),
                    "blue_area_ratio": float(
                        blue_diag.get("area_ratio", math.nan)
                    ),

                    # Derived task phase
                    **phase,

                    "frame_width": int(frame_w),
                    "frame_height": int(frame_h),
                }

                self._vision_seq += 1
                hz_count += 1

                now = time.perf_counter()

                if now - hz_t0 >= 1.0:
                    measured_hz = hz_count / (now - hz_t0)
                    hz_t0 = now
                    hz_count = 0
                else:
                    measured_hz = None

                with self._lock:
                    self._latest = result

                    if measured_hz is not None:
                        self._measured_hz = measured_hz

                    self._error = None

                if self.media_recorder is not None:
                    self.media_recorder.submit_vision(result)

            except Exception as exc:
                self.tracker.reset()
                self.blue_tracker.reset()

                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"

                print(f"[VISION ERROR] {self._error}")
                time.sleep(0.01)


class CameraCapture(threading.Thread):
    def __init__(self, index, width, height, fps, backend_name, frame_sink=None):
        super().__init__(daemon=True)
        self.index = index
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.backend_name = backend_name
        self.frame_sink = frame_sink
        self.running = False
        self.cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._capture_ns = 0
        self._measured_fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()
        self.actual_width = 0
        self.actual_height = 0
        self.reported_fps = 0.0
        self.actual_fourcc = "????"

    @staticmethod
    def _backend_code(name):
        if name == "dshow":
            return cv2.CAP_DSHOW
        if name == "msmf":
            return cv2.CAP_MSMF
        return cv2.CAP_ANY

    def open(self):
        backend = self._backend_code(self.backend_name)
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.index} with backend={self.backend_name}"
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.requested_fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.actual_width = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.actual_height = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_fourcc = fourcc_to_text(self.cap.get(cv2.CAP_PROP_FOURCC))
        print(
            "[CAM] requested: "
            f"index={self.index} {self.width}x{self.height}@{self.requested_fps:.1f} "
            f"MJPG backend={self.backend_name}"
        )
        print(
            "[CAM] reported : "
            f"{self.actual_width}x{self.actual_height}@{self.reported_fps:.2f} "
            f"FOURCC={self.actual_fourcc!r}"
        )

    def run(self):
        self.running = True
        self._fps_t0 = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            capture_ns = time.perf_counter_ns()
            if not ok or frame is None:
                time.sleep(0.001)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                frame_id = self._frame_id
                self._capture_ns = capture_ns
            if self.frame_sink is not None:
                self.frame_sink.submit_frame(frame_id, capture_ns, frame)
            self._fps_count += 1
            now = time.perf_counter()
            dt = now - self._fps_t0
            if dt >= 1.0:
                self._measured_fps = self._fps_count / dt
                self._fps_count = 0
                self._fps_t0 = now

    def latest(self):
        with self._lock:
            return self._frame, self._frame_id, self._capture_ns, self._measured_fps

    def close(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class OperatorInputState:
    """Thread-safe bridge from pygame/UI input to the fixed-rate control thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._captured = True
        self._pending_dx = 0
        self._pending_dy = 0

    def set_captured(self, captured):
        with self._lock:
            self._captured = bool(captured)
            self._pending_dx = 0
            self._pending_dy = 0

    def add_mouse_delta(self, dx, dy):
        with self._lock:
            if self._captured:
                self._pending_dx += int(dx)
                self._pending_dy += int(dy)

    def consume(self):
        with self._lock:
            captured = self._captured
            dx = self._pending_dx
            dy = self._pending_dy
            self._pending_dx = 0
            self._pending_dy = 0
            return captured, dx, dy


class ControlLoop(threading.Thread):
    """Owns the teleop clock, shadow integration, serial TX, and robot-state logging."""

    def __init__(self, ser, shadow, shadow_lock, recorder, input_state, args, vision_worker=None):
        super().__init__(daemon=True)
        self.ser = ser
        self.shadow = shadow
        self.shadow_lock = shadow_lock
        self.recorder = recorder
        self.input_state = input_state
        self.vision_worker = vision_worker
        self.args = args
        self.period_s = 1.0 / max(float(args.hz), 1.0)
        self.stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_cmd = (0.0, 0.0, 0.0, 0.0)
        self._measured_hz = 0.0
        self._last_dt_ms = 0.0
        self._last_jitter_ms = 0.0
        self._tick_count = 0
        self._late_ticks = 0
        self._error = None

    def stop(self):
        self.stop_event.set()

    def status(self):
        with self._state_lock:
            return {
                "last_cmd": self._last_cmd,
                "measured_hz": self._measured_hz,
                "last_dt_ms": self._last_dt_ms,
                "last_jitter_ms": self._last_jitter_ms,
                "tick_count": self._tick_count,
                "late_ticks": self._late_ticks,
                "error": self._error,
            }

    def _send_line(self, text):
        self.ser.write((text + "\n").encode("ascii"))

    def _set_status(self, cmd, dt_s, measured_hz=None, late=False, error=None):
        with self._state_lock:
            self._last_cmd = tuple(cmd)
            self._last_dt_ms = float(dt_s) * 1000.0
            self._last_jitter_ms = (float(dt_s) - self.period_s) * 1000.0
            self._tick_count += 1
            if late:
                self._late_ticks += 1
            if measured_hz is not None:
                self._measured_hz = float(measured_hz)
            if error is not None:
                self._error = str(error)

    def run(self):
        period = self.period_s
        last_tick = time.perf_counter()
        next_tick = last_tick + period
        hz_t0 = last_tick
        hz_count = 0
        last_debug = last_tick

        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                remaining = next_tick - now
                if remaining > 0.0015:
                    time.sleep(remaining - 0.001)
                    continue
                if remaining > 0.0:
                    # A short yield/spin gives much tighter 20 ms scheduling on Windows
                    # without keeping a CPU core busy for the whole period.
                    time.sleep(0)
                    continue

                now = time.perf_counter()
                actual_dt = max(now - last_tick, 0.001)
                last_tick = now
                late = now - next_tick > period * 0.35

                # Preserve a fixed cadence instead of scheduling the next packet from
                # however late this packet happened to be.
                next_tick += period
                if now > next_tick + period:
                    skipped = int((now - next_tick) // period) + 1
                    next_tick += skipped * period
                    late = True

                captured, dx, dy = self.input_state.consume()
                if captured:
                    dt_scale = clamp(period / actual_dt, 0.25, 4.0)
                    l = clamp(
                        -dx * self.args.mouse_x * dt_scale,
                        -abs(self.args.max_left), abs(self.args.max_left),
                    )
                    p = clamp(
                        dy * self.args.mouse_y * dt_scale,
                        -abs(self.args.max_pitch), abs(self.args.max_pitch),
                    )
                    up = key_down(VK_UP)
                    down = key_down(VK_DOWN)
                    mouse_left = key_down(VK_LBUTTON)
                    mouse_right = key_down(VK_RBUTTON)
                    z = 0.0
                    if up and not down:
                        z = abs(self.args.vertical)
                    elif down and not up:
                        z = -abs(self.args.vertical)
                    f = 0.0
                    if mouse_left and not mouse_right:
                        f = abs(self.args.forward)
                    elif mouse_right and not mouse_left:
                        f = -abs(self.args.forward)
                else:
                    l = z = p = f = 0.0

                host_time_ns = time.perf_counter_ns()
                with self.shadow_lock:
                    # The previous command acted during actual_dt. Advance to the
                    # current command instant, snapshot it, then install the new command.
                    self.shadow.advance(actual_dt)
                    snap = self.shadow.snapshot()
                    self.shadow.apply_teleop(l, z, p, f, actual_dt)

                vision = self.vision_worker.latest() if self.vision_worker is not None else None
                self.recorder.record_sample(
                    host_time_ns, actual_dt, l, z, p, f, snapshot=snap, vision=vision
                )

                cmd = f"teleop {l:.4f} {z:.4f} {p:.4f} {f:.4f}"
                self._send_line(cmd)

                hz_count += 1
                measured_hz = None
                elapsed_hz = now - hz_t0
                if elapsed_hz >= 1.0:
                    measured_hz = hz_count / elapsed_hz
                    hz_t0 = now
                    hz_count = 0

                self._set_status((l, z, p, f), actual_dt, measured_hz, late=late)

                if self.args.debug_tx and now - last_debug >= 0.20:
                    with self.shadow_lock:
                        q = self.shadow.q_deg[:]
                        pose = self.shadow.record_pose()
                    print(
                        f"[TX] {cmd} | dt={actual_dt*1000:.2f}ms | q="
                        + " ".join(f"{v:+.2f}" for v in q)
                        + f" | tip=({pose['x_mm']:.1f},{pose['y_mm']:.1f},{pose['z_mm']:.1f}) "
                        + f"yaw={pose['yaw_deg']:.1f} pitch={pose['pitch_deg']:.1f}"
                    )
                    last_debug = now

        except Exception as exc:
            self._set_status(self.status()["last_cmd"], self.period_s, error=exc)
            print(f"[CONTROL ERROR] {exc}")
            self.stop_event.set()


def _image_to_screen(camera_rect, frame_w, frame_h, u, v):
    """Map image pixel coordinates into the pygame camera preview rectangle."""
    fw = max(float(frame_w), 1.0)
    fh = max(float(frame_h), 1.0)
    x = camera_rect.left + float(u) / fw * camera_rect.width
    y = camera_rect.top + float(v) / fh * camera_rect.height
    return int(round(x)), int(round(y))


def _draw_image_ellipse(screen, camera_rect, frame_w, frame_h,
                        center_u, center_v, major, minor, angle_deg,
                        color, width=2):
    """Draw a rotated image-space ellipse on the resized pygame preview."""
    vals = (center_u, center_v, major, minor, angle_deg)
    if not all(math.isfinite(float(v)) for v in vals):
        return
    if major <= 1.0 or minor <= 1.0:
        return
    th = math.radians(float(angle_deg))
    ct, st = math.cos(th), math.sin(th)
    a = 0.5 * float(major)
    b = 0.5 * float(minor)
    pts = []
    for k in range(64):
        phi = 2.0 * math.pi * k / 64.0
        ex = a * math.cos(phi)
        ey = b * math.sin(phi)
        u = float(center_u) + ex * ct - ey * st
        v = float(center_v) + ex * st + ey * ct
        pts.append(_image_to_screen(camera_rect, frame_w, frame_h, u, v))
    if len(pts) >= 3:
        pygame.draw.lines(screen, color, True, pts, max(1, int(width)))


def frame_to_surface(frame_bgr, target_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
    tw, th = target_size
    scale = min(tw / w, th / h)
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    if (sw, sh) != (w, h):
        surface = pygame.transform.smoothscale(surface, (sw, sh))
    return surface


def print_episode_result(result, label):
    if result is None:
        return
    print(
        f"[REC] {result['episode_id']} {label} "
        f"samples={result['sample_count']} duration={result['duration_s']:.2f}s"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--port", required=True, help="e.g. COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument(
        "--boot-wait",
        type=float,
        default=4.2,
        help="Seconds to wait after opening serial before sending commands",
    )
    ap.add_argument(
        "--record-dir",
        default=None,
        help="Episode output directory. Default: <script_dir>/recordings",
    )
    ap.add_argument(
        "--record-tool-length-mm",
        type=float,
        default=177.0,
        help=(
            "Wrist-pitch joint to the recorded rod-tip point. The control solver "
            "always stays at the firmware value 177 mm; this parameter changes "
            "only the recorded tip pose."
        ),
    )

    ap.add_argument("--forward", type=float, default=0.85)
    ap.add_argument("--vertical", type=float, default=0.85)
    ap.add_argument("--mouse-x", type=float, default=0.040)
    ap.add_argument("--mouse-y", type=float, default=0.050)
    ap.add_argument("--max-left", type=float, default=0.45)
    ap.add_argument("--max-pitch", type=float, default=0.85)
    ap.add_argument("--claw-gap", type=float, default=1.5)
    ap.add_argument("--debug-tx", action="store_true")

    ap.add_argument(
        "--vision-config",
        default=str(Path(__file__).resolve().parent / "vision_tracker_config.json"),
        help="Tuned red-ring tracker JSON config",
    )
    ap.add_argument("--no-vision", action="store_true", help="Disable ring detection")
    ap.add_argument("--no-video", action="store_true", help="Do not save raw episode video")
    ap.add_argument("--media-queue", type=int, default=24, help="Async media queue depth")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--camera-fps", type=float, default=None)
    ap.add_argument(
        "--camera-backend", choices=["dshow", "msmf", "auto"], default="dshow"
    )
    ap.add_argument("--preview-fps", type=float, default=60.0)
    ap.add_argument("--crosshair-size", type=int, default=18)
    ap.add_argument("--crosshair-gap", type=int, default=5)
    ap.add_argument("--crosshair-thickness", type=int, default=2)
    args = ap.parse_args()

    vision_cfg_path = Path(args.vision_config).expanduser().resolve()
    vision_store = ring_tracker.ConfigStore(str(vision_cfg_path))
    vision_cfg = vision_store.load()

    # Frozen red color thresholds validated on the real camera.
    # Do not let an older broad red config (e.g. high branch starting at 60)
    # silently return.
    vision_cfg.h_low_end = 7
    vision_cfg.h_high_start = 173
    vision_cfg.s_min = 160
    vision_cfg.v_min = 90

    if args.camera is None:
        args.camera = int(vision_cfg.camera_index)
    if args.width is None:
        args.width = int(vision_cfg.width)
    if args.height is None:
        args.height = int(vision_cfg.height)
    if args.camera_fps is None:
        args.camera_fps = float(vision_cfg.fps)

    if not (0.0 <= abs(args.forward) <= 1.0):
        ap.error("--forward magnitude must be in [0.0, 1.0]")
    if not (0.0 <= abs(args.vertical) <= 1.0):
        ap.error("--vertical magnitude must be in [0.0, 1.0]")
    if not (0.0 < abs(args.max_left) <= 1.0):
        ap.error("--max-left magnitude must be in (0.0, 1.0]")
    if not (0.0 < abs(args.max_pitch) <= 1.0):
        ap.error("--max-pitch magnitude must be in (0.0, 1.0]")
    if not (0.0 <= args.claw_gap <= 5.8):
        ap.error("--claw-gap must be in [0.0, 5.8] cm")
    if args.record_tool_length_mm <= 0.0:
        ap.error("--record-tool-length-mm must be > 0")
    if args.hz <= 0.0:
        ap.error("--hz must be > 0")

    # Improve sleep granularity for the dedicated 50 Hz control clock on Windows.
    timer_resolution_enabled = False
    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            timer_resolution_enabled = True
        except Exception:
            pass

    media = EpisodeMediaRecorder(
        queue_size=args.media_queue,
        save_video=not args.no_video,
    )
    camera = CameraCapture(
        args.camera, args.width, args.height, args.camera_fps, args.camera_backend,
        frame_sink=media if not args.no_video else None,
    )
    camera.open()
    camera.start()

    vision = None
    if not args.no_vision:
        vision = VisionWorker(camera, vision_cfg, media_recorder=media)
        vision.start()
        print(f"[VISION] red-ring + blue-marker phase tracker started | config={vision_cfg_path}")
    else:
        print("[VISION] disabled")

    shadow = ShadowLeArm(record_wrist_to_tip_mm=args.record_tool_length_mm)
    shadow_lock = threading.RLock()
    record_root = (
        Path(args.record_dir).expanduser().resolve()
        if args.record_dir
        else Path(__file__).resolve().parent / "recordings"
    )
    recorder = EpisodeRecorder(
        record_root, args.hz, args.port, args.baud, shadow, args.claw_gap
    )
    print(f"[REC] output root: {record_root}")
    print("[REC] R=start | S=success+save | F=failure+save")
    print(
        "[SHADOW] initial q="
        + " ".join(f"{v:+.1f}" for v in shadow.q_deg)
        + f" | recorded wrist->tip={shadow.record_wrist_to_tip_mm:.1f} mm"
    )

    ser = serial.Serial(
        args.port, args.baud, timeout=0.05, write_timeout=0.2
    )
    reader = SerialReader(ser)
    reader.start()

    print(f"[SER] waiting {args.boot_wait:.1f}s for firmware boot...")
    time.sleep(max(args.boot_wait, 0.0))

    def send_line(text):
        ser.write((text + "\n").encode("ascii"))

    send_line("stop")
    ser.flush()
    time.sleep(0.05)
    send_line(f"claw {args.claw_gap:.3f}")
    ser.flush()
    time.sleep(0.05)
    print(f"[SER] control active; claw target={args.claw_gap:.2f} cm")

    pygame.init()
    screen_w, screen_h = 1120, 840
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("LeArm real-robot demonstration collector")
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 23)
    clock = pygame.time.Clock()

    captured = True
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    input_state = OperatorInputState()
    input_state.set_captured(True)
    control = ControlLoop(
        ser, shadow, shadow_lock, recorder, input_state, args, vision_worker=vision
    )
    control.start()
    print(f"[CONTROL] dedicated thread started: target={args.hz:.1f} Hz")

    running = True
    preview_period = 1.0 / max(args.preview_fps, 1.0)
    next_preview = time.perf_counter()
    last_surface = None
    last_frame_id = 0
    camera_display_rect = None
    f1_prev = r_prev = s_prev = f_prev = False

    def patch_episode_meta(result, media_summary):
        if result is None:
            return
        meta_path = Path(result["episode_dir"]) / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["vision"] = {
                "enabled": not args.no_vision,
                "tracker": "red_ring_and_blue_tip_tracker",
                "config_file": vision_cfg_path.name,
                "camera_width": camera.actual_width,
                "camera_height": camera.actual_height,
                "camera_reported_fps": camera.reported_fps,
                "center_reference": "image_center_crosshair",
                "prediction_hold_s": float(vision_cfg.predict_hold_s),
                "blue_marker": {
                    "h_low": BLUE_H_LOW,
                    "h_high": BLUE_H_HIGH,
                    "s_min": BLUE_S_MIN,
                    "v_min": BLUE_V_MIN,
                    "close_kernel": BLUE_CLOSE_KERNEL,
                    "open_kernel": BLUE_OPEN_KERNEL,
                    "min_area": BLUE_MIN_AREA,
                    "max_area": BLUE_MAX_AREA,
                    "search_radius_px": BLUE_SEARCH_RADIUS_PX,
                    "prediction_hold_s": BLUE_PREDICT_HOLD_S,
                    "episode_anchor_radius_px": BLUE_EPISODE_GATE_RADIUS_PX,
                    "episode_area_ratio_min": BLUE_EPISODE_AREA_RATIO_MIN,
                    "episode_area_ratio_max": BLUE_EPISODE_AREA_RATIO_MAX,
                    "global_reacquisition_during_episode": False,
                    "display_stale_prediction_as_detection": False,
                },
                "task_phase": {
                    "training_field": "pierced",
                    "signal": "continuous_actual_blue_marker_loss",
                    "uses_blue_measured_not_prediction": True,
                    "arm_visible_s": PHASE_ARM_VISIBLE_S,
                    "pierce_lost_s": PHASE_PIERCE_LOST_S,
                    "pierced_latched_until_episode_end": True,
                },
            }
            meta["media"] = media_summary
            files = meta.setdefault("files", {})
            if media_summary.get("video_file"):
                files["camera_raw"] = media_summary["video_file"]
                files["camera_frames"] = media_summary["frame_index_file"]
            files["vision"] = media_summary["vision_file"]
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[META ERROR] {exc}")

    def start_episode():
        if recorder.active:
            return

        # Freeze a REAL currently visible blue marker before recording.
        # If it is not visible, do not start an ambiguous episode.
        if vision is not None:
            phase_ok = vision.start_episode_phase()

            if not phase_ok:
                print(
                    "[REC BLOCKED] Blue marker is not actually measured now. "
                    "Make the marker visible/stable, then press R again."
                )
                return

        started = recorder.start()

        if not started:
            if vision is not None:
                vision.end_episode_phase()
            return
        nominal_video_fps = (
            camera.reported_fps if camera.reported_fps and camera.reported_fps > 1.0
            else args.camera_fps
        )
        media.start(started[1], recorder.start_perf_ns, nominal_video_fps)
        print(f"[REC] {started[0]} STARTED -> {started[1]}")

    def finish_episode(outcome, label):
        if not recorder.active:
            return None
        media.stop_accepting()
        result = recorder.finish(outcome)
        media_summary = media.finish()
        patch_episode_meta(result, media_summary)

        if vision is not None:
            vision.end_episode_phase()

        print_episode_result(result, label)
        if media_summary.get("media_error"):
            print(f"[MEDIA] error={media_summary['media_error']}")
        else:
            print(
                f"[MEDIA] video_frames={media_summary['video_frames_written']} "
                f"vision_rows={media_summary['vision_rows_written']} "
                f"drops={media_summary['dropped_video_frames']}/"
                f"{media_summary['dropped_vision_rows']}"
            )
        return result

    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

            if key_down(VK_ESCAPE):
                running = False

            f1 = key_down(VK_F1)
            if f1 and not f1_prev:
                captured = not captured
                pygame.event.set_grab(captured)
                pygame.mouse.set_visible(not captured)
                pygame.mouse.get_rel()
                input_state.set_captured(captured)
            f1_prev = f1

            r_key = key_down(VK_R)
            s_key = key_down(VK_S)
            f_key = key_down(VK_F)

            if r_key and not r_prev and not recorder.active:
                start_episode()

            if s_key and not s_prev and recorder.active:
                finish_episode("success", "SUCCESS")

            if f_key and not f_prev and recorder.active:
                finish_episode("failure", "FAILURE")

            r_prev, s_prev, f_prev = r_key, s_key, f_key

            if captured:
                dx, dy = pygame.mouse.get_rel()
                if dx or dy:
                    input_state.add_mouse_delta(dx, dy)
            else:
                pygame.mouse.get_rel()

            now = time.perf_counter()
            if now >= next_preview:
                frame, frame_id, capture_ns, measured_cam_fps = camera.latest()
                if frame is not None and frame_id != last_frame_id:
                    last_surface = frame_to_surface(frame, (screen_w - 40, 570))
                    last_frame_id = frame_id
                next_preview += preview_period
                if now > next_preview + preview_period:
                    next_preview = now + preview_period

            frame, frame_id, capture_ns, measured_cam_fps = camera.latest()
            screen.fill((22, 22, 22))

            camera_area_y = 14
            camera_area_h = 570
            if last_surface is not None:
                x = (screen_w - last_surface.get_width()) // 2
                y = camera_area_y + (camera_area_h - last_surface.get_height()) // 2
                camera_display_rect = pygame.Rect(
                    x, y, last_surface.get_width(), last_surface.get_height()
                )
                screen.blit(last_surface, camera_display_rect)
                cx, cy = camera_display_rect.center
                size = max(4, args.crosshair_size)
                gap = max(0, min(args.crosshair_gap, size - 1))
                thickness = max(1, args.crosshair_thickness)
                outline = (0, 0, 0)
                reticle = (255, 255, 255)
                outline_w = thickness + 2
                segments = [
                    ((cx - size, cy), (cx - gap, cy)),
                    ((cx + gap, cy), (cx + size, cy)),
                    ((cx, cy - size), (cx, cy - gap)),
                    ((cx, cy + gap), (cx, cy + size)),
                ]
                for start, end in segments:
                    pygame.draw.line(screen, outline, start, end, outline_w)
                for start, end in segments:
                    pygame.draw.line(screen, reticle, start, end, thickness)
                pygame.draw.circle(screen, outline, (cx, cy), max(2, gap + 2), 2)
                pygame.draw.circle(screen, reticle, (cx, cy), max(1, gap), 1)

                if vision is not None:
                    vr = vision.latest()
                    fw = max(int(vr.get("frame_width", camera.actual_width) or camera.actual_width), 1)
                    fh = max(int(vr.get("frame_height", camera.actual_height) or camera.actual_height), 1)
                    valid_target = bool(vr.get("valid", False))
                    measured_target = bool(vr.get("measured", False))

                    # Draw the measured outer-ring ellipse whenever this vision
                    # result came from a real detection. Magenta distinguishes
                    # detector geometry from the green/yellow tracked center.
                    if measured_target:
                        _draw_image_ellipse(
                            screen, camera_display_rect, fw, fh,
                            float(vr.get("outer_u_px", math.nan)),
                            float(vr.get("outer_v_px", math.nan)),
                            float(vr.get("major_px", math.nan)),
                            float(vr.get("minor_px", math.nan)),
                            float(vr.get("angle_deg", math.nan)),
                            (255, 80, 255), 2,
                        )

                        raw_u = float(vr.get("raw_u_px", math.nan))
                        raw_v = float(vr.get("raw_v_px", math.nan))
                        if math.isfinite(raw_u) and math.isfinite(raw_v):
                            rx, ry = _image_to_screen(
                                camera_display_rect, fw, fh, raw_u, raw_v
                            )
                            pygame.draw.circle(screen, (255, 80, 255), (rx, ry), 4, 1)

                    if valid_target and math.isfinite(float(vr.get("u_px", math.nan))):
                        tx, ty = _image_to_screen(
                            camera_display_rect, fw, fh,
                            float(vr["u_px"]), float(vr["v_px"])
                        )
                        tcolor = (0, 255, 0) if measured_target else (255, 230, 0)

                        # Visual error vector from tool-tip crosshair to ring center.
                        pygame.draw.line(screen, (0, 0, 0), (cx, cy), (tx, ty), 4)
                        pygame.draw.line(screen, tcolor, (cx, cy), (tx, ty), 1)

                        # Tracked center marker.
                        pygame.draw.circle(screen, (0, 0, 0), (tx, ty), 11, 4)
                        pygame.draw.circle(screen, tcolor, (tx, ty), 10, 2)
                        pygame.draw.line(screen, tcolor, (tx - 15, ty), (tx + 15, ty), 2)
                        pygame.draw.line(screen, tcolor, (tx, ty - 15), (tx, ty + 15), 2)

                        state_txt = "TRACK" if measured_target else "PREDICT"
                        source_txt = str(vr.get("source", "") or "-")
                        du = float(vr.get("du_px", math.nan))
                        dv = float(vr.get("dv_px", math.nan))
                        label = f"{state_txt} {source_txt}  du={du:+.0f} dv={dv:+.0f}px"
                        label_surf = small_font.render(label, True, tcolor)
                        lx = min(max(camera_display_rect.left + 4, tx + 14),
                                 camera_display_rect.right - label_surf.get_width() - 4)
                        ly = max(camera_display_rect.top + 4, ty - 28)
                        bg = pygame.Rect(
                            lx - 4, ly - 2, label_surf.get_width() + 8,
                            label_surf.get_height() + 4
                        )
                        pygame.draw.rect(screen, (0, 0, 0), bg)
                        screen.blit(label_surf, (lx, ly))
                    else:
                        lost = small_font.render("VISION: SEARCH / LOST", True, (255, 90, 90))
                        screen.blit(lost, (camera_display_rect.left + 10, camera_display_rect.top + 10))

                    # Blue marker overlay. The marker should sit near the
                    # tool-tip crosshair while visible.
                    blue_u = float(vr.get("blue_u_px", math.nan))
                    blue_v = float(vr.get("blue_v_px", math.nan))
                    blue_measured = bool(vr.get("blue_measured", False))

                    # Frozen expected marker position (episode anchor).
                    anchor_u = float(vr.get("blue_anchor_u_px", math.nan))
                    anchor_v = float(vr.get("blue_anchor_v_px", math.nan))

                    if (
                        bool(vr.get("blue_anchor_locked", False))
                        and math.isfinite(anchor_u)
                        and math.isfinite(anchor_v)
                    ):
                        ax, ay = _image_to_screen(
                            camera_display_rect, fw, fh, anchor_u, anchor_v
                        )
                        pygame.draw.circle(
                            screen, (100, 100, 100), (ax, ay), 11, 1
                        )

                    # Actual blue detection only. No stale predicted marker.
                    if (
                        blue_measured
                        and math.isfinite(blue_u)
                        and math.isfinite(blue_v)
                    ):
                        bx, by = _image_to_screen(
                            camera_display_rect, fw, fh, blue_u, blue_v
                        )
                        bcolor = (80, 190, 255)
                        pygame.draw.circle(screen, (0, 0, 0), (bx, by), 8, 3)
                        pygame.draw.circle(screen, bcolor, (bx, by), 7, 2)

                    if bool(vr.get("pierced", False)):
                        phase_surf = small_font.render(
                            "PHASE: PIERCED", True, (80, 255, 255)
                        )
                        phase_bg = pygame.Rect(
                            camera_display_rect.left + 8,
                            camera_display_rect.bottom - phase_surf.get_height() - 12,
                            phase_surf.get_width() + 10,
                            phase_surf.get_height() + 6,
                        )
                        pygame.draw.rect(screen, (0, 0, 0), phase_bg)
                        screen.blit(
                            phase_surf,
                            (phase_bg.left + 5, phase_bg.top + 3)
                        )
            else:
                camera_display_rect = None
                img = font.render("Waiting for camera frame...", True, (235, 235, 235))
                screen.blit(img, (30, 40))

            age_ms = 0.0
            if capture_ns:
                age_ms = (time.perf_counter_ns() - capture_ns) / 1_000_000.0
            with shadow_lock:
                snap = shadow.snapshot()
            rec_active, rec_episode, rec_samples = recorder.status()
            ctl = control.status()
            last_cmd = ctl["last_cmd"]
            vr = vision.latest() if vision is not None else {}
            vst = vision.status() if vision is not None else {"measured_hz": 0.0, "error": None}
            vision_age_ms = (
                (time.perf_counter_ns() - int(vr.get("capture_ns", 0))) / 1e6
                if vr.get("capture_ns") else math.nan
            )
            vision_label = "OFF" if vision is None else f"{vst['measured_hz']:.1f}Hz"
            vision_du = float(vr.get("du_px", math.nan))
            vision_dv = float(vr.get("dv_px", math.nan))

            hud_y = 580
            lines = [
                (
                    f"CAM {camera.actual_width}x{camera.actual_height} "
                    f"reported={camera.reported_fps:.1f}fps measured={measured_cam_fps:.1f}fps "
                    f"latest_age={age_ms:.1f}ms frame={frame_id}"
                ),
                (
                    f"VISION {vision_label} valid={int(bool(vr.get('valid', False)))} "
                    f"measured={int(bool(vr.get('measured', False)))} "
                    f"src={vr.get('source', '-') or '-'} du={vision_du:+.1f} "
                    f"dv={vision_dv:+.1f} age={vision_age_ms:.1f}ms"
                ),
                (
                    f"BLUE measured={int(bool(vr.get('blue_measured', False)))} "
                    f"anchor={int(bool(vr.get('blue_anchor_locked', False)))} "
                    f"area={float(vr.get('blue_area', 0.0)):.0f}px2 "
                    f"gate_d={float(vr.get('blue_gate_distance_px', math.nan)):.1f}px "
                    f"lost={float(vr.get('blue_lost_s', 0.0)):.3f}s "
                    f"pierced={int(bool(vr.get('pierced', False)))}"
                ),
                (
                    f"CONTROL target={args.hz:.1f}Hz actual={ctl['measured_hz']:.1f}Hz "
                    f"dt={ctl['last_dt_ms']:.2f}ms jitter={ctl['last_jitter_ms']:+.2f}ms "
                    f"late={ctl['late_ticks']}"
                ),
                (
                    f"TX L={last_cmd[0]:+.3f} Z={last_cmd[1]:+.3f} "
                    f"P={last_cmd[2]:+.3f} F={last_cmd[3]:+.3f} | "
                    f"REC={'ON' if rec_active else 'OFF'} {rec_episode or '-'} samples={rec_samples}"
                ),
                (
                    "SHADOW q=[" + ", ".join(
                        f"{snap[k]:+.1f}" for k in (
                            "q_base_deg", "q_shoulder_deg", "q_elbow_deg",
                            "q_wrist_pitch_deg", "q_wrist_roll_deg"
                        )
                    ) + "]"
                ),
                (
                    f"TIP x={snap['tool_x_mm']:+.1f} y={snap['tool_y_mm']:+.1f} "
                    f"z={snap['tool_z_mm']:+.1f} yaw={snap['tool_yaw_deg']:+.1f} "
                    f"pitch={snap['tool_pitch_deg']:+.1f} | R=start S=success F=failure"
                ),
            ]
            for i, line in enumerate(lines):
                img = small_font.render(line, True, (235, 235, 235))
                screen.blit(img, (18, hud_y + i * 34))

            pygame.display.flip()
            clock.tick(120)

            if ctl["error"]:
                print(f"[CONTROL] stopping UI because control thread failed: {ctl['error']}")
                running = False

    finally:
        control.stop()
        if control.is_alive():
            control.join(timeout=1.5)

        try:
            if recorder.active:
                finish_episode("aborted", "ABORTED")
            elif media.active:
                media.finish()
            send_line("teleop 0 0 0 0")
            ser.flush()
            time.sleep(0.03)
            send_line("stop")
            ser.flush()
        except Exception:
            pass

        reader.running = False
        try:
            ser.close()
        except Exception:
            pass
        if vision is not None:
            vision.stop()
            if vision.is_alive():
                vision.join(timeout=1.5)
        camera.close()
        pygame.quit()
        if timer_resolution_enabled:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This teleop script currently uses Windows GetAsyncKeyState.")
    main()

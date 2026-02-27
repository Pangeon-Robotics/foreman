"""Heading-based and wheeled navigation for target game.

Provides NavigatorMixin with _tick_walk() and _tick_walk_wheeled() methods
that are mixed into TargetGame. These implement the non-DWA navigation paths:
heading-proportional control for legged robots and differential wheel drive.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import clamp as _clamp


class NavigatorMixin:
    """Heading-based and wheeled navigation methods for TargetGame."""

    def _tick_walk(self):
        """Walk toward target using L4 GaitParams directly.

        Matches GA training episode logic (episode.py:1771-1811):
        - Large heading error -> turn in place (L4 arc mode)
        - Small heading error -> walk with differential stride

        When SLAM odometry is active, heading/distance use the estimated
        pose. Fall detection still uses ground truth (z-height).
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        # Path critic: record position at 10Hz (skip TIP -- stationary phase)
        if (self._path_critic is not None
                and self._target_step_count % 10 == 0
                and abs(heading_err) <= C.THETA_THRESHOLD):
            self._path_critic.record(nav_x, nav_y, t=self._target_step_count * C.CONTROL_DT)

        if abs(heading_err) > C.THETA_THRESHOLD:
            # TURN IN PLACE -- arc-based rotation with diagonal stepping
            wz = (C.TURN_WZ if heading_err > 0 else -C.TURN_WZ) * C._TIP_WZ_SCALE
            params = self._L4GaitParams(
                gait_type='trot', wz=wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
        else:
            # WALK -- L4 differential stride (layer_4/generator.py:134-164)
            self._reset_tip()
            # Decel taper in last 30% before threshold so WALK->TURN
            # transition doesn't pitch the robot forward from momentum.
            taper_start = 0.7 * C.THETA_THRESHOLD
            if abs(heading_err) > taper_start:
                decel = (C.THETA_THRESHOLD - abs(heading_err)) / (C.THETA_THRESHOLD - taper_start)
                decel = max(0.0, decel)
            else:
                decel = 1.0
            heading_mod = decel * max(0.0, math.cos(heading_err))
            wz = _clamp(C.KP_YAW * heading_err, -C.WZ_LIMIT, C.WZ_LIMIT)
            params = self._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * heading_mod,
                gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH, wz=wz,
                body_height=C.BODY_HEIGHT,
            )

        self._send_l4(params)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "T" if abs(heading_err) > C.THETA_THRESHOLD else "W"
        if len(self._rp_log) > 0:
            prev = self._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        self._rp_log.append((self._step_count, r_deg, p_deg, droll, dpitch, mode))

        if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * C.CONTROL_DT
            is_tip = abs(heading_err) > C.THETA_THRESHOLD
            mode_str = "TIP" if is_tip else "WALK"
            if is_tip:
                sent_wz = C.TURN_WZ if heading_err > 0 else -C.TURN_WZ
                sent_step = 0.0
            else:
                sent_wz = wz
                sent_step = C.STEP_LENGTH * heading_mod
            slam_info = ""
            if self._odometry is not None:
                drift = math.sqrt((nav_x - x_truth)**2 + (nav_y - y_truth)**2)
                slam_info = f"  drift={drift:.3f}m"
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
                ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                            f"rg={_rg:.2f} reg={_reg:.1f}m")
            occ_str = self._get_occ_str()
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str:<7} dist={dist:.1f}m  "
                  f"h_err={math.degrees(heading_err):+.0f}deg  "
                  f"step={sent_step:.2f}  wz={sent_wz:+.2f}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                  f"{slam_info}{ato_info}{occ_str}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _tick_walk_wheeled(self):
        """Drive toward target using differential wheel torque.

        No gait generator -- legs hold rigid at home pose via PD,
        wheels get pure torque for forward drive + differential steering.
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        # Forward: proportional to alignment^2, taper near target.
        # Squaring alignment prioritizes turning: at 60 deg off, fwd drops to 6%
        # instead of 25%. This prevents heavy robots from spiraling at high
        # heading errors -- they turn first, then drive straight.
        alignment = max(0.0, math.cos(heading_err))
        alignment = alignment * alignment
        dist_taper = min(1.0, dist / 1.0)  # slow in last 1m
        fwd = C.WHEEL_FWD_TORQUE * alignment * dist_taper

        # Turn: proportional + saturated
        turn = _clamp(C.WHEEL_KP_YAW * heading_err, -C.WHEEL_MAX_TURN, C.WHEEL_MAX_TURN)

        self._send_wheeled(fwd, turn)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "D"  # Drive (wheeled)
        if len(self._rp_log) > 0:
            prev = self._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        self._rp_log.append((self._step_count, r_deg, p_deg, droll, dpitch, mode))

        if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * C.CONTROL_DT
            body = self._sim.get_body("base")
            vx = float(body.linvel[0]) if body else 0
            vy = float(body.linvel[1]) if body else 0
            occ_str = self._get_occ_str()
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"DRIVE  dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"z={z:.2f}  fwd={fwd:.1f}Nm  turn={turn:.1f}Nm  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  v=({vx:+.2f},{vy:+.2f})  t={t:.1f}s"
                  f"{occ_str}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _send_wheeled(self, fwd_torque, turn_torque, z=None):
        """Construct and send LowCmd for differential wheel drive.

        Legs (slots 0-11): rigid PD hold at home pose. With a properly
        splayed stance (wide support polygon + CoM analysis), no tracking
        or height correction is needed -- the robot rolls like a car.
        Wheels (slots 12-15): pure torque, differential drive.
          FR=12, FL=13, RR=14, RL=15
          right wheels (FR, RR) = fwd + turn
          left wheels (FL, RL) = fwd - turn
        """
        cmd = self._make_low_cmd()
        # All 12 leg joints: rigid hold at home pose
        for i in range(12):
            cmd.motor_cmd[i].q = float(self._home_q[i])
            cmd.motor_cmd[i].kp = self._kp
            cmd.motor_cmd[i].kd = self._kd
        # Wheels: pure torque, differential drive
        right = fwd_torque + turn_torque
        left = fwd_torque - turn_torque
        for i, tau in [(12, right), (13, left), (14, right), (15, left)]:
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = tau
        self._stamp_cmd(cmd)
        self._sim._lowcmd_pub.Write(cmd)
        self._sim._t += C.CONTROL_DT

"""Heading-based and wheeled navigation for target game.

Provides Navigator with tick_walk_heading() and tick_walk_wheeled() methods.
These implement the non-DWA navigation paths: heading-proportional control
for legged robots and differential wheel drive.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import clamp as _clamp


class Navigator:
    """Heading-based and wheeled navigation helper for TargetGame."""

    def __init__(self, game):
        self.game = game

    def tick_walk_heading(self):
        """Walk toward target using L4 GaitParams directly.

        Matches GA training episode logic (episode.py:1771-1811):
        - Large heading error -> turn in place (L4 arc mode)
        - Small heading error -> walk with differential stride

        When SLAM odometry is active, heading/distance use the estimated
        pose. Fall detection still uses ground truth (z-height).
        """
        g = self.game
        x_truth, y_truth, yaw_truth, z, roll, pitch = g._get_robot_pose()
        target = g._spawner.current_target
        g._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if g._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = g._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        # Path critic: record position at 10Hz (skip TIP -- stationary phase)
        if (g._path_critic is not None
                and g._target_step_count % 10 == 0
                and abs(heading_err) <= C.THETA_THRESHOLD):
            v_cmd = C.STEP_LENGTH * max(0.0, math.cos(heading_err)) * C.GAIT_FREQ
            g._path_critic.record(
                nav_x, nav_y, t=g._target_step_count * C.CONTROL_DT,
                v_cmd=v_cmd)

        if abs(heading_err) > C.THETA_THRESHOLD:
            # TURN IN PLACE -- arc-based rotation with diagonal stepping
            wz = (C.TURN_WZ if heading_err > 0 else -C.TURN_WZ) * C._TIP_WZ_SCALE
            params = g._L4GaitParams(
                gait_type='trot', wz=wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
        else:
            # WALK -- L4 differential stride (layer_4/generator.py:134-164)
            g._reset_tip()
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
            params = g._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * heading_mod,
                gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH, wz=wz,
                body_height=C.BODY_HEIGHT,
            )

        g._send_l4(params)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "T" if abs(heading_err) > C.THETA_THRESHOLD else "W"
        if len(g._rp_log) > 0:
            prev = g._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        g._rp_log.append((g._step_count, r_deg, p_deg, droll, dpitch, mode))

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            is_tip = abs(heading_err) > C.THETA_THRESHOLD
            mode_str = "TIP" if is_tip else "WALK"
            if is_tip:
                sent_wz = C.TURN_WZ if heading_err > 0 else -C.TURN_WZ
                sent_step = 0.0
            else:
                sent_wz = wz
                sent_step = C.STEP_LENGTH * heading_mod
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            occ = g._get_occ_str()
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"{mode_str:<5} t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_truth:.1f},{y_truth:.1f})"
                f"{clr_s}{occ}")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

    def tick_walk_wheeled(self):
        """Drive toward target using differential wheel torque.

        No gait generator -- legs hold rigid at home pose via PD,
        wheels get pure torque for forward drive + differential steering.
        """
        g = self.game
        x_truth, y_truth, yaw_truth, z, roll, pitch = g._get_robot_pose()
        target = g._spawner.current_target
        g._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if g._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = g._get_nav_pose()
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
        if len(g._rp_log) > 0:
            prev = g._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        g._rp_log.append((g._step_count, r_deg, p_deg, droll, dpitch, mode))

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            occ = g._get_occ_str()
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"DRIVE t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_truth:.1f},{y_truth:.1f})"
                f"{clr_s}{occ}")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

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
        g = self.game
        cmd = g._make_low_cmd()
        # All 12 leg joints: rigid hold at home pose
        for i in range(12):
            cmd.motor_cmd[i].q = float(g._home_q[i])
            cmd.motor_cmd[i].kp = g._kp
            cmd.motor_cmd[i].kd = g._kd
        # Wheels: pure torque, differential drive
        right = fwd_torque + turn_torque
        left = fwd_torque - turn_torque
        for i, tau in [(12, right), (13, left), (14, right), (15, left)]:
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = tau
        g._stamp_cmd(cmd)
        g._sim._lowcmd_pub.Write(cmd)
        g._sim._t += C.CONTROL_DT

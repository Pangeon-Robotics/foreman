"""Heading-based and wheeled navigation for target game.

Provides Navigator with tick_walk_heading() and tick_walk_wheeled() methods.
Heading-proportional control for legged robots and differential wheel drive.

When a costmap is available, tick_walk_heading() follows the committed A*
path (green dots) for obstacle avoidance. Without a costmap, it steers
directly at the target.
"""
from __future__ import annotations

import math

from . import game_config as C
from .dwa_path_export import export_path, waypoint_from_path
from .utils import clamp as _clamp, normalize_angle as _normalize_angle


class Navigator:
    """Heading-based and wheeled navigation helper for TargetGame."""

    def __init__(self, game):
        self.game = game

    # ------------------------------------------------------------------
    # Waypoint from committed path (green dots = navigation path)
    # ------------------------------------------------------------------

    def _waypoint_heading(self, nav_x, nav_y, nav_yaw, target, dist):
        """Return heading error — toward next waypoint on committed path.

        The committed path (green dots) is the single A* path used for
        both visualization and navigation. We extract a waypoint 2m
        ahead along that path. Falls back to target heading when no
        path is available or when close to the target.
        """
        g = self.game
        if dist > 1.5 and g._committed_path is not None and len(g._committed_path) >= 2:
            wp = waypoint_from_path(g._committed_path, nav_x, nav_y,
                                    lookahead=2.0)
            if wp is not None:
                wp_dx = wp[0] - nav_x
                wp_dy = wp[1] - nav_y
                wp_err = _normalize_angle(
                    math.atan2(wp_dy, wp_dx) - nav_yaw)
                # Sanity: if waypoint is behind us, steer at target instead
                if abs(wp_err) < math.pi / 2:
                    return wp_err
        return target.heading_error(nav_x, nav_y, nav_yaw)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick_walk_heading(self):
        """Walk toward target using L4 GaitParams directly.

        Steers toward A* waypoints when a costmap is available,
        otherwise heads directly at the target.
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
        dist = target.distance_to(nav_x, nav_y)

        # A* path planning — single path for both green dots and navigation.
        # export_path() manages its own replan cooldown (PATH_HOLD_TICKS)
        # and writes the committed path to the temp file (green dots).
        if g._path_critic is not None and g._target_step_count % 10 == 0:
            export_path(g, target.x, target.y)
        heading_err = self._waypoint_heading(
            nav_x, nav_y, nav_yaw, target, dist)

        # TIP threshold with hysteresis.
        # B2 differential stride produces limited yaw rate — TIP handles
        # large heading corrections. Narrow hysteresis band (0.7× exit)
        # reduces overshoot by exiting TIP sooner.
        tip_enter = C.THETA_THRESHOLD
        tip_exit = C.THETA_THRESHOLD * 0.7
        in_tip = getattr(g, '_heading_in_tip', False)
        should_tip = (abs(heading_err) > tip_enter
                      or (in_tip and abs(heading_err) > tip_exit))

        # Path critic: record position at 10Hz (skip TIP -- stationary phase)
        if (g._path_critic is not None
                and g._target_step_count % 10 == 0
                and not should_tip):
            v_cmd = C.STEP_LENGTH * max(0.15, math.cos(heading_err)) * C.GAIT_FREQ
            g._path_critic.record(
                nav_x, nav_y, t=g._target_step_count * C.CONTROL_DT,
                v_cmd=v_cmd)

        # Smooth wz to prevent heading oscillation from gait-induced yaw wobble.
        # Without smoothing, each stride cycle produces a yaw perturbation that
        # reverses the wz command, causing limit-cycle oscillation.
        _prev_wz = getattr(g, '_smooth_wz', 0.0)

        if should_tip:
            # TURN IN PLACE with proportional wz (40-100% based on error)
            g._heading_in_tip = True
            err_frac = min(1.0, abs(heading_err) / tip_enter)
            tip_wz = C.TURN_WZ * (0.4 + 0.6 * err_frac)
            wz = math.copysign(tip_wz, heading_err) * C._TIP_WZ_SCALE
            g._smooth_wz = 0.0  # reset walk wz when entering TIP
            params = g._L4GaitParams(
                gait_type='trot', wz=wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
        else:
            # WALK with differential stride turning
            g._heading_in_tip = False
            g._reset_tip()
            heading_mod = max(0.15, math.cos(heading_err))
            kp_yaw = C.KP_YAW
            wz_lim = C.WZ_LIMIT
            wz = _clamp(kp_yaw * heading_err, -wz_lim, wz_lim)
            # EMA smooth wz to dampen gait-induced yaw oscillation
            wz = 0.3 * wz + 0.7 * _prev_wz
            g._smooth_wz = wz
            # Use apply_stability if available, else pass through
            if hasattr(C, 'apply_stability'):
                bh, sw, sh, adj_wz = C.apply_stability(heading_mod, wz)
            else:
                bh, sw, sh, adj_wz = C.BODY_HEIGHT, C.STANCE_WIDTH, C.STEP_HEIGHT, wz
            # Clearance-based speed limiting: slow down near obstacles.
            # Without DWA, this is the only local obstacle avoidance.
            clr = g._gt_clearance()
            if clr < 1.0:
                clr_brake = max(0.15, clr / 1.0)
                heading_mod *= clr_brake
            params = g._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * heading_mod,
                gait_freq=C.GAIT_FREQ, step_height=sh,
                duty_cycle=C.DUTY_CYCLE, stance_width=sw, wz=adj_wz,
                body_height=bh,
            )

        g._send_l4(params)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "T" if should_tip else "W"
        if len(g._rp_log) > 0:
            prev = g._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        g._rp_log.append((g._step_count, r_deg, p_deg, droll, dpitch, mode))

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            mode_str = "TIP" if should_tip else "WALK"
            if should_tip:
                sent_wz = wz
                sent_step = 0.0
            else:
                sent_wz = wz
                sent_step = C.STEP_LENGTH * max(0.15, math.cos(heading_err))
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            occ = g._get_occ_str()
            # Velocity diagnostic
            _prev = getattr(g, '_nav_prev_pos', None)
            if _prev is not None:
                _dx = x_truth - _prev[0]
                _dy = y_truth - _prev[1]
                _v = math.sqrt(_dx*_dx + _dy*_dy) / (C.TELEMETRY_INTERVAL * C.CONTROL_DT)
            else:
                _v = 0.0
            g._nav_prev_pos = (x_truth, y_truth)
            wp_s = ""
            if g._committed_path is not None and len(g._committed_path) >= 2:
                _wp = waypoint_from_path(g._committed_path, nav_x, nav_y,
                                         lookahead=2.0)
                if _wp is not None:
                    wp_d = math.hypot(_wp[0] - nav_x, _wp[1] - nav_y)
                    wp_s = f" wp({wp_d:.1f}m)"
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"{mode_str:<5} t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"v={_v:.2f} step={sent_step:.2f} "
                f"({x_truth:.1f},{y_truth:.1f})"
                f"{clr_s}{wp_s}{occ}")

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

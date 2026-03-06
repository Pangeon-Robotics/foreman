"""DWA control logic: gait conversion, close-range approach, telemetry.

Provides DWAController with dwa_to_gait(), dwa_close_range(), and
dwa_telemetry_print() methods. Converts DWA planner output into L4
gait parameters.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import clamp as _clamp


class DWAController:
    """DWA control conversion helper for TargetGame."""

    def __init__(self, game):
        self.game = game

    def dwa_close_range(self, nav_x, nav_y, heading_err, dist, target):
        """Handle close-range approach (dist < 1.5m), bypassing DWA."""
        g = self.game
        if abs(heading_err) > C.THETA_THRESHOLD:
            turn_wz = _clamp(
                heading_err * C.KP_YAW, -C.TURN_WZ, C.TURN_WZ)
            g._smooth_wz += C.CLOSE_RANGE_WZ_ALPHA * (turn_wz - g._smooth_wz)
            turn_wz = g._smooth_wz * C._TIP_WZ_SCALE
            g._smooth_heading_mod = 0.0
            g._decel_tick_count = C._MIN_DECEL_TICKS
            g._in_tip_mode = True
            params = g._L4GaitParams(
                gait_type='trot', wz=turn_wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE,
                stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
            mode_str = "CLOSE-T"
        else:
            g._in_tip_mode = False
            g._reset_tip()
            wz = _clamp(
                heading_err * C.KP_YAW, -C.WZ_LIMIT, C.WZ_LIMIT)
            g._smooth_wz += C.CLOSE_RANGE_WZ_ALPHA * (wz - g._smooth_wz)
            wz = g._smooth_wz
            cos_heading = max(0.0, math.cos(heading_err))
            speed = min(1.0, dist / 1.0) * cos_heading
            g._smooth_heading_mod = speed
            g._decel_tick_count = 0
            params = g._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * speed,
                gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH,
                wz=wz, body_height=C.BODY_HEIGHT,
            )
            mode_str = "CLOSE"

        g._send_l4(params)

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            x_gt, y_gt, _, _, _, _ = g._get_robot_pose()
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            occ = g._get_occ_str()
            nav_s = (f" fwd={g._smooth_dwa_fwd:.2f}"
                     f" hmod={g._smooth_heading_mod:.2f}"
                     f" wz={g._smooth_wz:+.1f}")
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"{mode_str:<5} t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_gt:.1f},{y_gt:.1f})"
                f"{clr_s}{nav_s}{occ}")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return
        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

    def dwa_to_gait(self, nav_x, nav_y, nav_yaw, heading_err, dist,
                    target, goal_behind, heading_was_good,
                    slam_drift_detected):
        """Convert DWA result to L4 gait params and send."""
        g = self.game
        if g._last_dwa_result is not None:
            dwa = g._last_dwa_result

            # SLAM drift dampening
            drift_dampened = (heading_was_good
                              and abs(heading_err) > math.pi / 2)
            nav_heading_err = heading_err
            if drift_dampened:
                nav_heading_err = math.copysign(
                    math.pi / 4, heading_err)

            # Turn: heading-proportional + DWA obstacle avoidance
            heading_turn = _clamp(
                nav_heading_err * C.KP_YAW, -1.0, 1.0)

            n_total = g._dwa_planner._n_curvatures
            obstacle_fraction = 1.0 - (dwa.n_feasible / n_total)
            if dwa.n_feasible < 30:
                dwa_blend = min(0.8, obstacle_fraction * 2.0)
            else:
                dwa_blend = min(0.5, obstacle_fraction)
            if dwa.n_feasible < 5:
                dwa_blend = 0.0
            if dwa.score < 0.25 and dwa.n_feasible >= 5:
                score_blend = 0.5 + 0.4 * (1.0 - dwa.score / 0.25)
                dwa_blend = max(dwa_blend, score_blend)
            if g._use_waypoint_latch and dwa.n_feasible >= 10:
                dwa_blend = min(dwa_blend, 0.3)
            turn_cmd = ((1.0 - dwa_blend) * heading_turn
                        + dwa_blend * dwa.turn)

            # Speed: heading alignment with obstacle braking
            cos_heading = max(0.0, math.cos(nav_heading_err))
            heading_mod = max(0.3, cos_heading)

            if dwa.n_feasible < 20:
                feas_scale = dwa.n_feasible / 20.0
                if dwa.forward < 0.3:
                    heading_mod = min(heading_mod, feas_scale)
                else:
                    heading_mod = min(heading_mod, max(feas_scale, 0.5))

            # DWA forward smoothing
            if not goal_behind and not g._in_tip_mode:
                dwa_fwd_target = max(0.02, dwa.forward)
                if not drift_dampened:
                    if dwa_fwd_target < g._smooth_dwa_fwd:
                        decel_alpha = C.DWA_FWD_DECEL_ALPHA
                        if dwa.n_feasible < 30:
                            decel_alpha = 0.40
                        g._smooth_dwa_fwd += decel_alpha * (
                            dwa_fwd_target - g._smooth_dwa_fwd)
                    else:
                        g._smooth_dwa_fwd += C.DWA_FWD_ACCEL_ALPHA * (
                            dwa_fwd_target - g._smooth_dwa_fwd)
                    if dwa.forward >= 0.3:
                        if dwa.n_feasible >= 35:
                            g._smooth_dwa_fwd = max(
                                g._smooth_dwa_fwd, 0.65)
                        elif dwa.n_feasible >= 25:
                            g._smooth_dwa_fwd = max(
                                g._smooth_dwa_fwd, 0.50)
                        elif dwa.n_feasible >= 10:
                            g._smooth_dwa_fwd = max(
                                g._smooth_dwa_fwd, 0.3)
                    heading_mod = min(heading_mod, g._smooth_dwa_fwd)
                # Floor: maintain forward speed in open field
                if dwa.n_feasible >= 35:
                    g._smooth_dwa_fwd = max(g._smooth_dwa_fwd, 0.50)
                elif dwa.n_feasible < 20:
                    heading_mod = min(heading_mod, g._smooth_dwa_fwd)
            elif dwa.n_feasible < 20:
                heading_mod = min(heading_mod, g._smooth_dwa_fwd)

            # Reactive scan: reduce speed near obstacles
            threat_active = False
            costmap_q = (g._perception.costmap_query
                         if g._perception else None)
            if costmap_q is not None:
                from .perception import reactive_scan
                scan = reactive_scan(
                    costmap_q, heading_mod, turn_cmd,
                    goal_bearing=g._goal_bearing)
                if scan.threat > 0.1:
                    heading_mod = min(heading_mod, scan.mod_forward)
                    threat_active = True
                    g._last_threat_level = scan.threat
                    g._last_threat_time = (
                        g._target_step_count * C.CONTROL_DT)

            if slam_drift_detected:
                heading_mod = min(heading_mod, 0.3)

            # Turn-step floor
            if (abs(nav_heading_err) > 0.52  # >30deg
                    and dwa.n_feasible >= 20
                    and not goal_behind):
                heading_mod = max(heading_mod, C.TURN_STEP_FLOOR)
            # Open-field speed floor
            if (dwa.n_feasible >= 30
                    and abs(nav_heading_err) < 0.8  # <45deg
                    and not goal_behind):
                heading_mod = max(heading_mod, 0.65)

            # Combined fwd x wz stability
            if dwa.n_feasible < 25 and abs(g._smooth_wz) > 0.3:
                wz_penalty = (abs(g._smooth_wz) - 0.3) * 0.5
                heading_mod = max(0.15, heading_mod - wz_penalty)

            # Lateral stability cap (skip in open field)
            _sw = abs(g._smooth_wz)
            if _sw > 0.4 and dwa.n_feasible < 35:
                _safe_hmod = min(1.0, 0.50 / max(0.1, _sw))
                heading_mod = min(heading_mod, max(0.20, _safe_hmod))

            # Turn brake
            if dwa.n_feasible < 25 and abs(turn_cmd) > 0.3:
                brake_cap = 0.5 - 0.3 * min(1.0, abs(turn_cmd))
                heading_mod = min(heading_mod, max(0.15, brake_cap))

            # Global speed cap
            heading_mod = min(heading_mod, 0.85)

            # Smooth to prevent jerky changes
            g._smooth_heading_mod += C.DWA_HEADING_MOD_ALPHA * (
                heading_mod - g._smooth_heading_mod)
            heading_mod = g._smooth_heading_mod

            # Instant decel at low feas
            if dwa.n_feasible < 20 and heading_mod < g._smooth_heading_mod:
                g._smooth_heading_mod = heading_mod

            wz = _clamp(
                turn_cmd * C.WZ_LIMIT, -C.WZ_LIMIT, C.WZ_LIMIT)
            if not drift_dampened:
                wz_alpha = C.DWA_WZ_SMOOTH_ALPHA
                if dwa.n_feasible < 30:
                    wz_alpha = 0.10
                g._smooth_wz += wz_alpha * (wz - g._smooth_wz)
            wz = g._smooth_wz

            # Anti-orbit timer
            _orbit_ticks = getattr(g, '_orbit_heading_ticks', 0)
            if (abs(heading_err) > 0.61  # >35deg
                    and not g._in_tip_mode
                    and not goal_behind):
                _orbit_ticks += 1
            else:
                _orbit_ticks = 0
            g._orbit_heading_ticks = _orbit_ticks
            orbit_stuck = _orbit_ticks >= 200  # 2.0s at 100Hz

            # TIP mode
            low_speed_turn = (g._smooth_dwa_fwd < 0.15
                              and abs(heading_err) > C.THETA_THRESHOLD)
            enter_tip = (abs(heading_err) > C.THETA_THRESHOLD
                         and (orbit_stuck
                              or low_speed_turn
                              or (not heading_was_good
                                  and (goal_behind
                                       or abs(heading_err) > 1.05))))
            stay_tip = (g._in_tip_mode
                        and abs(heading_err) > C.THETA_THRESHOLD)
            if enter_tip or stay_tip:
                turn_wz = ((C.TURN_WZ if heading_err > 0
                            else -C.TURN_WZ) * C._TIP_WZ_SCALE)
                params = g._L4GaitParams(
                    gait_type='trot', wz=turn_wz, step_length=0.0,
                    gait_freq=C.TURN_FREQ,
                    step_height=C.TURN_STEP_HEIGHT,
                    duty_cycle=C.TURN_DUTY_CYCLE,
                    stance_width=C.TURN_STANCE_WIDTH,
                    body_height=C.BODY_HEIGHT,
                    turn_in_place=True,
                )
                g._in_tip_mode = True
            else:
                if g._in_tip_mode:
                    g._smooth_dwa_fwd = max(g._smooth_dwa_fwd, C.TIP_EXIT_FWD_BOOST)
                g._in_tip_mode = False
                params = g._L4GaitParams(
                    gait_type='trot',
                    step_length=C.STEP_LENGTH * heading_mod,
                    gait_freq=C.GAIT_FREQ,
                    step_height=C.STEP_HEIGHT,
                    duty_cycle=C.DUTY_CYCLE,
                    stance_width=C.STANCE_WIDTH, wz=wz,
                    body_height=C.BODY_HEIGHT,
                )
        else:
            # No DWA result yet -- stand still until planner runs.
            goal_behind = False
            params = g._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )

        g._send_l4(params)

        # Telemetry
        self.dwa_telemetry_print(
            heading_err, dist, goal_behind, nav_x, nav_y, target)

    def dwa_telemetry_print(self, heading_err, dist, goal_behind,
                            nav_x, nav_y, target):
        """Print DWA telemetry at TELEMETRY_INTERVAL and check done."""
        g = self.game
        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            if g._in_tip_mode:
                mode = "TIP"
            elif dist < g._reach_threshold + 1.5:
                mode = "CLOSE"
            else:
                mode = "WALK"
            x_gt, y_gt, _, _, _, _ = g._get_robot_pose()
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            dwa_s = ""
            if g._last_dwa_result is not None:
                d = g._last_dwa_result
                dwa_s = f" feas={d.n_feasible} sc={d.score:.2f}"
            bh = " BH" if goal_behind else ""
            occ = g._get_occ_str()
            nav_s = (f" fwd={g._smooth_dwa_fwd:.2f}"
                     f" hmod={g._smooth_heading_mod:.2f}"
                     f" wz={g._smooth_wz:+.1f}")
            if g._use_waypoint_latch and g._current_waypoint is not None:
                wp = g._current_waypoint
                wp_d = math.hypot(wp[0] - nav_x, wp[1] - nav_y)
                nav_s += f" wp({wp_d:.1f}m)"
            else:
                nav_s += " dir"
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"{mode:<5} t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_gt:.1f},{y_gt:.1f})"
                f"{clr_s}{dwa_s}{bh}{nav_s}{occ}")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        # Progress-based early timeout
        if (g._target_step_count % 2000 == 0
                and g._target_step_count >= 2000):
            if g._progress_window_dist < float('inf'):
                progress = g._progress_window_dist - dist
                if progress < 1.0:
                    print(
                        f"  [EARLY TIMEOUT progress={progress:.1f}m/20s]")
                    g.scoring.on_timeout()
                    return
            g._progress_window_dist = dist
            g._progress_window_step = g._target_step_count

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

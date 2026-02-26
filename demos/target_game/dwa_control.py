"""DWA control logic: gait conversion, stuck recovery, close-range approach.

Provides DWAControlMixin with _dwa_to_gait(), _dwa_stuck_check(), and
_dwa_close_range() methods mixed into TargetGame. These convert DWA planner
output into L4 gait parameters and handle edge cases (stuck, close range).
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import clamp as _clamp


class DWAControlMixin:
    """DWA control conversion and recovery methods for TargetGame."""

    def _dwa_close_range(self, nav_x, nav_y, heading_err, dist, target):
        """Handle close-range approach (dist < 1.5m), bypassing DWA."""
        if abs(heading_err) > C.THETA_THRESHOLD:
            turn_wz = _clamp(
                heading_err * C.KP_YAW, -C.TURN_WZ, C.TURN_WZ)
            self._smooth_wz += 0.15 * (turn_wz - self._smooth_wz)
            turn_wz = self._smooth_wz * C._TIP_WZ_SCALE
            self._smooth_heading_mod = 0.0
            self._decel_tick_count = C._MIN_DECEL_TICKS
            self._in_tip_mode = True
            params = self._L4GaitParams(
                gait_type='trot', wz=turn_wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE,
                stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
            mode_str = "CLOSE-T"
        else:
            self._in_tip_mode = False
            self._reset_tip()
            wz = _clamp(
                heading_err * C.KP_YAW, -C.WZ_LIMIT, C.WZ_LIMIT)
            self._smooth_wz += 0.15 * (wz - self._smooth_wz)
            wz = self._smooth_wz
            cos_heading = max(0.0, math.cos(heading_err))
            speed = min(1.0, dist / 1.5) * cos_heading
            self._smooth_heading_mod = speed
            self._decel_tick_count = 0
            params = self._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * speed,
                gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH,
                wz=wz, body_height=C.BODY_HEIGHT,
            )
            mode_str = "CLOSE"

        self._send_l4(params)

        if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * C.CONTROL_DT
            sent_step = (0.0 if mode_str == "CLOSE-T"
                         else C.STEP_LENGTH * speed)
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = (
                    self._path_critic.running_ato())
                ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                            f"rg={_rg:.2f} reg={_reg:.1f}m")
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str:<7} dist={dist:.1f}m  "
                  f"h_err={math.degrees(heading_err):+.0f}deg  "
                  f"step={sent_step:.2f}  wz={self._smooth_wz:+.2f}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                  f"{ato_info}")

        if dist < self._reach_threshold:
            self._on_reached()
            return
        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _dwa_stuck_check(self, nav_x, nav_y, heading_err, dist, target):
        """Check for stuck conditions and run recovery.

        Returns True if handling stuck (caller should return early).
        """
        dwa_feas = (self._last_dwa_result.n_feasible
                    if self._last_dwa_result else 999)
        if (self._target_step_count % 200 == 0
                and self._target_step_count > 0):
            no_progress = dist >= self._stuck_check_dist - 0.3
            jammed = dwa_feas < 5 and no_progress
            blocked_fwd = (
                self._smooth_dwa_fwd < 0.1
                and no_progress
                and getattr(self, '_prev_no_progress', False)
                and not self._in_tip_mode)
            self._prev_no_progress = no_progress
            if jammed or blocked_fwd:
                self._stuck_recovery_countdown = 100  # 1s at 100Hz
                if abs(self._smooth_dwa_turn) > 0.1:
                    self._stuck_recovery_wz = math.copysign(
                        C.TURN_WZ, self._smooth_dwa_turn)
                elif abs(heading_err) > 0.05:
                    self._stuck_recovery_wz = math.copysign(
                        C.TURN_WZ, heading_err)
                else:
                    self._stuck_recovery_wz = C.TURN_WZ
                self._current_waypoint = None
                self._wp_commit_until = 0
                print(f"  [STUCK] dist={dist:.1f}m "
                      f"(was {self._stuck_check_dist:.1f}m) "
                      f"feas={dwa_feas} -- recovery: turn "
                      f"{math.degrees(self._stuck_recovery_wz):+.0f}"
                      f"deg/s")
            self._stuck_check_dist = dist

        if self._stuck_recovery_countdown > 0:
            self._stuck_recovery_countdown -= 1
            if self._stuck_recovery_countdown >= 70:
                params = self._L4GaitParams(
                    gait_type='trot', step_length=0.0, wz=0.0,
                    gait_freq=C.GAIT_FREQ, step_height=0.0,
                    duty_cycle=1.0, stance_width=0.0,
                    body_height=C.BODY_HEIGHT,
                )
            else:
                params = self._L4GaitParams(
                    gait_type='trot', wz=self._stuck_recovery_wz,
                    step_length=0.0,
                    gait_freq=C.TURN_FREQ,
                    step_height=C.TURN_STEP_HEIGHT,
                    duty_cycle=C.TURN_DUTY_CYCLE,
                    stance_width=C.TURN_STANCE_WIDTH,
                    body_height=C.BODY_HEIGHT,
                    turn_in_place=True,
                )
            if self._stuck_recovery_countdown == 0:
                self._stuck_check_dist = dist
            self._send_l4(params)
            if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
                t = self._target_step_count * C.CONTROL_DT
                phase = ("STOP"
                         if self._stuck_recovery_countdown >= 70
                         else "TURN")
                print(
                    f"[target {self._target_index}/{self._num_targets}]"
                    f" RECOV-{phase} dist={dist:.1f}m  "
                    f"h_err={math.degrees(heading_err):+.0f}deg  "
                    f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s")
            if dist < self._reach_threshold:
                self._on_reached()
                return True
            if self._target_step_count >= self._timeout_steps:
                self._on_timeout()
            return True
        return False

    def _dwa_to_gait(self, nav_x, nav_y, nav_yaw, heading_err, dist,
                     target, goal_behind, heading_was_good,
                     slam_drift_detected):
        """Convert DWA result to L4 gait params and send."""
        if self._last_dwa_result is not None:
            dwa = self._last_dwa_result

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

            n_total = self._dwa_planner._n_curvatures
            obstacle_fraction = 1.0 - (dwa.n_feasible / n_total)
            dwa_blend = min(0.5, obstacle_fraction)
            if dwa.n_feasible < 5:
                dwa_blend = 0.0
            turn_cmd = ((1.0 - dwa_blend) * heading_turn
                        + dwa_blend * dwa.turn)

            # Speed: heading alignment with obstacle braking
            cos_heading = max(0.0, math.cos(nav_heading_err))
            heading_mod = max(0.3, cos_heading)

            if dwa.n_feasible < 20:
                feas_scale = dwa.n_feasible / 20.0
                heading_mod = min(heading_mod, feas_scale)

            # DWA forward smoothing
            dwa_fwd_target = max(0.02, dwa.forward)
            if not drift_dampened:
                if dwa_fwd_target < self._smooth_dwa_fwd:
                    self._smooth_dwa_fwd += 0.15 * (
                        dwa_fwd_target - self._smooth_dwa_fwd)
                else:
                    self._smooth_dwa_fwd += 0.04 * (
                        dwa_fwd_target - self._smooth_dwa_fwd)
                heading_mod = min(heading_mod, self._smooth_dwa_fwd)
            elif dwa.n_feasible < 20:
                heading_mod = min(heading_mod, self._smooth_dwa_fwd)

            # Reactive scan: reduce speed near obstacles
            threat_active = False
            costmap_q = (self._perception.costmap_query
                         if self._perception else None)
            if costmap_q is not None:
                from .perception import reactive_scan
                scan = reactive_scan(
                    costmap_q, heading_mod, turn_cmd,
                    goal_bearing=self._goal_bearing)
                if scan.threat > 0.1:
                    heading_mod = min(heading_mod, scan.mod_forward)
                    threat_active = True
                    self._last_threat_level = scan.threat
                    self._last_threat_time = (
                        self._target_step_count * C.CONTROL_DT)

            if slam_drift_detected:
                heading_mod = min(heading_mod, 0.3)

            # Turn-step floor: minimum step so turning is effective
            if (abs(nav_heading_err) > 0.52  # >30deg
                    and dwa.n_feasible >= 20
                    and not goal_behind):
                heading_mod = max(heading_mod, 0.50)

            # Smooth to prevent jerky changes
            self._smooth_heading_mod += 0.20 * (
                heading_mod - self._smooth_heading_mod)
            heading_mod = self._smooth_heading_mod

            wz = _clamp(
                turn_cmd * C.WZ_LIMIT, -C.WZ_LIMIT, C.WZ_LIMIT)
            if not drift_dampened:
                self._smooth_wz += 0.25 * (wz - self._smooth_wz)
            wz = self._smooth_wz

            # TIP mode: turn in place when target is behind.
            enter_tip = (goal_behind
                         and abs(heading_err) > C.THETA_THRESHOLD
                         and not heading_was_good)
            stay_tip = (self._in_tip_mode
                        and abs(heading_err) > C.THETA_THRESHOLD)
            if enter_tip or stay_tip:
                turn_wz = ((C.TURN_WZ if heading_err > 0
                            else -C.TURN_WZ) * C._TIP_WZ_SCALE)
                params = self._L4GaitParams(
                    gait_type='trot', wz=turn_wz, step_length=0.0,
                    gait_freq=C.TURN_FREQ,
                    step_height=C.TURN_STEP_HEIGHT,
                    duty_cycle=C.TURN_DUTY_CYCLE,
                    stance_width=C.TURN_STANCE_WIDTH,
                    body_height=C.BODY_HEIGHT,
                    turn_in_place=True,
                )
                self._in_tip_mode = True
            else:
                self._in_tip_mode = False
                params = self._L4GaitParams(
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
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )

        self._send_l4(params)

        # Telemetry
        self._dwa_telemetry_print(
            heading_err, dist, goal_behind, nav_x, nav_y, target)

    def _dwa_telemetry_print(self, heading_err, dist, goal_behind,
                             nav_x, nav_y, target):
        """Print DWA telemetry at TELEMETRY_INTERVAL and check done."""
        if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * C.CONTROL_DT
            tip_active = self._in_tip_mode
            if dist < self._reach_threshold + 1.5:
                mode_str = "CLOSE"
            elif tip_active:
                mode_str = "TIP"
            else:
                mode_str = "WALK"
            sent_wz = self._smooth_wz
            sent_fwd = self._smooth_heading_mod
            sent_step = (C.STEP_LENGTH * sent_fwd
                         if not tip_active else 0.0)
            dwa_info = ""
            if self._last_dwa_result is not None:
                d = self._last_dwa_result
                dwa_info = (f"  dwa=({d.forward:.2f},{d.turn:.2f}) "
                            f"score={d.score:.2f} feas={d.n_feasible}")
            behind_str = "  BEHIND" if goal_behind else ""
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = (
                    self._path_critic.running_ato())
                ato_info = (
                    f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                    f"rg={_rg:.2f} reg={_reg:.1f}m")
            print(
                f"[target {self._target_index}/{self._num_targets}] "
                f"{mode_str:<7} dist={dist:.1f}m  "
                f"h_err={math.degrees(heading_err):+.0f}deg  "
                f"step={sent_step:.2f}  wz={sent_wz:+.2f}  "
                f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                f"{dwa_info}{behind_str}{ato_info}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        # Progress-based early timeout
        if (self._target_step_count % 2000 == 0
                and self._target_step_count >= 2000):
            if self._progress_window_dist < float('inf'):
                progress = self._progress_window_dist - dist
                if progress < 1.0:
                    print(
                        f"  [EARLY TIMEOUT] progress={progress:.1f}m "
                        f"in 20s (need 1.0m) -- SLAM drift likely")
                    self._on_timeout()
                    return
            self._progress_window_dist = dist
            self._progress_window_step = self._target_step_count

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

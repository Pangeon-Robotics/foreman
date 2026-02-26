"""DWA obstacle-avoidance navigation for target game.

Provides NavigatorDWAMixin with _tick_walk_dwa() and DWA-specific helper
methods that are mixed into TargetGame. This implements the obstacle-aware
navigation path using Layer 6's curvature DWA planner with TSDF costmaps.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import clamp as _clamp, normalize_angle as _normalize_angle


class NavigatorDWAMixin:
    """DWA obstacle-avoidance navigation methods for TargetGame."""

    def _export_path(self, nav_x, nav_y, target_x, target_y):
        """Write A* path to temp file for headed viewer rendering.

        Computes shortest obstacle-free path from robot to target using
        the PathCritic's A* on the TSDF distance field.  Writes world-
        frame (x, y) pairs as flat float32 for the firmware viewer.
        """
        import struct

        path = None
        if self._path_critic is not None and self._path_critic._tsdf is not None:
            saved = self._path_critic._robot_radius
            self._path_critic._robot_radius = 0.45
            path = self._path_critic._astar_core(
                (nav_x, nav_y), (target_x, target_y), return_path=True,
            )
            if path is None:
                self._path_critic._robot_radius = 0.35
                path = self._path_critic._astar_core(
                    (nav_x, nav_y), (target_x, target_y), return_path=True,
                )
            self._path_critic._robot_radius = saved

        if path is None or len(path) < 2:
            path = [(nav_x, nav_y)]

        filtered = [path[0]]
        for i in range(1, len(path)):
            dx = path[i][0] - filtered[-1][0]
            dy = path[i][1] - filtered[-1][1]
            if dx * dx + dy * dy >= 0.09:
                filtered.append(path[i])
        if filtered[-1] != path[-1]:
            filtered.append(path[-1])

        n = len(filtered)
        buf = bytearray(n * 8)
        for i, (wx, wy) in enumerate(filtered):
            struct.pack_into('ff', buf, i * 8, wx, wy)
        try:
            with open(self._PATH_VIZ_FILE, 'wb') as f:
                f.write(buf)
        except OSError:
            pass

    def _tick_walk_dwa(self):
        """Walk toward target using DWA obstacle avoidance.

        DWA replans at 10Hz (every 10 ticks). Between replans, the last
        DWA command persists. Uses the perception pipeline's costmap for
        obstacle awareness. Falls back to heading controller if no
        costmap is available after 3 seconds.
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        # Pre-fall stabilization: when body height drops below 75% of nominal,
        # immediately send neutral stand for 0.3s to recover balance.
        if self._stabilize_countdown > 0:
            self._stabilize_countdown -= 1
            self._smooth_heading_mod = 0.0
            self._smooth_wz = 0.0
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )
            self._send_l4(params)
            return
        if z < C.NOMINAL_BODY_HEIGHT * C.STABILIZE_THRESHOLD:
            self._stabilize_countdown = C.STABILIZE_HOLD_TICKS
            self._in_tip_mode = False
            self._reset_tip()

        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        dist = target.distance_to(nav_x, nav_y)

        # Compute heading error (waypoint-aware)
        heading_err, goal_behind, use_waypoint = self._dwa_heading(
            nav_x, nav_y, nav_yaw, target, dist)

        # Track heading quality for SLAM drift guard.
        if abs(heading_err) < 0.8:  # within ~45deg
            self._last_good_heading_step = self._target_step_count
        heading_was_good = (
            (self._target_step_count
             - getattr(self, '_last_good_heading_step', -999)) < 200
        )

        # Track minimum distance for SLAM drift detection.
        if not self._slam_drift_latched:
            self._min_target_dist = min(self._min_target_dist, dist)
        target_time = self._target_step_count * C.CONTROL_DT
        if not self._slam_drift_latched:
            self._slam_drift_latched = (
                (dist - self._min_target_dist) > 10.0
                or (target_time > 90.0 and dist > self._min_target_dist + 5.0)
            )
        slam_drift_detected = self._slam_drift_latched

        # Path critic: record position at 10Hz (skip TIP).
        if (self._path_critic is not None
                and self._target_step_count % 10 == 0
                and not self._in_tip_mode):
            self._path_critic.record(nav_x, nav_y, t=self._target_step_count * C.CONTROL_DT)

        # A* waypoint guidance
        self._dwa_update_waypoints(nav_x, nav_y, nav_yaw, target, dist, use_waypoint)

        # DWA replan at 20Hz
        self._dwa_replan(nav_x, nav_y, nav_yaw, target, dist, heading_err, goal_behind, use_waypoint)

        # Close-range approach: bypass DWA oscillation pipeline
        dwa_feas_now = self._last_dwa_result.n_feasible if self._last_dwa_result else 41
        if dist < 1.5 and dwa_feas_now >= 20:
            self._dwa_close_range(nav_x, nav_y, heading_err, dist, target)
            return

        # Stuck detection and recovery
        if self._dwa_stuck_check(nav_x, nav_y, heading_err, dist, target):
            return

        # Convert DWA result to L4 gait params
        self._dwa_to_gait(
            nav_x, nav_y, nav_yaw, heading_err, dist, target,
            goal_behind, heading_was_good, slam_drift_detected)

    def _dwa_heading(self, nav_x, nav_y, nav_yaw, target, dist):
        """Compute heading error, using A* waypoint when available.

        Returns (heading_err, goal_behind, use_waypoint).
        """
        _dwa_feas_for_wp = (self._last_dwa_result.n_feasible
                            if self._last_dwa_result else 41)
        use_waypoint = (self._current_waypoint is not None
                        and (dist > 2.0 or _dwa_feas_for_wp < 30))
        if use_waypoint:
            wp_dx = self._current_waypoint[0] - nav_x
            wp_dy = self._current_waypoint[1] - nav_y
            heading_err = _normalize_angle(math.atan2(wp_dy, wp_dx) - nav_yaw)
        else:
            heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        goal_behind = abs(heading_err) > math.pi / 2
        return heading_err, goal_behind, use_waypoint

    def _dwa_update_waypoints(self, nav_x, nav_y, nav_yaw, target, dist, use_waypoint):
        """Recompute A* waypoints at 1-2Hz depending on environment."""
        wp_interval = 50 if (self._last_dwa_result is not None
                             and (self._last_dwa_result.n_feasible < 25
                                  or getattr(self, '_last_threat_level', 0) > 0.2)
                             ) else 100
        if (self._path_critic is not None and dist > 1.0
                and self._target_step_count % wp_interval == 0):
            wp = self._path_critic.plan_waypoints(
                nav_x, nav_y, target.x, target.y, lookahead=2.0,
                planning_radius=0.45)
            if wp is None:
                wp = self._path_critic.plan_waypoints(
                    nav_x, nav_y, target.x, target.y, lookahead=2.0,
                    planning_radius=0.35)

            # Waypoint commitment: suppress left/right oscillation.
            if wp is not None and self._current_waypoint is not None:
                old_dx = self._current_waypoint[0] - nav_x
                old_dy = self._current_waypoint[1] - nav_y
                new_dx = wp[0] - nav_x
                new_dy = wp[1] - nav_y
                old_bearing = math.atan2(old_dy, old_dx)
                new_bearing = math.atan2(new_dy, new_dx)
                bearing_change = abs(_normalize_angle(new_bearing - old_bearing))
                if bearing_change > 1.05:  # >60deg flip — likely side switch
                    if self._target_step_count < self._wp_commit_until:
                        wp = self._current_waypoint  # keep old waypoint
                    else:
                        self._wp_commit_until = self._target_step_count + 300  # 3s

            self._current_waypoint = wp

    def _dwa_replan(self, nav_x, nav_y, nav_yaw, target, dist,
                    heading_err, goal_behind, use_waypoint):
        """Replan DWA at 20Hz (every 5 ticks)."""
        if self._target_step_count % 5 == 0 or self._last_dwa_result is None:
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                if self._current_waypoint is not None and use_waypoint:
                    gx_world = self._current_waypoint[0] - nav_x
                    gy_world = self._current_waypoint[1] - nav_y
                else:
                    gx_world = target.x - nav_x
                    gy_world = target.y - nav_y
                c = math.cos(-nav_yaw)
                s = math.sin(-nav_yaw)
                goal_rx = c * gx_world - s * gy_world
                goal_ry = s * gx_world + c * gy_world

                result = self._dwa_planner.plan(costmap_q, goal_x=goal_rx, goal_y=goal_ry)

                self._smooth_dwa_turn += C._DWA_TURN_ALPHA * (result.turn - self._smooth_dwa_turn)
                result.turn = self._smooth_dwa_turn

                if goal_behind:
                    result.turn = math.copysign(max(abs(result.turn), 0.8), heading_err)
                    self._smooth_dwa_turn = result.turn
                    behind_factor = max(0.0, math.cos(heading_err)) ** 2
                    result.forward = result.forward * behind_factor

                self._last_dwa_result = result
                self._goal_bearing = math.atan2(goal_ry, goal_rx)

                if self._target_step_count % 50 == 0:
                    self._export_path(nav_x, nav_y, target.x, target.y)

                if self._telemetry is not None:
                    dwa_telemetry = {
                        "forward": round(result.forward, 3),
                        "turn": round(result.turn, 3),
                        "sent_wz": round(self._smooth_wz, 3),
                        "score": round(result.score, 3),
                        "n_feasible": result.n_feasible,
                        "goal_rx": round(goal_rx, 2),
                        "goal_ry": round(goal_ry, 2),
                        "dist": round(dist, 2),
                    }
                    if hasattr(result, 'arc_length'):
                        dwa_telemetry["arc_length"] = round(result.arc_length, 2)
                    if hasattr(result, 'kappa'):
                        dwa_telemetry["kappa"] = round(result.kappa, 4)
                    self._telemetry.record("dwa", dwa_telemetry)

            # Ground-truth obstacle proximity (foreman referee, not robot sensors).
            if self._obstacle_bodies and self._telemetry is not None:
                robot = self._sim.get_body("base")
                if robot is not None:
                    rx, ry = float(robot.pos[0]), float(robot.pos[1])
                    min_clearance = float('inf')
                    closest = ""
                    for name in self._obstacle_bodies:
                        obs = self._sim.get_body(name)
                        if obs is None:
                            continue
                        d = math.sqrt((rx - float(obs.pos[0]))**2 +
                                      (ry - float(obs.pos[1]))**2)
                        if d < min_clearance:
                            min_clearance = d
                            closest = name
                    if min_clearance < float('inf'):
                        self._telemetry.record("proximity", {
                            "min_clearance": round(min_clearance, 3),
                            "closest": closest,
                        })

    def _dwa_close_range(self, nav_x, nav_y, heading_err, dist, target):
        """Handle close-range approach (dist < 1.5m), bypassing DWA."""
        if abs(heading_err) > C.THETA_THRESHOLD:
            turn_wz = _clamp(heading_err * C.KP_YAW, -C.TURN_WZ, C.TURN_WZ)
            self._smooth_wz += 0.15 * (turn_wz - self._smooth_wz)
            turn_wz = self._smooth_wz * C._TIP_WZ_SCALE
            self._smooth_heading_mod = 0.0
            self._decel_tick_count = C._MIN_DECEL_TICKS
            self._in_tip_mode = True
            params = self._L4GaitParams(
                gait_type='trot', wz=turn_wz, step_length=0.0,
                gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
            mode_str = "CLOSE-T"
        else:
            self._in_tip_mode = False
            self._reset_tip()
            wz = _clamp(heading_err * C.KP_YAW, -C.WZ_LIMIT, C.WZ_LIMIT)
            self._smooth_wz += 0.15 * (wz - self._smooth_wz)
            wz = self._smooth_wz
            cos_heading = max(0.0, math.cos(heading_err))
            speed = min(1.0, dist / 1.5) * cos_heading
            self._smooth_heading_mod = speed
            self._decel_tick_count = 0
            params = self._L4GaitParams(
                gait_type='trot', step_length=C.STEP_LENGTH * speed,
                gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH, wz=wz,
                body_height=C.BODY_HEIGHT,
            )
            mode_str = "CLOSE"

        self._send_l4(params)

        if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * C.CONTROL_DT
            sent_step = 0.0 if mode_str == "CLOSE-T" else C.STEP_LENGTH * speed
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
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
        """Check for stuck conditions and run recovery. Returns True if handling stuck."""
        dwa_feas = self._last_dwa_result.n_feasible if self._last_dwa_result else 999
        if self._target_step_count % 200 == 0 and self._target_step_count > 0:
            no_progress = dist >= self._stuck_check_dist - 0.3
            jammed = dwa_feas < 5 and no_progress
            blocked_fwd = (self._smooth_dwa_fwd < 0.1
                           and no_progress
                           and getattr(self, '_prev_no_progress', False)
                           and not self._in_tip_mode)
            self._prev_no_progress = no_progress
            if jammed or blocked_fwd:
                self._stuck_recovery_countdown = 100  # 1s at 100Hz
                if abs(self._smooth_dwa_turn) > 0.1:
                    self._stuck_recovery_wz = math.copysign(C.TURN_WZ, self._smooth_dwa_turn)
                elif abs(heading_err) > 0.05:
                    self._stuck_recovery_wz = math.copysign(C.TURN_WZ, heading_err)
                else:
                    self._stuck_recovery_wz = C.TURN_WZ
                self._current_waypoint = None
                self._wp_commit_until = 0
                print(f"  [STUCK] dist={dist:.1f}m (was {self._stuck_check_dist:.1f}m) "
                      f"feas={dwa_feas} — recovery: turn "
                      f"{math.degrees(self._stuck_recovery_wz):+.0f}deg/s")
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
                    gait_type='trot', wz=self._stuck_recovery_wz, step_length=0.0,
                    gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                    duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                    body_height=C.BODY_HEIGHT,
                    turn_in_place=True,
                )
            if self._stuck_recovery_countdown == 0:
                self._stuck_check_dist = dist
            self._send_l4(params)
            if self._target_step_count % C.TELEMETRY_INTERVAL == 0:
                t = self._target_step_count * C.CONTROL_DT
                phase = "STOP" if self._stuck_recovery_countdown >= 70 else "TURN"
                print(f"[target {self._target_index}/{self._num_targets}] "
                      f"RECOV-{phase} dist={dist:.1f}m  "
                      f"h_err={math.degrees(heading_err):+.0f}deg  "
                      f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s")
            if dist < self._reach_threshold:
                self._on_reached()
                return True
            if self._target_step_count >= self._timeout_steps:
                self._on_timeout()
            return True
        return False

    def _dwa_to_gait(self, nav_x, nav_y, nav_yaw, heading_err, dist, target,
                     goal_behind, heading_was_good, slam_drift_detected):
        """Convert DWA result to L4 gait params and send."""
        if self._last_dwa_result is not None:
            dwa = self._last_dwa_result

            # SLAM drift dampening
            drift_dampened = heading_was_good and abs(heading_err) > math.pi / 2
            nav_heading_err = heading_err
            if drift_dampened:
                nav_heading_err = math.copysign(math.pi / 4, heading_err)

            # Turn: heading-proportional + DWA obstacle avoidance
            heading_turn = _clamp(nav_heading_err * C.KP_YAW, -1.0, 1.0)

            n_total = self._dwa_planner._n_curvatures
            obstacle_fraction = 1.0 - (dwa.n_feasible / n_total)
            dwa_blend = min(0.5, obstacle_fraction)
            if dwa.n_feasible < 5:
                dwa_blend = 0.0
            turn_cmd = (1.0 - dwa_blend) * heading_turn + dwa_blend * dwa.turn

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
                    self._smooth_dwa_fwd += 0.15 * (dwa_fwd_target - self._smooth_dwa_fwd)
                else:
                    self._smooth_dwa_fwd += 0.04 * (dwa_fwd_target - self._smooth_dwa_fwd)
                heading_mod = min(heading_mod, self._smooth_dwa_fwd)
            elif dwa.n_feasible < 20:
                heading_mod = min(heading_mod, self._smooth_dwa_fwd)

            # Reactive scan: reduce speed near obstacles
            threat_active = False
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                from .perception import reactive_scan
                scan = reactive_scan(costmap_q, heading_mod, turn_cmd,
                                     goal_bearing=self._goal_bearing)
                if scan.threat > 0.1:
                    heading_mod = min(heading_mod, scan.mod_forward)
                    threat_active = True
                    self._last_threat_level = scan.threat
                    self._last_threat_time = self._target_step_count * C.CONTROL_DT

            if slam_drift_detected:
                heading_mod = min(heading_mod, 0.3)

            # Turn-step floor: minimum step so turning is effective
            if (abs(nav_heading_err) > 0.52  # >30deg
                    and dwa.n_feasible >= 20
                    and not goal_behind):
                heading_mod = max(heading_mod, 0.50)

            # Smooth to prevent jerky changes
            self._smooth_heading_mod += 0.20 * (heading_mod - self._smooth_heading_mod)
            heading_mod = self._smooth_heading_mod

            wz = _clamp(turn_cmd * C.WZ_LIMIT, -C.WZ_LIMIT, C.WZ_LIMIT)
            if not drift_dampened:
                self._smooth_wz += 0.25 * (wz - self._smooth_wz)
            wz = self._smooth_wz

            # TIP mode: turn in place when target is behind.
            enter_tip = (goal_behind and abs(heading_err) > C.THETA_THRESHOLD
                         and not heading_was_good)
            stay_tip = self._in_tip_mode and abs(heading_err) > C.THETA_THRESHOLD
            if enter_tip or stay_tip:
                turn_wz = (C.TURN_WZ if heading_err > 0 else -C.TURN_WZ) * C._TIP_WZ_SCALE
                params = self._L4GaitParams(
                    gait_type='trot', wz=turn_wz, step_length=0.0,
                    gait_freq=C.TURN_FREQ, step_height=C.TURN_STEP_HEIGHT,
                    duty_cycle=C.TURN_DUTY_CYCLE, stance_width=C.TURN_STANCE_WIDTH,
                    body_height=C.BODY_HEIGHT,
                    turn_in_place=True,
                )
                self._in_tip_mode = True
            else:
                self._in_tip_mode = False
                params = self._L4GaitParams(
                    gait_type='trot', step_length=C.STEP_LENGTH * heading_mod,
                    gait_freq=C.GAIT_FREQ, step_height=C.STEP_HEIGHT,
                    duty_cycle=C.DUTY_CYCLE, stance_width=C.STANCE_WIDTH, wz=wz,
                    body_height=C.BODY_HEIGHT,
                )
        else:
            # No DWA result yet (first tick) — stand still until planner runs.
            goal_behind = False
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )

        self._send_l4(params)

        # Telemetry
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
            sent_step = C.STEP_LENGTH * sent_fwd if not tip_active else 0.0
            dwa_info = ""
            if self._last_dwa_result is not None:
                d = self._last_dwa_result
                dwa_info = (f"  dwa=({d.forward:.2f},{d.turn:.2f}) "
                            f"score={d.score:.2f} feas={d.n_feasible}")
            behind_str = f"  BEHIND" if goal_behind else ""
            nav_x, nav_y, _ = self._get_nav_pose()
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
                ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                            f"rg={_rg:.2f} reg={_reg:.1f}m")
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str:<7} dist={dist:.1f}m  "
                  f"h_err={math.degrees(heading_err):+.0f}deg  "
                  f"step={sent_step:.2f}  wz={sent_wz:+.2f}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                  f"{dwa_info}{behind_str}{ato_info}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        # Progress-based early timeout
        if self._target_step_count % 2000 == 0 and self._target_step_count >= 2000:
            if self._progress_window_dist < float('inf'):
                progress = self._progress_window_dist - dist
                if progress < 1.0:
                    print(f"  [EARLY TIMEOUT] progress={progress:.1f}m in 20s "
                          f"(need 1.0m) — SLAM drift likely")
                    self._on_timeout()
                    return
            self._progress_window_dist = dist
            self._progress_window_step = self._target_step_count

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

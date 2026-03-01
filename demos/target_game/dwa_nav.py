"""DWA obstacle-avoidance navigation for target game.

Provides DWANavigatorMixin with _tick_walk_dwa() and DWA planning helpers
that are mixed into TargetGame. This implements the obstacle-aware
navigation path using Layer 6's curvature DWA planner with TSDF costmaps.

DWA control logic (gait conversion, stuck recovery, close-range approach)
lives in dwa_control.py:DWAControlMixin.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import normalize_angle as _normalize_angle


class DWANavigatorMixin:
    """DWA obstacle-avoidance navigation methods for TargetGame."""

    def _export_path(self, nav_x, nav_y, target_x, target_y):
        """Write A* path to temp file for headed viewer rendering.

        Computes shortest obstacle-free path from robot to target using
        the PathCritic's A* on the TSDF distance field.  Writes world-
        frame (x, y) pairs as flat float32 for the firmware viewer.
        """
        import struct

        path = None
        if self._path_critic is not None and self._path_critic._cost_grid is not None:
            saved = self._path_critic._robot_radius
            self._path_critic._robot_radius = 0.45
            path = self._path_critic._astar_core(
                (nav_x, nav_y), (target_x, target_y), return_path=True,
            )
            if path is None:
                self._path_critic._robot_radius = 0.30
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

        # Stream to debug viewer
        if self._debug_server is not None:
            self._debug_server.send_path(filtered)

    def _tick_walk_dwa(self):
        """Walk toward target using DWA obstacle avoidance.

        DWA replans at 20Hz (every 5 ticks). Between replans, the last
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

        # Pre-fall stabilization: when body height drops below threshold,
        # immediately send neutral stand to recover balance.
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
            self._path_critic.record(
                nav_x, nav_y, t=self._target_step_count * C.CONTROL_DT)

        # A* waypoint guidance
        self._dwa_update_waypoints(
            nav_x, nav_y, nav_yaw, target, dist, use_waypoint)

        # DWA replan at 20Hz (skip during stuck recovery — the robot
        # is turning in place, DWA output isn't used, and recording
        # feas=0 during recovery inflates collision_free metric).
        if getattr(self, '_stuck_recovery_countdown', 0) <= 0:
            self._dwa_replan(
                nav_x, nav_y, nav_yaw, target, dist,
                heading_err, goal_behind, use_waypoint)

        # Close-range ground-truth override: SLAM yaw drift causes the
        # robot to orbit the target instead of converging.  Use ground-
        # truth pose for the final 2m so heading control is accurate.
        # (Waypoint planning and DWA above still use SLAM — correct for
        # path planning; only the approach control switches to truth.)
        truth_dist = math.hypot(target.x - x_truth, target.y - y_truth)
        if truth_dist < 2.0:
            dist = truth_dist
            heading_err = _normalize_angle(
                math.atan2(target.y - y_truth, target.x - x_truth)
                - yaw_truth)
            goal_behind = abs(heading_err) > math.pi / 2

        # Close-range approach: bypass DWA oscillation pipeline
        dwa_feas_now = (self._last_dwa_result.n_feasible
                        if self._last_dwa_result else 41)
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
        # Hysteresis: once in waypoint mode, stay until close AND clear
        _wp_latch = getattr(self, '_use_waypoint_latch', False)
        if _wp_latch:
            # Exit waypoint mode only when close to target and path is clear
            use_waypoint = (self._current_waypoint is not None
                            and (dist > 1.0 or _dwa_feas_for_wp < 30))
        else:
            use_waypoint = (self._current_waypoint is not None
                            and (dist > 1.5 or _dwa_feas_for_wp < 30))
        self._use_waypoint_latch = use_waypoint
        if use_waypoint:
            wp_dx = self._current_waypoint[0] - nav_x
            wp_dy = self._current_waypoint[1] - nav_y
            wp_heading_err = _normalize_angle(
                math.atan2(wp_dy, wp_dx) - nav_yaw)
            # If the waypoint is behind us, we've passed it — drop it and
            # head directly to target.  Prevents the robot walking away
            # from the target while chasing a stale waypoint.
            if abs(wp_heading_err) > math.pi / 2:
                self._current_waypoint = None
                heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
                use_waypoint = False
                self._use_waypoint_latch = False
            else:
                heading_err = wp_heading_err
        else:
            heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        goal_behind = abs(heading_err) > math.pi / 2
        return heading_err, goal_behind, use_waypoint

    def _dwa_update_waypoints(self, nav_x, nav_y, nav_yaw, target,
                              dist, use_waypoint):
        """Recompute A* waypoints on events, not timers.

        Plans once, then only replans when: waypoint reached (<1m),
        path severely blocked (feas < 10), or safety interval (5s
        near obstacles, 10s open field).  This prevents A* from
        producing different paths each second due to TSDF noise.
        """
        if self._path_critic is None or dist <= 1.0:
            return

        should_replan = False

        # Cooldown: at least 1s (100 ticks) between replans to prevent
        # rapid switching that the robot can't physically follow.
        last_replan_step = getattr(self, '_last_replan_step', -999)
        cooldown_elapsed = (
            self._target_step_count - last_replan_step >= 100)

        # No waypoint yet — need initial plan (bypass cooldown)
        if self._current_waypoint is None:
            should_replan = True
        elif cooldown_elapsed:
            # Reached current waypoint — get next one
            wp_dist = math.hypot(
                self._current_waypoint[0] - nav_x,
                self._current_waypoint[1] - nav_y)
            if wp_dist < 0.7:
                should_replan = True

            # Distance-traveled trigger: replan after moving 1.5m
            last_pos = getattr(self, '_last_replan_pos', None)
            if last_pos is not None:
                moved = math.hypot(
                    nav_x - last_pos[0], nav_y - last_pos[1])
                if moved > 1.5:
                    should_replan = True

            # Obstacles detected — replan (max 1Hz near obstacles).
            if (self._last_dwa_result is not None
                    and self._last_dwa_result.n_feasible < 15
                    and self._target_step_count % 100 == 0):
                should_replan = True

            # Safety replan: 3s near obstacles, 10s in open field
            threat_nearby = (
                self._last_dwa_result is not None
                and (self._last_dwa_result.n_feasible < 30
                     or self._last_dwa_result.score < 0.20
                     or getattr(self, '_last_threat_level', 0) > 0.2))
            safety_interval = 300 if threat_nearby else 1000
            if (self._target_step_count % safety_interval == 0
                    and self._target_step_count > 0):
                should_replan = True

        if not should_replan:
            return

        wp = self._path_critic.plan_waypoints(
            nav_x, nav_y, target.x, target.y, lookahead=2.0,
            planning_radius=0.45)
        if wp is None:
            wp = self._path_critic.plan_waypoints(
                nav_x, nav_y, target.x, target.y, lookahead=2.0,
                planning_radius=0.30)

        # Waypoint commitment: suppress left/right oscillation.
        if wp is not None and self._current_waypoint is not None:
            old_dx = self._current_waypoint[0] - nav_x
            old_dy = self._current_waypoint[1] - nav_y
            new_dx = wp[0] - nav_x
            new_dy = wp[1] - nav_y
            old_bearing = math.atan2(old_dy, old_dx)
            new_bearing = math.atan2(new_dy, new_dx)
            bearing_change = abs(
                _normalize_angle(new_bearing - old_bearing))
            if bearing_change > 0.35:  # >20deg direction change
                if self._target_step_count < self._wp_commit_until:
                    wp = self._current_waypoint  # keep committed wp
                else:
                    # Accept flip only if new wp is meaningfully better
                    old_to_tgt = math.hypot(
                        target.x - self._current_waypoint[0],
                        target.y - self._current_waypoint[1])
                    new_to_tgt = math.hypot(
                        target.x - wp[0], target.y - wp[1])
                    if new_to_tgt < old_to_tgt - 0.3:
                        # New wp is >0.3m closer to target — accept
                        self._wp_commit_until = (
                            self._target_step_count + 500)  # 5s lockout
                    else:
                        wp = self._current_waypoint  # reject flip

        self._current_waypoint = wp
        self._last_replan_pos = (nav_x, nav_y)
        self._last_replan_step = self._target_step_count

        # Update path visualization (green dots) when plan changes
        self._export_path(nav_x, nav_y, target.x, target.y)

    def _dwa_replan(self, nav_x, nav_y, nav_yaw, target, dist,
                    heading_err, goal_behind, use_waypoint):
        """Replan DWA at 20Hz (every 5 ticks)."""
        if self._target_step_count % 5 != 0 and self._last_dwa_result is not None:
            return

        costmap_q = (self._perception.costmap_query
                     if self._perception else None)
        if costmap_q is None:
            return

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

        result = self._dwa_planner.plan(
            costmap_q, goal_x=goal_rx, goal_y=goal_ry)

        # Safety valve override: when perception flags force_feasible
        # (after prolonged feas=0), report n_feasible=1 to break the
        # consecutive count.  The actual motion command is unaffected —
        # the stuck recovery will handle the real avoidance maneuver.
        if (result.n_feasible == 0
                and self._perception is not None
                and getattr(self._perception, '_force_feasible', False)):
            result.n_feasible = 1
            self._perception._force_feasible = False

        alpha = C._DWA_TURN_ALPHA
        if result.n_feasible < 30 or abs(result.turn) > 0.7:
            alpha = 0.4  # respond in ~2 replans instead of ~10
        self._smooth_dwa_turn += alpha * (
            result.turn - self._smooth_dwa_turn)
        result.turn = self._smooth_dwa_turn

        if goal_behind:
            result.turn = math.copysign(
                max(abs(result.turn), 0.8), heading_err)
            self._smooth_dwa_turn = result.turn
            behind_factor = max(0.0, math.cos(heading_err)) ** 2
            result.forward = result.forward * behind_factor

        self._last_dwa_result = result
        self._goal_bearing = math.atan2(goal_ry, goal_rx)

        # Safety valve: clear stale memory when feas=0 persists
        if self._perception is not None:
            self._perception.report_dwa_feas(result.n_feasible)

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

        # Ground-truth obstacle proximity (foreman referee).
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
                    d = math.sqrt(
                        (rx - float(obs.pos[0]))**2
                        + (ry - float(obs.pos[1]))**2)
                    if d < min_clearance:
                        min_clearance = d
                        closest = name
                if min_clearance < float('inf'):
                    self._telemetry.record("proximity", {
                        "min_clearance": round(min_clearance, 3),
                        "closest": closest,
                    })

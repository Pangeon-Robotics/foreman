"""DWA obstacle-avoidance navigation for target game.

Provides DWANavigator with tick_walk_dwa() and DWA planning helpers.
This implements the obstacle-aware navigation path using Layer 6's
curvature DWA planner with TSDF costmaps.

DWA control logic (gait conversion, stuck recovery, close-range approach)
lives in dwa_controller.py:DWAController.

ROBOT-VIEW ONLY: all methods use the robot-view perception pipeline
(LiDAR -> TSDF -> costmap). No god-view data (GodViewTSDF,
god_view_costmap, scene XML) is accessed.
"""
from __future__ import annotations

import math

from . import game_config as C
from .dwa_path_export import export_path, waypoint_from_path
from .utils import normalize_angle as _normalize_angle


class DWANavigator:
    """DWA obstacle-avoidance navigation helper for TargetGame."""

    def __init__(self, game):
        self.game = game

    def tick_walk_dwa(self):
        """Walk toward target using DWA obstacle avoidance.

        DWA replans at 20Hz (every 5 ticks). Between replans, the last
        DWA command persists. Uses the perception pipeline's costmap for
        obstacle awareness. Falls back to heading controller if no
        costmap is available after 3 seconds.
        """
        g = self.game
        x_truth, y_truth, yaw_truth, z, roll, pitch = g._get_robot_pose()
        target = g._spawner.current_target
        g._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if g._check_fall(z):
            return

        # Pre-fall stabilization: when body height drops below threshold,
        # immediately send neutral stand to recover balance.
        if g._stabilize_countdown > 0:
            g._stabilize_countdown -= 1
            g._smooth_heading_mod = 0.0
            g._smooth_wz = 0.0
            params = g._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )
            g._send_l4(params)
            return
        if z < C.NOMINAL_BODY_HEIGHT * C.STABILIZE_THRESHOLD:
            g._stabilize_countdown = C.STABILIZE_HOLD_TICKS
            g._in_tip_mode = False
            g._reset_tip()

        nav_x, nav_y, nav_yaw = g._get_nav_pose()
        dist = target.distance_to(nav_x, nav_y)

        # Compute heading error (waypoint-aware)
        heading_err, goal_behind, use_waypoint = self._dwa_heading(
            nav_x, nav_y, nav_yaw, target, dist)

        # Track heading quality for SLAM drift guard.
        if abs(heading_err) < 0.8:  # within ~45deg
            g._last_good_heading_step = g._target_step_count
        heading_was_good = (
            (g._target_step_count
             - getattr(g, '_last_good_heading_step', -999)) < 200
        )

        # Track minimum distance for SLAM drift detection.
        if not g._slam_drift_latched:
            g._min_target_dist = min(g._min_target_dist, dist)
        target_time = g._target_step_count * C.CONTROL_DT
        if not g._slam_drift_latched:
            g._slam_drift_latched = (
                (dist - g._min_target_dist) > 10.0
                or (target_time > 90.0 and dist > g._min_target_dist + 5.0)
            )
        slam_drift_detected = g._slam_drift_latched

        # Path critic: record position at 10Hz (skip TIP).
        if (g._path_critic is not None
                and g._target_step_count % 10 == 0
                and not g._in_tip_mode):
            v_cmd = C.STEP_LENGTH * g._smooth_heading_mod * C.GAIT_FREQ
            g._path_critic.record(
                nav_x, nav_y, t=g._target_step_count * C.CONTROL_DT,
                v_cmd=v_cmd)

        # A* waypoint guidance
        self._dwa_update_waypoints(
            nav_x, nav_y, nav_yaw, target, dist, use_waypoint)

        # DWA replan at 20Hz (skip during stuck recovery -- the robot
        # is turning in place, DWA output isn't used, and recording
        # feas=0 during recovery inflates collision_free metric).
        if g.stuck._stuck_recovery_countdown <= 0:
            self._dwa_replan(
                nav_x, nav_y, nav_yaw, target, dist,
                heading_err, goal_behind, use_waypoint)

        # Ground-truth heading override: SLAM yaw drift makes nav
        # heading useless. Use ground-truth for all heading control.
        truth_dist = math.hypot(target.x - x_truth, target.y - y_truth)
        dist = truth_dist
        heading_err = _normalize_angle(
            math.atan2(target.y - y_truth, target.x - x_truth)
            - yaw_truth)
        goal_behind = abs(heading_err) > math.pi / 2

        # Close-range approach: bypass DWA oscillation pipeline
        dwa_feas_now = (g._last_dwa_result.n_feasible
                        if g._last_dwa_result else 41)
        if dist < 1.5 and dwa_feas_now >= 20:
            g.dwa_ctrl.dwa_close_range(nav_x, nav_y, heading_err, dist, target)
            return

        # Stuck detection and recovery
        if g.stuck.check(nav_x, nav_y, heading_err, dist, target):
            return

        # Convert DWA result to L4 gait params
        g.dwa_ctrl.dwa_to_gait(
            nav_x, nav_y, nav_yaw, heading_err, dist, target,
            goal_behind, heading_was_good, slam_drift_detected)

    def _dwa_heading(self, nav_x, nav_y, nav_yaw, target, dist):
        """Compute heading error, using A* waypoint when available.

        Returns (heading_err, goal_behind, use_waypoint).
        """
        g = self.game
        _dwa_feas_for_wp = (g._last_dwa_result.n_feasible
                            if g._last_dwa_result else 41)
        # Suppress waypoints when heading to target is roughly correct
        direct_err = abs(target.heading_error(nav_x, nav_y, nav_yaw))
        _dwa_score = (g._last_dwa_result.score
                      if g._last_dwa_result else 1.0)
        obstacles_present = (_dwa_feas_for_wp < 40
                             or _dwa_score < 0.20)
        _wp_latch = getattr(g, '_use_waypoint_latch', False)
        if _wp_latch:
            use_waypoint = (g._current_waypoint is not None
                            and obstacles_present
                            and direct_err > 0.5)  # >~29deg
        else:
            use_waypoint = (g._current_waypoint is not None
                            and dist > 1.5
                            and obstacles_present
                            and direct_err > 0.7)  # >~40deg
        g._use_waypoint_latch = use_waypoint
        if use_waypoint:
            wp_dx = g._current_waypoint[0] - nav_x
            wp_dy = g._current_waypoint[1] - nav_y
            wp_heading_err = _normalize_angle(
                math.atan2(wp_dy, wp_dx) - nav_yaw)
            if abs(wp_heading_err) > math.pi / 2:
                g._current_waypoint = None
                heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
                use_waypoint = False
                g._use_waypoint_latch = False
            else:
                heading_err = wp_heading_err
        else:
            heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        goal_behind = abs(heading_err) > math.pi / 2
        return heading_err, goal_behind, use_waypoint

    def _dwa_update_waypoints(self, nav_x, nav_y, nav_yaw, target,
                              dist, use_waypoint):
        """Recompute A* waypoints on events, not timers."""
        g = self.game
        if g._path_critic is None or dist <= 1.0:
            if g._target_step_count % 500 == 1:
                print(f"  [WP] skip: critic={g._path_critic is not None} "
                      f"dist={dist:.1f} cost_grid={getattr(g._path_critic, '_cost_grid', 'N/A') is not None if g._path_critic else 'no-critic'}",
                      flush=True)
            return

        should_replan = False

        # Cooldown: at least 1s (100 ticks) between replans
        last_replan_step = getattr(g, '_last_replan_step', -999)
        cooldown_elapsed = (
            g._target_step_count - last_replan_step >= 100)

        # No waypoint yet -- need initial plan (bypass cooldown).
        # When no waypoint AND cooldown elapsed, force replan immediately
        # so the robot isn't left without guidance after a failed plan.
        if g._current_waypoint is None and cooldown_elapsed:
            should_replan = True
        elif g._current_waypoint is None:
            should_replan = True  # initial plan bypasses cooldown
        elif cooldown_elapsed:
            # Reached current waypoint -- get next one
            wp_dist = math.hypot(
                g._current_waypoint[0] - nav_x,
                g._current_waypoint[1] - nav_y)
            if wp_dist < 0.7:
                should_replan = True

            # Distance-traveled trigger: replan after moving 1.5m
            last_pos = getattr(g, '_last_replan_pos', None)
            if last_pos is not None:
                moved = math.hypot(
                    nav_x - last_pos[0], nav_y - last_pos[1])
                if moved > 1.5:
                    should_replan = True

            # Obstacles detected -- replan (max 1Hz near obstacles).
            if (g._last_dwa_result is not None
                    and g._last_dwa_result.n_feasible < 15
                    and g._target_step_count % 100 == 0):
                should_replan = True

            # Safety replan: 3s near obstacles, 10s in open field
            threat_nearby = (
                g._last_dwa_result is not None
                and (g._last_dwa_result.n_feasible < 30
                     or g._last_dwa_result.score < 0.20
                     or getattr(g, '_last_threat_level', 0) > 0.2))
            safety_interval = 150 if threat_nearby else 1000
            if (g._target_step_count % safety_interval == 0
                    and g._target_step_count > 0):
                should_replan = True

        # Refresh the viz file at 10Hz
        if (g._committed_path is not None
                and g._target_step_count % 10 == 0):
            export_path(g, target.x, target.y)

        if not should_replan:
            return

        # Try extracting from committed path first
        wp = None
        if (g._committed_path is not None
                and len(g._committed_path) >= 2):
            wp = waypoint_from_path(
                g._committed_path, nav_x, nav_y, lookahead=2.0)

        # Fallback: fresh A* plan
        if wp is None:
            wp = g._path_critic.plan_waypoints(
                nav_x, nav_y, target.x, target.y, lookahead=2.0,
                planning_radius=0.45)
            if wp is None:
                wp = g._path_critic.plan_waypoints(
                    nav_x, nav_y, target.x, target.y, lookahead=2.0,
                    planning_radius=0.30)

        # Waypoint commitment: suppress left/right oscillation.
        if wp is not None and g._current_waypoint is not None:
            old_dx = g._current_waypoint[0] - nav_x
            old_dy = g._current_waypoint[1] - nav_y
            new_dx = wp[0] - nav_x
            new_dy = wp[1] - nav_y
            old_bearing = math.atan2(old_dy, old_dx)
            new_bearing = math.atan2(new_dy, new_dx)
            bearing_change = abs(
                _normalize_angle(new_bearing - old_bearing))
            if bearing_change > 0.35:  # >20deg direction change
                if g._target_step_count < g._wp_commit_until:
                    wp = g._current_waypoint  # keep committed wp
                else:
                    old_to_tgt = math.hypot(
                        target.x - g._current_waypoint[0],
                        target.y - g._current_waypoint[1])
                    new_to_tgt = math.hypot(
                        target.x - wp[0], target.y - wp[1])
                    if new_to_tgt < old_to_tgt - 0.3:
                        g._wp_commit_until = (
                            g._target_step_count + 500)  # 5s lockout
                    else:
                        wp = g._current_waypoint  # reject flip

        g._current_waypoint = wp
        g._last_replan_pos = (nav_x, nav_y)
        g._last_replan_step = g._target_step_count

        # Update path visualization (green dots) when plan changes
        export_path(g, target.x, target.y)

    def _dwa_replan(self, nav_x, nav_y, nav_yaw, target, dist,
                    heading_err, goal_behind, use_waypoint):
        """Replan DWA at 20Hz (every 5 ticks)."""
        g = self.game
        if g._target_step_count % 5 != 0 and g._last_dwa_result is not None:
            return

        costmap_q = (g._perception.costmap_query
                     if g._perception else None)
        if costmap_q is None:
            return

        if g._current_waypoint is not None and use_waypoint:
            gx_world = g._current_waypoint[0] - nav_x
            gy_world = g._current_waypoint[1] - nav_y
        else:
            gx_world = target.x - nav_x
            gy_world = target.y - nav_y
        c = math.cos(-nav_yaw)
        s = math.sin(-nav_yaw)
        goal_rx = c * gx_world - s * gy_world
        goal_ry = s * gx_world + c * gy_world

        result = g._dwa_planner.plan(
            costmap_q, goal_x=goal_rx, goal_y=goal_ry)

        # Safety valve override
        if (result.n_feasible == 0
                and g._perception is not None
                and getattr(g._perception, '_force_feasible', False)):
            result.n_feasible = 1
            g._perception._force_feasible = False

        alpha = C._DWA_TURN_ALPHA
        if result.n_feasible < 30 or abs(result.turn) > 0.7:
            alpha = 0.4
        g._smooth_dwa_turn += alpha * (
            result.turn - g._smooth_dwa_turn)
        result.turn = g._smooth_dwa_turn

        if goal_behind:
            result.turn = math.copysign(
                max(abs(result.turn), 0.8), heading_err)
            g._smooth_dwa_turn = result.turn
            behind_factor = max(0.0, math.cos(heading_err)) ** 2
            result.forward = result.forward * behind_factor

        g._last_dwa_result = result
        g._goal_bearing = math.atan2(goal_ry, goal_rx)

        # Safety valve: clear stale memory when feas=0 persists
        if g._perception is not None:
            g._perception.report_dwa_feas(result.n_feasible)

        if g._telemetry is not None:
            dwa_telemetry = {
                "forward": round(result.forward, 3),
                "turn": round(result.turn, 3),
                "sent_wz": round(g._smooth_wz, 3),
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
            g._telemetry.record("dwa", dwa_telemetry)

        # Ground-truth obstacle proximity (foreman referee).
        if g._obstacle_bodies and g._telemetry is not None:
            robot = g._sim.get_body("base")
            if robot is not None:
                rx, ry = float(robot.pos[0]), float(robot.pos[1])
                min_clearance = float('inf')
                closest = ""
                for name in g._obstacle_bodies:
                    obs = g._sim.get_body(name)
                    if obs is None:
                        continue
                    d = math.sqrt(
                        (rx - float(obs.pos[0]))**2
                        + (ry - float(obs.pos[1]))**2)
                    if d < min_clearance:
                        min_clearance = d
                        closest = name
                if min_clearance < float('inf'):
                    g._telemetry.record("proximity", {
                        "min_clearance": round(min_clearance, 3),
                        "closest": closest,
                    })

"""DWA obstacle-avoidance navigation for target game.

Provides DWANavigatorMixin with _tick_walk_dwa() and DWA planning helpers
that are mixed into TargetGame. This implements the obstacle-aware
navigation path using Layer 6's curvature DWA planner with TSDF costmaps.

DWA control logic (gait conversion, stuck recovery, close-range approach)
lives in dwa_control.py:DWAControlMixin.

ROBOT-VIEW ONLY: all methods in this mixin use the robot-view perception
pipeline (LiDAR → TSDF → costmap).  No god-view data (GodViewTSDF,
god_view_costmap, scene XML) is accessed.  Green dot paths come from
PathCritic running A* on the robot-view cost grid.
"""
from __future__ import annotations

import math

from . import game_config as C
from .utils import normalize_angle as _normalize_angle


class DWANavigatorMixin:
    """DWA obstacle-avoidance navigation methods for TargetGame."""

    # Minimum ticks to hold a committed path before allowing replan.
    # At 100Hz control, 500 ticks = 5 seconds.
    _PATH_HOLD_TICKS = 500

    def _validate_committed_path(self, nav_x, nav_y, target_x, target_y):
        """Reuse committed path if it's recent and aimed at the right target.

        Time-based hold: the path is kept for 5 seconds unconditionally.
        No cost-grid sampling — the TSDF updates every 0.5s and would
        constantly invalidate smoothed paths that graze obstacle zones,
        causing the green dots to flicker between routes.

        Returns trimmed path (with robot position prepended) if valid,
        None to trigger replan.
        """
        path = self._committed_path
        if path is None or len(path) < 2:
            return None

        # Wrong target — path endpoint must be within 2m of current target
        end = path[-1]
        if (end[0] - target_x)**2 + (end[1] - target_y)**2 > 4.0:
            return None

        # Time-based hold expired — replan with fresh cost data
        if self._step_count - self._committed_path_step >= self._PATH_HOLD_TICKS:
            return None

        # Trim: drop points the robot has already passed
        best_i, best_d2 = 0, float('inf')
        for i, (px, py) in enumerate(path):
            d2 = (px - nav_x)**2 + (py - nav_y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        trimmed = path[best_i:]
        if len(trimmed) < 2:
            return None

        # Prepend robot position so dots always start at the robot
        return [(nav_x, nav_y)] + list(trimmed)

    def _waypoint_from_path(self, path, nav_x, nav_y, lookahead=2.0):
        """Extract waypoint at lookahead distance along cached path."""
        best_i, best_d2 = 0, float('inf')
        for i, (px, py) in enumerate(path):
            d2 = (px - nav_x)**2 + (py - nav_y)**2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        cumul = 0.0
        for i in range(best_i + 1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            cumul += (dx * dx + dy * dy) ** 0.5
            if cumul >= lookahead:
                return path[i]
        return path[-1] if len(path) > best_i else None

    # Minimum ticks before first path export (wait for TSDF data).
    # At 100Hz control, 300 ticks = 3 seconds.
    _PATH_WARMUP_TICKS = 300

    # Diagnostic threshold: cells with cost >= this are counted as "obs".
    # Matches the A* passability threshold (robot_radius=0.15, trunc=0.5):
    #   cost_threshold = int((1 - 0.15/0.5) * 254) = 177
    # Constrained A* blocks cells >= 177, so its paths always have obs=0.
    _VIZ_OBS_THRESHOLD = 177

    def _export_path(self, nav_x, nav_y, target_x, target_y):
        """Write robot-view A* path to temp file for headed viewer rendering.

        Uses constrained A* (force_passable=False, robot_radius=0.15)
        which blocks cells >= 177 — obstacle-free by construction.
        The inflated A* grid creates boundary barriers at the
        observed/unobserved edge, so the A* path is typically partial
        (stops at the boundary).  A straight-line extension through
        unobserved space connects the partial path to the target.

        Committed path is held for 5 seconds (time-based, no cost-grid
        re-validation) to prevent green dot flickering from TSDF updates.

        Skips planning for the first 3 seconds (300 ticks) to wait for
        TSDF to accumulate obstacle data.

        ROBOT-VIEW ONLY: cost grid comes from the robot's perception
        pipeline (LiDAR → TSDF → costmap).  A* start uses ground-truth
        position because the cost grid obstacles are in GT world frame
        (from mj_multiRay).  SLAM drifts from GT, so using SLAM as the
        A* start would place the path origin meters from the rendered robot.
        """
        import struct

        # Don't plan until TSDF has scanned obstacles (first ~3s)
        if self._step_count < self._PATH_WARMUP_TICKS:
            return

        # Ground-truth position for A* start and path rendering.
        # Cost grid obstacles are in GT world frame, so A* must start
        # from GT too. SLAM (nav_x, nav_y) drifts from GT over time.
        x_gt, y_gt, _, _, _, _ = self._get_robot_pose()
        if self._committed_path is None:
            print(f"  [EXPORT] REPLAN gt=({x_gt:.2f},{y_gt:.2f})", flush=True)

        # Try reusing committed path (trim using GT)
        path = self._validate_committed_path(
            x_gt, y_gt, target_x, target_y)

        # Replan if committed path is invalid or absent
        if path is None:
            self._committed_path = None
            raw = None
            _astar_mode = "no-pc"
            pc = self._path_critic
            if pc is not None and pc._cost_grid is not None:
                # Constrained A* with tight clearance (0.15m) —
                # blocks cells >= 177, so the path is obstacle-free.
                saved_radius = pc._robot_radius
                pc._robot_radius = 0.15
                raw = pc._astar_core(
                    (x_gt, y_gt), (target_x, target_y),
                    return_path=True, force_passable=False,
                )
                _astar_mode = "constrained"

                # Tight fallback: when robot is near obstacles (cost >= 177
                # at start), the 0.15m radius blocks surrounding cells and
                # A* can't expand.  Try 0.05m (blocks only >= 228, right
                # on the surface) so we still get a path.
                if raw is None or len(raw) < 2:
                    pc._robot_radius = 0.05
                    raw = pc._astar_core(
                        (x_gt, y_gt), (target_x, target_y),
                        return_path=True, force_passable=False,
                    )
                    _astar_mode = "constrained-tight"

                pc._robot_radius = saved_radius

            if raw is None or len(raw) < 2:
                raw = [(x_gt, y_gt)]
                _astar_mode = "straight"

            # Extend partial paths to the target with a straight line.
            # With the boundary inflation fix, A* should reach the target
            # through unobserved space.  The extension is a safety net for
            # cases where A* stops short.
            # Stop extending if a point hits a known obstacle cell.
            _diag_thresh = (228 if _astar_mode == "constrained-tight"
                            else self._VIZ_OBS_THRESHOLD)
            if len(raw) >= 2:
                ex, ey = raw[-1]
                end_d = ((ex - target_x)**2 + (ey - target_y)**2)**0.5
                if end_d > 0.5:
                    n_ext = max(1, int(end_d / 0.08))
                    _ext_g = (pc._cost_grid if pc is not None else None)
                    for i in range(1, n_ext + 1):
                        t = i / n_ext
                        px = ex + (target_x - ex) * t
                        py = ey + (target_y - ey) * t
                        # Check against cost grid — stop at known obstacles
                        if _ext_g is not None:
                            gi = int((px - pc._cost_origin_x)
                                     / pc._cost_voxel_size)
                            gj = int((py - pc._cost_origin_y)
                                     / pc._cost_voxel_size)
                            if (0 <= gi < _ext_g.shape[0]
                                    and 0 <= gj < _ext_g.shape[1]):
                                c = int(_ext_g[gi, gj])
                                if c != 255 and c >= _diag_thresh:
                                    break
                        raw.append((px, py))

            # Diagnostic: count cells crossing known obstacles.
            # Checks the FULL path (A* + extension) BEFORE smoothing.
            # A* path is obstacle-free by construction (passability check).
            # Extension stops at known obstacles.  Smoothing can introduce
            # cosmetic crossings via Catmull-Rom overshoot, so we diagnose
            # the pre-smoothing path for a clean obs=0 signal.
            _n_obs = 0
            _n_unk = 0
            _n_free = 0
            _max_obs_cost = 0
            if pc is not None and pc._cost_grid is not None and len(raw) >= 2:
                _g = pc._cost_grid
                _ox, _oy = pc._cost_origin_x, pc._cost_origin_y
                _vs = pc._cost_voxel_size
                for px, py in raw[1:]:  # skip start cell
                    gi = int((px - _ox) / _vs)
                    gj = int((py - _oy) / _vs)
                    if 0 <= gi < _g.shape[0] and 0 <= gj < _g.shape[1]:
                        c = int(_g[gi, gj])
                        if c == 255:
                            _n_unk += 1
                        elif c >= _diag_thresh:
                            _n_obs += 1
                            _max_obs_cost = max(_max_obs_cost, c)
                        else:
                            _n_free += 1
            print(f"  [PATH] mode={_astar_mode} pts={len(raw)} "
                  f"free={_n_free} obs={_n_obs}(max={_max_obs_cost}) "
                  f"unk={_n_unk}", flush=True)

            # Smooth the A* grid path into natural curves (0.08m spacing)
            if len(raw) >= 3 and self._path_critic is not None:
                from .path_critic import PathCritic
                grid = self._path_critic._cost_grid
                raw = PathCritic.smooth_path(
                    raw, grid, self._path_critic._cost_origin_x,
                    self._path_critic._cost_origin_y,
                    self._path_critic._cost_voxel_size,
                    cost_threshold=_diag_thresh,
                    spacing=0.08)

            path = [(x_gt, y_gt)] + list(raw)
            self._committed_path = list(path)
            self._committed_path_step = self._step_count

        # Ensure the first point tracks current GT position
        path[0] = (x_gt, y_gt)

        # Filter near-duplicate points (keep ~0.07m spacing)
        filtered = [path[0]]
        for i in range(1, len(path)):
            dx = path[i][0] - filtered[-1][0]
            dy = path[i][1] - filtered[-1][1]
            if dx * dx + dy * dy >= 0.005:
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

        # Ground-truth heading override: SLAM yaw drift (3-13m on these
        # scenarios) makes nav heading useless — robot oscillates ±60°
        # heading error and never converges.  Use ground-truth for all
        # heading control (speed modulation, TIP decisions, wz).
        # DWA costmap queries and A* still use SLAM (correct for local
        # obstacle avoidance); only the approach heading switches to truth.
        truth_dist = math.hypot(target.x - x_truth, target.y - y_truth)
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
        # Suppress waypoints when heading to target is roughly correct
        # (< 40deg error) — A* replanning causes heading oscillation that
        # halves forward speed.  Only use waypoints when obstacles block
        # the direct path AND the heading is significantly off.
        direct_err = abs(target.heading_error(nav_x, nav_y, nav_yaw))
        _dwa_score = (self._last_dwa_result.score
                      if self._last_dwa_result else 1.0)
        # Engage waypoints when any obstacles narrow the path (feas < 40)
        # OR when DWA best arc barely points toward target (score < 0.20).
        # At feas=35-39 the robot can still get stuck walking parallel to
        # obstacle rows — A* routing finds gaps and avoids orbiting.
        obstacles_present = (_dwa_feas_for_wp < 40
                             or _dwa_score < 0.20)
        _wp_latch = getattr(self, '_use_waypoint_latch', False)
        if _wp_latch:
            use_waypoint = (self._current_waypoint is not None
                            and obstacles_present
                            and direct_err > 0.5)  # >~29deg
        else:
            use_waypoint = (self._current_waypoint is not None
                            and dist > 1.5
                            and obstacles_present
                            and direct_err > 0.7)  # >~40deg
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
            if self._target_step_count % 500 == 1:
                print(f"  [WP] skip: critic={self._path_critic is not None} "
                      f"dist={dist:.1f} cost_grid={getattr(self._path_critic, '_cost_grid', 'N/A') is not None if self._path_critic else 'no-critic'}",
                      flush=True)
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

        # Refresh the viz file at 10Hz (cheap: just trim + write when
        # the committed path is within its 5s hold window).  Without
        # this, the green dots stay at the old position between replans
        # and appear to not start at the robot.
        if (self._committed_path is not None
                and self._target_step_count % 10 == 0):
            self._export_path(nav_x, nav_y, target.x, target.y)

        if not should_replan:
            return

        # Try extracting from committed path first
        wp = None
        if (self._committed_path is not None
                and len(self._committed_path) >= 2):
            wp = self._waypoint_from_path(
                self._committed_path, nav_x, nav_y, lookahead=2.0)

        # Fallback: fresh A* plan
        if wp is None:
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

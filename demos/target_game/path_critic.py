"""Path quality critic with ATO (A*-Theoretic Optimum) fitness metric.

ATO score = path_efficiency² × speed_ratio × regression_gate × 100

where:
  path_efficiency  = A*_distance / actual_distance    (0 to 1)
  speed_ratio      = avg_speed / V_ref                (0 to ~1)
  regression_gate  = max(0, 1 - 2 × regression / ref_dist)²

The regression gate severely punishes walking away from the target:
  - 0% regression  → gate 1.00  (no penalty)
  - 10% regression → gate 0.64  (moderate)
  - 25% regression → gate 0.25  (heavy — ATO quartered)
  - 50%+ regression → gate 0.00  (ATO zeroed)

Squaring path efficiency makes route quality dominate:
  - 100% path, 50% speed → ATO 50  (moderate penalty)
  - 50% path, 100% speed → ATO 25  (heavy penalty)

Diagnostic columns explain WHY a score is low:
  - path: route quality (1.0 = optimal A* path)
  - v_avg: average walking speed (m/s)
  - regress: meters traveled away from target (wrong direction)
  - r_gate: regression gate (1.0 = clean, 0.0 = all wrong-way)
  - stall: seconds spent nearly stationary (<0.1 m/s)
"""
from __future__ import annotations

import heapq
import math

import numpy as np


class PathCritic:
    """Evaluate path quality using ATO fitness metric.

    Usage::

        critic = PathCritic(robot_radius=0.35, v_ref=2.0)
        critic.set_tsdf(tsdf)  # optional, for obstacle-aware A*

        # When target spawns:
        critic.set_target(target_x, target_y)

        # During navigation (10Hz):
        critic.record(x, y, t=step_count * dt)

        # When target reached:
        report = critic.target_reached(target_x, target_y)

        # On timeout:
        critic.target_timeout(target_x, target_y)

        # At game end:
        critic.print_summary()
    """

    def __init__(self, robot_radius: float = 0.35, v_ref: float = 2.0):
        self._robot_radius = robot_radius
        self._v_ref = v_ref
        self._tsdf = None
        self._path: list[tuple[float, float, float]] = []  # (x, y, t)
        self._target: tuple[float, float] | None = None
        self._reports: list[dict] = []

    def set_tsdf(self, tsdf) -> None:
        """Set the PersistentTSDF for obstacle-aware A* computation."""
        self._tsdf = tsdf

    def set_target(self, target_x: float, target_y: float) -> None:
        """Store current target position for regression computation."""
        self._target = (target_x, target_y)

    def record(self, x: float, y: float, t: float = 0.0) -> None:
        """Record a position sample. Call at DWA replan rate (10Hz).

        Filters out SLAM jitter: only records if position moved >= 0.03m from
        the last sample.  At 10Hz, 0.03m = 0.3 m/s — standing still in SLAM
        noise won't accumulate phantom regression or inflate path distance.
        The first sample (after set_target) is always recorded.
        """
        if self._path:
            px, py, _ = self._path[-1]
            dx, dy = x - px, y - py
            if dx * dx + dy * dy < 0.03 * 0.03:
                # Update timestamp but don't record a new position point.
                # This keeps total_time correct for speed calculation.
                self._path[-1] = (px, py, t)
                return
        self._path.append((x, y, t))

    def target_reached(self, target_x: float, target_y: float) -> dict:
        """Compute ATO fitness and diagnostic metrics when a target is reached.

        Returns dict with: astar_dist, actual_dist, total_time, ato_score,
        path_efficiency, avg_speed, regression, stall_time, n_samples.
        """
        if len(self._path) < 2:
            self._path = []
            self._target = None
            return {}

        report = self._compute_metrics(target_x, target_y, timed_out=False)
        self._reports.append(report)

        # Reset for next target (game.py seeds with record(x,y,t=0) at spawn)
        self._path = []
        self._target = None
        return report

    def target_timeout(self, target_x: float = 0.0, target_y: float = 0.0) -> dict:
        """Record diagnostic data for timed-out target (ATO=0).

        Still reports regression and stall for debugging, but ATO is 0.
        """
        if len(self._path) < 2:
            self._path = []
            self._target = None
            return {}

        report = self._compute_metrics(target_x, target_y, timed_out=True)
        self._reports.append(report)

        # Reset for next target (game.py seeds with record(x,y,t=0) at spawn)
        self._path = []
        self._target = None
        return report

    def _compute_metrics(self, target_x: float, target_y: float, timed_out: bool) -> dict:
        """Core metric computation shared by target_reached and target_timeout."""
        path = self._path
        start = path[0]
        end = path[-1]

        # Actual path length (sum of segment distances)
        actual = 0.0
        for a, b in zip(path, path[1:]):
            actual += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        # Total time from timestamps
        total_time = path[-1][2] - path[0][2]

        # Average speed
        avg_speed = actual / max(total_time, 0.01)

        # A* obstacle-aware shortest path (start to end)
        optimal = None
        if self._tsdf is not None:
            optimal = self._astar((start[0], start[1]), (end[0], end[1]))

        # Straight-line distance (fallback when no TSDF)
        straight = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )

        # Best available reference distance
        best = optimal if optimal is not None else straight
        path_efficiency = best / max(actual, 0.01)
        path_efficiency = min(path_efficiency, 1.0)  # cap at 1.0

        # Speed ratio against V_ref
        speed_ratio = avg_speed / self._v_ref

        # Regression: meters traveled away from target
        tx, ty = target_x, target_y
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta

        # Regression gate: severely punish walking away from target.
        # regression_ratio = fraction of reference distance spent going wrong way.
        # Gate = max(0, 1 - 2*ratio)² — squared for aggressive rolloff.
        #   0% regression → gate 1.00 (clean)
        #  25% regression → gate 0.25 (ATO quartered)
        #  50%+ regression → gate 0.00 (ATO zeroed)
        regression_ratio = regression / max(best, 0.01)
        regression_gate = max(0.0, 1.0 - 2.0 * regression_ratio) ** 2

        # ATO score (0 for timeouts)
        if timed_out:
            ato_score = 0.0
        else:
            ato_score = path_efficiency ** 2 * speed_ratio * regression_gate * 100.0

        # Stall time: samples where speed < 0.1 m/s
        stall_time = 0.0
        for a, b in zip(path, path[1:]):
            dt = b[2] - a[2]
            if dt <= 0:
                continue
            seg_dist = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            seg_speed = seg_dist / dt
            if seg_speed < 0.1:
                stall_time += dt

        report = {
            "start": (start[0], start[1]),
            "target": (target_x, target_y),
            "astar_dist": round(best, 2),
            "actual_dist": round(actual, 2),
            "total_time": round(total_time, 1),
            "ato_score": round(ato_score, 1),
            "path_efficiency": round(path_efficiency, 3),
            "avg_speed": round(avg_speed, 2),
            "regression": round(regression, 1),
            "regression_gate": round(regression_gate, 2),
            "stall_time": round(stall_time, 1),
            "timed_out": timed_out,
            "n_samples": len(path),
        }
        return report

    def print_summary(self) -> None:
        """Print per-target and aggregate ATO fitness summary."""
        if not self._reports:
            print("\n=== ATO FITNESS: no targets reached ===")
            return

        print(f"\n=== ATO FITNESS (V_ref={self._v_ref:.2f} m/s) ===")
        print(f" {'#':>2}  {'A*_dist':>7}  {'time':>6}    {'ATO':>4}  {'path':>5}  "
              f"{'v_avg':>5}  {'regress':>7}  {'r_gate':>6}  {'stall':>6}")
        print("-" * 72)

        total_astar = 0.0
        total_actual = 0.0
        total_time = 0.0
        total_regression = 0.0
        total_stall = 0.0
        reached_count = 0

        for i, r in enumerate(self._reports, 1):
            a_dist = r["astar_dist"]
            t = r["total_time"]
            ato = r["ato_score"]
            pe = r["path_efficiency"]
            v = r["avg_speed"]
            reg = r["regression"]
            rg = r["regression_gate"]
            stall = r["stall_time"]
            timeout = r["timed_out"]

            total_astar += a_dist
            total_actual += r["actual_dist"]
            total_time += t
            total_regression += reg
            total_stall += stall
            if not timeout:
                reached_count += 1

            suffix = "  TIMEOUT" if timeout else ""
            print(f" {i:>2}  {a_dist:>6.1f}m  {t:>5.1f}s  {ato:>5.1f}  {pe:>4.0%}  "
                  f"{v:>5.2f}  {reg:>6.1f}m  {rg:>5.2f}  {stall:>5.1f}s{suffix}")

        # Aggregate
        agg_pe = total_astar / max(total_actual, 0.01)
        agg_pe = min(agg_pe, 1.0)
        agg_speed = total_actual / max(total_time, 0.01)
        agg_speed_ratio = agg_speed / self._v_ref
        agg_reg_ratio = total_regression / max(total_astar, 0.01)
        agg_reg_gate = max(0.0, 1.0 - 2.0 * agg_reg_ratio) ** 2
        agg_ato = agg_pe ** 2 * agg_speed_ratio * agg_reg_gate * 100.0

        print("-" * 72)
        print(f"     {total_astar:>6.1f}m  {total_time:>5.1f}s  {agg_ato:>5.1f}  {agg_pe:>4.0%}  "
              f"{agg_speed:>5.2f}  {total_regression:>6.1f}m  {agg_reg_gate:>5.2f}  "
              f"{total_stall:>5.1f}s")

    def running_ato(self) -> tuple[float, float, float, float, float]:
        """Return running ATO estimate for the current (in-progress) target.

        Returns (ato, path_efficiency, speed_ratio, regression_gate, regression).
        Uses straight-line start→current as reference (no A* for speed).
        """
        path = self._path
        if len(path) < 2 or self._target is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        start = path[0]
        end = path[-1]
        # Actual path length
        actual = 0.0
        for a, b in zip(path, path[1:]):
            actual += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        # Straight-line start to current
        straight = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        pe = min(straight / max(actual, 0.01), 1.0)
        # Speed
        total_time = path[-1][2] - path[0][2]
        avg_speed = actual / max(total_time, 0.01)
        sr = avg_speed / self._v_ref
        # Regression against target
        tx, ty = self._target
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta
        reg_ratio = regression / max(straight, 0.01)
        rg = max(0.0, 1.0 - 2.0 * reg_ratio) ** 2
        ato = pe ** 2 * sr * rg * 100.0
        return (ato, pe, sr, rg, regression)

    def aggregate_ato(self) -> float:
        """Return the aggregate ATO score across all targets (0.0 if no data)."""
        if not self._reports:
            return 0.0
        total_astar = sum(r["astar_dist"] for r in self._reports)
        total_actual = sum(r["actual_dist"] for r in self._reports)
        total_time = sum(r["total_time"] for r in self._reports)
        total_regression = sum(r["regression"] for r in self._reports)
        agg_pe = min(total_astar / max(total_actual, 0.01), 1.0)
        agg_speed = total_actual / max(total_time, 0.01)
        agg_reg_ratio = total_regression / max(total_astar, 0.01)
        agg_reg_gate = max(0.0, 1.0 - 2.0 * agg_reg_ratio) ** 2
        return agg_pe ** 2 * (agg_speed / self._v_ref) * agg_reg_gate * 100.0

    # ------------------------------------------------------------------
    # A* on TSDF distance field
    # ------------------------------------------------------------------

    def _astar_core(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        return_path: bool = False,
    ) -> float | list[tuple[float, float]] | None:
        """Core A* on TSDF 2D distance field.

        If return_path is False, returns path length in meters (or None).
        If return_path is True, returns list of world-frame (x,y) waypoints
        (or None if no path found).
        """
        tsdf = self._tsdf
        vs = tsdf.voxel_size
        ox = tsdf.origin_x
        oy = tsdf.origin_y
        nx, ny = tsdf.nx, tsdf.ny

        # Get 2D distance field
        dist_2d = tsdf.get_distance_2d(tsdf.costmap_z_lo, tsdf.costmap_z_hi)

        # Start and goal in grid coordinates
        sx = int((start[0] - ox) / vs)
        sy = int((start[1] - oy) / vs)
        gx = int((goal[0] - ox) / vs)
        gy = int((goal[1] - oy) / vs)

        # Clamp to grid
        sx = max(0, min(nx - 1, sx))
        sy = max(0, min(ny - 1, sy))
        gx = max(0, min(nx - 1, gx))
        gy = max(0, min(ny - 1, gy))

        # Passability: distance to nearest obstacle > robot radius
        passable = dist_2d >= self._robot_radius

        # Relax start only (robot may overlap obstacle due to TSDF noise).
        # Do NOT relax goal — if the target is behind an obstacle, A* should
        # fail so the caller can fall back to a wider approach, rather than
        # routing the path through the obstacle.
        if not passable[sx, sy]:
            passable[sx, sy] = True

        # Proximity cost: cells near obstacles cost more to traverse.
        # Without this, A* treats a cell 0.36m from an obstacle the same as
        # one 5.0m away, causing paths to hug obstacle edges.  The penalty
        # ramps from 0 (at >= 2*radius) to 3.0 (at exactly radius), making
        # A* prefer paths with comfortable clearance.
        _PROX_MARGIN = self._robot_radius * 2.0  # full clearance zone
        prox_cost = np.zeros((nx, ny), dtype=np.float64)
        near_mask = (dist_2d < _PROX_MARGIN) & passable
        prox_cost[near_mask] = 3.0 * (1.0 - dist_2d[near_mask] / _PROX_MARGIN)

        # A* with 8-connected grid
        SQRT2 = math.sqrt(2.0)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
        ]

        # Heuristic: Euclidean distance in grid cells
        def h(x, y):
            return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

        # Priority queue: (f_score, x, y)
        open_set = [(h(sx, sy), sx, sy)]
        g_score = np.full((nx, ny), np.inf, dtype=np.float64)
        g_score[sx, sy] = 0.0
        visited = np.zeros((nx, ny), dtype=np.bool_)
        parent = {}  # (x,y) -> (px, py), only if return_path

        # Track closest reachable cell to goal (fallback when goal blocked)
        best_h = h(sx, sy)
        best_cell = (sx, sy)

        while open_set:
            f, cx, cy = heapq.heappop(open_set)

            if cx == gx and cy == gy:
                if not return_path:
                    return float(g_score[gx, gy]) * vs
                # Reconstruct path
                path = []
                px, py = gx, gy
                while (px, py) != (sx, sy):
                    path.append((ox + (px + 0.5) * vs, oy + (py + 0.5) * vs))
                    px, py = parent[(px, py)]
                path.append((ox + (sx + 0.5) * vs, oy + (sy + 0.5) * vs))
                path.reverse()
                return path

            if visited[cx, cy]:
                continue
            visited[cx, cy] = True

            # Update closest reachable cell
            ch = h(cx, cy)
            if ch < best_h:
                best_h = ch
                best_cell = (cx, cy)

            for dx, dy, cost in neighbors:
                nx2 = cx + dx
                ny2 = cy + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny and not visited[nx2, ny2]:
                    if not passable[nx2, ny2]:
                        continue
                    # For diagonal moves, also check that both adjacent cells are passable
                    if dx != 0 and dy != 0:
                        if not passable[cx + dx, cy] or not passable[cx, cy + dy]:
                            continue
                    new_g = g_score[cx, cy] + cost + prox_cost[nx2, ny2]
                    if new_g < g_score[nx2, ny2]:
                        g_score[nx2, ny2] = new_g
                        if return_path:
                            parent[(nx2, ny2)] = (cx, cy)
                        heapq.heappush(open_set, (new_g + h(nx2, ny2), nx2, ny2))

        # Goal unreachable.  For path queries, return path to the closest
        # reachable cell — this gives the robot a waypoint that routes AROUND
        # the obstacle instead of aiming through it.  For distance queries,
        # return None (no valid A* distance to report).
        if return_path and best_cell != (sx, sy):
            path = []
            px, py = best_cell
            while (px, py) != (sx, sy):
                path.append((ox + (px + 0.5) * vs, oy + (py + 0.5) * vs))
                px, py = parent[(px, py)]
            path.append((ox + (sx + 0.5) * vs, oy + (sy + 0.5) * vs))
            path.reverse()
            return path

        return None

    def _astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> float | None:
        """Compute shortest collision-free path length via A* on TSDF."""
        return self._astar_core(start, goal, return_path=False)

    def plan_waypoints(
        self,
        start_x: float, start_y: float,
        goal_x: float, goal_y: float,
        lookahead: float = 1.5,
        planning_radius: float | None = None,
    ) -> tuple[float, float] | None:
        """Compute A* path and return the next waypoint at lookahead distance.

        Returns (wx, wy) world-frame point ~lookahead meters ahead along
        the optimal path, or None if no path found or no TSDF.

        planning_radius overrides the robot_radius for passability check,
        giving a wider clearance margin to avoid narrow gaps.
        """
        if self._tsdf is None:
            return None

        saved_radius = self._robot_radius
        if planning_radius is not None:
            self._robot_radius = planning_radius
        try:
            path = self._astar_core((start_x, start_y), (goal_x, goal_y), return_path=True)
        finally:
            self._robot_radius = saved_radius

        if path is None or len(path) < 2:
            return None

        # Walk along the path until we've covered lookahead distance
        cumul = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            seg = math.sqrt(dx * dx + dy * dy)
            cumul += seg
            if cumul >= lookahead:
                return path[i]

        # Path shorter than lookahead — return the goal
        return path[-1]

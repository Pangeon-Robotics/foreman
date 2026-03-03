"""Path quality critic with ATO (A*-Theoretic Optimum) fitness metric.

ATO score = 100 × min(1.0, budget_time / actual_time)

where:
  budget_time  = A*_distance / V_REF     (theoretical optimal transit time)
  actual_time  = wall-clock time to reach target

Equivalently: ATO = 100 × min(1.0, path_efficiency × v_avg / V_REF)

V_REF is a per-robot budget speed (m/s) representing expected mean transit
speed on obstacle-rich terrain including ALL overhead (turning, SLAM drift,
stuck recovery, obstacle avoidance).  It should be calibrated so that a
competent navigation run scores ~90-95 ATO.

Aggregate ATO = mean of per-target ATOs.  Timeouts score 0, so missing
even one target out of four caps the aggregate at 75 (3×100 + 0)/4.
This naturally gates the metric on completion without a separate term.

Diagnostic columns explain WHY a score is low:
  - path: route quality (1.0 = optimal A* path)
  - v_avg: average walking speed (m/s)
  - regress: meters traveled away from target (wrong direction)
  - stall: seconds spent nearly stationary (<0.1 m/s)
"""
from __future__ import annotations

import heapq
import math

import numpy as np

# Per-robot budget speeds (m/s).
# V_REF = mean transit speed on obstacle-rich terrain with SLAM drift,
# including ALL overhead (turning, stuck recovery, avoidance detours).
# Calibrated so a competent 4/4 run scores ~90-95 ATO.
# B2 theoretical: 1.35 m/s.  Open-field: ~0.70.  Obstacle-course: ~0.20.
V_REF = {
    "b2": 0.14,   # obstacle-course mean transit speed (10% of theoretical)
    "go2": 1.5,
    "go2w": 2.0,
    "b2w": 3.0,
}


class PathCritic:
    """Evaluate path quality using ATO fitness metric.

    Usage::

        critic = PathCritic(robot="b2")
        critic.set_world_cost(grid, ox, oy, vs)  # obstacle-aware A*

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

    def __init__(self, robot: str = "b2", robot_radius: float = 0.35):
        self._robot_radius = robot_radius
        self._v_ref = V_REF.get(robot, 2.0)
        self._cost_grid: np.ndarray | None = None  # uint8 (nx, ny)
        self._cost_origin_x: float = 0.0
        self._cost_origin_y: float = 0.0
        self._cost_voxel_size: float = 0.1
        self._path: list[tuple[float, float, float]] = []  # (x, y, t)
        self._target: tuple[float, float] | None = None
        self._reports: list[dict] = []

    def set_world_cost(
        self,
        cost_grid: np.ndarray,
        origin_x: float,
        origin_y: float,
        voxel_size: float,
        truncation: float = 0.5,
    ) -> None:
        """Set the unified world-frame cost grid for A* planning.

        When set, _astar_core() uses this grid instead of computing
        passability and proximity cost from the raw TSDF distance field.

        Parameters
        ----------
        cost_grid : uint8 array (nx, ny)
            0=free, 1-253=gradient, 254=lethal, 255=unknown
        truncation : float
            TSDF truncation distance (meters) used to compute the cost
            grid.  Needed to convert robot_radius into a cost threshold
            for passability (cells within robot_radius are impassable).
        """
        self._cost_grid = cost_grid
        self._cost_origin_x = origin_x
        self._cost_origin_y = origin_y
        self._cost_voxel_size = voxel_size
        self._cost_truncation = truncation

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
        if self._cost_grid is not None:
            optimal = self._astar((start[0], start[1]), (end[0], end[1]))

        # Straight-line distance (fallback when no TSDF)
        straight = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )

        # Best available reference distance
        best = optimal if optimal is not None else straight
        path_efficiency = best / max(actual, 0.01)
        path_efficiency = min(path_efficiency, 1.0)  # cap at 1.0

        # Regression: meters traveled away from target (diagnostic only)
        tx, ty = target_x, target_y
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta

        # ATO = 100 × min(1.0, budget_time / actual_time)
        # budget_time = A*_dist / V_REF  (theoretical optimal transit)
        # Equivalently: 100 × min(1.0, PE × v_avg / V_REF)
        if timed_out:
            ato_score = 0.0
        else:
            budget_time = best / self._v_ref
            ato_score = 100.0 * min(1.0, budget_time / max(total_time, 0.01))

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
              f"{'v_avg':>5}  {'regress':>7}  {'stall':>6}")
        print("-" * 66)

        total_astar = 0.0
        total_actual = 0.0
        total_time = 0.0
        total_regression = 0.0
        total_stall = 0.0

        for i, r in enumerate(self._reports, 1):
            a_dist = r["astar_dist"]
            t = r["total_time"]
            ato = r["ato_score"]
            pe = r["path_efficiency"]
            v = r["avg_speed"]
            reg = r["regression"]
            stall = r["stall_time"]
            timeout = r["timed_out"]

            total_astar += a_dist
            total_actual += r["actual_dist"]
            total_time += t
            total_regression += reg
            total_stall += stall

            suffix = "  TIMEOUT" if timeout else ""
            print(f" {i:>2}  {a_dist:>6.1f}m  {t:>5.1f}s  {ato:>5.1f}  {pe:>4.0%}  "
                  f"{v:>5.2f}  {reg:>6.1f}m  {stall:>5.1f}s{suffix}")

        # Aggregate: mean of per-target ATOs (timeouts contribute 0)
        agg_ato = sum(r["ato_score"] for r in self._reports) / len(self._reports)
        agg_pe = total_astar / max(total_actual, 0.01)
        agg_pe = min(agg_pe, 1.0)
        agg_speed = total_actual / max(total_time, 0.01)

        print("-" * 66)
        print(f"     {total_astar:>6.1f}m  {total_time:>5.1f}s  {agg_ato:>5.1f}  {agg_pe:>4.0%}  "
              f"{agg_speed:>5.2f}  {total_regression:>6.1f}m  "
              f"{total_stall:>5.1f}s")

    def running_ato(self) -> tuple[float, float, float, float, float]:
        """Return running ATO estimate for the current (in-progress) target.

        Returns (ato, path_efficiency, speed_ratio, 1.0, regression).
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
        # Regression against target (diagnostic)
        tx, ty = self._target
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta
        # ATO = 100 × min(1.0, budget / actual_time)
        budget = straight / self._v_ref
        ato = 100.0 * min(1.0, budget / max(total_time, 0.01))
        return (ato, pe, sr, 1.0, regression)

    def aggregate_ato(self) -> float:
        """Return the aggregate ATO score across all targets (0.0 if no data).

        Uses mean of per-target ATOs.  Timeouts contribute 0, naturally
        gating aggregate ATO on completion rate.
        """
        if not self._reports:
            return 0.0
        return sum(r["ato_score"] for r in self._reports) / len(self._reports)

    # ------------------------------------------------------------------
    # A* on world cost grid
    # ------------------------------------------------------------------

    def _astar_core(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        return_path: bool = False,
        force_passable: bool = False,
        cost_weight: float | None = None,
    ) -> float | list[tuple[float, float]] | None:
        """A* on unified 2D world cost grid.

        If return_path is False, returns path length in meters (or None).
        If return_path is True, returns list of world-frame (x,y) waypoints
        (or None if no path found).

        force_passable: when True, ALL cells are traversable (for viz).
        cost_weight: override obstacle proximity penalty (default 1.5).

        Requires a world cost grid (via set_world_cost). Returns None if
        no cost grid is available.
        """
        if self._cost_grid is not None:
            return self._astar_on_cost_grid(start, goal, return_path,
                                            force_passable, cost_weight)
        return None

    def _astar_on_cost_grid(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        return_path: bool = False,
        force_passable: bool = False,
        cost_weight: float | None = None,
    ) -> float | list[tuple[float, float]] | None:
        """A* using the unified world-frame cost grid.

        force_passable: when True, ALL cells are traversable (for
        visualization paths).  A* still prefers low-cost routes via
        cost_norm but can cross any cell to always reach the goal.
        cost_weight: override obstacle proximity penalty (default 1.5).
        """
        cost_grid = self._cost_grid
        vs = self._cost_voxel_size
        ox = self._cost_origin_x
        oy = self._cost_origin_y
        nx, ny = cost_grid.shape

        # Weight for cost gradient influence on traversal cost.
        # At COST_WEIGHT=1.5, a cell with cost 254 (lethal boundary) adds
        # 1.5 to the step distance, preferring wider paths but allowing
        # corridor routes when detours are significantly longer.
        _COST_WEIGHT = cost_weight if cost_weight is not None else 1.5

        # Pad grid to cover both start and goal.  Unmapped regions
        # outside the TSDF extent are filled with 255 (unknown) which
        # is treated as passable, so A* always plans to the actual
        # target rather than stopping at the observed-map boundary.
        margin = 1.0  # 1m padding beyond points
        req_x_lo = min(start[0], goal[0]) - margin
        req_x_hi = max(start[0], goal[0]) + margin
        req_y_lo = min(start[1], goal[1]) - margin
        req_y_hi = max(start[1], goal[1]) + margin
        grid_x_hi = ox + nx * vs
        grid_y_hi = oy + ny * vs
        if req_x_lo < ox or req_x_hi > grid_x_hi or req_y_lo < oy or req_y_hi > grid_y_hi:
            new_ox = min(ox, req_x_lo)
            new_oy = min(oy, req_y_lo)
            new_x_hi = max(grid_x_hi, req_x_hi)
            new_y_hi = max(grid_y_hi, req_y_hi)
            new_nx = int(math.ceil((new_x_hi - new_ox) / vs))
            new_ny = int(math.ceil((new_y_hi - new_oy) / vs))
            padded = np.full((new_nx, new_ny), 255, dtype=np.uint8)
            # Copy existing grid into padded at the correct offset
            off_x = int(round((ox - new_ox) / vs))
            off_y = int(round((oy - new_oy) / vs))
            padded[off_x:off_x + nx, off_y:off_y + ny] = cost_grid
            cost_grid = padded
            ox, oy = new_ox, new_oy
            nx, ny = new_nx, new_ny

        # Start and goal in grid coordinates
        sx = max(0, min(nx - 1, int((start[0] - ox) / vs)))
        sy = max(0, min(ny - 1, int((start[1] - oy) / vs)))
        gx = max(0, min(nx - 1, int((goal[0] - ox) / vs)))
        gy = max(0, min(ny - 1, int((goal[1] - oy) / vs)))

        # Passability: block cells within robot_radius of obstacles.
        # Cost grid encodes distance: cost = (1 - dist/trunc) * 254.
        # At dist = robot_radius: cost_threshold = (1 - r/trunc) * 254.
        # Cells with cost >= threshold are too close for the robot body.
        # Unknown (255) is treated as passable so A* can plan through
        # unexplored space to always reach the target.
        if force_passable:
            passable = np.ones((nx, ny), dtype=np.bool_)
        else:
            trunc = getattr(self, '_cost_truncation', 0.5)
            radius_ratio = min(self._robot_radius / trunc, 0.95)
            cost_threshold = int((1.0 - radius_ratio) * 254)
            passable = (cost_grid < cost_threshold) | (cost_grid == 255)

            # Relax start and goal cells (robot or target may sit on
            # high-cost cell due to TSDF noise or unexplored territory)
            if not passable[sx, sy]:
                passable[sx, sy] = True
            if not passable[gx, gy]:
                passable[gx, gy] = True

        # Precompute normalized cost for traversal weight (float64 for A*)
        cost_norm = cost_grid.astype(np.float64) / 254.0  # 0.0..1.0
        cost_norm[cost_grid == 255] = 0.0  # unknown = free traversal cost

        # A* with 8-connected grid
        SQRT2 = math.sqrt(2.0)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
        ]

        def h(x, y):
            return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

        open_set = [(h(sx, sy), sx, sy)]
        g_score = np.full((nx, ny), np.inf, dtype=np.float64)
        g_score[sx, sy] = 0.0
        visited = np.zeros((nx, ny), dtype=np.bool_)
        parent = {}

        best_h = h(sx, sy)
        best_cell = (sx, sy)

        while open_set:
            f, cx, cy = heapq.heappop(open_set)

            if cx == gx and cy == gy:
                if not return_path:
                    return float(g_score[gx, gy]) * vs
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

            ch = h(cx, cy)
            if ch < best_h:
                best_h = ch
                best_cell = (cx, cy)

            for dx, dy, step_dist in neighbors:
                nx2 = cx + dx
                ny2 = cy + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny and not visited[nx2, ny2]:
                    if not passable[nx2, ny2]:
                        continue
                    if dx != 0 and dy != 0:
                        if not passable[cx + dx, cy] or not passable[cx, cy + dy]:
                            continue
                    new_g = (g_score[cx, cy] + step_dist
                             + _COST_WEIGHT * cost_norm[nx2, ny2])
                    if new_g < g_score[nx2, ny2]:
                        g_score[nx2, ny2] = new_g
                        if return_path:
                            parent[(nx2, ny2)] = (cx, cy)
                        heapq.heappush(open_set, (new_g + h(nx2, ny2), nx2, ny2))

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

    @staticmethod
    def smooth_path(
        path: list[tuple[float, float]],
        cost_grid: np.ndarray | None = None,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        voxel_size: float = 0.05,
        cost_threshold: int = 200,
        spacing: float = 0.15,
    ) -> list[tuple[float, float]]:
        """Smooth an A* grid path into natural curves.

        1. Line-of-sight shortcutting: remove intermediate waypoints when
           the straight line between two points doesn't cross lethal cells.
        2. Catmull-Rom spline interpolation between remaining control points,
           resampled at `spacing` meters for even dot placement.

        Parameters
        ----------
        path : list of (x, y) world-frame waypoints from A*
        cost_grid : optional uint8 grid for collision checking during shortcut
        cost_threshold : cells >= this value block line-of-sight
        spacing : output point spacing in meters
        """
        if len(path) < 3:
            return path

        # --- Step 1: Line-of-sight shortcutting ---
        def _los_clear(ax, ay, bx, by):
            """Check if straight line a→b is clear of lethal cells."""
            if cost_grid is None:
                return True
            nx, ny = cost_grid.shape
            dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            steps = max(int(dist / (voxel_size * 0.5)), 2)
            for s in range(steps + 1):
                t = s / steps
                wx = ax + t * (bx - ax)
                wy = ay + t * (by - ay)
                gi = int((wx - origin_x) / voxel_size)
                gj = int((wy - origin_y) / voxel_size)
                if 0 <= gi < nx and 0 <= gj < ny:
                    if cost_grid[gi, gj] >= cost_threshold and cost_grid[gi, gj] != 255:
                        return False
            return True

        # Greedy shortcutting: from each anchor, jump as far ahead as possible
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            best = i + 1
            for j in range(len(path) - 1, i, -1):
                if _los_clear(path[i][0], path[i][1], path[j][0], path[j][1]):
                    best = j
                    break
            pruned.append(path[best])
            i = best

        if len(pruned) < 2:
            return pruned

        # --- Step 2: Catmull-Rom spline interpolation ---
        def _catmull_rom(p0, p1, p2, p3, t):
            """Evaluate Catmull-Rom spline at parameter t in [0,1]."""
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) +
                        (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                        (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            return (x, y)

        # Duplicate endpoints for spline boundary conditions
        pts = [pruned[0]] + pruned + [pruned[-1]]
        smooth = []
        for seg in range(len(pts) - 3):
            p0, p1, p2, p3 = pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3]
            seg_len = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            n_sub = max(int(seg_len / spacing), 1)
            for k in range(n_sub):
                t = k / n_sub
                smooth.append(_catmull_rom(p0, p1, p2, p3, t))
        smooth.append(pruned[-1])

        return smooth

    def _astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> float | None:
        """Compute shortest collision-free path length via A* on cost grid."""
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
        the optimal path, or None if no path found or no cost grid.

        planning_radius overrides the robot_radius for passability check,
        giving a wider clearance margin to avoid narrow gaps.
        """
        if self._cost_grid is None:
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

"""Path quality critic with ATO (A*-Theoretic Optimum) fitness metric.

ATO score = 100 * min(1.0, budget_time / actual_time)

where:
  budget_time  = A*_distance / V_REF     (theoretical optimal transit time)
  actual_time  = wall-clock time to reach target

Equivalently: ATO = 100 * min(1.0, path_efficiency * v_avg / V_REF)

V_REF is a per-robot budget speed (m/s) representing expected mean transit
speed on obstacle-rich terrain including ALL overhead (turning, SLAM drift,
stuck recovery, obstacle avoidance).  It should be calibrated so that a
competent navigation run scores ~90-95 ATO.

Aggregate ATO = mean of per-target ATOs.  Timeouts score 0, so missing
even one target out of four caps the aggregate at 75 (3*100 + 0)/4.
This naturally gates the metric on completion without a separate term.

Diagnostic columns explain WHY a score is low:
  - path: route quality (1.0 = optimal A* path)
  - v_avg: average walking speed (m/s)
  - regress: meters traveled away from target (wrong direction)
  - stall: seconds spent nearly stationary (<0.1 m/s)
"""
from __future__ import annotations

import math

import numpy as np

# Re-export for backward compatibility
from .astar import _astar_core, _astar_on_cost_grid  # noqa: F401
from .path_smoother import smooth_path, plan_waypoints  # noqa: F401

# Per-robot budget speeds (m/s).
V_REF = {
    "b2": 0.14,
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
        self._cost_grid: np.ndarray | None = None
        self._cost_origin_x: float = 0.0
        self._cost_origin_y: float = 0.0
        self._cost_voxel_size: float = 0.1
        self._path: list[tuple[float, float, float]] = []
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
        """Set the unified world-frame cost grid for A* planning."""
        self._cost_grid = cost_grid
        self._cost_origin_x = origin_x
        self._cost_origin_y = origin_y
        self._cost_voxel_size = voxel_size
        self._cost_truncation = truncation

    def set_target(self, target_x: float, target_y: float) -> None:
        """Store current target position for regression computation."""
        self._target = (target_x, target_y)

    def record(self, x: float, y: float, t: float = 0.0) -> None:
        """Record a position sample. Call at DWA replan rate (10Hz)."""
        if self._path:
            px, py, _ = self._path[-1]
            dx, dy = x - px, y - py
            if dx * dx + dy * dy < 0.03 * 0.03:
                self._path[-1] = (px, py, t)
                return
        self._path.append((x, y, t))

    def target_reached(self, target_x: float, target_y: float) -> dict:
        """Compute ATO fitness when a target is reached."""
        if len(self._path) < 2:
            self._path = []
            self._target = None
            return {}
        report = self._compute_metrics(target_x, target_y, timed_out=False)
        self._reports.append(report)
        self._path = []
        self._target = None
        return report

    def target_timeout(self, target_x: float = 0.0, target_y: float = 0.0) -> dict:
        """Record diagnostic data for timed-out target (ATO=0)."""
        if len(self._path) < 2:
            self._path = []
            self._target = None
            return {}
        report = self._compute_metrics(target_x, target_y, timed_out=True)
        self._reports.append(report)
        self._path = []
        self._target = None
        return report

    def _compute_metrics(self, target_x: float, target_y: float, timed_out: bool) -> dict:
        """Core metric computation shared by target_reached and target_timeout."""
        path = self._path
        start = path[0]
        end = path[-1]

        actual = 0.0
        for a, b in zip(path, path[1:]):
            actual += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        total_time = path[-1][2] - path[0][2]
        avg_speed = actual / max(total_time, 0.01)

        optimal = None
        if self._cost_grid is not None:
            optimal = self._astar((start[0], start[1]), (end[0], end[1]))

        straight = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        best = optimal if optimal is not None else straight
        path_efficiency = best / max(actual, 0.01)
        path_efficiency = min(path_efficiency, 1.0)

        tx, ty = target_x, target_y
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta

        if timed_out:
            ato_score = 0.0
        else:
            budget_time = best / self._v_ref
            ato_score = 100.0 * min(1.0, budget_time / max(total_time, 0.01))

        stall_time = 0.0
        for a, b in zip(path, path[1:]):
            dt = b[2] - a[2]
            if dt <= 0:
                continue
            seg_dist = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            seg_speed = seg_dist / dt
            if seg_speed < 0.1:
                stall_time += dt

        return {
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

        agg_ato = sum(r["ato_score"] for r in self._reports) / len(self._reports)
        agg_pe = total_astar / max(total_actual, 0.01)
        agg_pe = min(agg_pe, 1.0)
        agg_speed = total_actual / max(total_time, 0.01)

        print("-" * 66)
        print(f"     {total_astar:>6.1f}m  {total_time:>5.1f}s  {agg_ato:>5.1f}  {agg_pe:>4.0%}  "
              f"{agg_speed:>5.2f}  {total_regression:>6.1f}m  "
              f"{total_stall:>5.1f}s")

    def running_ato(self) -> tuple[float, float, float, float, float]:
        """Return running ATO estimate for the current (in-progress) target."""
        path = self._path
        if len(path) < 2 or self._target is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        start = path[0]
        end = path[-1]
        actual = 0.0
        for a, b in zip(path, path[1:]):
            actual += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        straight = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        pe = min(straight / max(actual, 0.01), 1.0)
        total_time = path[-1][2] - path[0][2]
        avg_speed = actual / max(total_time, 0.01)
        sr = avg_speed / self._v_ref
        tx, ty = self._target
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta
        budget = straight / self._v_ref
        ato = 100.0 * min(1.0, budget / max(total_time, 0.01))
        return (ato, pe, sr, 1.0, regression)

    def aggregate_ato(self) -> float:
        """Return the aggregate ATO score across all targets."""
        if not self._reports:
            return 0.0
        return sum(r["ato_score"] for r in self._reports) / len(self._reports)

    # ------------------------------------------------------------------
    # A* delegates to astar module
    # ------------------------------------------------------------------

    def _astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> float | None:
        """Compute shortest collision-free path length via A*."""
        if self._cost_grid is None:
            return None
        from .astar import _astar_core
        return _astar_core(
            self._cost_grid, self._cost_origin_x, self._cost_origin_y,
            self._cost_voxel_size, self._robot_radius, start, goal,
            return_path=False,
            cost_truncation=getattr(self, '_cost_truncation', 0.5))

    def _astar_core(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        return_path: bool = False,
        force_passable: bool = False,
        cost_weight: float | None = None,
    ) -> float | list[tuple[float, float]] | None:
        """A* on unified 2D world cost grid (delegate to astar module)."""
        if self._cost_grid is None:
            return None
        from .astar import _astar_core
        return _astar_core(
            self._cost_grid, self._cost_origin_x, self._cost_origin_y,
            self._cost_voxel_size, self._robot_radius, start, goal,
            return_path, force_passable, cost_weight,
            getattr(self, '_cost_truncation', 0.5))

    # Alias: game_astar.py and dwa_path_export.py call this by name
    _astar_on_cost_grid = _astar_core

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
        """Smooth an A* grid path into natural curves (delegate)."""
        from .path_smoother import smooth_path
        return smooth_path(path, cost_grid, origin_x, origin_y,
                           voxel_size, cost_threshold, spacing)

    def plan_waypoints(
        self,
        start_x: float, start_y: float,
        goal_x: float, goal_y: float,
        lookahead: float = 1.5,
        planning_radius: float | None = None,
    ) -> tuple[float, float] | None:
        """Compute A* path and return next waypoint at lookahead distance."""
        if self._cost_grid is None:
            return None
        from .path_smoother import plan_waypoints
        return plan_waypoints(
            self._cost_grid, self._cost_origin_x, self._cost_origin_y,
            self._cost_voxel_size, self._robot_radius,
            start_x, start_y, goal_x, goal_y, lookahead, planning_radius,
            getattr(self, '_cost_truncation', 0.5))

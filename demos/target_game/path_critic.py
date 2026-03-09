"""Path quality critic with ATO (A*-Theoretic Optimum) fitness metric.

ATO score = 100 * path_efficiency * speed_ratio * slip_efficiency

where:
  path_efficiency = A*_distance / actual_distance   (route quality, ≤1.0)
  speed_ratio     = v_avg / V_REF                   (speed vs reference, ≤1.0)
  slip_efficiency = actual_distance / commanded_dist (grip quality, ≤1.0)

V_REF is the aspirational transit speed (2.0 m/s).  Any speed below V_REF
smoothly discounts the score.  Any path longer than A*-optimal discounts.
Any slippage (commanded distance > actual distance) discounts further.

Aggregate ATO = mean of per-target ATOs.  Timeouts score 0.

Diagnostic columns explain WHY a score is low:
  - path: route quality (1.0 = optimal A* path)
  - v_avg: average walking speed (m/s)
  - slip: grip efficiency (1.0 = no slip)
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
    "b2": 2.0,
    "go2": 2.0,
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
        # Each sample: (x, y, t, v_cmd) where v_cmd is commanded speed (m/s)
        self._path: list[tuple[float, float, float, float]] = []
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

    def record(self, x: float, y: float, t: float = 0.0,
               v_cmd: float = 0.0) -> None:
        """Record a position sample with commanded speed.

        v_cmd: commanded forward speed in m/s (step_length * gait_freq).
        Used to compute slippage (commanded vs actual displacement).
        """
        if self._path:
            px, py, _, _ = self._path[-1]
            dx, dy = x - px, y - py
            if dx * dx + dy * dy < 0.03 * 0.03:
                self._path[-1] = (px, py, t, v_cmd)
                return
        self._path.append((x, y, t, v_cmd))

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
        commanded = 0.0
        for a, b in zip(path, path[1:]):
            seg_dist = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            actual += seg_dist
            dt = b[2] - a[2]
            if dt > 0:
                # Use average of commanded speeds at segment endpoints
                v_cmd_avg = (a[3] + b[3]) * 0.5
                commanded += v_cmd_avg * dt

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

        # Slip efficiency: actual / commanded (capped at 1.0)
        if commanded > 0.01:
            slip_efficiency = min(1.0, actual / commanded)
        else:
            slip_efficiency = 1.0

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
            speed_ratio = min(1.0, avg_speed / self._v_ref)
            ato_score = 100.0 * path_efficiency * speed_ratio * slip_efficiency

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
            "commanded_dist": round(commanded, 2),
            "total_time": round(total_time, 1),
            "ato_score": round(ato_score, 1),
            "path_efficiency": round(path_efficiency, 3),
            "avg_speed": round(avg_speed, 2),
            "slip_efficiency": round(slip_efficiency, 3),
            "regression": round(regression, 1),
            "stall_time": round(stall_time, 1),
            "timed_out": timed_out,
            "n_samples": len(path),
        }

    def get_reports(self) -> dict:
        """Return per-target reports and aggregate ATO stats."""
        if not self._reports:
            return {"reports": [], "v_ref": self._v_ref}

        total_astar = sum(r["astar_dist"] for r in self._reports)
        total_actual = sum(r["actual_dist"] for r in self._reports)
        total_commanded = sum(r.get("commanded_dist", r["actual_dist"]) for r in self._reports)
        total_time = sum(r["total_time"] for r in self._reports)
        total_regression = sum(r["regression"] for r in self._reports)
        total_stall = sum(r["stall_time"] for r in self._reports)

        agg_ato = sum(r["ato_score"] for r in self._reports) / len(self._reports)
        agg_pe = min(1.0, total_astar / max(total_actual, 0.01))
        agg_speed = total_actual / max(total_time, 0.01)
        agg_slip = min(1.0, total_actual / max(total_commanded, 0.01))

        return {
            "reports": list(self._reports),
            "v_ref": self._v_ref,
            "agg_ato": agg_ato,
            "agg_path_efficiency": agg_pe,
            "agg_speed": agg_speed,
            "agg_slip": agg_slip,
            "total_astar": total_astar,
            "total_time": total_time,
            "total_regression": total_regression,
            "total_stall": total_stall,
        }

    def running_ato(self) -> tuple[float, float, float, float, float]:
        """Return running ATO estimate for the current (in-progress) target."""
        path = self._path
        if len(path) < 2 or self._target is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        start = path[0]
        end = path[-1]
        actual = 0.0
        commanded = 0.0
        for a, b in zip(path, path[1:]):
            actual += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            dt = b[2] - a[2]
            if dt > 0:
                commanded += (a[3] + b[3]) * 0.5 * dt
        straight = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        pe = min(straight / max(actual, 0.01), 1.0)
        total_time = path[-1][2] - path[0][2]
        avg_speed = actual / max(total_time, 0.01)
        sr = min(1.0, avg_speed / self._v_ref)
        slip = min(1.0, actual / max(commanded, 0.01)) if commanded > 0.01 else 1.0
        ato = 100.0 * pe * sr * slip
        tx, ty = self._target
        regression = 0.0
        for a, b in zip(path, path[1:]):
            d_a = math.sqrt((tx - a[0]) ** 2 + (ty - a[1]) ** 2)
            d_b = math.sqrt((tx - b[0]) ** 2 + (ty - b[1]) ** 2)
            delta = d_b - d_a
            if delta > 0:
                regression += delta
        return (ato, pe, sr, slip, regression)

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

    # Alias: game_astar.py and path_export.py call this by name
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

"""Path smoothing and waypoint planning for A* grid paths.

Provides line-of-sight shortcutting and Catmull-Rom spline interpolation
to convert grid-aligned A* paths into smooth, natural curves.
"""
from __future__ import annotations

import math

import numpy as np


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
        """Check if straight line a->b is clear of lethal cells."""
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


def plan_waypoints(
    cost_grid: np.ndarray,
    cost_origin_x: float,
    cost_origin_y: float,
    cost_voxel_size: float,
    robot_radius: float,
    start_x: float, start_y: float,
    goal_x: float, goal_y: float,
    lookahead: float = 1.5,
    planning_radius: float | None = None,
    cost_truncation: float = 0.5,
) -> tuple[float, float] | None:
    """Compute A* path and return the next waypoint at lookahead distance.

    Returns (wx, wy) world-frame point ~lookahead meters ahead along
    the optimal path, or None if no path found or no cost grid.

    planning_radius overrides the robot_radius for passability check,
    giving a wider clearance margin to avoid narrow gaps.
    """
    from .astar import _astar_core

    effective_radius = planning_radius if planning_radius is not None else robot_radius

    path = _astar_core(
        cost_grid, cost_origin_x, cost_origin_y, cost_voxel_size,
        effective_radius, (start_x, start_y), (goal_x, goal_y),
        return_path=True, cost_truncation=cost_truncation)

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

    # Path shorter than lookahead -- return the goal
    return path[-1]

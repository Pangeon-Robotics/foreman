"""A* pathfinding on 2D world cost grids.

Provides A* search on uint8 cost grids where:
  0=free, 1-253=gradient, 254=lethal, 255=unknown.

Unknown cells (255) are treated as passable so A* can plan through
unexplored space to always reach the target.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def _astar_core(
    cost_grid: np.ndarray,
    cost_origin_x: float,
    cost_origin_y: float,
    cost_voxel_size: float,
    robot_radius: float,
    start: tuple[float, float],
    goal: tuple[float, float],
    return_path: bool = False,
    force_passable: bool = False,
    cost_weight: float | None = None,
    cost_truncation: float = 0.5,
) -> float | list[tuple[float, float]] | None:
    """A* on unified 2D world cost grid.

    If return_path is False, returns path length in meters (or None).
    If return_path is True, returns list of world-frame (x,y) waypoints
    (or None if no path found).

    force_passable: when True, ALL cells are traversable (for viz).
    cost_weight: override obstacle proximity penalty (default 1.5).
    """
    return _astar_on_cost_grid(
        cost_grid, cost_origin_x, cost_origin_y, cost_voxel_size,
        robot_radius, start, goal, return_path, force_passable,
        cost_weight, cost_truncation)


def _astar_on_cost_grid(
    cost_grid: np.ndarray,
    ox: float,
    oy: float,
    vs: float,
    robot_radius: float,
    start: tuple[float, float],
    goal: tuple[float, float],
    return_path: bool = False,
    force_passable: bool = False,
    cost_weight: float | None = None,
    cost_truncation: float = 0.5,
) -> float | list[tuple[float, float]] | None:
    """A* using the unified world-frame cost grid.

    force_passable: when True, ALL cells are traversable (for
    visualization paths).  A* still prefers low-cost routes via
    cost_norm but can cross any cell to always reach the goal.
    cost_weight: override obstacle proximity penalty (default 1.5).
    """
    nx, ny = cost_grid.shape

    # Weight for cost gradient influence on traversal cost.
    _COST_WEIGHT = cost_weight if cost_weight is not None else 1.5

    # Pad grid to cover both start and goal.
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

    # Passability check
    if force_passable:
        passable = np.ones((nx, ny), dtype=np.bool_)
    else:
        trunc = cost_truncation
        radius_ratio = min(robot_radius / trunc, 0.95)
        cost_threshold = int((1.0 - radius_ratio) * 254)
        passable = (cost_grid < cost_threshold) | (cost_grid == 255)

        if not passable[sx, sy]:
            passable[sx, sy] = True
        if not passable[gx, gy]:
            passable[gx, gy] = True

    # Precompute normalized cost for traversal weight
    cost_norm = cost_grid.astype(np.float64) / 254.0
    cost_norm[cost_grid == 255] = 0.0

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

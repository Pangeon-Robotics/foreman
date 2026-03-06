"""Lazy Theta* pathfinding on 2D world cost grids.

Any-angle variant of A* that produces shorter, more direct paths by
allowing line-of-sight shortcuts during the search itself. "Lazy"
defers the LOS check to expansion time rather than insertion time,
reducing the number of expensive raycasts.

Cost grid conventions:
  0=free, 1-253=gradient, 254=lethal, 255=unknown.

Unknown cells (255) are treated as passable so the planner can route
through unexplored space to always reach the target.
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
    """Lazy Theta* on unified 2D world cost grid.

    Drop-in replacement for A*.  Returns same types:
      - return_path=False: path length in meters (or None)
      - return_path=True: list of world-frame (x,y) waypoints (or None)

    force_passable: when True, ALL cells are traversable (for viz).
    cost_weight: override obstacle proximity penalty (default 1.5).
    """
    return _lazy_theta_star(
        cost_grid, cost_origin_x, cost_origin_y, cost_voxel_size,
        robot_radius, start, goal, return_path, force_passable,
        cost_weight, cost_truncation)


# Keep old name as alias for any callers that use it directly.
_astar_on_cost_grid = _astar_core


def _lazy_theta_star(
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
    """Lazy Theta* using the unified world-frame cost grid.

    Any-angle pathfinding: nodes can have non-neighbor parents when
    line-of-sight exists, producing shorter paths than grid-constrained
    A*. The "lazy" variant defers LOS checks to expansion time.
    """
    nx, ny = cost_grid.shape
    _COST_WEIGHT = cost_weight if cost_weight is not None else 1.5

    # ── Pad grid to cover both start and goal ──────────────────────
    margin = 1.0
    req_x_lo = min(start[0], goal[0]) - margin
    req_x_hi = max(start[0], goal[0]) + margin
    req_y_lo = min(start[1], goal[1]) - margin
    req_y_hi = max(start[1], goal[1]) + margin
    grid_x_hi = ox + nx * vs
    grid_y_hi = oy + ny * vs
    if (req_x_lo < ox or req_x_hi > grid_x_hi
            or req_y_lo < oy or req_y_hi > grid_y_hi):
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

    # ── Grid coordinates ──────────────────────────────────────────
    sx = max(0, min(nx - 1, int((start[0] - ox) / vs)))
    sy = max(0, min(ny - 1, int((start[1] - oy) / vs)))
    gx = max(0, min(nx - 1, int((goal[0] - ox) / vs)))
    gy = max(0, min(ny - 1, int((goal[1] - oy) / vs)))

    # ── Passability ───────────────────────────────────────────────
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

    # ── Cost normalization ────────────────────────────────────────
    cost_norm = cost_grid.astype(np.float64) / 254.0
    cost_norm[cost_grid == 255] = 0.0

    # ── Line-of-sight check ───────────────────────────────────────
    def _los(x0, y0, x1, y1):
        """Bresenham-style LOS check on the passability grid.

        Returns True if a straight line from (x0,y0) to (x1,y1) crosses
        only passable cells. Also accumulates cost along the line for
        the cost-weighted g-score.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx_ = 1 if x1 > x0 else -1
        sy_ = 1 if y1 > y0 else -1
        cx, cy = x0, y0
        acc_cost = 0.0
        n_steps = 0

        if dx >= dy:
            # X-major
            err = dx // 2
            for _ in range(dx + 1):
                if not (0 <= cx < nx and 0 <= cy < ny and passable[cx, cy]):
                    return False, 0.0
                acc_cost += cost_norm[cx, cy]
                n_steps += 1
                err -= dy
                if err < 0:
                    cy += sy_
                    err += dx
                cx += sx_
        else:
            # Y-major
            err = dy // 2
            for _ in range(dy + 1):
                if not (0 <= cx < nx and 0 <= cy < ny and passable[cx, cy]):
                    return False, 0.0
                acc_cost += cost_norm[cx, cy]
                n_steps += 1
                err -= dx
                if err < 0:
                    cx += sx_
                    err += dy
                cy += sy_

        return True, acc_cost

    # ── Lazy Theta* search ────────────────────────────────────────
    SQRT2 = math.sqrt(2.0)
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    ]

    def h(x, y):
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

    g_score = np.full((nx, ny), np.inf, dtype=np.float64)
    g_score[sx, sy] = 0.0
    visited = np.zeros((nx, ny), dtype=np.bool_)
    parent = {}
    parent[(sx, sy)] = (sx, sy)

    open_set = [(h(sx, sy), sx, sy)]

    best_h = h(sx, sy)
    best_cell = (sx, sy)

    while open_set:
        _, cx, cy = heapq.heappop(open_set)

        if visited[cx, cy]:
            continue

        # ── Lazy LOS check at expansion time ──────────────────
        # The node was optimistically given its grandparent.
        # Now verify LOS; if it fails, fall back to best grid neighbor.
        if (cx, cy) != (sx, sy):
            px, py = parent[(cx, cy)]
            if (px, py) != (cx, cy):
                los_ok, _ = _los(px, py, cx, cy)
                if not los_ok:
                    # LOS failed — recompute g from grid neighbors
                    best_g = np.inf
                    best_par = parent[(cx, cy)]
                    for dx, dy, step_dist in neighbors:
                        nx2 = cx + dx
                        ny2 = cy + dy
                        if (0 <= nx2 < nx and 0 <= ny2 < ny
                                and visited[nx2, ny2]):
                            cand_g = (g_score[nx2, ny2] + step_dist
                                      + _COST_WEIGHT * cost_norm[cx, cy])
                            if cand_g < best_g:
                                best_g = cand_g
                                best_par = (nx2, ny2)
                    parent[(cx, cy)] = best_par
                    g_score[cx, cy] = best_g

        # ── Goal reached ──────────────────────────────────────
        if cx == gx and cy == gy:
            if not return_path:
                return float(g_score[gx, gy]) * vs
            path = _reconstruct(parent, sx, sy, gx, gy, ox, oy, vs)
            return path

        visited[cx, cy] = True

        ch = h(cx, cy)
        if ch < best_h:
            best_h = ch
            best_cell = (cx, cy)

        # ── Expand neighbors (lazy: optimistically set grandparent) ─
        px, py = parent[(cx, cy)]
        for dx, dy, step_dist in neighbors:
            nx2 = cx + dx
            ny2 = cy + dy
            if not (0 <= nx2 < nx and 0 <= ny2 < ny):
                continue
            if visited[nx2, ny2] or not passable[nx2, ny2]:
                continue
            # Corner-cutting check for diagonal moves
            if dx != 0 and dy != 0:
                if not passable[cx + dx, cy] or not passable[cx, cy + dy]:
                    continue

            # Path 1: through grandparent (any-angle)
            gp_dx = nx2 - px
            gp_dy = ny2 - py
            gp_dist = math.sqrt(gp_dx * gp_dx + gp_dy * gp_dy)
            # Approximate cost along LOS as endpoint cost * distance
            gp_g = (g_score[px, py] + gp_dist
                    + _COST_WEIGHT * cost_norm[nx2, ny2] * gp_dist
                    / max(step_dist, 0.01))

            # Path 2: through current node (grid-constrained)
            grid_g = (g_score[cx, cy] + step_dist
                      + _COST_WEIGHT * cost_norm[nx2, ny2])

            if gp_g <= grid_g + 1e-10:
                new_g = gp_g
                new_parent = (px, py)
            else:
                new_g = grid_g
                new_parent = (cx, cy)

            if new_g < g_score[nx2, ny2]:
                g_score[nx2, ny2] = new_g
                parent[(nx2, ny2)] = new_parent
                heapq.heappush(open_set, (new_g + h(nx2, ny2), nx2, ny2))

    # ── Partial path (couldn't reach goal) ────────────────────────
    if return_path and best_cell != (sx, sy):
        return _reconstruct(parent, sx, sy, best_cell[0], best_cell[1],
                            ox, oy, vs)

    return None


def _reconstruct(parent, sx, sy, ex, ey, ox, oy, vs):
    """Reconstruct path from parent dict, converting grid→world coords."""
    path = []
    px, py = ex, ey
    while (px, py) != (sx, sy):
        path.append((ox + (px + 0.5) * vs, oy + (py + 0.5) * vs))
        px, py = parent[(px, py)]
    path.append((ox + (sx + 0.5) * vs, oy + (sy + 0.5) * vs))
    path.reverse()
    return path

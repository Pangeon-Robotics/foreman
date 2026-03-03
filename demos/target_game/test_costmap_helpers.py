"""God-view costmap builders and route comparison utilities.

Extracted from test_costmap_compare.py to keep files under 400 lines.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _rasterize_obstacles(
    scene_xml_path: str,
    z_lo: float, z_hi: float,
    output_resolution: float, xy_extent: float,
) -> tuple[np.ndarray, dict]:
    """Rasterize obstacle XY footprints into a bool grid.

    Returns (occupied_2d, meta) where occupied_2d is (gs, gs) bool.
    """
    from .scene_parser import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    origin = -xy_extent
    gs = int(round(2 * xy_extent / output_resolution))
    res = output_resolution
    occupied = np.zeros((gs, gs), dtype=bool)

    for obs in obstacles:
        px, py, pz = obs['pos']
        otype, size = obs['type'], obs['size']

        if otype == 'box':
            hx, hy, hz = size
            if pz + hz < z_lo or pz - hz > z_hi:
                continue
            ix_lo = max(0, int(math.floor((px - hx - origin) / res)))
            ix_hi = min(gs, int(math.ceil((px + hx - origin) / res)))
            iy_lo = max(0, int(math.floor((py - hy - origin) / res)))
            iy_hi = min(gs, int(math.ceil((py + hy - origin) / res)))
            occupied[ix_lo:ix_hi, iy_lo:iy_hi] = True

        elif otype == 'cylinder':
            radius, half_h = size[0], size[1]
            if pz + half_h < z_lo or pz - half_h > z_hi:
                continue
            ix_lo = max(0, int(math.floor((px - radius - origin) / res)))
            ix_hi = min(gs, int(math.ceil((px + radius - origin) / res)))
            iy_lo = max(0, int(math.floor((py - radius - origin) / res)))
            iy_hi = min(gs, int(math.ceil((py + radius - origin) / res)))
            for ix in range(ix_lo, ix_hi):
                wx = origin + (ix + 0.5) * res
                for iy in range(iy_lo, iy_hi):
                    wy = origin + (iy + 0.5) * res
                    if (wx - px)**2 + (wy - py)**2 <= radius**2:
                        occupied[ix, iy] = True

    meta = {'origin_x': origin, 'origin_y': origin, 'voxel_size': res,
            'nx': gs, 'ny': gs}
    return occupied, meta


def build_god_view_binary(
    scene_xml_path: str,
    z_lo: float = 0.05, z_hi: float = 0.55,
    output_resolution: float = 0.05,
    xy_extent: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """Build binary god-view grid: 254 for occupied cells, 0 for free."""
    occupied, meta = _rasterize_obstacles(
        scene_xml_path, z_lo, z_hi, output_resolution, xy_extent)
    grid = np.zeros(occupied.shape, dtype=np.uint8)
    grid[occupied] = 254
    return grid, meta


def build_god_view_costmap(
    scene_xml_path: str,
    z_lo: float = 0.05, z_hi: float = 0.55,
    output_resolution: float = 0.05,
    xy_extent: float = 10.0, truncation: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Build god-view costmap with EDT gradient (for comparison metrics)."""
    from scipy.ndimage import distance_transform_edt

    occupied, meta = _rasterize_obstacles(
        scene_xml_path, z_lo, z_hi, output_resolution, xy_extent)
    dist_m = distance_transform_edt(~occupied) * output_resolution
    cost_u8 = ((1.0 - np.clip(dist_m / truncation, 0, 1)) * 254).astype(np.uint8)
    meta['truncation'] = truncation
    return cost_u8, meta


def _astar_path(
    cost_grid: np.ndarray, start_xy: tuple[float, float],
    goal_xy: tuple[float, float], voxel_size: float,
    origin_x: float, origin_y: float,
    robot_radius: float = 0.35, truncation: float = 0.5,
) -> list[tuple[int, int]] | None:
    """Standalone A* on uint8 cost grid. Returns list of (ix,iy) or None."""
    import heapq

    vs, ox, oy = voxel_size, origin_x, origin_y
    nx, ny = cost_grid.shape
    SQRT2 = math.sqrt(2.0)
    _COST_WEIGHT = 1.5

    sx = max(0, min(nx - 1, int((start_xy[0] - ox) / vs)))
    sy = max(0, min(ny - 1, int((start_xy[1] - oy) / vs)))
    gx = max(0, min(nx - 1, int((goal_xy[0] - ox) / vs)))
    gy = max(0, min(ny - 1, int((goal_xy[1] - oy) / vs)))

    radius_ratio = min(robot_radius / truncation, 0.95)
    cost_threshold = int((1.0 - radius_ratio) * 254)
    passable = (cost_grid < cost_threshold) | (cost_grid == 255)
    passable[sx, sy] = True
    passable[gx, gy] = True

    _UNKNOWN_COST = 0.5
    cost_norm = cost_grid.astype(np.float64) / 254.0
    cost_norm[cost_grid == 255] = _UNKNOWN_COST
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

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if cx == gx and cy == gy:
            path = []
            px, py = gx, gy
            while (px, py) != (sx, sy):
                path.append((px, py))
                px, py = parent[(px, py)]
            path.append((sx, sy))
            path.reverse()
            return path
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True
        for dx, dy, step_dist in neighbors:
            nx2, ny2 = cx + dx, cy + dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny and not visited[nx2, ny2]:
                if not passable[nx2, ny2]:
                    continue
                if dx != 0 and dy != 0:
                    if not passable[cx + dx, cy] or not passable[cx, cy + dy]:
                        continue
                new_g = g_score[cx, cy] + step_dist + _COST_WEIGHT * cost_norm[nx2, ny2]
                if new_g < g_score[nx2, ny2]:
                    g_score[nx2, ny2] = new_g
                    parent[(nx2, ny2)] = (cx, cy)
                    heapq.heappush(open_set, (new_g + h(nx2, ny2), nx2, ny2))
    return None


def compute_route_score(
    god_grid: np.ndarray, god_meta: dict,
    robot_grid: np.ndarray, robot_meta: dict,
    robot_xy: tuple[float, float], target_xy: tuple[float, float],
    robot_radius: float = 0.35,
) -> dict:
    """Compare A* paths on god-view vs robot-view costmaps."""
    trunc_g = god_meta.get('truncation', 0.5)
    trunc_r = robot_meta.get('truncation', 0.5)
    vs = god_meta['voxel_size']

    god_path = _astar_path(
        god_grid, robot_xy, target_xy, vs,
        god_meta['origin_x'], god_meta['origin_y'], robot_radius, trunc_g)
    robot_path = _astar_path(
        robot_grid, robot_xy, target_xy, robot_meta['voxel_size'],
        robot_meta['origin_x'], robot_meta['origin_y'], robot_radius, trunc_r)

    def path_length_m(path, voxel_sz):
        d = 0.0
        for a, b in zip(path, path[1:]):
            d += math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) * voxel_sz
        return d

    fail = {'rs_length_ratio': 0.0, 'rs_hausdorff': float('inf'),
            'rs_overlap': 0.0, 'rs_score': 0.0,
            'god_path_len': 0, 'robot_path_len': 0,
            'god_path_m': 0.0, 'robot_path_m': 0.0}
    if god_path is None and robot_path is None:
        return {**fail, 'rs_score': 50.0}
    if god_path is None:
        return {**fail, 'rs_score': 15.0, 'robot_path_len': len(robot_path)}
    if robot_path is None:
        return {**fail, 'rs_score': 20.0, 'god_path_len': len(god_path),
                'god_path_m': path_length_m(god_path, vs)}

    god_len = path_length_m(god_path, vs)
    robot_len = path_length_m(robot_path, robot_meta['voxel_size'])

    rs_length_ratio = god_len / max(robot_len, 0.01) if god_len > 0 else 1.0
    rs_length_ratio = min(rs_length_ratio, 1.0)

    def to_world(path, meta):
        o_x, o_y, v = meta['origin_x'], meta['origin_y'], meta['voxel_size']
        return [(o_x + (ix + 0.5) * v, o_y + (iy + 0.5) * v) for ix, iy in path]

    gw = to_world(god_path, god_meta)
    rw = to_world(robot_path, robot_meta)

    def directed_hausdorff(a, b):
        worst = 0.0
        for ax, ay in a:
            best = float('inf')
            for bx, by in b:
                d = math.sqrt((ax - bx)**2 + (ay - by)**2)
                if d < best:
                    best = d
            if best > worst:
                worst = best
        return worst

    h_ab = directed_hausdorff(gw, rw)
    h_ba = directed_hausdorff(rw, gw)
    rs_hausdorff = max(h_ab, h_ba)

    tol = 2 * vs
    match = 0
    for rx, ry in rw:
        for gx, gy in gw:
            if math.sqrt((rx - gx)**2 + (ry - gy)**2) <= tol:
                match += 1
                break
    rs_overlap = match / len(rw)

    len_score = rs_length_ratio
    haus_score = max(0.0, 1.0 - rs_hausdorff / 2.0)
    rs_score = (0.4 * len_score + 0.3 * rs_overlap + 0.3 * haus_score) * 100.0

    return {'rs_length_ratio': round(rs_length_ratio, 3),
            'rs_hausdorff': round(rs_hausdorff, 3),
            'rs_overlap': round(rs_overlap, 3),
            'rs_score': round(rs_score, 1),
            'god_path_len': len(god_path), 'robot_path_len': len(robot_path),
            'god_path_m': round(god_len, 2), 'robot_path_m': round(robot_len, 2)}

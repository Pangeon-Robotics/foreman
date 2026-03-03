"""Perception F1 scores: Surface, Cost, Router.

All three compare robot-view TSDF against god-view TSDF, producing
simple P/R/F1 on a 0-100 scale. Same ground truth source (god TSDF).
"""
from __future__ import annotations

import numpy as np


def compute_surface_f1(
    god_tsdf, robot_tsdf, resolution: float = 0.05,
) -> dict:
    """Voxel-set F1 between god and robot surface voxels.

    Snaps both voxel clouds to a common grid and computes set P/R/F1.
    """
    god_vox = god_tsdf.get_surface_voxels(include_history=True)
    robot_vox = robot_tsdf.get_surface_voxels(include_history=True)

    def _to_set(voxels):
        if len(voxels) == 0:
            return set()
        keys = np.floor(voxels / resolution).astype(np.int32)
        return set(map(tuple, keys))

    god_set = _to_set(god_vox)
    robot_set = _to_set(robot_vox)

    n_god = len(god_set)
    n_robot = len(robot_set)
    inter = len(god_set & robot_set)

    P = (inter / n_robot * 100.0) if n_robot > 0 else 0.0
    R = (inter / n_god * 100.0) if n_god > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    return {
        "f1": round(F1, 1),
        "precision": round(P, 1),
        "recall": round(R, 1),
        "n_god": n_god,
        "n_robot": n_robot,
    }


def compute_cost_f1(
    god_tsdf, robot_tsdf,
    z_lo: float, z_hi: float, resolution: float = 0.05,
) -> dict:
    """Binary grid F1 between god and robot cost grids.

    Only scores cells observed by both (neither == 255).
    Occupied threshold: cost >= 200 (near-surface, not soft gradient).
    """
    god_grid, god_meta = god_tsdf.get_world_cost_grid(z_lo, z_hi, resolution)
    robot_grid, robot_meta = robot_tsdf.get_world_cost_grid(z_lo, z_hi, resolution)

    # Align to smaller shape
    nx = min(god_grid.shape[0], robot_grid.shape[0])
    ny = min(god_grid.shape[1], robot_grid.shape[1])
    god_grid = god_grid[:nx, :ny]
    robot_grid = robot_grid[:nx, :ny]

    valid = (god_grid != 255) & (robot_grid != 255)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    g_occ = god_grid[valid] >= 200
    r_occ = robot_grid[valid] >= 200

    tp = int(np.sum(g_occ & r_occ))
    fp = int(np.sum(~g_occ & r_occ))
    fn = int(np.sum(g_occ & ~r_occ))

    P = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    R = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    return {
        "f1": round(F1, 1),
        "precision": round(P, 1),
        "recall": round(R, 1),
    }


def compute_router_f1(
    god_tsdf, robot_tsdf,
    start_xy: tuple[float, float], goal_xy: tuple[float, float],
    z_lo: float, z_hi: float,
    resolution: float = 0.05, tolerance: int = 2,
) -> dict:
    """A* path F1 between god and robot cost grids.

    Runs A* on both grids, then tolerance-matched P/R (Chebyshev
    distance within `tolerance` grid cells).
    """
    from .test_costmap_compare import _astar_path

    god_grid, god_meta = god_tsdf.get_world_cost_grid(z_lo, z_hi, resolution)
    robot_grid, robot_meta = robot_tsdf.get_world_cost_grid(z_lo, z_hi, resolution)

    # Align
    nx = min(god_grid.shape[0], robot_grid.shape[0])
    ny = min(god_grid.shape[1], robot_grid.shape[1])
    god_grid = god_grid[:nx, :ny]
    robot_grid = robot_grid[:nx, :ny]

    # Use 1.5× robot half-width so A* avoids corridors narrower than
    # 1.05m (1.5 × 0.70m robot diameter).
    _ROUTER_RADIUS = 0.525

    god_path = _astar_path(
        god_grid, start_xy, goal_xy,
        god_meta['voxel_size'], god_meta['origin_x'], god_meta['origin_y'],
        robot_radius=_ROUTER_RADIUS)
    robot_path = _astar_path(
        robot_grid, start_xy, goal_xy,
        robot_meta['voxel_size'], robot_meta['origin_x'], robot_meta['origin_y'],
        robot_radius=_ROUTER_RADIUS)

    def _path_length_m(path, vs):
        d = 0.0
        for a, b in zip(path, path[1:]):
            d += ((b[0] - a[0])**2 + (b[1] - a[1])**2) ** 0.5 * vs
        return d

    # Both fail → agree on impassable
    if god_path is None and robot_path is None:
        return {"f1": 100.0, "precision": 100.0, "recall": 100.0,
                "god_path_m": 0.0, "robot_path_m": 0.0}
    # One fails → total mismatch
    if god_path is None or robot_path is None:
        god_m = _path_length_m(god_path, god_meta['voxel_size']) if god_path else 0.0
        robot_m = _path_length_m(robot_path, robot_meta['voxel_size']) if robot_path else 0.0
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "god_path_m": round(god_m, 2), "robot_path_m": round(robot_m, 2)}

    god_set = set(god_path)
    robot_set = set(robot_path)

    def _matched_fraction(source, reference, tol):
        """Fraction of source cells within Chebyshev `tol` of any reference cell."""
        if not source:
            return 0.0
        matched = 0
        for sx, sy in source:
            for rx, ry in reference:
                if abs(sx - rx) <= tol and abs(sy - ry) <= tol:
                    matched += 1
                    break
        return matched / len(source)

    P = _matched_fraction(robot_path, god_path, tolerance) * 100.0
    R = _matched_fraction(god_path, robot_path, tolerance) * 100.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    god_m = _path_length_m(god_path, god_meta['voxel_size'])
    robot_m = _path_length_m(robot_path, robot_meta['voxel_size'])

    return {
        "f1": round(F1, 1),
        "precision": round(P, 1),
        "recall": round(R, 1),
        "god_path_m": round(god_m, 2),
        "robot_path_m": round(robot_m, 2),
    }

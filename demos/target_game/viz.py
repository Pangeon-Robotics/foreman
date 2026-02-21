"""Visualization helpers for the target game.

Renders SLAM trails, costmap overlays, and DWA arcs using mjv_initGeom
on the MuJoCo viewer's user_scn. All functions are no-ops if the viewer
is not available (headless mode).

Usage in the physics render loop:
    viz.draw_slam_trail(viewer, truth_trail, slam_trail)
    viz.draw_costmap_voxels(viewer, costmap, max_geoms=200)
    viz.draw_dwa_arcs(viewer, arcs, scores)
"""
from __future__ import annotations

import math


def draw_slam_trail(viewer, truth_trail, slam_trail, max_points: int = 200) -> int:
    """Render ground truth (green) and SLAM estimate (yellow) trails.

    Parameters
    ----------
    viewer : mujoco.viewer handle
        Must have user_scn with ngeom counter.
    truth_trail : list of (x, y)
        Ground truth positions.
    slam_trail : list of (x, y)
        SLAM estimated positions.
    max_points : int
        Max trail points to render (most recent).

    Returns
    -------
    int
        Number of geoms added.
    """
    import mujoco

    added = 0
    scn = viewer.user_scn

    # Subsample if too many points
    truth = truth_trail[-max_points:]
    slam = slam_trail[-max_points:]

    # Ground truth trail: green spheres
    for x, y in truth:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.02, 0, 0],     # size (radius)
            [x, y, 0.02],     # pos
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # identity mat
            [0.2, 0.8, 0.2, 0.6],  # green, semi-transparent
        )
        scn.ngeom += 1
        added += 1

    # SLAM trail: yellow spheres
    for x, y in slam:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.02, 0, 0],
            [x, y, 0.04],     # slightly above ground truth
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.9, 0.9, 0.1, 0.6],  # yellow
        )
        scn.ngeom += 1
        added += 1

    return added


def draw_costmap_voxels(viewer, costmap, threshold: float = 0.5, max_geoms: int = 200) -> int:
    """Render high-cost costmap cells as colored boxes.

    Parameters
    ----------
    viewer : mujoco.viewer handle
    costmap : Costmap2D
        2D costmap with grid, resolution, origin_x, origin_y.
    threshold : float
        Only render cells with cost >= threshold.
    max_geoms : int
        Max voxels to render.

    Returns
    -------
    int
        Number of geoms added.
    """
    import mujoco
    import numpy as np

    if costmap is None:
        return 0

    scn = viewer.user_scn
    grid = costmap.grid
    res = costmap.resolution
    half = res / 2

    # Find cells above threshold
    iy, ix = np.where(grid >= threshold)
    if len(ix) == 0:
        return 0

    # Subsample if too many
    if len(ix) > max_geoms:
        indices = np.linspace(0, len(ix) - 1, max_geoms, dtype=int)
        ix = ix[indices]
        iy = iy[indices]

    # Grid is in body frame — rotate cells to world frame for rendering.
    # origin_x/y = robot world position, robot_yaw = heading at scan time.
    half_ext = grid.shape[0] * res / 2.0
    c_yaw = math.cos(costmap.robot_yaw)
    s_yaw = math.sin(costmap.robot_yaw)

    added = 0
    for i, j in zip(ix, iy):
        if scn.ngeom >= scn.maxgeom:
            break
        cost = float(grid[i, j])
        # Cell center in body frame (robot at origin, +X = forward)
        bx = j * res + half - half_ext
        by = i * res + half - half_ext
        # Rotate to world frame
        wx = costmap.origin_x + c_yaw * bx - s_yaw * by
        wy = costmap.origin_y + s_yaw * bx + c_yaw * by

        # Color: red intensity proportional to cost
        r = min(1.0, cost)
        g = max(0.0, 0.3 * (1.0 - cost))
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            [half, half, 0.05],   # size
            [wx, wy, 0.05],      # pos (on ground)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [r, g, 0.0, 0.4],    # rgba
        )
        scn.ngeom += 1
        added += 1

    return added


def draw_dwa_arcs(viewer, arc_points, scores, top_n: int = 3) -> int:
    """Render top-scoring DWA arcs as colored line strips.

    Parameters
    ----------
    viewer : mujoco.viewer handle
    arc_points : ndarray (n_arcs, n_steps, 2)
        All arc waypoints in robot-centered frame.
    scores : ndarray (n_arcs,)
        Arc scores (higher = better).
    top_n : int
        How many top arcs to render.

    Returns
    -------
    int
        Number of geoms added.
    """
    import mujoco
    import numpy as np

    if arc_points is None or scores is None:
        return 0

    scn = viewer.user_scn
    added = 0

    # Get top-N arcs by score
    valid = scores > -np.inf
    if not np.any(valid):
        return 0

    top_indices = np.argsort(scores[valid])[-top_n:][::-1]
    valid_indices = np.where(valid)[0]

    colors = [
        [0.0, 1.0, 0.0, 0.8],  # best: green
        [1.0, 1.0, 0.0, 0.6],  # 2nd: yellow
        [1.0, 0.5, 0.0, 0.4],  # 3rd: orange
    ]

    for rank, idx in enumerate(top_indices):
        arc_idx = valid_indices[idx]
        arc = arc_points[arc_idx]  # (n_steps, 2)
        color = colors[min(rank, len(colors) - 1)]

        # Render arc as series of small spheres
        for step in range(len(arc)):
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.015, 0, 0],
                [float(arc[step, 0]), float(arc[step, 1]), 0.1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                color,
            )
            scn.ngeom += 1
            added += 1

    return added

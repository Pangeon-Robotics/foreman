"""God-view A* path planning.

GOD-VIEW ONLY: uses god-view costmap (from scene XML or GodViewTSDF)
and ground-truth robot position. Renders as blue dots. Not used by
the robot's perception/navigation pipeline.
"""
from __future__ import annotations


def astar_on_god_view(
    grid, meta, start: tuple[float, float], goal: tuple[float, float],
    robot_radius: float = 0.525,
) -> list[tuple[float, float]] | None:
    """Run A* on a god-view cost grid and return world-frame waypoints.

    robot_radius defaults to 0.525m = 1.5 x B2 half-width, so A* avoids
    corridors narrower than 1.05m (1.5 x robot diameter).
    """
    from .path_critic import PathCritic
    critic = PathCritic.__new__(PathCritic)
    critic._cost_grid = grid
    critic._cost_origin_x = meta['origin_x']
    critic._cost_origin_y = meta['origin_y']
    critic._cost_voxel_size = meta['voxel_size']
    critic._cost_truncation = meta.get('truncation', 0.5)
    critic._robot_radius = robot_radius
    path = critic._astar_on_cost_grid(start, goal, return_path=True,
                                      force_passable=True)
    if path is not None and len(path) >= 3:
        path = PathCritic.smooth_path(
            path, grid, meta['origin_x'], meta['origin_y'],
            meta['voxel_size'])
    return path

"""Cost grid builder for the perception pipeline.

Builds A* cost grid from TSDF surface voxels (including history).
Uses the same data source as the grey cube visualization, so red
areas in the viewer perfectly match visible TSDF cubes.
"""
from __future__ import annotations

import math
import time

import numpy as np


# Robot clearance radius for A* exclusion zones.
# B2 half-width ≈ 0.28m.  0.25m gives ~0.30m total from obstacle
# surface (TSDF voxels are already at the surface).
_CLEARANCE_M = 0.25


def build_cost_grids(pipeline, tsdf, imu_x: float,
                     imu_y: float) -> None:
    """Build cost grid from TSDF surface voxels (including history).

    Uses get_surface_voxels(include_history=True) — the same data that
    drives the grey cube visualization.  This guarantees the costmap
    (red overlay) appears everywhere a TSDF cube is visible.

    Binary grid: 0=free, 254=blocked.  Circular dilation at _CLEARANCE_M
    provides robot-width clearance from obstacle surfaces.

    Parameters
    ----------
    pipeline : PerceptionPipeline-like
        Attributes are read/written.
    tsdf : TSDF
        The TSDF instance.
    imu_x, imu_y : float
        Robot position in world frame.
    """
    vs = getattr(pipeline._cfg, 'tsdf_output_resolution', 0.05)
    truncation = pipeline._cfg.tsdf_truncation

    t0 = time.monotonic()

    # Surface voxels with history = same data as grey cube viz.
    voxels = tsdf.get_surface_voxels(include_history=True)

    # Z-band filter: only obstacles in relevant height range
    if len(voxels) > 0:
        z_lo = getattr(tsdf, 'costmap_z_lo', 0.05)
        z_hi = getattr(tsdf, 'costmap_z_hi', 0.80)
        z_mask = (voxels[:, 2] >= z_lo) & (voxels[:, 2] <= z_hi)
        voxels = voxels[z_mask]

    if len(voxels) == 0:
        pipeline._world_cost_grid = None
        pipeline._world_cost_meta = None
        pipeline._dwa_cost_grid = None
        pipeline._last_cg_ms = 0
        pipeline._last_astar_ms = 0
        return

    # Grid bounds covering all voxels + margin for dilation
    margin_m = _CLEARANCE_M + vs * 2
    wx_min = float(voxels[:, 0].min()) - margin_m
    wy_min = float(voxels[:, 1].min()) - margin_m
    wx_max = float(voxels[:, 0].max()) + margin_m
    wy_max = float(voxels[:, 1].max()) + margin_m

    nx = int(math.ceil((wx_max - wx_min) / vs))
    ny = int(math.ceil((wy_max - wy_min) / vs))

    # Project voxels to 2D grid
    gx = ((voxels[:, 0] - wx_min) / vs).astype(np.int32)
    gy = ((voxels[:, 1] - wy_min) / vs).astype(np.int32)
    occupied = np.zeros((nx, ny), dtype=np.bool_)
    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    occupied[gx[valid], gy[valid]] = True

    # Circular dilation for robot clearance
    from scipy.ndimage import binary_dilation
    r_cells = max(1, int(round(_CLEARANCE_M / vs)))
    y_k, x_k = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    disk = (x_k * x_k + y_k * y_k) <= r_cells * r_cells
    inflated = binary_dilation(occupied, structure=disk)
    astar_u8 = np.where(inflated, np.uint8(254), np.uint8(0))

    # Robot footprint clearing (0.35m)
    cx = int((imu_x - wx_min) / vs)
    cy = int((imu_y - wy_min) / vs)
    fp_r = int(0.35 / vs)
    ax_lo = max(0, cx - fp_r)
    ax_hi = min(nx, cx + fp_r + 1)
    ay_lo = max(0, cy - fp_r)
    ay_hi = min(ny, cy + fp_r + 1)
    if ax_lo < ax_hi and ay_lo < ay_hi:
        aix, aiy = np.ogrid[ax_lo:ax_hi, ay_lo:ay_hi]
        fp_mask = (aix - cx) ** 2 + (aiy - cy) ** 2 <= fp_r ** 2
        astar_u8[ax_lo:ax_hi, ay_lo:ay_hi][fp_mask] = 0

    t1 = time.monotonic()

    meta = {
        'origin_x': wx_min,
        'origin_y': wy_min,
        'voxel_size': vs,
        'nx': nx, 'ny': ny,
        'truncation': truncation,
    }
    pipeline._world_cost_grid = astar_u8
    pipeline._world_cost_meta = meta
    # LiveCostmapQuery reads _dwa_cost_grid — point it at the same grid.
    pipeline._dwa_cost_grid = astar_u8
    pipeline._last_cg_ms = (t1 - t0) * 1000
    pipeline._last_astar_ms = pipeline._last_cg_ms

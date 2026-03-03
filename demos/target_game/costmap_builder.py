"""Cost grid builder for the perception pipeline.

Builds DWA and A* cost grids from TSDF data. Extracted from
PerceptionPipeline._build_cost_grids() to keep perception.py under 400 lines.
"""
from __future__ import annotations

import time

import numpy as np


def build_cost_grids(pipeline, tsdf, imu_x: float,
                     imu_y: float) -> None:
    """Build cost grids from TSDF.

    TSDF has built-in persistence (log-odds accumulation) so the
    obstacle memory layer is NOT used -- it amplifies noise into phantom
    obstacles that oscillate DWA between feas=41 and feas=0.

    The TSDF cost grid (EDT-based) feeds both DWA and A* directly.

    Parameters
    ----------
    pipeline : PerceptionPipeline
        The perception pipeline instance (attributes are read/written).
    tsdf : TSDF
        The TSDF instance to extract cost grids from.
    imu_x, imu_y : float
        Robot position in world frame.
    """
    out_res = getattr(pipeline._cfg, 'tsdf_output_resolution', 0.05)
    truncation = pipeline._cfg.tsdf_truncation

    # Get cost grid from TSDF (handles projection + EDT)
    _t_cg0 = time.monotonic()
    cost_u8, meta = tsdf.get_world_cost_grid(
        tsdf.costmap_z_lo, tsdf.costmap_z_hi, out_res)
    _t_cg1 = time.monotonic()

    out_nx = meta['nx']
    out_ny = meta['ny']
    observed = meta.get('observed_mask')

    # Robot position in output grid coordinates
    vs = out_res
    cx = int((imu_x - tsdf.origin_x) / vs)
    cy = int((imu_y - tsdf.origin_y) / vs)

    # --- DWA cost grid: direct from TSDF ---
    # cost_u8 already has EDT-based distance falloff.
    # No memory overlay needed -- TSDF persistence handles it.
    dwa_u8 = cost_u8.copy()
    if observed is not None:
        dwa_u8[~observed] = 255

    # DWA robot footprint clearing (0.4m)
    dwa_fp_r = int(0.4 / vs)
    dx_lo = max(0, cx - dwa_fp_r)
    dx_hi = min(out_nx, cx + dwa_fp_r + 1)
    dy_lo = max(0, cy - dwa_fp_r)
    dy_hi = min(out_ny, cy + dwa_fp_r + 1)
    if dx_lo < dx_hi and dy_lo < dy_hi:
        dix, diy = np.ogrid[dx_lo:dx_hi, dy_lo:dy_hi]
        dwa_fp = (dix - cx)**2 + (diy - cy)**2 <= dwa_fp_r**2
        dwa_u8[dx_lo:dx_hi, dy_lo:dy_hi][dwa_fp] = 0
    pipeline._dwa_cost_grid = dwa_u8

    # Yield GIL before A* grid build
    time.sleep(0.005)

    # --- A* cost grid: TSDF + inflation ---
    # A* inflation EDT (~65ms) is rate-limited to every 5th build.
    # A* paths change slowly (waypoint update rate), so stale-by-4
    # builds (~1s at 5Hz scan) is acceptable.
    _t_a0 = time.monotonic()
    if not hasattr(pipeline, '_astar_build_counter'):
        pipeline._astar_build_counter = 0
    pipeline._astar_build_counter += 1
    astar_changed = (not hasattr(pipeline, '_astar_cache_id')
                     or pipeline._astar_build_counter % 5 == 0)
    if astar_changed:
        from scipy.ndimage import binary_dilation, distance_transform_edt
        occupied_seed = (cost_u8 > 200) & (cost_u8 != 255)
        if np.any(occupied_seed):
            inflated = binary_dilation(occupied_seed, iterations=6)
            dist_2d = distance_transform_edt(~inflated).astype(
                np.float32) * vs
        else:
            dist_2d = np.full(
                (out_nx, out_ny), out_nx * vs, dtype=np.float32)
        astar_cost_f = 1.0 - np.clip(dist_2d / truncation, 0.0, 1.0)
        astar_u8 = (astar_cost_f * 254).astype(np.uint8)
        if observed is not None:
            astar_u8[~observed] = 255
        pipeline._astar_base = astar_u8

    astar_u8 = pipeline._astar_base.copy()

    # A* robot footprint clearing (0.35m)
    astar_r = int(0.35 / vs)
    ax_lo = max(0, cx - astar_r)
    ax_hi = min(out_nx, cx + astar_r + 1)
    ay_lo = max(0, cy - astar_r)
    ay_hi = min(out_ny, cy + astar_r + 1)
    if ax_lo < ax_hi and ay_lo < ay_hi:
        aix, aiy = np.ogrid[ax_lo:ax_hi, ay_lo:ay_hi]
        astar_fp = (aix - cx)**2 + (aiy - cy)**2 <= astar_r**2
        astar_u8[ax_lo:ax_hi, ay_lo:ay_hi][astar_fp] = 0
    pipeline._world_cost_grid = astar_u8
    pipeline._last_cg_ms = (_t_cg1 - _t_cg0) * 1000
    pipeline._last_astar_ms = (time.monotonic() - _t_a0) * 1000

    pipeline._world_cost_meta = {
        'origin_x': tsdf.origin_x, 'origin_y': tsdf.origin_y,
        'voxel_size': vs,
        'nx': out_nx, 'ny': out_ny,
        'truncation': truncation,
    }


def clear_local_tsdf(pipeline) -> None:
    """Clear TSDF voxels near robot (contradicting evidence).

    The robot's presence at a location is evidence that the space
    is traversable.  Clears log_odds and obs_count in all TSDF
    chunks within 2.5m, then forces one feasible DWA arc.
    """
    tsdf = pipeline._tsdf
    with pipeline._imu_lock:
        rx, ry = pipeline._imu_x, pipeline._imu_y
    cs = 16 * tsdf.voxel_size  # chunk side in meters
    r2 = 2.5 * 2.5
    for key, chunk in list(tsdf._chunks.items()):
        cx, cy, cz = key
        chunk_wx = (cx + 0.5) * cs + tsdf.origin_x
        chunk_wy = (cy + 0.5) * cs + tsdf.origin_y
        if (chunk_wx - rx)**2 + (chunk_wy - ry)**2 <= r2:
            chunk.log_odds[:] = 0.0
            chunk.obs_count[:] = 0
            chunk.dirty = True
            tsdf._occ_chunk_dirty.add(key)
    tsdf._dirty = True
    pipeline._force_feasible = True

"""Visualization file writers and debug viewer streaming.

Writes robot-view TSDF surface voxels to temp file (from subprocess).
Game process reads TSDF voxels, builds costmap, writes costmap viz file.
Also handles debug viewer streaming and god-view A* path file writing.
"""
from __future__ import annotations

import math
import os
import struct

import numpy as np

ROBOT_TSDF_FILE = "/tmp/robot_view_tsdf.bin"
ROBOT_COSTMAP_FILE = "/tmp/robot_view_costmap.bin"

# Robot clearance radius for A* exclusion zones
_CLEARANCE_M = 0.25


def _write_costmap_viz(grid, meta):
    """Write A* cost grid as the viewer costmap file.

    Viewer expects 16-byte header: u16 rows, u16 cols, f32 ox, f32 oy, f32 vs.
    Grid in (rows=Y, cols=X) layout.
    """
    grid_t = grid.T  # (nx, ny) → (ny, nx) = (rows, cols)
    rows, cols = grid_t.shape
    hdr = struct.pack('<HHfff', rows, cols,
                      meta['origin_x'], meta['origin_y'],
                      meta['voxel_size'])
    tmp = ROBOT_COSTMAP_FILE + ".tmp"
    try:
        with open(tmp, 'wb') as f:
            f.write(hdr)
            f.write(np.ascontiguousarray(grid_t).tobytes())
        os.replace(tmp, ROBOT_COSTMAP_FILE)
    except OSError:
        pass


def write_tsdf_binary(tsdf, path: str, display_resolution: float = 0.05) -> None:
    """Write TSDF surface voxels to binary file for MuJoCo renderer.

    Format: u32 n_voxels + f32 voxel_half_size (8 bytes header)
            N x 3 float32 xyz (12 bytes per voxel)
    Downsamples to display_resolution for bounded voxel count.
    """
    voxels = tsdf.get_surface_voxels(include_history=True)
    if len(voxels) == 0:
        buf = bytearray(8)
        struct.pack_into('<If', buf, 0, 0, 0.0)
        try:
            with open(path, 'wb') as f:
                f.write(buf)
        except OSError:
            pass
        return

    keys = np.floor(voxels / display_resolution).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    voxels = (keys[idx] + 0.5) * display_resolution

    n = len(voxels)
    half = display_resolution / 2.0
    buf = bytearray(8 + n * 12)
    struct.pack_into('<If', buf, 0, n, half)
    buf[8:] = voxels.astype(np.float32).tobytes()

    try:
        with open(path, 'wb') as f:
            f.write(buf)
    except OSError:
        pass


def write_robot_tsdf_file(tsdf, display_resolution: float = 0.05) -> None:
    """Write robot-view TSDF surface voxels to temp file for MuJoCo viewer."""
    write_tsdf_binary(tsdf, ROBOT_TSDF_FILE, display_resolution)


def _read_tsdf_voxels(path=ROBOT_TSDF_FILE):
    """Read TSDF voxel file written by subprocess. Returns Nx3 float32 or None.

    Format: u32 n_voxels + f32 voxel_half_size (8 bytes header)
            N x 3 float32 xyz (12 bytes per voxel)
    """
    try:
        with open(path, 'rb') as f:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            n, _half = struct.unpack('<If', hdr)
            if n == 0:
                return None
            body = f.read(n * 12)
            if len(body) < n * 12:
                return None
    except (OSError, FileNotFoundError):
        return None
    return np.frombuffer(body, dtype=np.float32).reshape(n, 3).copy()


def _build_costmap(voxels, robot_x, robot_y, voxel_size=0.05, truncation=0.5):
    """Build A* cost grid from TSDF surface voxels.

    Binary grid: 0=free, 254=blocked. Circular dilation at _CLEARANCE_M
    for robot-width clearance. Robot footprint cleared.

    Returns (grid, meta) or None.
    """
    from scipy.ndimage import binary_dilation

    # Z-band filter
    z_mask = (voxels[:, 2] >= 0.05) & (voxels[:, 2] <= 0.80)
    voxels = voxels[z_mask]
    if len(voxels) == 0:
        return None

    vs = voxel_size

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
    r_cells = max(1, int(round(_CLEARANCE_M / vs)))
    y_k, x_k = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    disk = (x_k * x_k + y_k * y_k) <= r_cells * r_cells
    inflated = binary_dilation(occupied, structure=disk)
    grid = np.where(inflated, np.uint8(254), np.uint8(0))

    # Robot footprint clearing (0.35m)
    cx = int((robot_x - wx_min) / vs)
    cy = int((robot_y - wy_min) / vs)
    fp_r = int(0.35 / vs)
    ax_lo = max(0, cx - fp_r)
    ax_hi = min(nx, cx + fp_r + 1)
    ay_lo = max(0, cy - fp_r)
    ay_hi = min(ny, cy + fp_r + 1)
    if ax_lo < ax_hi and ay_lo < ay_hi:
        aix, aiy = np.ogrid[ax_lo:ax_hi, ay_lo:ay_hi]
        fp_mask = (aix - cx) ** 2 + (aiy - cy) ** 2 <= fp_r ** 2
        grid[ax_lo:ax_hi, ay_lo:ay_hi][fp_mask] = 0

    meta = {
        'origin_x': wx_min, 'origin_y': wy_min,
        'voxel_size': vs, 'nx': nx, 'ny': ny,
        'truncation': truncation,
    }
    return grid, meta


def stream_debug_viewer(game) -> None:
    """Stream state to Godot debug viewer at 10Hz/2Hz.

    Called from tick() when debug_server has a connected client.
    """
    from .game_astar import astar_on_god_view

    # Send ground-truth obstacle volumes once per connection
    geoms = getattr(game, '_obstacle_geoms', None)
    if geoms and not getattr(game, '_obstacles_sent', False):
        game._debug_server.send_obstacles(geoms)
        game._obstacles_sent = True

    if game._step_count % 10 == 0:
        x, y, yaw, z, roll, pitch = game._get_robot_pose()
        target = game._spawner.current_target
        tx = target.x if target else 0.0
        ty = target.y if target else 0.0
        joints = [0.0] * 12
        try:
            st = game._sim.get_state()
            joints = list(st.joint_positions[:12])
        except Exception:
            pass
        game._debug_server.send_robot_state(
            x, y, yaw, z, roll, pitch,
            joints, tx, ty,
            game._state.value,
            0.0, 0.0, 0,
            0.0, 0.0,
            False,  # heading_in_tip (unused)
        )

    if game._step_count % 25 == 0 and game._perception is not None:
        tsdf = game._perception._tsdf
        if tsdf is not None:
            game._debug_server.send_tsdf(tsdf)
        cost_grid = game._perception.world_cost_grid
        meta = game._perception.world_cost_meta
        if cost_grid is not None and meta is not None:
            game._debug_server.send_costmap_2d(
                cost_grid, meta['origin_x'], meta['origin_y'],
                meta['voxel_size'])
        if game._god_view_costmap is not None:
            gv_grid, gv_meta = game._god_view_costmap
            game._debug_server.send_god_view_costmap(
                gv_grid, gv_meta['origin_x'], gv_meta['origin_y'],
                gv_meta['voxel_size'])
            target = game._spawner.current_target
            if target is not None:
                gv_x, gv_y, _, _, _, _ = game._get_robot_pose()
                god_path = astar_on_god_view(
                    gv_grid, gv_meta, (gv_x, gv_y),
                    (target.x, target.y))
                if god_path is not None:
                    game._debug_server.send_god_view_path(god_path)


def write_god_view_path(game) -> None:
    """Write god-view A* path to temp file for MuJoCo overlay."""
    if (game._god_view_path_file is None
            or game._god_view_costmap is None
            or game._step_count % 25 != 0):
        return

    from .game_astar import astar_on_god_view

    gv_grid, gv_meta = game._god_view_costmap
    target = game._spawner.current_target
    if target is None:
        return

    gv_x, gv_y, _, _, _, _ = game._get_robot_pose()
    god_path = astar_on_god_view(
        gv_grid, gv_meta, (gv_x, gv_y),
        (target.x, target.y))
    if god_path is None:
        return

    buf = bytearray(len(god_path) * 8)
    for i, (wx, wy) in enumerate(god_path):
        struct.pack_into('ff', buf, i * 8, wx, wy)
    try:
        with open(game._god_view_path_file, 'wb') as f:
            f.write(buf)
    except OSError:
        pass


def tick_perception(game) -> None:
    """Send pose to perception subprocess + build costmap from TSDF voxels.

    Subprocess writes TSDF voxel file at scan_hz.
    Game process reads it, builds costmap, feeds path_critic and viewer.

    God-view TSDF (if enabled) updates here for F1 scoring only.
    """
    if game._step_count % 10 != 0 and game._step_count != 1:
        return

    _gt_x, _gt_y, _gt_yaw, _gt_z, _gt_roll, _gt_pitch = game._get_robot_pose()

    # God-view TSDF (F1 scoring only, not used for navigation)
    # Skip in headless mode — mj_multiRay blocks GIL for 10-50ms,
    # causing DDS timing instability and falls. Replay post-game instead.
    if game._god_view_tsdf is not None and not game._headless:
        if game._step_count % 25 == 0 or game._step_count == 1:
            game._god_view_tsdf.update(_gt_x, _gt_y, _gt_yaw, _gt_z)
            game._god_view_tsdf.write_temp_file("/tmp/god_view_tsdf.bin")

    # Send pose to perception subprocess (10Hz)
    perc_sub = getattr(game, '_perception_subprocess', None)
    if perc_sub is not None:
        nav_x, nav_y, nav_yaw = game._get_nav_pose()
        perc_sub.update_pose(nav_x, nav_y, _gt_z, nav_yaw,
                             _gt_roll, _gt_pitch)

        # Check if TSDF voxel file has been updated
        try:
            mt = os.path.getmtime(ROBOT_TSDF_FILE)
        except OSError:
            mt = 0.0
        if not hasattr(game, '_tsdf_mtime'):
            game._tsdf_mtime = 0.0
        if mt == game._tsdf_mtime:
            return  # No new TSDF data
        game._tsdf_mtime = mt

        # Read TSDF voxels → build costmap for viz (not navigation).
        # Navigation uses the god-view costmap seeded at startup —
        # robot-view costmap is too sparse early on and overwrites it.
        voxels = _read_tsdf_voxels()
        if voxels is not None:
            result = _build_costmap(voxels, nav_x, nav_y)
            if result is not None:
                grid, meta = result
                _write_costmap_viz(grid, meta)


def tick_slam_trails(game) -> None:
    """Record SLAM vs truth trails for visualization (10Hz)."""
    import math
    if game._odometry is None or game._step_count % 10 != 0:
        return
    truth_x, truth_y, _, _, _, _ = game._get_robot_pose()
    game._truth_trail.append((truth_x, truth_y))
    p = game._odometry.pose
    game._slam_trail.append((p.x, p.y))
    if game._telemetry is not None:
        drift = math.sqrt(
            (p.x - truth_x)**2 + (p.y - truth_y)**2)
        game._telemetry.record("slam", {
            "drift": round(drift, 4),
            "slam_x": round(p.x, 3),
            "slam_y": round(p.y, 3),
            "truth_x": round(truth_x, 3),
            "truth_y": round(truth_y, 3),
        })

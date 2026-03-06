"""Visualization file writers and debug viewer streaming.

Writes robot-view TSDF surface voxels and costmap data to temp files
that the MuJoCo viewer subprocess reads for rendering. Also handles
debug viewer state streaming and god-view A* path file writing.
"""
from __future__ import annotations

import struct

ROBOT_TSDF_FILE = "/tmp/robot_view_tsdf.bin"
ROBOT_COSTMAP_FILE = "/tmp/robot_view_costmap.bin"


def write_robot_tsdf_file(tsdf, display_resolution: float = 0.05) -> None:
    """Write robot-view TSDF surface voxels to temp file for MuJoCo viewer.

    Same binary format as god-view: u32 n + f32 half, then N x 3 float32 xyz.
    Downsamples to display_resolution for bounded voxel count.
    """
    import struct
    import numpy as np

    voxels = tsdf.get_surface_voxels(include_history=True)
    if len(voxels) == 0:
        buf = bytearray(8)
        struct.pack_into('<If', buf, 0, 0, 0.0)
        try:
            with open(ROBOT_TSDF_FILE, 'wb') as f:
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
        with open(ROBOT_TSDF_FILE, 'wb') as f:
            f.write(buf)
    except OSError:
        pass


def write_robot_costmap_file(tsdf, display_resolution: float = 0.05,
                             cost_floor: float = 0.20) -> None:
    """Write robot-view costmap built from TSDF surface history.

    Uses the same surface voxels as the grey cube visualization
    (include_history=True), so the costmap always matches visible cubes.
    Runs a cropped EDT for speed.

    Cost = 1 / (1 + dist^2) -- inverse-square falloff, no distance cutoff.
    Cells below cost_floor (5%) are zeroed so the data is honest.

    Header: u16 rows, u16 cols, f32 origin_x, f32 origin_y, f32 voxel_size
    Body: rows * cols uint8
    """
    import struct
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    voxels = tsdf.get_surface_voxels(include_history=True)
    if len(voxels) == 0:
        return

    res = display_resolution
    ox, oy = tsdf.origin_x, tsdf.origin_y

    # Project to 2D grid indices
    gx = ((voxels[:, 0] - ox) / res).astype(np.int32)
    gy = ((voxels[:, 1] - oy) / res).astype(np.int32)

    # Crop to bounding box + margin sized from where cost hits floor
    # 1/(1+d^2) = floor -> d = sqrt(1/floor - 1)
    max_dist = (1.0 / cost_floor - 1.0) ** 0.5
    margin = int(max_dist / res) + 2
    x_lo, x_hi = int(gx.min()) - margin, int(gx.max()) + margin + 1
    y_lo, y_hi = int(gy.min()) - margin, int(gy.max()) + margin + 1

    # Build occupied grid on cropped region
    cnx, cny = x_hi - x_lo, y_hi - y_lo
    occupied = np.zeros((cnx, cny), dtype=np.bool_)
    lgx, lgy = gx - x_lo, gy - y_lo
    valid = (lgx >= 0) & (lgx < cnx) & (lgy >= 0) & (lgy < cny)
    occupied[lgx[valid], lgy[valid]] = True

    # EDT -> inverse-square cost with scale factor for steep dropoff
    dist = distance_transform_edt(~occupied).astype(np.float32) * res
    scale = 0.10  # characteristic distance (m)
    cost_f = 1.0 / (1.0 + (dist / scale) ** 2)
    cost_u8 = (cost_f * 254).astype(np.uint8)
    cost_u8[cost_u8 < int(cost_floor * 254)] = 0

    # Transpose to (row=Y, col=X) for renderer
    grid_t = cost_u8.T
    rows, cols = grid_t.shape
    crop_ox = ox + x_lo * res
    crop_oy = oy + y_lo * res
    buf = struct.pack('<HHfff', rows, cols, crop_ox, crop_oy, res)
    try:
        with open(ROBOT_COSTMAP_FILE, 'wb') as f:
            f.write(buf)
            f.write(np.ascontiguousarray(grid_t).tobytes())
    except OSError:
        pass


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
        slam_x, slam_y, slam_yaw = x, y, yaw
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
            slam_x, slam_y, slam_yaw, z, roll, pitch,
            joints, tx, ty,
            game._state.value,
            0.0, 0.0, 0,
            0.0, 0.0,
            getattr(game, '_heading_in_tip', False),
        )

    if game._step_count % 50 == 0 and game._perception is not None:
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
            or game._step_count % 50 != 0):
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
    """Run staggered perception work within the tick cycle.

    Spreads expensive operations across different ticks to avoid
    blocking the control loop for 200-300ms on a single tick.
    Each operation runs at 4Hz (every 25 ticks) but offset.
    """
    _phase = game._step_count % 25

    # Phase 0: God-view + DirectScanner raycasts (same pose for F1 scoring)
    if _phase == 0:
        _gt_x, _gt_y, _gt_yaw, _gt_z, _gt_roll, _gt_pitch = game._get_robot_pose()
        if game._god_view_tsdf is not None:
            game._god_view_tsdf.update(_gt_x, _gt_y, _gt_yaw, _gt_z)
            game._god_view_tsdf.write_temp_file("/tmp/god_view_tsdf.bin")
        if (game._perception is not None
                and getattr(game._perception, '_direct_ready', False)):
            nav_x, nav_y, nav_yaw = game._get_nav_pose()
            game._perception.direct_scan_only(
                nav_x, nav_y, _gt_z, nav_yaw,
                roll=_gt_roll, pitch=_gt_pitch)

    # Phase 10: Cost grid build (~65-100ms).
    # Runs every OTHER cycle (2Hz) to limit GIL blocking.
    _cycle = game._step_count // 25
    if _phase == 10 and _cycle % 2 == 0 and game._perception is not None:
        game._perception.build_cost_grids_from_main()

    # Phase 15: Path critic update + viewer file writes
    if _phase == 15:
        if (game._perception is not None
                and game._path_critic is not None):
            cost_grid = game._perception.world_cost_grid
            meta = game._perception.world_cost_meta
            if cost_grid is not None and meta is not None:
                game._path_critic.set_world_cost(
                    cost_grid, meta['origin_x'], meta['origin_y'],
                    meta['voxel_size'],
                    truncation=meta.get('truncation', 0.5))
        if (not game._headless
                and game._perception is not None
                and game._perception._tsdf is not None):
            write_robot_tsdf_file(game._perception._tsdf)
            write_robot_costmap_file(game._perception._tsdf)


def get_occ_str(game) -> str:
    """Return formatted occupancy accuracy string, recomputing every 2000 ticks."""
    if game._scene_xml_path is None or game._perception is None:
        return ""
    if game._step_count - game._occ_compute_step >= 2000:
        try:
            from .test_occupancy import compute_3ds_v2, compute_3ds_god
            tsdf = game._perception._tsdf
            if tsdf is not None:
                game._cached_occ = compute_3ds_v2(
                    tsdf, game._scene_xml_path)
                # Z-filter GT for completeness: exclude surfaces
                # below LiDAR min detection height
                _lidar_min_z = getattr(
                    game._perception, '_MIN_WORLD_Z', 0.05)
                _tsdf_z_hi = getattr(
                    game._perception._cfg, 'costmap_z_hi', 0.80)
                game._cached_occ['god'] = compute_3ds_god(
                    tsdf, game._scene_xml_path,
                    gt_z_range=(_lidar_min_z, _tsdf_z_hi))
                game._occ_compute_step = game._step_count
        except Exception:
            pass
    if game._cached_occ is not None:
        o = game._cached_occ
        adh = o['adherence_mm']
        cpl = o['completeness_pct']
        phn = o['phantom_pct']
        parts = f" 3DS: adh={adh:.1f}mm cpl={cpl:.1f}% phn={phn:.1f}%"
        god = o.get('god')
        if god is not None:
            parts += f" god={god['score']:.0f}"
        return parts
    return ""


def gt_clearance(game) -> float:
    """Minimum ground-truth distance from robot surface to nearest obstacle."""
    import math
    if not game._obstacle_bodies:
        return float('inf')
    robot = game._sim.get_body("base")
    if robot is None:
        return float('inf')
    rx, ry = float(robot.pos[0]), float(robot.pos[1])
    min_d = float('inf')
    for name in game._obstacle_bodies:
        obs = game._sim.get_body(name)
        if obs is None:
            continue
        d = math.sqrt(
            (rx - float(obs.pos[0]))**2 + (ry - float(obs.pos[1]))**2)
        if d < min_d:
            min_d = d
    return max(0.0, min_d - 0.60)


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

"""Direct MuJoCo raycasting for motion-blur-free TSDF integration.

Bypasses DDS LiDAR (which has motion blur across 50 physics steps).
Casts all 32K rays instantaneously via mj_multiRay from game process.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


def init_direct_scanner(pipeline, scene_xml_path: str,
                        robot: str = "b2",
                        mj_model=None, mj_data=None) -> None:
    """Set up direct MuJoCo raycasting for motion-blur-free TSDF.

    Loads a static copy of the scene and pre-computes ray directions
    matching the real LiDAR (Hesai XT16: 2000 H x 16 V = 32000 rays).
    Call direct_scan() from game tick at desired rate.

    If mj_model/mj_data are provided, reuse them instead of loading
    a new model. This ensures identical raycasting results when both
    god-view and robot TSDFs share the same collision geometry.
    """
    import mujoco

    if mj_model is not None and mj_data is not None:
        pipeline._direct_model = mj_model
        pipeline._direct_data = mj_data
    else:
        pipeline._direct_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        pipeline._direct_data = mujoco.MjData(pipeline._direct_model)
        mujoco.mj_forward(pipeline._direct_model, pipeline._direct_data)

    # Find robot root body
    site_id = mujoco.mj_name2id(
        pipeline._direct_model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    exclude_body = -1
    if site_id >= 0:
        body_id = pipeline._direct_model.site_bodyid[site_id]
        while pipeline._direct_model.body_parentid[body_id] != 0:
            body_id = pipeline._direct_model.body_parentid[body_id]
        exclude_body = body_id
    pipeline._direct_exclude_body = exclude_body

    # Collect ALL robot geom IDs (root + child bodies)
    pipeline._direct_robot_geoms = set()
    if exclude_body >= 0:
        bodies = [exclude_body]
        visited = set()
        while bodies:
            bid = bodies.pop()
            if bid in visited:
                continue
            visited.add(bid)
            for g in range(pipeline._direct_model.ngeom):
                if pipeline._direct_model.geom_bodyid[g] == bid:
                    pipeline._direct_robot_geoms.add(g)
            for child in range(pipeline._direct_model.nbody):
                if pipeline._direct_model.body_parentid[child] == bid:
                    bodies.append(child)

    # Target geom IDs
    pipeline._direct_target_geoms = set()
    target_body = mujoco.mj_name2id(
        pipeline._direct_model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if target_body >= 0:
        for g in range(pipeline._direct_model.ngeom):
            if pipeline._direct_model.geom_bodyid[g] == target_body:
                pipeline._direct_target_geoms.add(g)
    pipeline._direct_exclude_geoms = (
        pipeline._direct_robot_geoms | pipeline._direct_target_geoms)

    # Pre-compute ray directions. Use reduced ray count when SLAM is
    # active to avoid GIL contention (numpy ops in scan thread block
    # the control loop, destabilizing the robot at 100Hz).
    # Full: 128 V × 500 H = 64000 rays (~200ms with obstacles).
    # Reduced: 32 V × 250 H = 8000 rays (~25ms with obstacles).
    _use_reduced = getattr(pipeline, '_use_reduced_rays', False)
    if _use_reduced:
        h_angles = np.deg2rad(np.arange(0.0, 360.0, 1.44))  # 250 rays
        v_angles_deg = np.linspace(-30.0, 30.0, 32).tolist()
    else:
        h_angles = np.deg2rad(np.arange(0.0, 360.0, 0.72))  # 500 rays
        v_angles_deg = np.linspace(-30.0, 30.0, 128).tolist()
    dirs = []
    for v_deg in v_angles_deg:
        v_rad = np.deg2rad(v_deg)
        cos_v, sin_v = np.cos(v_rad), np.sin(v_rad)
        channel = np.column_stack([
            np.cos(h_angles) * cos_v,
            np.sin(h_angles) * cos_v,
            np.full_like(h_angles, sin_v),
        ])
        dirs.append(channel)
    pipeline._direct_ray_dirs = np.ascontiguousarray(
        np.vstack(dirs), dtype=np.float64)
    pipeline._direct_n_rays = len(pipeline._direct_ray_dirs)

    # Working arrays
    pipeline._direct_geomid = np.zeros(pipeline._direct_n_rays, dtype=np.int32)
    pipeline._direct_dist = np.zeros(pipeline._direct_n_rays, dtype=np.float64)

    # Sensor offset per robot
    offsets = {
        "b2": [0.34, 0.0, 0.18],
        "go2": [0.20, 0.0, 0.10],
        "go2w": [0.20, 0.0, 0.10],
        "b2w": [0.34, 0.0, 0.18],
    }
    pipeline._direct_sensor_offset = np.array(
        offsets.get(robot, [0.34, 0.0, 0.18]), dtype=np.float64)

    pipeline._direct_ready = True
    pipeline._direct_scan_count = 0


def _cast_rays(pipeline, x: float, y: float, z: float, yaw: float,
               roll: float = 0.0, pitch: float = 0.0):
    """Cast rays and return (hits, sx, sy) or None if no valid hits."""
    import mujoco

    if not getattr(pipeline, '_direct_ready', False):
        return None

    c, s = np.cos(yaw), np.sin(yaw)
    off = pipeline._direct_sensor_offset
    sx = x + c * off[0] - s * off[1]
    sy = y + s * off[0] + c * off[1]
    sz = z + off[2]
    sensor_pos = np.array([sx, sy, sz], dtype=np.float64)

    # Full rotation: R(yaw) * R(pitch) * R(roll) for IMU-corrected rays
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    R = R_yaw @ R_pitch @ R_roll
    rays_world = (R @ pipeline._direct_ray_dirs.T).T
    rays_flat = np.ascontiguousarray(rays_world.flatten())

    mujoco.mj_multiRay(
        m=pipeline._direct_model, d=pipeline._direct_data,
        pnt=sensor_pos, vec=rays_flat,
        geomgroup=None, flg_static=1,
        bodyexclude=pipeline._direct_exclude_body,
        geomid=pipeline._direct_geomid,
        dist=pipeline._direct_dist,
        nray=pipeline._direct_n_rays, cutoff=10.0,
    )

    valid = ((pipeline._direct_dist >= 0.05)
             & (pipeline._direct_dist <= 10.0))
    if not np.any(valid):
        return None

    dists = pipeline._direct_dist[valid]
    dirs_v = rays_world[valid]
    geomids = pipeline._direct_geomid[valid]

    if pipeline._direct_exclude_geoms:
        keep = np.ones(len(geomids), dtype=bool)
        for gid in pipeline._direct_exclude_geoms:
            keep &= geomids != gid
        dists = dists[keep]
        dirs_v = dirs_v[keep]

    if len(dists) == 0:
        return None

    hits = sensor_pos + dirs_v * dists[:, np.newaxis]
    hits = hits.astype(np.float32)
    hits = hits[hits[:, 2] >= pipeline._MIN_WORLD_Z]
    if len(hits) == 0:
        return None

    return hits, float(sx), float(sy)


def direct_scan(pipeline, x: float, y: float, z: float,
                yaw: float, roll: float = 0.0, pitch: float = 0.0) -> int:
    """Cast rays and integrate into TSDF. Call from game tick.

    Returns number of valid hits integrated.
    """
    result = _cast_rays(pipeline, x, y, z, yaw, roll, pitch)
    if result is None:
        return 0

    hits, sx, sy = result

    # Integrate into TSDF
    pipeline._tsdf.integrate_scan_world(hits, sx, sy)
    pipeline._direct_scan_count += 1

    # Build cost grids after every integration (main thread, safe).
    with pipeline._imu_lock:
        ix, iy = pipeline._imu_x, pipeline._imu_y

    from .costmap_builder import build_cost_grids
    build_cost_grids(pipeline, pipeline._tsdf, ix, iy)

    from .perception import LiveCostmapQuery
    query = LiveCostmapQuery(pipeline)
    with pipeline._lock:
        pipeline._costmap_query = query

    return len(hits)


def direct_scan_only(pipeline, x: float, y: float, z: float,
                     yaw: float, roll: float = 0.0,
                     pitch: float = 0.0) -> int:
    """Raycast + TSDF integration only, NO cost grid build.

    Cost grid build is deferred to build_cost_grids_from_main()
    on a later tick to avoid blocking the control loop for 200+ms.
    """
    result = _cast_rays(pipeline, x, y, z, yaw, roll, pitch)
    if result is None:
        return 0

    hits, sx, sy = result

    pipeline._tsdf.integrate_scan_world(hits, sx, sy)
    pipeline._direct_scan_count += 1
    return len(hits)


def build_cost_grids_from_main(pipeline) -> None:
    """Build cost grids from TSDF. Call from game tick on a staggered phase.

    This is the expensive operation (~65-100ms for EDT) that was
    previously bundled into direct_scan(), blocking the control loop.
    """
    if pipeline._tsdf is None:
        return
    with pipeline._imu_lock:
        ix, iy = pipeline._imu_x, pipeline._imu_y

    from .costmap_builder import build_cost_grids
    build_cost_grids(pipeline, pipeline._tsdf, ix, iy)

    from .perception import LiveCostmapQuery
    query = LiveCostmapQuery(pipeline)
    with pipeline._lock:
        pipeline._costmap_query = query

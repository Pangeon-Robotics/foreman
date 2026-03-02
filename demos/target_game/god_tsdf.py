"""God-view TSDF: perfect LiDAR → TSDF rendering.

Uses mj_multiRay against known obstacle geometry to produce noise-free
TSDF voxels from the robot's actual sensor position. Shows "what the
TSDF should look like if perception were perfect."

Same sensor viewpoint + integration logic as the real pipeline, but
zero noise.
"""
from __future__ import annotations

import struct

import mujoco
import numpy as np


# LiDAR config matching layers_1_2/lidar.py defaults
_H_FOV = 360.0
_H_RESOLUTION = 0.18  # degrees
# Hesai XT16: 16 channels at 2° spacing, -15° to +15°
_V_ANGLES_DEG = [-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                   1.0,   3.0,   5.0,  7.0,  9.0, 11.0, 13.0, 15.0]
_MAX_RANGE = 10.0
_MIN_RANGE = 0.05

# Sensor body-frame offset (B2: <site name="lidar" pos="0.34 0 0.18"/>)
_SENSOR_OFFSETS = {
    "b2": np.array([0.34, 0.0, 0.18], dtype=np.float64),
    "go2": np.array([0.20, 0.0, 0.10], dtype=np.float64),
    "go2w": np.array([0.20, 0.0, 0.10], dtype=np.float64),
    "b2w": np.array([0.34, 0.0, 0.18], dtype=np.float64),
}


def _build_ray_dirs() -> np.ndarray:
    """Pre-compute sensor-local ray directions (N, 3) float64.

    Matches lidar.py: 2000 horizontal x 5 vertical = 10000 rays.
    """
    h_angles = np.deg2rad(np.arange(0.0, _H_FOV, _H_RESOLUTION))
    dirs = []
    for v_deg in _V_ANGLES_DEG:
        v_rad = np.deg2rad(v_deg)
        cos_v = np.cos(v_rad)
        sin_v = np.sin(v_rad)
        channel = np.column_stack([
            np.cos(h_angles) * cos_v,
            np.sin(h_angles) * cos_v,
            np.full_like(h_angles, sin_v),
        ])
        dirs.append(channel)
    return np.ascontiguousarray(np.vstack(dirs), dtype=np.float64)


class GodViewTSDF:
    """Perfect-raycast TSDF for god-view visualization.

    Loads a separate MuJoCo model (static, no physics stepping),
    casts rays from the robot's actual sensor position, and integrates
    hits into a real TSDF instance.
    """

    def __init__(self, scene_xml_path: str, robot: str = "b2",
                 voxel_size: float = 0.01, xy_extent: float = 10.0):
        # Load static model (obstacles only, no physics)
        self._model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Find robot root body to exclude from raycasting.
        # Walk up from the lidar site's body to the first non-world body,
        # matching lidar.py's _find_root_body logic.
        site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        if site_id >= 0:
            body_id = self._model.site_bodyid[site_id]
            while self._model.body_parentid[body_id] != 0:
                body_id = self._model.body_parentid[body_id]
            self._exclude_body = body_id
        else:
            self._exclude_body = -1

        # Build complete set of robot geom IDs (root + all child bodies).
        # bodyexclude only filters geoms on the specified body itself,
        # NOT child bodies. We must filter child-body geom hits manually.
        self._robot_geom_ids = set()
        if self._exclude_body >= 0:
            bodies = [self._exclude_body]
            visited = set()
            while bodies:
                bid = bodies.pop()
                if bid in visited:
                    continue
                visited.add(bid)
                for g in range(self._model.ngeom):
                    if self._model.geom_bodyid[g] == bid:
                        self._robot_geom_ids.add(g)
                for child in range(self._model.nbody):
                    if self._model.body_parentid[child] == bid:
                        bodies.append(child)

        # Pre-compute ray directions
        self._ray_dirs = _build_ray_dirs()
        self._n_rays = len(self._ray_dirs)

        # Working arrays for mj_multiRay output
        self._geomid = np.zeros(self._n_rays, dtype=np.int32)
        self._dist = np.zeros(self._n_rays, dtype=np.float64)

        # Sensor offset in body frame
        self._sensor_offset = _SENSOR_OFFSETS.get(
            robot, np.array([0.34, 0.0, 0.18], dtype=np.float64))

        # Find target body geom IDs to exclude from raycasting
        self._target_geom_ids = set()
        target_body = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if target_body >= 0:
            for g in range(self._model.ngeom):
                if self._model.geom_bodyid[g] == target_body:
                    self._target_geom_ids.add(g)

        # Create TSDF with a config tuned for god-view (no noise).
        # log_odds_hit=3.0 so a single hit crosses the 2.0 surface
        # threshold — perfect rays don't need multi-scan convergence.
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            tsdf_xy_extent=xy_extent,
            tsdf_z_range=(-0.5, 1.5),
            tsdf_voxel_size=voxel_size,
            tsdf_truncation=0.5,
            tsdf_log_odds_hit=3.0,
            tsdf_log_odds_free=0.25,
            tsdf_log_odds_max=5.0,
            tsdf_log_odds_min=-2.0,
            tsdf_unknown_cell_cost=0.5,
            costmap_z_lo=0.05,
            costmap_z_hi=0.80,
            tsdf_output_resolution=0.05,
            tsdf_depth_extension=5,
            tsdf_decay_rate=0.0,  # god-view: no decay, perfect rays
        )
        from layer_6.world_model.tsdf import TSDF
        self._tsdf = TSDF(cfg)

    def update(self, robot_x: float, robot_y: float,
               robot_yaw: float, robot_z: float) -> int:
        """Cast perfect rays and integrate hits into TSDF.

        Returns number of surface voxels.
        """
        # Sensor world position: rotate body-frame offset by yaw
        c = np.cos(robot_yaw)
        s = np.sin(robot_yaw)
        off = self._sensor_offset
        sx = robot_x + c * off[0] - s * off[1]
        sy = robot_y + s * off[0] + c * off[1]
        sz = robot_z + off[2]

        sensor_pos = np.array([sx, sy, sz], dtype=np.float64)

        # Rotate ray directions from sensor-local to world frame
        R_yaw = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ], dtype=np.float64)
        rays_world = (R_yaw @ self._ray_dirs.T).T
        rays_flat = np.ascontiguousarray(rays_world.flatten())

        # Cast all rays
        mujoco.mj_multiRay(
            m=self._model,
            d=self._data,
            pnt=sensor_pos,
            vec=rays_flat,
            geomgroup=None,
            flg_static=1,
            bodyexclude=self._exclude_body,
            geomid=self._geomid,
            dist=self._dist,
            nray=self._n_rays,
            cutoff=_MAX_RANGE,
        )

        # Filter valid hits
        valid = (self._dist >= _MIN_RANGE) & (self._dist <= _MAX_RANGE)
        if not np.any(valid):
            return 0

        dists = self._dist[valid]
        dirs = rays_world[valid]
        geomids = self._geomid[valid]

        # Filter robot child-body geom hits (bodyexclude only covers root
        # body geoms, not child bodies like limbs) and target geom hits.
        exclude_geoms = self._robot_geom_ids | self._target_geom_ids
        if exclude_geoms:
            keep = np.ones(len(geomids), dtype=bool)
            for gid in exclude_geoms:
                keep &= geomids != gid
            dists = dists[keep]
            dirs = dirs[keep]

        if len(dists) == 0:
            return 0

        # Compute world hit points
        hits = sensor_pos + dirs * dists[:, np.newaxis]
        hits = hits.astype(np.float32)

        # Filter ground hits (below 0.05m)
        hits = hits[hits[:, 2] >= 0.05]
        if len(hits) == 0:
            return 0

        # Integrate into TSDF
        self._tsdf.integrate_scan_world(hits, float(sx), float(sy))

        voxels = self._tsdf.get_surface_voxels(include_history=True)
        return len(voxels)

    def write_temp_file(self, path: str = "/tmp/god_view_tsdf.bin",
                        display_resolution: float = 0.05) -> None:
        """Write surface voxels to binary file for MuJoCo renderer.

        Downsamples from internal 1cm voxels to display_resolution (default
        5cm) to keep the voxel count bounded.  Without this, the renderer's
        stride-based subsampling creates visual churn as the history set
        grows across frames.

        Format: u32 n_voxels + f32 voxel_half_size (8 bytes header)
                N x 3 float32 xyz (12 bytes per voxel)
        """
        voxels = self._tsdf.get_surface_voxels(include_history=True)
        if len(voxels) == 0:
            buf = bytearray(8)
            struct.pack_into('<If', buf, 0, 0, 0.0)
            with open(path, 'wb') as f:
                f.write(buf)
            return

        # Snap to display grid and deduplicate for stable rendering
        keys = (voxels / display_resolution).astype(np.int32)
        _, idx = np.unique(keys, axis=0, return_index=True)
        voxels = voxels[idx]

        n = len(voxels)
        half = display_resolution / 2.0

        buf = bytearray(8 + n * 12)
        struct.pack_into('<If', buf, 0, n, half)
        voxels_f32 = voxels.astype(np.float32)
        buf[8:] = voxels_f32.tobytes()

        with open(path, 'wb') as f:
            f.write(buf)

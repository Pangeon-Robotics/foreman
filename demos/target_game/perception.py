"""Perception pipeline for the target game.

Subscribes to LiDAR point cloud, builds TSDF/costmap, and exposes
the latest costmap for A* path planning. Runs the heavy work (EDT)
on point cloud arrival (~10Hz), not at control rate (100Hz).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np



class LiveCostmapQuery:
    """CostmapQuery that samples the world-frame grid with live pose.

    Instead of building a body-frame snapshot (which goes stale between
    LiDAR callbacks), this rotates body-frame query points to world frame
    using the CURRENT robot pose and looks up the world-frame DWA cost
    grid.  Eliminates the stale-snapshot bug where obstacles appear at
    wrong body-frame positions.

    Compatible with CurvatureDWAPlanner (exposes sample / sample_batch).
    """

    def __init__(self, perception, oob_cost: float = 0.35):
        self._p = perception
        self._oob_cost = oob_cost
        # Expose _grid=None so diagnostic code doesn't crash
        self._grid = None
        self._costmap = None
        self._res = getattr(
            perception._cfg, 'tsdf_output_resolution', 0.05)

    def sample(self, x: float, y: float) -> float:
        p = self._p
        grid = p._dwa_cost_grid
        meta = p._world_cost_meta
        if grid is None or meta is None:
            return self._oob_cost
        yaw = p._imu_yaw
        c = math.cos(yaw)
        s = math.sin(yaw)
        wx = c * x - s * y + p._imu_x
        wy = s * x + c * y + p._imu_y
        vs = meta['voxel_size']
        ix = int((wx - meta['origin_x']) / vs)
        iy = int((wy - meta['origin_y']) / vs)
        if 0 <= ix < meta['nx'] and 0 <= iy < meta['ny']:
            raw = int(grid[ix, iy])
            if raw == 255:
                return self._oob_cost
            return raw / 254.0
        return self._oob_cost

    def sample_batch(self, points: np.ndarray) -> np.ndarray:
        p = self._p
        grid = p._dwa_cost_grid
        meta = p._world_cost_meta
        if grid is None or meta is None:
            return np.full(len(points), self._oob_cost, dtype=np.float32)
        yaw = p._imu_yaw
        c = np.cos(yaw)
        s = np.sin(yaw)
        wx = c * points[:, 0] - s * points[:, 1] + p._imu_x
        wy = s * points[:, 0] + c * points[:, 1] + p._imu_y
        vs = meta['voxel_size']
        ix = ((wx - meta['origin_x']) / vs).astype(np.intp)
        iy = ((wy - meta['origin_y']) / vs).astype(np.intp)
        nx, ny = meta['nx'], meta['ny']
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        costs = np.full(len(points), self._oob_cost, dtype=np.float32)
        raw = grid[ix[valid], iy[valid]]
        known = raw != 255
        costs_valid = costs[valid]
        costs_valid[known] = raw[known].astype(np.float32) / 254.0
        costs[valid] = costs_valid
        return costs


class PerceptionPipeline:
    """Lightweight perception wrapper for target game integration.

    Thread safety: the DDS callback runs in a listener thread.
    The game tick reads the latest costmap from the main thread.
    A lock protects the shared costmap reference.

    Parameters
    ----------
    odometry : Odometry
        SLAM odometry (for pose at time of scan).
    perception_config : PerceptionConfig
        Robot-specific TSDF/costmap parameters.
    """

    def __init__(self, odometry, perception_config):
        from layer_6.world_model.costmap import CostmapQuery
        from layer_6.types import Costmap2D, Pose2D, PointCloud

        self._odometry = odometry
        self._cfg = perception_config

        from layer_6.world_model.tsdf import TSDF
        self._tsdf = TSDF(perception_config)

        # Thread-safe costmap storage
        self._lock = threading.Lock()
        self._costmap_query: CostmapQuery | None = None
        self._costmap: Costmap2D | None = None
        self._world_cost_grid: np.ndarray | None = None  # uint8 (nx, ny)
        self._world_cost_meta: dict | None = None
        self._last_cloud_time: float = 0.0
        self._build_times: list[float] = []

        # Safety valve: consecutive feas=0 counter.

        self._force_feasible = False

        # Scan throttle
        self._scan_min_interval = getattr(
            perception_config, 'scan_min_interval', 0.50)
        self._scan_last_time = 0.0

        # Shutdown flag
        self._shutdown = False

        # Direct scanner flag (set by init_direct_scanner)
        self._direct_ready = False

        # Full 6-DOF IMU pose (set by game tick, read by DDS callback)
        self._imu_lock = threading.Lock()
        self._imu_x = 0.0
        self._imu_y = 0.0
        self._imu_z = 0.18
        self._imu_roll = 0.0
        self._imu_pitch = 0.0
        self._imu_yaw = 0.0

        # Target marker positions (set by game, used to filter LiDAR hits)
        self._target_positions: list[tuple[float, float]] = [(5.0, 0.0)]
        self._TARGET_FILTER_RADIUS = 0.35

        # Keep references to avoid re-importing
        self._CostmapQuery = CostmapQuery
        self._PointCloud = PointCloud
        self._Pose2D = Pose2D

    # ------------------------------------------------------------------
    # Direct scanner: delegated to direct_scanner module
    # ------------------------------------------------------------------

    def init_direct_scanner(self, scene_xml_path: str,
                            robot: str = "b2",
                            mj_model=None, mj_data=None) -> None:
        """Set up direct MuJoCo raycasting for motion-blur-free TSDF."""
        from .direct_scanner import init_direct_scanner
        init_direct_scanner(self, scene_xml_path, robot, mj_model, mj_data)

    def direct_scan(self, x: float, y: float, z: float,
                    yaw: float, roll: float = 0.0,
                    pitch: float = 0.0) -> int:
        """Cast rays and integrate into TSDF. Returns number of valid hits."""
        from .direct_scanner import direct_scan
        return direct_scan(self, x, y, z, yaw, roll, pitch)

    def direct_scan_only(self, x: float, y: float, z: float,
                         yaw: float, roll: float = 0.0,
                         pitch: float = 0.0) -> int:
        """Raycast + TSDF integration only, NO cost grid build."""
        from .direct_scanner import direct_scan_only
        return direct_scan_only(self, x, y, z, yaw, roll, pitch)

    def build_cost_grids_from_main(self) -> None:
        """Build cost grids from TSDF. Call from game tick on a staggered phase."""
        from .direct_scanner import build_cost_grids_from_main
        build_cost_grids_from_main(self)

    def shutdown(self) -> None:
        """Stop the DDS callback from processing new scans."""
        self._shutdown = True

    def set_imu_pose(self, x: float, y: float, z: float,
                     roll: float, pitch: float, yaw: float) -> None:
        """Update full 6-DOF pose from game tick (ground-truth MuJoCo body)."""
        with self._imu_lock:
            self._imu_x = x
            self._imu_y = y
            self._imu_z = z
            self._imu_roll = roll
            self._imu_pitch = pitch
            self._imu_yaw = yaw

    def set_target_position(self, x: float, y: float) -> None:
        """Add target marker position for LiDAR hit filtering."""
        self._target_positions.append((x, y))

    @staticmethod
    def _build_rotation(roll, pitch, yaw):
        """Build ZYX Euler rotation matrix from roll, pitch, yaw."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr            ],
        ], dtype=np.float32)

    @staticmethod
    def _transform_points_world(pts_local, sensor_offset, body_pos, R):
        """Transform sensor-local points to world frame."""
        pts_body = pts_local + sensor_offset
        world = (R @ pts_body.T).T
        world[:, 0] += body_pos[0]
        world[:, 1] += body_pos[1]
        world[:, 2] += body_pos[2]
        return world

    # LiDAR site offset in body frame
    _SENSOR_OFFSET = np.array([0.34, 0.0, 0.18], dtype=np.float32)

    # Minimum world-frame Z for valid LiDAR hits
    _MIN_WORLD_Z = 0.30

    def on_point_cloud(self, msg) -> None:
        """DDS callback: new point cloud received."""
        if self._shutdown:
            return
        if self._direct_ready:
            return

        now = time.monotonic()
        if self._scan_min_interval > 0:
            if now - self._scan_last_time < self._scan_min_interval:
                return

        t0 = now
        num_pts = int(msg.num_points)
        if num_pts == 0:
            return

        data = np.array(msg.data[:num_pts * 3], dtype=np.float32).reshape(-1, 3)
        dist_sq = data[:, 0]**2 + data[:, 1]**2
        data = data[dist_sq > 0.64]  # 0.8m squared
        if len(data) == 0:
            return

        with self._imu_lock:
            imu_x = self._imu_x
            imu_y = self._imu_y
            imu_z = self._imu_z
            imu_roll = self._imu_roll
            imu_pitch = self._imu_pitch
            imu_yaw = self._imu_yaw

        if imu_z < 0.30:
            return

        R = self._build_rotation(imu_roll, imu_pitch, imu_yaw)
        body_pos = np.array([imu_x, imu_y, imu_z], dtype=np.float32)
        pts_world = self._transform_points_world(
            data, self._SENSOR_OFFSET, body_pos, R)

        valid = pts_world[:, 2] >= self._MIN_WORLD_Z
        pts_world = pts_world[valid]
        if len(pts_world) == 0:
            return

        r2 = self._TARGET_FILTER_RADIUS ** 2
        for tx, ty in self._target_positions:
            dx = pts_world[:, 0] - tx
            dy = pts_world[:, 1] - ty
            pts_world = pts_world[dx * dx + dy * dy > r2]
            if len(pts_world) == 0:
                return

        sensor_world = R @ self._SENSOR_OFFSET + body_pos
        pose = self._odometry.pose
        sensor_pose = self._Pose2D(x=pose.x, y=pose.y, yaw=pose.yaw, stamp=pose.stamp)

        self._tsdf.integrate_scan_world(
            pts_world, float(sensor_world[0]), float(sensor_world[1]))

        time.sleep(0.005)

        from .costmap_builder import build_cost_grids
        build_cost_grids(self, self._tsdf, imu_x, imu_y)

        query = LiveCostmapQuery(self)
        with self._lock:
            self._costmap_query = query
            self._last_cloud_time = msg.timestamp

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._build_times.append(elapsed_ms)
        self._scan_last_time = time.monotonic()
        if elapsed_ms > 150:
            n = len(self._build_times)
            cg_ms = getattr(self, '_last_cg_ms', 0)
            astar_ms = getattr(self, '_last_astar_ms', 0)
            pass  # slow build logged internally (>150ms)

    @property
    def costmap_query(self):
        """Latest CostmapQuery, or None if no scan received yet."""
        with self._lock:
            return self._costmap_query

    @property
    def costmap(self):
        """Latest Costmap2D, or None if no scan received yet."""
        with self._lock:
            return self._costmap

    @property
    def world_cost_grid(self) -> np.ndarray | None:
        """Latest world-frame cost grid (uint8), or None if no scan yet."""
        return self._world_cost_grid

    @property
    def world_cost_meta(self) -> dict | None:
        """Metadata for world_cost_grid."""
        return self._world_cost_meta

    @property
    def stats(self) -> dict:
        """Perception timing statistics."""
        if not self._build_times:
            return {"builds": 0}
        s = {
            "builds": len(self._build_times),
            "mean_ms": sum(self._build_times) / len(self._build_times),
            "max_ms": max(self._build_times),
            "min_ms": min(self._build_times),
        }
        tsdf = self._tsdf
        s["n_chunks"] = tsdf.n_chunks
        s["n_converged"] = tsdf.n_converged
        s["memory_mb"] = tsdf.memory_mb
        return s

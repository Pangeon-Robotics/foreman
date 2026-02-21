"""Perception pipeline for the target game.

Subscribes to LiDAR point cloud, builds TSDF/costmap, and exposes
the latest costmap for the DWA planner. Runs the heavy work (EDT)
on point cloud arrival (~10Hz), not at control rate (100Hz).
"""
from __future__ import annotations

import threading
import time

import numpy as np


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
        from layer_6.world_model.tsdf import TSDFGrid, build_costmap_2d_direct
        from layer_6.world_model.costmap import CostmapBuilder, CostmapQuery
        from layer_6.types import Costmap2D, Pose2D, PointCloud

        self._odometry = odometry
        self._cfg = perception_config

        # TSDF grid (reused each scan)
        self._tsdf = TSDFGrid(
            xy_extent=perception_config.tsdf_xy_extent,
            z_range=perception_config.tsdf_z_range,
            voxel_size=perception_config.tsdf_voxel_size,
            truncation=perception_config.tsdf_truncation,
        )

        # Costmap projection
        self._costmap_builder = CostmapBuilder(
            z_slice=(perception_config.costmap_z_lo, perception_config.costmap_z_hi),
            truncation=perception_config.tsdf_truncation,
            z_min=perception_config.tsdf_z_range[0],
            voxel_size=perception_config.tsdf_voxel_size,
        )

        # Thread-safe costmap storage
        self._lock = threading.Lock()
        self._costmap_query: CostmapQuery | None = None
        self._costmap: Costmap2D | None = None
        self._last_cloud_time: float = 0.0
        self._build_times: list[float] = []

        # Keep references to avoid re-importing
        self._build_costmap_2d_direct = build_costmap_2d_direct
        self._CostmapQuery = CostmapQuery
        self._PointCloud = PointCloud
        self._Pose2D = Pose2D

    def on_point_cloud(self, msg) -> None:
        """DDS callback: new point cloud received.

        Runs in the DDS listener thread. Builds costmap and stores it.
        """
        t0 = time.monotonic()

        # Parse point cloud from DDS message
        num_pts = int(msg.num_points)
        if num_pts == 0:
            return

        data = np.array(msg.data[:num_pts * 3], dtype=np.float32).reshape(-1, 3)

        # LiDAR points are in sensor-local frame (Z=0 for horizontal rays).
        # The costmap height filter uses strict z > z_lo (z_lo=0.0), which
        # excludes horizontal returns at Z=0. Offset by sensor height above
        # body center so points register within the costmap z-range.
        data[:, 2] += getattr(self._cfg, 'lidar_z_offset', 0.18)

        # Get current SLAM pose for the costmap origin
        pose = self._odometry.pose
        # Build costmap in BODY frame (yaw=0) so it aligns with DWA's
        # body-frame arc queries.  build_costmap_2d_direct rotates LiDAR
        # points by pose.yaw via frame_transform — passing yaw=0 keeps
        # points in sensor/body frame, which is the same frame the DWA
        # planner uses for its arc waypoints.
        l6_pose = self._Pose2D(x=pose.x, y=pose.y, yaw=0.0, stamp=pose.stamp)

        # Build 2D costmap directly (Phase 1: skip 3D TSDF for speed)
        cloud = self._PointCloud(points=data, ring_index=None, stamp=msg.timestamp)
        cost_grid = self._build_costmap_2d_direct(
            cloud, l6_pose,
            xy_extent=self._cfg.tsdf_xy_extent,
            z_range=(self._cfg.costmap_z_lo, self._cfg.costmap_z_hi),
            resolution=self._cfg.tsdf_voxel_size,
            truncation=self._cfg.tsdf_truncation,
            square_cost=getattr(self._cfg, 'square_cost', False),
        )

        from layer_6.types import Costmap2D
        costmap = Costmap2D(
            grid=cost_grid,
            resolution=self._cfg.tsdf_voxel_size,
            origin_x=pose.x - self._cfg.tsdf_xy_extent,
            origin_y=pose.y - self._cfg.tsdf_xy_extent,
            robot_yaw=pose.yaw,
            stamp=msg.timestamp,
        )

        query = self._CostmapQuery(costmap)

        with self._lock:
            self._costmap_query = query
            self._costmap = costmap
            self._last_cloud_time = msg.timestamp

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._build_times.append(elapsed_ms)

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
    def stats(self) -> dict:
        """Perception timing statistics."""
        if not self._build_times:
            return {"builds": 0}
        return {
            "builds": len(self._build_times),
            "mean_ms": sum(self._build_times) / len(self._build_times),
            "max_ms": max(self._build_times),
            "min_ms": min(self._build_times),
        }

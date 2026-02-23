"""Perception pipeline for the target game.

Subscribes to LiDAR point cloud, builds TSDF/costmap, and exposes
the latest costmap for the DWA planner. Runs the heavy work (EDT)
on point cloud arrival (~10Hz), not at control rate (100Hz).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Reactive forward scan: costmap probes beyond DWA horizon
# ---------------------------------------------------------------------------
# 5 rays at [-50, -25, 0, +25, +50] degrees, sampled at [0.5, 1.0, 1.5] m.
# Pre-computed as body-frame (x, y) points -- 15 total.

_RAY_ANGLES_DEG = np.array([-50, -25, 0, 25, 50], dtype=np.float64)
_RAY_ANGLES = np.deg2rad(_RAY_ANGLES_DEG)
_RAY_DISTS = np.array([0.5, 1.0, 1.5], dtype=np.float64)

# Shape (15, 2): body-frame [x, y] for each (angle, distance) pair.
_SCAN_POINTS = np.array([
    [d * math.cos(a), d * math.sin(a)]
    for a in _RAY_ANGLES
    for d in _RAY_DISTS
], dtype=np.float32)

# Distance weights for 3 samples along each ray: closer = higher weight
_DIST_WEIGHTS = np.array([1.0, 0.6, 0.3], dtype=np.float64)
_DIST_WEIGHTS_NORM = _DIST_WEIGHTS / _DIST_WEIGHTS.sum()

# Lateral weights: sin(angle) for each ray. Used to compute asymmetry
# from all 5 rays -- positive means left side more obstructed.
_LATERAL_WEIGHTS = np.sin(_RAY_ANGLES)  # [-sin50, -sin25, 0, sin25, sin50]

# Tuning constants
_MIN_SPEED_FACTOR = 0.55        # floor -- always maintain walking gait
_ASYMMETRY_GAIN = 3.0           # turn bias per unit asymmetry
_SYMMETRY_TIEBREAKER = 0.2      # forced asymmetry when obstacle is dead-ahead
_HIGH_THREAT = 0.55             # above this, override DWA turn entirely


@dataclass
class ReactiveScanResult:
    """Output of reactive_scan(), for telemetry."""
    mod_forward: float
    mod_turn: float
    threat: float
    asymmetry: float
    emergency: bool


def reactive_scan(
    costmap_query, dwa_forward: float, dwa_turn: float,
    goal_bearing: float = 0.0,
) -> ReactiveScanResult:
    """Probe costmap ahead of robot and modulate DWA output.

    Queries 15 body-frame points (5 rays x 3 distances) in the costmap
    and computes two signals:
      1. Forward threat -> speed reduction (with floor to maintain walking)
      2. Lateral asymmetry -> turn bias away from obstructed side

    When an obstacle is dead-ahead (high threat, near-zero asymmetry),
    uses goal_bearing to break symmetry and commit to an avoidance
    direction.  At high threat, overrides DWA turn entirely (DWA
    oscillates +/-0.1 when confused -- its signal is useless).
    """
    costs = costmap_query.sample_batch(_SCAN_POINTS)  # (15,)

    # Per-ray distance-weighted costs (5 rays)
    ray_costs = costs.reshape(5, 3)              # (5 rays, 3 dists)
    per_ray = ray_costs @ _DIST_WEIGHTS_NORM     # (5,) cost per ray

    # Forward threat: max cost across center 3 rays (-25deg, 0deg, +25deg)
    threat = float(np.max(per_ray[1:4]))
    threat = min(max(threat, 0.0), 1.0)

    # Lateral asymmetry from all 5 rays weighted by sin(angle).
    # Positive = left side more obstructed -> robot should turn right.
    asymmetry = float(per_ray @ _LATERAL_WEIGHTS)

    # Symmetry breaking -- when obstacle is dead-ahead and scan
    # can't tell which side is better, use goal bearing to choose.
    if threat > 0.3 and abs(asymmetry) < 0.15:
        if abs(goal_bearing) > 0.05:
            asymmetry += math.copysign(_SYMMETRY_TIEBREAKER, -goal_bearing)
        else:
            asymmetry += -_SYMMETRY_TIEBREAKER

    # Speed reduction: quadratic -- gentle braking at moderate threat,
    # hard braking only when genuinely close.  Linear (1.0 - threat)
    # was too aggressive at threat=0.65.
    speed_factor = max(_MIN_SPEED_FACTOR, 1.0 - threat * threat)
    mod_forward = dwa_forward * speed_factor

    # At high threat, DWA turn is unreliable (oscillating +/-0.1).
    # Override it entirely with the reactive scan's avoidance direction.
    if threat > _HIGH_THREAT:
        avoidance = -asymmetry * _ASYMMETRY_GAIN
        mod_turn = max(-1.0, min(1.0, avoidance))
    else:
        # Blend: add scan bias to DWA's output
        mod_turn = dwa_turn + (-asymmetry * _ASYMMETRY_GAIN * threat)

    return ReactiveScanResult(
        mod_forward=mod_forward,
        mod_turn=mod_turn,
        threat=threat,
        asymmetry=asymmetry,
        emergency=False,
    )


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

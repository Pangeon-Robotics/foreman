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
# 5 rays at [-50, -25, 0, +25, +50] degrees, sampled at [0.5, 1.0, 1.5, 2.5] m.
# Pre-computed as body-frame (x, y) points -- 20 total.
# Extended to 2.5m (from 1.5m) so reactive scan detects obstacles before the
# robot commits to a path at 1.3m/s — gives ~1.9s warning instead of ~1.1s.

_RAY_ANGLES_DEG = np.array([-50, -25, 0, 25, 50], dtype=np.float64)
_RAY_ANGLES = np.deg2rad(_RAY_ANGLES_DEG)
_RAY_DISTS = np.array([0.5, 1.0, 1.5, 2.5], dtype=np.float64)

# Shape (20, 2): body-frame [x, y] for each (angle, distance) pair.
_SCAN_POINTS = np.array([
    [d * math.cos(a), d * math.sin(a)]
    for a in _RAY_ANGLES
    for d in _RAY_DISTS
], dtype=np.float32)

# Distance weights for 4 samples along each ray: closer = higher weight
_DIST_WEIGHTS = np.array([1.0, 0.6, 0.3, 0.15], dtype=np.float64)
_DIST_WEIGHTS_NORM = _DIST_WEIGHTS / _DIST_WEIGHTS.sum()

# Lateral weights: sin(angle) for each ray. Used to compute asymmetry
# from all 5 rays -- positive means left side more obstructed.
_LATERAL_WEIGHTS = np.sin(_RAY_ANGLES)  # [-sin50, -sin25, 0, sin25, sin50]

# Tuning constants
_MIN_SPEED_FACTOR = 0.40        # floor -- 0.40 * 0.30m = 0.12m stride, above minimum effective
_ASYMMETRY_GAIN = 3.0           # turn bias per unit asymmetry
_SYMMETRY_TIEBREAKER = 0.35     # stronger goal-bearing bias when obstacle dead-ahead
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

    Queries 20 body-frame points (5 rays x 4 distances) in the costmap
    and computes two signals:
      1. Forward threat -> speed reduction (with floor to maintain walking)
      2. Lateral asymmetry -> turn bias away from obstructed side

    When an obstacle is dead-ahead (high threat, near-zero asymmetry),
    uses goal_bearing to break symmetry and commit to an avoidance
    direction.  At high threat, overrides DWA turn entirely (DWA
    oscillates +/-0.1 when confused -- its signal is useless).
    """
    costs = costmap_query.sample_batch(_SCAN_POINTS)  # (20,)

    # Per-ray distance-weighted costs (5 rays)
    n_dists = len(_RAY_DISTS)
    ray_costs = costs.reshape(5, n_dists)        # (5 rays, 4 dists)
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
        if abs(asymmetry) < 0.15:
            # Symmetric corridor: DWA has better spatial awareness than 5-ray scan.
            # Blend rather than override.
            mod_turn = 0.5 * dwa_turn + 0.5 * max(-1.0, min(1.0, avoidance))
        else:
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
        from layer_6.world_model.tsdf import PersistentTSDF
        from layer_6.world_model.costmap import CostmapQuery
        from layer_6.types import Costmap2D, Pose2D, PointCloud

        self._odometry = odometry
        self._cfg = perception_config

        # Persistent world-frame TSDF (Bayesian, accumulates across scans)
        self._tsdf = PersistentTSDF(perception_config)

        # Thread-safe costmap storage
        self._lock = threading.Lock()
        self._costmap_query: CostmapQuery | None = None
        self._costmap: Costmap2D | None = None
        self._world_cost_grid: np.ndarray | None = None  # uint8 (nx, ny)
        self._world_cost_meta: dict | None = None
        self._last_cloud_time: float = 0.0
        self._build_times: list[float] = []

        # Full 6-DOF IMU pose (set by game tick, read by DDS callback)
        self._imu_lock = threading.Lock()
        self._imu_x = 0.0
        self._imu_y = 0.0
        self._imu_z = 0.18  # fallback: approximate sensor height
        self._imu_roll = 0.0
        self._imu_pitch = 0.0
        self._imu_yaw = 0.0

        # Target marker positions (set by game, used to filter LiDAR hits).
        # The target mocap sphere is hit by LiDAR but is not an obstacle.
        # Tracks ALL historical positions since old TSDF voxels persist.
        # Default includes scene XML initial position (5, 0).
        self._target_positions: list[tuple[float, float]] = [(5.0, 0.0)]
        self._TARGET_FILTER_RADIUS = 0.35  # filter hits within 0.35m of target

        # Keep references to avoid re-importing
        self._CostmapQuery = CostmapQuery
        self._PointCloud = PointCloud
        self._Pose2D = Pose2D

    def set_imu_pose(self, x: float, y: float, z: float,
                     roll: float, pitch: float, yaw: float) -> None:
        """Update full 6-DOF pose from game tick (ground-truth MuJoCo body).

        Called from the main thread at control rate (100Hz). The DDS
        callback reads these values to build the full 3D rotation matrix
        for transforming sensor-local LiDAR points to world frame.
        """
        with self._imu_lock:
            self._imu_x = x
            self._imu_y = y
            self._imu_z = z
            self._imu_roll = roll
            self._imu_pitch = pitch
            self._imu_yaw = yaw

    def set_target_position(self, x: float, y: float) -> None:
        """Add target marker position for LiDAR hit filtering.

        Tracks all historical positions since TSDF voxels from old
        target positions persist until free rays clear them.
        """
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
        """Transform sensor-local points to world frame.

        Accounts for the LiDAR sensor's body-frame offset (the sensor
        is NOT at the body center). In body frame:
            pt_body = sensor_offset + pt_sensor
        In world:
            pt_world = R @ pt_body + body_pos
                     = R @ (sensor_offset + pt_sensor) + body_pos

        Parameters
        ----------
        pts_local : (N, 3) sensor-local points
        sensor_offset : (3,) sensor position in body frame (from site XML)
        body_pos : (3,) body center position in world frame
        R : (3, 3) body-to-world rotation matrix
        """
        # Shift from sensor-local to body-local by adding sensor offset
        pts_body = pts_local + sensor_offset
        # Rotate from body to world and translate
        world = (R @ pts_body.T).T
        world[:, 0] += body_pos[0]
        world[:, 1] += body_pos[1]
        world[:, 2] += body_pos[2]
        return world

    # LiDAR site offset in body frame — from B2 XML: <site name="lidar" pos="0.34 0 0.18"/>
    # TODO: read from model at init for multi-robot support
    _SENSOR_OFFSET = np.array([0.34, 0.0, 0.18], dtype=np.float32)

    # Minimum world-frame Z for valid LiDAR hits. Points below this are
    # ground-plane artifacts that slip through lidar.py's filter after
    # 3D rotation shifts their Z. B2 LiDAR at ~0.65m height; downward
    # channels (-15°, -7°) at 2-4m range hit ground at z=0.13-0.25m.
    # Raising from 0.10 to 0.25 cuts most ground-ring false positives
    # while preserving obstacle base detection above 0.25m.
    _MIN_WORLD_Z = 0.25

    def on_point_cloud(self, msg) -> None:
        """DDS callback: new point cloud received.

        Runs in the DDS listener thread. Integrates scan into persistent
        TSDF and extracts a body-frame costmap for the DWA planner.

        Uses full 3D IMU rotation (roll/pitch/yaw) to transform sensor-local
        LiDAR points to world frame before TSDF integration. Accounts for
        the sensor's body-frame offset (LiDAR is 0.34m forward, 0.18m above
        body center on B2).
        """
        t0 = time.monotonic()

        # Parse point cloud from DDS message
        num_pts = int(msg.num_points)
        if num_pts == 0:
            return

        data = np.array(msg.data[:num_pts * 3], dtype=np.float32).reshape(-1, 3)

        # Filter robot self-hits: discard sensor-local points within 0.8m.
        # mj_multiRay's bodyexclude only covers the root body, not child
        # bodies (limbs). Limb hits at 0.3-0.8m create persistent FP voxels
        # as the robot walks. Real obstacles are always >0.5m from sensor.
        dist_sq = data[:, 0]**2 + data[:, 1]**2
        data = data[dist_sq > 0.64]  # 0.8m squared
        if len(data) == 0:
            return

        # Snapshot IMU pose (set by game tick at 100Hz)
        with self._imu_lock:
            imu_x = self._imu_x
            imu_y = self._imu_y
            imu_z = self._imu_z
            imu_roll = self._imu_roll
            imu_pitch = self._imu_pitch
            imu_yaw = self._imu_yaw

        # Skip integration when robot is fallen — disoriented LiDAR creates
        # TSDF artifacts at wrong z-levels that persist and corrupt the costmap.
        if imu_z < 0.30:  # B2 nominal height ~0.47, fall threshold ~0.35
            return

        # Build rotation matrix once (used for both points and sensor origin)
        R = self._build_rotation(imu_roll, imu_pitch, imu_yaw)
        body_pos = np.array([imu_x, imu_y, imu_z], dtype=np.float32)

        # Transform sensor-local points to world frame, accounting for
        # the LiDAR's body-frame offset (0.34m forward, 0.18m above center).
        pts_world = self._transform_points_world(
            data, self._SENSOR_OFFSET, body_pos, R,
        )

        # Filter ground-plane artifacts: discard points below min world Z.
        # Downward LiDAR channels (-15deg) can produce near-ground hits
        # that create false voxels after 3D rotation.
        valid = pts_world[:, 2] >= self._MIN_WORLD_Z
        pts_world = pts_world[valid]

        if len(pts_world) == 0:
            return

        # Filter target marker hits: the mocap target sphere is hit by
        # LiDAR but is not an obstacle. Discard points near any historical
        # target position (old TSDF voxels persist until free rays clear them).
        r2 = self._TARGET_FILTER_RADIUS ** 2
        for tx, ty in self._target_positions:
            dx = pts_world[:, 0] - tx
            dy = pts_world[:, 1] - ty
            pts_world = pts_world[dx * dx + dy * dy > r2]
            if len(pts_world) == 0:
                return

        # Sensor world position (for DDA ray origin — rays originate from
        # the sensor, not body center).
        sensor_world = R @ self._SENSOR_OFFSET + body_pos

        # Get current SLAM pose for costmap extraction (body-frame output).
        pose = self._odometry.pose
        sensor_pose = self._Pose2D(x=pose.x, y=pose.y, yaw=pose.yaw, stamp=pose.stamp)

        # Integrate pre-transformed world-frame points into TSDF.
        # Use sensor world XY as ray origin (not body center).
        self._tsdf.integrate_scan_world(
            pts_world, float(sensor_world[0]), float(sensor_world[1]),
        )

        # Clear TSDF log-odds within robot footprint.
        # The robot's own body produces LiDAR returns that create persistent
        # occupied voxels. As the robot moves, these remain as a trail of
        # grey cubes in the viewer. Clear them at the current body position.
        tsdf = self._tsdf
        vs = tsdf.voxel_size
        cx = int((imu_x - tsdf.origin_x) / vs)
        cy = int((imu_y - tsdf.origin_y) / vs)
        mask_r = int(0.8 / vs)  # 0.8m radius (covers B2 body + limbs)
        x_lo = max(0, cx - mask_r)
        x_hi = min(tsdf.nx, cx + mask_r + 1)
        y_lo = max(0, cy - mask_r)
        y_hi = min(tsdf.ny, cy + mask_r + 1)
        if x_lo < x_hi and y_lo < y_hi:
            ix_idx, iy_idx = np.ogrid[x_lo:x_hi, y_lo:y_hi]
            footprint = (ix_idx - cx)**2 + (iy_idx - cy)**2 <= mask_r**2
            # Clear low-confidence occupied cells (0 < lo < 2.0) — likely
            # self-hits from limbs that got 1-2 hits.  Confirmed obstacles
            # (lo >= 2.0, i.e. 3+ hits) are preserved so the robot doesn't
            # forget nearby walls.  Free-space evidence (lo < 0) is also
            # preserved for convergence.
            lo_slice = tsdf._log_odds[x_lo:x_hi, y_lo:y_hi, :]
            uncertain = footprint[:, :, None] & (lo_slice > 0.0) & (lo_slice < 2.0)
            lo_slice[uncertain] = 0.0

        # --- World-frame 2D cost grid (unified authority for A* and DWA) ---
        # Use higher log-odds threshold (>1.0 = 2+ LiDAR hits) for path
        # planning.  Single hits (lo=0.85) are too noisy for A* routing.
        # DWA body-frame costmap still uses lo>0.5 for reactive safety.
        from scipy.ndimage import binary_opening, distance_transform_edt

        lo = tsdf._log_odds
        iz_lo = max(0, int((tsdf.costmap_z_lo - tsdf.z_min) / vs))
        iz_hi = min(tsdf.nz, int((tsdf.costmap_z_hi - tsdf.z_min) / vs) + 1)
        lo_slice = lo[:, :, iz_lo:iz_hi]

        occupied_2d = np.any(lo_slice > 1.0, axis=2)

        # Morphological opening: remove isolated 1-2 cell noise (self-hits,
        # target residue, edge artifacts).  Real obstacles are 3+ cells wide.
        occupied_2d = binary_opening(occupied_2d, iterations=1)

        # EDT: distance from every free cell to nearest occupied cell
        if np.any(occupied_2d):
            dist_2d = distance_transform_edt(~occupied_2d).astype(
                np.float32) * vs
        else:
            dist_2d = np.full(
                (tsdf.nx, tsdf.ny), tsdf.nx * vs, dtype=np.float32)

        truncation = self._cfg.tsdf_truncation  # 0.5m
        cost_f = 1.0 - np.clip(dist_2d / truncation, 0.0, 1.0)
        cost_u8 = (cost_f * 254).astype(np.uint8)

        # Mark unobserved cells as unknown (255)
        observed = np.any(lo_slice != 0.0, axis=2)
        cost_u8[~observed] = 255

        # Zero robot footprint (reuse same circular mask)
        if x_lo < x_hi and y_lo < y_hi:
            cost_u8[x_lo:x_hi, y_lo:y_hi][footprint] = 0

        self._world_cost_grid = cost_u8
        self._world_cost_meta = {
            'origin_x': tsdf.origin_x, 'origin_y': tsdf.origin_y,
            'voxel_size': tsdf.voxel_size,
            'nx': tsdf.nx, 'ny': tsdf.ny,
            'truncation': truncation,
        }

        # Extract body-frame costmap for DWA (extent covers full arc range)
        costmap_extent = self._cfg.tsdf_xy_extent
        cost_grid, unknown_mask = self._tsdf.extract_body_costmap(
            sensor_pose, costmap_extent=costmap_extent,
        )

        # Mask robot footprint: zero cost within 0.5m of robot origin.
        # The robot's own body produces LiDAR returns that register as
        # obstacles in the persistent TSDF, poisoning the DWA's origin
        # and causing all arcs to fail the lethal check (feas=0).
        voxel = self._cfg.tsdf_voxel_size
        n = cost_grid.shape[0]
        center = n // 2
        mask_radius_cells = int(0.5 / voxel)
        y_idx, x_idx = np.ogrid[:n, :n]
        robot_mask = (x_idx - center)**2 + (y_idx - center)**2 <= mask_radius_cells**2
        cost_grid[robot_mask] = 0.0

        # Increase cost of unknown cells: the robot shouldn't blast through
        # unseen territory at full speed. Default Layer 6 cost is 0.15 (nearly
        # free). At 0.35, arcs through unknown space are significantly penalized,
        # causing the DWA to prefer scanned routes and naturally decelerate
        # near the scanning frontier.
        _UNKNOWN_CELL_COST = 0.35
        if unknown_mask is not None:
            cost_grid[unknown_mask] = _UNKNOWN_CELL_COST

        from layer_6.types import Costmap2D
        costmap = Costmap2D(
            grid=cost_grid,
            resolution=self._cfg.tsdf_voxel_size,
            origin_x=pose.x - costmap_extent,
            origin_y=pose.y - costmap_extent,
            robot_yaw=pose.yaw,
            stamp=msg.timestamp,
            unknown_mask=unknown_mask,
        )

        query = self._CostmapQuery(costmap, oob_cost=_UNKNOWN_CELL_COST)

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
    def world_cost_grid(self) -> np.ndarray | None:
        """Latest world-frame cost grid (uint8), or None if no scan yet.

        Values: 0=free, 1-253=gradient, 254=lethal, 255=unknown.
        """
        return self._world_cost_grid

    @property
    def world_cost_meta(self) -> dict | None:
        """Metadata for world_cost_grid: origin_x, origin_y, voxel_size, nx, ny."""
        return self._world_cost_meta

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

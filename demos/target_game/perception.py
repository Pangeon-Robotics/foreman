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

        # Safety valve: consecutive feas=0 counter.  When DWA can't find
        # feasible arcs for 15+ replans, clear local TSDF voxels as
        # contradicting evidence (robot is here, space should be free).
        self._feas_zero_count = 0
        # Force-feasible flag: when True, next DWA replan should
        # guarantee at least one feasible arc (clears after read).
        self._force_feasible = False

        # Scan throttle: TSDF integration is too heavy (~80ms) to run
        # at full LiDAR rate — it holds the GIL and starves the control
        # loop.  In headless mode, firmware can flood scans at 50-100Hz
        # wall time.  Time-based throttle limits processing to max 5Hz
        # wall time regardless of incoming rate.
        self._scan_min_interval = getattr(
            perception_config, 'scan_min_interval', 0.50)  # cooldown gap AFTER build (not period)
        self._scan_last_time = 0.0

        # Shutdown flag: when True, DDS callback exits immediately.
        self._shutdown = False

        # When True, DDS callback skips TSDF integration (direct scanner
        # handles it instead). DDS still updates costmaps.
        self._direct_ready = False

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

    # ------------------------------------------------------------------
    # Direct scanner: instant raycasting bypassing DDS progressive scan
    # ------------------------------------------------------------------

    def init_direct_scanner(self, scene_xml_path: str,
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
            self._direct_model = mj_model
            self._direct_data = mj_data
        else:
            self._direct_model = mujoco.MjModel.from_xml_path(scene_xml_path)
            self._direct_data = mujoco.MjData(self._direct_model)
            mujoco.mj_forward(self._direct_model, self._direct_data)

        # Find robot root body
        site_id = mujoco.mj_name2id(
            self._direct_model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        exclude_body = -1
        if site_id >= 0:
            body_id = self._direct_model.site_bodyid[site_id]
            while self._direct_model.body_parentid[body_id] != 0:
                body_id = self._direct_model.body_parentid[body_id]
            exclude_body = body_id
        self._direct_exclude_body = exclude_body

        # Collect ALL robot geom IDs (root + child bodies)
        self._direct_robot_geoms = set()
        if exclude_body >= 0:
            bodies = [exclude_body]
            visited = set()
            while bodies:
                bid = bodies.pop()
                if bid in visited:
                    continue
                visited.add(bid)
                for g in range(self._direct_model.ngeom):
                    if self._direct_model.geom_bodyid[g] == bid:
                        self._direct_robot_geoms.add(g)
                for child in range(self._direct_model.nbody):
                    if self._direct_model.body_parentid[child] == bid:
                        bodies.append(child)

        # Target geom IDs
        self._direct_target_geoms = set()
        target_body = mujoco.mj_name2id(
            self._direct_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if target_body >= 0:
            for g in range(self._direct_model.ngeom):
                if self._direct_model.geom_bodyid[g] == target_body:
                    self._direct_target_geoms.add(g)
        self._direct_exclude_geoms = (
            self._direct_robot_geoms | self._direct_target_geoms)

        # Pre-compute ray directions (matching lidar.py defaults)
        h_angles = np.deg2rad(np.arange(0.0, 360.0, 0.18))
        v_angles_deg = [-15, -13, -11, -9, -7, -5, -3, -1,
                        1, 3, 5, 7, 9, 11, 13, 15]
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
        self._direct_ray_dirs = np.ascontiguousarray(
            np.vstack(dirs), dtype=np.float64)
        self._direct_n_rays = len(self._direct_ray_dirs)

        # Working arrays
        self._direct_geomid = np.zeros(self._direct_n_rays, dtype=np.int32)
        self._direct_dist = np.zeros(self._direct_n_rays, dtype=np.float64)

        # Sensor offset per robot
        offsets = {
            "b2": [0.34, 0.0, 0.18],
            "go2": [0.20, 0.0, 0.10],
            "go2w": [0.20, 0.0, 0.10],
            "b2w": [0.34, 0.0, 0.18],
        }
        self._direct_sensor_offset = np.array(
            offsets.get(robot, [0.34, 0.0, 0.18]), dtype=np.float64)

        self._direct_ready = True
        self._direct_scan_count = 0

    def direct_scan(self, x: float, y: float, z: float,
                    yaw: float) -> int:
        """Cast rays and integrate into TSDF. Call from game tick.

        Returns number of valid hits integrated.
        """
        import mujoco

        if not getattr(self, '_direct_ready', False):
            return 0

        # Sensor world position (yaw-only rotation, matching god-view)
        c, s = np.cos(yaw), np.sin(yaw)
        off = self._direct_sensor_offset
        sx = x + c * off[0] - s * off[1]
        sy = y + s * off[0] + c * off[1]
        sz = z + off[2]
        sensor_pos = np.array([sx, sy, sz], dtype=np.float64)

        # Rotate ray directions from sensor-local to world frame
        R_yaw = np.array([
            [c, -s, 0], [s, c, 0], [0, 0, 1],
        ], dtype=np.float64)
        rays_world = (R_yaw @ self._direct_ray_dirs.T).T
        rays_flat = np.ascontiguousarray(rays_world.flatten())

        # Cast all rays at once (no progressive scan, no motion blur)
        mujoco.mj_multiRay(
            m=self._direct_model, d=self._direct_data,
            pnt=sensor_pos, vec=rays_flat,
            geomgroup=None, flg_static=1,
            bodyexclude=self._direct_exclude_body,
            geomid=self._direct_geomid,
            dist=self._direct_dist,
            nray=self._direct_n_rays, cutoff=10.0,
        )

        # Filter valid hits
        valid = ((self._direct_dist >= 0.05)
                 & (self._direct_dist <= 10.0))
        if not np.any(valid):
            return 0

        dists = self._direct_dist[valid]
        dirs_v = rays_world[valid]
        geomids = self._direct_geomid[valid]

        # Exclude robot + target geoms
        if self._direct_exclude_geoms:
            keep = np.ones(len(geomids), dtype=bool)
            for gid in self._direct_exclude_geoms:
                keep &= geomids != gid
            dists = dists[keep]
            dirs_v = dirs_v[keep]

        if len(dists) == 0:
            return 0

        # Compute world hit points
        hits = sensor_pos + dirs_v * dists[:, np.newaxis]
        hits = hits.astype(np.float32)

        # Ground filter
        hits = hits[hits[:, 2] >= self._MIN_WORLD_Z]
        if len(hits) == 0:
            return 0

        # Target geoms already excluded by geom ID (no XY distance
        # filter needed — that's only for the DDS path where geom IDs
        # are unavailable).

        # Integrate into TSDF
        self._tsdf.integrate_scan_world(hits, float(sx), float(sy))
        self._direct_scan_count += 1

        # Build cost grids after every integration (main thread, safe).
        with self._imu_lock:
            ix, iy = self._imu_x, self._imu_y
        self._build_cost_grids(self._tsdf, ix, iy)
        query = LiveCostmapQuery(self)
        with self._lock:
            self._costmap_query = query

        return len(hits)

    def direct_scan_only(self, x: float, y: float, z: float,
                         yaw: float) -> int:
        """Raycast + TSDF integration only, NO cost grid build.

        Cost grid build is deferred to build_cost_grids_from_main()
        on a later tick to avoid blocking the control loop for 200+ms.
        """
        import mujoco

        if not getattr(self, '_direct_ready', False):
            return 0

        c, s = np.cos(yaw), np.sin(yaw)
        off = self._direct_sensor_offset
        sx = x + c * off[0] - s * off[1]
        sy = y + s * off[0] + c * off[1]
        sz = z + off[2]
        sensor_pos = np.array([sx, sy, sz], dtype=np.float64)

        R_yaw = np.array([
            [c, -s, 0], [s, c, 0], [0, 0, 1],
        ], dtype=np.float64)
        rays_world = (R_yaw @ self._direct_ray_dirs.T).T
        rays_flat = np.ascontiguousarray(rays_world.flatten())

        mujoco.mj_multiRay(
            m=self._direct_model, d=self._direct_data,
            pnt=sensor_pos, vec=rays_flat,
            geomgroup=None, flg_static=1,
            bodyexclude=self._direct_exclude_body,
            geomid=self._direct_geomid,
            dist=self._direct_dist,
            nray=self._direct_n_rays, cutoff=10.0,
        )

        valid = ((self._direct_dist >= 0.05)
                 & (self._direct_dist <= 10.0))
        if not np.any(valid):
            return 0

        dists = self._direct_dist[valid]
        dirs_v = rays_world[valid]
        geomids = self._direct_geomid[valid]

        if self._direct_exclude_geoms:
            keep = np.ones(len(geomids), dtype=bool)
            for gid in self._direct_exclude_geoms:
                keep &= geomids != gid
            dists = dists[keep]
            dirs_v = dirs_v[keep]

        if len(dists) == 0:
            return 0

        hits = sensor_pos + dirs_v * dists[:, np.newaxis]
        hits = hits.astype(np.float32)
        hits = hits[hits[:, 2] >= self._MIN_WORLD_Z]
        if len(hits) == 0:
            return 0

        self._tsdf.integrate_scan_world(hits, float(sx), float(sy))
        self._direct_scan_count += 1
        return len(hits)

    def build_cost_grids_from_main(self) -> None:
        """Build cost grids from TSDF. Call from game tick on a staggered phase.

        This is the expensive operation (~65-100ms for EDT) that was
        previously bundled into direct_scan(), blocking the control loop.
        """
        if self._tsdf is None:
            return
        with self._imu_lock:
            ix, iy = self._imu_x, self._imu_y
        self._build_cost_grids(self._tsdf, ix, iy)
        query = LiveCostmapQuery(self)
        with self._lock:
            self._costmap_query = query

    def shutdown(self) -> None:
        """Stop the DDS callback from processing new scans."""
        self._shutdown = True

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
    _MIN_WORLD_Z = 0.30

    def on_point_cloud(self, msg) -> None:
        """DDS callback: new point cloud received.

        Runs in the DDS listener thread. Integrates scan into persistent
        TSDF and extracts a body-frame costmap for the DWA planner.

        Uses full 3D IMU rotation (roll/pitch/yaw) to transform sensor-local
        LiDAR points to world frame before TSDF integration. Accounts for
        the sensor's body-frame offset (LiDAR is 0.34m forward, 0.18m above
        body center on B2).
        """
        if self._shutdown:
            return

        # When direct scanner is active, skip DDS callback entirely.
        # Cost grids are built in direct_scan() (main thread) to avoid
        # thread-safety issues with concurrent TSDF access.
        if self._direct_ready:
            return

        # Time-based throttle: skip scans that arrive too fast (wall time).
        # In headless mode, firmware floods scans at 50-100Hz wall time.
        # TSDF integration (~80ms) would starve the control loop.
        now = time.monotonic()
        if self._scan_min_interval > 0:
            if now - self._scan_last_time < self._scan_min_interval:
                return
            # Note: last_time is set AFTER build completes (_scan_finish_time)
            # so the interval is a cooldown gap, not a period.  This guarantees
            # the control loop gets at least scan_min_interval of GIL time
            # between builds.

        t0 = now

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

        # No subsampling: hit-only integration is O(N), full scans
        # (~2000 pts) take <15ms.  TSDF decay (0.25/scan) requires
        # ~2 hits/voxel/scan for net convergence — subsampling below
        # 1000 causes net decay and near-zero completeness.

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

        # Yield GIL between TSDF integration and cost grid build.
        # The main thread needs the GIL at 100Hz to send GaitParams.
        # Without this yield, 100-165ms continuous GIL hold causes
        # gait parameter jumps when the main thread resumes.
        time.sleep(0.005)

        tsdf = self._tsdf

        self._build_cost_grids(tsdf, imu_x, imu_y)

        # --- Live costmap query for DWA ---
        # Instead of building a body-frame snapshot (which goes stale
        # between LiDAR callbacks), DWA samples the world-frame grid
        # on-the-fly using the current robot pose.  LiveCostmapQuery
        # reads _dwa_cost_grid and _imu_yaw at query time, so the
        # costmap is never stale.
        query = LiveCostmapQuery(self)

        with self._lock:
            self._costmap_query = query
            self._last_cloud_time = msg.timestamp

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._build_times.append(elapsed_ms)
        # Set last_time AFTER build completes so the interval is a cooldown
        # gap that guarantees the control loop GIL time between builds.
        self._scan_last_time = time.monotonic()
        if elapsed_ms > 150:
            n = len(self._build_times)
            cg_ms = getattr(self, '_last_cg_ms', 0)
            astar_ms = getattr(self, '_last_astar_ms', 0)
            print(f"  [perc] build#{n} {elapsed_ms:.0f}ms "
                  f"(cg={cg_ms:.0f} astar={astar_ms:.0f}) "
                  f"chunks={self._tsdf.n_chunks}")

    # ------------------------------------------------------------------
    # Cost grid builder
    # ------------------------------------------------------------------

    def _build_cost_grids(self, tsdf, imu_x: float,
                          imu_y: float) -> None:
        """Build cost grids from TSDF.

        TSDF has built-in persistence (log-odds accumulation) so the
        obstacle memory layer is NOT used — it amplifies noise into phantom
        obstacles that oscillate DWA between feas=41 and feas=0.

        The TSDF cost grid (EDT-based) feeds both DWA and A* directly.
        """
        out_res = getattr(self._cfg, 'tsdf_output_resolution', 0.05)
        truncation = self._cfg.tsdf_truncation

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
        # No memory overlay needed — TSDF persistence handles it.
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
        self._dwa_cost_grid = dwa_u8

        # Yield GIL before A* grid build
        time.sleep(0.005)

        # --- A* cost grid: TSDF + inflation ---
        # A* inflation EDT (~65ms) is rate-limited to every 5th build.
        # A* paths change slowly (waypoint update rate), so stale-by-4
        # builds (~1s at 5Hz scan) is acceptable.
        _t_a0 = time.monotonic()
        if not hasattr(self, '_astar_build_counter'):
            self._astar_build_counter = 0
        self._astar_build_counter += 1
        astar_changed = (not hasattr(self, '_astar_cache_id')
                         or self._astar_build_counter % 5 == 0)
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
            self._astar_base = astar_u8

        astar_u8 = self._astar_base.copy()

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
        self._world_cost_grid = astar_u8
        self._last_cg_ms = (_t_cg1 - _t_cg0) * 1000
        self._last_astar_ms = (time.monotonic() - _t_a0) * 1000

        self._world_cost_meta = {
            'origin_x': tsdf.origin_x, 'origin_y': tsdf.origin_y,
            'voxel_size': vs,
            'nx': out_nx, 'ny': out_ny,
            'truncation': truncation,
        }

    def report_dwa_feas(self, n_feasible: int) -> None:
        """Report DWA feasibility count (called at 20Hz replan rate).

        When n_feasible == 0 for 15+ consecutive replans (0.75s), clear
        local TSDF voxels as contradicting evidence — the robot is
        physically present at this location, so nearby obstacle voxels
        are likely phantom.  If a real obstacle exists, the next LiDAR
        scan will repopulate it instantly (lo_hit=3.0).
        """
        if n_feasible == 0:
            self._feas_zero_count += 1
            if self._feas_zero_count >= 15:
                self._clear_local_tsdf()
                self._feas_zero_count = 0
        else:
            self._feas_zero_count = 0

    def _clear_local_tsdf(self) -> None:
        """Clear TSDF voxels near robot (contradicting evidence).

        The robot's presence at a location is evidence that the space
        is traversable.  Clears log_odds and obs_count in all TSDF
        chunks within 2.5m, then forces one feasible DWA arc.
        """
        tsdf = self._tsdf
        with self._imu_lock:
            rx, ry = self._imu_x, self._imu_y
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
        self._force_feasible = True

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

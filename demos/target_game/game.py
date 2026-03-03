"""Target game state machine.

Core TargetGame class with state machine, SLAM integration, and pose
helpers. Navigation methods are provided by:
  - NavigatorMixin (heading-based and wheeled) in navigator.py
  - DWANavigatorMixin (DWA planning) in dwa_nav.py
  - DWAControlMixin (DWA gait conversion) in dwa_control.py
  - ScoringMixin (lifecycle, scoring, analysis) in scoring.py

Configuration constants, GameState, GameStatistics, and configure_for_robot
live in game_config.py. All names are re-exported here for backward
compatibility.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .game_config import (
    BODY_HEIGHT, CONTROL_DT, FALL_CONFIRM_TICKS, FALL_THRESHOLD,
    GameState, GameStatistics,
    GAIT_FREQ, KP_FULL, KP_START, KD_FULL, KD_START,
    NOMINAL_BODY_HEIGHT, REACH_DISTANCE, ROBOT_DEFAULTS,
    STARTUP_RAMP_SECONDS, STARTUP_SETTLE_STEPS,
    TARGET_TIMEOUT_STEPS, TELEMETRY_INTERVAL, TURN_FREQ,
    WHEEL_HOME_Q, WHEELED,
    configure_for_robot,
)
from .navigator import NavigatorMixin
from .dwa_nav import DWANavigatorMixin
from .dwa_control import DWAControlMixin
from .scoring import ScoringMixin
from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, quat_to_rpy as _quat_to_rpy

if TYPE_CHECKING:
    from layer_6.slam.odometry import Odometry

# Re-export everything from game_config for backward compatibility.
from .game_config import *  # noqa: F401,F403

_ROBOT_TSDF_FILE = "/tmp/robot_view_tsdf.bin"
_ROBOT_COSTMAP_FILE = "/tmp/robot_view_costmap.bin"


def _write_robot_tsdf_file(tsdf, display_resolution: float = 0.05) -> None:
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
            with open(_ROBOT_TSDF_FILE, 'wb') as f:
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
        with open(_ROBOT_TSDF_FILE, 'wb') as f:
            f.write(buf)
    except OSError:
        pass


def _write_robot_costmap_file(tsdf, display_resolution: float = 0.05,
                              cost_floor: float = 0.20) -> None:
    """Write robot-view costmap built from TSDF surface history.

    Uses the same surface voxels as the grey cube visualization
    (include_history=True), so the costmap always matches visible cubes.
    Runs a cropped EDT for speed.

    Cost = 1 / (1 + dist^2) — inverse-square falloff, no distance cutoff.
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
    # 1/(1+d^2) = floor → d = sqrt(1/floor - 1)
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

    # EDT → inverse-square cost with scale factor for steep dropoff
    dist = distance_transform_edt(~occupied).astype(np.float32) * res
    scale = 0.10  # characteristic distance (m) — cost halves at this distance
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
        with open(_ROBOT_COSTMAP_FILE, 'wb') as f:
            f.write(buf)
            f.write(np.ascontiguousarray(grid_t).tobytes())
    except OSError:
        pass


class TargetGame(
    NavigatorMixin,
    DWANavigatorMixin,
    DWAControlMixin,
    ScoringMixin,
):
    """Random target navigation game.

    Runs a state machine that spawns targets and drives the robot toward
    them using L4 GaitParams directly (same path as GA training).
    """

    def __init__(
        self,
        sim,
        L4GaitParams=None,
        make_low_cmd=None,
        stamp_cmd=None,
        num_targets: int = 5,
        min_dist: float = 3.0,
        max_dist: float = 6.0,
        reach_threshold: float = 0.5,
        timeout_steps: int = TARGET_TIMEOUT_STEPS,
        seed: int | None = None,
        angle_range: tuple[float, float] | list[tuple[float, float]] = (
            -math.pi / 2, math.pi / 2),
        odometry: Odometry | None = None,
    ):
        self._sim = sim
        self._L4GaitParams = L4GaitParams
        self._make_low_cmd = make_low_cmd
        self._stamp_cmd = stamp_cmd
        self._num_targets = num_targets
        self._reach_threshold_base = reach_threshold
        self._reach_threshold = reach_threshold
        self._timeout_steps = timeout_steps
        self._angle_range = angle_range

        # SLAM odometry
        self._odometry = odometry
        self._slam_trail: list[tuple[float, float]] = []
        self._truth_trail: list[tuple[float, float]] = []
        self._pose_pub = None
        self._PoseMsg = None
        self._pose_stamp = None
        self._perception = None
        self._dwa_planner = None
        self._last_dwa_result = None
        self._telemetry = None
        self._path_critic = None

        # Avoidance commitment
        self._avoidance_sign = 0
        self._avoidance_since = 0.0
        self._avoidance_dist = 0.0
        self._goal_bearing = 0.0

        # DWA smoothing
        self._smooth_dwa_turn = 0.0
        self._smooth_dwa_fwd = 1.0

        # Regression tracking
        self._dist_ring = []
        self._recent_regression = 0.0

        # Fall detection
        self._fall_tick_count = 0
        self._post_fall_settle = 0
        self._stabilize_countdown = 0

        # Gait parameter smoothing (EMA)
        self._smooth_heading_mod = 1.0
        self._smooth_wz = 0.0
        self._decel_tick_count = 0
        self._last_threat_time = 0.0

        # SLAM drift detection
        self._min_target_dist = float('inf')
        self._slam_drift_latched = False

        # Gains
        self._kp = KP_START
        self._kd = KD_START

        self._spawner = TargetSpawner(
            min_distance=min_dist, max_distance=max_dist,
            reach_threshold=reach_threshold, seed=seed)
        self._spawn_fn = None

        self._state = GameState.STARTUP
        self._stats = GameStatistics()
        self._target_index = 0
        self._step_count = 0
        self._target_step_count = 0

        # Wheeled mode
        self._home_q = WHEEL_HOME_Q if WHEELED else None

        # Debug viewer
        self._debug_server = None
        self._god_view_costmap: tuple | None = None  # (grid_u8, meta) from scene XML
        self._god_view_path_file: str | None = None  # temp file for MuJoCo overlay
        self._god_view_tsdf = None  # GodViewTSDF instance for perfect raycast

        # Obstacle proximity
        self._obstacle_bodies: list[str] = []

        # Occupancy accuracy (periodic cache)
        self._scene_xml_path: str | None = None
        self._cached_occ: dict | None = None
        self._occ_compute_step: int = -999

        # TIP state
        self._step_and_shift = None
        self._in_tip_mode = False
        self._tip_start_step = 0
        self._tip_start_heading = 0.0
        self._tip_cooldown_until = 0

        # A* waypoint guidance
        self._current_waypoint = None
        self._wp_commit_until = 0

        # Stuck detection
        self._stuck_check_dist = float('inf')
        self._stuck_check_step = 0
        self._stuck_recovery_countdown = 0
        self._stuck_recovery_wz = 0.0
        self._prev_no_progress = False

        # Progress-gated early timeout
        self._progress_window_dist = float('inf')
        self._progress_window_step = 0

        # Regression tracking
        self._closest_approach = float('inf')
        self._reg_start_step = 0

        # Pitch/roll diagnostics
        self._rp_log = []

        # Pose glitch rejection
        self._cached_pose: (
            tuple[float, float, float, float, float, float] | None
        ) = None

    # --- Pose helpers ---

    def _get_robot_pose(self):
        """Return (x, y, yaw, z, roll, pitch) from ground truth.

        Includes glitch rejection: jumps > 2m return cached pose.
        """
        body = self._sim.get_body("base")
        x = float(body.pos[0])
        y = float(body.pos[1])
        z = float(body.pos[2])
        yaw = _quat_to_yaw(body.quat)
        roll, pitch, _ = _quat_to_rpy(body.quat)
        pose = (x, y, yaw, z, roll, pitch)
        if self._cached_pose is not None:
            dx = x - self._cached_pose[0]
            dy = y - self._cached_pose[1]
            if math.sqrt(dx * dx + dy * dy) > 2.0:
                return self._cached_pose
        self._cached_pose = pose
        return pose

    def _update_slam(self, truth_yaw: float) -> None:
        """Feed SLAM odometry and IMU pose to perception pipeline."""
        if self._odometry is None:
            return
        from types import SimpleNamespace
        l5_state = self._sim.get_state()
        imu_quat = l5_state.imu_quaternion
        body = self._sim.get_body("base")
        vx_world = float(body.linvel[0])
        vy_world = float(body.linvel[1])
        c = math.cos(truth_yaw)
        s = math.sin(truth_yaw)
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world
        state = SimpleNamespace(
            imu_quaternion=imu_quat,
            body_velocity=(vx_body, vy_body),
            timestamp=(l5_state.timestamp if l5_state.timestamp > 0
                       else self._step_count * CONTROL_DT),
        )
        self._odometry.update(state, CONTROL_DT)
        if self._pose_pub is not None:
            self._publish_pose()

        # Feed full 6-DOF pose to perception for IMU-corrected LiDAR transforms.
        # Uses ground-truth MuJoCo body pose (x, y, z, roll, pitch, yaw).
        if self._perception is not None:
            x, y, yaw, z, roll, pitch = self._get_robot_pose()
            self._perception.set_imu_pose(x, y, z, roll, pitch, yaw)

    # --- Setters ---

    def set_dwa_planner(self, planner) -> None:
        self._dwa_planner = planner

    _PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"

    def set_path_critic(self, critic) -> None:
        self._path_critic = critic

    def set_scene_path(self, path: str) -> None:
        """Set scene XML path for occupancy accuracy computation."""
        self._scene_xml_path = path

    def set_god_view_costmap(self, grid: 'np.ndarray', meta: dict) -> None:
        """Set god-view binary costmap for Godot overlay (red = obstacle footprint)."""
        self._god_view_costmap = (grid, meta)

    def set_obstacle_bodies(self, names: list[str]) -> None:
        self._obstacle_bodies = names

    def set_pose_publisher(self, pub, msg_class, stamp_fn) -> None:
        self._pose_pub = pub
        self._PoseMsg = msg_class
        self._pose_stamp = stamp_fn

    def set_telemetry(self, telemetry_log) -> None:
        self._telemetry = telemetry_log

    def _publish_pose(self) -> None:
        if self._odometry is None or self._pose_pub is None:
            return
        p = self._odometry.pose
        msg = self._PoseMsg(
            x=p.x, y=p.y, yaw=p.yaw, timestamp=p.stamp, crc=0)
        self._pose_stamp(msg)
        self._pose_pub.Write(msg)

    def _get_occ_str(self) -> str:
        """Return formatted occupancy accuracy string, recomputing every 500 ticks."""
        if (self._scene_xml_path is None
                or self._perception is None):
            return ""
        # Recompute every 2000 ticks (20s at 100Hz).  3DS computations
        # (KD tree build + chunk iteration) block the main thread for
        # 50-100ms, preventing GaitParams sending.  Less frequent updates
        # reduce GIL spikes that cause falls.
        if self._step_count - self._occ_compute_step >= 2000:
            try:
                from .test_occupancy import compute_3ds_v2, compute_3ds_god
                tsdf = self._perception._tsdf
                if tsdf is not None:
                    self._cached_occ = compute_3ds_v2(
                        tsdf, self._scene_xml_path)
                    self._cached_occ['god'] = compute_3ds_god(
                        tsdf, self._scene_xml_path)
                    self._occ_compute_step = self._step_count
            except Exception:
                pass
        if self._cached_occ is not None:
            o = self._cached_occ
            adh = o['adherence_mm']
            cpl = o['completeness_pct']
            phn = o['phantom_pct']
            parts = f" 3DS: adh={adh:.1f}mm cpl={cpl:.1f}% phn={phn:.1f}%"
            god = o.get('god')
            if god is not None:
                parts += f" god={god['score']:.0f}"
            return parts
        return ""

    def _gt_clearance(self) -> float:
        """Minimum ground-truth distance from robot surface to nearest obstacle."""
        if not self._obstacle_bodies:
            return float('inf')
        robot = self._sim.get_body("base")
        if robot is None:
            return float('inf')
        rx, ry = float(robot.pos[0]), float(robot.pos[1])
        min_d = float('inf')
        for name in self._obstacle_bodies:
            obs = self._sim.get_body(name)
            if obs is None:
                continue
            d = math.sqrt(
                (rx - float(obs.pos[0]))**2 + (ry - float(obs.pos[1]))**2)
            if d < min_d:
                min_d = d
        # Subtract robot radius (0.35m) + obstacle half-width (~0.25m)
        return max(0.0, min_d - 0.60)

    def _get_nav_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) for navigation (always ground truth)."""
        x, y, yaw, _, _, _ = self._get_robot_pose()
        return x, y, yaw

    def _get_tip_scales(self):
        if self._step_and_shift is None:
            from step_and_shift import StepAndShift
            self._step_and_shift = StepAndShift(gait_freq=TURN_FREQ)
        return self._step_and_shift.update(CONTROL_DT)

    def _reset_tip(self):
        if self._step_and_shift is not None:
            self._step_and_shift.reset()

    @property
    def slam_trail(self) -> list[tuple[float, float]]:
        return self._slam_trail

    @property
    def truth_trail(self) -> list[tuple[float, float]]:
        return self._truth_trail

    # --- Core ---

    def _check_fall(self, z: float) -> bool:
        """Check for sustained fall. Returns True if fallen."""
        if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD:
            self._fall_tick_count += 1
            if self._fall_tick_count >= FALL_CONFIRM_TICKS:
                self._stats.falls += 1
                self._fall_tick_count = 0
                self._post_fall_settle = 500
                print(f"FALL DETECTED at z={z:.3f}m "
                      f"(sustained {FALL_CONFIRM_TICKS} ticks)")
                self._state = GameState.SPAWN_TARGET
                return True
        else:
            self._fall_tick_count = 0
        return False

    def _send_l4(self, params):
        """Send L4 GaitParams directly to Layer 4."""
        try:
            self._sim.send_command(
                params, self._sim._t, kp=self._kp, kd=self._kd)
            self._sim._t += CONTROL_DT
        except RuntimeError:
            print("Simulation stopped unexpectedly")
            self._state = GameState.DONE

    def tick(self) -> bool:
        """Advance one control step. Returns False when done."""
        if self._state == GameState.DONE:
            return False

        self._step_count += 1
        self._update_gains()

        # Update SLAM odometry
        _, _, truth_yaw, _, _, _ = self._get_robot_pose()
        self._update_slam(truth_yaw)

        # Record trails for visualization (10Hz)
        if self._odometry is not None and self._step_count % 10 == 0:
            truth_x, truth_y, _, _, _, _ = self._get_robot_pose()
            self._truth_trail.append((truth_x, truth_y))
            p = self._odometry.pose
            self._slam_trail.append((p.x, p.y))
            if self._telemetry is not None:
                drift = math.sqrt(
                    (p.x - truth_x)**2 + (p.y - truth_y)**2)
                self._telemetry.record("slam", {
                    "drift": round(drift, 4),
                    "slam_x": round(p.x, 3),
                    "slam_y": round(p.y, 3),
                    "truth_x": round(truth_x, 3),
                    "truth_y": round(truth_y, 3),
                })

        # Debug viewer: stream state at 10Hz, TSDF at 2Hz
        if self._debug_server is not None and self._debug_server.has_client:
            # Send ground-truth obstacle volumes once per connection
            geoms = getattr(self, '_obstacle_geoms', None)
            if geoms and not getattr(self, '_obstacles_sent', False):
                self._debug_server.send_obstacles(geoms)
                self._obstacles_sent = True
            if self._step_count % 10 == 0:
                x, y, yaw, z, roll, pitch = self._get_robot_pose()
                # Send ground-truth pose to Godot viewer (not SLAM).
                # TSDF voxels, cost map, obstacles, and target are all in
                # world frame — robot must match to avoid visual offset.
                slam_x, slam_y, slam_yaw = x, y, yaw
                target = self._spawner.current_target
                tx = target.x if target else 0.0
                ty = target.y if target else 0.0
                joints = [0.0] * 12
                try:
                    st = self._sim.get_state()
                    joints = list(st.joint_positions[:12])
                except Exception:
                    pass
                dwa = self._last_dwa_result
                self._debug_server.send_robot_state(
                    slam_x, slam_y, slam_yaw, z, roll, pitch,
                    joints, tx, ty,
                    self._state.value,
                    dwa.forward if dwa else 0.0,
                    dwa.turn if dwa else 0.0,
                    dwa.n_feasible if dwa else 0,
                    0.0, 0.0,
                    self._in_tip_mode,
                )
            if self._step_count % 50 == 0 and self._perception is not None:
                tsdf = self._perception._tsdf
                if tsdf is not None:
                    self._debug_server.send_tsdf(tsdf)
                # Send robot-view costmap (red in Godot)
                cost_grid = self._perception.world_cost_grid
                meta = self._perception.world_cost_meta
                if cost_grid is not None and meta is not None:
                    self._debug_server.send_costmap_2d(
                        cost_grid, meta['origin_x'], meta['origin_y'],
                        meta['voxel_size'])
                # Send god-view costmap (blue in Godot)
                if self._god_view_costmap is not None:
                    gv_grid, gv_meta = self._god_view_costmap
                    self._debug_server.send_god_view_costmap(
                        gv_grid, gv_meta['origin_x'], gv_meta['origin_y'],
                        gv_meta['voxel_size'])
                    # God-view A* path (blue dots in Godot)
                    target = self._spawner.current_target
                    if target is not None:
                        gv_x, gv_y, _, _, _, _ = self._get_robot_pose()
                        god_path = _astar_on_god_view(
                            gv_grid, gv_meta, (gv_x, gv_y),
                            (target.x, target.y))
                        if god_path is not None:
                            self._debug_server.send_god_view_path(god_path)

        # God-view A* path for MuJoCo overlay (written to temp file)
        if (self._god_view_path_file is not None
                and self._god_view_costmap is not None
                and self._step_count % 50 == 0):
            import struct as _struct
            gv_grid, gv_meta = self._god_view_costmap
            target = self._spawner.current_target
            if target is not None:
                gv_x, gv_y, _, _, _, _ = self._get_robot_pose()
                god_path = _astar_on_god_view(
                    gv_grid, gv_meta, (gv_x, gv_y),
                    (target.x, target.y))
                if god_path is not None:
                    buf = bytearray(len(god_path) * 8)
                    for i, (wx, wy) in enumerate(god_path):
                        _struct.pack_into('ff', buf, i * 8, wx, wy)
                    try:
                        with open(self._god_view_path_file, 'wb') as _f:
                            _f.write(buf)
                    except OSError:
                        pass

        # --- Staggered perception work ---
        # Spread expensive operations across different ticks to avoid
        # blocking the control loop for 200-300ms on a single tick.
        # Each operation runs at 4Hz (every 25 ticks) but offset.
        # Previous 50-tick cycle (2Hz) let robot move 1.14m between
        # scans at v_theory=1.14 m/s — outrunning perception.
        _phase = self._step_count % 25

        # Phase 0: God-view + DirectScanner raycasts (same pose for F1 scoring)
        # Raycast is fast (~5ms each via mj_multiRay C call).
        # Cost grid build is DEFERRED to phase 10.
        if _phase == 0:
            _scan_x, _scan_y, _scan_yaw, _scan_z, _, _ = self._get_robot_pose()

            if self._god_view_tsdf is not None:
                self._god_view_tsdf.update(_scan_x, _scan_y, _scan_yaw, _scan_z)
                self._god_view_tsdf.write_temp_file("/tmp/god_view_tsdf.bin")

            if (self._perception is not None
                    and getattr(self._perception, '_direct_ready', False)):
                self._perception.direct_scan_only(
                    _scan_x, _scan_y, _scan_z, _scan_yaw)

        # Phase 10: Cost grid build (~65-100ms — the expensive part).
        # Runs every OTHER cycle (2Hz) to limit GIL blocking.
        # Scans run at 4Hz above so TSDF has fresh data.
        _cycle = self._step_count // 25
        if _phase == 10 and _cycle % 2 == 0 and self._perception is not None:
            self._perception.build_cost_grids_from_main()

        # Phase 15: Path critic update + viewer file writes
        if _phase == 15:
            if (self._perception is not None
                    and self._path_critic is not None):
                cost_grid = self._perception.world_cost_grid
                meta = self._perception.world_cost_meta
                if cost_grid is not None and meta is not None:
                    self._path_critic.set_world_cost(
                        cost_grid, meta['origin_x'], meta['origin_y'],
                        meta['voxel_size'],
                        truncation=meta.get('truncation', 0.5))
            # Viewer file writes (headed mode only)
            if (not self._headless
                    and self._perception is not None
                    and self._perception._tsdf is not None):
                _write_robot_tsdf_file(self._perception._tsdf)
                _write_robot_costmap_file(self._perception._tsdf)

        # Dynamic reach threshold
        if self._state == GameState.WALK_TO_TARGET:
            t = self._target_step_count * CONTROL_DT
            self._reach_threshold = min(
                self._reach_threshold_base + t * 0.02,
                self._reach_threshold_base + 1.0)

        if self._state == GameState.STARTUP:
            self._tick_startup()
        elif self._state == GameState.SPAWN_TARGET:
            self._tick_spawn()
        elif self._state == GameState.WALK_TO_TARGET:
            if WHEELED:
                self._tick_walk_wheeled()
            elif (self._dwa_planner is not None
                    and self._perception is not None):
                self._tick_walk_dwa()
            else:
                self._tick_walk()

        return self._state != GameState.DONE

    def _update_gains(self):
        """Smoothly ramp PD gains over first 2.5s."""
        if self._kp >= KP_FULL:
            return
        t = self._step_count * CONTROL_DT
        progress = min(1.0, t / STARTUP_RAMP_SECONDS)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        self._kp = KP_START + alpha * (KP_FULL - KP_START)
        self._kd = KD_START + alpha * (KD_FULL - KD_START)


def _astar_on_god_view(
    grid, meta, start: tuple[float, float], goal: tuple[float, float],
    robot_radius: float = 0.525,
) -> list[tuple[float, float]] | None:
    """Run A* on a god-view cost grid and return world-frame waypoints.

    robot_radius defaults to 0.525m = 1.5 × B2 half-width, so A* avoids
    corridors narrower than 1.05m (1.5 × robot diameter).
    """
    from .path_critic import PathCritic
    critic = PathCritic.__new__(PathCritic)
    critic._cost_grid = grid
    critic._cost_origin_x = meta['origin_x']
    critic._cost_origin_y = meta['origin_y']
    critic._cost_voxel_size = meta['voxel_size']
    critic._cost_truncation = meta.get('truncation', 0.5)
    critic._robot_radius = robot_radius
    return critic._astar_on_cost_grid(start, goal, return_path=True)

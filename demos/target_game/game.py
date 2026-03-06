"""Target game state machine.

Core TargetGame class with state machine, SLAM integration, and pose
helpers. Navigation and scoring are provided by composed helpers:
  - Navigator (heading-based and wheeled) in navigator_helper.py
  - GameScoring (lifecycle, scoring, analysis) in game_scoring.py

Configuration constants live in game_config.py; all names are re-exported
here for backward compatibility. Robot-view code must NEVER access god-view
data (GodViewTSDF, god_view_costmap, scene XML geometry).
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
from .navigator_helper import Navigator
from .game_scoring import GameScoring
from .game_viz import (
    stream_debug_viewer, write_god_view_path,
    tick_perception, tick_slam_trails,
    get_occ_str as _get_occ_str_fn, gt_clearance as _gt_clearance_fn,
)
from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, quat_to_rpy as _quat_to_rpy

if TYPE_CHECKING:
    from layer_6.slam.odometry import Odometry

# Re-export everything from game_config for backward compatibility.
from .game_config import *  # noqa: F401,F403


class TargetGame:
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

        self._odometry = odometry
        self._slam_trail: list[tuple[float, float]] = []
        self._truth_trail: list[tuple[float, float]] = []
        self._pose_pub = None
        self._PoseMsg = None
        self._pose_stamp = None
        self._perception = None
        self._telemetry = None
        self._path_critic = None
        self._fall_tick_count = 0
        self._post_fall_settle = 0
        self._min_target_dist = float('inf')
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
        self._home_q = WHEEL_HOME_Q if WHEELED else None
        self._debug_server = None
        self._god_view_costmap: tuple | None = None
        self._god_view_path_file: str | None = None
        self._god_view_tsdf = None
        self._obstacle_bodies: list[str] = []
        self._scene_xml_path: str | None = None
        self._cached_occ: dict | None = None
        self._occ_compute_step: int = -999
        self._step_and_shift = None
        self._current_waypoint = None  # DEPRECATED — navigation uses _committed_path
        self._committed_path = None
        self._committed_path_step = 0
        self._rp_log = []
        self._cached_pose: (
            tuple[float, float, float, float, float, float] | None
        ) = None
        self._glitch_ticks = 0

        # --- Composed helpers ---
        self.nav = Navigator(self)
        self.scoring = GameScoring(self)

    _PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"

    # --- Pose helpers ---

    def _get_robot_pose(self):
        """Return (x, y, yaw, z, roll, pitch) from ground truth.

        Includes glitch rejection: jumps > 2m are rejected for up to
        0.5s (50 ticks at 100Hz).  If the new position persists beyond
        the grace period, it's accepted as real (fall recovery, collision).
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
                if self._glitch_ticks < 50:
                    self._glitch_ticks += 1
                    return self._cached_pose
                # Grace period expired — accept new position
        self._glitch_ticks = 0
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

        if self._perception is not None:
            _, _, _, z, roll, pitch = self._get_robot_pose()
            p = self._odometry.pose
            self._perception.set_imu_pose(p.x, p.y, z, roll, pitch, p.yaw)

    # --- Setters ---

    def set_path_critic(self, critic) -> None:
        self._path_critic = critic

    def set_scene_path(self, path: str) -> None:
        self._scene_xml_path = path

    def set_god_view_costmap(self, grid, meta: dict) -> None:
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
        return _get_occ_str_fn(self)

    def _gt_clearance(self) -> float:
        return _gt_clearance_fn(self)

    def _get_nav_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) for navigation -- ground truth.

        SLAM odometry drifts too much for heading-proportional control
        (6m+ position drift, 40°+ yaw noise).  SLAM is still used for
        perception (TSDF scan integration) via _update_slam().
        """
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
            rc = None
            try:
                fw = self._sim._firmware
                if fw is not None and fw._proc is not None:
                    rc = fw._proc.poll()
            except Exception:
                pass
            print(f"Simulation stopped unexpectedly (firmware exit={rc})")
            self._state = GameState.DONE

    # --- Public API forwarding ---

    def run(self) -> GameStatistics:
        """Run the full game loop and return statistics."""
        return self.scoring.run()

    # --- Tick ---

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
        tick_slam_trails(self)

        # Debug viewer: stream state at 10Hz, TSDF at 2Hz
        if self._debug_server is not None and self._debug_server.has_client:
            stream_debug_viewer(self)

        # God-view A* path for MuJoCo overlay (written to temp file)
        write_god_view_path(self)

        # Staggered perception work (god-view, scan, cost grid, path critic)
        tick_perception(self)

        # Dynamic reach threshold
        if self._state == GameState.WALK_TO_TARGET:
            t = self._target_step_count * CONTROL_DT
            self._reach_threshold = min(
                self._reach_threshold_base + t * 0.02,
                self._reach_threshold_base + 1.0)

        if self._state == GameState.STARTUP:
            self.scoring.tick_startup()
        elif self._state == GameState.SPAWN_TARGET:
            self.scoring.tick_spawn()
        elif self._state == GameState.WALK_TO_TARGET:
            if WHEELED:
                self.nav.tick_walk_wheeled()
            else:
                self.nav.tick_walk_heading()

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

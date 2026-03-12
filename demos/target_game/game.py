"""Target game state machine.

Core TargetGame class with state machine, SLAM integration, and pose
helpers. Navigation and scoring are provided by composed helpers:
  - Navigator (heading-based and wheeled) in navigator_helper.py
  - GameScoring (lifecycle, scoring, analysis) in game_scoring.py

Configuration constants live in game_config.py; all names are re-exported
here for backward compatibility. Robot-view code must NEVER access god-view
data (GodViewTSDF, god_view_costmap, scene XML geometry).

Layer compliance: legged robots send MotionCommands through Layer 5's
send_motion_command(). Layer 5 handles gait selection, gain scheduling,
and Layer 4 dispatch. Wheeled robots bypass L5 (documented violation).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .game_config import (
    CONTROL_DT, FALL_CONFIRM_TICKS, FALL_THRESHOLD,
    GameState, GameStatistics,
    NOMINAL_BODY_HEIGHT, REACH_DISTANCE, ROBOT_DEFAULTS,
    STARTUP_SETTLE_STEPS,
    TARGET_TIMEOUT_STEPS, TELEMETRY_INTERVAL,
    WHEELED,
    configure_for_robot,
)
from .navigator_helper import Navigator
from .game_scoring import GameScoring
from .telemetry import GameTelemetry
from .game_viz import (
    stream_debug_viewer, write_god_view_path,
    tick_perception, tick_slam_trails,
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
    them. Legged robots use Layer 5 MotionCommands (L5 handles gait
    selection, gains, L4 dispatch). Wheeled robots bypass L5.
    """

    def __init__(
        self,
        sim,
        robot: str = 'b2',
        num_targets: int = 5,
        min_dist: float = 3.0,
        max_dist: float = 6.0,
        reach_threshold: float = REACH_DISTANCE,
        timeout_steps: int = TARGET_TIMEOUT_STEPS,
        seed: int | None = None,
        angle_range: tuple[float, float] | list[tuple[float, float]] = (
            -math.pi / 2, math.pi / 2),
        odometry: 'Odometry | None' = None,
    ):
        self._sim = sim
        self._robot = robot
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
        self._post_reach_settle = 0
        self._spawner = TargetSpawner(
            min_distance=min_dist, max_distance=max_dist,
            reach_threshold=reach_threshold, seed=seed)
        self._spawn_fn = None
        self._state = GameState.STARTUP
        self._stats = GameStatistics()
        self._target_index = 0
        self._step_count = 0
        self._target_step_count = 0
        self._debug_server = None
        self._god_view_costmap: tuple | None = None
        self._god_view_path_file: str | None = None
        self._god_view_tsdf = None
        self._obstacle_bodies: list[str] = []
        self._scene_xml_path: str | None = None
        self._costmap_changed = False
        self._nav_prev_pos = None
        self._rp_log = []
        self._cached_pose: (
            tuple[float, float, float, float, float, float] | None
        ) = None
        self._glitch_ticks = 0

        # L5 MotionCommand class — resolved lazily to avoid config collision
        self._MotionCommand = None

        # --- Telemetry (in-memory, never prints) ---
        self._gt = GameTelemetry()

        # --- Composed helpers ---
        self.nav = Navigator(self)
        self.scoring = GameScoring(self)

    _PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"

    # --- L5 MotionCommand ---

    def _get_motion_command_class(self):
        """Lazily resolve L5's MotionCommand class."""
        if self._MotionCommand is None:
            from config.defaults import MotionCommand
            self._MotionCommand = MotionCommand
        return self._MotionCommand

    def _send_motion(self, vx: float = 0.0, wz: float = 0.0,
                     behavior: str = 'walk') -> None:
        """Send a MotionCommand through Layer 5."""
        MC = self._get_motion_command_class()
        cmd = MC(vx=vx, wz=wz, behavior=behavior, robot=self._robot)
        try:
            self._sim.send_motion_command(cmd, dt=CONTROL_DT,
                                            terrain=False)
        except RuntimeError:
            rc = None
            try:
                fw = self._sim._firmware
                if fw is not None and fw._proc is not None:
                    rc = fw._proc.poll()
            except Exception:
                pass
            self._gt.record_event(
                "sim_stopped", step=self._step_count,
                t=self._step_count * CONTROL_DT,
                firmware_exit=rc)
            self._state = GameState.DONE

    # --- Pose helpers ---

    def _get_robot_pose(self):
        """Return (x, y, yaw, z, roll, pitch) from ground truth.

        Includes glitch rejection: jumps > 2m are rejected for up to
        0.5s (25 ticks at 50Hz).  If the new position persists beyond
        the grace period, it's accepted as real (fall recovery, collision).
        """
        body = self._sim.get_body("base")
        if body is None:
            if self._cached_pose is not None:
                return self._cached_pose
            return (0.0, 0.0, 0.0, 0.465, 0.0, 0.0)  # default standing pose
        self._cached_body = body  # cache for _update_slam
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
                if self._glitch_ticks < 50:  # 0.5s at 100Hz
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
        body = getattr(self, '_cached_body', None) or self._sim.get_body("base")
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
        # Pose publishing disabled during deferred perception runs:
        # DDS write traffic + GIL contention destabilizes gait timing.
        if (self._pose_pub is not None
                and self._step_count % 10 == 0):
            self._publish_pose()

        if self._perception is not None and self._cached_pose is not None:
            z, roll, pitch = self._cached_pose[3], self._cached_pose[4], self._cached_pose[5]
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

    def _get_nav_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) for navigation -- ground truth.

        SLAM odometry drifts too much for heading-proportional control
        (6m+ position drift, 40°+ yaw noise).  SLAM is still used for
        perception (TSDF scan integration) via _update_slam().
        """
        x, y, yaw, _, _, _ = self._get_robot_pose()
        return x, y, yaw

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
                self._post_fall_settle = 500  # 5s at 100Hz
                self._gt.record_event(
                    "fall", step=self._step_count,
                    t=self._step_count * CONTROL_DT,
                    target_index=self._target_index,
                    z=z, ticks=FALL_CONFIRM_TICKS)
                self._state = GameState.SPAWN_TARGET
                return True
        else:
            self._fall_tick_count = 0
        return False

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

        # Update SLAM odometry
        _, _, truth_yaw, _, _, _ = self._get_robot_pose()
        self._update_slam(truth_yaw)

        # Record trails for visualization (10Hz)
        tick_slam_trails(self)

        # Debug viewer: stream state at 10Hz, TSDF at 2Hz
        if self._debug_server is not None and self._debug_server.has_client:
            stream_debug_viewer(self)

        # God-view A* path for MuJoCo overlay (only with debug viewer)
        if self._debug_server is not None:
            write_god_view_path(self)

        # Perception: LiDAR scan → TSDF → costmap → route replan flag
        tick_perception(self)

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

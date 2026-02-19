"""Target game state machine."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto

from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, clamp as _clamp


# --- Game constants ---

CONTROL_DT = 0.01          # 100 Hz
STARTUP_SETTLE = 1.5       # seconds (0.5s ramp + 1.0s stabilize)
WALK_SPEED = 1.0            # m/s forward
KP_YAW = 2.0               # proportional yaw gain
TURN_WZ_LIMIT = 2.0         # max angular velocity command (rad/s)
REACH_DISTANCE = 0.5        # m
TARGET_TIMEOUT_STEPS = 6000  # 60 seconds at 100 Hz (faster with improved locomotion)
TELEMETRY_INTERVAL = 100    # steps between prints (1 Hz at 100 Hz)


class GameState(Enum):
    STARTUP = auto()
    SPAWN_TARGET = auto()
    WALK_TO_TARGET = auto()
    DONE = auto()


@dataclass
class GameStatistics:
    targets_spawned: int = 0
    targets_reached: int = 0
    targets_timeout: int = 0
    falls: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.targets_spawned == 0:
            return 0.0
        return self.targets_reached / self.targets_spawned


class TargetGame:
    """Random target navigation game.

    Runs a state machine that spawns targets and drives the robot toward
    them using continuous cos(heading_err) steering.
    """

    def __init__(
        self,
        sim,
        num_targets: int = 5,
        min_dist: float = 3.0,
        max_dist: float = 6.0,
        reach_threshold: float = 0.5,
        timeout_steps: int = TARGET_TIMEOUT_STEPS,
        seed: int | None = None,
        angle_range: tuple[float, float] | list[tuple[float, float]] = (-math.pi / 2, math.pi / 2),
    ):
        self._sim = sim
        self._num_targets = num_targets
        self._reach_threshold = reach_threshold
        self._timeout_steps = timeout_steps
        self._angle_range = angle_range

        self._spawner = TargetSpawner(
            min_distance=min_dist,
            max_distance=max_dist,
            reach_threshold=reach_threshold,
            seed=seed,
        )

        self._state = GameState.STARTUP
        self._stats = GameStatistics()
        self._target_index = 0
        self._step_count = 0
        self._target_step_count = 0
        self._startup_steps = 0
        self._startup_total = int(STARTUP_SETTLE / CONTROL_DT)
        # hysteresis state removed — using continuous cos(heading_err) steering

    def _get_robot_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) from simulation."""
        body = self._sim.get_body("base")
        x = float(body.pos[0])
        y = float(body.pos[1])
        yaw = _quat_to_yaw(body.quat)
        return x, y, yaw

    def _send_cmd(self, vx: float = 0.0, wz: float = 0.0, behavior: str = "walk"):
        """Send a motion command through the simulation."""
        # Lazy import: Layer 5's config.defaults must be on sys.path, which
        # is set up by __main__.py before this method is ever called.
        from config.defaults import MotionCommand
        cmd = MotionCommand(vx=vx, wz=wz, behavior=behavior)
        try:
            self._sim.send_motion_command(cmd, dt=CONTROL_DT, terrain=False)
        except RuntimeError:
            # Firmware process exited — end the game gracefully
            print("Simulation stopped unexpectedly")
            self._state = GameState.DONE

    def tick(self) -> bool:
        """Advance one control step. Returns False when game is over."""
        if self._state == GameState.DONE:
            return False

        self._step_count += 1

        if self._state == GameState.STARTUP:
            self._tick_startup()
        elif self._state == GameState.SPAWN_TARGET:
            self._tick_spawn()
        elif self._state == GameState.WALK_TO_TARGET:
            self._tick_walk()

        return self._state != GameState.DONE

    def _tick_startup(self):
        """Send stand commands until ramp completes and settle time elapses."""
        self._send_cmd(behavior="stand")
        self._startup_steps += 1

        ramp_done = not self._sim.is_ramping
        settle_done = self._startup_steps >= self._startup_total

        if ramp_done and settle_done:
            print("Robot ready!")
            self._state = GameState.SPAWN_TARGET

    def _tick_spawn(self):
        """Spawn a new target and transition to turning."""
        if self._target_index >= self._num_targets:
            self._state = GameState.DONE
            return

        x, y, yaw = self._get_robot_pose()
        target = self._spawner.spawn_relative(x, y, yaw, angle_range=self._angle_range)
        self._target_index += 1
        self._target_step_count = 0
        self._stats.targets_spawned += 1

        # Move visual target marker (mocap body) to target position
        try:
            self._sim.set_mocap_pos(0, [target.x, target.y, 0.15])
        except (RuntimeError, AttributeError):
            pass  # No mocap body in scene (headless without target scene)

        dist = target.distance_to(x, y)
        print(f"\n[target {self._target_index}/{self._num_targets}] "
              f"spawned at ({target.x:.1f}, {target.y:.1f})  dist={dist:.1f}m")

        self._state = GameState.WALK_TO_TARGET

    def _tick_walk(self):
        """Walk toward target with continuous cos(heading_err) steering.

        Forward speed scales smoothly from WALK_SPEED (aligned) to 0
        (perpendicular or beyond), while yaw rate is proportional to
        heading error.  Matches the GA v9 evaluation steering model.
        """
        x, y, yaw = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        heading_err = target.heading_error(x, y, yaw)
        dist = target.distance_to(x, y)

        # Continuous steering: cos modulates forward speed, P-control on yaw
        vx = WALK_SPEED * max(0.0, math.cos(heading_err))
        wz = _clamp(KP_YAW * heading_err, -TURN_WZ_LIMIT, TURN_WZ_LIMIT)

        self._send_cmd(vx=vx, wz=wz)

        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"vx={vx:.2f}  pos=({x:.1f}, {y:.1f})  t={t:.1f}s")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _on_reached(self):
        t = self._target_step_count * CONTROL_DT
        self._stats.targets_reached += 1
        print(f"TARGET {self._target_index} REACHED in {t:.1f}s")
        self._state = GameState.SPAWN_TARGET

    def _on_timeout(self):
        t = self._target_step_count * CONTROL_DT
        self._stats.targets_timeout += 1
        print(f"TARGET {self._target_index} TIMEOUT after {t:.1f}s")
        self._state = GameState.SPAWN_TARGET

    def run(self) -> GameStatistics:
        """Run the full game loop and return statistics."""
        start_time = time.monotonic()

        while self.tick():
            time.sleep(CONTROL_DT)

        self._stats.total_time = time.monotonic() - start_time
        self._print_summary()
        return self._stats

    def _print_summary(self):
        s = self._stats
        print("\n=== GAME OVER ===")
        print(f"Targets: {s.targets_reached}/{s.targets_spawned} reached "
              f"({s.success_rate:.0%})")
        print(f"Timeouts: {s.targets_timeout}")
        print(f"Total time: {s.total_time:.1f}s")

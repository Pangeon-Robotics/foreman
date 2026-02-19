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
REACH_DISTANCE = 0.5        # m
TARGET_TIMEOUT_STEPS = 6000  # 60 seconds at 100 Hz
TELEMETRY_INTERVAL = 100    # steps between prints (1 Hz at 100 Hz)
NOMINAL_BODY_HEIGHT = 0.465 # m (B2)
FALL_THRESHOLD = 0.4        # fraction of nominal height (matches GA episode)

# --- Genome parameters (patched by _apply_genome in __main__.py) ---

GAIT_FREQ = 1.5
STEP_LENGTH = 0.30
STEP_HEIGHT = 0.08
DUTY_CYCLE = 0.55
STANCE_WIDTH = 0.0
KP_YAW = 2.0
WZ_LIMIT = 1.5
TURN_FREQ = 1.0
TURN_STEP_HEIGHT = 0.07
TURN_DUTY_CYCLE = 0.55
TURN_STANCE_WIDTH = 0.04
TURN_WZ = 1.2
THETA_THRESHOLD = 0.4


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
    them using L4 GaitParams directly (same path as GA training).
    """

    def __init__(
        self,
        sim,
        L4GaitParams=None,
        num_targets: int = 5,
        min_dist: float = 3.0,
        max_dist: float = 6.0,
        reach_threshold: float = 0.5,
        timeout_steps: int = TARGET_TIMEOUT_STEPS,
        seed: int | None = None,
        angle_range: tuple[float, float] | list[tuple[float, float]] = (-math.pi / 2, math.pi / 2),
    ):
        self._sim = sim
        self._L4GaitParams = L4GaitParams
        self._num_targets = num_targets
        self._reach_threshold = reach_threshold
        self._timeout_steps = timeout_steps
        self._angle_range = angle_range

        # Gains captured after startup ramp completes
        self._kp = None
        self._kd = None

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

    def _get_robot_pose(self) -> tuple[float, float, float, float]:
        """Return (x, y, yaw, z) from simulation."""
        body = self._sim.get_body("base")
        x = float(body.pos[0])
        y = float(body.pos[1])
        z = float(body.pos[2])
        yaw = _quat_to_yaw(body.quat)
        return x, y, yaw, z

    def _send_l4(self, params):
        """Send L4 GaitParams directly to Layer 4 (bypassing L5)."""
        try:
            self._sim.send_command(params, self._sim._t, kp=self._kp, kd=self._kd)
            self._sim._t += CONTROL_DT
        except RuntimeError:
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
        """Send stand commands via L5 until ramp completes and settle time elapses."""
        from config.defaults import MotionCommand
        cmd = MotionCommand(vx=0.0, wz=0.0, behavior="stand")
        try:
            self._sim.send_motion_command(cmd, dt=CONTROL_DT, terrain=False)
        except RuntimeError:
            print("Simulation stopped unexpectedly")
            self._state = GameState.DONE
            return

        self._startup_steps += 1

        ramp_done = not self._sim.is_ramping
        settle_done = self._startup_steps >= self._startup_total

        if ramp_done and settle_done:
            # Capture final gains for L4-direct control
            self._kp, self._kd = self._sim.locomotion.startup_gains()
            print(f"Robot ready! (kp={self._kp:.0f}, kd={self._kd:.1f})")
            self._state = GameState.SPAWN_TARGET

    def _tick_spawn(self):
        """Spawn a new target and transition to walking."""
        if self._target_index >= self._num_targets:
            self._state = GameState.DONE
            return

        x, y, yaw, _z = self._get_robot_pose()
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
        """Walk toward target using L4 GaitParams directly.

        Matches GA training episode logic (episode.py:1771-1811):
        - Large heading error -> turn in place (L4 arc mode)
        - Small heading error -> walk with differential stride
        """
        x, y, yaw, z = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection (matches GA episode: rz < nominal_z * 0.4)
        if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD:
            self._stats.falls += 1
            print(f"FALL DETECTED at z={z:.3f}m (threshold={NOMINAL_BODY_HEIGHT * FALL_THRESHOLD:.3f}m)")
            self._state = GameState.SPAWN_TARGET
            return

        heading_err = target.heading_error(x, y, yaw)
        dist = target.distance_to(x, y)

        if abs(heading_err) > THETA_THRESHOLD:
            # TURN IN PLACE — L4 arc mode (layer_4/generator.py:100-132)
            wz = TURN_WZ if heading_err > 0 else -TURN_WZ
            params = self._L4GaitParams(
                gait_type='trot', turn_in_place=True, wz=wz,
                gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
            )
        else:
            # WALK — L4 differential stride (layer_4/generator.py:134-164)
            heading_mod = max(0.0, math.cos(heading_err))
            wz = _clamp(KP_YAW * heading_err, -WZ_LIMIT, WZ_LIMIT)
            params = self._L4GaitParams(
                gait_type='trot', step_length=STEP_LENGTH * heading_mod,
                gait_freq=GAIT_FREQ, step_height=STEP_HEIGHT,
                duty_cycle=DUTY_CYCLE, stance_width=STANCE_WIDTH, wz=wz,
            )

        self._send_l4(params)

        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            mode = "TURN" if abs(heading_err) > THETA_THRESHOLD else "WALK"
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode}  dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"z={z:.2f}  pos=({x:.1f}, {y:.1f})  t={t:.1f}s")

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
        print(f"Timeouts: {s.targets_timeout}  Falls: {s.falls}")
        print(f"Total time: {s.total_time:.1f}s")

"""Target game state machine."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto

from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, quat_to_rpy as _quat_to_rpy, clamp as _clamp


# --- Game constants ---

CONTROL_DT = 0.01          # 100 Hz
STARTUP_RAMP_SECONDS = 2.5 # seconds — slow L4-direct gain ramp (L5's 0.5s causes vibration)
REACH_DISTANCE = 0.5        # m
TARGET_TIMEOUT_STEPS = 6000  # 60 seconds at 100 Hz
TELEMETRY_INTERVAL = 100    # steps between prints (1 Hz at 100 Hz)
NOMINAL_BODY_HEIGHT = 0.465 # m (B2)
FALL_THRESHOLD = 0.5        # fraction of nominal height (catches robot on its side at z=0.22m, threshold=0.233m)

# --- Startup gain ramp (bypasses L5 to avoid vibration) ---
KP_START = 500.0
KP_FULL = 4000.0
KD_START = 25.0
KD_FULL = 126.5             # sqrt(KP_FULL) * 2 — critically damped

# --- Genome parameters (patched by _apply_genome in __main__.py) ---

GAIT_FREQ = 1.5
STEP_LENGTH = 0.30
STEP_HEIGHT = 0.07
DUTY_CYCLE = 0.65
STANCE_WIDTH = 0.0
KP_YAW = 2.0
WZ_LIMIT = 1.5
TURN_FREQ = 1.0
TURN_STEP_HEIGHT = 0.07
TURN_DUTY_CYCLE = 0.65
TURN_STANCE_WIDTH = 0.08
TURN_WZ = 1.0
THETA_THRESHOLD = 0.6


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

        # Gains — start low, ramp during first 2.5s of navigation
        self._kp = KP_START
        self._kd = KD_START

        self._spawner = TargetSpawner(
            min_distance=min_dist,
            max_distance=max_dist,
            reach_threshold=reach_threshold,
            seed=seed,
        )

        self._state = GameState.SPAWN_TARGET  # No startup phase — ramp gains during navigation
        self._stats = GameStatistics()
        self._target_index = 0
        self._step_count = 0
        self._target_step_count = 0

        # Pitch/roll diagnostics
        self._rp_log = []  # (step, roll_deg, pitch_deg, droll, dpitch, mode)

    def _get_robot_pose(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, yaw, z, roll, pitch) from simulation."""
        body = self._sim.get_body("base")
        x = float(body.pos[0])
        y = float(body.pos[1])
        z = float(body.pos[2])
        yaw = _quat_to_yaw(body.quat)
        roll, pitch, _ = _quat_to_rpy(body.quat)
        return x, y, yaw, z, roll, pitch

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
        self._update_gains()

        if self._state == GameState.SPAWN_TARGET:
            self._tick_spawn()
        elif self._state == GameState.WALK_TO_TARGET:
            self._tick_walk()

        return self._state != GameState.DONE

    def _update_gains(self):
        """Smoothly ramp PD gains over first 2.5s of operation."""
        if self._kp >= KP_FULL:
            return
        t = self._step_count * CONTROL_DT
        progress = min(1.0, t / STARTUP_RAMP_SECONDS)
        alpha = progress * progress * (3.0 - 2.0 * progress)  # smoothstep
        self._kp = KP_START + alpha * (KP_FULL - KP_START)
        self._kd = KD_START + alpha * (KD_FULL - KD_START)

    def _tick_spawn(self):
        """Spawn a new target and transition to walking."""
        if self._target_index >= self._num_targets:
            self._state = GameState.DONE
            return

        x, y, yaw, _z, _r, _p = self._get_robot_pose()
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
        x, y, yaw, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection (matches GA episode: rz < nominal_z * 0.5)
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
            # Decel taper in last 30% before threshold so WALK→TURN
            # transition doesn't pitch the robot forward from momentum.
            taper_start = 0.7 * THETA_THRESHOLD
            if abs(heading_err) > taper_start:
                decel = (THETA_THRESHOLD - abs(heading_err)) / (THETA_THRESHOLD - taper_start)
                decel = max(0.0, decel)
            else:
                decel = 1.0
            heading_mod = decel * max(0.0, math.cos(heading_err))
            wz = _clamp(KP_YAW * heading_err, -WZ_LIMIT, WZ_LIMIT)
            params = self._L4GaitParams(
                gait_type='trot', step_length=STEP_LENGTH * heading_mod,
                gait_freq=GAIT_FREQ, step_height=STEP_HEIGHT,
                duty_cycle=DUTY_CYCLE, stance_width=STANCE_WIDTH, wz=wz,
            )

        self._send_l4(params)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "T" if abs(heading_err) > THETA_THRESHOLD else "W"
        if len(self._rp_log) > 0:
            prev = self._rp_log[-1]
            droll = (r_deg - prev[1]) / CONTROL_DT
            dpitch = (p_deg - prev[2]) / CONTROL_DT
        else:
            droll = dpitch = 0.0
        self._rp_log.append((self._step_count, r_deg, p_deg, droll, dpitch, mode))

        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            mode_str = "TURN" if abs(heading_err) > THETA_THRESHOLD else "WALK"
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str}  dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"z={z:.2f}  R={r_deg:+.1f}° P={p_deg:+.1f}°  "
                  f"pos=({x:.1f}, {y:.1f})  t={t:.1f}s")

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
        self._print_rp_analysis()

    def _print_rp_analysis(self):
        """Print pitch/roll dynamics analysis."""
        if not self._rp_log:
            return
        rolls = [r[1] for r in self._rp_log]
        pitches = [r[2] for r in self._rp_log]
        drolls = [r[3] for r in self._rp_log]
        dpitches = [r[4] for r in self._rp_log]
        modes = [r[5] for r in self._rp_log]

        print("\n=== PITCH/ROLL ANALYSIS ===")
        print(f"Samples: {len(self._rp_log)} ({len(self._rp_log)*CONTROL_DT:.1f}s)")
        print(f"Roll:   min={min(rolls):+.1f}°  max={max(rolls):+.1f}°  "
              f"mean={sum(abs(r) for r in rolls)/len(rolls):.1f}°")
        print(f"Pitch:  min={min(pitches):+.1f}°  max={max(pitches):+.1f}°  "
              f"mean={sum(abs(p) for p in pitches)/len(pitches):.1f}°")
        print(f"dRoll:  min={min(drolls):+.0f}°/s  max={max(drolls):+.0f}°/s")
        print(f"dPitch: min={min(dpitches):+.0f}°/s  max={max(dpitches):+.0f}°/s")

        # Top 10 worst moments by combined |roll| + |pitch|
        scored = [(abs(r[1]) + abs(r[2]), i, r) for i, r in enumerate(self._rp_log)]
        scored.sort(reverse=True)
        print("\nTop 10 worst tilt moments (|R|+|P|):")
        print(f"  {'step':>6}  {'t':>5}  {'R':>7}  {'P':>7}  {'dR':>8}  {'dP':>8}  mode")
        for _, idx, r in scored[:10]:
            step, roll, pitch, dr, dp, mode = r
            t = step * CONTROL_DT
            print(f"  {step:6d}  {t:5.2f}  {roll:+7.1f}  {pitch:+7.1f}  "
                  f"{dr:+8.0f}  {dp:+8.0f}  {mode}")

        # Mode breakdown
        walk_rolls = [abs(r[1]) for r in self._rp_log if r[5] == "W"]
        turn_rolls = [abs(r[1]) for r in self._rp_log if r[5] == "T"]
        walk_pitches = [abs(r[2]) for r in self._rp_log if r[5] == "W"]
        turn_pitches = [abs(r[2]) for r in self._rp_log if r[5] == "T"]
        if walk_rolls:
            print(f"\nWALK ({len(walk_rolls)} steps):  "
                  f"avg|R|={sum(walk_rolls)/len(walk_rolls):.1f}°  "
                  f"avg|P|={sum(walk_pitches)/len(walk_pitches):.1f}°  "
                  f"max|R|={max(walk_rolls):.1f}°  max|P|={max(walk_pitches):.1f}°")
        if turn_rolls:
            print(f"TURN ({len(turn_rolls)} steps):  "
                  f"avg|R|={sum(turn_rolls)/len(turn_rolls):.1f}°  "
                  f"avg|P|={sum(turn_pitches)/len(turn_pitches):.1f}°  "
                  f"max|R|={max(turn_rolls):.1f}°  max|P|={max(turn_pitches):.1f}°")

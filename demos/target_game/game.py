"""Target game state machine.

Core TargetGame class with state machine, SLAM integration, and summary
reporting. Navigation methods are provided by NavigatorMixin (heading-based
and wheeled) and NavigatorDWAMixin (obstacle-avoidance DWA).

Configuration constants, GameState, GameStatistics, and configure_for_robot
live in game_config.py. All names are re-exported here for backward
compatibility.
"""
from __future__ import annotations

import math
import time
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
from .navigator_dwa import NavigatorDWAMixin
from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, quat_to_rpy as _quat_to_rpy

if TYPE_CHECKING:
    from layer_6.slam.odometry import Odometry

# Re-export everything from game_config for backward compatibility.
# __main__.py and other consumers import from .game — this ensures
# they continue to work without changes.
from .game_config import *  # noqa: F401,F403


class TargetGame(NavigatorMixin, NavigatorDWAMixin):
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
        angle_range: tuple[float, float] | list[tuple[float, float]] = (-math.pi / 2, math.pi / 2),
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

        # SLAM odometry (Phase 1: replaces ground-truth position for nav)
        self._odometry = odometry
        self._slam_trail: list[tuple[float, float]] = []   # estimated (x, y)
        self._truth_trail: list[tuple[float, float]] = []   # ground truth (x, y)
        self._pose_pub = None       # DDS publisher, set via set_pose_publisher()
        self._PoseMsg = None
        self._pose_stamp = None
        self._perception = None     # PerceptionPipeline, set from __main__.py
        self._dwa_planner = None    # DWAPlanner, set via set_dwa_planner()
        self._last_dwa_result = None
        self._telemetry = None      # TelemetryLog, set via set_telemetry()
        self._path_critic = None    # PathCritic, set via set_path_critic()

        # Avoidance commitment — holds turn direction to prevent flickering
        self._avoidance_sign = 0       # +1 left, -1 right, 0 no commitment
        self._avoidance_since = 0.0    # tick time when direction was chosen
        self._avoidance_dist = 0.0     # dist-to-target when commitment started
        self._goal_bearing = 0.0       # cached goal bearing for reactive scan

        # DWA turn/forward smoothing
        self._smooth_dwa_turn = 0.0
        self._smooth_dwa_fwd = 1.0

        # Regression tracking: detect sustained wrong-way walking.
        # Ring buffer of recent dist-to-target samples (at 10Hz).
        self._dist_ring = []          # recent (step, dist) samples
        self._recent_regression = 0.0 # accumulated regression in last 2s

        # Fall detection: require sustained low z to avoid false positives
        # from dynamic gait oscillation. Body dips during turns/walks are
        # temporary; real falls are sustained.
        self._fall_tick_count = 0
        self._post_fall_settle = 0     # ticks remaining to settle after fall
        self._stabilize_countdown = 0  # pre-fall stabilization hold timer

        # Gait parameter smoothing (EMA state for DWA mode)
        self._smooth_heading_mod = 1.0
        self._smooth_wz = 0.0
        self._decel_tick_count = 0
        self._last_threat_time = 0.0  # tracks last obstacle encounter

        # SLAM drift detection: track minimum distance to target
        self._min_target_dist = float('inf')
        self._slam_drift_latched = False

        # Gains — start low, ramp during first 2.5s of navigation
        self._kp = KP_START
        self._kd = KD_START

        self._spawner = TargetSpawner(
            min_distance=min_dist,
            max_distance=max_dist,
            reach_threshold=reach_threshold,
            seed=seed,
        )
        self._spawn_fn = None  # Optional: (robot_x, robot_y, robot_yaw, idx) -> Target

        self._state = GameState.STARTUP  # Settle before first gait command
        self._stats = GameStatistics()
        self._target_index = 0
        self._step_count = 0
        self._target_step_count = 0

        # Wheeled mode: use keyframe pose for rigid leg PD hold
        self._home_q = WHEEL_HOME_Q if WHEELED else None

        # Obstacle proximity (foreman ground-truth referee, not robot sensors)
        self._obstacle_bodies: list[str] = []

        # Step-and-shift TIP: diagonal pair stepping for stable turns
        self._step_and_shift = None
        self._in_tip_mode = False
        self._tip_start_step = 0
        self._tip_start_heading = 0.0  # heading error at TIP entry
        self._tip_cooldown_until = 0  # step count when cooldown expires

        # A* waypoint guidance: next waypoint along optimal path
        self._current_waypoint = None  # (wx, wy) world-frame or None
        self._wp_commit_until = 0      # step count: suppress side-flips until then

        # Stuck detection: force recovery when no progress in tight spaces
        self._stuck_check_dist = float('inf')  # distance at last check
        self._stuck_check_step = 0              # step of last check
        self._stuck_recovery_countdown = 0      # ticks remaining in recovery maneuver
        self._stuck_recovery_wz = 0.0           # turn direction during recovery
        self._prev_no_progress = False          # stalled detection (two consecutive windows)

        # Progress-gated early timeout: skip hopeless targets faster
        self._progress_window_dist = float('inf')  # distance at window start
        self._progress_window_step = 0              # step at window start

        # Regression tracking (not currently active)
        self._closest_approach = float('inf')
        self._reg_start_step = 0

        # Pitch/roll diagnostics
        self._rp_log = []  # (step, roll_deg, pitch_deg, droll, dpitch, mode)

        # Pose glitch rejection: DDS buffer reuse can return stale/zeroed data
        self._cached_pose: tuple[float, float, float, float, float, float] | None = None

    def _get_robot_pose(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, yaw, z, roll, pitch) from simulation ground truth.

        Includes glitch rejection: if position jumps > 2m in a single tick
        (impossible for a walking robot at ~0.5 m/s), return the cached pose
        instead.
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
            jump = math.sqrt(dx * dx + dy * dy)
            if jump > 2.0:
                return self._cached_pose

        self._cached_pose = pose
        return pose

    def _update_slam(self, truth_yaw: float) -> None:
        """Feed SLAM odometry with current sensor data."""
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
            timestamp=l5_state.timestamp if l5_state.timestamp > 0 else self._step_count * CONTROL_DT,
        )
        self._odometry.update(state, CONTROL_DT)

        if self._pose_pub is not None:
            self._publish_pose()

    def set_dwa_planner(self, planner) -> None:
        """Set DWA planner for obstacle-aware navigation."""
        self._dwa_planner = planner

    _PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"

    def set_path_critic(self, critic) -> None:
        """Set PathCritic for path quality evaluation."""
        self._path_critic = critic

    def set_obstacle_bodies(self, names: list[str]) -> None:
        """Register obstacle body names for ground-truth proximity checking."""
        self._obstacle_bodies = names

    def set_pose_publisher(self, pub, msg_class, stamp_fn) -> None:
        """Set DDS publisher for rt/pose_estimate topic."""
        self._pose_pub = pub
        self._PoseMsg = msg_class
        self._pose_stamp = stamp_fn

    def set_telemetry(self, telemetry_log) -> None:
        """Set JSONL telemetry logger for Layer 6 modules."""
        self._telemetry = telemetry_log

    def _publish_pose(self) -> None:
        """Publish current SLAM pose on DDS."""
        if self._odometry is None or self._pose_pub is None:
            return
        p = self._odometry.pose
        msg = self._PoseMsg(x=p.x, y=p.y, yaw=p.yaw, timestamp=p.stamp, crc=0)
        self._pose_stamp(msg)
        self._pose_pub.Write(msg)

    def _get_nav_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) for navigation decisions.

        Always uses ground truth. SLAM dead-reckoning drifts too fast
        for reliable navigation.
        """
        x, y, yaw, _, _, _ = self._get_robot_pose()
        return x, y, yaw

    def _get_tip_scales(self):
        """Get leg_scales from step-and-shift state machine for TIP."""
        if self._step_and_shift is None:
            from step_and_shift import StepAndShift
            self._step_and_shift = StepAndShift(gait_freq=TURN_FREQ)
        return self._step_and_shift.update(CONTROL_DT)

    def _reset_tip(self):
        """Reset step-and-shift state machine when exiting TIP."""
        if self._step_and_shift is not None:
            self._step_and_shift.reset()

    @property
    def slam_trail(self) -> list[tuple[float, float]]:
        """SLAM estimated positions (10Hz), for visualization."""
        return self._slam_trail

    @property
    def truth_trail(self) -> list[tuple[float, float]]:
        """Ground truth positions (10Hz), for visualization."""
        return self._truth_trail

    def _check_fall(self, z: float) -> bool:
        """Check for sustained fall. Returns True if robot has fallen."""
        if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD:
            self._fall_tick_count += 1
            if self._fall_tick_count >= FALL_CONFIRM_TICKS:
                self._stats.falls += 1
                self._fall_tick_count = 0
                self._post_fall_settle = 500
                print(f"FALL DETECTED at z={z:.3f}m (sustained {FALL_CONFIRM_TICKS} ticks)")
                self._state = GameState.SPAWN_TARGET
                return True
        else:
            self._fall_tick_count = 0
        return False

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

        # Update SLAM odometry (every tick, before navigation)
        _, _, truth_yaw, _, _, _ = self._get_robot_pose()
        self._update_slam(truth_yaw)

        # Record trails for visualization (every 10 steps = 10Hz)
        if self._odometry is not None and self._step_count % 10 == 0:
            truth_x, truth_y, _, _, _, _ = self._get_robot_pose()
            self._truth_trail.append((truth_x, truth_y))
            p = self._odometry.pose
            self._slam_trail.append((p.x, p.y))

            if self._telemetry is not None:
                drift = math.sqrt((p.x - truth_x)**2 + (p.y - truth_y)**2)
                self._telemetry.record("slam", {
                    "drift": round(drift, 4),
                    "slam_x": round(p.x, 3), "slam_y": round(p.y, 3),
                    "truth_x": round(truth_x, 3), "truth_y": round(truth_y, 3),
                })

        # Dynamic reach threshold
        if self._state == GameState.WALK_TO_TARGET:
            t = self._target_step_count * CONTROL_DT
            self._reach_threshold = min(
                self._reach_threshold_base + t * 0.02,
                self._reach_threshold_base + 1.0,
            )

        if self._state == GameState.STARTUP:
            self._tick_startup()
        elif self._state == GameState.SPAWN_TARGET:
            self._tick_spawn()
        elif self._state == GameState.WALK_TO_TARGET:
            if WHEELED:
                self._tick_walk_wheeled()
            elif self._dwa_planner is not None and self._perception is not None:
                self._tick_walk_dwa()
            else:
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

    def _tick_startup(self):
        """Hold a neutral stand for STARTUP_SETTLE_STEPS to let the robot settle."""
        if WHEELED and self._home_q is not None:
            self._send_wheeled(0.0, 0.0)
        else:
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )
            self._send_l4(params)
        if self._step_count >= STARTUP_SETTLE_STEPS:
            x, y, yaw, z, roll, pitch = self._get_robot_pose()
            roll_deg = abs(math.degrees(roll))
            if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD or roll_deg > 5.0:
                print(f"Startup FAILED: z={z:.3f}m  "
                      f"R={math.degrees(roll):+.1f}\u00b0  P={math.degrees(pitch):+.1f}\u00b0  "
                      f"(fell or unstable settle)")
                self._stats.falls += 1
                self._state = GameState.DONE
                return
            if WHEELED:
                if self._home_q is None:
                    state = self._sim.get_robot_state()
                    self._home_q = [state.joint_positions[i] for i in range(12)]
                    print(f"Startup complete (wheeled, L4 pose): z={z:.3f}m  "
                          f"home_q=[{self._home_q[0]:.3f}, {self._home_q[1]:.3f}, {self._home_q[2]:.3f}]")
                else:
                    print(f"Startup complete (wheeled): z={z:.3f}m  home_q={WHEEL_HOME_Q[:3]}")
            else:
                print(f"Startup complete: z={z:.3f}m  R={math.degrees(roll):+.1f}\u00b0  "
                      f"P={math.degrees(pitch):+.1f}\u00b0")
            self._state = GameState.SPAWN_TARGET

    def _tick_spawn(self):
        """Spawn a new target and transition to walking."""
        if self._target_index >= self._num_targets:
            self._state = GameState.DONE
            return

        if self._post_fall_settle > 0:
            self._post_fall_settle -= 1
            _, _, _, z, _, _ = self._get_robot_pose()
            recovered = z > NOMINAL_BODY_HEIGHT * 0.85
            if not recovered and self._post_fall_settle > 0:
                params = self._L4GaitParams(
                    gait_type='trot', step_length=0.0,
                    gait_freq=GAIT_FREQ, step_height=0.0,
                    duty_cycle=1.0, stance_width=0.0, wz=0.0,
                    body_height=BODY_HEIGHT,
                )
                self._send_l4(params)
                return

        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        if self._spawn_fn is not None:
            target = self._spawn_fn(nav_x, nav_y, nav_yaw, self._target_index)
            self._spawner._current_target = target
        else:
            target = self._spawner.spawn_relative(nav_x, nav_y, nav_yaw, angle_range=self._angle_range)
        self._target_index += 1
        self._target_step_count = 0
        self._reach_threshold = self._reach_threshold_base
        self._min_target_dist = float('inf')
        self._slam_drift_latched = False
        self._in_tip_mode = False
        self._tip_start_step = 0
        self._tip_start_heading = 0.0
        self._tip_cooldown_until = 0
        self._reset_tip()
        self._smooth_heading_mod = 0.0
        self._smooth_wz = 0.0
        self._decel_tick_count = 0
        self._smooth_dwa_turn = 0.0
        self._smooth_dwa_fwd = 1.0
        self._avoidance_sign = 0
        self._avoidance_dist = 0.0
        self._last_threat_level = 0.0
        self._current_waypoint = None
        self._wp_commit_until = 0
        self._stuck_check_dist = float('inf')
        self._stuck_check_step = 0
        self._stuck_recovery_countdown = 0
        self._stuck_recovery_wz = 0.0
        self._prev_no_progress = False
        self._last_good_heading_step = -999
        self._progress_window_dist = float('inf')
        self._progress_window_step = 0
        self._dist_ring = []
        self._recent_regression = 0.0
        self._closest_approach = float('inf')
        self._reg_start_step = 0
        self._stats.targets_spawned += 1

        target_z = min(NOMINAL_BODY_HEIGHT, 0.30)
        try:
            self._sim.set_mocap_pos(0, [target.x, target.y, target_z])
        except (RuntimeError, AttributeError):
            pass

        dist = target.distance_to(nav_x, nav_y)
        src = "SLAM" if self._odometry else "truth"
        print(f"\n[target {self._target_index}/{self._num_targets}] "
              f"spawned at ({target.x:.1f}, {target.y:.1f})  dist={dist:.1f}m  (nav={src})")

        if self._path_critic is not None:
            self._path_critic.set_target(target.x, target.y)
            self._path_critic.record(nav_x, nav_y, t=0.0)

        self._state = GameState.WALK_TO_TARGET

    def _on_reached(self):
        t = self._target_step_count * CONTROL_DT
        self._stats.targets_reached += 1
        print(f"TARGET {self._target_index} REACHED in {t:.1f}s")
        if self._path_critic is not None:
            target = self._spawner.current_target
            report = self._path_critic.target_reached(target.x, target.y)
            if report:
                agg = self._path_critic.aggregate_ato()
                print(f"  ATO={report['ato_score']:.1f}  r_gate={report['regression_gate']:.2f}  "
                      f"agg={agg:.1f}")
        self._state = GameState.SPAWN_TARGET

    def _on_timeout(self):
        t = self._target_step_count * CONTROL_DT
        self._stats.targets_timeout += 1
        print(f"TARGET {self._target_index} TIMEOUT after {t:.1f}s")
        if self._path_critic is not None:
            target = self._spawner.current_target
            report = self._path_critic.target_timeout(target.x, target.y)
            if report:
                agg = self._path_critic.aggregate_ato()
                print(f"  ATO=0.0  r_gate={report['regression_gate']:.2f}  "
                      f"agg={agg:.1f}")
        self._state = GameState.SPAWN_TARGET

    def run(self) -> GameStatistics:
        """Run the full game loop and return statistics."""
        start_time = time.monotonic()

        while self.tick():
            time.sleep(CONTROL_DT)

        self._stats.total_time = time.monotonic() - start_time
        if self._telemetry is not None:
            self._telemetry.close()
        import os
        try:
            os.unlink(self._PATH_VIZ_FILE)
        except OSError:
            pass
        self._print_summary()
        if self._path_critic is not None:
            self._path_critic.print_summary()
        return self._stats

    def _print_summary(self):
        s = self._stats
        print("\n=== GAME OVER ===")
        print(f"Targets: {s.targets_reached}/{s.targets_spawned} reached "
              f"({s.success_rate:.0%})")
        print(f"Timeouts: {s.targets_timeout}  Falls: {s.falls}")
        print(f"Total time: {s.total_time:.1f}s")
        self._print_slam_analysis()
        self._print_rp_analysis()

    def _print_slam_analysis(self):
        """Print SLAM drift analysis if odometry was active."""
        if self._odometry is None or not self._truth_trail:
            return
        drifts = [
            math.sqrt((s[0] - t[0])**2 + (s[1] - t[1])**2)
            for s, t in zip(self._slam_trail, self._truth_trail)
        ]
        if not drifts:
            return
        print("\n=== SLAM DRIFT ANALYSIS ===")
        print(f"Samples: {len(drifts)} ({len(drifts) * 0.1:.1f}s at 10Hz)")
        print(f"Drift: min={min(drifts):.3f}m  max={max(drifts):.3f}m  "
              f"mean={sum(drifts)/len(drifts):.3f}m  final={drifts[-1]:.3f}m")

    def _print_rp_analysis(self):
        """Print pitch/roll dynamics analysis."""
        if not self._rp_log:
            return
        rolls = [r[1] for r in self._rp_log]
        pitches = [r[2] for r in self._rp_log]
        drolls = [r[3] for r in self._rp_log]
        dpitches = [r[4] for r in self._rp_log]

        print("\n=== PITCH/ROLL ANALYSIS ===")
        print(f"Samples: {len(self._rp_log)} ({len(self._rp_log)*CONTROL_DT:.1f}s)")
        print(f"Roll:   min={min(rolls):+.1f}\u00b0  max={max(rolls):+.1f}\u00b0  "
              f"mean={sum(abs(r) for r in rolls)/len(rolls):.1f}\u00b0")
        print(f"Pitch:  min={min(pitches):+.1f}\u00b0  max={max(pitches):+.1f}\u00b0  "
              f"mean={sum(abs(p) for p in pitches)/len(pitches):.1f}\u00b0")
        print(f"dRoll:  min={min(drolls):+.0f}\u00b0/s  max={max(drolls):+.0f}\u00b0/s")
        print(f"dPitch: min={min(dpitches):+.0f}\u00b0/s  max={max(dpitches):+.0f}\u00b0/s")

        scored = [(abs(r[1]) + abs(r[2]), i, r) for i, r in enumerate(self._rp_log)]
        scored.sort(reverse=True)
        print("\nTop 10 worst tilt moments (|R|+|P|):")
        print(f"  {'step':>6}  {'t':>5}  {'R':>7}  {'P':>7}  {'dR':>8}  {'dP':>8}  mode")
        for _, idx, r in scored[:10]:
            step, roll, pitch, dr, dp, mode = r
            t = step * CONTROL_DT
            print(f"  {step:6d}  {t:5.2f}  {roll:+7.1f}  {pitch:+7.1f}  "
                  f"{dr:+8.0f}  {dp:+8.0f}  {mode}")

        for label, code in [("WALK", "W"), ("TURN", "T"), ("DRIVE", "D")]:
            mode_rolls = [abs(r[1]) for r in self._rp_log if r[5] == code]
            mode_pitches = [abs(r[2]) for r in self._rp_log if r[5] == code]
            if mode_rolls:
                print(f"\n{label} ({len(mode_rolls)} steps):  "
                      f"avg|R|={sum(mode_rolls)/len(mode_rolls):.1f}\u00b0  "
                      f"avg|P|={sum(mode_pitches)/len(mode_pitches):.1f}\u00b0  "
                      f"max|R|={max(mode_rolls):.1f}\u00b0  max|P|={max(mode_pitches):.1f}\u00b0")

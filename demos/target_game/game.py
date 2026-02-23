"""Target game state machine."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from .target import TargetSpawner
from .utils import quat_to_yaw as _quat_to_yaw, quat_to_rpy as _quat_to_rpy, clamp as _clamp

if TYPE_CHECKING:
    from layer_6.slam.odometry import Odometry


# --- Game constants ---

CONTROL_DT = 0.01          # 100 Hz
STARTUP_RAMP_SECONDS = 2.5 # seconds — slow L4-direct gain ramp (L5's 0.5s causes vibration)
REACH_DISTANCE = 0.5        # m
TARGET_TIMEOUT_STEPS = 6000  # 60 seconds at 100 Hz
TELEMETRY_INTERVAL = 100    # steps between prints (1 Hz at 100 Hz)
NOMINAL_BODY_HEIGHT = 0.465 # m (B2)
FALL_THRESHOLD = 0.5        # fraction of nominal height
FALL_CONFIRM_TICKS = 20     # consecutive ticks below threshold to confirm fall (0.2s)

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
BODY_HEIGHT = 0.465
KP_YAW = 2.0
WZ_LIMIT = 1.5
TURN_FREQ = 1.0
TURN_STEP_HEIGHT = 0.06
TURN_DUTY_CYCLE = 0.65
TURN_STANCE_WIDTH = 0.12
TURN_WZ = 0.6
THETA_THRESHOLD = 0.6

# --- Gait parameter smoothing (context-dependent EMA) ---
# Near obstacles: slow decel preserves smooth avoidance curves
_EMA_ALPHA_DOWN_OBSTACLE = 0.02   # tau ~0.5s
_EMA_ALPHA_UP_OBSTACLE = 0.08     # tau ~0.125s
# Open field: faster tracking prevents orbit, still smooth enough to prevent falls
_EMA_ALPHA_DOWN_OPEN = 0.08       # tau ~0.13s — 13 ticks decel, tracks DWA better
_EMA_ALPHA_UP_OPEN = 0.15         # tau ~0.07s — fast turn→walk recovery
# Wz smoothing
_EMA_ALPHA_WZ_OBSTACLE = 0.10     # tau ~0.1s
_EMA_ALPHA_WZ_OPEN = 0.15         # tau ~0.07s
# DWA turn smoothing: suppress frame-to-frame oscillation
_DWA_TURN_ALPHA = 0.15            # tau ~0.7s at 10Hz replan — moderate smoothing to suppress oscillation
# Minimum decel before walk→turn mode switch
_MIN_DECEL_TICKS = 15             # 0.15s at 100 Hz

# --- Wheeled robot parameters (set by configure_for_robot) ---

WHEELED = False
WHEEL_FWD_TORQUE = 2.0
WHEEL_KP_YAW = 4.0
WHEEL_MAX_TURN = 12.0
WHEEL_HOME_Q = None  # Keyframe leg pose for PD hold (12 values, actuator order)

# --- Per-robot defaults ---
# Scaling rationale: Go2 legs are 60% of B2 (0.213/0.35), mass is 25% (15/60 kg).
# Step params scale with leg length, gains scale with mass, freq scales inversely.

ROBOT_DEFAULTS = {
    "b2": {
        "NOMINAL_BODY_HEIGHT": 0.465, "BODY_HEIGHT": 0.465,
        "KP_START": 500.0, "KP_FULL": 4000.0,
        "KD_START": 25.0, "KD_FULL": 126.5,
        "GAIT_FREQ": 1.5, "STEP_LENGTH": 0.30, "STEP_HEIGHT": 0.07,
        "DUTY_CYCLE": 0.65, "STANCE_WIDTH": 0.0,
        "KP_YAW": 2.0, "WZ_LIMIT": 1.5,
        "TURN_FREQ": 1.0, "TURN_STEP_HEIGHT": 0.06,
        "TURN_DUTY_CYCLE": 0.65, "TURN_STANCE_WIDTH": 0.12,
        "TURN_WZ": 0.6, "THETA_THRESHOLD": 0.6,
    },
    "go2": {
        "NOMINAL_BODY_HEIGHT": 0.27,       # MJCF keyframe z (actual sim standing height)
        "BODY_HEIGHT": 0.34,               # L4 NOMINAL_HEIGHT — target standing height for IK
        # Gains: go2 motors max 23.7 Nm. KP>200 causes torque saturation → backward walking.
        # KP=150 keeps thigh tracking in linear PD zone (saturation at 0.158 rad error).
        "KP_START": 75.0, "KP_FULL": 150.0,
        "KD_START": 3.8, "KD_FULL": 7.5,
        "GAIT_FREQ": 1.5, "STEP_LENGTH": 0.18, "STEP_HEIGHT": 0.06,
        "DUTY_CYCLE": 0.65, "STANCE_WIDTH": 0.0,
        "KP_YAW": 2.0, "WZ_LIMIT": 1.5,
        "TURN_FREQ": 1.5, "TURN_STEP_HEIGHT": 0.04,
        "TURN_DUTY_CYCLE": 0.65, "TURN_STANCE_WIDTH": 0.04,
        "TURN_WZ": 0.8, "THETA_THRESHOLD": 0.6,
    },
    "go2w": {
        "WHEELED": True,
        "NOMINAL_BODY_HEIGHT": 0.34,       # Proportional go2 target height (73% of B2's 0.465m)
        # Elegantly poised stance: splayed hips + fore/aft stagger for wide support polygon.
        # CoM analysis: tipping force (97.6N) > friction limit (74.5N) → can't tip over.
        # Actuator order: FR, FL, RR, RL — each [hip_abd, thigh, knee]
        "WHEEL_HOME_Q": [
            -0.25, 0.85, -1.6,   # FR: hip splay out, front thigh, knee
            +0.25, 0.85, -1.6,   # FL: hip splay out, front thigh, knee
            -0.25, 0.75, -1.6,   # RR: hip splay out, rear thigh, knee
            +0.25, 0.75, -1.6,   # RL: hip splay out, rear thigh, knee
        ],
        # Gains: go2w motors max 23.7 Nm. Rigid hold at home pose.
        "KP_START": 150.0, "KP_FULL": 150.0,
        "KD_START": 7.5, "KD_FULL": 7.5,
        # With rigid legs and wide support polygon, robot rolls on wheels like a car.
        # Higher torque OK because splayed stance prevents tipping.
        "WHEEL_FWD_TORQUE": 2.0,           # Nm per wheel (5.0 caused 44° nose-dive)
        "WHEEL_KP_YAW": 4.0,              # Nm/rad heading proportional
        "WHEEL_MAX_TURN": 6.0,            # Nm max differential
    },
    "b2w": {
        "WHEELED": True,
        "NOMINAL_BODY_HEIGHT": 0.46,       # B2 target height (~0.465m), splayed stance
        # Elegantly poised stance: splayed hips + fore/aft stagger.
        # B2w legs: thigh=0.35m, calf=0.35m. Motors: 200 Nm (hip/thigh), 300 Nm (calf).
        # Front height: 0.35*cos(0.95) + 0.35*cos(-0.75) = 0.460m
        # Rear height:  0.35*cos(0.85) + 0.35*cos(-0.85) = 0.460m
        # Actuator order: FR, FL, RR, RL — each [hip_abd, thigh, knee]
        "WHEEL_HOME_Q": [
            -0.25, 0.95, -1.7,   # FR: hip splay out, front thigh, knee
            +0.25, 0.95, -1.7,   # FL: hip splay out, front thigh, knee
            -0.25, 0.85, -1.7,   # RR: hip splay out, rear thigh, knee
            +0.25, 0.85, -1.7,   # RL: hip splay out, rear thigh, knee
        ],
        # Gains: b2w is 65 kg, needs high PD to hold pose rigidly.
        "KP_START": 4000.0, "KP_FULL": 4000.0,
        "KD_START": 126.5, "KD_FULL": 126.5,
        # b2w: 65 kg robot, 20 Nm wheel motors. Strong torque OK with splayed stance.
        "WHEEL_FWD_TORQUE": 15.0,          # Nm per wheel
        "WHEEL_KP_YAW": 8.0,              # Nm/rad heading proportional
        "WHEEL_MAX_TURN": 12.0,           # Nm max differential
    },
}


def configure_for_robot(robot: str) -> None:
    """Set module-level game constants for the given robot.

    Call before _apply_genome so genome values override these defaults.
    """
    import sys
    mod = sys.modules[__name__]
    defaults = ROBOT_DEFAULTS.get(robot, ROBOT_DEFAULTS["b2"])
    for name, value in defaults.items():
        setattr(mod, name, value)
    if defaults.get('WHEELED'):
        print(f"Configured game for {robot} (wheeled): height={defaults['NOMINAL_BODY_HEIGHT']}m, "
              f"KP={defaults['KP_FULL']}, fwd_torque={defaults['WHEEL_FWD_TORQUE']}Nm")
    else:
        print(f"Configured game for {robot}: height={defaults['NOMINAL_BODY_HEIGHT']}m, "
              f"KP={defaults['KP_FULL']}, step_h={defaults['STEP_HEIGHT']}m")


STARTUP_SETTLE_STEPS = 100  # 1s at 100 Hz — let robot settle before first gait command


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

        # Avoidance commitment — holds turn direction to prevent flickering
        self._avoidance_sign = 0       # +1 left, -1 right, 0 no commitment
        self._avoidance_since = 0.0    # tick time when direction was chosen
        self._goal_bearing = 0.0       # cached goal bearing for reactive scan

        # DWA turn smoothing
        self._smooth_dwa_turn = 0.0

        # Fall detection: require sustained low z to avoid false positives
        # from dynamic gait oscillation. Body dips during turns/walks are
        # temporary; real falls are sustained.
        self._fall_tick_count = 0

        # Gait parameter smoothing (EMA state for DWA mode)
        self._smooth_heading_mod = 1.0
        self._smooth_wz = 0.0
        self._decel_tick_count = 0

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

        # Pitch/roll diagnostics
        self._rp_log = []  # (step, roll_deg, pitch_deg, droll, dpitch, mode)

        # Pose glitch rejection: DDS buffer reuse can return stale/zeroed data
        self._cached_pose: tuple[float, float, float, float, float, float] | None = None

    def _get_robot_pose(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, yaw, z, roll, pitch) from simulation ground truth.

        Includes glitch rejection: if position jumps > 2m in a single tick
        (impossible for a walking robot at ~0.5 m/s), return the cached pose
        instead. This guards against CycloneDDS buffer reuse races in
        simulation_manager._on_world_physics.
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
        """Feed SLAM odometry with current sensor data.

        Uses IMU quaternion from L5 upward state, and body velocity
        from simulation (ground truth linvel rotated to body frame).
        The target game bypasses L5 for commands, so L5's commanded
        velocity is (0,0). We use true velocity as a stand-in for
        what leg odometry would provide on a real robot.
        """
        if self._odometry is None:
            return

        from types import SimpleNamespace

        # IMU quaternion from L5 upward state (real sensor)
        l5_state = self._sim.get_state()
        imu_quat = l5_state.imu_quaternion

        # Body velocity: transform world-frame linvel to body frame
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

        # Publish pose on DDS (if publisher available)
        if self._pose_pub is not None:
            self._publish_pose()

    def set_dwa_planner(self, planner) -> None:
        """Set DWA planner for obstacle-aware navigation.

        When set, the DWA planner replaces the heading-based controller
        in _tick_walk. This is demo-only — training uses the heading
        controller to avoid the v12 train/demo parity issue.
        """
        self._dwa_planner = planner

    def set_obstacle_bodies(self, names: list[str]) -> None:
        """Register obstacle body names for ground-truth proximity checking.

        FOREMAN VALIDATION ONLY — not used by the robot's control loop.
        The robot avoids obstacles using its sensors (LiDAR → costmap → DWA).
        This uses MuJoCo physics ground truth as a referee to verify the
        robot actually maintained clearance.

        Requires obstacle bodies in SimulationManager's track_bodies list
        so their positions are published over DDS.
        """
        self._obstacle_bodies = names

    def set_pose_publisher(self, pub, msg_class, stamp_fn) -> None:
        """Set DDS publisher for rt/pose_estimate topic.

        Called from __main__.py after DDS init. Optional — if not called,
        SLAM still works but pose is not published on DDS.
        """
        self._pose_pub = pub
        self._PoseMsg = msg_class
        self._pose_stamp = stamp_fn

    def set_telemetry(self, telemetry_log) -> None:
        """Set JSONL telemetry logger for Layer 6 modules.

        When set, the game records SLAM drift, DWA decisions, and perception
        timing into a JSONL file for post-run analysis.
        """
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

        Always uses ground truth.  SLAM dead-reckoning drifts too fast
        for reliable navigation (yaw drift > 0.5 rad causes DWA goal
        vector to rotate so robot walks away from target).  SLAM still
        runs for drift analysis/telemetry; when loop-closure or
        magnetometer correction is added to Layer 6, switch back.
        """
        x, y, yaw, _, _, _ = self._get_robot_pose()
        return x, y, yaw

    @property
    def slam_trail(self) -> list[tuple[float, float]]:
        """SLAM estimated positions (10Hz), for visualization."""
        return self._slam_trail

    @property
    def truth_trail(self) -> list[tuple[float, float]]:
        """Ground truth positions (10Hz), for visualization."""
        return self._truth_trail

    def _check_fall(self, z: float) -> bool:
        """Check for sustained fall. Returns True if robot has fallen.

        Requires z below threshold for FALL_CONFIRM_TICKS consecutive ticks
        to avoid false positives from dynamic gait oscillation. Body height
        dips during turns/walks are temporary; real falls are sustained.
        """
        if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD:
            self._fall_tick_count += 1
            if self._fall_tick_count >= FALL_CONFIRM_TICKS:
                self._stats.falls += 1
                self._fall_tick_count = 0
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

            # Telemetry: SLAM drift
            if self._telemetry is not None:
                drift = math.sqrt((p.x - truth_x)**2 + (p.y - truth_y)**2)
                self._telemetry.record("slam", {
                    "drift": round(drift, 4),
                    "slam_x": round(p.x, 3), "slam_y": round(p.y, 3),
                    "truth_x": round(truth_x, 3), "truth_y": round(truth_y, 3),
                })

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
            # Pre-specified home pose: hold at keyframe directly.
            self._send_wheeled(0.0, 0.0)
        else:
            # Walking robots or wheeled without pre-specified home (e.g. b2w):
            # use L4 to bring robot to standing height.
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )
            self._send_l4(params)
        if self._step_count >= STARTUP_SETTLE_STEPS:
            x, y, yaw, z, roll, pitch = self._get_robot_pose()
            # Bail out if robot fell or settled unstably (DDS timing race).
            roll_deg = abs(math.degrees(roll))
            if z < NOMINAL_BODY_HEIGHT * FALL_THRESHOLD or roll_deg > 5.0:
                print(f"Startup FAILED: z={z:.3f}m  "
                      f"R={math.degrees(roll):+.1f}°  P={math.degrees(pitch):+.1f}°  "
                      f"(fell or unstable settle)")
                self._stats.falls += 1
                self._state = GameState.DONE
                return
            if WHEELED:
                if self._home_q is None:
                    # Capture home pose from L4 standing position
                    state = self._sim.get_robot_state()
                    self._home_q = [state.joint_positions[i] for i in range(12)]
                    print(f"Startup complete (wheeled, L4 pose): z={z:.3f}m  "
                          f"home_q=[{self._home_q[0]:.3f}, {self._home_q[1]:.3f}, {self._home_q[2]:.3f}]")
                else:
                    print(f"Startup complete (wheeled): z={z:.3f}m  home_q={WHEEL_HOME_Q[:3]}")
            else:
                print(f"Startup complete: z={z:.3f}m  R={math.degrees(roll):+.1f}°  "
                      f"P={math.degrees(pitch):+.1f}°")
            self._state = GameState.SPAWN_TARGET

    def _tick_spawn(self):
        """Spawn a new target and transition to walking."""
        if self._target_index >= self._num_targets:
            self._state = GameState.DONE
            return

        # Use SLAM pose for spawning (target relative to estimated position)
        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        if self._spawn_fn is not None:
            target = self._spawn_fn(nav_x, nav_y, nav_yaw, self._target_index)
            self._spawner._current_target = target
        else:
            target = self._spawner.spawn_relative(nav_x, nav_y, nav_yaw, angle_range=self._angle_range)
        self._target_index += 1
        self._target_step_count = 0
        self._stats.targets_spawned += 1

        # Move visual target marker (mocap body) to target position.
        # Place at robot body height so marker is visible at "eye level."
        # Capped at 0.30m to prevent tall robots from having targets too high.
        target_z = min(NOMINAL_BODY_HEIGHT, 0.30)
        try:
            self._sim.set_mocap_pos(0, [target.x, target.y, target_z])
        except (RuntimeError, AttributeError):
            pass  # No mocap body in scene (headless without target scene)

        dist = target.distance_to(nav_x, nav_y)
        src = "SLAM" if self._odometry else "truth"
        print(f"\n[target {self._target_index}/{self._num_targets}] "
              f"spawned at ({target.x:.1f}, {target.y:.1f})  dist={dist:.1f}m  (nav={src})")

        self._state = GameState.WALK_TO_TARGET

    def _tick_walk(self):
        """Walk toward target using L4 GaitParams directly.

        Matches GA training episode logic (episode.py:1771-1811):
        - Large heading error -> turn in place (L4 arc mode)
        - Small heading error -> walk with differential stride

        When SLAM odometry is active, heading/distance use the estimated
        pose. Fall detection still uses ground truth (z-height).
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        if abs(heading_err) > THETA_THRESHOLD:
            # TURN IN PLACE — L4 arc mode (layer_4/generator.py:100-132)
            wz = TURN_WZ if heading_err > 0 else -TURN_WZ
            params = self._L4GaitParams(
                gait_type='trot', turn_in_place=True, wz=wz,
                gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
                body_height=BODY_HEIGHT,
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
                body_height=BODY_HEIGHT,
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
            # Get velocity for direction analysis
            body = self._sim.get_body("base")
            vx = float(body.linvel[0]) if body else 0
            vy = float(body.linvel[1]) if body else 0
            v_heading = math.atan2(vy, vx) if (abs(vx) + abs(vy)) > 0.01 else 0
            slam_info = ""
            if self._odometry is not None:
                drift = math.sqrt((nav_x - x_truth)**2 + (nav_y - y_truth)**2)
                slam_info = f"  drift={drift:.3f}m"
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str}  dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"z={z:.2f}  R={r_deg:+.1f}° P={p_deg:+.1f}°  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  yaw={nav_yaw:+.2f}  "
                  f"v=({vx:+.2f},{vy:+.2f}) v_dir={v_heading:+.2f}  t={t:.1f}s"
                  f"{slam_info}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _tick_walk_dwa(self):
        """Walk toward target using DWA obstacle avoidance.

        DWA replans at 10Hz (every 10 ticks). Between replans, the last
        DWA command persists. Uses the perception pipeline's costmap for
        obstacle awareness. Falls back to heading controller if no
        costmap is available after 3 seconds.
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        dist = target.distance_to(nav_x, nav_y)
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        goal_behind = abs(heading_err) > math.pi / 2

        # Replan at 10Hz (every 10 ticks), or immediately if no result yet.
        # The immediate trigger on first tick ensures DWA activates right away
        # instead of falling through to _tick_walk() forever.
        if self._target_step_count % 10 == 0 or self._last_dwa_result is None:
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                # Goal in robot-centered frame
                goal_x = target.x - nav_x
                goal_y = target.y - nav_y
                # Rotate to robot frame
                c = math.cos(-nav_yaw)
                s = math.sin(-nav_yaw)
                goal_rx = c * goal_x - s * goal_y
                goal_ry = s * goal_x + c * goal_y

                result = self._dwa_planner.plan(costmap_q, goal_x=goal_rx, goal_y=goal_ry)

                # Smooth raw DWA turn to suppress frame-to-frame oscillation.
                # DWA flips sign when multiple arcs score similarly.
                self._smooth_dwa_turn += _DWA_TURN_ALPHA * (result.turn - self._smooth_dwa_turn)
                result.turn = self._smooth_dwa_turn

                # Goal-behind lock: when target is behind (heading_err > pi/2),
                # DWA oscillates because left and right arcs score similarly.
                # Lock the turn direction to face the target.
                if goal_behind:
                    result.turn = math.copysign(max(abs(result.turn), 0.5), heading_err)
                    self._smooth_dwa_turn = result.turn

                self._last_dwa_result = result
                self._goal_bearing = math.atan2(goal_ry, goal_rx)

                # Telemetry: DWA decision
                if self._telemetry is not None:
                    self._telemetry.record("dwa", {
                        "forward": round(result.forward, 3),
                        "turn": round(result.turn, 3),
                        "score": round(result.score, 3),
                        "n_feasible": result.n_feasible,
                        "goal_rx": round(goal_rx, 2),
                        "goal_ry": round(goal_ry, 2),
                        "dist": round(dist, 2),
                    })

            # Ground-truth obstacle proximity (foreman referee, not robot sensors).
            # Uses physics engine body positions via Layer 3's get_body() API.
            if self._obstacle_bodies and self._telemetry is not None:
                robot = self._sim.get_body("base")
                if robot is not None:
                    rx, ry = float(robot.pos[0]), float(robot.pos[1])
                    min_clearance = float('inf')
                    closest = ""
                    for name in self._obstacle_bodies:
                        obs = self._sim.get_body(name)
                        if obs is None:
                            continue
                        d = math.sqrt((rx - float(obs.pos[0]))**2 +
                                      (ry - float(obs.pos[1]))**2)
                        if d < min_clearance:
                            min_clearance = d
                            closest = name
                    if min_clearance < float('inf'):
                        self._telemetry.record("proximity", {
                            "min_clearance": round(min_clearance, 3),
                            "closest": closest,
                        })

        # Convert DWA result to L4 gait params
        if self._last_dwa_result is not None and self._last_dwa_result.n_feasible > 0:
            dwa = self._last_dwa_result

            # Reactive forward scan: probe costmap beyond DWA horizon and
            # modulate (forward, turn) before gait conversion.
            threat_active = False  # gates EMA smoothing (only near obstacles)
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                from .perception import reactive_scan
                scan = reactive_scan(costmap_q, dwa.forward, dwa.turn,
                                     goal_bearing=self._goal_bearing)
                heading_mod = scan.mod_forward
                turn_cmd = scan.mod_turn

                # A4: Avoidance commitment — hold direction for 2s to
                # prevent flickering when DWA oscillates ±0.1.
                now = self._target_step_count * CONTROL_DT
                if scan.threat > 0.3:
                    if (self._avoidance_sign == 0
                            or (now - self._avoidance_since) > 2.0):
                        self._avoidance_sign = 1 if turn_cmd >= 0 else -1
                        self._avoidance_since = now
                    # If scan output is weak, enforce committed direction
                    if abs(turn_cmd) < 0.2:
                        turn_cmd = self._avoidance_sign * 0.4 * scan.threat
                elif scan.threat < 0.1:
                    self._avoidance_sign = 0  # clear in open field

                threat_active = scan.threat > 0.1

                # Telemetry: reactive scan events (only when modulating)
                if self._telemetry is not None and scan.threat > 0.05:
                    self._telemetry.record("reactive_scan", {
                        "threat": round(scan.threat, 3),
                        "asymmetry": round(scan.asymmetry, 3),
                        "emergency": scan.emergency,
                        "dwa_fwd": round(dwa.forward, 3),
                        "dwa_turn": round(dwa.turn, 3),
                        "mod_fwd": round(scan.mod_forward, 3),
                        "mod_turn": round(turn_cmd, 3),
                        "smooth_fwd": round(self._smooth_heading_mod, 3),
                        "smooth_wz": round(self._smooth_wz, 3),
                        "avoidance_sign": self._avoidance_sign,
                    })
            else:
                heading_mod = dwa.forward
                turn_cmd = dwa.turn

            # Forward-turn coupling: decelerate for tight turns.
            # At full speed + max turn, L4 walk-mode differential stride can't
            # deliver the commanded turn rate (~0.4 rad/s actual vs 1.05 commanded).
            # Linear taper: heading_mod -> 0 at turn_cmd=1.0, which triggers
            # turn-in-place mode (pure pivot at full WZ_LIMIT).
            if abs(turn_cmd) > 0.4:
                turn_frac = min(1.0, (abs(turn_cmd) - 0.4) / 0.6)
                heading_mod = min(heading_mod, 1.0 - turn_frac)

            # --- Always-on heading_mod smoothing (AFTER coupling) ---
            # Context-dependent alphas: slow near obstacles, faster in open field
            if heading_mod < self._smooth_heading_mod:
                alpha_hm = _EMA_ALPHA_DOWN_OBSTACLE if threat_active else _EMA_ALPHA_DOWN_OPEN
            else:
                alpha_hm = _EMA_ALPHA_UP_OBSTACLE if threat_active else _EMA_ALPHA_UP_OPEN
            self._smooth_heading_mod += alpha_hm * (heading_mod - self._smooth_heading_mod)
            heading_mod = self._smooth_heading_mod

            # Speed-dependent turn rate limit: at full forward speed, cap wz
            # to 40% of WZ_LIMIT to prevent the gait from destabilizing.
            # At zero forward speed, full WZ_LIMIT is available for turning.
            wz_scale = 1.0 - 0.5 * heading_mod
            wz_limit = WZ_LIMIT * wz_scale
            wz = _clamp(turn_cmd * WZ_LIMIT, -wz_limit, wz_limit)

            # --- Always-on wz smoothing ---
            alpha_wz = _EMA_ALPHA_WZ_OBSTACLE if threat_active else _EMA_ALPHA_WZ_OPEN
            self._smooth_wz += alpha_wz * (wz - self._smooth_wz)
            wz = self._smooth_wz

            # --- Mode switch with decel guard ---
            # Require heading_mod to be low AND sustained for _MIN_DECEL_TICKS
            # before switching to turn-in-place. This prevents the 65kg B2 from
            # pitching forward when forward momentum is killed in a single tick.
            wants_turn = heading_mod < 0.1 and abs(wz) > 0.3
            if wants_turn:
                self._decel_tick_count += 1
            else:
                self._decel_tick_count = 0

            if wants_turn and self._decel_tick_count >= _MIN_DECEL_TICKS:
                # DWA says turn — use arc mode (decel guard satisfied).
                # Cap wz to TURN_WZ: the GA-evolved turn rate is safe for B2's
                # mass. WZ_LIMIT (1.5) is for walk-mode differential stride,
                # not for turn-in-place pivots which stress the stance legs.
                turn_wz = _clamp(wz, -TURN_WZ, TURN_WZ)
                params = self._L4GaitParams(
                    gait_type='trot', turn_in_place=True, wz=turn_wz,
                    gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                    duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
                    body_height=BODY_HEIGHT,
                )
            else:
                params = self._L4GaitParams(
                    gait_type='trot', step_length=STEP_LENGTH * heading_mod,
                    gait_freq=GAIT_FREQ, step_height=STEP_HEIGHT,
                    duty_cycle=DUTY_CYCLE, stance_width=STANCE_WIDTH, wz=wz,
                    body_height=BODY_HEIGHT,
                )
        else:
            # Emergency stop or no costmap yet — stand still
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )

        self._send_l4(params)

        # Telemetry
        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            dwa_info = ""
            if self._last_dwa_result is not None:
                d = self._last_dwa_result
                dwa_info = f"  dwa=({d.forward:.2f},{d.turn:.2f}) score={d.score:.2f} feasible={d.n_feasible}"
            behind_str = f"  BEHIND({math.degrees(heading_err):.0f}deg)" if goal_behind else ""
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"DWA  dist={dist:.1f}m  smooth_fwd={self._smooth_heading_mod:.2f}  "
                  f"decel={self._decel_tick_count}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s{dwa_info}{behind_str}")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _tick_walk_wheeled(self):
        """Drive toward target using differential wheel torque.

        No gait generator — legs hold rigid at home pose via PD,
        wheels get pure torque for forward drive + differential steering.
        """
        x_truth, y_truth, yaw_truth, z, roll, pitch = self._get_robot_pose()
        target = self._spawner.current_target
        self._target_step_count += 1

        # Fall detection uses ground truth (sustained check)
        if self._check_fall(z):
            return

        # Navigation uses SLAM pose when available
        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        # Forward: proportional to alignment², taper near target.
        # Squaring alignment prioritizes turning: at 60° off, fwd drops to 6%
        # instead of 25%. This prevents heavy robots from spiraling at high
        # heading errors — they turn first, then drive straight.
        alignment = max(0.0, math.cos(heading_err))
        alignment = alignment * alignment
        dist_taper = min(1.0, dist / 1.0)  # slow in last 1m
        fwd = WHEEL_FWD_TORQUE * alignment * dist_taper

        # Turn: proportional + saturated
        turn = _clamp(WHEEL_KP_YAW * heading_err, -WHEEL_MAX_TURN, WHEEL_MAX_TURN)

        self._send_wheeled(fwd, turn)

        # Log pitch/roll every step for diagnostics
        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "D"  # Drive (wheeled)
        if len(self._rp_log) > 0:
            prev = self._rp_log[-1]
            droll = (r_deg - prev[1]) / CONTROL_DT
            dpitch = (p_deg - prev[2]) / CONTROL_DT
        else:
            droll = dpitch = 0.0
        self._rp_log.append((self._step_count, r_deg, p_deg, droll, dpitch, mode))

        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            body = self._sim.get_body("base")
            vx = float(body.linvel[0]) if body else 0
            vy = float(body.linvel[1]) if body else 0
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"DRIVE  dist={dist:.1f}m  heading_err={heading_err:+.2f}rad  "
                  f"z={z:.2f}  fwd={fwd:.1f}Nm  turn={turn:.1f}Nm  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  v=({vx:+.2f},{vy:+.2f})  t={t:.1f}s")

        if dist < self._reach_threshold:
            self._on_reached()
            return

        if self._target_step_count >= self._timeout_steps:
            self._on_timeout()

    def _send_wheeled(self, fwd_torque, turn_torque, z=None):
        """Construct and send LowCmd for differential wheel drive.

        Legs (slots 0-11): rigid PD hold at home pose. With a properly
        splayed stance (wide support polygon + CoM analysis), no tracking
        or height correction is needed — the robot rolls like a car.
        Wheels (slots 12-15): pure torque, differential drive.
          FR=12, FL=13, RR=14, RL=15
          right wheels (FR, RR) = fwd + turn
          left wheels (FL, RL) = fwd - turn
        """
        cmd = self._make_low_cmd()
        # All 12 leg joints: rigid hold at home pose
        for i in range(12):
            cmd.motor_cmd[i].q = float(self._home_q[i])
            cmd.motor_cmd[i].kp = self._kp
            cmd.motor_cmd[i].kd = self._kd
        # Wheels: pure torque, differential drive
        right = fwd_torque + turn_torque
        left = fwd_torque - turn_torque
        for i, tau in [(12, right), (13, left), (14, right), (15, left)]:
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = tau
        self._stamp_cmd(cmd)
        self._sim._lowcmd_pub.Write(cmd)
        self._sim._t += CONTROL_DT

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
        if self._telemetry is not None:
            self._telemetry.close()
        self._print_summary()
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
        for label, code in [("WALK", "W"), ("TURN", "T"), ("DRIVE", "D")]:
            mode_rolls = [abs(r[1]) for r in self._rp_log if r[5] == code]
            mode_pitches = [abs(r[2]) for r in self._rp_log if r[5] == code]
            if mode_rolls:
                print(f"\n{label} ({len(mode_rolls)} steps):  "
                      f"avg|R|={sum(mode_rolls)/len(mode_rolls):.1f}°  "
                      f"avg|P|={sum(mode_pitches)/len(mode_pitches):.1f}°  "
                      f"max|R|={max(mode_rolls):.1f}°  max|P|={max(mode_pitches):.1f}°")

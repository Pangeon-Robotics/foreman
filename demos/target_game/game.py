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
STABILIZE_THRESHOLD = 0.0   # DISABLED — guard interrupts gait mid-stride and causes falls
STABILIZE_HOLD_TICKS = 50   # 0.5s neutral stand to recover balance

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
TIP_STEP_LENGTH = 0.10   # m — stride for differential TIP (minimal forward drift)
_TIP_WZ_SCALE = 2.0      # multiplier: gentle differential (one side walks, other stationary)

# --- Gait parameter smoothing (context-dependent EMA) ---
# Near obstacles: slow decel preserves smooth avoidance curves
_EMA_ALPHA_DOWN_OBSTACLE = 0.04   # tau ~0.25s — faster decel near obstacles, tracks DWA better
_EMA_ALPHA_UP_OBSTACLE = 0.12     # tau ~0.08s — faster recovery after clearing obstacle
# Open field: faster tracking prevents orbit, still smooth enough to prevent falls
_EMA_ALPHA_DOWN_OPEN = 0.08       # tau ~0.13s — 13 ticks decel, tracks DWA better
_EMA_ALPHA_UP_OPEN = 0.15         # tau ~0.07s — fast turn→walk recovery
# Wz smoothing
_EMA_ALPHA_WZ_OBSTACLE = 0.10     # tau ~0.1s
_EMA_ALPHA_WZ_OPEN = 0.15         # tau ~0.07s
# DWA turn smoothing: suppress frame-to-frame oscillation
_DWA_TURN_ALPHA = 0.10            # tau ~0.5s at 20Hz replan — moderate smoothing to suppress oscillation
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
        "TURN_FREQ": 2.0, "TURN_STEP_HEIGHT": 0.06,
        "TURN_DUTY_CYCLE": 0.60, "TURN_STANCE_WIDTH": 0.10,
        "TURN_WZ": 0.8, "THETA_THRESHOLD": 0.6,
        "V_REF": 0.30,
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
        "V_REF": 1.5,
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
        "V_REF": 1.5,
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
        "V_REF": 2.0,
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


STARTUP_SETTLE_STEPS = 50  # 0.5s at 100 Hz — let robot settle before first gait command


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
        self._path_critic = None    # PathCritic, set via set_path_critic()

        # Avoidance commitment — holds turn direction to prevent flickering
        self._avoidance_sign = 0       # +1 left, -1 right, 0 no commitment
        self._avoidance_since = 0.0    # tick time when direction was chosen
        self._avoidance_dist = 0.0     # dist-to-target when commitment started
        self._goal_bearing = 0.0       # cached goal bearing for reactive scan

        # DWA turn smoothing
        self._smooth_dwa_turn = 0.0

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

        # Stuck detection: force recovery when no progress in tight spaces
        self._stuck_check_dist = float('inf')  # distance at last check
        self._stuck_check_step = 0              # step of last check

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

    _PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"

    def _export_path(self, nav_x, nav_y, target_x, target_y):
        """Write A* path to temp file for headed viewer rendering.

        Computes shortest obstacle-free path from robot to target using
        the PathCritic's A* on the TSDF distance field.  Writes world-
        frame (x, y) pairs as flat float32 for the firmware viewer.
        """
        import struct

        path = None
        if self._path_critic is not None and self._path_critic._tsdf is not None:
            path = self._path_critic._astar_core(
                (nav_x, nav_y), (target_x, target_y), return_path=True,
            )

        if path is None or len(path) < 2:
            # No A* path — fall back to straight line
            path = [(nav_x, nav_y), (target_x, target_y)]

        # Subsample to ~0.3m spacing for viz (A* grid is 0.1m)
        filtered = [path[0]]
        for i in range(1, len(path)):
            dx = path[i][0] - filtered[-1][0]
            dy = path[i][1] - filtered[-1][1]
            if dx * dx + dy * dy >= 0.09:  # 0.3m squared
                filtered.append(path[i])
        # Always include the final point (the target)
        if filtered[-1] != path[-1]:
            filtered.append(path[-1])

        n = len(filtered)
        buf = bytearray(n * 8)
        for i, (wx, wy) in enumerate(filtered):
            struct.pack_into('ff', buf, i * 8, wx, wy)
        try:
            with open(self._PATH_VIZ_FILE, 'wb') as f:
                f.write(buf)
        except OSError:
            pass

    def set_path_critic(self, critic) -> None:
        """Set PathCritic for path quality evaluation."""
        self._path_critic = critic

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
                self._post_fall_settle = 200  # 2s settle before next target
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

        # Post-fall settle: send neutral stand to let robot recover balance.
        # Without this, the robot goes from lying on the ground to full-speed
        # walking in 1 tick, causing cascading falls.
        if self._post_fall_settle > 0:
            self._post_fall_settle -= 1
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )
            self._send_l4(params)
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
        self._min_target_dist = float('inf')
        self._slam_drift_latched = False
        self._in_tip_mode = False
        self._tip_start_step = 0
        self._tip_start_heading = 0.0
        self._tip_cooldown_until = 0
        self._reset_tip()
        # Reset EMA state so new target starts gently (ramps up via EMA)
        self._smooth_heading_mod = 0.0
        self._smooth_wz = 0.0
        self._decel_tick_count = 0
        self._smooth_dwa_turn = 0.0
        self._avoidance_sign = 0
        self._avoidance_dist = 0.0
        self._last_threat_level = 0.0
        self._current_waypoint = None
        self._stuck_check_dist = float('inf')
        self._stuck_check_step = 0
        self._progress_window_dist = float('inf')
        self._progress_window_step = 0
        self._dist_ring = []
        self._recent_regression = 0.0
        self._closest_approach = float('inf')
        self._reg_start_step = 0
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

        # Path critic: seed with initial position and target
        if self._path_critic is not None:
            self._path_critic.set_target(target.x, target.y)
            self._path_critic.record(nav_x, nav_y, t=0.0)

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

        # Path critic: record position at 10Hz
        if self._path_critic is not None and self._target_step_count % 10 == 0:
            self._path_critic.record(nav_x, nav_y, t=self._target_step_count * CONTROL_DT)

        if abs(heading_err) > THETA_THRESHOLD:
            # TURN IN PLACE — differential stride (zero lateral slippage)
            wz = (TURN_WZ if heading_err > 0 else -TURN_WZ) * _TIP_WZ_SCALE
            params = self._L4GaitParams(
                gait_type='trot', wz=wz, step_length=TIP_STEP_LENGTH,
                gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
                body_height=BODY_HEIGHT,
            )
        else:
            # WALK — L4 differential stride (layer_4/generator.py:134-164)
            self._reset_tip()
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
            is_tip = abs(heading_err) > THETA_THRESHOLD
            mode_str = "TIP" if is_tip else "WALK"
            if is_tip:
                sent_wz = TURN_WZ if heading_err > 0 else -TURN_WZ
                sent_step = 0.0
            else:
                sent_wz = wz
                sent_step = STEP_LENGTH * heading_mod
            slam_info = ""
            if self._odometry is not None:
                drift = math.sqrt((nav_x - x_truth)**2 + (nav_y - y_truth)**2)
                slam_info = f"  drift={drift:.3f}m"
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
                ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                            f"rg={_rg:.2f} reg={_reg:.1f}m")
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str:<7} dist={dist:.1f}m  "
                  f"h_err={math.degrees(heading_err):+.0f}deg  "
                  f"step={sent_step:.2f}  wz={sent_wz:+.2f}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                  f"{slam_info}{ato_info}")

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

        # Pre-fall stabilization: when body height drops below 75% of nominal,
        # immediately send neutral stand for 0.3s to recover balance.
        # This catches destabilization from TIP or walk-while-turning before
        # it becomes a full fall (which triggers at 50% threshold).
        if self._stabilize_countdown > 0:
            self._stabilize_countdown -= 1
            self._smooth_heading_mod = 0.0
            self._smooth_wz = 0.0
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )
            self._send_l4(params)
            return
        if z < NOMINAL_BODY_HEIGHT * STABILIZE_THRESHOLD:
            self._stabilize_countdown = STABILIZE_HOLD_TICKS
            self._in_tip_mode = False
            self._reset_tip()

        nav_x, nav_y, nav_yaw = self._get_nav_pose()
        dist = target.distance_to(nav_x, nav_y)
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        goal_behind = abs(heading_err) > math.pi / 2

        # Track minimum distance for SLAM drift detection.
        # Once drift is latched, stop updating min_dist (prevents the
        # oscillate-and-reset pattern where robot accidentally gets closer,
        # resets min_dist, then walks away again at full speed).
        if not self._slam_drift_latched:
            self._min_target_dist = min(self._min_target_dist, dist)
        target_time = self._target_step_count * CONTROL_DT
        # Drift detected: distance grew >2m beyond best approach, OR
        # target has been active >60s with no significant progress (circling).
        # Once latched, stays latched until next target spawn.
        # Heading_mod stays at 30% (not 0) so robot can still crawl toward
        # targets that are reachable despite moderate drift.
        if not self._slam_drift_latched:
            self._slam_drift_latched = (
                (dist - self._min_target_dist) > 10.0
                or (target_time > 90.0 and dist > self._min_target_dist + 5.0)
            )
        slam_drift_detected = self._slam_drift_latched

        # Path critic: record position at 10Hz
        if self._path_critic is not None and self._target_step_count % 10 == 0:
            self._path_critic.record(nav_x, nav_y, t=self._target_step_count * CONTROL_DT)

        # A* waypoint guidance: recompute at 1-2Hz depending on environment.
        # Near obstacles (low feasibility or high threat), recompute at 2Hz
        # for faster course correction. In open field, 1Hz suffices.
        wp_interval = 50 if (self._last_dwa_result is not None
                             and (self._last_dwa_result.n_feasible < 25
                                  or getattr(self, '_last_threat_level', 0) > 0.2)
                             ) else 100
        if (self._path_critic is not None and dist > 2.0
                and self._target_step_count % wp_interval == 0):
            wp = self._path_critic.plan_waypoints(
                nav_x, nav_y, target.x, target.y, lookahead=2.0,
                planning_radius=0.45)
            if wp is None:
                # Tight gap: fall back to evaluation radius
                wp = self._path_critic.plan_waypoints(
                    nav_x, nav_y, target.x, target.y, lookahead=2.0,
                    planning_radius=0.35)
            self._current_waypoint = wp

        # Replan at 20Hz (every 5 ticks), or immediately if no result yet.
        # The immediate trigger on first tick ensures DWA activates right away
        # instead of falling through to _tick_walk() forever.
        # Avoidance commitment (1.5s hold) suppresses oscillation; DWA needs
        # fast replans for responsive obstacle navigation.
        if self._target_step_count % 5 == 0 or self._last_dwa_result is None:
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                # Goal in robot-centered frame: use A* waypoint if available,
                # else raw target.  Waypoint guides DWA along the globally
                # optimal path instead of beelining toward the target.
                if self._current_waypoint is not None and dist > 2.0:
                    gx_world = self._current_waypoint[0] - nav_x
                    gy_world = self._current_waypoint[1] - nav_y
                else:
                    gx_world = target.x - nav_x
                    gy_world = target.y - nav_y
                # Rotate to robot frame
                c = math.cos(-nav_yaw)
                s = math.sin(-nav_yaw)
                goal_rx = c * gx_world - s * gy_world
                goal_ry = s * gx_world + c * gy_world

                result = self._dwa_planner.plan(costmap_q, goal_x=goal_rx, goal_y=goal_ry)

                # Light DWA turn smoothing to suppress frame-to-frame oscillation.
                self._smooth_dwa_turn += _DWA_TURN_ALPHA * (result.turn - self._smooth_dwa_turn)
                result.turn = self._smooth_dwa_turn

                # Goal-behind lock: when target is behind (heading_err > pi/2),
                # DWA oscillates because left and right arcs score similarly.
                # Lock the turn direction and suppress forward speed to prevent
                # the robot from walking away from the target while turning.
                if goal_behind:
                    result.turn = math.copysign(max(abs(result.turn), 0.8), heading_err)
                    self._smooth_dwa_turn = result.turn
                    # Suppress forward: cos²(heading_err) goes from 0 at ±90° to 0
                    # at ±180°.  This prevents full-speed walk-away while still
                    # allowing slight forward motion during the initial turn.
                    behind_factor = max(0.0, math.cos(heading_err)) ** 2
                    result.forward = result.forward * behind_factor

                self._last_dwa_result = result
                self._goal_bearing = math.atan2(goal_ry, goal_rx)

                # Export A* path to temp file for headed viewer rendering.
                # Run at 2Hz (every 50 ticks) — A* is heavier than arc export.
                if self._target_step_count % 50 == 0:
                    self._export_path(nav_x, nav_y, target.x, target.y)

                # Telemetry: DWA decision
                if self._telemetry is not None:
                    dwa_telemetry = {
                        "forward": round(result.forward, 3),
                        "turn": round(result.turn, 3),
                        "sent_wz": round(self._smooth_wz, 3),
                        "score": round(result.score, 3),
                        "n_feasible": result.n_feasible,
                        "goal_rx": round(goal_rx, 2),
                        "goal_ry": round(goal_ry, 2),
                        "dist": round(dist, 2),
                    }
                    # CurvatureDWAResult has extra fields
                    if hasattr(result, 'arc_length'):
                        dwa_telemetry["arc_length"] = round(result.arc_length, 2)
                    if hasattr(result, 'kappa'):
                        dwa_telemetry["kappa"] = round(result.kappa, 4)
                    self._telemetry.record("dwa", dwa_telemetry)

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

        # --- Close-range approach: bypass DWA oscillation pipeline ---
        # Within 2.0m, heading error changes fast (0.5m offset at 1m = 27°).
        # DWA multi-arc scoring oscillates, EMA can't track, robot orbits.
        # Use simple proportional heading control with distance-scaled speed.
        # When heading error is large (>THETA_THRESHOLD), turn in place first.
        if dist < 1.5:
            if abs(heading_err) > THETA_THRESHOLD:
                # Too far off — turn in place to re-orient before approaching
                turn_wz = _clamp(heading_err * KP_YAW, -TURN_WZ, TURN_WZ)
                self._smooth_wz += 0.15 * (turn_wz - self._smooth_wz)
                turn_wz = self._smooth_wz * _TIP_WZ_SCALE
                self._smooth_heading_mod = 0.0
                self._decel_tick_count = _MIN_DECEL_TICKS  # pre-satisfy guard
                self._in_tip_mode = True
                # Differential stride TIP: zero lateral slippage
                params = self._L4GaitParams(
                    gait_type='trot', wz=turn_wz, step_length=TIP_STEP_LENGTH,
                    gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                    duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
                    body_height=BODY_HEIGHT,
                )
                mode_str = "CLOSE-T"
            else:
                self._in_tip_mode = False
                self._reset_tip()
                wz = _clamp(heading_err * KP_YAW, -WZ_LIMIT, WZ_LIMIT)
                self._smooth_wz += 0.15 * (wz - self._smooth_wz)
                wz = self._smooth_wz
                # Speed proportional to distance, reduced when heading is off
                cos_heading = max(0.0, math.cos(heading_err))
                speed = min(1.0, dist / 1.5) * cos_heading
                # Update EMA state to prevent discontinuity on mode exit
                self._smooth_heading_mod = speed
                self._decel_tick_count = 0
                params = self._L4GaitParams(
                    gait_type='trot', step_length=STEP_LENGTH * speed,
                    gait_freq=GAIT_FREQ, step_height=STEP_HEIGHT,
                    duty_cycle=DUTY_CYCLE, stance_width=STANCE_WIDTH, wz=wz,
                    body_height=BODY_HEIGHT,
                )
                mode_str = "CLOSE"

            self._send_l4(params)

            if self._target_step_count % TELEMETRY_INTERVAL == 0:
                t = self._target_step_count * CONTROL_DT
                sent_step = 0.0 if mode_str == "CLOSE-T" else STEP_LENGTH * speed
                ato_info = ""
                if self._path_critic is not None:
                    _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
                    ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                                f"rg={_rg:.2f} reg={_reg:.1f}m")
                print(f"[target {self._target_index}/{self._num_targets}] "
                      f"{mode_str:<7} dist={dist:.1f}m  "
                      f"h_err={math.degrees(heading_err):+.0f}deg  "
                      f"step={sent_step:.2f}  wz={self._smooth_wz:+.2f}  "
                      f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                      f"{ato_info}")

            if dist < self._reach_threshold:
                self._on_reached()
                return
            if self._target_step_count >= self._timeout_steps:
                self._on_timeout()
            return

        # Convert DWA result to L4 gait params.
        #
        # Heading-proportional control is the base turn controller (matches
        # GA training path).  DWA provides obstacle avoidance: when some
        # arcs are blocked, blend DWA's turn direction to steer around them.
        if self._last_dwa_result is not None:
            dwa = self._last_dwa_result

            # --- Turn: heading-proportional + DWA obstacle avoidance ---
            heading_turn = _clamp(heading_err * KP_YAW, -1.0, 1.0)

            # Blend DWA turn as a nudge, not an override.
            # The DWA is a local planner — it picks the best 10m arc, but
            # can't reason about going AROUND obstacles.  When it overrides
            # heading control, it steers the robot away from the target
            # indefinitely.  Cap blend at 0.5 so heading control always
            # retains influence.  The reactive scan (1.5m) handles close-
            # range avoidance.
            n_total = self._dwa_planner._n_curvatures
            obstacle_fraction = 1.0 - (dwa.n_feasible / n_total)
            dwa_blend = min(0.5, obstacle_fraction)
            turn_cmd = (1.0 - dwa_blend) * heading_turn + dwa_blend * dwa.turn

            # --- Speed: heading alignment with obstacle braking ---
            cos_heading = max(0.0, math.cos(heading_err))
            heading_mod = max(0.3, cos_heading)

            # Reactive scan: reduce speed near obstacles
            threat_active = False
            costmap_q = self._perception.costmap_query if self._perception else None
            if costmap_q is not None:
                from .perception import reactive_scan
                scan = reactive_scan(costmap_q, heading_mod, turn_cmd,
                                     goal_bearing=self._goal_bearing)
                if scan.threat > 0.1:
                    heading_mod = min(heading_mod, scan.mod_forward)
                    threat_active = True
                    self._last_threat_level = scan.threat
                    self._last_threat_time = self._target_step_count * CONTROL_DT

            if slam_drift_detected:
                heading_mod = min(heading_mod, 0.3)

            # Smooth to prevent jerky changes
            self._smooth_heading_mod += 0.10 * (heading_mod - self._smooth_heading_mod)
            heading_mod = self._smooth_heading_mod

            wz = _clamp(turn_cmd * WZ_LIMIT, -WZ_LIMIT, WZ_LIMIT)
            self._smooth_wz += 0.15 * (wz - self._smooth_wz)
            wz = self._smooth_wz

            # TIP mode: turn in place when target is behind
            if goal_behind and abs(heading_err) > THETA_THRESHOLD:
                turn_wz = (TURN_WZ if heading_err > 0 else -TURN_WZ) * _TIP_WZ_SCALE
                params = self._L4GaitParams(
                    gait_type='trot', wz=turn_wz, step_length=TIP_STEP_LENGTH,
                    gait_freq=TURN_FREQ, step_height=TURN_STEP_HEIGHT,
                    duty_cycle=TURN_DUTY_CYCLE, stance_width=TURN_STANCE_WIDTH,
                    body_height=BODY_HEIGHT,
                )
                self._in_tip_mode = True
            else:
                self._in_tip_mode = False
                params = self._L4GaitParams(
                    gait_type='trot', step_length=STEP_LENGTH * heading_mod,
                    gait_freq=GAIT_FREQ, step_height=STEP_HEIGHT,
                    duty_cycle=DUTY_CYCLE, stance_width=STANCE_WIDTH, wz=wz,
                    body_height=BODY_HEIGHT,
                )
        else:
            # No DWA result yet (first tick) — stand still until planner runs.
            params = self._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=BODY_HEIGHT,
            )

        self._send_l4(params)

        # Telemetry — must capture what was actually sent to L4
        if self._target_step_count % TELEMETRY_INTERVAL == 0:
            t = self._target_step_count * CONTROL_DT
            # Determine actual mode sent
            tip_active = self._in_tip_mode
            if dist < self._reach_threshold + 1.5:
                mode_str = "CLOSE"
            elif tip_active:
                mode_str = "TIP"
            else:
                mode_str = "WALK"
            # What was actually sent to L4
            sent_wz = self._smooth_wz
            sent_fwd = self._smooth_heading_mod
            sent_step = STEP_LENGTH * sent_fwd if not tip_active else 0.0
            # DWA raw output
            dwa_info = ""
            if self._last_dwa_result is not None:
                d = self._last_dwa_result
                dwa_info = (f"  dwa=({d.forward:.2f},{d.turn:.2f}) "
                            f"score={d.score:.2f} feas={d.n_feasible}")
            behind_str = f"  BEHIND" if goal_behind else ""
            ato_info = ""
            if self._path_critic is not None:
                _a, _pe, _sr, _rg, _reg = self._path_critic.running_ato()
                ato_info = (f"  ATO={_a:.0f} pe={_pe:.0%} sr={_sr:.2f} "
                            f"rg={_rg:.2f} reg={_reg:.1f}m")
            print(f"[target {self._target_index}/{self._num_targets}] "
                  f"{mode_str:<7} dist={dist:.1f}m  "
                  f"h_err={math.degrees(heading_err):+.0f}deg  "
                  f"step={sent_step:.2f}  wz={sent_wz:+.2f}  "
                  f"pos=({nav_x:.1f}, {nav_y:.1f})  t={t:.1f}s"
                  f"{dwa_info}{behind_str}{ato_info}")

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
        # Clean up viz temp file
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

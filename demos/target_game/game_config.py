"""Target game configuration: constants, enums, and per-robot defaults."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


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
TURN_FREQ = 3.0
TURN_STEP_HEIGHT = 0.08
TURN_DUTY_CYCLE = 0.55
TURN_STANCE_WIDTH = 0.12
TURN_WZ = 1.0
THETA_THRESHOLD = 0.6
TIP_STEP_LENGTH = 0.10   # m — stride for differential TIP (minimal forward drift)
_TIP_WZ_SCALE = 1.0      # 1:1 — TURN_WZ is the actual TIP yaw rate (matches tip_demo.py)

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
        "TURN_FREQ": 3.0, "TURN_STEP_HEIGHT": 0.08,
        "TURN_DUTY_CYCLE": 0.55, "TURN_STANCE_WIDTH": 0.12,
        "TURN_WZ": 1.0, "THETA_THRESHOLD": 0.6,
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


def configure_for_robot(robot: str) -> None:
    """Set module-level game constants for the given robot.

    Updates both game_config and game modules so that code referencing
    constants from either location sees the updated values. The game
    module re-exports game_config constants via star-import, so its
    copies must also be patched.

    Call before _apply_genome so genome values override these defaults.
    """
    import sys
    mod = sys.modules[__name__]
    defaults = ROBOT_DEFAULTS.get(robot, ROBOT_DEFAULTS["b2"])
    for name, value in defaults.items():
        setattr(mod, name, value)
    # Also patch the game module (re-exports our constants via star-import)
    game_mod = sys.modules.get("foreman.demos.target_game.game")
    if game_mod is not None:
        for name, value in defaults.items():
            setattr(game_mod, name, value)
    if defaults.get('WHEELED'):
        print(f"Configured game for {robot} (wheeled): height={defaults['NOMINAL_BODY_HEIGHT']}m, "
              f"KP={defaults['KP_FULL']}, fwd_torque={defaults['WHEEL_FWD_TORQUE']}Nm")
    else:
        print(f"Configured game for {robot}: height={defaults['NOMINAL_BODY_HEIGHT']}m, "
              f"KP={defaults['KP_FULL']}, step_h={defaults['STEP_HEIGHT']}m")

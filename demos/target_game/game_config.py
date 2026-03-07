"""Target game configuration: constants, enums, and per-robot defaults.

Game-level constants only. Gait parameters, gain scheduling, and locomotion
behavior belong in Layer 5. Navigation parameters belong in Layer 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# --- Game constants ---

CONTROL_DT = 0.01          # 100 Hz (headed gold standard)
REACH_DISTANCE = 0.5        # m
TARGET_TIMEOUT_STEPS = 6000  # 60 seconds at 100 Hz
TELEMETRY_INTERVAL = 100    # steps between prints (1 Hz at 100 Hz)
STARTUP_SETTLE_STEPS = 50   # 0.5s at 100 Hz

# --- Fall detection ---

NOMINAL_BODY_HEIGHT = 0.465 # m (B2 default, set by configure_for_robot)
FALL_THRESHOLD = 0.5        # fraction of nominal height
FALL_CONFIRM_TICKS = 20     # consecutive ticks below threshold to confirm fall (0.2s)

# --- Wheeled robot parameters ---
# Used by navigator_helper.py for proportional steering torque computation.
# Actual wheel command execution goes through L5's send_wheel_command().

WHEELED = False
WHEEL_FWD_TORQUE = 2.0
WHEEL_KP_YAW = 4.0
WHEEL_MAX_TURN = 12.0

# --- Per-robot defaults ---
# Only game-level constants (fall detection height, wheeled config).
# Gait parameters are owned by Layer 5 robot configs.

ROBOT_DEFAULTS = {
    "b2": {
        "NOMINAL_BODY_HEIGHT": 0.465,
    },
    "go2": {
        "NOMINAL_BODY_HEIGHT": 0.27,
    },
    "go2w": {
        "WHEELED": True,
        "NOMINAL_BODY_HEIGHT": 0.34,
        "WHEEL_FWD_TORQUE": 2.0,
        "WHEEL_KP_YAW": 4.0,
        "WHEEL_MAX_TURN": 6.0,
    },
    "b2w": {
        "WHEELED": True,
        "NOMINAL_BODY_HEIGHT": 0.46,
        "WHEEL_FWD_TORQUE": 15.0,
        "WHEEL_KP_YAW": 8.0,
        "WHEEL_MAX_TURN": 12.0,
    },
}


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

    Only sets game-level constants (fall detection height, wheeled config).
    Gait parameters are owned by Layer 5.
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
        print(f"Configured game for {robot} (wheeled): "
              f"height={defaults['NOMINAL_BODY_HEIGHT']}m")
    else:
        print(f"Configured game for {robot}: "
              f"height={defaults['NOMINAL_BODY_HEIGHT']}m")

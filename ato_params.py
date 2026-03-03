"""ATO optimization parameters, bounds, and helper functions.

Extracted from ato_optimize.py to keep files under 400 lines.
"""
import json

# V_REF for B2 — must match path_critic.py
V_REF = 2.0

# Parameter definitions: (initial, min, max, step)
PARAM_BOUNDS = {
    "STEP_LENGTH":     (0.30, 0.10, 0.70, 0.05),
    "GAIT_FREQ":       (1.5,  1.0,  6.0,  0.5),
    "STEP_HEIGHT":     (0.07, 0.03, 0.15, 0.01),
    "DUTY_CYCLE":      (0.65, 0.50, 0.80, 0.05),
    "TURN_WZ":         (1.0,  0.5,  2.0,  0.1),
    "THETA_THRESHOLD": (0.6,  0.3,  0.8,  0.05),
    "KP_YAW":          (2.0,  1.0,  5.0,  0.5),
    # Turn params (keep stable during speed ramp)
    "TURN_FREQ":       (3.0,  1.5,  5.0,  0.5),
    "TURN_STEP_HEIGHT": (0.08, 0.04, 0.12, 0.01),
    "TURN_DUTY_CYCLE": (0.55, 0.45, 0.65, 0.05),
    "TURN_STANCE_WIDTH": (0.12, 0.08, 0.20, 0.02),
}

# Speed ramp progression: (STEP_LENGTH, GAIT_FREQ, STEP_HEIGHT)
# Ratio STEP_LENGTH/GAIT_FREQ ~0.13-0.16, STEP_HEIGHT ~23% of STEP_LENGTH
SPEED_RAMP = [
    (0.40, 2.5, 0.09),
    (0.45, 3.0, 0.10),
    (0.50, 3.5, 0.11),
    (0.55, 4.0, 0.12),
    (0.60, 4.5, 0.13),
    (0.65, 5.0, 0.14),
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def theoretical_speed(step_length: float, gait_freq: float) -> float:
    """Estimate theoretical max forward speed from gait params."""
    return step_length * gait_freq


def make_genome_json(params: dict) -> dict:
    """Create a v12-format genome JSON from parameter dict."""
    return {
        "genome": {
            "GAIT_FREQ": params["GAIT_FREQ"],
            "STEP_LENGTH": params["STEP_LENGTH"],
            "STEP_HEIGHT": params["STEP_HEIGHT"],
            "DUTY_CYCLE": params["DUTY_CYCLE"],
            "STANCE_WIDTH": params.get("STANCE_WIDTH", 0.0),
            "KP_YAW": params["KP_YAW"],
            "WZ_LIMIT": params.get("WZ_LIMIT", 1.5),
            "TURN_FREQ": params["TURN_FREQ"],
            "TURN_STEP_HEIGHT": params["TURN_STEP_HEIGHT"],
            "TURN_DUTY_CYCLE": params["TURN_DUTY_CYCLE"],
            "TURN_STANCE_WIDTH": params["TURN_STANCE_WIDTH"],
            "TURN_WZ": params["TURN_WZ"],
            "THETA_THRESHOLD": params["THETA_THRESHOLD"],
        },
        "generation": "ato_opt",
        "fitness": 0.0,
    }


def get_initial_params() -> dict:
    """Return the initial parameter set (current B2 defaults)."""
    return {name: bounds[0] for name, bounds in PARAM_BOUNDS.items()}

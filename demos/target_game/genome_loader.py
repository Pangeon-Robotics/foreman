"""Genome loading and expansion for GA-evolved parameter sets.

Handles v9 (8 params), v10 (8 + turn joint deltas), and v12 (13 params)
genome formats. Expands compact genome representations to full Layer 5
constants and patches game module globals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _expand_v9_genome(params: dict) -> dict:
    """Expand v9 genome (8 params) to all Layer 5 constants.

    Maps unified gait/steering params to both walk and turn-in-place
    constants, matching training/ga/episode.py:inject_genome_v9().
    """
    freq = params.get("FREQ", 1.5)
    step_length = params.get("STEP_LENGTH", 0.20)
    step_height = params.get("STEP_HEIGHT", 0.06)
    duty_cycle = params.get("DUTY_CYCLE", 0.55)
    walk_speed = params.get("WALK_SPEED", 1.0)
    kp_yaw = params.get("KP_YAW", 2.0)
    wz_limit = params.get("WZ_LIMIT", 1.5)
    stance_width = params.get("STANCE_WIDTH", 0.0)

    return {
        # Walk gait
        "BASE_FREQ": freq, "MIN_FREQ": freq, "MAX_FREQ": freq,
        "FREQ_SCALE": 0.0,
        "STEP_LENGTH_SCALE": step_length,
        "MAX_STEP_LENGTH": step_length,
        "TROT_STEP_HEIGHT": step_height,
        "WALK_STEP_HEIGHT": step_height,
        # Turn gait (unified with walk)
        "TURN_IN_PLACE_FREQ": freq,
        "TURN_IN_PLACE_STEP_HEIGHT": step_height,
        "TURN_IN_PLACE_STEP_LENGTH": step_length,
        "TURN_IN_PLACE_DUTY_CYCLE": duty_cycle,
        "TURN_IN_PLACE_WZ_SCALE": 1.0,
        "TURN_IN_PLACE_STANCE_WIDTH": stance_width,
        # Steering
        "KP_YAW": kp_yaw, "WALK_SPEED": walk_speed,
        "WALK_SPEED_MIN": 0.0, "TURN_WZ_LIMIT": wz_limit,
    }


def _is_v12_genome(params: dict) -> bool:
    """Detect v12 sovereign genome by presence of GAIT_FREQ key."""
    return "GAIT_FREQ" in params


def _expand_v12_genome(params: dict) -> dict:
    """Expand v12 genome (13 params) to all Layer 5 constants.

    Maps sovereign walk + turn genes to L5's TURN_IN_PLACE_* and
    walk constants. WALK_SPEED is derived from STEP_LENGTH * GAIT_FREQ.
    """
    gait_freq = params.get("GAIT_FREQ", 1.5)
    step_length = params.get("STEP_LENGTH", 0.20)
    step_height = params.get("STEP_HEIGHT", 0.06)
    duty_cycle = params.get("DUTY_CYCLE", 0.55)
    stance_width = params.get("STANCE_WIDTH", 0.0)
    kp_yaw = params.get("KP_YAW", 2.0)
    wz_limit = params.get("WZ_LIMIT", 1.5)

    return {
        # Walk gait
        "BASE_FREQ": gait_freq, "MIN_FREQ": gait_freq, "MAX_FREQ": gait_freq,
        "FREQ_SCALE": 0.0,
        "STEP_LENGTH_SCALE": step_length,
        "MAX_STEP_LENGTH": step_length,
        "TROT_STEP_HEIGHT": step_height,
        "WALK_STEP_HEIGHT": step_height,
        # Turn gait (v12 has separate turn params)
        "TURN_IN_PLACE_FREQ": params.get("TURN_FREQ", 1.0),
        "TURN_IN_PLACE_STEP_HEIGHT": params.get("TURN_STEP_HEIGHT", 0.06),
        "TURN_IN_PLACE_STEP_LENGTH": step_length,
        "TURN_IN_PLACE_DUTY_CYCLE": params.get("TURN_DUTY_CYCLE", 0.55),
        "TURN_IN_PLACE_WZ_SCALE": 1.0,
        "TURN_IN_PLACE_STANCE_WIDTH": params.get("TURN_STANCE_WIDTH", 0.04),
        # Steering (WALK_SPEED derived from step_length * gait_freq)
        "KP_YAW": kp_yaw,
        "WALK_SPEED": step_length * gait_freq,
        "WALK_SPEED_MIN": 0.0,
        "TURN_WZ_LIMIT": wz_limit,
    }


def _is_v10_genome(params: dict) -> bool:
    """Detect v10 genome by presence of turn joint delta keys."""
    return "P1_FL_HIP" in params


def _apply_genome(genome_path: str) -> None:
    """Load a GA-evolved genome JSON and patch game + Layer 5 parameters.

    v12+ genomes: dual patch -- game module constants for L4-direct walk
    control, plus L5 expansion for the startup/stand phase.
    v9/v10 genomes: expand to L5 constants and patch L5 modules.
    """
    genome = json.loads(Path(genome_path).read_text())

    # Flatten: handle both export format {"locomotion": {...}, "steering": {...}}
    # and GA checkpoint format {"genome": {...}, "fitness": ...}
    params = {}
    if "genome" in genome and isinstance(genome["genome"], dict):
        params.update(genome["genome"])
    for group in ("locomotion", "steering"):
        if group in genome:
            params.update(genome[group])

    from . import game as game_mod
    from . import game_config as game_cfg

    if _is_v12_genome(params):
        # v12+ sovereign genome: dual patch

        # 1. Patch game module constants for L4-direct walk/turn control.
        v12_game_params = [
            "GAIT_FREQ", "STEP_LENGTH", "STEP_HEIGHT", "DUTY_CYCLE", "STANCE_WIDTH",
            "KP_YAW", "WZ_LIMIT", "TURN_FREQ", "TURN_STEP_HEIGHT", "TURN_DUTY_CYCLE",
            "TURN_STANCE_WIDTH", "TURN_WZ", "THETA_THRESHOLD",
        ]
        for name in v12_game_params:
            if name in params:
                setattr(game_cfg, name, params[name])
                setattr(game_mod, name, params[name])
                print(f"  game.{name} = {params[name]:.4f}")

        # 2. Patch extended control parameters
        for key, val in params.items():
            if key not in v12_game_params and hasattr(game_cfg, key):
                setattr(game_cfg, key, val)
                setattr(game_mod, key, val)
                print(f"  game.{key} = {val}")

        # 3. Expand to L5 constants for startup/stand phase
        expanded = _expand_v12_genome(params)
        theta = params.get("THETA_THRESHOLD", "?")
        turn_wz = params.get("TURN_WZ", "?")
        n_extra = sum(1 for k in params if k not in v12_game_params and hasattr(game_cfg, k))
        extra_str = f" + {n_extra} control params" if n_extra else ""
        print(f"  v12 sovereign genome: 13 genes{extra_str}, theta_threshold={theta}, turn_wz={turn_wz}")

    elif _is_v10_genome(params):
        # Extract walk subset (the 8 v9 params) for L5 expansion
        walk_keys = ["FREQ", "STEP_LENGTH", "STEP_HEIGHT", "DUTY_CYCLE",
                     "WALK_SPEED", "KP_YAW", "WZ_LIMIT", "STANCE_WIDTH"]
        walk_params = {k: params[k] for k in walk_keys if k in params}
        expanded = _expand_v9_genome(walk_params)
        # Merge: expanded walk + original turn/timing genes
        for k, v in params.items():
            if k not in expanded:
                expanded[k] = v
        turn_count = sum(1 for k in params if k.startswith(("P1_", "P2_")))
        timing = {k: params[k] for k in ["T_PHASE1", "T_PHASE2", "T_PHASE3"] if k in params}
        print(f"  v10 turn genes: {turn_count} joint deltas, timing={timing}")

    elif "FREQ" in params:
        expanded = _expand_v9_genome(params)

    else:
        expanded = params

    # Patch Layer 5 config.defaults and downstream modules
    locomotion_params = [
        "BASE_FREQ", "FREQ_SCALE", "MAX_FREQ", "MIN_FREQ",
        "STEP_LENGTH_SCALE", "MAX_STEP_LENGTH", "TROT_STEP_HEIGHT",
        "WALK_STEP_HEIGHT",
        "TURN_IN_PLACE_FREQ", "TURN_IN_PLACE_STEP_HEIGHT",
        "TURN_IN_PLACE_STEP_LENGTH", "TURN_IN_PLACE_DUTY_CYCLE",
        "TURN_IN_PLACE_WZ_SCALE", "TURN_IN_PLACE_STANCE_WIDTH",
    ]
    defaults_mod = sys.modules.get("config.defaults")
    downstream_mods = ["velocity_mapper", "gait_selector", "locomotion",
                       "transition", "terrain_gait"]

    for name in locomotion_params:
        if name not in expanded:
            continue
        if defaults_mod and hasattr(defaults_mod, name):
            setattr(defaults_mod, name, expanded[name])
        for mod_name in downstream_mods:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, name):
                setattr(mod, name, expanded[name])

    gen = genome.get("generation", "?")
    fitness = genome.get("fitness", "?")
    print(f"Applied genome gen={gen}, fitness={fitness}")

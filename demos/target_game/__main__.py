"""Entry point: python -m foreman.demos.target_game"""
from __future__ import annotations

# CycloneDDS 0.10.4 preload: the pip-installed cyclonedds (0.10.5) has an
# incompatible libddsc. Preload the system 0.10.4 before anything imports it.
import ctypes as _ctypes
import os as _os
_SYS_DDSC = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"
if _os.path.exists(_SYS_DDSC):
    _ctypes.CDLL(_SYS_DDSC, mode=_ctypes.RTLD_GLOBAL)

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup for cross-layer imports.
#
# Layers 3, 4, and 5 each have a `config/` package with different contents.
# Layer 5's simulation.py swaps config modules at import time, but the
# SimulationManager constructor calls config functions at runtime from
# both Layer 3 and Layer 4. Since both layers use
# importlib.import_module("config.b2") which resolves via sys.modules,
# the wrong config.b2 gets loaded.
#
# Fix: pre-populate each layer's _active_config cache by loading their
# config.b2 modules directly by file path, bypassing sys.modules entirely.
# Also replace Layer 4's load_config call in the constructor to use the
# pre-loaded config.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[3]  # workspace root
_layer5 = str(_root / "layer_5")

if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)

# Import Layer 5 API (triggers import-time config swap for Layer 4/3)
from simulation import SimulationManager
from config.defaults import MotionCommand  # noqa: F401 — captured for game.py

from .game import TargetGame
from .utils import load_module_by_path, patch_layer_configs


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

    Detects v9 genomes (have "FREQ" key) and expands them to all Layer 5
    constants. Patches steering constants in game.py and locomotion
    constants in Layer 5's config.defaults + downstream modules.
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

    # v12 genomes: sovereign — expand to L5 constants (no joint angles)
    # v10 genomes: expand walk params via v9 path, turn genes used directly
    # v9 genomes: expand to all Layer 5 constants
    if _is_v12_genome(params):
        expanded = _expand_v12_genome(params)
        params = expanded
        theta = genome.get("genome", {}).get("THETA_THRESHOLD", params.get("THETA_THRESHOLD", "?"))
        turn_wz = genome.get("genome", {}).get("TURN_WZ", params.get("TURN_WZ", "?"))
        print(f"  v12 sovereign genome: 13 genes, theta_threshold={theta}, turn_wz={turn_wz}")
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
        params = expanded
        # Print turn gene summary
        turn_count = sum(1 for k in params if k.startswith(("P1_", "P2_")))
        timing = {k: params[k] for k in ["T_PHASE1", "T_PHASE2", "T_PHASE3"] if k in params}
        print(f"  v10 turn genes: {turn_count} joint deltas, timing={timing}")
    elif "FREQ" in params:
        params = _expand_v9_genome(params)

    # Patch game module steering constants
    from . import game as game_mod
    steering_params = [
        "KP_YAW", "WALK_SPEED", "WALK_SPEED_MIN", "TURN_WZ_LIMIT",
    ]
    for name in steering_params:
        if name in params:
            setattr(game_mod, name, params[name])
            print(f"  game.{name} = {params[name]:.4f}")

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
        if name not in params:
            continue
        if defaults_mod and hasattr(defaults_mod, name):
            setattr(defaults_mod, name, params[name])
        for mod_name in downstream_mods:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, name):
                setattr(mod, name, params[name])

    gen = genome.get("generation", "?")
    fitness = genome.get("fitness", "?")
    print(f"Applied genome gen={gen}, fitness={fitness}")


def main():
    parser = argparse.ArgumentParser(description="Random Target Game")
    parser.add_argument("--robot", default="b2")
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to GA-evolved genome JSON file")
    parser.add_argument("--full-circle", action="store_true",
                        help="Spawn targets at any angle (uniform -π to π)")
    parser.add_argument("--domain", type=int, default=None,
                        help="DDS domain ID (avoids conflict with running firmware)")
    args = parser.parse_args()

    print(f"Starting target game: robot={args.robot}, targets={args.targets}")

    if args.genome:
        print(f"\nApplying evolved genome: {args.genome}")
        _apply_genome(args.genome)

    # Pre-populate config caches to avoid runtime namespace collision
    patch_layer_configs(args.robot, Path(_root))

    # If --domain specified, monkey-patch FirmwareLauncher to use it
    # (avoids DDS session conflict with other running firmware)
    if args.domain is not None:
        from launcher import FirmwareLauncher
        _orig_init = FirmwareLauncher.__init__
        _domain = args.domain
        def _patched_init(self, *a, **kw):
            kw["domain"] = _domain
            _orig_init(self, *a, **kw)
        FirmwareLauncher.__init__ = _patched_init

    # Use scene with mocap target body for visual markers
    scene_path = str(_root / "layers_1_2" / "unitree_robots" / args.robot / "scene_target.xml")
    sim = SimulationManager(
        args.robot,
        headless=args.headless,
        scene=scene_path,
        track_bodies=["target"],
    )
    sim.start()
    try:
        import math
        if args.full_circle:
            angle_range = (-math.pi, math.pi)
        else:
            # Exclude front cone (±30°): spawn to the sides and behind
            angle_range = [(math.pi / 6, math.pi), (-math.pi, -math.pi / 6)]
        game = TargetGame(
            sim,
            num_targets=args.targets,
            seed=args.seed,
            angle_range=angle_range,
        )
        stats = game.run()
        print(f"\nFinal: {stats.targets_reached}/{stats.targets_spawned} "
              f"reached ({stats.success_rate:.0%})")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()

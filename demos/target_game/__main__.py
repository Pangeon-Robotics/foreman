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


def _apply_genome(genome_path: str) -> None:
    """Load a GA-evolved genome JSON and patch game + Layer 5 parameters.

    Patches steering constants in game.py and locomotion constants in
    Layer 5's config.defaults + downstream modules.
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

    # Patch game module steering constants
    from . import game as game_mod
    steering_params = [
        "KP_YAW", "HEADING_FULL_SPEED", "HEADING_SLOW_SPEED",
        "HEADING_TURN_ONLY", "WALK_SPEED", "WALK_SPEED_MIN",
    ]
    for name in steering_params:
        if name in params:
            setattr(game_mod, name, params[name])
            print(f"  game.{name} = {params[name]:.4f}")

    # Patch Layer 5 config.defaults and downstream modules
    locomotion_params = [
        "BASE_FREQ", "FREQ_SCALE", "MAX_FREQ", "MIN_FREQ",
        "STEP_LENGTH_SCALE", "MAX_STEP_LENGTH", "TROT_STEP_HEIGHT",
    ]
    defaults_mod = sys.modules.get("config.defaults")
    downstream_mods = ["velocity_mapper", "gait_selector", "locomotion",
                       "transition", "terrain_gait"]

    for name in locomotion_params:
        if name not in params:
            continue
        if defaults_mod and hasattr(defaults_mod, name):
            setattr(defaults_mod, name, params[name])
            print(f"  config.defaults.{name} = {params[name]:.4f}")
        for mod_name in downstream_mods:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, name):
                setattr(mod, name, params[name])

    # NOTE: genome "KP"/"KD" are GPU tier PD gains (100-3000, with torque
    # clamp). Layer 5 uses "KP_FULL"/"KD_FULL" (5000/141.4) at firmware
    # scale. Do NOT map between them — different control architectures.

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

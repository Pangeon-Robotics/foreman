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
import importlib.util
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
_layer4 = str(_root / "layer_4")
_layer3 = str(_root / "layer_3")

if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)

# Import Layer 5 API (triggers import-time config swap for Layer 4/3)
from simulation import SimulationManager
from config.defaults import MotionCommand  # noqa: F401 — captured for game.py

from .game import TargetGame


def _load_module_by_path(name: str, path: str):
    """Load a Python module by file path without polluting sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_layer_configs(robot: str) -> None:
    """Pre-populate _active_config for Layer 3 and Layer 4 config modules.

    After Layer 5's import-time swap, each layer's get_active_config() function
    has _active_config=None in its globals. When called at runtime, it tries
    importlib.import_module("config.b2") which collides between layers.

    We bypass this by:
    1. Loading each layer's config/b2.py directly by file path
    2. Injecting the result into get_active_config's __globals__["_active_config"]
    3. Replacing Layer 4's _load_l4_config with a no-op (config already cached)
    """
    # Load each layer's config/robot.py by file path (no sys.modules)
    l3_b2 = _load_module_by_path(
        "_l3_config_b2", str(Path(_layer3) / "config" / f"{robot}.py"))
    l4_b2 = _load_module_by_path(
        "_l4_config_b2", str(Path(_layer4) / "config" / f"{robot}.py"))

    # Patch Layer 3's config: policy.cartesian and ik share the same globals
    cart_mod = sys.modules.get("policy.cartesian")
    if cart_mod and hasattr(cart_mod, "get_active_config"):
        cart_mod.get_active_config.__globals__["_active_config"] = l3_b2

    # Patch Layer 4's config: generator uses its own config globals
    gen_mod = sys.modules.get("generator")
    if gen_mod and hasattr(gen_mod, "get_active_config"):
        gen_mod.get_active_config.__globals__["_active_config"] = l4_b2

    # Replace Layer 4's _load_l4_config with a no-op — config is already cached.
    # Without this, Layer 4's SimulationManager.__init__ calls load_config(robot)
    # which does importlib.import_module("config.b2") and gets the wrong one.
    l4_sim = sys.modules.get("_l4_simulation")
    if l4_sim:
        l4_sim._load_l4_config = lambda robot: l4_b2


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
    _patch_layer_configs(args.robot)

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
        angle_range = (-math.pi, math.pi) if args.full_circle else (-math.pi / 2, math.pi / 2)
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

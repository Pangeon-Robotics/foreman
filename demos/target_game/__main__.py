"""Entry point: python -m foreman.demos.target_game"""
from __future__ import annotations

import argparse
import importlib.util
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


def main():
    parser = argparse.ArgumentParser(description="Random Target Game")
    parser.add_argument("--robot", default="b2")
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Starting target game: robot={args.robot}, targets={args.targets}")

    # Pre-populate config caches to avoid runtime namespace collision
    _patch_layer_configs(args.robot)

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
        game = TargetGame(
            sim,
            num_targets=args.targets,
            seed=args.seed,
        )
        stats = game.run()
        print(f"\nFinal: {stats.targets_reached}/{stats.targets_spawned} "
              f"reached ({stats.success_rate:.0%})")
    finally:
        sim.stop()


if __name__ == "__main__":
    main()

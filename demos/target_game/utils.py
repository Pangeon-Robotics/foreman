"""Cross-layer utilities shared between foreman and training.

Quaternion math, angle helpers, and config-patching functions used by
the target game demo and the GA training pipeline.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def quat_to_yaw(quat) -> float:
    """Extract yaw from [w, x, y, z] quaternion."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_rpy(quat) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw from [w, x, y, z] quaternion."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def load_module_by_path(name: str, path: str):
    """Load a Python module by file path without polluting sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_layer_configs(robot: str, workspace_root: Path) -> None:
    """Pre-populate _active_config for Layer 3 and Layer 4 config modules.

    After Layer 5's import-time swap, each layer's get_active_config() function
    has _active_config=None in its globals. When called at runtime, it tries
    importlib.import_module("config.b2") which collides between layers.

    We bypass this by:
    1. Loading each layer's config/robot.py directly by file path
    2. Injecting the result into get_active_config's __globals__["_active_config"]
    3. Replacing Layer 4's _load_l4_config with a no-op (config already cached)

    Parameters
    ----------
    robot : str
        Robot name (e.g. "b2", "go2").
    workspace_root : Path
        Absolute path to the workspace root (parent of layer directories).
    """
    layer3 = workspace_root / "layer_3"
    layer4 = workspace_root / "layer_4"

    # Load each layer's config/robot.py by file path (no sys.modules)
    l3_cfg_path = str(layer3 / "config" / f"{robot}.py")
    l4_cfg_path = str(layer4 / "config" / f"{robot}.py")

    try:
        l3_b2 = load_module_by_path("_l3_config_b2", l3_cfg_path)
    except FileNotFoundError:
        print(f"Warning: Layer 3 config not found: {l3_cfg_path}")
        l3_b2 = None

    try:
        l4_b2 = load_module_by_path("_l4_config_b2", l4_cfg_path)
    except FileNotFoundError:
        print(f"Warning: Layer 4 config not found: {l4_cfg_path}")
        l4_b2 = None

    # Patch Layer 3's config: policy.cartesian and ik share the same globals
    if l3_b2 is not None:
        cart_mod = sys.modules.get("policy.cartesian")
        if cart_mod and hasattr(cart_mod, "get_active_config"):
            cart_mod.get_active_config.__globals__["_active_config"] = l3_b2

    # Patch Layer 4's config: generator uses its own config globals
    if l4_b2 is not None:
        gen_mod = sys.modules.get("generator")
        if gen_mod and hasattr(gen_mod, "get_active_config"):
            gen_mod.get_active_config.__globals__["_active_config"] = l4_b2

    # Replace Layer 4's _load_l4_config with a no-op -- config is already cached.
    # Without this, Layer 4's SimulationManager.__init__ calls load_config(robot)
    # which does importlib.import_module("config.b2") and gets the wrong one.
    if l4_b2 is not None:
        l4_sim = sys.modules.get("_l4_simulation")
        if l4_sim:
            l4_sim._load_l4_config = lambda robot: l4_b2

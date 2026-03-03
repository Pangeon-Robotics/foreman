"""Scene XML parsing utilities for obstacle discovery.

Foreman-level utilities that read MuJoCo scene XML to discover obstacle
bodies and their geometries for ground-truth proximity checking.
"""
from __future__ import annotations

from pathlib import Path


def _find_obstacle_bodies(scene_path: Path) -> list[str]:
    """Parse scene XML to discover obstacle body names (obs_*).

    Foreman-level utility -- reads the scene to learn what obstacles exist
    so it can track their physics state for ground-truth proximity checking.
    """
    import xml.etree.ElementTree as ET
    names = []
    try:
        tree = ET.parse(scene_path)
        for body in tree.iter("body"):
            name = body.get("name", "")
            if name.startswith("obs_"):
                names.append(name)
    except (ET.ParseError, FileNotFoundError):
        pass
    return names


def _find_obstacle_geoms(scene_path: Path) -> list[dict]:
    """Parse scene XML to extract obstacle geom volumes.

    Finds bodies whose names start with 'obs_' or 'wall_'.
    Returns list of dicts with 'type' ('box'|'cylinder'), 'pos' (x,y,z),
    and 'size' (half-extents for box, radius+half-height for cylinder).
    Used by debug_server (viewer wireframes) and test_occupancy (ground truth).
    """
    import xml.etree.ElementTree as ET
    _PREFIXES = ("obs_", "wall_")
    obstacles = []
    try:
        tree = ET.parse(scene_path)
        for body in tree.iter("body"):
            name = body.get("name", "")
            if not any(name.startswith(p) for p in _PREFIXES):
                continue
            pos_str = body.get("pos", "0 0 0")
            pos = tuple(float(v) for v in pos_str.split())
            for geom in body.findall("geom"):
                gtype = geom.get("type", "sphere")
                size_str = geom.get("size", "0.25")
                size = tuple(float(v) for v in size_str.split())
                obstacles.append({
                    "type": gtype,
                    "pos": pos,
                    "size": size,
                })
    except (ET.ParseError, FileNotFoundError):
        pass
    return obstacles

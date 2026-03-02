"""Procedural scene generation for randomized obstacle scenarios.

Generates MuJoCo XML scenes with random obstacle layouts. The base scene
structure (robot include, floor, skybox, lights, target marker) comes from
the existing scattered scene template; only the obstacle bodies are replaced
with procedurally generated ones.

Usage:
    from foreman.demos.target_game.scene_gen import generate_scattered_scene
    path = generate_scattered_scene("b2", seed=42)
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple


class Obstacle(NamedTuple):
    """A generated obstacle with position and geometry."""
    name: str
    x: float
    y: float
    geom_type: str  # "cylinder" or "box"
    # For cylinder: (radius, half_height). For box: (hx, hy, hz).
    size: tuple[float, ...]


_SCENE_TEMPLATE = """\
<mujoco model="b2 scenario: scattered (seed={seed})">
  <include file="{robot}.xml"/>

  <statistic center="5 0 0.5" extent="10"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="obstacle" rgba="0.6 0.6 0.6 1" specular="0.3" shininess="0.5"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Target marker: mocap body (no dynamics, no collision) -->
    <body name="target" mocap="true" pos="5 0 0.1">
      <geom type="sphere" size="0.15" rgba="1 0.2 0.2 0.8" contype="0" conaffinity="0"/>
    </body>

{obstacles}
  </worldbody>
</mujoco>
"""

# Field bounds for obstacle placement
_FIELD_X_MIN = 1.5
_FIELD_X_MAX = 9.5
_FIELD_Y_MIN = -3.5
_FIELD_Y_MAX = 3.5

# Obstacle geometry ranges
_CYL_R_MIN, _CYL_R_MAX = 0.15, 0.35
_CYL_HH_MIN, _CYL_HH_MAX = 0.30, 0.50
_BOX_H_MIN, _BOX_H_MAX = 0.15, 0.35

# Minimum center-to-center distance between obstacles
_MIN_SPACING = 1.5

# Number of obstacles
_N_MIN, _N_MAX = 15, 20


def _generate_obstacles(rng: random.Random, n: int) -> list[Obstacle]:
    """Generate n non-overlapping obstacles with random positions and shapes."""
    obstacles: list[Obstacle] = []
    max_attempts = n * 50  # prevent infinite loop

    attempts = 0
    while len(obstacles) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(_FIELD_X_MIN, _FIELD_X_MAX)
        y = rng.uniform(_FIELD_Y_MIN, _FIELD_Y_MAX)

        # Check minimum spacing against existing obstacles
        too_close = False
        for obs in obstacles:
            dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            if dist < _MIN_SPACING:
                too_close = True
                break
        if too_close:
            continue

        # Random shape: ~50% cylinder, ~50% box
        name = f"obs_{len(obstacles) + 1:02d}"
        if rng.random() < 0.5:
            r = rng.uniform(_CYL_R_MIN, _CYL_R_MAX)
            hh = rng.uniform(_CYL_HH_MIN, _CYL_HH_MAX)
            obstacles.append(Obstacle(name, x, y, "cylinder", (r, hh)))
        else:
            hx = rng.uniform(_BOX_H_MIN, _BOX_H_MAX)
            hy = rng.uniform(_BOX_H_MIN, _BOX_H_MAX)
            hz = rng.uniform(_CYL_HH_MIN, _CYL_HH_MAX)  # same height range
            obstacles.append(Obstacle(name, x, y, "box", (hx, hy, hz)))

    return obstacles


def _obstacle_to_xml(obs: Obstacle) -> str:
    """Convert an Obstacle to a MuJoCo XML body element."""
    if obs.geom_type == "cylinder":
        r, hh = obs.size
        z = hh  # center at half-height so it sits on the floor
        size_str = f"{r:.2f} {hh:.2f}"
    else:
        hx, hy, hz = obs.size
        z = hz
        size_str = f"{hx:.2f} {hy:.2f} {hz:.2f}"
    return (
        f'    <body name="{obs.name}" pos="{obs.x:.2f} {obs.y:.2f} {z:.2f}">\n'
        f'      <geom type="{obs.geom_type}" size="{size_str}" '
        f'material="obstacle" contype="1" conaffinity="1"/>\n'
        f'    </body>'
    )


def generate_scattered_scene(
    robot: str, seed: int, output_dir: str | None = None,
) -> str:
    """Generate a scattered obstacle scene with random positions.

    Parameters
    ----------
    robot : str
        Robot model name (b2, go2, etc.).
    seed : int
        Random seed for reproducible obstacle placement.
    output_dir : str or None
        Directory for the generated XML file. Defaults to the robot's
        Assets directory so ``<include file="b2.xml"/>`` resolves.

    Returns
    -------
    str
        Absolute path to the generated scene XML.
    """
    rng = random.Random(seed)
    n_obstacles = rng.randint(_N_MIN, _N_MAX)
    obstacles = _generate_obstacles(rng, n_obstacles)

    obstacle_xml = "\n".join(_obstacle_to_xml(obs) for obs in obstacles)
    scene_xml = _SCENE_TEMPLATE.format(
        seed=seed, robot=robot, obstacles=obstacle_xml,
    )

    if output_dir is None:
        # Place next to robot XML so <include file="b2.xml"/> resolves
        workspace = Path(__file__).resolve().parents[3]
        output_dir = str(workspace / "Assets" / "unitree_robots" / robot)

    out_path = Path(output_dir) / f"scene_scattered_{seed}.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(scene_xml)
    return str(out_path)

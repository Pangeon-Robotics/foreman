"""Scenario testing for obstacle avoidance.

Registry of progressively harder scenarios with automated critics
that evaluate robot behavior. Each scenario specifies a MuJoCo scene,
target spawning parameters, and pass criteria for critics.

Usage:
    python -m foreman.demos.target_game.scenarios --robot b2
    python -m foreman.demos.target_game.scenarios --robot b2 --scenario corridor
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_ASSETS = Path(__file__).resolve().parents[4] / "Assets" / "unitree_robots"


@dataclass
class ScenarioDefinition:
    """Definition of a single test scenario."""
    name: str
    scene_xml: str           # filename in Assets/unitree_robots/{robot}/
    num_targets: int
    target_seed: int
    timeout_per_target: float  # seconds per target before timeout
    full_circle: bool        # spawn targets in all directions?
    has_obstacles: bool = True  # scene has obstacles (enables DWA + perception)
    use_slam: bool = True      # enable SLAM odometry
    min_dist: float = 3.0
    max_dist: float = 6.0
    angle_range: tuple[float, float] | None = None  # override full_circle logic
    pass_criteria: dict = field(default_factory=dict)
    spawn_fn_factory: Callable[[int], Callable] | None = None  # (seed) -> spawn_fn

    def scene_path(self, robot: str) -> Path:
        """Resolve absolute path to the scene XML for the given robot."""
        return _ASSETS / robot / self.scene_xml


def _make_backforth_spawn_fn(
    seed: int,
    field_near_x: float = 2.0,
    field_far_x: float = 9.0,
    y_range: tuple[float, float] = (-2.0, 2.0),
):
    """Create a spawn function that alternates targets across an obstacle field.

    Even targets (0, 2, ...) land beyond the far edge, odd targets (1, 3, ...)
    land before the near edge, forcing the robot to traverse the obstacle field
    in both directions.
    """
    from ..target import Target

    rng = random.Random(seed)

    def spawn_fn(robot_x: float, robot_y: float, robot_yaw: float, idx: int) -> Target:
        y = rng.uniform(y_range[0], y_range[1])
        if idx % 2 == 0:
            # Far side: beyond obstacle field
            x = field_far_x + rng.uniform(0.5, 1.5)
        else:
            # Near side: before obstacle field (back toward origin)
            x = field_near_x - rng.uniform(1.0, 2.0)
        heading = math.atan2(y - robot_y, x - robot_x)
        return Target(x=x, y=y, heading=heading)

    return spawn_fn


SCENARIOS: dict[str, ScenarioDefinition] = {
    "open": ScenarioDefinition(
        name="open",
        scene_xml="scene_scenario_open.xml",
        num_targets=3,
        target_seed=42,
        timeout_per_target=60.0,
        full_circle=True,
        has_obstacles=False,  # baseline: no obstacles, no DWA
        pass_criteria={
            "target_success_rate": 1.0,
            "max_falls": 0,
            "max_slam_drift_mean": 0.5,
            "max_slam_drift_final": 1.0,
            "forward_progress_pct": 0.80,
        },
    ),
    "scattered": ScenarioDefinition(
        name="scattered",
        scene_xml="scene_scenario_scattered.xml",
        num_targets=4,
        target_seed=42,
        timeout_per_target=120.0,
        full_circle=False,
        min_dist=3.0,
        max_dist=5.0,
        # Back-and-forth: targets alternate between far side and near side
        # of the obstacle field (x=3..8), forcing traversal every time.
        spawn_fn_factory=lambda seed: _make_backforth_spawn_fn(
            seed, field_near_x=2.0, field_far_x=9.0, y_range=(-2.0, 2.0),
        ),
        pass_criteria={
            "target_success_rate": 0.25,  # at least 1 of 4 (back-and-forth is hard)
            "max_falls": 3,
            "max_slam_drift_mean": 0.5,
            "max_slam_drift_final": 2.0,
            "min_dwa_feasible_mean": 15,
            "max_consecutive_estops": 5,  # tight spaces cause brief 0-feasible
            "min_obstacle_clearance": 0.5,  # center-to-center (obs radius + robot half-width)
        },
    ),
    "corridor": ScenarioDefinition(
        name="corridor",
        scene_xml="scene_scenario_corridor.xml",
        num_targets=3,
        target_seed=42,
        timeout_per_target=120.0,
        full_circle=False,
        angle_range=(-math.pi / 6, math.pi / 6),  # forward cone only
        min_dist=4.0,
        max_dist=7.0,
        pass_criteria={
            "target_success_rate": 0.33,  # at least 1 of 3
            "max_falls": 2,
            "max_slam_drift_mean": 0.5,
            "min_dwa_feasible_mean": 15,
            "min_obstacle_clearance": 0.5,
        },
    ),
    "L_wall": ScenarioDefinition(
        name="L_wall",
        scene_xml="scene_scenario_L_wall.xml",
        num_targets=2,
        target_seed=42,
        timeout_per_target=120.0,
        full_circle=False,
        angle_range=(-math.pi / 4, math.pi / 4),  # forward 90-degree cone
        min_dist=4.0,
        max_dist=6.0,
        pass_criteria={
            "target_success_rate": 0.5,  # at least 1 of 2
            "max_falls": 0,
            "min_dwa_feasible_mean": 15,
            "min_obstacle_clearance": 0.5,
        },
    ),
    "dead_end": ScenarioDefinition(
        name="dead_end",
        scene_xml="scene_scenario_dead_end.xml",
        num_targets=2,
        target_seed=42,
        timeout_per_target=120.0,
        full_circle=False,
        angle_range=(-math.pi / 3, math.pi / 3),  # wider cone — targets may be to the side
        min_dist=4.0,
        max_dist=6.0,
        pass_criteria={
            "target_success_rate": 0.5,
            "max_falls": 0,
            "min_dwa_feasible_mean": 10,
            "min_obstacle_clearance": 0.5,
        },
    ),
    "dense": ScenarioDefinition(
        name="dense",
        scene_xml="scene_scenario_dense.xml",
        num_targets=2,
        target_seed=42,
        timeout_per_target=180.0,
        full_circle=True,
        min_dist=3.0,
        max_dist=5.0,
        pass_criteria={
            "target_success_rate": 0.5,
            "max_falls": 1,
            "min_dwa_feasible_mean": 10,
            "min_obstacle_clearance": 0.5,
        },
    ),
}

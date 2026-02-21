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
from dataclasses import dataclass, field
from pathlib import Path

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

    def scene_path(self, robot: str) -> Path:
        """Resolve absolute path to the scene XML for the given robot."""
        return _ASSETS / robot / self.scene_xml


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
        num_targets=3,
        target_seed=42,
        timeout_per_target=90.0,
        full_circle=True,
        pass_criteria={
            "target_success_rate": 1.0,
            "max_falls": 0,
            "max_slam_drift_mean": 0.5,
            "max_slam_drift_final": 1.0,
            "forward_progress_pct": 0.70,
            "min_dwa_feasible_mean": 20,
        },
    ),
    "corridor": ScenarioDefinition(
        name="corridor",
        scene_xml="scene_scenario_corridor.xml",
        num_targets=2,
        target_seed=42,
        timeout_per_target=120.0,
        full_circle=False,
        angle_range=(-math.pi / 6, math.pi / 6),  # forward cone only
        min_dist=4.0,
        max_dist=7.0,
        pass_criteria={
            "target_success_rate": 1.0,
            "max_falls": 0,
            "max_slam_drift_mean": 0.5,
            "min_dwa_feasible_mean": 15,
            "max_dwa_oscillation_per_target": 8,
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
            "max_falls": 0,
            "min_dwa_feasible_mean": 10,
        },
    ),
}

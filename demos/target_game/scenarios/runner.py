"""Scenario runner: orchestrates execution and evaluation.

Constructs args for run_game(), invokes it per scenario, runs critics,
and collects results. Each scenario runs in-process sequentially (DDS
and SimulationManager are created/destroyed per run).
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace

from . import ScenarioDefinition
from .critics import CriticResult, run_all_critics


@dataclass
class ScenarioResult:
    """Result from running a single scenario."""
    scenario_name: str
    game_result: object | None = None  # GameRunResult
    critic_results: list[CriticResult] = field(default_factory=list)
    error: str | None = None


class ScenarioRunner:
    """Orchestrates scenario execution and evaluation.

    Parameters
    ----------
    robot : str
        Robot model (b2, go2, etc.).
    genome : str or None
        Path to genome JSON file.
    domain : int
        DDS domain ID (default 2, avoids firmware conflicts).
    headless : bool
        Run without MuJoCo viewer.
    """

    def __init__(
        self,
        robot: str = "b2",
        genome: str | None = None,
        domain: int = 2,
        headless: bool = True,
        viewer: bool = False,
    ):
        self._robot = robot
        self._genome = genome
        self._domain = domain
        self._headless = headless
        self._viewer = viewer

    def run_all(
        self,
        scenarios: dict[str, ScenarioDefinition],
        seed_override: int | None = None,
        targets_override: int | None = None,
    ) -> dict[str, ScenarioResult]:
        """Run all scenarios and return results keyed by name."""
        self._seed_override = seed_override
        self._targets_override = targets_override
        results = {}
        total = len(scenarios)
        for i, (name, scenario) in enumerate(scenarios.items(), 1):
            seed = seed_override if seed_override is not None else scenario.target_seed
            n_targets = targets_override if targets_override is not None else scenario.num_targets
            print(f"\n{'='*60}")
            print(f"SCENARIO [{i}/{total}]: {name}")
            print(f"  scene: {scenario.scene_xml}")
            print(f"  targets: {n_targets}, seed: {seed}")
            print(f"  timeout: {scenario.timeout_per_target}s/target")
            print(f"{'='*60}")
            result = self._run_one(name, scenario)
            results[name] = result

            # Quick summary after each scenario
            if result.error:
                print(f"\n>>> {name}: ERROR - {result.error}")
            else:
                n_pass = sum(1 for c in result.critic_results if c.passed)
                n_total = len(result.critic_results)
                status = "PASS" if n_pass == n_total else "FAIL"
                print(f"\n>>> {name}: {status} ({n_pass}/{n_total} critics)")

        return results

    def _run_one(self, name: str, scenario: ScenarioDefinition) -> ScenarioResult:
        """Run a single scenario: game + critics."""
        # Import run_game lazily (triggers DDS preload and Layer 5 imports)
        from ..__main__ import run_game

        # Build spawn_fn if scenario has a factory
        seed = (self._seed_override if self._seed_override is not None
                else scenario.target_seed)
        spawn_fn = None
        if scenario.spawn_fn_factory is not None:
            spawn_fn = scenario.spawn_fn_factory(seed)

        # Build args namespace matching what run_game expects
        args = SimpleNamespace(
            robot=self._robot,
            targets=(self._targets_override if self._targets_override is not None
                     else scenario.num_targets),
            headless=self._headless,
            seed=seed,
            genome=self._genome,
            full_circle=scenario.full_circle,
            domain=self._domain,
            slam=scenario.use_slam,
            obstacles=scenario.has_obstacles,
            scene_path=str(scenario.scene_path(self._robot)),
            timeout_per_target=scenario.timeout_per_target,
            min_dist=scenario.min_dist,
            max_dist=scenario.max_dist,
            angle_range=scenario.angle_range,
            spawn_fn=spawn_fn,
            viewer=self._viewer,
        )

        try:
            game_result = run_game(args)

            # Run all critics
            critic_results = run_all_critics(
                game_result,
                scenario.pass_criteria,
                game_result.telemetry_path,
            )

            return ScenarioResult(
                scenario_name=name,
                game_result=game_result,
                critic_results=critic_results,
            )
        except Exception as e:
            traceback.print_exc()
            return ScenarioResult(
                scenario_name=name,
                game_result=None,
                error=str(e),
            )

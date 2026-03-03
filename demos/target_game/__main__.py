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
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup for cross-layer imports.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[3]  # workspace root
_layer5 = str(_root / "layer_5")

if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)

from simulation import SimulationManager
from config.defaults import MotionCommand  # noqa: F401

_l4_sim = sys.modules["_l4_simulation"]
L4GaitParams = _l4_sim.GaitParams

from .game import TargetGame, configure_for_robot, GameStatistics, CONTROL_DT, TARGET_TIMEOUT_STEPS
from .utils import load_module_by_path, patch_layer_configs
from .genome_loader import _apply_genome
from .scene_parser import _find_obstacle_bodies, _find_obstacle_geoms


@dataclass
class GameRunResult:
    """Result from a single game run, consumed by scenario critics."""
    stats: GameStatistics
    telemetry_path: Path | None = None
    slam_trail: list[tuple[float, float]] = field(default_factory=list)
    truth_trail: list[tuple[float, float]] = field(default_factory=list)
    perception_stats: dict | None = None
    ato_score: float | None = None
    occupancy_accuracy: dict | None = None
    tsdf: object | None = None
    scene_path: str | None = None
    scores: dict | None = None


def _cleanup_stale_data(domain: int | None = None) -> None:
    """Remove stale temp files and processes from previous runs."""
    import glob
    import subprocess
    import time

    if domain is not None:
        result = subprocess.run(
            ["pgrep", "-af", "firmware_sim.py"],
            capture_output=True, text=True)
        if result.stdout.strip():
            killed = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                pid, cmdline = parts
                if f"--domain {domain}" in cmdline:
                    killed.append(pid)
                    subprocess.run(["kill", "-9", pid],
                                   capture_output=True)
            if killed:
                print(f"  [CLEANUP] killing {len(killed)} firmware PIDs on domain {domain}: {killed}")
                time.sleep(2.0)
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.5)
    else:
        result = subprocess.run(["pgrep", "-f", "firmware_sim.py"],
                                capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"  [CLEANUP] killing {len(pids)} firmware PIDs: {pids}")
            subprocess.run(["pkill", "-9", "-f", "firmware_sim.py"],
                           capture_output=True)
            time.sleep(2.0)
        else:
            time.sleep(0.5)

    for pattern in [
        "/tmp/god_view_costmap.bin",
        "/tmp/god_view_tsdf.bin",
        "/tmp/god_view_path.bin",
        "/tmp/robot_view_tsdf.bin",
        "/tmp/robot_view_costmap.bin",
        "/tmp/dwa_best_arc.bin",
        "/tmp/god_view_tsdf_test.bin",
    ]:
        for f in glob.glob(pattern):
            try:
                _os.remove(f)
            except OSError:
                pass

    if domain is not None:
        session_file = f"/tmp/robo_sessions/b2_domain{domain}.json"
        try:
            _os.remove(session_file)
        except OSError:
            pass
    else:
        for f in glob.glob("/tmp/robo_sessions/*.json"):
            try:
                _os.remove(f)
            except OSError:
                pass


def run_game(args) -> GameRunResult:
    """Run target game with given configuration. Returns structured result."""
    _cleanup_stale_data(getattr(args, 'domain', None))

    print(f"Starting target game: robot={args.robot}, targets={args.targets}")
    configure_for_robot(args.robot)

    genome = getattr(args, 'genome', None)
    if genome:
        print(f"\nApplying evolved genome: {genome}")
        _apply_genome(genome)

    from . import game as game_mod
    from . import game_config as game_cfg
    if args.robot == "b2":
        if game_mod.TURN_WZ > 3.0:
            game_mod.TURN_WZ = 3.0
            game_cfg.TURN_WZ = 3.0
            print(f"  Safety cap: TURN_WZ capped to {game_mod.TURN_WZ} for B2")
        if game_mod.TURN_STANCE_WIDTH < 0.12:
            game_mod.TURN_STANCE_WIDTH = 0.12
            game_cfg.TURN_STANCE_WIDTH = 0.12
            print(f"  Safety cap: TURN_STANCE_WIDTH raised to {game_mod.TURN_STANCE_WIDTH} for B2")
        if game_mod.TURN_DUTY_CYCLE > 0.55:
            game_mod.TURN_DUTY_CYCLE = 0.55
            game_cfg.TURN_DUTY_CYCLE = 0.55
            print(f"  Safety cap: TURN_DUTY_CYCLE lowered to {game_mod.TURN_DUTY_CYCLE} for B2")

    patch_layer_configs(args.robot, Path(_root))

    domain = getattr(args, 'domain', None)
    if domain is not None:
        from launcher import FirmwareLauncher
        _orig_init = FirmwareLauncher.__init__
        _domain = domain
        def _patched_init(self, *a, **kw):
            kw["domain"] = _domain
            _orig_init(self, *a, **kw)
        FirmwareLauncher.__init__ = _patched_init

    odometry = None
    slam = getattr(args, 'slam', False)
    if slam:
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from layer_6.slam.odometry import Odometry
        odometry = Odometry()
        print("SLAM odometry enabled (Phase 1: IMU yaw + body velocity)")

    telemetry_log = None
    telem_path = None
    if slam:
        from layer_6.telemetry import TelemetryLog
        telem_dir = _root / "tmp" / "telemetry"
        telem_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        telem_path = telem_dir / f"target_game_{args.robot}_{int(_time.time())}.jsonl"
        telemetry_log = TelemetryLog(telem_path)
        print(f"Telemetry: {telem_path}")

    scene_path_override = getattr(args, 'scene_path', None)
    if scene_path_override:
        scene_path = Path(scene_path_override)
    elif getattr(args, 'obstacles', False):
        scene_path = _root / "layers_1_2" / "unitree_robots" / args.robot / "scene_obstacles.xml"
        if not scene_path.exists():
            print(f"Warning: scene_obstacles.xml not found for {args.robot}, falling back")
            scene_path = _root / "layers_1_2" / "unitree_robots" / args.robot / "scene_target.xml"
    else:
        scene_path = _root / "layers_1_2" / "unitree_robots" / args.robot / "scene_target.xml"
    if not scene_path.exists():
        scene_path = _root / "layers_1_2" / "unitree_robots" / args.robot / "scene.xml"
        print(f"Warning: scene_target.xml not found for {args.robot}, using scene.xml")

    obstacle_bodies = _find_obstacle_bodies(scene_path)
    track_bodies = ["target"] + obstacle_bodies
    print(f"Scene: {scene_path} ({len(obstacle_bodies)} obstacles)")

    headless = getattr(args, 'headless', False)
    sim = SimulationManager(
        args.robot, headless=headless, scene=str(scene_path),
        track_bodies=track_bodies, domain=domain)
    sim.start()
    perception = None
    try:
        import math
        full_circle = getattr(args, 'full_circle', False)
        if full_circle:
            angle_range = (-math.pi, math.pi)
        else:
            angle_range = [(math.pi / 6, math.pi), (-math.pi, -math.pi / 6)]

        angle_range_override = getattr(args, 'angle_range', None)
        if angle_range_override is not None:
            angle_range = angle_range_override

        timeout_steps = TARGET_TIMEOUT_STEPS
        timeout_per_target = getattr(args, 'timeout_per_target', None)
        if timeout_per_target is not None:
            timeout_steps = int(timeout_per_target / CONTROL_DT)

        game = TargetGame(
            sim, L4GaitParams=L4GaitParams,
            make_low_cmd=_l4_sim._make_low_cmd, stamp_cmd=_l4_sim._stamp_cmd,
            num_targets=args.targets, seed=getattr(args, 'seed', None),
            angle_range=angle_range, odometry=odometry,
            min_dist=getattr(args, 'min_dist', 3.0),
            max_dist=getattr(args, 'max_dist', 6.0),
            timeout_steps=timeout_steps)

        game._headless = args.headless
        if obstacle_bodies:
            game.set_obstacle_bodies(obstacle_bodies)
        game.set_scene_path(str(scene_path))

        spawn_fn = getattr(args, 'spawn_fn', None)
        if spawn_fn is not None:
            game._spawn_fn = spawn_fn
        if telemetry_log is not None:
            game.set_telemetry(telemetry_log)

        from .game_setup import setup_perception, compute_occupancy, compute_scores
        perception = setup_perception(args, game, sim, odometry,
                                      obstacle_bodies, scene_path)

        if game._path_critic is None:
            from .path_critic import PathCritic
            game.set_path_critic(PathCritic(robot=args.robot, robot_radius=0.35))

        stats = game.run()
        ato = game._path_critic.aggregate_ato() if game._path_critic else None
        occ_accuracy = compute_occupancy(perception, args, scene_path)
        scores = compute_scores(game, perception, args)

        return GameRunResult(
            stats=stats, telemetry_path=telem_path,
            slam_trail=list(game.slam_trail),
            truth_trail=list(game.truth_trail),
            perception_stats=perception.stats if perception is not None else None,
            ato_score=ato, occupancy_accuracy=occ_accuracy,
            tsdf=perception._tsdf if perception is not None else None,
            scene_path=str(scene_path), scores=scores)
    finally:
        if perception is not None:
            perception.shutdown()
        sim.stop()


def main():
    parser = argparse.ArgumentParser(description="Random Target Game")
    parser.add_argument("--robot", default="b2")
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to GA-evolved genome JSON file")
    parser.add_argument("--full-circle", action="store_true",
                        help="Spawn targets at any angle (uniform -pi to pi)")
    parser.add_argument("--domain", type=int, default=None,
                        help="DDS domain ID (avoids conflict with running firmware)")
    parser.add_argument("--slam", action="store_true",
                        help="Use SLAM odometry instead of ground-truth position")
    parser.add_argument("--obstacles", action="store_true",
                        help="Use scene_obstacles.xml with static obstacles")
    parser.add_argument("--viewer", action="store_true",
                        help="Start TCP debug server for Godot TSDF viewer")
    args = parser.parse_args()

    result = run_game(args)

    print(f"\nFinal: {result.stats.targets_reached}/{result.stats.targets_spawned} "
          f"reached ({result.stats.success_rate:.0%})")

    if result.perception_stats is not None:
        ps = result.perception_stats
        if ps.get("builds", 0) > 0:
            print(f"\nPerception: {ps['builds']} costmaps built, "
                  f"mean={ps['mean_ms']:.1f}ms, max={ps['max_ms']:.1f}ms")
            print(f"  TSDF: {ps['n_chunks']} chunks, "
                  f"{ps['n_converged']} converged, "
                  f"{ps['memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()

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

# Get L4 GaitParams (Layer 5 registers _l4_simulation at import time)
_l4_sim = sys.modules["_l4_simulation"]
L4GaitParams = _l4_sim.GaitParams

from .game import TargetGame, configure_for_robot, GameStatistics, CONTROL_DT, TARGET_TIMEOUT_STEPS
from .utils import load_module_by_path, patch_layer_configs


@dataclass
class GameRunResult:
    """Result from a single game run, consumed by scenario critics."""
    stats: GameStatistics
    telemetry_path: Path | None = None
    slam_trail: list[tuple[float, float]] = field(default_factory=list)
    truth_trail: list[tuple[float, float]] = field(default_factory=list)
    perception_stats: dict | None = None


def _expand_v9_genome(params: dict) -> dict:
    """Expand v9 genome (8 params) to all Layer 5 constants.

    Maps unified gait/steering params to both walk and turn-in-place
    constants, matching training/ga/episode.py:inject_genome_v9().
    """
    freq = params.get("FREQ", 1.5)
    step_length = params.get("STEP_LENGTH", 0.20)
    step_height = params.get("STEP_HEIGHT", 0.06)
    duty_cycle = params.get("DUTY_CYCLE", 0.55)
    walk_speed = params.get("WALK_SPEED", 1.0)
    kp_yaw = params.get("KP_YAW", 2.0)
    wz_limit = params.get("WZ_LIMIT", 1.5)
    stance_width = params.get("STANCE_WIDTH", 0.0)

    return {
        # Walk gait
        "BASE_FREQ": freq, "MIN_FREQ": freq, "MAX_FREQ": freq,
        "FREQ_SCALE": 0.0,
        "STEP_LENGTH_SCALE": step_length,
        "MAX_STEP_LENGTH": step_length,
        "TROT_STEP_HEIGHT": step_height,
        "WALK_STEP_HEIGHT": step_height,
        # Turn gait (unified with walk)
        "TURN_IN_PLACE_FREQ": freq,
        "TURN_IN_PLACE_STEP_HEIGHT": step_height,
        "TURN_IN_PLACE_STEP_LENGTH": step_length,
        "TURN_IN_PLACE_DUTY_CYCLE": duty_cycle,
        "TURN_IN_PLACE_WZ_SCALE": 1.0,
        "TURN_IN_PLACE_STANCE_WIDTH": stance_width,
        # Steering
        "KP_YAW": kp_yaw, "WALK_SPEED": walk_speed,
        "WALK_SPEED_MIN": 0.0, "TURN_WZ_LIMIT": wz_limit,
    }


def _is_v12_genome(params: dict) -> bool:
    """Detect v12 sovereign genome by presence of GAIT_FREQ key."""
    return "GAIT_FREQ" in params


def _expand_v12_genome(params: dict) -> dict:
    """Expand v12 genome (13 params) to all Layer 5 constants.

    Maps sovereign walk + turn genes to L5's TURN_IN_PLACE_* and
    walk constants. WALK_SPEED is derived from STEP_LENGTH * GAIT_FREQ.
    """
    gait_freq = params.get("GAIT_FREQ", 1.5)
    step_length = params.get("STEP_LENGTH", 0.20)
    step_height = params.get("STEP_HEIGHT", 0.06)
    duty_cycle = params.get("DUTY_CYCLE", 0.55)
    stance_width = params.get("STANCE_WIDTH", 0.0)
    kp_yaw = params.get("KP_YAW", 2.0)
    wz_limit = params.get("WZ_LIMIT", 1.5)

    return {
        # Walk gait
        "BASE_FREQ": gait_freq, "MIN_FREQ": gait_freq, "MAX_FREQ": gait_freq,
        "FREQ_SCALE": 0.0,
        "STEP_LENGTH_SCALE": step_length,
        "MAX_STEP_LENGTH": step_length,
        "TROT_STEP_HEIGHT": step_height,
        "WALK_STEP_HEIGHT": step_height,
        # Turn gait (v12 has separate turn params)
        "TURN_IN_PLACE_FREQ": params.get("TURN_FREQ", 1.0),
        "TURN_IN_PLACE_STEP_HEIGHT": params.get("TURN_STEP_HEIGHT", 0.06),
        "TURN_IN_PLACE_STEP_LENGTH": step_length,
        "TURN_IN_PLACE_DUTY_CYCLE": params.get("TURN_DUTY_CYCLE", 0.55),
        "TURN_IN_PLACE_WZ_SCALE": 1.0,
        "TURN_IN_PLACE_STANCE_WIDTH": params.get("TURN_STANCE_WIDTH", 0.04),
        # Steering (WALK_SPEED derived from step_length * gait_freq)
        "KP_YAW": kp_yaw,
        "WALK_SPEED": step_length * gait_freq,
        "WALK_SPEED_MIN": 0.0,
        "TURN_WZ_LIMIT": wz_limit,
    }


def _is_v10_genome(params: dict) -> bool:
    """Detect v10 genome by presence of turn joint delta keys."""
    return "P1_FL_HIP" in params


def _apply_genome(genome_path: str) -> None:
    """Load a GA-evolved genome JSON and patch game + Layer 5 parameters.

    v12+ genomes: dual patch — game module constants for L4-direct walk
    control, plus L5 expansion for the startup/stand phase.
    v9/v10 genomes: expand to L5 constants and patch L5 modules.
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

    from . import game as game_mod

    if _is_v12_genome(params):
        # v12+ sovereign genome: dual patch

        # 1. Patch game module constants for L4-direct walk/turn control
        v12_game_params = [
            "GAIT_FREQ", "STEP_LENGTH", "STEP_HEIGHT", "DUTY_CYCLE", "STANCE_WIDTH",
            "KP_YAW", "WZ_LIMIT", "TURN_FREQ", "TURN_STEP_HEIGHT", "TURN_DUTY_CYCLE",
            "TURN_STANCE_WIDTH", "TURN_WZ", "THETA_THRESHOLD",
        ]
        for name in v12_game_params:
            if name in params:
                setattr(game_mod, name, params[name])
                print(f"  game.{name} = {params[name]:.4f}")

        # 2. Expand to L5 constants for startup/stand phase (send_motion_command)
        expanded = _expand_v12_genome(params)
        theta = params.get("THETA_THRESHOLD", "?")
        turn_wz = params.get("TURN_WZ", "?")
        print(f"  v12 sovereign genome: 13 genes, theta_threshold={theta}, turn_wz={turn_wz}")

    elif _is_v10_genome(params):
        # Extract walk subset (the 8 v9 params) for L5 expansion
        walk_keys = ["FREQ", "STEP_LENGTH", "STEP_HEIGHT", "DUTY_CYCLE",
                     "WALK_SPEED", "KP_YAW", "WZ_LIMIT", "STANCE_WIDTH"]
        walk_params = {k: params[k] for k in walk_keys if k in params}
        expanded = _expand_v9_genome(walk_params)
        # Merge: expanded walk + original turn/timing genes
        for k, v in params.items():
            if k not in expanded:
                expanded[k] = v
        # Print turn gene summary
        turn_count = sum(1 for k in params if k.startswith(("P1_", "P2_")))
        timing = {k: params[k] for k in ["T_PHASE1", "T_PHASE2", "T_PHASE3"] if k in params}
        print(f"  v10 turn genes: {turn_count} joint deltas, timing={timing}")

    elif "FREQ" in params:
        expanded = _expand_v9_genome(params)

    else:
        expanded = params

    # Patch Layer 5 config.defaults and downstream modules
    locomotion_params = [
        "BASE_FREQ", "FREQ_SCALE", "MAX_FREQ", "MIN_FREQ",
        "STEP_LENGTH_SCALE", "MAX_STEP_LENGTH", "TROT_STEP_HEIGHT",
        "WALK_STEP_HEIGHT",
        "TURN_IN_PLACE_FREQ", "TURN_IN_PLACE_STEP_HEIGHT",
        "TURN_IN_PLACE_STEP_LENGTH", "TURN_IN_PLACE_DUTY_CYCLE",
        "TURN_IN_PLACE_WZ_SCALE", "TURN_IN_PLACE_STANCE_WIDTH",
    ]
    defaults_mod = sys.modules.get("config.defaults")
    downstream_mods = ["velocity_mapper", "gait_selector", "locomotion",
                       "transition", "terrain_gait"]

    for name in locomotion_params:
        if name not in expanded:
            continue
        if defaults_mod and hasattr(defaults_mod, name):
            setattr(defaults_mod, name, expanded[name])
        for mod_name in downstream_mods:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, name):
                setattr(mod, name, expanded[name])

    gen = genome.get("generation", "?")
    fitness = genome.get("fitness", "?")
    print(f"Applied genome gen={gen}, fitness={fitness}")


def _find_obstacle_bodies(scene_path: Path) -> list[str]:
    """Parse scene XML to discover obstacle body names (obs_*).

    Foreman-level utility — reads the scene to learn what obstacles exist
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


def run_game(args) -> GameRunResult:
    """Run target game with given configuration. Returns structured result.

    Called by both the CLI main() and the scenario runner. The args namespace
    should have at minimum: robot, targets, headless, seed, genome,
    full_circle, domain, slam, obstacles.

    Optional attributes (used by scenario runner):
        scene_path (str): Override scene XML path (bypasses obstacles logic)
        timeout_per_target (float): Seconds per target before timeout
        min_dist (float): Min target spawn distance (default 3.0)
        max_dist (float): Max target spawn distance (default 6.0)
        angle_range: Override target spawning angle range
    """
    print(f"Starting target game: robot={args.robot}, targets={args.targets}")

    # Configure game constants for this robot (before genome, so genome overrides)
    configure_for_robot(args.robot)

    genome = getattr(args, 'genome', None)
    if genome:
        print(f"\nApplying evolved genome: {genome}")
        _apply_genome(genome)

    # Safety cap: untrained genomes may have unstable turn params for heavy robots.
    # B2 (65kg) topples at TURN_WZ > 0.7 and needs wider stance (>= 0.12).
    from . import game as game_mod
    if args.robot == "b2":
        if game_mod.TURN_WZ > 0.7:
            game_mod.TURN_WZ = 0.6
            print(f"  Safety cap: TURN_WZ capped to {game_mod.TURN_WZ} for B2")
        if game_mod.TURN_STANCE_WIDTH < 0.12:
            game_mod.TURN_STANCE_WIDTH = 0.12
            print(f"  Safety cap: TURN_STANCE_WIDTH raised to {game_mod.TURN_STANCE_WIDTH} for B2")

    # Pre-populate config caches to avoid runtime namespace collision
    patch_layer_configs(args.robot, Path(_root))

    # If domain specified, monkey-patch FirmwareLauncher to use it
    # (avoids DDS session conflict with other running firmware)
    domain = getattr(args, 'domain', None)
    if domain is not None:
        from launcher import FirmwareLauncher
        _orig_init = FirmwareLauncher.__init__
        _domain = domain
        def _patched_init(self, *a, **kw):
            kw["domain"] = _domain
            _orig_init(self, *a, **kw)
        FirmwareLauncher.__init__ = _patched_init

    # Initialize SLAM odometry if requested
    odometry = None
    slam = getattr(args, 'slam', False)
    if slam:
        # Add workspace root to path for layer_6 imports
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from layer_6.slam.odometry import Odometry
        odometry = Odometry()
        print("SLAM odometry enabled (Phase 1: IMU yaw + body velocity)")

    # Initialize telemetry log if SLAM is active
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

    # Select scene variant (scene_path override takes priority)
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

    # Discover obstacle bodies from scene XML for ground-truth proximity
    # checking (foreman referee, not robot sensors).
    obstacle_bodies = _find_obstacle_bodies(scene_path)
    track_bodies = ["target"] + obstacle_bodies

    headless = getattr(args, 'headless', False)
    sim = SimulationManager(
        args.robot,
        headless=headless,
        scene=str(scene_path),
        track_bodies=track_bodies,
        domain=domain,
    )
    sim.start()
    perception = None
    try:
        import math
        full_circle = getattr(args, 'full_circle', False)
        if full_circle:
            angle_range = (-math.pi, math.pi)
        else:
            # Exclude front cone (+/-30deg): spawn to the sides and behind
            angle_range = [(math.pi / 6, math.pi), (-math.pi, -math.pi / 6)]

        # Allow per-scenario override of angle_range
        angle_range_override = getattr(args, 'angle_range', None)
        if angle_range_override is not None:
            angle_range = angle_range_override

        # Per-scenario timeout override
        timeout_steps = TARGET_TIMEOUT_STEPS
        timeout_per_target = getattr(args, 'timeout_per_target', None)
        if timeout_per_target is not None:
            timeout_steps = int(timeout_per_target / CONTROL_DT)

        min_dist = getattr(args, 'min_dist', 3.0)
        max_dist = getattr(args, 'max_dist', 6.0)

        game = TargetGame(
            sim,
            L4GaitParams=L4GaitParams,
            make_low_cmd=_l4_sim._make_low_cmd,
            stamp_cmd=_l4_sim._stamp_cmd,
            num_targets=args.targets,
            seed=getattr(args, 'seed', None),
            angle_range=angle_range,
            odometry=odometry,
            min_dist=min_dist,
            max_dist=max_dist,
            timeout_steps=timeout_steps,
        )

        # Register obstacle bodies for ground-truth proximity checking
        if obstacle_bodies:
            game.set_obstacle_bodies(obstacle_bodies)

        # Optional custom spawn function (e.g. scattered scenario back-and-forth)
        spawn_fn = getattr(args, 'spawn_fn', None)
        if spawn_fn is not None:
            game._spawn_fn = spawn_fn

        # Wire up telemetry log
        if telemetry_log is not None:
            game.set_telemetry(telemetry_log)

        # Wire up DDS publishers/subscribers if SLAM is active
        obstacles = getattr(args, 'obstacles', False)
        if odometry is not None:
            try:
                from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
                # PoseEstimate_ and PointCloud_ are in layers_1_2/messages.py,
                # not layer_3/messages.py (which is cached in sys.modules from L5).
                _l12_msgs = load_module_by_path("_l12_messages", str(_root / "layers_1_2" / "messages.py"))
                PoseEstimate_ = _l12_msgs.PoseEstimate_
                PointCloud_ = _l12_msgs.PointCloud_
                from dds import dds_init, stamp_cmd
                dds_domain = domain if domain is not None else 1
                dds_init(domain_id=dds_domain, interface="lo")

                # Pose publisher
                pose_pub = ChannelPublisher("rt/pose_estimate", PoseEstimate_)
                pose_pub.Init()
                # PoseEstimate_ is a Layer 6 internal message — no CRC needed.
                # stamp_cmd only handles Layer 3 motor commands.
                _noop_stamp = lambda msg: None
                game.set_pose_publisher(pose_pub, PoseEstimate_, _noop_stamp)
                print(f"DDS pose publisher on rt/pose_estimate (domain={dds_domain})")

                # Point cloud subscriber + perception pipeline (if obstacles)
                if obstacles:
                    from layer_6.config.defaults import load_config
                    from layer_6.planner.dwa import CurvatureDWAPlanner
                    from .perception import PerceptionPipeline
                    pcfg = load_config(args.robot)

                    # Apply perception overrides (from tuning script)
                    for key, val in getattr(args, 'perception_overrides', {}).items():
                        setattr(pcfg, key, val)
                    # Apply DWA overrides to pcfg fields (tuning names → pcfg names)
                    for key, val in getattr(args, 'dwa_overrides', {}).items():
                        setattr(pcfg, key, val)

                    perception = PerceptionPipeline(odometry, pcfg)
                    cloud_sub = ChannelSubscriber("rt/pointcloud", PointCloud_)
                    cloud_sub.Init(perception.on_point_cloud, 5)
                    game._perception = perception
                    print(f"Persistent TSDF active: ±{pcfg.tsdf_xy_extent}m, "
                          f"voxel={pcfg.tsdf_voxel_size}m, Bayesian log-odds")

                    # Wire up curvature-based DWA planner
                    game.set_dwa_planner(CurvatureDWAPlanner(pcfg))
                    print(f"Curvature DWA active: {pcfg.dwa_n_curvatures} arcs, "
                          f"max_arc={pcfg.dwa_max_arc_length}m, "
                          f"v_max={pcfg.v_max}, w_max={pcfg.w_max}")

                    # Wait for first point cloud (DDS discovery + LiDAR scan)
                    import time as _time
                    _pc_deadline = _time.monotonic() + 5.0
                    while perception.costmap_query is None and _time.monotonic() < _pc_deadline:
                        _time.sleep(0.1)
                    if perception.costmap_query is not None:
                        print(f"First costmap received ({perception.stats['builds']} builds)")
                    else:
                        print("Warning: no point cloud received after 5s (DWA will use fallback)")

            except Exception as e:
                print(f"Warning: DDS setup failed: {e} (SLAM still works)")

        stats = game.run()

        return GameRunResult(
            stats=stats,
            telemetry_path=telem_path,
            slam_trail=list(game.slam_trail),
            truth_trail=list(game.truth_trail),
            perception_stats=perception.stats if perception is not None else None,
        )
    finally:
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
    args = parser.parse_args()

    result = run_game(args)

    print(f"\nFinal: {result.stats.targets_reached}/{result.stats.targets_spawned} "
          f"reached ({result.stats.success_rate:.0%})")

    # Print perception stats
    if result.perception_stats is not None:
        ps = result.perception_stats
        if ps.get("builds", 0) > 0:
            print(f"\nPerception: {ps['builds']} costmaps built, "
                  f"mean={ps['mean_ms']:.1f}ms, max={ps['max_ms']:.1f}ms")


if __name__ == "__main__":
    main()

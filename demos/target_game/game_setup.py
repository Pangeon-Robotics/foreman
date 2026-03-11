"""Game setup helpers for DDS, perception, viewers, and scoring.

Extracted from __main__.py to keep it under 400 lines.
These functions are called only by run_game() / _run_game_inner().
No prints — callers decide what to display.
"""
from __future__ import annotations

from pathlib import Path

# Late-bound: _root is injected at first use via __main__._root
_root = None


def _get_root():
    global _root
    if _root is None:
        _root = Path(__file__).resolve().parents[3]
    return _root


def setup_perception(args, game, sim, odometry, obstacle_bodies, scene_path):
    """Wire up DDS, perception pipeline, DWA planner, and debug viewers."""
    from .utils import load_module_by_path

    perception = None
    obstacles = getattr(args, 'obstacles', False)
    domain = getattr(args, 'domain', None)
    root = _get_root()

    if odometry is None:
        return None

    try:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        _l12_msgs = load_module_by_path(
            "_l12_messages", str(root / "layers_1_2" / "messages.py"))
        PoseEstimate_ = _l12_msgs.PoseEstimate_
        PointCloud_ = _l12_msgs.PointCloud_
        from dds import dds_init, stamp_cmd
        dds_domain = domain if domain is not None else 1
        dds_init(domain_id=dds_domain, interface="lo")

        pose_pub = ChannelPublisher("rt/pose_estimate", PoseEstimate_)
        pose_pub.Init()
        _noop_stamp = lambda msg: None
        game.set_pose_publisher(pose_pub, PoseEstimate_, _noop_stamp)

        if obstacles:
            perception = _setup_obstacle_perception(
                args, game, obstacle_bodies, scene_path, odometry,
                ChannelSubscriber, PointCloud_, dds_domain)

        setup_viewers(args, game, obstacle_bodies, scene_path)

    except Exception as e:
        import sys
        print(f"[game_setup] Perception init failed: {e}", file=sys.stderr)

    return perception


def _perception_config_dict(args):
    """Build serializable perception config dict for subprocess."""
    from layer_6.config.defaults import load_config
    pcfg = load_config(args.robot)
    for key, val in getattr(args, 'perception_overrides', {}).items():
        setattr(pcfg, key, val)
    for key, val in getattr(args, 'dwa_overrides', {}).items():
        setattr(pcfg, key, val)

    pcfg.scan_min_interval = 0.25
    pcfg.costmap_z_hi = 0.80
    pcfg.tsdf_voxel_size = 0.01
    pcfg.tsdf_output_resolution = 0.05
    pcfg.tsdf_log_odds_hit = 3.0
    pcfg.tsdf_log_odds_max = 5.0
    pcfg.tsdf_log_odds_free = 0.25
    pcfg.tsdf_truncation = 0.5
    pcfg.tsdf_xy_extent = 10.0
    pcfg.tsdf_depth_extension = 5
    pcfg.tsdf_decay_rate = 0.0
    pcfg.tsdf_unknown_cell_cost = 0.5
    pcfg.min_world_z = 0.05

    # Serialize to dict (multiprocessing spawn can't pickle arbitrary objects)
    return {k: v for k, v in vars(pcfg).items()
            if isinstance(v, (int, float, str, bool, tuple, list))}


def _setup_obstacle_perception(args, game, obstacle_bodies, scene_path,
                               odometry, ChannelSubscriber, PointCloud_,
                               dds_domain):
    """Spawn perception subprocess for obstacle avoidance.

    Perception runs in a separate process (no GIL contention).
    Communicates via shared memory (pose) and temp files (costmap).
    """
    from .perception_worker import PerceptionSubprocess

    config_dict = _perception_config_dict(args)

    # God-view TSDF for F1 scoring (separate from perception subprocess)
    if getattr(args, 'god', False) and obstacle_bodies:
        from .god_tsdf import GodViewTSDF
        god_tsdf = GodViewTSDF(str(scene_path), robot=args.robot)
        game._god_view_tsdf = god_tsdf

    # Spawn perception subprocess
    perc_sub = PerceptionSubprocess(
        scene_xml=str(scene_path),
        robot=args.robot,
        config_dict=config_dict,
        scan_hz=4.0,
    )
    perc_sub.start()
    game._perception_subprocess = perc_sub

    from .path_critic import PathCritic
    critic = PathCritic(robot=args.robot, robot_radius=0.35)
    game.set_path_critic(critic)

    # Return None — no in-process PerceptionPipeline anymore
    return None


def setup_viewers(args, game, obstacle_bodies, scene_path):
    """Set up debug viewer, god-view costmap, and god-view TSDF."""
    from .scene_parser import _find_obstacle_geoms

    viewer = getattr(args, 'viewer', False)
    if viewer:
        from .debug_server import DebugServer
        debug_server = DebugServer(port=9877)
        debug_server.start()
        game._debug_server = debug_server
        game._obstacle_geoms = _find_obstacle_geoms(scene_path)
        if obstacle_bodies:
            from .test_costmap_compare import build_god_view_binary
            from layer_6.config.defaults import load_config as _load_pcfg
            _pcfg = _load_pcfg(args.robot)
            gv_grid, gv_meta = build_god_view_binary(
                str(scene_path), z_lo=_pcfg.costmap_z_lo, z_hi=_pcfg.costmap_z_hi,
                output_resolution=_pcfg.tsdf_output_resolution,
                xy_extent=_pcfg.tsdf_xy_extent)
            game.set_god_view_costmap(gv_grid, gv_meta)

    if getattr(args, 'god', False) and obstacle_bodies and not getattr(args, 'headless', False):
        from .test_costmap_compare import build_god_view_costmap
        from layer_6.config.defaults import load_config as _load_pcfg
        import struct as _struct
        import numpy as _np

        _pcfg = _load_pcfg(args.robot)
        gv_grid, gv_meta = build_god_view_costmap(
            str(scene_path), z_lo=_pcfg.costmap_z_lo, z_hi=_pcfg.costmap_z_hi,
            output_resolution=_pcfg.tsdf_output_resolution,
            xy_extent=_pcfg.tsdf_xy_extent, truncation=_pcfg.tsdf_truncation)
        _god_file = "/tmp/god_view_costmap.bin"
        gv_grid_t = gv_grid.T
        rows, cols = gv_grid_t.shape
        with open(_god_file, 'wb') as _f:
            _f.write(_struct.pack('<HHfff', rows, cols,
                                  gv_meta['origin_x'], gv_meta['origin_y'],
                                  gv_meta['voxel_size']))
            _f.write(gv_grid_t.tobytes())
        game.set_god_view_costmap(gv_grid, gv_meta)
        game._god_view_path_file = "/tmp/god_view_path.bin"

    if getattr(args, 'god', False) and obstacle_bodies:
        if game._god_view_tsdf is None:
            from .god_tsdf import GodViewTSDF
            god_tsdf = GodViewTSDF(str(scene_path), robot=args.robot)
            game._god_view_tsdf = god_tsdf


def compute_occupancy(perception, args, scene_path):
    """Compute occupancy accuracy if perception is active."""
    if perception is None or not getattr(args, 'obstacles', False):
        return None
    try:
        from .test_occupancy import compute_3ds_v2, compute_3ds_god
        occ = compute_3ds_v2(perception._tsdf, str(scene_path))
        occ['god'] = compute_3ds_god(perception._tsdf, str(scene_path))
        return occ
    except Exception:
        return None


def compute_scores(game, perception, args):
    """Compute perception F1 scores (god TSDF vs robot TSDF). Returns dict or None."""
    god_tsdf_obj = getattr(game, '_god_view_tsdf', None)
    robot_tsdf_obj = perception._tsdf if perception is not None else None
    if god_tsdf_obj is None or robot_tsdf_obj is None:
        return None
    try:
        from .scores import compute_surface_f1, compute_cost_f1, compute_router_f1
        from layer_6.config.defaults import load_config as _load_score_cfg
        _scfg = _load_score_cfg(args.robot)
        _z_lo = getattr(_scfg, 'costmap_z_lo', 0.05)
        _z_hi = 0.80
        _res = getattr(_scfg, 'tsdf_output_resolution', 0.05)

        scores = {
            "surface": compute_surface_f1(god_tsdf_obj._tsdf, robot_tsdf_obj, _res),
            "cost": compute_cost_f1(god_tsdf_obj._tsdf, robot_tsdf_obj, _z_lo, _z_hi, _res),
        }
        trail = list(game.truth_trail)
        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]
            if (dx * dx + dy * dy) > 1.0:
                scores["router"] = compute_router_f1(
                    god_tsdf_obj._tsdf, robot_tsdf_obj,
                    trail[0], trail[-1], _z_lo, _z_hi, _res)
        if "router" not in scores:
            scores["router"] = {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                                "god_path_m": 0.0, "robot_path_m": 0.0}
        return scores
    except Exception:
        return None

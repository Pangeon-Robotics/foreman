"""Game setup helpers for DDS, perception, viewers, and scoring.

Extracted from __main__.py to keep it under 400 lines.
These functions are called only by run_game() / _run_game_inner().
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
        print(f"DDS pose publisher on rt/pose_estimate (domain={dds_domain})")

        if obstacles:
            perception = _setup_obstacle_perception(
                args, game, obstacle_bodies, scene_path, odometry,
                ChannelSubscriber, PointCloud_, dds_domain)

        setup_viewers(args, game, obstacle_bodies, scene_path)

    except Exception as e:
        print(f"Warning: DDS setup failed: {e} (SLAM still works)")

    return perception


def _setup_obstacle_perception(args, game, obstacle_bodies, scene_path,
                               odometry, ChannelSubscriber, PointCloud_,
                               dds_domain):
    """Set up TSDF and direct scanner for obstacle avoidance."""
    from layer_6.config.defaults import load_config
    from .perception import PerceptionPipeline

    pcfg = load_config(args.robot)
    for key, val in getattr(args, 'perception_overrides', {}).items():
        setattr(pcfg, key, val)
    for key, val in getattr(args, 'dwa_overrides', {}).items():
        setattr(pcfg, key, val)

    if not hasattr(pcfg, 'scan_min_interval'):
        pcfg.scan_min_interval = 0.25
    pcfg.costmap_z_hi = 0.80
    pcfg.tsdf_voxel_size = 0.01  # 1cm internal voxels for high 3DS
    pcfg.tsdf_output_resolution = 0.05  # 5cm costmap output
    pcfg.tsdf_log_odds_hit = 3.0
    pcfg.tsdf_log_odds_max = 5.0
    pcfg.tsdf_log_odds_free = 0.25
    pcfg.tsdf_truncation = 0.5
    pcfg.tsdf_xy_extent = 10.0
    pcfg.tsdf_depth_extension = 5
    pcfg.tsdf_decay_rate = 0.0
    pcfg.tsdf_unknown_cell_cost = 0.5

    perception = PerceptionPipeline(odometry, pcfg)
    perception._MIN_WORLD_Z = 0.05

    _god_mj_model = None
    _god_mj_data = None
    if getattr(args, 'god', False) and obstacle_bodies:
        from .god_tsdf import GodViewTSDF
        god_tsdf = GodViewTSDF(str(scene_path), robot=args.robot)
        game._god_view_tsdf = god_tsdf
        _god_mj_model = god_tsdf._model
        _god_mj_data = god_tsdf._data
        print(f"God-view TSDF: {god_tsdf._n_rays} rays, "
              f"exclude {len(god_tsdf._robot_geom_ids)} robot geoms, "
              f"voxel={god_tsdf._tsdf.voxel_size}m")

    # Reduce ray count when SLAM is active to limit GIL blocking.
    # Full 64K rays take 200-450ms; 8K rays take ~25ms.
    slam = getattr(args, 'slam', False)
    if slam:
        perception._use_reduced_rays = True
    perception.init_direct_scanner(
        str(scene_path), robot=args.robot,
        mj_model=_god_mj_model, mj_data=_god_mj_data)
    print(f"Direct scanner: {perception._direct_n_rays} rays, "
          f"exclude {len(perception._direct_robot_geoms)} robot + "
          f"{len(perception._direct_target_geoms)} target geoms")

    # Only subscribe to DDS point clouds if DirectScanner isn't active.
    # The DDS subscriber's background thread acquires the GIL on every
    # callback (even if the callback returns immediately), starving the
    # control loop and destabilizing the gait.
    if not perception._direct_ready:
        cloud_sub = ChannelSubscriber("rt/pointcloud", PointCloud_)
        cloud_sub.Init(perception.on_point_cloud, 5)
    game._perception = perception
    print(f"TSDF active: +/-{pcfg.tsdf_xy_extent}m, "
          f"voxel={pcfg.tsdf_voxel_size}m, "
          f"output={pcfg.tsdf_output_resolution}m, "
          f"lo_hit={pcfg.tsdf_log_odds_hit}, "
          f"decay={pcfg.tsdf_decay_rate}")

    if not perception._direct_ready:
        import time as _time
        _pc_deadline = _time.monotonic() + 5.0
        while perception.costmap_query is None and _time.monotonic() < _pc_deadline:
            _time.sleep(0.1)
        if perception.costmap_query is not None:
            print(f"First costmap received ({perception.stats['builds']} builds)")
        else:
            print("Warning: no point cloud received after 5s (DWA will use fallback)")

    from .path_critic import PathCritic
    critic = PathCritic(robot=args.robot, robot_radius=0.35)
    game.set_path_critic(critic)

    return perception


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
            import numpy as _np
            print(f"God-view costmap: {gv_grid.shape}, "
                  f"{int(_np.sum(gv_grid == 254))} obstacle cells")

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
        n_lethal = int(_np.sum(gv_grid >= 253))
        n_grad = int(_np.sum((gv_grid > 0) & (gv_grid < 253)))
        print(f"God-view costmap: {gv_grid.shape}, "
              f"{n_lethal} lethal + {n_grad} gradient cells -> {_god_file}")
        game.set_god_view_costmap(gv_grid, gv_meta)
        game._god_view_path_file = "/tmp/god_view_path.bin"

    if getattr(args, 'god', False) and obstacle_bodies:
        if game._god_view_tsdf is None:
            from .god_tsdf import GodViewTSDF
            god_tsdf = GodViewTSDF(str(scene_path), robot=args.robot)
            game._god_view_tsdf = god_tsdf
            print(f"God-view TSDF: {god_tsdf._n_rays} rays, "
                  f"exclude {len(god_tsdf._robot_geom_ids)} robot geoms, "
                  f"voxel={god_tsdf._tsdf.voxel_size}m")


def compute_occupancy(perception, args, scene_path):
    """Compute occupancy accuracy if perception is active."""
    if perception is None or not getattr(args, 'obstacles', False):
        return None
    try:
        from .test_occupancy import compute_3ds_v2, compute_3ds_god
        occ = compute_3ds_v2(perception._tsdf, str(scene_path))
        occ['god'] = compute_3ds_god(perception._tsdf, str(scene_path))
        return occ
    except Exception as e:
        print(f"Warning: occupancy accuracy failed: {e}")
        return None


def compute_scores(game, perception, args):
    """Compute perception F1 scores (god TSDF vs robot TSDF)."""
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
        sf = scores['surface']
        print(f"\nScores: surface={sf['f1']:.1f} "
              f"(P={sf['precision']:.1f} R={sf['recall']:.1f} "
              f"n_god={sf['n_god']} n_robot={sf['n_robot']}) "
              f"cost={scores['cost']['f1']:.1f} "
              f"router={scores['router']['f1']:.1f}")

        _print_voxel_diagnostics(god_tsdf_obj, robot_tsdf_obj, game)
        return scores
    except Exception as e:
        print(f"Warning: score computation failed: {e}")
        return None


def _print_voxel_diagnostics(god_tsdf_obj, robot_tsdf_obj, game):
    """Print diagnostic info about surface voxels."""
    import numpy as _np
    god_vox = god_tsdf_obj._tsdf.get_surface_voxels(include_history=True)
    robot_vox = robot_tsdf_obj.get_surface_voxels(include_history=True)
    if len(god_vox) == 0 or len(robot_vox) == 0:
        return
    print(f"\n=== SURFACE VOXEL DIAGNOSTICS ===")
    print(f"God voxels: {len(god_vox)}, Robot voxels: {len(robot_vox)}")
    print(f"God Z: min={god_vox[:,2].min():.3f} max={god_vox[:,2].max():.3f} "
          f"mean={god_vox[:,2].mean():.3f}")
    print(f"Robot Z: min={robot_vox[:,2].min():.3f} max={robot_vox[:,2].max():.3f} "
          f"mean={robot_vox[:,2].mean():.3f}")
    for label, vox in [("God", god_vox), ("Robot", robot_vox)]:
        zbins = _np.arange(-0.5, 1.6, 0.1)
        hist, _ = _np.histogram(vox[:, 2], bins=zbins)
        nz = [(f"{zbins[i]:.1f}:{h}") for i, h in enumerate(hist) if h > 0]
        print(f"{label} Z hist: {' '.join(nz)}")
    print(f"God XY: x=[{god_vox[:,0].min():.1f},{god_vox[:,0].max():.1f}] "
          f"y=[{god_vox[:,1].min():.1f},{god_vox[:,1].max():.1f}]")
    print(f"Robot XY: x=[{robot_vox[:,0].min():.1f},{robot_vox[:,0].max():.1f}] "
          f"y=[{robot_vox[:,1].min():.1f},{robot_vox[:,1].max():.1f}]")
    print(f"God history: {len(god_tsdf_obj._tsdf._surface_history)}")
    print(f"Robot history: {len(robot_tsdf_obj._surface_history)}")
    print(f"God chunks: {len(god_tsdf_obj._tsdf._chunks)}")
    print(f"Robot chunks: {len(robot_tsdf_obj._chunks)}")
    if hasattr(game._perception, '_direct_scan_count'):
        print(f"Robot direct scans: {game._perception._direct_scan_count}")
    print(f"Robot DDS scans: {game._perception.stats.get('builds', 0)}")

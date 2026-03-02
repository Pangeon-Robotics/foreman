"""God-view vs robot-view costmap comparison.

Builds a "god-view" costmap from scene XML obstacle geometry (perfect
knowledge), runs the robot to build a TSDF-derived costmap, then compares
the two in the observed region. Produces a 2x2 diagnostic PNG + metrics JSON.

Usage:
    python -m foreman.demos.target_game.test_costmap_compare scattered
    python -m foreman.demos.target_game.test_costmap_compare scattered --god-only
    python -m foreman.demos.target_game.test_costmap_compare scattered --targets 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _rasterize_obstacles(
    scene_xml_path: str,
    z_lo: float, z_hi: float,
    output_resolution: float, xy_extent: float,
) -> tuple[np.ndarray, dict]:
    """Rasterize obstacle XY footprints into a bool grid.

    Returns (occupied_2d, meta) where occupied_2d is (gs, gs) bool.
    Shared by build_god_view_costmap (EDT gradient) and
    build_god_view_binary (binary for Godot).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from foreman.demos.target_game.__main__ import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    origin = -xy_extent
    gs = int(round(2 * xy_extent / output_resolution))
    res = output_resolution
    occupied = np.zeros((gs, gs), dtype=bool)

    for obs in obstacles:
        px, py, pz = obs['pos']
        otype, size = obs['type'], obs['size']

        if otype == 'box':
            hx, hy, hz = size
            if pz + hz < z_lo or pz - hz > z_hi:
                continue
            ix_lo = max(0, int(math.floor((px - hx - origin) / res)))
            ix_hi = min(gs, int(math.ceil((px + hx - origin) / res)))
            iy_lo = max(0, int(math.floor((py - hy - origin) / res)))
            iy_hi = min(gs, int(math.ceil((py + hy - origin) / res)))
            occupied[ix_lo:ix_hi, iy_lo:iy_hi] = True

        elif otype == 'cylinder':
            radius, half_h = size[0], size[1]
            if pz + half_h < z_lo or pz - half_h > z_hi:
                continue
            ix_lo = max(0, int(math.floor((px - radius - origin) / res)))
            ix_hi = min(gs, int(math.ceil((px + radius - origin) / res)))
            iy_lo = max(0, int(math.floor((py - radius - origin) / res)))
            iy_hi = min(gs, int(math.ceil((py + radius - origin) / res)))
            for ix in range(ix_lo, ix_hi):
                wx = origin + (ix + 0.5) * res
                for iy in range(iy_lo, iy_hi):
                    wy = origin + (iy + 0.5) * res
                    if (wx - px)**2 + (wy - py)**2 <= radius**2:
                        occupied[ix, iy] = True

    meta = {'origin_x': origin, 'origin_y': origin, 'voxel_size': res,
            'nx': gs, 'ny': gs}
    return occupied, meta


def build_god_view_binary(
    scene_xml_path: str,
    z_lo: float = 0.05, z_hi: float = 0.55,
    output_resolution: float = 0.05,
    xy_extent: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """Build binary god-view grid: 254 for occupied cells, 0 for free.

    For Godot display — red appears ONLY at obstacle footprints.
    """
    occupied, meta = _rasterize_obstacles(
        scene_xml_path, z_lo, z_hi, output_resolution, xy_extent)
    grid = np.zeros(occupied.shape, dtype=np.uint8)
    grid[occupied] = 254
    return grid, meta


def build_god_view_costmap(
    scene_xml_path: str,
    z_lo: float = 0.05, z_hi: float = 0.55,
    output_resolution: float = 0.05,
    xy_extent: float = 10.0, truncation: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Build god-view costmap with EDT gradient (for comparison metrics)."""
    from scipy.ndimage import distance_transform_edt

    occupied, meta = _rasterize_obstacles(
        scene_xml_path, z_lo, z_hi, output_resolution, xy_extent)
    dist_m = distance_transform_edt(~occupied) * output_resolution
    cost_u8 = ((1.0 - np.clip(dist_m / truncation, 0, 1)) * 254).astype(np.uint8)
    meta['truncation'] = truncation
    return cost_u8, meta


def _astar_path(
    cost_grid: np.ndarray, start_xy: tuple[float, float],
    goal_xy: tuple[float, float], voxel_size: float,
    origin_x: float, origin_y: float,
    robot_radius: float = 0.35, truncation: float = 0.5,
) -> list[tuple[int, int]] | None:
    """Standalone A* on uint8 cost grid. Returns list of (ix,iy) or None."""
    import heapq

    vs, ox, oy = voxel_size, origin_x, origin_y
    nx, ny = cost_grid.shape
    SQRT2 = math.sqrt(2.0)
    _COST_WEIGHT = 1.5

    sx = max(0, min(nx - 1, int((start_xy[0] - ox) / vs)))
    sy = max(0, min(ny - 1, int((start_xy[1] - oy) / vs)))
    gx = max(0, min(nx - 1, int((goal_xy[0] - ox) / vs)))
    gy = max(0, min(ny - 1, int((goal_xy[1] - oy) / vs)))

    radius_ratio = min(robot_radius / truncation, 0.95)
    cost_threshold = int((1.0 - radius_ratio) * 254)
    passable = (cost_grid < cost_threshold) | (cost_grid == 255)
    passable[sx, sy] = True
    passable[gx, gy] = True

    # Unknown cells (255) get moderate traversal cost — they're uncertain,
    # not obstacles. Using 0.5 (half-cost) encourages A* to prefer known-free
    # space but allows traversal through unknown regions when needed.
    _UNKNOWN_COST = 0.5
    cost_norm = cost_grid.astype(np.float64) / 254.0
    cost_norm[cost_grid == 255] = _UNKNOWN_COST
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    ]

    def h(x, y):
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

    open_set = [(h(sx, sy), sx, sy)]
    g_score = np.full((nx, ny), np.inf, dtype=np.float64)
    g_score[sx, sy] = 0.0
    visited = np.zeros((nx, ny), dtype=np.bool_)
    parent = {}

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if cx == gx and cy == gy:
            path = []
            px, py = gx, gy
            while (px, py) != (sx, sy):
                path.append((px, py))
                px, py = parent[(px, py)]
            path.append((sx, sy))
            path.reverse()
            return path
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True
        for dx, dy, step_dist in neighbors:
            nx2, ny2 = cx + dx, cy + dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny and not visited[nx2, ny2]:
                if not passable[nx2, ny2]:
                    continue
                if dx != 0 and dy != 0:
                    if not passable[cx + dx, cy] or not passable[cx, cy + dy]:
                        continue
                new_g = g_score[cx, cy] + step_dist + _COST_WEIGHT * cost_norm[nx2, ny2]
                if new_g < g_score[nx2, ny2]:
                    g_score[nx2, ny2] = new_g
                    parent[(nx2, ny2)] = (cx, cy)
                    heapq.heappush(open_set, (new_g + h(nx2, ny2), nx2, ny2))
    return None


def compute_route_score(
    god_grid: np.ndarray, god_meta: dict,
    robot_grid: np.ndarray, robot_meta: dict,
    robot_xy: tuple[float, float], target_xy: tuple[float, float],
    robot_radius: float = 0.35,
) -> dict:
    """Compare A* paths on god-view vs robot-view costmaps.

    Returns dict with rs_length_ratio, rs_hausdorff, rs_overlap, rs_score.
    """
    trunc_g = god_meta.get('truncation', 0.5)
    trunc_r = robot_meta.get('truncation', 0.5)
    vs = god_meta['voxel_size']

    god_path = _astar_path(
        god_grid, robot_xy, target_xy, vs,
        god_meta['origin_x'], god_meta['origin_y'], robot_radius, trunc_g)
    robot_path = _astar_path(
        robot_grid, robot_xy, target_xy, robot_meta['voxel_size'],
        robot_meta['origin_x'], robot_meta['origin_y'], robot_radius, trunc_r)

    def path_length_m(path, voxel_sz):
        d = 0.0
        for a, b in zip(path, path[1:]):
            d += math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) * voxel_sz
        return d

    fail = {'rs_length_ratio': 0.0, 'rs_hausdorff': float('inf'),
            'rs_overlap': 0.0, 'rs_score': 0.0,
            'god_path_len': 0, 'robot_path_len': 0,
            'god_path_m': 0.0, 'robot_path_m': 0.0}
    if god_path is None and robot_path is None:
        # Both views can't find a path — terrain genuinely impassable
        # at this robot radius. Not the robot's fault, return neutral.
        return {**fail, 'rs_score': 50.0}
    if god_path is None:
        # God-view can't find a path but robot can — unusual, penalty score
        return {**fail, 'rs_score': 15.0, 'robot_path_len': len(robot_path)}
    if robot_path is None:
        # Robot-view can't find a path but god-view can — degraded perception
        return {**fail, 'rs_score': 20.0, 'god_path_len': len(god_path),
                'god_path_m': path_length_m(god_path, vs)}

    god_len = path_length_m(god_path, vs)
    robot_len = path_length_m(robot_path, robot_meta['voxel_size'])

    # Length ratio: god/robot (1.0 = same, <1.0 = robot detours)
    rs_length_ratio = god_len / max(robot_len, 0.01) if god_len > 0 else 1.0
    rs_length_ratio = min(rs_length_ratio, 1.0)

    # Convert paths to world-frame points for spatial comparison
    def to_world(path, meta):
        o_x, o_y, v = meta['origin_x'], meta['origin_y'], meta['voxel_size']
        return [(o_x + (ix + 0.5) * v, o_y + (iy + 0.5) * v) for ix, iy in path]

    gw = to_world(god_path, god_meta)
    rw = to_world(robot_path, robot_meta)

    # Hausdorff distance (max bidirectional deviation)
    def directed_hausdorff(a, b):
        worst = 0.0
        for ax, ay in a:
            best = float('inf')
            for bx, by in b:
                d = math.sqrt((ax - bx)**2 + (ay - by)**2)
                if d < best:
                    best = d
            if best > worst:
                worst = best
        return worst

    h_ab = directed_hausdorff(gw, rw)
    h_ba = directed_hausdorff(rw, gw)
    rs_hausdorff = max(h_ab, h_ba)

    # Overlap: fraction of robot path cells within 2 cells of god path
    tol = 2 * vs  # 2 cells in meters
    match = 0
    for rx, ry in rw:
        for gx, gy in gw:
            if math.sqrt((rx - gx)**2 + (ry - gy)**2) <= tol:
                match += 1
                break
    rs_overlap = match / len(rw)

    # Composite score: 0-100
    len_score = rs_length_ratio  # 0..1
    haus_score = max(0.0, 1.0 - rs_hausdorff / 2.0)  # 2m+ deviation → 0
    rs_score = (0.4 * len_score + 0.3 * rs_overlap + 0.3 * haus_score) * 100.0

    return {'rs_length_ratio': round(rs_length_ratio, 3),
            'rs_hausdorff': round(rs_hausdorff, 3),
            'rs_overlap': round(rs_overlap, 3),
            'rs_score': round(rs_score, 1),
            'god_path_len': len(god_path), 'robot_path_len': len(robot_path),
            'god_path_m': round(god_len, 2), 'robot_path_m': round(robot_len, 2)}


def compare_costmaps(
    god: np.ndarray, robot: np.ndarray,
    observed_mask: np.ndarray | None,
) -> dict:
    """Compare god-view and robot-view costmaps in observed region."""
    valid = (observed_mask & (robot != 255)) if observed_mask is not None else (robot != 255)
    n_valid = int(np.sum(valid))
    total = god.shape[0] * god.shape[1]
    coverage = n_valid / total * 100.0 if total > 0 else 0.0

    zeros = {'mae': float('inf'), 'binary_precision': 0.0, 'binary_recall': 0.0,
             'binary_f1': 0.0, 'fn_dangerous': 0, 'fp_phantom': 0,
             'lethal_recall': 0.0, 'gradient_mae': float('inf'),
             'gradient_corr': 0.0, 'coverage_pct': coverage, 'n_observed': n_valid}
    if n_valid == 0:
        return zeros

    g = god[valid].astype(np.float32)
    r = robot[valid].astype(np.float32)
    mae = float(np.mean(np.abs(g - r)))

    # Binary at threshold 127
    thresh = 127
    g_occ, r_occ = g >= thresh, r >= thresh
    tp = int(np.sum(g_occ & r_occ))
    fp = int(np.sum(~g_occ & r_occ))
    fn = int(np.sum(g_occ & ~r_occ))
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    fn_dangerous = int(np.sum((g > 200) & (r < 50)))
    fp_phantom = int(np.sum((g < 10) & (r > 200)))

    lethal_gt = g >= 253
    n_lethal = int(np.sum(lethal_gt))
    lethal_recall = float(np.sum(lethal_gt & (r >= thresh)) / n_lethal) if n_lethal > 0 else 1.0

    # Gradient zone: both in 1-253
    grad = (g >= 1) & (g <= 253) & (r >= 1) & (r <= 253)
    n_grad = int(np.sum(grad))
    if n_grad > 10:
        gg, rg = g[grad], r[grad]
        gradient_mae = float(np.mean(np.abs(gg - rg)))
        gd, rd = gg - np.mean(gg), rg - np.mean(rg)
        den = np.sqrt(np.sum(gd**2) * np.sum(rd**2))
        gradient_corr = float(np.sum(gd * rd) / den) if den > 1e-9 else 0.0
    else:
        gradient_mae, gradient_corr = 0.0, 0.0

    return {'mae': mae, 'binary_precision': P, 'binary_recall': R,
            'binary_f1': f1, 'fn_dangerous': fn_dangerous, 'fp_phantom': fp_phantom,
            'lethal_recall': lethal_recall, 'gradient_mae': gradient_mae,
            'gradient_corr': gradient_corr, 'coverage_pct': coverage, 'n_observed': n_valid}


def save_comparison_png(
    god: np.ndarray, robot: np.ndarray,
    observed: np.ndarray | None, metrics: dict, path: str,
) -> None:
    """Save 2x2 comparison: god-view, robot-view, difference, classification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    valid = (observed & (robot != 255)) if observed is not None else (robot != 255)

    # God-view
    ax = axes[0, 0]
    ax.set_title('God-View Costmap')
    im = ax.imshow(god.T, cmap='Reds', vmin=0, vmax=254, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # Robot-view (255 → gray)
    ax = axes[0, 1]
    ax.set_title('Robot-View Costmap')
    rd = robot.T.astype(np.float32)
    rd[robot.T == 255] = np.nan
    cmap_r = plt.cm.Reds.copy()
    cmap_r.set_bad(color='lightgray')
    im = ax.imshow(rd, cmap=cmap_r, vmin=0, vmax=254, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # Signed difference
    ax = axes[1, 0]
    ax.set_title('Signed Difference (robot - god)')
    diff = np.zeros_like(god, dtype=np.float32)
    diff[valid] = robot[valid].astype(np.float32) - god[valid].astype(np.float32)
    dd = diff.T.copy()
    dd[~valid.T] = np.nan
    cmap_d = plt.cm.RdYlGn_r.copy()
    cmap_d.set_bad(color='lightgray')
    vm = max(abs(np.nanmin(dd)), abs(np.nanmax(dd)), 1)
    im = ax.imshow(dd, cmap=cmap_d, vmin=-vm, vmax=vm, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # Binary classification
    ax = axes[1, 1]
    ax.set_title('Binary Classification (thresh=127)')
    cls = np.full(god.shape, 4, dtype=np.uint8)  # 4=unobserved
    go, ro = god >= 127, robot >= 127
    cls[valid & go & ro] = 1    # TP
    cls[valid & ~go & ro] = 2   # FP
    cls[valid & go & ~ro] = 3   # FN
    cls[valid & ~go & ~ro] = 0  # TN
    ax.imshow(cls.T, cmap=ListedColormap(
        ['#1a1a2e', '#2ecc71', '#f1c40f', '#e74c3c', '#cccccc']),
        vmin=0, vmax=4, origin='lower')
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in
              [('#2ecc71','TP'),('#f1c40f','FP'),('#e74c3c','FN'),
               ('#1a1a2e','TN'),('#cccccc','Unobs')]],
              loc='upper right', fontsize=8)

    m = metrics
    text = (f"MAE: {m['mae']:.1f}  |  F1: {m['binary_f1']:.3f}  "
            f"(P={m['binary_precision']:.3f} R={m['binary_recall']:.3f})\n"
            f"Lethal recall: {m['lethal_recall']:.3f}  |  "
            f"FN dangerous: {m['fn_dangerous']}  |  FP phantom: {m['fp_phantom']}\n"
            f"Gradient MAE: {m['gradient_mae']:.1f}  |  "
            f"Gradient corr: {m['gradient_corr']:.3f}  |  "
            f"Coverage: {m['coverage_pct']:.1f}%")
    fig.text(0.5, 0.01, text, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison PNG: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="God-view vs robot-view costmap comparison")
    parser.add_argument("scenario", nargs="?", default="scattered")
    parser.add_argument("--robot", default="b2")
    parser.add_argument("--targets", type=int, default=None)
    parser.add_argument("--domain", type=int, default=20)
    parser.add_argument("--god-only", action="store_true",
                        help="Only build and visualize god-view costmap")
    parser.add_argument("--random-obstacles", action="store_true",
                        help="Randomize obstacle positions (uses --obstacle-seed)")
    parser.add_argument("--obstacle-seed", type=int, default=42,
                        help="Seed for obstacle randomization (default: 42)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from foreman.demos.target_game.scenarios import SCENARIOS
    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = SCENARIOS[args.scenario]
    if not scenario.has_obstacles:
        print(f"Scenario '{args.scenario}' has no obstacles — nothing to compare")
        sys.exit(1)

    obstacle_seed = args.obstacle_seed if args.random_obstacles else None
    scene_path = str(scenario.scene_path(args.robot, obstacle_seed))
    out_path = args.output or f"/tmp/costmap_compare_{args.scenario}.png"

    # Load perception config for matching parameters
    workspace_root = Path(__file__).resolve().parents[3]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from layer_6.config.defaults import load_config
    pcfg = load_config(args.robot)

    print(f"Scene: {scene_path}")
    print(f"Config: z_lo={pcfg.costmap_z_lo}, z_hi={pcfg.costmap_z_hi}, "
          f"res={pcfg.tsdf_output_resolution}, trunc={pcfg.tsdf_truncation}")

    # God-view costmap
    print("\nBuilding god-view costmap...")
    god_grid, god_meta = build_god_view_costmap(
        scene_path, z_lo=pcfg.costmap_z_lo, z_hi=pcfg.costmap_z_hi,
        output_resolution=pcfg.tsdf_output_resolution,
        xy_extent=pcfg.tsdf_xy_extent, truncation=pcfg.tsdf_truncation)
    print(f"  Grid: {god_grid.shape}, occupied(>=127): {int(np.sum(god_grid >= 127))}, "
          f"lethal(>=253): {int(np.sum(god_grid >= 253))}")

    if args.god_only:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(f'God-View Costmap: {args.scenario}')
        im = ax.imshow(god_grid.T, cmap='Reds', vmin=0, vmax=254, origin='lower')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved god-view PNG: {out_path}")
        return

    # Run game to build robot-view TSDF
    print("\nRunning robot simulation...")
    from foreman.demos.target_game.__main__ import run_game

    num_targets = args.targets if args.targets is not None else scenario.num_targets
    spawn_fn = None
    if scenario.spawn_fn_factory is not None:
        spawn_fn = scenario.spawn_fn_factory(scenario.target_seed)

    run_args = SimpleNamespace(
        robot=args.robot, targets=num_targets, headless=True,
        seed=scenario.target_seed, genome=None,
        full_circle=scenario.full_circle, domain=args.domain,
        slam=scenario.use_slam, obstacles=scenario.has_obstacles,
        scene_path=scene_path, timeout_per_target=scenario.timeout_per_target,
        min_dist=scenario.min_dist, max_dist=scenario.max_dist,
        angle_range=scenario.angle_range, spawn_fn=spawn_fn, viewer=False)

    result = run_game(run_args)
    print(f"\nGame: {result.stats.targets_reached}/{result.stats.targets_spawned} "
          f"reached, {result.stats.falls} falls")

    if result.tsdf is None:
        print("ERROR: No TSDF available (perception was not active)")
        sys.exit(1)

    # Robot-view costmap
    print("\nExtracting robot-view costmap...")
    robot_grid, robot_meta = result.tsdf.get_world_cost_grid(
        pcfg.costmap_z_lo, pcfg.costmap_z_hi, pcfg.tsdf_output_resolution)
    observed_mask = robot_meta.get('observed_mask')
    print(f"  Grid: {robot_grid.shape}, occupied: "
          f"{int(np.sum((robot_grid >= 127) & (robot_grid != 255)))}, "
          f"observed: {int(np.sum(observed_mask)) if observed_mask is not None else 0}")

    # Align grids if needed
    if god_grid.shape != robot_grid.shape:
        print(f"WARNING: size mismatch god={god_grid.shape} robot={robot_grid.shape}")
        nx = min(god_grid.shape[0], robot_grid.shape[0])
        ny = min(god_grid.shape[1], robot_grid.shape[1])
        god_grid, robot_grid = god_grid[:nx, :ny], robot_grid[:nx, :ny]
        if observed_mask is not None:
            observed_mask = observed_mask[:nx, :ny]

    # Compare
    metrics = compare_costmaps(god_grid, robot_grid, observed_mask)

    # 3DS god-mode metric (surface-level accuracy)
    from .test_occupancy import compute_3ds_god
    god_3ds = compute_3ds_god(result.tsdf, scene_path)
    metrics['3ds_god'] = god_3ds

    print(f"\n{'='*60}")
    print("COSTMAP COMPARISON METRICS")
    print(f"{'='*60}")
    for k, fmt in [('coverage_pct', '.1f'), ('mae', '.1f'), ('binary_f1', '.3f'),
                   ('binary_precision', '.3f'), ('binary_recall', '.3f'),
                   ('lethal_recall', '.3f'), ('fn_dangerous', 'd'),
                   ('fp_phantom', 'd'), ('gradient_mae', '.1f'),
                   ('gradient_corr', '.3f')]:
        print(f"  {k:20s}: {metrics[k]:{fmt}}")
    g = god_3ds
    print(f"  {'3ds_god':20s}: {g['score']:.1f}  "
          f"(prec={g['precision_score']:.1f} cpl={g['completeness_pct']:.1f}% "
          f"phn={g['phantom_penalty']:.1f})")
    print(f"{'='*60}")

    # Route score: compare A* paths if we have robot position and a target
    if result.stats.targets_spawned > 0 and hasattr(result, 'final_pos'):
        robot_xy = result.final_pos[:2] if result.final_pos is not None else (0.0, 0.0)
        target_xy = result.stats.last_target_pos if hasattr(result.stats, 'last_target_pos') else None
        if target_xy is not None:
            rs = compute_route_score(god_grid, god_meta, robot_grid, robot_meta,
                                     robot_xy, target_xy)
            metrics.update(rs)
            print(f"\n{'='*60}")
            print("ROUTE SCORE")
            print(f"{'='*60}")
            for k in ['rs_score', 'rs_length_ratio', 'rs_hausdorff',
                       'rs_overlap', 'god_path_m', 'robot_path_m']:
                fmt = '.1f' if k == 'rs_score' else '.3f'
                print(f"  {k:20s}: {rs[k]:{fmt}}")
            print(f"{'='*60}")

    save_comparison_png(god_grid, robot_grid, observed_mask, metrics, out_path)

    json_path = out_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {json_path}")

    # Diagnosis
    print("\nDiagnosis:")
    if metrics['fn_dangerous'] > 50:
        print("  [!] High FN dangerous — check scan coverage, Z-band filtering")
    if metrics['fp_phantom'] > 50:
        print("  [!] High FP phantom — check ground hits, self-hit filter")
    if metrics['gradient_mae'] > 50:
        print("  [!] High gradient MAE — check truncation/resolution mismatch")
    if metrics['coverage_pct'] < 10:
        print("  [!] Low coverage — check LiDAR range, game duration")
    if metrics['binary_f1'] > 0.7:
        print("  [OK] F1 > 0.7 — obstacle detection is good")
    elif metrics['binary_f1'] > 0.4:
        print("  [~] F1 0.4-0.7 — moderate, room for improvement")
    else:
        print("  [!] F1 < 0.4 — poor detection, needs investigation")


if __name__ == "__main__":
    main()

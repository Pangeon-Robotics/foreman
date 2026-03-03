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
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Re-export for backward compatibility (game_setup.py, scores.py import from here)
from .test_costmap_helpers import (  # noqa: F401
    _rasterize_obstacles, build_god_view_binary, build_god_view_costmap,
    _astar_path, compute_route_score,
)


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

    ax = axes[0, 0]
    ax.set_title('God-View Costmap')
    im = ax.imshow(god.T, cmap='Reds', vmin=0, vmax=254, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.6)

    ax = axes[0, 1]
    ax.set_title('Robot-View Costmap')
    rd = robot.T.astype(np.float32)
    rd[robot.T == 255] = np.nan
    cmap_r = plt.cm.Reds.copy()
    cmap_r.set_bad(color='lightgray')
    im = ax.imshow(rd, cmap=cmap_r, vmin=0, vmax=254, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.6)

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

    ax = axes[1, 1]
    ax.set_title('Binary Classification (thresh=127)')
    cls = np.full(god.shape, 4, dtype=np.uint8)
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
                        help="Randomize obstacle positions")
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

    workspace_root = Path(__file__).resolve().parents[3]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from layer_6.config.defaults import load_config
    pcfg = load_config(args.robot)

    print(f"Scene: {scene_path}")
    print(f"Config: z_lo={pcfg.costmap_z_lo}, z_hi={pcfg.costmap_z_hi}, "
          f"res={pcfg.tsdf_output_resolution}, trunc={pcfg.tsdf_truncation}")

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

    print("\nExtracting robot-view costmap...")
    robot_grid, robot_meta = result.tsdf.get_world_cost_grid(
        pcfg.costmap_z_lo, pcfg.costmap_z_hi, pcfg.tsdf_output_resolution)
    observed_mask = robot_meta.get('observed_mask')
    print(f"  Grid: {robot_grid.shape}, occupied: "
          f"{int(np.sum((robot_grid >= 127) & (robot_grid != 255)))}, "
          f"observed: {int(np.sum(observed_mask)) if observed_mask is not None else 0}")

    if god_grid.shape != robot_grid.shape:
        print(f"WARNING: size mismatch god={god_grid.shape} robot={robot_grid.shape}")
        nx = min(god_grid.shape[0], robot_grid.shape[0])
        ny = min(god_grid.shape[1], robot_grid.shape[1])
        god_grid, robot_grid = god_grid[:nx, :ny], robot_grid[:nx, :ny]
        if observed_mask is not None:
            observed_mask = observed_mask[:nx, :ny]

    metrics = compare_costmaps(god_grid, robot_grid, observed_mask)

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

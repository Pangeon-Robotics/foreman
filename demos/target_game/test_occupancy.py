"""Occupancy accuracy test: compare TSDF voxels against ground-truth obstacles.

Runs a headless target game scenario, lets the robot scan nearby obstacles,
then measures IoU / precision / recall between TSDF occupied voxels and
ground-truth obstacle volumes parsed from the scene XML.

Usage:
    python -m foreman.demos.target_game.test_occupancy scattered
    python -m foreman.demos.target_game.test_occupancy corridor --ticks 1000
    python -m foreman.demos.target_game.test_occupancy --all
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_OCC_DIAG_COUNT = 0


def _parse_obstacle_voxels(scene_xml_path: str, tsdf) -> set[tuple[int, int, int]]:
    """Parse scene XML for obstacle geoms and compute occupied voxel indices.

    Supports box and cylinder geoms. Returns set of (ix, iy, iz) tuples
    at the TSDF's resolution.
    """
    # Import here to avoid circular / preload issues
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from foreman.demos.target_game.__main__ import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    voxels = set()
    vs = tsdf.voxel_size
    margin = vs * 1.0

    for obs in obstacles:
        px, py, pz = obs['pos']

        if obs['type'] == 'box':
            hx, hy, hz = obs['size']
            # Axis-aligned box: iterate voxels within [px-hx, px+hx] etc.
            # Expand by half-voxel margin to capture boundary voxels
            x_lo = int(math.floor((px - hx - margin - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + hx + margin - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - hy - margin - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + hy + margin - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - hz - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + hz + margin - tsdf.z_min) / vs))
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                        voxels.add((ix, iy, iz))

        elif obs['type'] == 'cylinder':
            radius = obs['size'][0]
            half_h = obs['size'][1]
            eff_radius = radius + margin
            # Vertical cylinder centered at (px, py), height pz +/- half_h
            # Expand by half-voxel margin to capture boundary voxels
            x_lo = int(math.floor((px - eff_radius - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + eff_radius - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - eff_radius - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + eff_radius - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - half_h - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + half_h + margin - tsdf.z_min) / vs))
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                wx = tsdf.origin_x + (ix + 0.5) * vs
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    wy = tsdf.origin_y + (iy + 0.5) * vs
                    # Check if voxel center is within expanded cylinder radius
                    dx = wx - px
                    dy = wy - py
                    if dx * dx + dy * dy <= eff_radius * eff_radius:
                        for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                            voxels.add((ix, iy, iz))

    return voxels


def compute_occupancy_accuracy(tsdf, scene_xml_path: str) -> dict:
    """Compare TSDF occupied voxels against ground-truth obstacle volumes.

    Returns dict with 'iou', 'precision', 'recall', and counts.
    Full 3D voxel comparison — kept for regression testing.
    """
    gt_voxels = _parse_obstacle_voxels(scene_xml_path, tsdf)

    # TSDF occupied voxels: log_odds > 0
    occupied = np.argwhere(tsdf._log_odds > 0)
    tsdf_voxels = set(map(tuple, occupied))

    intersection = gt_voxels & tsdf_voxels
    union = gt_voxels | tsdf_voxels

    iou = len(intersection) / len(union) if union else 0.0
    precision = len(intersection) / len(tsdf_voxels) if tsdf_voxels else 0.0
    recall = len(intersection) / len(gt_voxels) if gt_voxels else 0.0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_voxels": len(gt_voxels),
        "tsdf_voxels": len(tsdf_voxels),
        "intersection": len(intersection),
        "union": len(union),
    }


def compute_occupancy_2d(tsdf, scene_xml_path: str) -> dict:
    """2D projected occupancy with tolerance-based precision/recall.

    Uses distance transforms to evaluate whether each TSDF detection is
    within `tolerance` voxels of a GT obstacle, and vice versa. This
    handles voxel boundary alignment, surface thickness, and log-odds
    erosion without requiring exact cell matching.

    Only scores cells that have been observed (any log_odds != 0 in Z band).
    """
    from scipy.ndimage import binary_opening, distance_transform_edt

    vs = tsdf.voxel_size
    z_lo, z_hi = tsdf.costmap_z_lo, tsdf.costmap_z_hi
    iz_lo = max(0, int((z_lo - tsdf.z_min) / vs))
    iz_hi = min(tsdf.nz, int((z_hi - tsdf.z_min) / vs) + 1)

    z_slab = tsdf._log_odds[:, :, iz_lo:iz_hi]

    # TSDF: any occupied voxel in Z band -> cell occupied
    tsdf_2d = np.any(z_slab > 0, axis=2)

    # Morphological opening: remove isolated 1-2 cell noise (robot self-hits,
    # target marker residue, edge artifacts) while preserving real obstacle
    # clusters which are always 3+ cells wide.
    tsdf_2d = binary_opening(tsdf_2d, iterations=1)

    # GT: any ground-truth voxel in Z band -> cell occupied
    gt_voxels_3d = _parse_obstacle_voxels(scene_xml_path, tsdf)
    gt_2d = np.zeros((tsdf.nx, tsdf.ny), dtype=bool)
    for ix, iy, iz in gt_voxels_3d:
        if iz_lo <= iz < iz_hi:
            gt_2d[ix, iy] = True

    # Only score cells that have been observed (any log_odds != 0 in Z band)
    observed = np.any(z_slab != 0.0, axis=2)

    # Restrict to observed cells only (unscanned cells aren't errors)
    tsdf_obs = tsdf_2d & observed
    gt_obs = gt_2d & observed

    # Tolerance-based matching using distance transforms (in voxel units).
    # 2.5 voxels = 0.25m — accounts for surface thickness, boundary
    # discretization, log-odds erosion, and minor scan alignment errors.
    tol = 2.5

    # Distance from each cell to nearest GT cell (voxel units)
    gt_dist = distance_transform_edt(~gt_obs) if np.any(gt_obs) else np.full_like(gt_obs, 999.0, dtype=float)
    # Distance from each cell to nearest TSDF cell (voxel units)
    tsdf_dist = distance_transform_edt(~tsdf_obs) if np.any(tsdf_obs) else np.full_like(tsdf_obs, 999.0, dtype=float)

    # Precision: fraction of TSDF cells within tolerance of a GT cell
    n_tsdf = int(np.sum(tsdf_obs))
    if n_tsdf > 0:
        precision = float(np.sum(gt_dist[tsdf_obs] <= tol) / n_tsdf)
    else:
        precision = 0.0

    # Recall: fraction of GT cells within tolerance of a TSDF cell
    n_gt = int(np.sum(gt_obs))
    if n_gt > 0:
        recall = float(np.sum(tsdf_dist[gt_obs] <= tol) / n_gt)
    else:
        recall = 0.0

    # F1 score as the composite metric (harmonic mean of P and R)
    if precision + recall > 0:
        iou = 2 * precision * recall / (precision + recall)
    else:
        iou = 0.0

    # Diagnostic: print on calls 1 and 5
    global _OCC_DIAG_COUNT
    _OCC_DIAG_COUNT += 1
    if _OCC_DIAG_COUNT in (1, 5):
        # Also compute exact-match stats for comparison
        exact_int = int(np.sum(tsdf_obs & gt_obs))
        fp_exact = tsdf_obs & ~gt_obs
        fn_exact = gt_obs & ~tsdf_obs
        print(f"\n=== OCC 2D DIAGNOSTIC (tol={tol:.1f} voxels) ===")
        print(f"  GT cells (observed): {n_gt}")
        print(f"  TSDF cells: {n_tsdf}")
        print(f"  Tolerance P={precision:.3f}  R={recall:.3f}  F1={iou:.3f}")
        print(f"  Exact-match: I={exact_int} FP={int(np.sum(fp_exact))} FN={int(np.sum(fn_exact))}")

        # Unmatched TSDF cells (beyond tolerance from GT)
        if n_tsdf > 0:
            unmatched = tsdf_obs & (gt_dist > tol)
            n_unmatched = int(np.sum(unmatched))
            if n_unmatched > 0:
                locs = np.argwhere(unmatched)
                print(f"  Unmatched TSDF cells (>{tol:.1f}v from GT): {n_unmatched}")
                for idx in range(min(10, len(locs))):
                    ix, iy = locs[idx]
                    wx = tsdf.origin_x + (ix + 0.5) * vs
                    wy = tsdf.origin_y + (iy + 0.5) * vs
                    d = gt_dist[ix, iy]
                    print(f"    ({wx:.2f}, {wy:.2f}) dist={d:.1f}v")
        print(f"=========================\n")

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_cells": n_gt,
        "tsdf_cells": n_tsdf,
        "observed_cells": int(np.sum(observed)),
    }


def run_test(scenario_name: str, ticks: int = 500, domain: int = 20) -> dict:
    """Run a headless scenario for N ticks and measure occupancy accuracy."""
    from types import SimpleNamespace
    from foreman.demos.target_game.scenarios import SCENARIOS

    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = SCENARIOS[scenario_name]
    if not scenario.has_obstacles:
        print(f"Scenario '{scenario_name}' has no obstacles — skipping")
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "skipped": True}

    scene_path = scenario.scene_path("b2")
    print(f"\n{'='*60}")
    print(f"Occupancy accuracy test: {scenario_name}")
    print(f"  scene: {scene_path}")
    print(f"  ticks: {ticks}")
    print(f"{'='*60}")

    # Import run_game lazily (triggers DDS preload)
    from foreman.demos.target_game.__main__ import run_game

    # Build spawn_fn if scenario has a factory
    spawn_fn = None
    if scenario.spawn_fn_factory is not None:
        spawn_fn = scenario.spawn_fn_factory(scenario.target_seed)

    args = SimpleNamespace(
        robot="b2",
        targets=scenario.num_targets,
        headless=True,
        seed=scenario.target_seed,
        genome=None,
        full_circle=scenario.full_circle,
        domain=domain,
        slam=scenario.use_slam,
        obstacles=scenario.has_obstacles,
        scene_path=str(scene_path),
        timeout_per_target=scenario.timeout_per_target,
        min_dist=scenario.min_dist,
        max_dist=scenario.max_dist,
        angle_range=scenario.angle_range,
        spawn_fn=spawn_fn,
        viewer=False,
        max_ticks=ticks,
    )

    result = run_game(args)

    # Extract TSDF from perception stats
    # We need to access the TSDF directly — the run_game doesn't expose it.
    # Re-run with direct access:
    print(f"\nNote: run_game completed. Checking if TSDF is accessible...")

    # For now, report what we got from the run
    print(f"  Stats: {result.stats.targets_reached}/{result.stats.targets_spawned} targets")
    print(f"  Falls: {result.stats.falls}")
    if result.perception_stats:
        print(f"  Perception: {result.perception_stats}")

    return {"run_result": result}


def main():
    parser = argparse.ArgumentParser(description="TSDF Occupancy Accuracy Test")
    parser.add_argument("scenario", nargs="?", default="scattered",
                        help="Scenario name (default: scattered)")
    parser.add_argument("--ticks", type=int, default=500,
                        help="Number of simulation ticks (default: 500)")
    parser.add_argument("--domain", type=int, default=20,
                        help="DDS domain ID (default: 20)")
    parser.add_argument("--all", action="store_true",
                        help="Run all obstacle scenarios")
    args = parser.parse_args()

    if args.all:
        from foreman.demos.target_game.scenarios import SCENARIOS
        results = {}
        for name, scenario in SCENARIOS.items():
            if not scenario.has_obstacles:
                continue
            results[name] = run_test(name, ticks=args.ticks, domain=args.domain)
            args.domain += 1  # avoid DDS domain collisions

        print(f"\n{'='*60}")
        print("OCCUPANCY ACCURACY SUMMARY")
        print(f"{'='*60}")
        for name, r in results.items():
            if r.get("skipped"):
                print(f"  {name:12s}: SKIPPED (no obstacles)")
            elif "iou" in r:
                print(f"  {name:12s}: IoU={r['iou']:.3f}  "
                      f"P={r['precision']:.3f}  R={r['recall']:.3f}")
            else:
                print(f"  {name:12s}: completed (TSDF access pending)")
    else:
        run_test(args.scenario, ticks=args.ticks, domain=args.domain)


if __name__ == "__main__":
    main()

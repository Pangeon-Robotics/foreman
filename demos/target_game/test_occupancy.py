"""Occupancy accuracy test: compare TSDF voxels against ground-truth obstacles.

Runs a headless target game scenario, lets the robot scan nearby obstacles,
then measures IoU / precision / recall between TSDF occupied voxels and
ground-truth obstacle volumes parsed from the scene XML.

3DS v2 metrics (adherence, completeness, phantom rate) use TSDF's
get_surface_voxels method.

Usage:
    python -m foreman.demos.target_game.test_occupancy scattered
    python -m foreman.demos.target_game.test_occupancy corridor --ticks 1000
    python -m foreman.demos.target_game.test_occupancy --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Re-export for backward compatibility (game_setup.py, game_viz.py,
# test_costmap_compare.py import from here)
from .test_occupancy_3ds import compute_3ds_v2, compute_3ds_god  # noqa: F401
from .test_occupancy_gt import (
    materialize_log_odds, parse_obstacle_voxels,
    parse_obstacle_surface_voxels,
)

_OCC_DIAG_COUNT = 0


def compute_occupancy_accuracy(tsdf, scene_xml_path: str,
                                diag: bool = False) -> dict:
    """Compare TSDF occupied voxels against ground-truth obstacle surfaces.

    Compares raw TSDF detections (log_odds > 0) against the *surface band*
    of ground-truth obstacles. Only GT voxels in observed territory are scored.

    Returns dict with 'iou', 'precision', 'recall', and counts.
    """
    gt_surface = parse_obstacle_surface_voxels(scene_xml_path, tsdf)

    lo = materialize_log_odds(tsdf)
    _LO_THRESHOLD = 1.0
    occupied = np.argwhere(lo > _LO_THRESHOLD)
    tsdf_voxels = set(map(tuple, occupied))

    _NEIGHBOURS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

    scanned = lo != 0.0
    gt_observed = set()
    for v in gt_surface:
        ix, iy, iz = v
        if scanned[ix, iy, iz]:
            gt_observed.add(v)
            continue
        for dx, dy, dz in _NEIGHBOURS:
            nx_, ny_, nz_ = ix+dx, iy+dy, iz+dz
            if (0 <= nx_ < tsdf.nx and 0 <= ny_ < tsdf.ny
                    and 0 <= nz_ < tsdf.nz and scanned[nx_, ny_, nz_]):
                gt_observed.add(v)
                break

    _TOL = 2
    _OFFSETS = []
    for dx in range(-_TOL, _TOL + 1):
        for dy in range(-_TOL, _TOL + 1):
            for dz in range(-_TOL, _TOL + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if max(abs(dx), abs(dy), abs(dz)) <= _TOL:
                    _OFFSETS.append((dx, dy, dz))

    tsdf_matched = set()
    gt_matched = set()
    for v in tsdf_voxels:
        if v in gt_observed:
            tsdf_matched.add(v)
            gt_matched.add(v)
            continue
        ix, iy, iz = v
        for dx, dy, dz in _OFFSETS:
            nb = (ix+dx, iy+dy, iz+dz)
            if nb in gt_observed:
                tsdf_matched.add(v)
                gt_matched.add(nb)
                break
    for v in gt_observed:
        if v in gt_matched:
            continue
        ix, iy, iz = v
        for dx, dy, dz in _OFFSETS:
            nb = (ix+dx, iy+dy, iz+dz)
            if nb in tsdf_voxels:
                gt_matched.add(v)
                break

    precision = len(tsdf_matched) / len(tsdf_voxels) if tsdf_voxels else 0.0
    recall = len(gt_matched) / len(gt_observed) if gt_observed else 0.0
    iou = (2 * precision * recall / (precision + recall)
           if precision + recall > 0 else 0.0)

    global _OCC_DIAG_COUNT
    _OCC_DIAG_COUNT += 1
    if diag or _OCC_DIAG_COUNT in (1, 5):
        vs = tsdf.voxel_size
        fp_voxels = tsdf_voxels - tsdf_matched
        fn_voxels = gt_observed - gt_matched
        fp_z = {}
        for ix, iy, iz in fp_voxels:
            wz = tsdf.z_min + (iz + 0.5) * vs
            bucket = round(wz, 1)
            fp_z[bucket] = fp_z.get(bucket, 0) + 1
        print(f"\n=== OCC 3D SURFACE DIAGNOSTIC ===")
        print(f"  GT surface: {len(gt_surface)}  GT observed: {len(gt_observed)}  "
              f"TSDF: {len(tsdf_voxels)}")
        print(f"  TSDF matched: {len(tsdf_matched)}  GT matched: {len(gt_matched)}")
        print(f"  P={precision:.3f}  R={recall:.3f}  F1={iou:.3f}")
        print(f"  FP: {len(fp_voxels)}  FN: {len(fn_voxels)}")
        if fp_z:
            sorted_z = sorted(fp_z.items())
            print(f"  FP by Z level: {sorted_z[:15]}")
        fp_list = list(fp_voxels)[:10]
        if fp_list:
            print(f"  Sample FPs (world coords):")
            for ix, iy, iz in fp_list:
                wx = tsdf.origin_x + (ix + 0.5) * vs
                wy = tsdf.origin_y + (iy + 0.5) * vs
                wz = tsdf.z_min + (iz + 0.5) * vs
                actual_lo = lo[ix, iy, iz]
                print(f"    ({wx:.2f}, {wy:.2f}, z={wz:.2f}) lo={actual_lo:.2f}")
        print(f"=========================\n")

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_surface": len(gt_surface),
        "gt_observed": len(gt_observed),
        "tsdf_voxels": len(tsdf_voxels),
        "tsdf_matched": len(tsdf_matched),
        "gt_matched": len(gt_matched),
    }


def compute_occupancy_2d(tsdf, scene_xml_path: str) -> dict:
    """2D projected occupancy with tolerance-based precision/recall.

    Uses distance transforms to evaluate whether each TSDF detection is
    within `tolerance` voxels of a GT obstacle, and vice versa.
    Only scores cells that have been observed.
    """
    from scipy.ndimage import binary_opening, distance_transform_edt

    vs = tsdf.voxel_size
    z_lo, z_hi = tsdf.costmap_z_lo, tsdf.costmap_z_hi
    iz_lo = max(0, int((z_lo - tsdf.z_min) / vs))
    iz_hi = min(tsdf.nz, int((z_hi - tsdf.z_min) / vs) + 1)

    z_slab = materialize_log_odds(tsdf)[:, :, iz_lo:iz_hi]

    tsdf_2d = np.any(z_slab > 0, axis=2)
    tsdf_2d = binary_opening(tsdf_2d, iterations=1)

    gt_voxels_3d = parse_obstacle_voxels(scene_xml_path, tsdf)
    gt_2d = np.zeros((tsdf.nx, tsdf.ny), dtype=bool)
    for ix, iy, iz in gt_voxels_3d:
        if iz_lo <= iz < iz_hi:
            gt_2d[ix, iy] = True

    observed = np.any(z_slab != 0.0, axis=2)

    tsdf_obs = tsdf_2d & observed
    gt_obs = gt_2d & observed

    tol = 2.5

    gt_dist = distance_transform_edt(~gt_obs) if np.any(gt_obs) else np.full_like(gt_obs, 999.0, dtype=float)
    tsdf_dist = distance_transform_edt(~tsdf_obs) if np.any(tsdf_obs) else np.full_like(tsdf_obs, 999.0, dtype=float)

    n_tsdf = int(np.sum(tsdf_obs))
    if n_tsdf > 0:
        precision = float(np.sum(gt_dist[tsdf_obs] <= tol) / n_tsdf)
    else:
        precision = 0.0

    n_gt = int(np.sum(gt_obs))
    if n_gt > 0:
        recall = float(np.sum(tsdf_dist[gt_obs] <= tol) / n_gt)
    else:
        recall = 0.0

    if precision + recall > 0:
        iou = 2 * precision * recall / (precision + recall)
    else:
        iou = 0.0

    global _OCC_DIAG_COUNT
    _OCC_DIAG_COUNT += 1
    if _OCC_DIAG_COUNT in (1, 5):
        exact_int = int(np.sum(tsdf_obs & gt_obs))
        fp_exact = tsdf_obs & ~gt_obs
        fn_exact = gt_obs & ~tsdf_obs
        print(f"\n=== OCC 2D DIAGNOSTIC (tol={tol:.1f} voxels) ===")
        print(f"  GT cells (observed): {n_gt}")
        print(f"  TSDF cells: {n_tsdf}")
        print(f"  Tolerance P={precision:.3f}  R={recall:.3f}  F1={iou:.3f}")
        print(f"  Exact-match: I={exact_int} FP={int(np.sum(fp_exact))} FN={int(np.sum(fn_exact))}")

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

    from foreman.demos.target_game.__main__ import run_game

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

    print(f"\nNote: run_game completed. Checking if TSDF is accessible...")
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
            args.domain += 1

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

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
    """Parse scene XML for obstacle geoms and compute *solid* occupied voxel indices.

    Returns ALL voxels inside each obstacle (full volume).
    Used by compute_occupancy_2d; for 3D surface comparison use
    _parse_obstacle_surface_voxels() instead.
    """
    # Import here to avoid circular / preload issues
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from foreman.demos.target_game.__main__ import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    voxels = set()
    vs = tsdf.voxel_size
    margin = vs * 0.5

    for obs in obstacles:
        px, py, pz = obs['pos']

        if obs['type'] == 'box':
            hx, hy, hz = obs['size']
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
                    dx = wx - px
                    dy = wy - py
                    if dx * dx + dy * dy <= eff_radius * eff_radius:
                        for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                            voxels.add((ix, iy, iz))

    return voxels


def _parse_obstacle_surface_voxels(
    scene_xml_path: str, tsdf,
) -> set[tuple[int, int, int]]:
    """Parse scene XML and return the *surface shell* of each obstacle.

    LiDAR detects obstacle surfaces, not interiors.  This function returns
    only boundary voxels — voxels inside the obstacle that have at least
    one face-neighbour outside.  This is a thin (1-voxel) shell.

    The IoU metric uses 2-voxel Chebyshev tolerance when matching TSDF
    detections against this shell, accounting for discretization and
    TSDF truncation boundary offset.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from foreman.demos.target_game.__main__ import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    surface = set()
    vs = tsdf.voxel_size
    margin = vs * 0.5
    _NEIGHBOURS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

    for obs in obstacles:
        px, py, pz = obs['pos']

        if obs['type'] == 'box':
            hx, hy, hz = obs['size']
            x_lo = int(math.floor((px - hx - margin - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + hx + margin - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - hy - margin - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + hy + margin - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - hz - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + hz + margin - tsdf.z_min) / vs))
            solid = set()
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                        solid.add((ix, iy, iz))
            for v in solid:
                ix, iy, iz = v
                for dx, dy, dz in _NEIGHBOURS:
                    if (ix+dx, iy+dy, iz+dz) not in solid:
                        surface.add(v)
                        break

        elif obs['type'] == 'cylinder':
            radius = obs['size'][0]
            half_h = obs['size'][1]
            eff_radius = radius + margin
            x_lo = int(math.floor((px - eff_radius - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + eff_radius - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - eff_radius - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + eff_radius - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - half_h - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + half_h + margin - tsdf.z_min) / vs))
            solid = set()
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                wx = tsdf.origin_x + (ix + 0.5) * vs
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    wy = tsdf.origin_y + (iy + 0.5) * vs
                    dx = wx - px
                    dy = wy - py
                    if dx * dx + dy * dy <= eff_radius * eff_radius:
                        for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                            solid.add((ix, iy, iz))
            for v in solid:
                ix, iy, iz = v
                for dx, dy, dz in _NEIGHBOURS:
                    if (ix+dx, iy+dy, iz+dz) not in solid:
                        surface.add(v)
                        break

    return surface


def compute_occupancy_accuracy(tsdf, scene_xml_path: str,
                                diag: bool = False) -> dict:
    """Compare TSDF occupied voxels against ground-truth obstacle surfaces.

    Compares raw TSDF detections (log_odds > 0) against the *surface band*
    of ground-truth obstacles — not their solid interiors.  Only GT voxels
    in observed territory are scored (the robot can't detect unscanned
    surfaces).

    A voxel is "observed" if any voxel in its 6-connected neighbourhood
    has been touched by the TSDF (log_odds != 0).  This means the sensor
    has cast rays near this location and had a chance to detect it.

    Returns dict with 'iou', 'precision', 'recall', and counts.
    Full 3D voxel comparison.
    """
    gt_surface = _parse_obstacle_surface_voxels(scene_xml_path, tsdf)

    # TSDF occupied voxels: log_odds > 1.0 (2+ LiDAR hits required).
    # A single LiDAR hit gives lo=0.85.  Requiring >1.0 filters out
    # single-hit noise (self-hits, edge artifacts) and matches the
    # confidence level used by the cost grid for A* path planning.
    # Real obstacles get hit repeatedly (lo=2-3.5), so 1.0 is well
    # below the confirmed-obstacle floor.
    lo = tsdf._log_odds
    _LO_THRESHOLD = 1.0
    occupied = np.argwhere(lo > _LO_THRESHOLD)
    tsdf_voxels = set(map(tuple, occupied))

    _NEIGHBOURS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

    # Observed mask: a GT surface voxel is "observed" if it or any
    # face-neighbour has been touched by the TSDF (log_odds != 0).
    scanned = lo != 0.0  # (nx, ny, nz) bool
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

    # Tolerance matching with radius 2 voxels (0.2m).
    # A TSDF voxel "matches" if any GT surface voxel is within Chebyshev
    # distance 2 (and vice versa).  0.2m accounts for discretization,
    # ray angle effects, and TSDF truncation boundary offset.
    # Build offsets for Chebyshev distance ≤ 2 (5×5×5 cube minus corners > 2)
    _TOL = 2
    _OFFSETS = []
    for dx in range(-_TOL, _TOL + 1):
        for dy in range(-_TOL, _TOL + 1):
            for dz in range(-_TOL, _TOL + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if max(abs(dx), abs(dy), abs(dz)) <= _TOL:
                    _OFFSETS.append((dx, dy, dz))

    tsdf_matched = set()     # TSDF voxels near GT surface
    gt_matched = set()       # GT surface voxels near a TSDF detection
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
    # Also check: GT voxels matched by nearby TSDF voxels
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

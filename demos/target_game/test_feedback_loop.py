"""Fast headless test harness for costmap feedback loop metrics.

Runs multiple headless scattered scenarios and aggregates Surface F1,
Cost F1, Router F1, ATO, targets reached, and falls across seeds.
Exit code 0 if passing, 1 if not.

Usage:
    python -m foreman.demos.target_game.test_feedback_loop
    python -m foreman.demos.target_game.test_feedback_loop --seeds 5 --domain 20
    python -m foreman.demos.target_game.test_feedback_loop --random-obstacles
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Passing thresholds
THRESH_TARGETS = 3    # mean targets reached (of 4)
THRESH_FALLS = 1      # mean falls
THRESH_SURFACE = 50.0  # mean Surface F1
THRESH_COST = 50.0     # mean Cost F1
THRESH_ROUTER = 50.0   # mean Router F1


def _run_seed(
    seed: int, robot: str, domain: int, random_obstacles: bool,
    *, no_slam: bool = False,
) -> dict:
    """Run a single headless scattered scenario and collect all metrics."""
    from foreman.demos.target_game.__main__ import run_game
    from foreman.demos.target_game.scenarios import SCENARIOS

    scenario = SCENARIOS["scattered"]
    obstacle_seed = seed if random_obstacles else None
    scene_path = str(scenario.scene_path(robot, obstacle_seed))

    spawn_fn = None
    if scenario.spawn_fn_factory is not None:
        spawn_fn = scenario.spawn_fn_factory(seed)

    run_args = SimpleNamespace(
        robot=robot, targets=scenario.num_targets, headless=True,
        seed=seed, genome=None, full_circle=scenario.full_circle,
        domain=domain, slam=(scenario.use_slam and not no_slam),
        obstacles=scenario.has_obstacles,
        scene_path=scene_path, timeout_per_target=scenario.timeout_per_target,
        min_dist=scenario.min_dist, max_dist=scenario.max_dist,
        angle_range=scenario.angle_range, spawn_fn=spawn_fn, viewer=False,
        god=True)

    t0 = time.monotonic()
    result = run_game(run_args)
    elapsed = time.monotonic() - t0

    stats = result.stats
    row = {
        "seed": seed,
        "targets": stats.targets_reached,
        "spawned": stats.targets_spawned,
        "falls": stats.falls,
        "ato": result.ato_score or 0.0,
        "time": round(elapsed, 1),
    }

    # New F1 scores from god-view vs robot-view TSDF
    if result.scores:
        row["surface"] = result.scores["surface"]["f1"]
        row["cost"] = result.scores["cost"]["f1"]
        row["router"] = result.scores["router"]["f1"]
    else:
        row["surface"] = 0.0
        row["cost"] = 0.0
        row["router"] = 0.0

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Fast headless feedback loop test")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run (default 5)")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting seed (default 42)")
    parser.add_argument("--robot", default="b2")
    parser.add_argument("--domain", type=int, default=20)
    parser.add_argument("--random-obstacles", action="store_true",
                        help="Randomize obstacle positions per seed")
    parser.add_argument("--no-slam", action="store_true",
                        help="Use ground-truth position (isolate costmap quality from SLAM drift)")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"Feedback loop test: {len(seeds)} seeds, robot={args.robot}, "
          f"domain={args.domain}")
    print(f"Seeds: {seeds}")
    if args.random_obstacles:
        print("Random obstacles: ON")
    if args.no_slam:
        print("SLAM: OFF (ground-truth position)")
    print()

    results = []
    t_total = time.monotonic()

    for i, seed in enumerate(seeds):
        # Kill stale firmware on THIS seed's domain (each seed gets unique domain)
        import subprocess, glob
        seed_domain_cleanup = args.domain + i
        subprocess.run(
            ["pkill", "-9", "-f",
             f"firmware_sim.py.*--domain {seed_domain_cleanup}"],
            capture_output=True)
        for f in glob.glob(f"/tmp/robo_sessions/*domain{seed_domain_cleanup}*.json"):
            try:
                os.remove(f)
            except OSError:
                pass
        # Also clean domain-unaware session files (legacy)
        for f in glob.glob(f"/tmp/robo_sessions/{args.robot}.json"):
            try:
                os.remove(f)
            except OSError:
                pass
        # Force GC to reclaim TSDF/perception memory from previous seed
        import gc
        gc.collect()
        time.sleep(4.0)  # let processes die and DDS resources release

        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        # Increment domain per seed to avoid DDS resource conflicts
        seed_domain = args.domain + i
        row = _run_seed(seed, args.robot, seed_domain, args.random_obstacles,
                        no_slam=args.no_slam)
        results.append(row)
        print(f"  => targets={row['targets']}/{row['spawned']} "
              f"falls={row['falls']} ato={row['ato']:.1f} "
              f"surf={row['surface']:.1f} cost={row['cost']:.1f} "
              f"rtr={row['router']:.1f} ({row['time']:.0f}s)")

    total_time = time.monotonic() - t_total

    # Summary table
    print(f"\n\n{'='*80}")
    print("FEEDBACK LOOP SUMMARY")
    print(f"{'='*80}")
    print(f" {'seed':>6}  {'tgt':>5}  {'fall':>4}  {'ATO':>6}  "
          f"{'surf':>6}  {'cost':>6}  {'rtr':>6}  {'time':>5}")
    print(f" {'-'*6}  {'-'*5}  {'-'*4}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")
    for r in results:
        print(f" {r['seed']:>6}  {r['targets']:>2}/{r['spawned']:<2}  "
              f"{r['falls']:>4}  {r['ato']:>6.1f}  "
              f"{r['surface']:>6.1f}  {r['cost']:>6.1f}  "
              f"{r['router']:>6.1f}  {r['time']:>4.0f}s")

    # Aggregates
    targets = [r["targets"] for r in results]
    falls = [r["falls"] for r in results]
    atos = [r["ato"] for r in results]
    scores_surf = [r["surface"] for r in results]
    scores_cost = [r["cost"] for r in results]
    scores_rtr = [r["router"] for r in results]

    m_tgt = np.mean(targets)
    m_fall = np.mean(falls)
    m_ato = np.mean(atos)
    m_surf = np.mean(scores_surf)
    m_cost = np.mean(scores_cost)
    m_rtr = np.mean(scores_rtr)

    print(f" {'-'*6}  {'-'*5}  {'-'*4}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}")
    print(f" {'mean':>6}  {m_tgt:>5.1f}  {m_fall:>4.1f}  {m_ato:>6.1f}  "
          f"{m_surf:>6.1f}  {m_cost:>6.1f}  {m_rtr:>6.1f}  {total_time:>4.0f}s")
    print(f" {'min':>6}  {min(targets):>5}  {min(falls):>4}  "
          f"{min(atos):>6.1f}  {min(scores_surf):>6.1f}  "
          f"{min(scores_cost):>6.1f}  {min(scores_rtr):>6.1f}")
    print(f" {'max':>6}  {max(targets):>5}  {max(falls):>4}  "
          f"{max(atos):>6.1f}  {max(scores_surf):>6.1f}  "
          f"{max(scores_cost):>6.1f}  {max(scores_rtr):>6.1f}")
    print(f"{'='*80}")

    # Pass/fail
    checks = [
        ("targets", m_tgt >= THRESH_TARGETS, f"{m_tgt:.1f} >= {THRESH_TARGETS}"),
        ("falls", m_fall <= THRESH_FALLS, f"{m_fall:.1f} <= {THRESH_FALLS}"),
        ("surface", m_surf >= THRESH_SURFACE, f"{m_surf:.1f} >= {THRESH_SURFACE}"),
        ("cost", m_cost >= THRESH_COST, f"{m_cost:.1f} >= {THRESH_COST}"),
        ("router", m_rtr >= THRESH_ROUTER, f"{m_rtr:.1f} >= {THRESH_ROUTER}"),
    ]

    print("\nThreshold checks:")
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        marker = " " if passed else "!"
        print(f"  [{marker}] {name:10s}: {detail}  {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nRESULT: PASS")
        return 0
    else:
        print("\nRESULT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())

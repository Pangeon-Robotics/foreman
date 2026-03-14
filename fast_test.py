#!/usr/bin/env python3
"""Parallel scattered scenario test harness.

Runs N instances of the scattered scenario in parallel, each on a separate
DDS domain, and prints a statistical summary of ATO scores.

Usage:
    python fast_test.py              # 4 parallel runs
    python fast_test.py -n 8         # 8 parallel runs
    python fast_test.py -n 2 --seed-offset 100  # 2 runs starting from seed 100
    python fast_test.py --targets 3 --config tmp/ato_candidates/iter_001.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from test_result_parser import parse_output

_FOREMAN = Path(__file__).resolve().parent
_WORKSPACE = _FOREMAN.parent

# Environment: LD_PRELOAD for CycloneDDS 0.10.4
_ENV = os.environ.copy()
_ENV["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"

# Module-level config for worker processes (set via _init_worker)
_worker_targets = 4


def _init_worker(targets):
    """Initialize module-level state in worker processes."""
    global _worker_targets
    _worker_targets = targets


def kill_firmware_on_domain(domain_id: int) -> None:
    """Kill any firmware processes using this DDS domain."""
    subprocess.run(
        ["pkill", "-9", "-f", f"firmware_sim.py.*domain.*{domain_id}"],
        capture_output=True,
    )
    # Clean up session files (both old and new naming conventions)
    for pattern in [f"b2_domain{domain_id}.json", f"b2.json"]:
        session_file = Path(f"/tmp/robo_sessions/{pattern}")
        if session_file.exists():
            session_file.unlink(missing_ok=True)


def run_worker(args: tuple) -> dict:
    """Run a single scattered scenario on the given domain.

    Args is a tuple of (domain_id, seed) to work with ProcessPoolExecutor.map().
    Uses module-level _worker_targets and _worker_genome set by _init_worker.

    Returns parsed result dict.
    """
    domain_id, seed = args
    kill_firmware_on_domain(domain_id)

    cmd = [
        sys.executable,
        str(_FOREMAN / "run_scenario.py"),
        "scattered",
        "--headless",
        "--domain", str(domain_id),
        "--seed", str(seed),
        "--targets", str(_worker_targets),
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=360,
            env=_ENV,
            cwd=str(_FOREMAN),
        )
        wall_time = time.monotonic() - t0
        result = parse_output(proc.stdout)
        result["wall_time"] = wall_time
        result["seed"] = seed
        result["domain"] = domain_id
        result["returncode"] = proc.returncode
        if proc.returncode != 0 and result["aggregate_ato"] is None:
            result["error"] = f"exit code {proc.returncode}"
            # Capture last 5 lines of stderr for debugging
            stderr_lines = proc.stderr.strip().split("\n")
            result["stderr_tail"] = "\n".join(stderr_lines[-5:])
        return result
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - t0
        return {
            "targets": [],
            "aggregate_ato": None,
            "reached": 0,
            "total": 0,
            "falls": 0,
            "wall_time": wall_time,
            "seed": seed,
            "domain": domain_id,
            "returncode": -1,
            "error": "timeout (360s)",
        }
    finally:
        kill_firmware_on_domain(domain_id)


def compute_speed_ratio(v_avg: float, v_ref: float = 2.0) -> float:
    """Compute speed_ratio component from v_avg and V_REF."""
    return v_avg / v_ref if v_ref > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Parallel scattered scenario test harness")
    parser.add_argument("-n", type=int, default=4,
                        help="Number of parallel runs (default: 4)")
    parser.add_argument("--domain-start", type=int, default=10,
                        help="Starting DDS domain ID (default: 10)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset for reproducibility")
    parser.add_argument("--targets", type=int, default=None,
                        help="Number of targets per run (default: scenario default)")
    parser.add_argument("--results-json", type=str, default=None,
                        help="Path to write results JSON (default: tmp/ato_results.json)")
    args = parser.parse_args()

    n = args.n
    domain_start = args.domain_start
    targets = args.targets

    seeds = [args.seed_offset + i for i in range(n)]
    targets_str = f"{targets} targets" if targets else "scenario default targets"
    print(f"Running {n} parallel scattered scenarios (domains {domain_start}-{domain_start + n - 1})")
    print(f"Seeds: {seeds}")
    print(f"Each run: {targets_str}")
    print()

    t0 = time.monotonic()

    # Launch all workers as subprocesses in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    domains = [domain_start + i for i in range(n)]
    results = {}

    worker_args = [(domain_start + i, seeds[i]) for i in range(n)]

    with ProcessPoolExecutor(
        max_workers=n,
        initializer=_init_worker,
        initargs=(targets or 4,),
    ) as executor:
        futures = {
            executor.submit(run_worker, wa): wa[0]
            for wa in worker_args
        }
        for future in as_completed(futures):
            dom = futures[future]
            try:
                results[dom] = future.result()
            except Exception as e:
                results[dom] = {
                    "targets": [],
                    "aggregate_ato": None,
                    "reached": 0,
                    "total": 0,
                    "falls": 0,
                    "wall_time": 0,
                    "seed": 0,
                    "domain": dom,
                    "returncode": -1,
                    "error": str(e),
                }

    total_wall = time.monotonic() - t0

    # Print results
    print(f"\n{'='*72}")
    print(f"PARALLEL TEST RESULTS ({n} runs)")
    print(f"{'='*72}")
    print()

    header = f"{'Run':>4}  {'Seed':>4}  {'Targets':>8}  {'Falls':>5}  {'ATO':>6}  {'Wall':>6}  {'Per-Target ATO':>30}"
    print(header)
    print("-" * len(header))

    ato_scores = []
    all_results = []  # for JSON export
    path_effs = []
    v_avgs = []
    slip_effs = []
    reg_gates = []
    total_falls = 0

    for i, dom in enumerate(sorted(results.keys())):
        r = results[dom]
        run_num = dom - domain_start + 1
        seed = r.get("seed", seeds[i])

        if r.get("error") and r["aggregate_ato"] is None:
            print(f"{run_num:>4}  {seed:>4}  {'ERROR':>8}  {'':>5}  {'':>6}  {r['wall_time']:>5.1f}s  {r['error']}")
            if r.get("stderr_tail"):
                for sl in r["stderr_tail"].split("\n"):
                    print(f"        {sl}")
            all_results.append(r)
            continue

        targets_str = f"{r['reached']}/{r['total']}"
        falls_str = str(r.get("falls", 0))
        total_falls += r.get("falls", 0)

        agg_ato = r["aggregate_ato"]
        if agg_ato is not None:
            ato_str = f"{agg_ato:.1f}"
            ato_scores.append(agg_ato)
        else:
            ato_str = "n/a"

        if r.get("path_efficiency") is not None:
            path_effs.append(r["path_efficiency"])
        if r.get("v_avg") is not None:
            v_avgs.append(r["v_avg"])
        if r.get("slip_efficiency") is not None:
            slip_effs.append(r["slip_efficiency"])
        if r.get("regression") is not None:
            reg_gates.append(r["regression"])

        wall_str = f"{r['wall_time']:.1f}s"

        # Per-target ATO breakdown
        per_target = []
        for t in r["targets"]:
            if t["timeout"]:
                per_target.append(f"T{t['idx']}:TOUT")
            else:
                per_target.append(f"T{t['idx']}:{t['ato']:.1f}")
        per_target_str = "  ".join(per_target) if per_target else "n/a"

        print(f"{run_num:>4}  {seed:>4}  {targets_str:>8}  {falls_str:>5}  {ato_str:>6}  {wall_str:>6}  {per_target_str}")
        all_results.append(r)

    # Statistics
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    if ato_scores:
        ato_scores.sort()
        mean_ato = sum(ato_scores) / len(ato_scores)
        median_idx = len(ato_scores) // 2
        if len(ato_scores) % 2 == 0 and len(ato_scores) > 1:
            median_ato = (ato_scores[median_idx - 1] + ato_scores[median_idx]) / 2
        else:
            median_ato = ato_scores[median_idx]
        min_ato = ato_scores[0]
        max_ato = ato_scores[-1]

        print(f"  Runs with ATO: {len(ato_scores)}/{n}")
        print(f"  Mean ATO:   {mean_ato:.1f}")
        print(f"  Median ATO: {median_ato:.1f}")
        print(f"  Min ATO:    {min_ato:.1f}")
        print(f"  Max ATO:    {max_ato:.1f}")
    else:
        mean_ato = 0.0
        median_ato = 0.0
        min_ato = 0.0
        max_ato = 0.0
        print("  No successful runs with ATO scores")

    # Component breakdown
    bottleneck = None
    if path_effs or v_avgs or reg_gates:
        print()
        print("  COMPONENT BREAKDOWN (bottleneck identification)")
        print("  " + "-" * 50)
        if path_effs:
            mean_pe = sum(path_effs) / len(path_effs)
            print(f"  Path efficiency:  {mean_pe:.1%}  (min={min(path_effs):.1%}, max={max(path_effs):.1%})")
        if v_avgs:
            mean_v = sum(v_avgs) / len(v_avgs)
            speed_ratio = compute_speed_ratio(mean_v)
            print(f"  Speed (v_avg):    {mean_v:.2f} m/s  (ratio={speed_ratio:.2f}, V_REF=2.0)")
        if slip_effs:
            mean_slip = sum(slip_effs) / len(slip_effs)
            print(f"  Slip efficiency:  {mean_slip:.1%}  (min={min(slip_effs):.1%}, max={max(slip_effs):.1%})")
        if reg_gates:
            mean_rg = sum(reg_gates) / len(reg_gates)
            print(f"  Regression gate:  {mean_rg:.2f}  (min={min(reg_gates):.2f})")
        print(f"  Total falls:      {total_falls}")

        # Identify bottleneck
        if path_effs and v_avgs and reg_gates:
            mean_pe = sum(path_effs) / len(path_effs)
            mean_v = sum(v_avgs) / len(v_avgs)
            speed_ratio = compute_speed_ratio(mean_v)
            mean_rg = sum(reg_gates) / len(reg_gates)

            bottleneck = "speed"
            bottleneck_val = speed_ratio
            if mean_pe ** 2 < bottleneck_val:
                bottleneck = "path_efficiency"
                bottleneck_val = mean_pe ** 2
            if mean_rg < bottleneck_val:
                bottleneck = "regression"
                bottleneck_val = mean_rg
            if total_falls > 0:
                bottleneck = "falls"

            print(f"\n  BOTTLENECK: {bottleneck}")

    errors = sum(1 for r in results.values() if r.get("error") and r["aggregate_ato"] is None)
    if errors:
        print(f"  Errors: {errors}/{n}")

    print(f"  Total wall time: {total_wall:.1f}s")
    print()

    # Write results JSON
    results_path = args.results_json or str(_FOREMAN / "tmp" / "ato_results.json")
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.time(),
        "n_runs": n,
        "targets_per_run": targets or 4,
        "genome": None,
        "seeds": seeds,
        "mean_ato": round(mean_ato, 1) if ato_scores else None,
        "median_ato": round(median_ato, 1) if ato_scores else None,
        "min_ato": round(min_ato, 1) if ato_scores else None,
        "max_ato": round(max_ato, 1) if ato_scores else None,
        "total_falls": total_falls,
        "errors": errors,
        "wall_time": round(total_wall, 1),
        "mean_path_efficiency": round(sum(path_effs) / len(path_effs), 3) if path_effs else None,
        "mean_v_avg": round(sum(v_avgs) / len(v_avgs), 3) if v_avgs else None,
        "mean_speed_ratio": round(compute_speed_ratio(sum(v_avgs) / len(v_avgs)), 3) if v_avgs else None,
        "mean_slip_efficiency": round(sum(slip_effs) / len(slip_effs), 3) if slip_effs else None,
        "mean_regression": round(sum(reg_gates) / len(reg_gates), 3) if reg_gates else None,
        "bottleneck": bottleneck if (path_effs and v_avgs and reg_gates) else None,
        "runs": all_results,
    }
    Path(results_path).write_text(json.dumps(summary, indent=2, default=str))
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()

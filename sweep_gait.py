#!/usr/bin/env python3
"""Sweep gait parameters through the real L5 pipeline via fast_test.py.

Tests each parameter combination by setting L5_GAIT_OVERRIDE env var
and running fast_test.py (which uses the full L5→L4→L3→L2→L1 stack).

Usage:
    python sweep_gait.py                    # Default grid
    python sweep_gait.py --episodes 2       # Fewer episodes per config
    python sweep_gait.py --quick            # Minimal grid for smoke test
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_FOREMAN = Path(__file__).resolve().parent
_WORKSPACE = _FOREMAN.parent

# Environment
_ENV = os.environ.copy()
_ENV["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"


def _make_override(freq: float, sl: float, sh: float) -> dict:
    """Build a gait override dict from freq, step_length, step_height."""
    return {
        "base_freq": freq,
        "min_freq": freq,
        "max_freq": freq,
        "turn_freq": freq,
        "max_step_length": sl,
        "step_length_scale": sl,
        "step_length_range": [0.05, sl],
        "trot_step_height": sh,
    }


def run_config(override: dict, n_runs: int, domain_start: int, targets: int,
               results_json: str = None) -> dict:
    """Run fast_test.py with the given gait override and return parsed results."""
    override_json = json.dumps(override)
    env = _ENV.copy()
    env["L5_GAIT_OVERRIDE"] = override_json

    # Use a unique results file per config to avoid race conditions
    if results_json is None:
        results_json = str(_FOREMAN / "tmp" / "sweep_run_result.json")

    cmd = [
        sys.executable,
        str(_FOREMAN / "fast_test.py"),
        "-n", str(n_runs),
        "--domain-start", str(domain_start),
        "--targets", str(targets),
        "--results-json", results_json,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(_FOREMAN),
        )
        return _parse_results_json(results_json, override)
    except subprocess.TimeoutExpired:
        return {"override": override, "error": "timeout", "ato": 0, "falls": 99,
                "reached": 0, "total": 0, "v_avg": 0.0}


def _parse_results_json(results_path: str, override: dict) -> dict:
    """Read fast_test.py's JSON output file and extract key metrics."""
    result = {
        "override": override,
        "ato": 0.0,
        "falls": 0,
        "reached": 0,
        "total": 0,
        "v_avg": 0.0,
    }

    try:
        with open(results_path) as f:
            data = json.load(f)
        result["ato"] = data.get("mean_ato") or 0.0
        result["falls"] = data.get("total_falls", 0)
        result["v_avg"] = data.get("mean_v_avg") or 0.0
        # Compute mean reached/total from individual runs
        runs = data.get("runs", [])
        if runs:
            result["reached"] = sum(r.get("reached", 0) for r in runs) / len(runs)
            result["total"] = data.get("targets_per_run", 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Sweep gait params through real pipeline")
    parser.add_argument("--episodes", type=int, default=4,
                        help="Parallel runs per config (default: 4)")
    parser.add_argument("--targets", type=int, default=4,
                        help="Targets per run (default: 4)")
    parser.add_argument("--domain-start", type=int, default=20,
                        help="Starting DDS domain (default: 20)")
    parser.add_argument("--quick", action="store_true",
                        help="Minimal grid for smoke test")
    parser.add_argument("--focused", action="store_true",
                        help="Focused grid (~10 configs near known-good)")
    args = parser.parse_args()

    if args.quick:
        # Smoke test: just 3 configs
        configs = [
            {},  # baseline (no override)
            {"base_freq": 2.5, "min_freq": 2.5, "max_freq": 2.5},
            {"base_freq": 2.0, "min_freq": 2.0, "max_freq": 2.0,
             "max_step_length": 0.40, "step_length_scale": 0.40,
             "step_length_range": [0.05, 0.40]},
        ]
    elif args.focused:
        # Focused grid: baseline + variations near known-good params
        # Baseline: 3.0Hz, sl=0.35, sh=0.10, kp=500
        # stride_speed = freq * step_length, must exceed body_speed (~0.88 m/s)
        configs = [
            {},  # baseline: 3.0Hz/0.35/0.10 → stride_speed=1.05
            # Higher step_length (more ground per step)
            _make_override(3.0, 0.40, 0.10),   # stride_speed=1.20
            _make_override(3.0, 0.45, 0.10),   # stride_speed=1.35
            # Lower step_height (less energy wasted lifting feet)
            _make_override(3.0, 0.35, 0.06),   # same stride, lower lift
            _make_override(3.0, 0.35, 0.08),   # same stride, medium lift
            _make_override(3.0, 0.40, 0.06),   # bigger stride, lower lift
            # Slightly higher freq
            _make_override(3.5, 0.35, 0.08),   # stride_speed=1.225
            _make_override(3.5, 0.35, 0.10),   # stride_speed=1.225
            # Lower freq, longer stride (keep stride_speed > 1.0)
            _make_override(2.5, 0.40, 0.08),   # stride_speed=1.00
            _make_override(2.5, 0.45, 0.08),   # stride_speed=1.125
        ]
    else:
        # Full grid: freq × step_length × step_height × kp
        freqs = [2.0, 2.5, 3.0, 3.5]
        step_lengths = [0.30, 0.35, 0.40]
        step_heights = [0.06, 0.08, 0.10]
        kps = [500]  # start with kp=500, add higher if safe

        configs = [{}]  # always include baseline
        for freq, sl, sh in itertools.product(freqs, step_lengths, step_heights):
            if freq == 3.0 and sl == 0.35 and sh == 0.10:
                continue  # skip baseline duplicate
            override = {
                "base_freq": freq,
                "min_freq": freq,
                "max_freq": freq,
                "turn_freq": freq,
                "max_step_length": sl,
                "step_length_scale": sl,
                "step_length_range": [0.05, sl],
                "trot_step_height": sh,
            }
            configs.append(override)

    print(f"Gait Parameter Sweep — Real Pipeline")
    print(f"  Configs: {len(configs)}")
    print(f"  Runs/config: {args.episodes}")
    print(f"  Targets/run: {args.targets}")
    print()

    results = []
    t0 = time.monotonic()

    for i, cfg in enumerate(configs):
        label = "BASELINE" if not cfg else (
            f"f={cfg.get('base_freq', 3.0):.1f} "
            f"sl={cfg.get('max_step_length', 0.35):.2f} "
            f"sh={cfg.get('trot_step_height', 0.10):.2f}"
        )
        print(f"[{i+1}/{len(configs)}] {label} ...", end=" ", flush=True)

        result = run_config(cfg, args.episodes, args.domain_start, args.targets)
        results.append(result)

        elapsed = time.monotonic() - t0
        print(f"ATO={result['ato']:.1f}  targets={result['reached']:.1f}/{result['total']}  "
              f"falls={result['falls']:.1f}  v_avg={result['v_avg']:.2f}  ({elapsed:.0f}s)")

    elapsed = time.monotonic() - t0
    print(f"\nSweep complete in {elapsed:.0f}s")

    # Rank by ATO among configs with 0 falls
    stable = [r for r in results if r["falls"] <= 0.5]
    stable.sort(key=lambda r: r["ato"], reverse=True)

    print(f"\n{'='*70}")
    print(f"Top configs (≤0.5 mean falls, ranked by ATO)")
    print(f"{'='*70}")
    for r in stable[:10]:
        cfg = r["override"]
        label = "BASELINE" if not cfg else (
            f"f={cfg.get('base_freq', 3.0):.1f} "
            f"sl={cfg.get('max_step_length', 0.35):.2f} "
            f"sh={cfg.get('trot_step_height', 0.10):.2f}"
        )
        print(f"  {label:40s} ATO={r['ato']:5.1f}  "
              f"targets={r['reached']:.1f}/{r['total']}  "
              f"falls={r['falls']:.1f}  v_avg={r['v_avg']:.2f}")

    # Save results
    out_path = _FOREMAN / "tmp" / "sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

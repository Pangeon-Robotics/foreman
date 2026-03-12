#!/usr/bin/env python3
"""Navigator parameter optimization for obstacle scenarios.

Tunes heading control + speed control parameters to minimize time-to-target
in scattered obstacle scenarios. Uses parallel headless runs with DDS isolation.

Usage:
    python nav_optimize.py                    # Run optimization (default 6 hours)
    python nav_optimize.py --hours 2          # Shorter run
    python nav_optimize.py --resume           # Resume from last checkpoint
    python nav_optimize.py --pop 12 -n 6      # 12 candidates, 6 seeds each
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

_FOREMAN = Path(__file__).resolve().parent
_WORKSPACE = _FOREMAN.parent
_CHECKPOINT = _FOREMAN / "tmp" / "nav_optimize_state.json"
_LOG = _FOREMAN / "tmp" / "nav_optimize.log"

# Navigator parameters to tune
PARAM_SPEC = {
    # name:          (default,   lo,    hi,  mutation_sigma)
    "WZ_ABS_MAX":    (0.6,      0.3,   1.0, 0.08),
    "VX_FLOOR":      (0.6,      0.3,   1.2, 0.10),
    "TURN_VX_LO":    (0.6,      0.3,   1.0, 0.08),
    "TURN_VX_HI":    (0.8,      0.5,   1.4, 0.10),
    "KP_HEADING":    (4.8,      1.5,   8.0, 0.50),
    "KD_HEADING":    (0.5,      0.1,   2.0, 0.15),
    "LOOKAHEAD":     (1.5,      0.5,   3.0, 0.20),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_individual(params=None):
    """Create an individual with given or default params."""
    if params is None:
        return {k: v[0] for k, v in PARAM_SPEC.items()}
    return dict(params)


def mutate(params, strength=1.0):
    """Gaussian mutation with bounds clamping."""
    child = dict(params)
    for k, (_, lo, hi, sigma) in PARAM_SPEC.items():
        child[k] = clamp(child[k] + random.gauss(0, sigma * strength), lo, hi)
    # Enforce TURN_VX_LO <= TURN_VX_HI
    if child["TURN_VX_LO"] > child["TURN_VX_HI"]:
        child["TURN_VX_LO"], child["TURN_VX_HI"] = child["TURN_VX_HI"], child["TURN_VX_LO"]
    return child


def crossover(a, b):
    """Uniform crossover."""
    child = {}
    for k in PARAM_SPEC:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    if child["TURN_VX_LO"] > child["TURN_VX_HI"]:
        child["TURN_VX_LO"], child["TURN_VX_HI"] = child["TURN_VX_HI"], child["TURN_VX_LO"]
    return child


def evaluate(params, seeds, domain_base=30):
    """Run headless obstacle scenario with given nav params, return fitness dict.

    Patches navigator_helper.py constants via monkey-patching before import.
    """
    # Write params to temp file for the worker script
    params_path = _FOREMAN / "tmp" / "nav_eval_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params))

    results = []
    for seed in seeds:
        domain = domain_base + seed
        cmd = [
            sys.executable, str(_FOREMAN / "nav_eval_worker.py"),
            "--seed", str(seed),
            "--domain", str(domain),
            "--params", str(params_path),
        ]
        env = os.environ.copy()
        env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120, cwd=str(_WORKSPACE), env=env)
            # Parse last line: "RESULT reached=X time=Y.Y falls=Z"
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("RESULT "):
                    parts = dict(p.split("=") for p in line[7:].split())
                    results.append({
                        "seed": seed,
                        "reached": int(parts.get("reached", "0")),
                        "time": float(parts.get("time", "30.0")),
                        "falls": int(parts.get("falls", "0")),
                    })
                    break
        except (subprocess.TimeoutExpired, Exception) as e:
            results.append({"seed": seed, "reached": 0, "time": 30.0, "falls": 1})

    if not results:
        return {"fitness": 0.0, "reached": 0, "avg_time": 30.0, "falls": 0}

    total_reached = sum(r["reached"] for r in results)
    total_falls = sum(r["falls"] for r in results)
    avg_time = sum(r["time"] for r in results) / len(results)

    # Fitness: reward reaching targets, penalize time and falls
    # 100 * reach_rate - 10 * fall_rate - avg_time
    reach_rate = total_reached / len(results)
    fall_rate = total_falls / len(results)
    fitness = 100 * reach_rate - 20 * fall_rate - avg_time

    return {
        "fitness": round(fitness, 2),
        "reached": total_reached,
        "avg_time": round(avg_time, 1),
        "falls": total_falls,
        "n_seeds": len(results),
        "reach_rate": round(reach_rate, 3),
    }


def save_checkpoint(state):
    _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT.write_text(json.dumps(state, indent=2))


def load_checkpoint():
    if _CHECKPOINT.exists():
        return json.loads(_CHECKPOINT.read_text())
    return None


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(_LOG, "a") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Navigator parameter optimization")
    parser.add_argument("--hours", type=float, default=6.0,
                        help="Maximum runtime in hours (default: 6)")
    parser.add_argument("--pop", type=int, default=8,
                        help="Population size (default: 8)")
    parser.add_argument("-n", type=int, default=6,
                        help="Seeds per evaluation (default: 6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed-offset", type=int, default=42,
                        help="Starting seed (default: 42)")
    args = parser.parse_args()

    deadline = time.monotonic() + args.hours * 3600
    seeds = list(range(args.seed_offset, args.seed_offset + args.n))

    # Initialize or resume population
    state = load_checkpoint() if args.resume else None
    if state:
        population = [make_individual(p) for p in state["population"]]
        fitnesses = state["fitnesses"]
        gen = state["generation"]
        best = state["best"]
        log(f"Resumed from gen {gen}, best fitness={best['fitness']}")
    else:
        # Seed population: default + random mutations
        population = [make_individual()]  # default params
        for _ in range(args.pop - 1):
            population.append(mutate(make_individual(), strength=1.5))
        fitnesses = [None] * args.pop
        gen = 0
        best = {"fitness": -999, "params": None}
        log(f"Starting optimization: pop={args.pop}, seeds={seeds}, "
            f"deadline={args.hours}h")

    # Evolution loop
    while time.monotonic() < deadline:
        gen += 1
        log(f"\n=== Generation {gen} ===")

        # Evaluate population
        for i, params in enumerate(population):
            if time.monotonic() >= deadline:
                break
            result = evaluate(params, seeds)
            fitnesses[i] = result
            status = (f"  [{i+1}/{len(population)}] fit={result['fitness']:.1f} "
                      f"reached={result['reached']}/{result['n_seeds']} "
                      f"time={result['avg_time']:.1f}s falls={result['falls']}")
            params_str = " ".join(f"{k}={v:.2f}" for k, v in params.items())
            log(f"{status}  {params_str}")

            if result["fitness"] > best["fitness"]:
                best = {"fitness": result["fitness"], "params": dict(params),
                        "result": result, "generation": gen}
                log(f"  *** NEW BEST: fitness={best['fitness']:.1f} ***")

        # Selection: rank-based, top half survives
        ranked = sorted(range(len(population)),
                        key=lambda i: (fitnesses[i] or {}).get("fitness", -999),
                        reverse=True)
        survivors = ranked[:len(population) // 2]

        # Reproduction
        new_pop = [dict(population[survivors[0]])]  # Elitism: keep best
        while len(new_pop) < len(population):
            if random.random() < 0.3:
                # Crossover
                a, b = random.sample(survivors, 2)
                child = crossover(population[a], population[b])
                child = mutate(child, strength=0.5)
            else:
                # Mutation from survivor
                parent = random.choice(survivors)
                child = mutate(population[parent], strength=1.0)
            new_pop.append(child)

        population = new_pop
        fitnesses = [None] * len(population)

        # Checkpoint
        save_checkpoint({
            "generation": gen,
            "population": population,
            "fitnesses": fitnesses,
            "best": best,
        })

        elapsed_h = (time.monotonic() - (deadline - args.hours * 3600)) / 3600
        log(f"Gen {gen} complete. Best fitness={best['fitness']:.1f} "
            f"({elapsed_h:.1f}h elapsed)")

    # Final report
    log(f"\n{'='*60}")
    log(f"OPTIMIZATION COMPLETE after {gen} generations")
    log(f"Best fitness: {best['fitness']:.1f}")
    log(f"Best params: {json.dumps(best.get('params', {}), indent=2)}")
    log(f"Best result: {json.dumps(best.get('result', {}), indent=2)}")

    # Write best params to file
    best_path = _FOREMAN / "tmp" / "nav_best_params.json"
    if best.get("params"):
        best_path.write_text(json.dumps(best["params"], indent=2))
        log(f"Best params written to {best_path}")


if __name__ == "__main__":
    main()

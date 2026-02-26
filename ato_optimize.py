#!/usr/bin/env python3
"""ATO Optimization Loop: iteratively tune gait parameters to maximize ATO score.

Runs fast_test.py with candidate gait parameters, analyzes results, and
proposes the next parameter set based on decision rules.

Usage:
    python ato_optimize.py                    # Run optimization loop
    python ato_optimize.py --max-iter 5       # Limit iterations
    python ato_optimize.py --target-ato 80    # Lower target
    python ato_optimize.py -n 6              # 6 parallel seeds per iteration
    python ato_optimize.py --resume           # Resume from last iteration
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

_FOREMAN = Path(__file__).resolve().parent

# V_REF for B2 — must match path_critic.py
V_REF = 2.0

# Parameter definitions: (initial, min, max, step)
PARAM_BOUNDS = {
    "STEP_LENGTH":     (0.30, 0.10, 0.70, 0.05),
    "GAIT_FREQ":       (1.5,  1.0,  6.0,  0.5),
    "STEP_HEIGHT":     (0.07, 0.03, 0.15, 0.01),
    "DUTY_CYCLE":      (0.65, 0.50, 0.80, 0.05),
    "TURN_WZ":         (1.0,  0.5,  2.0,  0.1),
    "THETA_THRESHOLD": (0.6,  0.3,  0.8,  0.05),
    "KP_YAW":          (2.0,  1.0,  5.0,  0.5),
    # Turn params (keep stable during speed ramp)
    "TURN_FREQ":       (3.0,  1.5,  5.0,  0.5),
    "TURN_STEP_HEIGHT": (0.08, 0.04, 0.12, 0.01),
    "TURN_DUTY_CYCLE": (0.55, 0.45, 0.65, 0.05),
    "TURN_STANCE_WIDTH": (0.12, 0.08, 0.20, 0.02),
}

# Speed ramp progression: (STEP_LENGTH, GAIT_FREQ, STEP_HEIGHT)
# Ratio STEP_LENGTH/GAIT_FREQ ~0.13-0.16, STEP_HEIGHT ~23% of STEP_LENGTH
SPEED_RAMP = [
    (0.40, 2.5, 0.09),
    (0.45, 3.0, 0.10),
    (0.50, 3.5, 0.11),
    (0.55, 4.0, 0.12),
    (0.60, 4.5, 0.13),
    (0.65, 5.0, 0.14),
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def theoretical_speed(step_length: float, gait_freq: float) -> float:
    """Estimate theoretical max forward speed from gait params."""
    return step_length * gait_freq


def make_genome_json(params: dict) -> dict:
    """Create a v12-format genome JSON from parameter dict."""
    return {
        "genome": {
            "GAIT_FREQ": params["GAIT_FREQ"],
            "STEP_LENGTH": params["STEP_LENGTH"],
            "STEP_HEIGHT": params["STEP_HEIGHT"],
            "DUTY_CYCLE": params["DUTY_CYCLE"],
            "STANCE_WIDTH": params.get("STANCE_WIDTH", 0.0),
            "KP_YAW": params["KP_YAW"],
            "WZ_LIMIT": params.get("WZ_LIMIT", 1.5),
            "TURN_FREQ": params["TURN_FREQ"],
            "TURN_STEP_HEIGHT": params["TURN_STEP_HEIGHT"],
            "TURN_DUTY_CYCLE": params["TURN_DUTY_CYCLE"],
            "TURN_STANCE_WIDTH": params["TURN_STANCE_WIDTH"],
            "TURN_WZ": params["TURN_WZ"],
            "THETA_THRESHOLD": params["THETA_THRESHOLD"],
        },
        "generation": "ato_opt",
        "fitness": 0.0,
    }


def get_initial_params() -> dict:
    """Return the initial parameter set (current B2 defaults)."""
    return {name: bounds[0] for name, bounds in PARAM_BOUNDS.items()}


def run_evaluation(params: dict, iteration: int, n_runs: int, targets: int,
                   seed_offset: int = 0) -> dict:
    """Run fast_test.py with candidate params and return parsed results."""
    # Write candidate genome
    candidates_dir = _FOREMAN / "tmp" / "ato_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    genome_path = candidates_dir / f"iter_{iteration:03d}.json"
    genome_path.write_text(json.dumps(make_genome_json(params), indent=2))

    results_path = _FOREMAN / "tmp" / "ato_results.json"

    cmd = [
        sys.executable,
        str(_FOREMAN / "fast_test.py"),
        "-n", str(n_runs),
        "--targets", str(targets),
        "--config", str(genome_path),
        "--results-json", str(results_path),
        "--seed-offset", str(seed_offset),
    ]

    print(f"\n{'='*72}")
    print(f"ITERATION {iteration}: running {n_runs} seeds, {targets} targets each")
    v_theory = theoretical_speed(params["STEP_LENGTH"], params["GAIT_FREQ"])
    print(f"  v_theory = {params['STEP_LENGTH']:.2f} × {params['GAIT_FREQ']:.1f} = {v_theory:.2f} m/s")
    print(f"  params: {json.dumps({k: round(v, 3) for k, v in params.items()}, indent=None)}")
    print(f"{'='*72}")

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True,
        timeout=600,  # 10 min max per iteration
        cwd=str(_FOREMAN),
    )
    wall_time = time.monotonic() - t0

    # Parse results
    if results_path.exists():
        results = json.loads(results_path.read_text())
        results["iteration"] = iteration
        results["params"] = params
        results["v_theory"] = round(v_theory, 2)
        results["eval_wall_time"] = round(wall_time, 1)
        return results

    return {
        "iteration": iteration,
        "params": params,
        "mean_ato": None,
        "total_falls": 0,
        "error": f"fast_test.py exited with code {proc.returncode}",
    }


def analyze_and_propose(current_params: dict, results: dict,
                        prev_params: dict | None,
                        prev_results: dict | None,
                        ramp_index: int,
                        step_scale: float = 1.0) -> tuple[dict, int, str]:
    """Analyze results and propose next parameters.

    Returns (next_params, next_ramp_index, decision_reason).
    """
    params = copy.deepcopy(current_params)
    mean_ato = results.get("mean_ato") or 0.0
    falls = results.get("total_falls", 0)
    n_runs = results.get("n_runs", 1)
    pe = results.get("mean_path_efficiency") or 1.0
    v_avg = results.get("mean_v_avg") or 0.0
    speed_ratio = results.get("mean_speed_ratio") or 0.0
    rg = results.get("mean_regression_gate") or 1.0
    bottleneck = results.get("bottleneck", "speed")

    fall_rate = falls / max(n_runs, 1)

    # Safety: too many falls — revert and stabilize
    if fall_rate > 0.25:
        if prev_params is not None:
            reason = f"REVERT: {falls} falls ({fall_rate:.0%}), reverting to previous params"
            reverted = copy.deepcopy(prev_params)
            # Try to stabilize: increase duty cycle or step height
            reverted["DUTY_CYCLE"] = clamp(
                reverted["DUTY_CYCLE"] + 0.05 * step_scale,
                *PARAM_BOUNDS["DUTY_CYCLE"][1:3])
            reverted["STEP_HEIGHT"] = clamp(
                reverted["STEP_HEIGHT"] + 0.01 * step_scale,
                *PARAM_BOUNDS["STEP_HEIGHT"][1:3])
            return reverted, max(ramp_index - 1, 0), reason
        else:
            reason = f"STABILIZE: {falls} falls on initial params, increasing duty cycle"
            params["DUTY_CYCLE"] = clamp(
                params["DUTY_CYCLE"] + 0.05,
                *PARAM_BOUNDS["DUTY_CYCLE"][1:3])
            return params, ramp_index, reason

    # Some falls but not catastrophic — try stability tweaks without reverting speed
    if falls > 0:
        reason = f"STABILITY: {falls} falls, increasing step_height and duty_cycle"
        params["STEP_HEIGHT"] = clamp(
            params["STEP_HEIGHT"] + 0.01 * step_scale,
            *PARAM_BOUNDS["STEP_HEIGHT"][1:3])
        params["DUTY_CYCLE"] = clamp(
            params["DUTY_CYCLE"] + 0.05 * step_scale,
            *PARAM_BOUNDS["DUTY_CYCLE"][1:3])
        return params, ramp_index, reason

    # No falls — decide based on bottleneck
    if bottleneck == "speed" or speed_ratio < 0.8:
        # Speed is the bottleneck — advance the speed ramp
        if ramp_index < len(SPEED_RAMP):
            sl, gf, sh = SPEED_RAMP[ramp_index]
            # Safety: never increase >30% in one step
            max_sl = current_params["STEP_LENGTH"] * 1.3
            max_gf = current_params["GAIT_FREQ"] * 1.3
            sl = min(sl, max_sl)
            gf = min(gf, max_gf)
            params["STEP_LENGTH"] = clamp(sl, *PARAM_BOUNDS["STEP_LENGTH"][1:3])
            params["GAIT_FREQ"] = clamp(gf, *PARAM_BOUNDS["GAIT_FREQ"][1:3])
            params["STEP_HEIGHT"] = clamp(sh, *PARAM_BOUNDS["STEP_HEIGHT"][1:3])
            v_new = theoretical_speed(params["STEP_LENGTH"], params["GAIT_FREQ"])
            reason = f"SPEED RAMP [{ramp_index}]: step={params['STEP_LENGTH']:.2f}, freq={params['GAIT_FREQ']:.1f}, v_theory={v_new:.2f}"
            return params, ramp_index + 1, reason
        else:
            # Ramp exhausted — try finer increments
            params["STEP_LENGTH"] = clamp(
                params["STEP_LENGTH"] + 0.03 * step_scale,
                *PARAM_BOUNDS["STEP_LENGTH"][1:3])
            params["GAIT_FREQ"] = clamp(
                params["GAIT_FREQ"] + 0.3 * step_scale,
                *PARAM_BOUNDS["GAIT_FREQ"][1:3])
            reason = f"FINE SPEED: step+={0.03*step_scale:.2f}, freq+={0.3*step_scale:.1f}"
            return params, ramp_index, reason

    if bottleneck == "regression" or rg < 0.90:
        reason = f"REGRESSION FIX: r_gate={rg:.2f}, increasing TURN_WZ, decreasing THETA_THRESHOLD"
        params["TURN_WZ"] = clamp(
            params["TURN_WZ"] + 0.1 * step_scale,
            *PARAM_BOUNDS["TURN_WZ"][1:3])
        params["THETA_THRESHOLD"] = clamp(
            params["THETA_THRESHOLD"] - 0.05 * step_scale,
            *PARAM_BOUNDS["THETA_THRESHOLD"][1:3])
        return params, ramp_index, reason

    if bottleneck == "path_efficiency" or pe < 0.85:
        reason = f"PATH FIX: pe={pe:.2f}, increasing KP_YAW for tighter tracking"
        params["KP_YAW"] = clamp(
            params["KP_YAW"] + 0.5 * step_scale,
            *PARAM_BOUNDS["KP_YAW"][1:3])
        return params, ramp_index, reason

    # All components reasonable — still push speed if below target
    if speed_ratio < 1.0 and ramp_index < len(SPEED_RAMP):
        sl, gf, sh = SPEED_RAMP[ramp_index]
        max_sl = current_params["STEP_LENGTH"] * 1.3
        max_gf = current_params["GAIT_FREQ"] * 1.3
        sl = min(sl, max_sl)
        gf = min(gf, max_gf)
        params["STEP_LENGTH"] = clamp(sl, *PARAM_BOUNDS["STEP_LENGTH"][1:3])
        params["GAIT_FREQ"] = clamp(gf, *PARAM_BOUNDS["GAIT_FREQ"][1:3])
        params["STEP_HEIGHT"] = clamp(sh, *PARAM_BOUNDS["STEP_HEIGHT"][1:3])
        v_new = theoretical_speed(params["STEP_LENGTH"], params["GAIT_FREQ"])
        reason = f"BALANCED RAMP [{ramp_index}]: all components OK, pushing speed to v_theory={v_new:.2f}"
        return params, ramp_index + 1, reason

    # Plateau — try orthogonal axis
    reason = "PLATEAU: all components saturated, trying KP_YAW and TURN_WZ nudge"
    params["KP_YAW"] = clamp(
        params["KP_YAW"] + 0.5 * step_scale,
        *PARAM_BOUNDS["KP_YAW"][1:3])
    params["TURN_WZ"] = clamp(
        params["TURN_WZ"] + 0.1 * step_scale,
        *PARAM_BOUNDS["TURN_WZ"][1:3])
    return params, ramp_index, reason


def main():
    parser = argparse.ArgumentParser(description="ATO Optimization Loop")
    parser.add_argument("--max-iter", type=int, default=15,
                        help="Maximum iterations (default: 15)")
    parser.add_argument("--target-ato", type=float, default=90.0,
                        help="Target mean ATO score (default: 90)")
    parser.add_argument("-n", type=int, default=6,
                        help="Parallel seeds per iteration (default: 6)")
    parser.add_argument("--targets", type=int, default=3,
                        help="Targets per run (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last logged iteration")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset for reproducibility")
    args = parser.parse_args()

    log_dir = _FOREMAN / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ato_log.jsonl"

    # Initialize or resume
    params = get_initial_params()
    ramp_index = 0
    start_iter = 1
    prev_params = None
    prev_results = None
    step_scale = 1.0
    consecutive_fall_iters = 0
    plateau_count = 0
    best_ato = 0.0
    best_params = None

    if args.resume and log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        if lines:
            last = json.loads(lines[-1])
            start_iter = last["iteration"] + 1
            params = last.get("next_params", last["params"])
            ramp_index = last.get("ramp_index", 0)
            best_ato = last.get("best_ato", 0.0)
            best_params = last.get("best_params")
            step_scale = last.get("step_scale", 1.0)
            print(f"Resuming from iteration {start_iter}, best ATO so far: {best_ato:.1f}")

    print(f"\n{'#'*72}")
    print(f"# ATO OPTIMIZATION LOOP")
    print(f"# Target: mean ATO > {args.target_ato}")
    print(f"# Max iterations: {args.max_iter}")
    print(f"# Seeds per iteration: {args.n}")
    print(f"# Targets per run: {args.targets}")
    print(f"{'#'*72}\n")

    for iteration in range(start_iter, start_iter + args.max_iter):
        # Run evaluation
        results = run_evaluation(
            params, iteration, args.n, args.targets, args.seed_offset)

        mean_ato = results.get("mean_ato") or 0.0
        falls = results.get("total_falls", 0)

        # Track best
        if mean_ato > best_ato and falls == 0:
            best_ato = mean_ato
            best_params = copy.deepcopy(params)

        # Check convergence
        if mean_ato >= args.target_ato and falls == 0:
            print(f"\n{'*'*72}")
            print(f"* CONVERGED at iteration {iteration}!")
            print(f"* Mean ATO: {mean_ato:.1f} >= {args.target_ato}")
            print(f"* Falls: 0")
            print(f"* Params: {json.dumps({k: round(v, 3) for k, v in params.items()})}")
            print(f"{'*'*72}\n")
            _log_iteration(log_path, iteration, params, results, "CONVERGED",
                           ramp_index, step_scale, best_ato, best_params, None)
            break

        # Track consecutive fall iterations for step_scale reduction
        if falls > 0:
            consecutive_fall_iters += 1
            if consecutive_fall_iters >= 2:
                step_scale *= 0.5
                step_scale = max(step_scale, 0.25)
                print(f"  [safety] 2 consecutive fall iterations, step_scale halved to {step_scale}")
                consecutive_fall_iters = 0
        else:
            consecutive_fall_iters = 0

        # Check plateau
        if prev_results and prev_results.get("mean_ato"):
            improvement = mean_ato - prev_results["mean_ato"]
            if abs(improvement) < 2.0 and falls == 0:
                plateau_count += 1
            else:
                plateau_count = 0

        if plateau_count >= 3:
            print(f"  [plateau] 3 iterations with <2 ATO improvement, trying orthogonal axis")
            plateau_count = 0

        # Propose next parameters
        next_params, ramp_index, reason = analyze_and_propose(
            params, results, prev_params, prev_results,
            ramp_index, step_scale)

        print(f"\n  DECISION: {reason}")

        # Log
        _log_iteration(log_path, iteration, params, results, reason,
                       ramp_index, step_scale, best_ato, best_params, next_params)

        # Advance
        prev_params = copy.deepcopy(params)
        prev_results = results
        params = next_params

    else:
        # Loop ended without converging
        print(f"\n{'='*72}")
        print(f"OPTIMIZATION COMPLETE (did not reach target ATO {args.target_ato})")
        print(f"Best ATO: {best_ato:.1f}")
        if best_params:
            print(f"Best params: {json.dumps({k: round(v, 3) for k, v in best_params.items()})}")
        print(f"{'='*72}\n")

    # Write final best params
    if best_params:
        final_path = _FOREMAN / "tmp" / "ato_best_params.json"
        final_path.write_text(json.dumps({
            "genome": best_params,
            "best_ato": round(best_ato, 1),
            "v_ref": V_REF,
        }, indent=2))
        print(f"Best params written to {final_path}")


def _log_iteration(log_path: Path, iteration: int, params: dict,
                   results: dict, reason: str, ramp_index: int,
                   step_scale: float, best_ato: float,
                   best_params: dict | None, next_params: dict | None):
    """Append one iteration record to the JSONL log."""
    entry = {
        "iteration": iteration,
        "timestamp": time.time(),
        "params": {k: round(v, 4) for k, v in params.items()},
        "v_theory": round(theoretical_speed(params["STEP_LENGTH"], params["GAIT_FREQ"]), 2),
        "mean_ato": results.get("mean_ato"),
        "median_ato": results.get("median_ato"),
        "min_ato": results.get("min_ato"),
        "max_ato": results.get("max_ato"),
        "total_falls": results.get("total_falls", 0),
        "mean_path_efficiency": results.get("mean_path_efficiency"),
        "mean_v_avg": results.get("mean_v_avg"),
        "mean_speed_ratio": results.get("mean_speed_ratio"),
        "mean_regression_gate": results.get("mean_regression_gate"),
        "bottleneck": results.get("bottleneck"),
        "decision": reason,
        "ramp_index": ramp_index,
        "step_scale": step_scale,
        "best_ato": round(best_ato, 1),
        "best_params": {k: round(v, 4) for k, v in best_params.items()} if best_params else None,
        "next_params": {k: round(v, 4) for k, v in next_params.items()} if next_params else None,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()

"""Automated critics for scenario evaluation.

Each critic examines a GameRunResult (stats + telemetry + SLAM data)
and returns a CriticResult with pass/fail, score, and diagnostic details.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CriticResult:
    """Result from a single critic evaluation."""
    name: str
    passed: bool
    score: float
    details: str


def run_all_critics(result, criteria: dict, telemetry_path: Path | None) -> list[CriticResult]:
    """Run all applicable critics and return results.

    Parameters
    ----------
    result : GameRunResult
        Output from run_game().
    criteria : dict
        Pass thresholds from ScenarioDefinition.pass_criteria.
    telemetry_path : Path or None
        Path to JSONL telemetry file.
    """
    results = []

    # Parse telemetry once for all critics that need it
    telem = _parse_telemetry(telemetry_path) if telemetry_path else {}

    # Universal critics (run on every scenario)
    results.append(target_success(result, criteria))
    results.append(no_falls(result, criteria))

    if result.slam_trail and result.truth_trail:
        results.append(slam_accuracy(result, criteria))

    # DWA-dependent critics (only if DWA telemetry exists)
    if "dwa" in telem:
        results.append(forward_progress(result, telem, criteria))
        results.append(dwa_feasibility(telem, criteria))
        results.append(dwa_oscillation(result, telem, criteria))
        results.append(collision_free(telem, criteria))

    # Ground-truth proximity critic (foreman referee using physics engine)
    if "proximity" in telem:
        results.append(obstacle_proximity(telem, criteria))

    return results


def _parse_telemetry(path: Path | None) -> dict[str, list[dict]]:
    """Parse JSONL telemetry into {module: [records]}."""
    if path is None or not path.exists():
        return {}
    records: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                mod = rec.get("module", "unknown")
                records.setdefault(mod, []).append(rec)
            except json.JSONDecodeError:
                continue
    return records


def target_success(result, criteria: dict) -> CriticResult:
    """Check if enough targets were reached."""
    threshold = criteria.get("target_success_rate", 1.0)
    rate = result.stats.success_rate
    passed = rate >= threshold
    return CriticResult(
        name="target_success",
        passed=passed,
        score=rate,
        details=(f"{result.stats.targets_reached}/{result.stats.targets_spawned} "
                 f"reached ({rate:.0%}), need {threshold:.0%}"),
    )


def no_falls(result, criteria: dict) -> CriticResult:
    """Check robot stayed upright."""
    max_falls = criteria.get("max_falls", 0)
    falls = result.stats.falls
    passed = falls <= max_falls
    return CriticResult(
        name="no_falls",
        passed=passed,
        score=0.0 if falls > 0 else 1.0,
        details=f"{falls} falls (max allowed: {max_falls})",
    )


def slam_accuracy(result, criteria: dict) -> CriticResult:
    """Check SLAM drift against thresholds."""
    max_mean = criteria.get("max_slam_drift_mean", 0.5)
    max_final = criteria.get("max_slam_drift_final", 1.0)

    if not result.slam_trail or not result.truth_trail:
        return CriticResult("slam_accuracy", True, 1.0, "No SLAM data (skipped)")

    n = min(len(result.slam_trail), len(result.truth_trail))
    drifts = [
        math.sqrt((result.slam_trail[i][0] - result.truth_trail[i][0])**2 +
                   (result.slam_trail[i][1] - result.truth_trail[i][1])**2)
        for i in range(n)
    ]

    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    final_drift = drifts[-1] if drifts else 0.0

    passed = mean_drift < max_mean and final_drift < max_final
    return CriticResult(
        name="slam_accuracy",
        passed=passed,
        score=max(0.0, 1.0 - mean_drift / max_mean),
        details=(f"mean={mean_drift:.3f}m (max {max_mean}), "
                 f"final={final_drift:.3f}m (max {max_final})"),
    )


def forward_progress(result, telem: dict, criteria: dict) -> CriticResult:
    """Check robot generally moved toward targets (monotonic distance decrease).

    Uses DWA telemetry distance-to-target readings. Allows small increases
    (0.1m tolerance) for obstacle detours. Measures percentage of 10Hz
    windows where distance decreased or held steady.
    """
    threshold = criteria.get("forward_progress_pct", 0.80)

    dwa_records = telem.get("dwa", [])
    dists = [r["dist"] for r in dwa_records if "dist" in r]
    if len(dists) < 2:
        return CriticResult("forward_progress", True, 1.0, "Not enough data (skipped)")

    # Count windows where distance decreased (with 0.1m tolerance for detours)
    windows_good = sum(1 for i in range(1, len(dists)) if dists[i] <= dists[i-1] + 0.1)
    total_windows = len(dists) - 1
    progress_pct = windows_good / total_windows if total_windows > 0 else 1.0

    passed = progress_pct >= threshold
    return CriticResult(
        name="forward_progress",
        passed=passed,
        score=progress_pct,
        details=(f"{progress_pct:.0%} of windows showed progress "
                 f"({windows_good}/{total_windows}), need {threshold:.0%}"),
    )


def dwa_feasibility(telem: dict, criteria: dict) -> CriticResult:
    """Check DWA had enough feasible arcs on average.

    Low feasibility means the costmap is so cluttered that DWA can't
    find paths, likely causing the robot to stand still.
    """
    min_mean = criteria.get("min_dwa_feasible_mean", 20)

    dwa_records = telem.get("dwa", [])
    if not dwa_records:
        return CriticResult("dwa_feasibility", True, 1.0, "No DWA data (skipped)")

    feasible_counts = [r.get("n_feasible", 0) for r in dwa_records]
    mean_feasible = sum(feasible_counts) / len(feasible_counts)

    passed = mean_feasible >= min_mean
    return CriticResult(
        name="dwa_feasibility",
        passed=passed,
        score=min(1.0, mean_feasible / 105.0),  # 105 arcs total in DWA grid
        details=f"mean {mean_feasible:.0f} feasible arcs (need {min_mean}+, max 105)",
    )


def dwa_oscillation(result, telem: dict, criteria: dict) -> CriticResult:
    """Check robot isn't flip-flopping between left/right turns.

    Counts significant turn direction reversals (ignoring small turns
    below 0.1 rad/s) and normalizes per target reached.
    """
    max_per_target = criteria.get("max_dwa_oscillation_per_target", 8)

    dwa_records = telem.get("dwa", [])
    if len(dwa_records) < 3:
        return CriticResult("dwa_oscillation", True, 1.0, "Not enough data (skipped)")

    turns = [r.get("turn", 0) for r in dwa_records]
    sign_changes = 0
    for i in range(1, len(turns)):
        # Only count reversals where both turns are significant
        if (turns[i] * turns[i-1] < 0
                and abs(turns[i]) > 0.1
                and abs(turns[i-1]) > 0.1):
            sign_changes += 1

    targets = max(1, result.stats.targets_reached)
    per_target = sign_changes / targets
    max_total = max_per_target * targets

    passed = sign_changes <= max_total
    return CriticResult(
        name="dwa_oscillation",
        passed=passed,
        score=max(0.0, 1.0 - per_target / (max_per_target * 2)),
        details=(f"{sign_changes} turn reversals ({per_target:.1f}/target, "
                 f"max {max_per_target}/target)"),
    )


def collision_free(telem: dict, criteria: dict) -> CriticResult:
    """Check no long sequences of DWA emergency stops.

    An emergency stop is when n_feasible == 0 (DWA found no valid arcs).
    Short single-step zeros are normal (costmap stale), but consecutive
    zeros mean the robot is stuck.
    """
    max_consecutive = criteria.get("max_consecutive_estops", 2)

    dwa_records = telem.get("dwa", [])
    if not dwa_records:
        return CriticResult("collision_free", True, 1.0, "No DWA data (skipped)")

    feasible_counts = [r.get("n_feasible", 0) for r in dwa_records]

    # Find longest run of n_feasible == 0
    max_run = 0
    current_run = 0
    for f in feasible_counts:
        if f == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    total_estops = sum(1 for f in feasible_counts if f == 0)
    passed = max_run <= max_consecutive
    return CriticResult(
        name="collision_free",
        passed=passed,
        score=max(0.0, 1.0 - max_run / (max_consecutive * 3)),
        details=(f"max {max_run} consecutive e-stops (limit {max_consecutive}), "
                 f"{total_estops} total out of {len(feasible_counts)} samples"),
    )


def obstacle_proximity(telem: dict, criteria: dict) -> CriticResult:
    """Check robot maintained minimum clearance from obstacles.

    FOREMAN REFEREE — uses MuJoCo ground-truth body positions, not robot
    sensors. The robot's control loop uses only LiDAR/costmap/DWA for
    obstacle avoidance; this critic validates the outcome using the
    physics engine's omniscient view.

    Measures center-to-center distance from robot base to each obstacle
    body. Since obstacle radii are 0.2-0.3m and the robot half-width is
    ~0.25m, a center distance below ~0.5m indicates contact.
    """
    min_clearance_threshold = criteria.get("min_obstacle_clearance", 0.5)

    prox_records = telem.get("proximity", [])
    if not prox_records:
        return CriticResult("obstacle_proximity", True, 1.0, "No proximity data (skipped)")

    clearances = [r["min_clearance"] for r in prox_records]
    min_seen = min(clearances)
    mean_clearance = sum(clearances) / len(clearances)
    # Count how many samples were below threshold (potential collisions)
    violations = sum(1 for c in clearances if c < min_clearance_threshold)

    passed = min_seen >= min_clearance_threshold
    score = min(1.0, min_seen / min_clearance_threshold) if min_clearance_threshold > 0 else 1.0
    return CriticResult(
        name="obstacle_proximity",
        passed=passed,
        score=score,
        details=(f"min clearance {min_seen:.2f}m (threshold {min_clearance_threshold:.1f}m), "
                 f"mean {mean_clearance:.2f}m, "
                 f"{violations}/{len(clearances)} samples below threshold"),
    )

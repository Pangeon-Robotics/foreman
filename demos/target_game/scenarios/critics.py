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
    """Run all applicable critics and return results."""
    results = []
    telem = _parse_telemetry(telemetry_path) if telemetry_path else {}

    # Universal critics
    results.append(target_success(result, criteria))
    results.append(no_falls(result, criteria))

    if result.slam_trail and result.truth_trail:
        results.append(slam_accuracy(result, criteria))

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


def obstacle_proximity(telem: dict, criteria: dict) -> CriticResult:
    """Check robot maintained minimum clearance from obstacles.

    FOREMAN REFEREE — uses MuJoCo ground-truth body positions, not robot
    sensors. The robot's control loop uses only LiDAR/costmap for
    obstacle avoidance; this critic validates the outcome using the
    physics engine's omniscient view.
    """
    min_clearance_threshold = criteria.get("min_obstacle_clearance", 0.5)

    prox_records = telem.get("proximity", [])
    if not prox_records:
        return CriticResult("obstacle_proximity", True, 1.0, "No proximity data (skipped)")

    clearances = [r["min_clearance"] for r in prox_records]
    min_seen = min(clearances)
    mean_clearance = sum(clearances) / len(clearances)
    violations = sum(1 for c in clearances if c < min_clearance_threshold)

    passed = min_seen >= min_clearance_threshold
    score = min(1.0, min_seen / min_clearance_threshold) if min_clearance_threshold > 0 else 1.0
    return CriticResult(
        name="obstacle_proximity",
        passed=passed,
        score=score,
        details=(f"min clearance {min_seen:.2f}m (threshold {min_clearance_threshold:.2f}m), "
                 f"mean {mean_clearance:.2f}m, "
                 f"{violations}/{len(clearances)} samples below threshold"),
    )

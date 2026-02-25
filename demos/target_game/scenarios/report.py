"""Scenario test report generation.

Prints a summary table of all scenario results with per-critic details
for failures.
"""
from __future__ import annotations

import math


def print_report(results: dict) -> None:
    """Print formatted summary table of scenario results.

    Parameters
    ----------
    results : dict[str, ScenarioResult]
        Output from ScenarioRunner.run_all().
    """
    print(f"\n{'='*80}")
    print("SCENARIO TEST REPORT")
    print(f"{'='*80}\n")

    # Header
    header = (f"{'Scenario':<15} {'Targets':>8} {'Success':>8} {'Falls':>6} "
              f"{'ATO':>6} {'SLAM Drift':>11} {'DWA Feas':>9} {'Critics':>10} {'Result':>8}")
    print(header)
    print("-" * len(header))

    for name, result in results.items():
        if result.error:
            print(f"{name:<15} {'ERROR':>8}  {result.error[:50]}")
            continue

        gr = result.game_result
        stats = gr.stats

        targets_str = f"{stats.targets_reached}/{stats.targets_spawned}"
        success_str = f"{stats.success_rate:.0%}"
        falls_str = str(stats.falls)

        # ATO fitness
        ato_str = "n/a"
        if gr.ato_score is not None:
            ato_str = f"{gr.ato_score:.1f}"

        # SLAM drift
        slam_str = "n/a"
        if gr.slam_trail and gr.truth_trail:
            n = min(len(gr.slam_trail), len(gr.truth_trail))
            drifts = [
                math.sqrt((gr.slam_trail[i][0] - gr.truth_trail[i][0])**2 +
                           (gr.slam_trail[i][1] - gr.truth_trail[i][1])**2)
                for i in range(n)
            ]
            if drifts:
                slam_str = f"{sum(drifts)/len(drifts):.2f}m"

        # DWA feasibility (from critic results)
        dwa_str = "n/a"
        for c in result.critic_results:
            if c.name == "dwa_feasibility":
                dwa_str = f"{c.score * 105:.0f}/105"
                break

        # Critics summary
        n_passed = sum(1 for c in result.critic_results if c.passed)
        n_total = len(result.critic_results)
        critics_str = f"{n_passed}/{n_total}"

        all_passed = n_passed == n_total
        result_str = "PASS" if all_passed else "FAIL"

        print(f"{name:<15} {targets_str:>8} {success_str:>8} {falls_str:>6} "
              f"{ato_str:>6} {slam_str:>11} {dwa_str:>9} {critics_str:>10} {result_str:>8}")

    # Detailed failures
    failures = [
        (name, r) for name, r in results.items()
        if r.error or (r.critic_results and not all(c.passed for c in r.critic_results))
    ]

    if failures:
        print(f"\n{'='*80}")
        print("FAILURE DETAILS")
        print(f"{'='*80}")

        for name, result in failures:
            print(f"\n--- {name} ---")
            if result.error:
                print(f"  ERROR: {result.error}")
                continue
            for critic in result.critic_results:
                if not critic.passed:
                    print(f"  FAIL {critic.name}: {critic.details}")

    # Overall summary
    total = len(results)
    passed = sum(
        1 for r in results.values()
        if not r.error and all(c.passed for c in r.critic_results)
    )
    errored = sum(1 for r in results.values() if r.error)
    failed = total - passed - errored

    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} scenarios passed", end="")
    if errored:
        print(f", {errored} errors", end="")
    if failed:
        print(f", {failed} failed", end="")
    print()
    print(f"{'='*80}")

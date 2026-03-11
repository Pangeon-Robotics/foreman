#!/usr/bin/env python3
"""Run an obstacle avoidance scenario (headed by default).

Usage:
    python run_scenario.py                      # list available scenarios
    python run_scenario.py open                 # run open field (headed)
    python run_scenario.py corridor             # run corridor scenario
    python run_scenario.py dead_end --headless  # headless, max speed
    python run_scenario.py all                  # run all scenarios sequentially
"""
import argparse
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent  # robotics/

SCENARIOS = ["open", "scattered", "corridor", "L_wall", "dead_end", "dense", "test"]


def main():
    parser = argparse.ArgumentParser(
        description="Run obstacle avoidance scenarios with MuJoCo viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Scenarios (easiest to hardest):
  open       - Empty field, baseline navigation (no obstacles)
  scattered  - 6 obstacles, clear paths between
  corridor   - Parallel walls, 1.5m gap
  L_wall     - L-shaped wall, must navigate around
  dead_end   - U-shaped obstacle, must find detour
  dense      - 14 obstacles, narrow gaps (~0.8m)
  all        - Run every scenario sequentially
  test       - Feedback loop test (5 seeds, headless, all metrics)
""",
    )
    parser.add_argument("scenario", nargs="?", default=None,
                        help="Scenario name (or 'all')")
    parser.add_argument("--robot", default="b2",
                        help="Robot model (default: b2)")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless at max speed (default: headed)")
    parser.add_argument("--viewer", action="store_true",
                        help="Start TCP debug server for Godot TSDF viewer")
    parser.add_argument("--domain", type=int, default=2,
                        help="DDS domain ID (default: 2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override target seed (default: use scenario's seed)")
    parser.add_argument("--targets", type=int, default=None,
                        help="Override number of targets (default: use scenario's count)")
    parser.add_argument("--random-obstacles", action="store_true",
                        help="Randomize obstacle positions (uses target seed)")
    parser.add_argument("--god", action="store_true",
                        help="Show god-view costmap overlay in MuJoCo viewer")
    args = parser.parse_args()

    if args.scenario is None:
        print("Available scenarios:")
        for s in SCENARIOS:
            print(f"  {s}")
        print(f"\nUsage: python {Path(__file__).name} <scenario>")
        print(f"       python {Path(__file__).name} all")
        sys.exit(0)

    if args.scenario != "all" and args.scenario not in SCENARIOS:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {', '.join(SCENARIOS)}, all")
        sys.exit(1)

    # Feedback loop test shortcut
    if args.scenario == "test":
        cmd = [
            sys.executable, "-m",
            "foreman.demos.target_game.test_feedback_loop",
            "--robot", args.robot,
            "--domain", str(args.domain),
        ]
        if args.seed is not None:
            cmd.extend(["--seed-start", str(args.seed)])
        if args.targets is not None:
            cmd.extend(["--seeds", str(args.targets)])
        if args.random_obstacles:
            cmd.append("--random-obstacles")
        sys.exit(subprocess.call(cmd, cwd=str(_WORKSPACE)))

    # Build command
    cmd = [
        sys.executable, "-m", "foreman.demos.target_game.scenarios",
        "--robot", args.robot,
        "--domain", str(args.domain),
    ]

    if not args.headless:
        cmd.append("--headed")

    if args.viewer:
        cmd.append("--viewer")

    if args.random_obstacles:
        cmd.append("--random-obstacles")

    if args.god:
        cmd.append("--god")

    if args.scenario != "all":
        cmd.extend(["--scenario", args.scenario])

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    if args.targets is not None:
        cmd.extend(["--targets", str(args.targets)])

    sys.exit(subprocess.call(cmd, cwd=str(_WORKSPACE)))


if __name__ == "__main__":
    main()

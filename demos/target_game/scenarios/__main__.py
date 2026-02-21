"""Entry point: python -m foreman.demos.target_game.scenarios"""
from __future__ import annotations

# CycloneDDS 0.10.4 preload: must happen before any DDS imports.
# Same pattern as foreman/demos/target_game/__main__.py.
import ctypes as _ctypes
import os as _os
_SYS_DDSC = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"
if _os.path.exists(_SYS_DDSC):
    _ctypes.CDLL(_SYS_DDSC, mode=_ctypes.RTLD_GLOBAL)

import argparse
import sys

from . import SCENARIOS
from .runner import ScenarioRunner
from .report import print_report


def main():
    parser = argparse.ArgumentParser(
        description="Obstacle Avoidance Scenario Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Scenarios (easiest to hardest):
  open       - No obstacles, baseline navigation
  scattered  - 6 obstacles, clear paths between
  corridor   - Parallel walls, 1.5m gap
  L_wall     - L-shaped wall, must navigate around
  dead_end   - U-shaped trap, must find alternate path
  dense      - 14 obstacles, narrow gaps (~0.8m)
""",
    )
    parser.add_argument("--robot", default="b2",
                        help="Robot model (default: b2)")
    parser.add_argument("--scenario", default=None,
                        help="Run single scenario (default: all)")
    parser.add_argument("--genome", default=None,
                        help="Path to GA-evolved genome JSON")
    parser.add_argument("--domain", type=int, default=2,
                        help="DDS domain ID (default: 2)")
    parser.add_argument("--headed", action="store_true",
                        help="Run with MuJoCo viewer (default: headless)")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, defn in SCENARIOS.items():
            print(f"  {name:<15} {defn.scene_xml:<35} "
                  f"targets={defn.num_targets}, timeout={defn.timeout_per_target}s")
        sys.exit(0)

    # Select scenarios
    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    # Run
    runner = ScenarioRunner(
        robot=args.robot,
        genome=args.genome,
        domain=args.domain,
        headless=not args.headed,
    )

    results = runner.run_all(scenarios)
    print_report(results)

    # Exit code: 0 if all passed, 1 if any failed
    all_passed = all(
        not r.error and all(c.passed for c in r.critic_results)
        for r in results.values()
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

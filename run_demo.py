#!/usr/bin/env python3
"""Run the headed target game demo.

Usage:
    python run_demo.py                    # B2 with L5 defaults
    python run_demo.py --robot go2        # Go2
    python run_demo.py --targets 5 --seed 42
    python run_demo.py --headless         # No viewer window

Gait parameters are owned by Layer 5's RobotConfig.
The --genome flag was removed (see v12 postmortem for why).
"""
import argparse
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent  # robotics/


def main():
    parser = argparse.ArgumentParser(description="Headed target game demo")
    parser.add_argument("--robot", default="b2",
                        help="Robot model (default: b2)")
    parser.add_argument("--targets", type=int, default=3,
                        help="Number of targets (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--headless", action="store_true",
                        help="Run without viewer window")
    parser.add_argument("--domain", type=int, default=2,
                        help="DDS domain ID (default: 2)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "foreman.demos.target_game",
        "--robot", args.robot,
        "--targets", str(args.targets),
        "--full-circle",
    ]

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.headless:
        cmd.append("--headless")
    if args.domain is not None:
        cmd.extend(["--domain", str(args.domain)])

    print(f"\n  Robot: {args.robot}, targets: {args.targets}, domain: {args.domain}")
    print(f"  Gait params: L5 RobotConfig defaults\n")
    sys.stdout.flush()

    sys.exit(subprocess.call(cmd, cwd=str(_WORKSPACE)))


if __name__ == "__main__":
    main()

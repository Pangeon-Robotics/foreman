#!/usr/bin/env python3
"""Run the headed target game demo.

Usage:
    python run_demo.py              # default: b2, 3 targets, v14 seed genome
    python run_demo.py --robot go2
    python run_demo.py --targets 5 --seed 42
    python run_demo.py --no-genome  # run without GA-evolved genome
"""
import argparse
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent  # robotics/
_SEED_GENOMES = {
    "b2": _WORKSPACE / "training" / "models" / "b2" / "ga_v14_seed.json",
}


def main():
    parser = argparse.ArgumentParser(description="Headed target game demo")
    parser.add_argument("--robot", default="b2",
                        help="Robot model (default: b2)")
    parser.add_argument("--targets", type=int, default=3,
                        help="Number of targets (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to genome JSON (default: auto-detect v14 seed)")
    parser.add_argument("--no-genome", action="store_true",
                        help="Run without any genome file")
    parser.add_argument("--headless", action="store_true",
                        help="Run without viewer window")
    parser.add_argument("--domain", type=int, default=2,
                        help="DDS domain ID (default: 2, avoids conflict with running firmware)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "foreman.demos.target_game",
        "--robot", args.robot,
        "--targets", str(args.targets),
        "--full-circle",
    ]

    if not args.no_genome:
        genome = args.genome
        if genome is None:
            genome = _SEED_GENOMES.get(args.robot)
            if genome and genome.exists():
                genome = str(genome)
            else:
                genome = None
        if genome:
            cmd.extend(["--genome", genome])

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.headless:
        cmd.append("--headless")
    if args.domain is not None:
        cmd.extend(["--domain", str(args.domain)])

    sys.exit(subprocess.call(cmd, cwd=str(_WORKSPACE)))


if __name__ == "__main__":
    main()

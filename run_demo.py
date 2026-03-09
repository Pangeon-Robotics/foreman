#!/usr/bin/env python3
"""Run the headed target game demo.

Usage:
    python run_demo.py                    # latest GA champion (b2)
    python run_demo.py --genome v16       # specific GA version
    python run_demo.py --no-genome        # L5 defaults (no genome)
    python run_demo.py --robot go2        # Go2
    python run_demo.py --targets 5 --seed 42
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent  # robotics/
_GENOMES = {
    "b2": {
        "latest": _WORKSPACE / "training" / "models" / "b2" / "ga_best" / "final_best.json",
        "v16": _WORKSPACE / "training" / "models" / "b2" / "ga_v16_champion_r1.json",
        "v17": _WORKSPACE / "training" / "models" / "b2" / "ga_v17_champion.json",
    },
}


def _flatten_genome(data):
    """Extract flat genome dict from either grouped or flat format."""
    if "genome" in data:
        return data["genome"]
    flat = {}
    for key in ("gait_timing", "trajectory", "body_pose", "navigation", "horizon"):
        if key in data:
            flat.update(data[key])
    return flat


def _print_genome_info(genome_path):
    """Print genome parameters and fitness function being evaluated."""
    data = json.loads(Path(genome_path).read_text())
    g = _flatten_genome(data)

    print(f"\n  Genome: {Path(genome_path).name}")
    print(f"  GA fitness: {data.get('fitness', '?')}")
    print()
    print(f"  v17 per-timestep fitness = mean of:")
    print(f"    4.0 x stride_elegance   quadratic reward for step_length (0.10-0.70m)")
    print(f"    2.0 x stability         1 - K*(roll^2 + pitch^2 + d_roll^2 + d_pitch^2)")
    print(f"    2.0 x grip              fraction of stance feet with slip < 0.15 m/s")
    print(f"    1.5 x speed             velocity toward target / 2.0 m/s")
    print(f"    1.0 x turn              heading error reduction rate")
    print(f"  fall = 0")
    print()


def main():
    parser = argparse.ArgumentParser(description="Headed target game demo")
    parser.add_argument("--robot", default="b2",
                        help="Robot model (default: b2)")
    parser.add_argument("--targets", type=int, default=3,
                        help="Number of targets (default: 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--genome", type=str, default=None,
                        help="Genome: path, version name (v16/v17), or 'latest' (default: latest)")
    parser.add_argument("--no-genome", action="store_true",
                        help="Run without any genome file")
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

    genome_path = None
    if not args.no_genome:
        genome = args.genome or "latest"
        robot_genomes = _GENOMES.get(args.robot, {})
        if genome in robot_genomes:
            genome = robot_genomes[genome]
        path = Path(genome)
        if path.exists():
            genome_path = str(path)
            cmd.extend(["--genome", genome_path])
        else:
            print(f"Genome not found: {genome}")
            print(f"Available: {', '.join(robot_genomes.keys())}")
            sys.exit(1)

    if genome_path:
        _print_genome_info(genome_path)
    else:
        print(f"\n  Running with L5 defaults (no genome)\n")
    sys.stdout.flush()

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.headless:
        cmd.append("--headless")
    if args.domain is not None:
        cmd.extend(["--domain", str(args.domain)])

    sys.exit(subprocess.call(cmd, cwd=str(_WORKSPACE)))


if __name__ == "__main__":
    main()

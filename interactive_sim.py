"""Interactive sim: headless target game with structured telemetry to stdout.

Usage:
    python foreman/interactive_sim.py [--ticks N] [--seed S] [--domain D]

Prints one JSON line per tick with robot state, navigation commands, and
gait parameters. Designed for programmatic querying — no user interaction needed.
"""
from __future__ import annotations

import ctypes, os, sys, json, math, time
_SYS_DDSC = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"
if os.path.exists(_SYS_DDSC):
    ctypes.CDLL(_SYS_DDSC, mode=ctypes.RTLD_GLOBAL)

import argparse
from pathlib import Path

_root = Path(__file__).resolve().parents[1]  # workspace root
_layer5 = str(_root / "layer_5")
if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)

from simulation import SimulationManager
from foreman.demos.target_game.game import (
    TargetGame, configure_for_robot, CONTROL_DT, TARGET_TIMEOUT_STEPS,
)
from foreman.demos.target_game.utils import patch_layer_configs, normalize_angle


def run_interactive(ticks=500, seed=42, domain=10, robot='b2'):
    """Run headless sim, print JSON telemetry per tick."""
    import subprocess, glob

    # Cleanup
    subprocess.run(["pgrep", "-af", "firmware_sim.py"], capture_output=True)
    for pat in ["/tmp/god_view_*.bin", "/tmp/robot_view_*.bin", "/tmp/dwa_best_arc.bin"]:
        for f in glob.glob(pat):
            try: os.remove(f)
            except OSError: pass
    sf = f"/tmp/robo_sessions/b2_domain{domain}.json"
    try: os.remove(sf)
    except OSError: pass

    # Kill any firmware on this domain
    result = subprocess.run(["pgrep", "-af", "firmware_sim.py"],
                            capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            parts = line.split(None, 1)
            if len(parts) >= 2 and f"--domain {domain}" in parts[1]:
                subprocess.run(["kill", "-9", parts[0]], capture_output=True)
        time.sleep(1.0)

    configure_for_robot(robot)
    patch_layer_configs(robot, _root)

    # Patch firmware launcher for domain
    from launcher import FirmwareLauncher
    _orig_init = FirmwareLauncher.__init__
    def _patched_init(self, *a, **kw):
        kw["domain"] = domain
        _orig_init(self, *a, **kw)
    FirmwareLauncher.__init__ = _patched_init

    scene_path = _root / "layers_1_2" / "unitree_robots" / robot / "scene_target.xml"
    sim = SimulationManager(robot, headless=True, scene=str(scene_path),
                            track_bodies=["target"], domain=domain)
    sim.start()

    try:
        game = TargetGame(sim, robot=robot, num_targets=3, seed=seed,
                          angle_range=[
                              (math.pi / 6, math.pi),
                              (-math.pi, -math.pi / 6)],
                          timeout_steps=TARGET_TIMEOUT_STEPS)
        game._headless = True

        # Instrument _send_motion to capture commands
        _last_cmd = {}
        _orig_send = game._send_motion
        # Intercept L5's send_motion_command to capture GaitParams
        _orig_l5_send = sim.send_motion_command
        def _intercept_l5(cmd, **kw):
            params = _orig_l5_send(cmd, **kw)
            _last_cmd['gait'] = params.gait_type if params else '?'
            _last_cmd['step'] = params.step_length if params else 0
            _last_cmd['freq'] = params.gait_freq if params else 0
            _last_cmd['duty'] = params.duty_cycle if params else 0
            return params
        sim.send_motion_command = _intercept_l5

        def _instrumented_send(vx=0.0, wz=0.0, behavior='walk'):
            # vx, wz passed through from navigator
            _last_cmd['vx'] = vx
            _last_cmd['wz'] = wz
            _last_cmd['behavior'] = behavior
            _orig_send(vx=vx, wz=wz, behavior=behavior)
            _last_cmd['ramp'] = sim.is_ramping if hasattr(sim, 'is_ramping') else '?'
        game._send_motion = _instrumented_send

        # Instrument navigator to capture heading_err
        _nav_data = {}
        _orig_tick = game.nav.tick_walk_heading
        def _instrumented_tick():
            # Capture pre-tick nav state
            nav_x, nav_y, nav_yaw = game._get_nav_pose()
            target = game._spawner.current_target
            if target is not None:
                dx = target.x - nav_x
                dy = target.y - nav_y
                desired = math.atan2(dy, dx)
                _nav_data['heading_err'] = normalize_angle(desired - nav_yaw)
                _nav_data['dist'] = math.sqrt(dx*dx + dy*dy)
                _nav_data['nav_yaw'] = nav_yaw
                _nav_data['desired_heading'] = desired
                _nav_data['target_x'] = target.x
                _nav_data['target_y'] = target.y
                _nav_data['nav_x'] = nav_x
                _nav_data['nav_y'] = nav_y
            return _orig_tick()
        game.nav.tick_walk_heading = _instrumented_tick

        print("INTERACTIVE_SIM_START", flush=True)

        _next_tick = time.perf_counter()
        for i in range(ticks):
            # Pace at 200Hz (2x control rate) to stay synced with firmware physics
            _next_tick += CONTROL_DT
            _now = time.perf_counter()
            if _next_tick > _now:
                time.sleep(_next_tick - _now)

            alive = game.tick()
            if not alive:
                print(json.dumps({"tick": i, "done": True,
                                  "targets_reached": game._stats.targets_reached,
                                  "falls": game._stats.falls}), flush=True)
                break

            # Output telemetry every 25 ticks (4Hz at 100Hz control)
            if i % 25 == 0:
                body = sim.get_body("base")
                rec = {
                    "tick": i,
                    "t": round(i * CONTROL_DT, 2),
                    "state": str(game._state.name),
                    "x": round(float(body.pos[0]), 3),
                    "y": round(float(body.pos[1]), 3),
                    "z": round(float(body.pos[2]), 3),
                    "vx_world": round(float(body.linvel[0]), 3),
                    "vy_world": round(float(body.linvel[1]), 3),
                    "speed": round(math.sqrt(
                        float(body.linvel[0])**2 + float(body.linvel[1])**2), 3),
                    "yaw_rate": round(float(body.angvel[2]), 3),
                }
                rec.update({
                    "cmd_vx": round(_last_cmd.get('vx', 0), 3),
                    "cmd_wz": round(_last_cmd.get('wz', 0), 3),
                    "cmd_beh": _last_cmd.get('behavior', ''),
                    "ramp": _last_cmd.get('ramp', '?'),
                    "gait": _last_cmd.get('gait', '?'),
                    "step": round(_last_cmd.get('step', 0), 3),
                    "freq": round(_last_cmd.get('freq', 0), 2),
                })
                rec.update({
                    "heading_err": round(_nav_data.get('heading_err', 0), 3),
                    "dist": round(_nav_data.get('dist', 0), 3),
                    "nav_yaw": round(_nav_data.get('nav_yaw', 0), 3),
                    "target_x": round(_nav_data.get('target_x', 0), 3),
                    "target_y": round(_nav_data.get('target_y', 0), 3),
                })
                rec["targets_reached"] = game._stats.targets_reached
                rec["target_index"] = game._target_index
                print(json.dumps(rec), flush=True)

        print("INTERACTIVE_SIM_END", flush=True)
        print(json.dumps({
            "summary": True,
            "targets_reached": game._stats.targets_reached,
            "targets_spawned": game._stats.targets_spawned,
            "falls": game._stats.falls,
            "total_ticks": game._step_count,
        }), flush=True)

    finally:
        sim.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", type=int, default=10)
    parser.add_argument("--robot", default="b2")
    args = parser.parse_args()
    run_interactive(ticks=args.ticks, seed=args.seed,
                    domain=args.domain, robot=args.robot)

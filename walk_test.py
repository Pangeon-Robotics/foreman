"""Continuous stride ramp-up test for B2 with slip detection.

Keeps gait_freq=0.5Hz constant. Ramps commanded vx from 0.1 to 1.0 m/s
over 120s so stride length grows visibly. Uses Layer 5's SlipDetector
for real per-tick traction monitoring (contact forces, torque-motion
mismatch, IMU response, diagonal symmetry).

Usage:
    python foreman/walk_test.py
"""

import ctypes
import glob as _glob
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_SYS_DDSC = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"
if os.path.exists(_SYS_DDSC):
    ctypes.CDLL(_SYS_DDSC, mode=ctypes.RTLD_GLOBAL)

subprocess.run(["pkill", "-9", "-f", "firmware_sim.py"], capture_output=True)
time.sleep(1)
subprocess.run(["rm", "-f", "/tmp/robo_sessions/b2_domain52.json"], capture_output=True)

# Clean stale visualization files
for f in _glob.glob("/tmp/robot_view_*.bin") + _glob.glob("/tmp/god_view_*.bin"):
    os.remove(f)
for f in _glob.glob("/tmp/costmap_*.bin"):
    os.remove(f)

_root = Path(__file__).resolve().parents[1]
_foreman = Path(__file__).resolve().parent
_layer5 = str(_root / "layer_5")

if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)
if str(_foreman.parent) not in sys.path:
    sys.path.insert(0, str(_foreman.parent))

from config.defaults import MotionCommand
from simulation import SimulationManager
from slip_detector import SlipDetector
from foreman.demos.target_game.utils import patch_layer_configs

DOMAIN = 52
ROBOT = "b2"
DT = 0.01
TICK_SLEEP = 0.005

VX_START = 0.1
VX_END = 1.0
RAMP_SECONDS = 120
SETTLE_SECONDS = 5
REPORT_INTERVAL = 5  # seconds


def _yaw_deg(body):
    q = body.quat
    return math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))


def main():
    patch_layer_configs(ROBOT, _root)
    from velocity_mapper import map_velocity

    print("=" * 80)
    print("B2 Stride Ramp (0.5Hz, vx 0.1→1.0 over 120s) + SlipDetector")
    print("=" * 80)

    _scene = str(_root / "Assets" / "unitree_robots" / "b2" / "scene_scenario_open.xml")
    sim = SimulationManager(ROBOT, headless=False, domain=DOMAIN, transitions=False,
                            scene=_scene)
    sim.start()
    time.sleep(2.0)

    slip = SlipDetector(mass=83.5)  # B2

    # Settle
    print(f"\nSettling ({SETTLE_SECONDS}s)...")
    for _ in range(SETTLE_SECONDS * 100):
        cmd = MotionCommand(vx=0.0, behavior="stand", robot=ROBOT)
        sim.send_motion_command(cmd, dt=DT, terrain=False)
        time.sleep(TICK_SLEEP)

    body = sim.get_body("base")
    print(f"  Start: x={float(body.pos[0]):.3f}, z={float(body.pos[2]):.3f}")

    # Ramp
    print(f"\nRamping vx from {VX_START} to {VX_END} over {RAMP_SECONDS}s...")
    print(f"\n{'t':>5}  {'vx_cmd':>6}  {'step':>5}  {'v_theory':>8}  {'v_actual':>8}  "
          f"{'trac':>5}  {'slip_v':>6}  {'conf':>4}  {'yaw':>6}  {'z_min':>5}")
    print("-" * 80)

    total_steps = RAMP_SECONDS * 100
    z_min = 999.0
    fell = False

    prev_x = float(body.pos[0])
    prev_y = float(body.pos[1])
    report_steps = REPORT_INTERVAL * 100

    # Accumulate traction readings per report interval
    trac_samples = []
    slip_v_samples = []
    conf_samples = []

    for i in range(total_steps):
        frac = i / total_steps
        vx_cmd = VX_START + (VX_END - VX_START) * frac

        cmd = MotionCommand(vx=vx_cmd, wz=0.0, behavior="trot", robot=ROBOT)
        sim.send_motion_command(cmd, dt=DT, terrain=False)

        body = sim.get_body("base")
        if body:
            z = float(body.pos[2])
            z_min = min(z_min, z)
            if z < 0.25:
                fell = True

        # Run slip detector every tick
        robot_state = sim.get_robot_state()
        if robot_state is not None:
            est = slip.update(robot_state, commanded_vx=vx_cmd, commanded_wz=0.0)
            trac_samples.append(est.traction)
            slip_v_samples.append(est.slip_velocity)
            conf_samples.append(est.confidence)

        # Report
        if (i + 1) % report_steps == 0 and body:
            t = (i + 1) * DT
            x_now = float(body.pos[0])
            y_now = float(body.pos[1])
            dx = x_now - prev_x
            dy = y_now - prev_y
            dist = math.sqrt(dx*dx + dy*dy)
            v_actual = dist / REPORT_INTERVAL

            sl, freq, _ = map_velocity(vx_cmd)
            v_theory = sl * freq
            yaw = _yaw_deg(body)

            mean_trac = np.mean(trac_samples) if trac_samples else 0.0
            mean_slip_v = np.mean(slip_v_samples) if slip_v_samples else 0.0
            mean_conf = np.mean(conf_samples) if conf_samples else 0.0

            print(f"{t:5.0f}  {vx_cmd:6.2f}  {sl:5.3f}  {v_theory:8.3f}  {v_actual:8.3f}  "
                  f"{mean_trac:5.2f}  {mean_slip_v:6.3f}  {mean_conf:4.2f}  "
                  f"{yaw:+5.1f}°  {z_min:5.3f}")

            prev_x, prev_y = x_now, y_now
            z_min = 999.0
            trac_samples.clear()
            slip_v_samples.clear()
            conf_samples.clear()

        time.sleep(TICK_SLEEP)

        if fell:
            t = (i + 1) * DT
            sl, _, _ = map_velocity(vx_cmd)
            print(f"\n  *** FELL at t={t:.1f}s, vx_cmd={vx_cmd:.2f}, step={sl:.3f}m ***")
            break

    print("\n" + "=" * 80)
    print(f"  Fell: {'YES' if fell else 'NO'}")
    print("Done.")
    sim.stop()


if __name__ == "__main__":
    main()

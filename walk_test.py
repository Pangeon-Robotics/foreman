"""Walk test for B2 quadruped — headed (MuJoCo viewer).

Walks straight for 15s, then turns (wz=0.3) for 10s.

Usage:
    python foreman/walk_test.py
"""

import ctypes
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# --- DDS preload (must happen before any DDS imports) ---
_SYS_DDSC = "/usr/lib/x86_64-linux-gnu/libddsc.so.0.10.4"
if os.path.exists(_SYS_DDSC):
    ctypes.CDLL(_SYS_DDSC, mode=ctypes.RTLD_GLOBAL)

# --- Kill stale firmware and session files ---
subprocess.run(["pkill", "-9", "-f", "firmware_sim.py"], capture_output=True)
time.sleep(1)
subprocess.run(["rm", "-f", "/tmp/robo_sessions/b2_domain52.json"], capture_output=True)

# --- Path setup ---
_root = Path(__file__).resolve().parents[1]   # workspace root
_foreman = Path(__file__).resolve().parent    # foreman/
_layer5 = str(_root / "layer_5")

if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)

if str(_foreman.parent) not in sys.path:
    sys.path.insert(0, str(_foreman.parent))

from config.defaults import MotionCommand
from simulation import SimulationManager
from foreman.demos.target_game.utils import patch_layer_configs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN = 52
ROBOT = "b2"
VX_CMD = 1.0          # m/s forward
DT = 0.01             # s per control tick (100 Hz)
SETTLE_STEPS = 300    # 3s stand
STRAIGHT_STEPS = 1500 # 15s straight walk
TURN_STEPS = 1000     # 10s turning walk
TURN_WZ = 0.3         # rad/s turn rate
TICK_SLEEP = 0.005    # half real-time for headed mode


def _yaw_deg(body):
    """Extract yaw in degrees from body quaternion [w,x,y,z]."""
    q = body.quat
    return math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))


def main():
    print("=" * 60)
    print("B2 Walk Test (headed)")
    print(f"  vx={VX_CMD} m/s, turn wz={TURN_WZ} rad/s")
    print(f"  Domain={DOMAIN}")
    print("=" * 60)

    patch_layer_configs(ROBOT, _root)

    from config.defaults import STEP_LENGTH_SCALE, MAX_STEP_LENGTH, BASE_FREQ, TROT_STEP_HEIGHT, DUTY_CYCLES
    print(f"\n  L5 config check:")
    print(f"    STEP_LENGTH_SCALE={STEP_LENGTH_SCALE}, MAX_STEP_LENGTH={MAX_STEP_LENGTH}")
    print(f"    BASE_FREQ={BASE_FREQ}, TROT_STEP_HEIGHT={TROT_STEP_HEIGHT}")
    print(f"    DUTY_CYCLES['trot']={DUTY_CYCLES['trot']}")
    from velocity_mapper import map_velocity
    sl, freq, sh = map_velocity(VX_CMD)
    print(f"    map_velocity({VX_CMD}) -> step_length={sl}, freq={freq}, step_height={sh}\n")

    _scene = str(_root / "Assets" / "unitree_robots" / "b2" / "scene_scenario_open.xml")
    sim = SimulationManager(ROBOT, headless=False, domain=DOMAIN, transitions=False,
                            scene=_scene)
    sim.start()

    # Give firmware time to initialise
    time.sleep(2.0)

    # --- Phase 1: Settle (stand 3s) ---
    print("\n[1/4] Settling (3s stand)...")
    for _ in range(SETTLE_STEPS):
        cmd = MotionCommand(vx=0.0, behavior="stand", robot=ROBOT)
        sim.send_motion_command(cmd, dt=DT, terrain=False)
        time.sleep(TICK_SLEEP)

    body = sim.get_body("base")
    if body:
        print(f"  After settle: x={float(body.pos[0]):.3f} z={float(body.pos[2]):.3f}")

    # --- Phase 2: Straight walk (15s) ---
    print(f"\n[2/4] Straight walk: vx={VX_CMD}, wz=0.0 for 15s...")
    x0 = float(body.pos[0]) if body else 0.0
    z_min = 999.0
    fell = False

    for i in range(STRAIGHT_STEPS):
        cmd = MotionCommand(vx=VX_CMD, wz=0.0, behavior="trot", robot=ROBOT)
        sim.send_motion_command(cmd, dt=DT, terrain=False)

        body = sim.get_body("base")
        if body:
            z = float(body.pos[2])
            z_min = min(z_min, z)
            if z < 0.25:
                fell = True

            # Print every 1s
            if i % 100 == 0:
                t = i * DT
                x = float(body.pos[0])
                vx_actual = float(body.linvel[0])
                yaw = _yaw_deg(body)
                print(f"  t={t:5.1f}s  x={x:+7.3f}  z={z:.3f}  vx={vx_actual:+.3f}  yaw={yaw:+6.1f}°")

        time.sleep(TICK_SLEEP)

    body = sim.get_body("base")
    x_after_straight = float(body.pos[0]) if body else 0.0
    straight_disp = x_after_straight - x0
    straight_speed = straight_disp / (STRAIGHT_STEPS * DT)
    print(f"  Straight phase: displacement={straight_disp:.3f}m, avg_speed={straight_speed:.3f} m/s")

    # --- Phase 3: Turn walk (10s) ---
    print(f"\n[3/4] Turn walk: vx={VX_CMD}, wz={TURN_WZ} for 10s...")
    import numpy as np
    turn_vx_samples = []
    turn_wz_samples = []

    for i in range(TURN_STEPS):
        cmd = MotionCommand(vx=VX_CMD, wz=TURN_WZ, behavior="trot", robot=ROBOT)
        sim.send_motion_command(cmd, dt=DT, terrain=False)

        body = sim.get_body("base")
        if body:
            z = float(body.pos[2])
            z_min = min(z_min, z)
            if z < 0.25:
                fell = True

            vx_actual = float(body.linvel[0])
            # Angular velocity around z
            wz_actual = float(body.angvel[2]) if hasattr(body, 'angvel') else 0.0
            turn_vx_samples.append(vx_actual)
            turn_wz_samples.append(wz_actual)

            # Print every 1s
            if i % 100 == 0:
                t = (STRAIGHT_STEPS + i) * DT
                x = float(body.pos[0])
                yaw = _yaw_deg(body)
                print(f"  t={t:5.1f}s  x={x:+7.3f}  z={z:.3f}  vx={vx_actual:+.3f}  yaw={yaw:+6.1f}°  wz={wz_actual:+.3f}")

        time.sleep(TICK_SLEEP)

    # --- Phase 4: Summary ---
    print("\n[4/4] Summary")
    print("=" * 60)

    body = sim.get_body("base")
    if body:
        x_final = float(body.pos[0])
        y_final = float(body.pos[1])
        total_disp = np.sqrt((x_final - x0) ** 2 + y_final ** 2)
        total_time = (STRAIGHT_STEPS + TURN_STEPS) * DT
        avg_speed = total_disp / total_time

        print(f"  Total displacement : {total_disp:.3f} m")
        print(f"  Total time         : {total_time:.1f} s")
        print(f"  Average speed      : {avg_speed:.3f} m/s")
        print(f"  Speed ratio (v/cmd): {avg_speed / VX_CMD:.2f}")

    if turn_vx_samples:
        vx_arr = np.array(turn_vx_samples)
        wz_arr = np.array(turn_wz_samples)
        print(f"\n  Turn phase vx: mean={vx_arr.mean():.3f}, min={vx_arr.min():.3f}, max={vx_arr.max():.3f}")
        print(f"  Turn phase wz: mean={wz_arr.mean():.3f}, min={wz_arr.min():.3f}, max={wz_arr.max():.3f}")

    slip_ratio = straight_speed / VX_CMD if VX_CMD > 0 else 0.0
    print(f"\n  Straight slip ratio: {slip_ratio:.2f} (1.0 = no slip)")
    print(f"  Min body z         : {z_min:.3f}")
    print(f"  Fell               : {'YES' if fell else 'NO'}")

    if fell:
        print("\n  [FAIL] Robot fell!")
    elif avg_speed < 0.1:
        print("\n  [FAIL] Robot barely moved")
    else:
        print(f"\n  [OK] Walked at {avg_speed:.2f} m/s, {'no falls' if not fell else 'FELL'}")

    print("\nDone. Stopping simulation.")
    sim.stop()


if __name__ == "__main__":
    main()

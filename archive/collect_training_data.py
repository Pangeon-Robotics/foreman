"""Collect training data for body-state predictor.

Runs B2 through varied gaits (ramp up/down, turns, mixed) while recording
all hardware-available sensor data at 100Hz. Saves to .npz for training.

Recorded per tick (21D input + 6D target):
  - body_state: [vx, vy, wz, roll, pitch, yaw] from IMU+kinematics (6D)
  - gait_cmd: [step_length, gait_freq, step_height, wz_cmd, body_height] (5D)
  - foot_contacts: [FR, FL, RR, RL] (4D)
  - imu: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z] (6D)

Usage:
    python foreman/collect_training_data.py
    python foreman/collect_training_data.py --episodes 5
"""

import argparse
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
subprocess.run(["rm", "-f", "/tmp/robo_sessions/b2_domain53.json"], capture_output=True)
for f in _glob.glob("/tmp/robot_view_*.bin") + _glob.glob("/tmp/god_view_*.bin") + _glob.glob("/tmp/costmap_*.bin"):
    os.remove(f)

_root = Path(__file__).resolve().parents[1]
_layer5 = str(_root / "layer_5")
if _layer5 not in sys.path:
    sys.path.insert(0, _layer5)
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.defaults import MotionCommand
from simulation import SimulationManager
from foreman.demos.target_game.utils import patch_layer_configs

DOMAIN = 53
ROBOT = "b2"
DT = 0.01


def _quat_to_rpy(q):
    """Quaternion [w,x,y,z] to roll, pitch, yaw in radians."""
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = math.asin(max(-1, min(1, sinp)))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _body_state(body):
    """Extract 6D body state [vx, vy, wz, roll, pitch, yaw]."""
    vx = float(body.linvel[0])
    vy = float(body.linvel[1])
    wz = float(body.angvel[2])
    roll, pitch, yaw = _quat_to_rpy(body.quat)
    return np.array([vx, vy, wz, roll, pitch, yaw], dtype=np.float32)


def _imu_vec(state):
    """Extract 6D IMU [gyro_xyz, accel_xyz]."""
    gyro = np.asarray(state.imu_gyroscope, dtype=np.float32)
    accel = np.asarray(state.imu_accelerometer, dtype=np.float32)
    return np.concatenate([gyro, accel])


def _foot_vec(state):
    """Extract 4D foot contacts."""
    return np.asarray(state.foot_contacts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode definitions — varied gaits to cover the operating envelope
# ---------------------------------------------------------------------------

def _make_episodes(n_repeats=1):
    """Generate episode definitions. Each is (name, [(vx, wz, seconds), ...])."""
    episodes = []

    # 1. Straight ramp up 0.1→1.0 over 60s
    episodes.append(("ramp_up", [("ramp", 0.1, 1.0, 0.0, 60)]))

    # 2. Straight ramp down 1.0→0.1 over 60s
    episodes.append(("ramp_down", [("ramp", 1.0, 0.1, 0.0, 60)]))

    # 3. Constant speeds
    for vx in [0.2, 0.5, 0.8]:
        episodes.append((f"const_vx{vx}", [("const", vx, 0.0, 30)]))

    # 4. Turns at various speeds
    for vx in [0.3, 0.6]:
        for wz in [-0.5, -0.3, 0.3, 0.5]:
            episodes.append((f"turn_vx{vx}_wz{wz}", [("const", vx, wz, 20)]))

    # 5. Mixed: walk then turn then walk
    episodes.append(("mixed_1", [
        ("const", 0.5, 0.0, 15),
        ("const", 0.5, 0.4, 10),
        ("const", 0.5, 0.0, 15),
        ("const", 0.5, -0.4, 10),
    ]))

    # 6. Speed changes (step function)
    episodes.append(("speed_steps", [
        ("const", 0.2, 0.0, 10),
        ("const", 0.6, 0.0, 10),
        ("const", 0.3, 0.0, 10),
        ("const", 0.8, 0.0, 10),
        ("const", 0.4, 0.0, 10),
    ]))

    # 7. Gentle sinusoidal turn
    episodes.append(("sine_turn", [("sine_wz", 0.5, 0.4, 40)]))

    if n_repeats > 1:
        episodes = episodes * n_repeats

    return episodes


def _run_episode(sim, segments, map_velocity):
    """Run one episode, return recorded data arrays."""
    body_states = []
    gait_cmds = []
    foot_contacts = []
    imus = []
    timestamps = []

    t = 0.0
    for seg in segments:
        seg_type = seg[0]

        if seg_type == "ramp":
            _, vx_start, vx_end, wz, duration = seg
            steps = int(duration / DT)
            for i in range(steps):
                frac = i / steps
                vx = vx_start + (vx_end - vx_start) * frac
                _tick(sim, vx, wz, t, body_states, gait_cmds,
                      foot_contacts, imus, timestamps, map_velocity)
                t += DT

        elif seg_type == "const":
            _, vx, wz, duration = seg
            steps = int(duration / DT)
            for _ in range(steps):
                _tick(sim, vx, wz, t, body_states, gait_cmds,
                      foot_contacts, imus, timestamps, map_velocity)
                t += DT

        elif seg_type == "sine_wz":
            _, vx, wz_amp, duration = seg
            steps = int(duration / DT)
            for i in range(steps):
                wz = wz_amp * math.sin(2 * math.pi * i / steps)
                _tick(sim, vx, wz, t, body_states, gait_cmds,
                      foot_contacts, imus, timestamps, map_velocity)
                t += DT

    return (np.array(body_states), np.array(gait_cmds),
            np.array(foot_contacts), np.array(imus), np.array(timestamps))


def _tick(sim, vx, wz, t, body_states, gait_cmds, foot_contacts, imus,
          timestamps, map_velocity):
    """One control tick: send command, record data."""
    cmd = MotionCommand(vx=vx, wz=wz, behavior="trot", robot=ROBOT)
    sim.send_motion_command(cmd, dt=DT, terrain=False)

    body = sim.get_body("base")
    state = sim.get_robot_state()

    if body is not None and state is not None:
        bs = _body_state(body)
        body_states.append(bs)

        sl, freq, sh = map_velocity(abs(vx))
        gait_cmds.append(np.array([sl, freq, sh, wz, 0.465], dtype=np.float32))

        foot_contacts.append(_foot_vec(state))
        imus.append(_imu_vec(state))
        timestamps.append(t)

    time.sleep(0.002)  # headless — fast as possible with minimal sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, help="Repeat all episodes N times")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    args = parser.parse_args()

    patch_layer_configs(ROBOT, _root)
    from velocity_mapper import map_velocity

    out_dir = _root / "foreman" / "tmp" / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(out_dir / "body_state_data.npz")

    episodes = _make_episodes(args.episodes)
    total_secs = sum(seg[-1] for _, segs in episodes for seg in segs)

    print("=" * 70)
    print(f"Collecting training data: {len(episodes)} episodes, ~{total_secs:.0f}s total")
    print(f"Output: {out_path}")
    print("=" * 70)

    scene = str(_root / "Assets" / "unitree_robots" / "b2" / "scene_scenario_open.xml")
    sim = SimulationManager(ROBOT, headless=True, domain=DOMAIN, transitions=False,
                            scene=scene)
    sim.start()
    time.sleep(2.0)

    all_body = []
    all_gait = []
    all_foot = []
    all_imu = []
    all_ts = []

    for ep_i, (name, segments) in enumerate(episodes):
        # Settle 3s between episodes
        for _ in range(300):
            cmd = MotionCommand(vx=0.0, behavior="stand", robot=ROBOT)
            sim.send_motion_command(cmd, dt=DT, terrain=False)
            time.sleep(0.001)

        print(f"  [{ep_i+1}/{len(episodes)}] {name} ...", end="", flush=True)
        t0 = time.time()

        bs, gc, fc, im, ts = _run_episode(sim, segments, map_velocity)

        elapsed = time.time() - t0
        print(f" {len(bs)} samples in {elapsed:.1f}s")

        if len(bs) > 0:
            # Offset timestamps to be globally monotonic
            offset = all_ts[-1] + DT if all_ts else 0.0
            all_body.append(bs)
            all_gait.append(gc)
            all_foot.append(fc)
            all_imu.append(im)
            all_ts.extend((ts + offset).tolist())

    sim.stop()

    body_arr = np.concatenate(all_body, axis=0)
    gait_arr = np.concatenate(all_gait, axis=0)
    foot_arr = np.concatenate(all_foot, axis=0)
    imu_arr = np.concatenate(all_imu, axis=0)
    ts_arr = np.array(all_ts, dtype=np.float32)

    np.savez_compressed(
        out_path,
        body_state=body_arr,
        gait_cmd=gait_arr,
        foot_contacts=foot_arr,
        imu=imu_arr,
        timestamps=ts_arr,
        dt=DT,
    )

    print(f"\nSaved {len(body_arr)} samples to {out_path}")
    print(f"  body_state:   {body_arr.shape} (vx, vy, wz, roll, pitch, yaw)")
    print(f"  gait_cmd:     {gait_arr.shape} (step_len, freq, step_h, wz_cmd, body_h)")
    print(f"  foot_contacts:{foot_arr.shape} (FR, FL, RR, RL)")
    print(f"  imu:          {imu_arr.shape}  (gyro_xyz, accel_xyz)")
    print(f"  timestamps:   {ts_arr.shape}")


if __name__ == "__main__":
    main()

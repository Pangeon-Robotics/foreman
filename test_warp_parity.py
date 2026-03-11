#!/usr/bin/env python3
"""Warp-to-CPU parity test: run v18 champion genome on CPU MuJoCo.

Tests the thesis: a genome should function the same in MJWarp (GPU) as
in MuJoCo CPU. Uses the exact same fused numpy control pipeline
(control_numpy.py) and model (b2_mjwarp.xml), only the physics backend
differs.

v18 champion reference (MJWarp): 2.77 m/s, 83.2m in 30s, fitness 138.7

Usage:
    python foreman/test_warp_parity.py                 # Straight walking (parity test)
    python foreman/test_warp_parity.py --headed         # With MuJoCo viewer
    python foreman/test_warp_parity.py --target-angle 90  # With navigation/turning
    python foreman/test_warp_parity.py --no-ramp        # No startup ramp (instant gait)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# Add training repo to path for control_numpy
_TRAINING_GA = str(Path(__file__).resolve().parent.parent / "training" / "ga")
sys.path.insert(0, _TRAINING_GA)

from control_numpy import (
    HOME_Q, NOMINAL_HEIGHT,
    compute_navigation, compute_phases, compute_foot_positions,
    batch_ik, compute_pd_torque,
    quat_to_yaw, quat_to_rpy,
)

# -----------------------------------------------------------------------
# v18 champion genome (from training/models/b2/ga_v18/champion_v18.json)
# -----------------------------------------------------------------------
CHAMPION = {
    "STEP_LENGTH":       0.417,
    "GAIT_FREQ":         4.37,
    "STEP_HEIGHT":       0.092,
    "DUTY_CYCLE":        0.55,
    "SWING_APEX_PHASE":  0.629,
    "SWING_HEIGHT_RATIO": 1.771,
    "SWING_OVERSHOOT":   0.073,
    "KP_HEADING":        4.817,
    "WZ_MAX":            1.204,
    "VX_WALK":           0.743,
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "ga" / "models" / "b2_mjwarp.xml"

# Same as mjwarp_runner.py
_HOME_QPOS = np.array([
    0, 0, 0.465, 1, 0, 0, 0,  # base: pos(3) + quat(4)
    0.0, 0.841, -1.689,  # FL (body tree order)
    0.0, 0.841, -1.689,  # FR
    0.0, 0.841, -1.689,  # RL
    0.0, 0.841, -1.689,  # RR
], dtype=np.float64)


def run_cpu_episode(target_heading_deg: float = 0.0, headed: bool = False,
                    straight: bool = True, ramp_seconds: float = 2.0) -> dict:
    """Run v18 champion genome on CPU MuJoCo, return results."""

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Physics parameters (match mjwarp_runner.py exactly)
    kp = 500.0
    kd = 25.0
    control_dt = 0.01
    physics_per_ctrl = 5
    startup_steps = 150  # 1.5s settle
    walk_steps = 3000    # 30s
    nominal_z = NOMINAL_HEIGHT

    # Genome as (1,) arrays for control_numpy (expects batch dim)
    step_length     = np.array([CHAMPION["STEP_LENGTH"]])
    gait_freq       = np.array([CHAMPION["GAIT_FREQ"]])
    step_height     = np.array([CHAMPION["STEP_HEIGHT"]])
    duty_cycle      = np.array([CHAMPION["DUTY_CYCLE"]])
    swing_apex      = np.array([CHAMPION["SWING_APEX_PHASE"]])
    swing_hratio    = np.array([CHAMPION["SWING_HEIGHT_RATIO"]])
    swing_overshoot = np.array([CHAMPION["SWING_OVERSHOOT"]])
    kp_heading      = np.array([CHAMPION["KP_HEADING"]])
    wz_max          = np.array([CHAMPION["WZ_MAX"]])
    vx_walk         = np.array([CHAMPION["VX_WALK"]])

    # Target
    target_heading = math.radians(target_heading_deg)
    target_distance = 25.0
    tx = np.array([target_distance * math.cos(target_heading)])
    ty = np.array([target_distance * math.sin(target_heading)])

    # Reset to home pose
    data.qpos[:] = _HOME_QPOS
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    q_prev = HOME_Q.reshape(1, 12).copy()

    # Optional viewer
    viewer = None
    if headed:
        from mujoco.viewer import launch_passive
        viewer = launch_passive(model, data)

    # Startup settle — hold standing pose
    for _ in range(startup_steps):
        q_act = data.sensordata[0:12].reshape(1, 12)
        dq_act = data.sensordata[12:24].reshape(1, 12)
        torque = compute_pd_torque(q_prev, q_act, np.zeros_like(q_act), dq_act, 500.0, 25.0)
        data.ctrl[:] = torque.flatten()
        for _ in range(physics_per_ctrl):
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

    # Record initial position
    rx0 = data.qpos[0]
    ry0 = data.qpos[1]
    start_dist = math.sqrt((tx[0] - rx0)**2 + (ty[0] - ry0)**2)

    fell = False
    fall_step = None
    speed_log = []

    t0 = time.perf_counter()

    # Walk loop
    for step in range(walk_steps):
        t = step * control_dt

        # Read state
        q_act = data.sensordata[0:12].reshape(1, 12)
        dq_act = data.sensordata[12:24].reshape(1, 12)

        rx = np.array([data.qpos[0]])
        ry = np.array([data.qpos[1]])
        rz = data.qpos[2]
        body_quat = data.qpos[3:7].reshape(1, 4)
        vx_w = data.qvel[0]
        vy_w = data.qvel[1]
        yaw = quat_to_yaw(body_quat)
        roll, pitch, _ = quat_to_rpy(body_quat)

        # Fall detection
        if rz < nominal_z * 0.5:
            fell = True
            fall_step = step
            break

        # Speed logging (every 10 steps = 10Hz)
        if step % 10 == 0:
            desired = math.atan2(ty[0] - ry[0], tx[0] - rx[0])
            v_toward = vx_w * math.cos(desired) + vy_w * math.sin(desired)
            v_ground = math.sqrt(vx_w**2 + vy_w**2)
            speed_log.append({
                "t": t, "x": rx[0], "y": ry[0], "z": rz,
                "v_ground": v_ground, "v_toward": v_toward,
                "roll": float(roll[0]), "pitch": float(pitch[0]),
            })

        # Navigation
        vx_cmd, wz_cmd, _ = compute_navigation(
            rx, ry, yaw, tx, ty, kp_heading, wz_max, vx_walk)

        if straight:
            wz_cmd = np.zeros(1)

        # Gait phase
        is_stance, local_phase, _ = compute_phases(t, gait_freq, duty_cycle)

        # Ramp step_length from 0 to full to avoid cold-start torque spike
        ramp = min(1.0, t / ramp_seconds) if ramp_seconds > 0 else 1.0
        sl_ramped = step_length * ramp

        # Foot positions, IK, PD
        feet = compute_foot_positions(
            is_stance, local_phase, sl_ramped, step_height, wz_cmd,
            swing_apex, swing_hratio, swing_overshoot)

        q_target = batch_ik(feet)
        dq_ff = 0.9 * (q_target - q_prev) / control_dt
        torque = compute_pd_torque(q_target, q_act, dq_ff, dq_act, kp, kd)
        data.ctrl[:] = torque.flatten()
        q_prev = q_target

        for _ in range(physics_per_ctrl):
            mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

    wall_time = time.perf_counter() - t0

    # Results
    final_x = data.qpos[0]
    final_y = data.qpos[1]
    distance = math.sqrt((final_x - rx0)**2 + (final_y - ry0)**2)
    final_dist = math.sqrt((tx[0] - final_x)**2 + (ty[0] - final_y)**2)
    dist_closed = start_dist - final_dist
    active_time = (fall_step if fell else walk_steps) * control_dt
    avg_speed = distance / active_time if active_time > 0 else 0

    # Speed stats from log
    if speed_log:
        v_grounds = [s["v_ground"] for s in speed_log]
        v_towards = [s["v_toward"] for s in speed_log]
        avg_v_ground = np.mean(v_grounds)
        avg_v_toward = np.mean(v_towards)
        max_v_ground = np.max(v_grounds)
        rolls = [abs(s["roll"]) for s in speed_log]
        pitches = [abs(s["pitch"]) for s in speed_log]
    else:
        avg_v_ground = avg_v_toward = max_v_ground = 0
        rolls = pitches = [0]

    # MJWarp fitness formula
    V_REF = 2.0
    max_possible = V_REF * walk_steps * control_dt  # 60m
    fitness = max(0.0, dist_closed / max_possible * 100.0)
    if fell:
        fitness *= 0.5

    if viewer is not None:
        viewer.close()

    return {
        "distance_m": distance,
        "dist_closed_m": dist_closed,
        "speed_ms": avg_speed,
        "avg_v_ground": avg_v_ground,
        "avg_v_toward": avg_v_toward,
        "max_v_ground": max_v_ground,
        "active_time_s": active_time,
        "fell": fell,
        "fall_step": fall_step,
        "fitness": fitness,
        "avg_roll_deg": math.degrees(np.mean(rolls)),
        "avg_pitch_deg": math.degrees(np.mean(pitches)),
        "wall_time_s": wall_time,
        "final_pos": (final_x, final_y),
    }


def main():
    parser = argparse.ArgumentParser(description="Warp-to-CPU parity test")
    parser.add_argument("--headed", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--target-angle", type=float, default=0.0,
                        help="Target heading in degrees (default: 0)")
    parser.add_argument("--navigate", action="store_true",
                        help="Enable navigation (turning toward target)")
    parser.add_argument("--no-ramp", action="store_true",
                        help="Disable startup ramp (instant full stride)")
    args = parser.parse_args()

    straight = not args.navigate
    ramp_seconds = 0.0 if args.no_ramp else 2.0

    print("=" * 60)
    print("Warp-to-CPU Parity Test: v18 Champion Genome")
    print("=" * 60)
    print(f"\nModel: {MODEL_PATH.name}")
    print(f"Mode:  {'straight' if straight else f'navigate to {args.target_angle}°'}")
    print(f"Ramp:  {ramp_seconds:.1f}s")
    print(f"\nGenome:")
    for k, v in CHAMPION.items():
        print(f"  {k:24s} = {v}")

    print(f"\nRunning 30s episode on CPU MuJoCo...")
    result = run_cpu_episode(
        target_heading_deg=args.target_angle,
        headed=args.headed,
        straight=straight,
        ramp_seconds=ramp_seconds,
    )

    print(f"\n{'─' * 60}")
    print(f"Results:")
    print(f"  Distance traveled:  {result['distance_m']:.2f} m")
    print(f"  Avg speed (d/t):    {result['speed_ms']:.2f} m/s")
    print(f"  Avg ground speed:   {result['avg_v_ground']:.2f} m/s")
    print(f"  Max ground speed:   {result['max_v_ground']:.2f} m/s")
    print(f"  Active time:        {result['active_time_s']:.1f} s")
    print(f"  Fell:               {result['fell']}")
    if result['fell']:
        print(f"  Fall at step:       {result['fall_step']} ({result['fall_step'] * 0.01:.1f}s)")
    print(f"  Avg |roll|:         {result['avg_roll_deg']:.2f}°")
    print(f"  Avg |pitch|:        {result['avg_pitch_deg']:.2f}°")
    print(f"  Wall time:          {result['wall_time_s']:.1f}s")

    print(f"\n{'─' * 60}")
    print(f"MJWarp reference:     2.77 m/s, 83.2m in 30s")
    print(f"CPU result:           {result['speed_ms']:.2f} m/s, "
          f"{result['distance_m']:.1f}m in {result['active_time_s']:.1f}s")

    if result['fell']:
        print(f"\nFAILED — fell at {result['fall_step'] * 0.01:.1f}s")
    elif abs(result['speed_ms'] - 2.77) < 0.5:
        print(f"\nPARITY CONFIRMED — within 0.5 m/s of MJWarp")
    else:
        delta = result['speed_ms'] - 2.77
        print(f"\nSpeed delta: {delta:+.2f} m/s vs MJWarp")


if __name__ == "__main__":
    main()

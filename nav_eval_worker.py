#!/usr/bin/env python3
"""Worker for nav_optimize.py: run single seed with given navigator params.

Usage:
    python nav_eval_worker.py --seed 42 --domain 72 --params tmp/nav_eval_params.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.getcwd())

# Kill any stale firmware on this domain
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--domain", type=int, required=True)
parser.add_argument("--params", type=str, required=True)
args = parser.parse_args()

# Load nav params
params = json.loads(open(args.params).read())

# Monkey-patch navigator before importing game
import foreman.demos.target_game.navigator_helper as nh
nh.Navigator.KP_HEADING = params.get("KP_HEADING", 4.8)
nh.Navigator.KD_HEADING = params.get("KD_HEADING", 0.5)
nh.Navigator.VX_WALK = 2.0  # Always 2.0 (V_REF is immutable)
nh.Navigator._WZ_ABS_MAX = params.get("WZ_ABS_MAX", 0.6)
nh.Navigator._LOOKAHEAD = params.get("LOOKAHEAD", 1.5)

# Patch the speed control block to use VX_FLOOR and TURN_VX range
_vx_floor = params.get("VX_FLOOR", 0.6)
_turn_lo = params.get("TURN_VX_LO", 0.6)
_turn_hi = params.get("TURN_VX_HI", 0.8)

import math
from foreman.demos.target_game.utils import normalize_angle as _normalize_angle
from foreman.demos.target_game.utils import clamp as _clamp
from foreman.demos.target_game import game_config as C

_orig_tick = nh.Navigator.tick_walk_heading

def _patched_tick(self):
    """Patched tick with configurable speed control."""
    g = self.game
    x, y, yaw, z, roll, pitch = g._get_robot_pose()
    target = g._spawner.current_target
    g._target_step_count += 1

    if g._check_fall(z):
        return

    dist = target.distance_to(x, y)

    if dist < g._reach_threshold:
        g.scoring.on_reached()
        return

    if g._target_step_count >= g._timeout_steps:
        g.scoring.on_timeout()
        return

    steer_x, steer_y = self._get_steer_target(x, y, target.x, target.y)
    heading_err = _normalize_angle(
        math.atan2(steer_y - y, steer_x - x) - yaw)

    heading_err_rate = (heading_err - self._prev_heading_err) / C.CONTROL_DT
    self._prev_heading_err = heading_err
    wz_raw = self.KP_HEADING * heading_err + self.KD_HEADING * heading_err_rate
    wz = _clamp(wz_raw, -self._WZ_ABS_MAX, self._WZ_ABS_MAX)

    cos_err = max(0.0, math.cos(heading_err))
    base_vx = max(_vx_floor, self.VX_WALK * cos_err)
    wz_frac = min(1.0, abs(wz) / self._WZ_ABS_MAX) if self._WZ_ABS_MAX > 0 else 0.0
    turn_vx = _clamp(base_vx, _turn_lo, _turn_hi)
    vx = base_vx * (1.0 - wz_frac) + turn_vx * wz_frac

    g._send_motion(vx=vx, wz=wz)

    # Minimal telemetry (skip slip, path_critic, diagnostics for speed)
    r_deg = math.degrees(roll)
    p_deg = math.degrees(pitch)
    mode = "W"
    if len(g._rp_log) > 0:
        prev = g._rp_log[-1]
        droll = (r_deg - prev[1]) / C.CONTROL_DT
        dpitch = (p_deg - prev[2]) / C.CONTROL_DT
    else:
        droll = dpitch = 0.0
    g._rp_log.append((g._step_count, r_deg, p_deg, droll, dpitch, mode))

nh.Navigator.tick_walk_heading = _patched_tick

# Clean up stale sessions for this domain
session_dir = "/tmp/robo_sessions"
if os.path.exists(session_dir):
    for f in os.listdir(session_dir):
        try:
            os.remove(os.path.join(session_dir, f))
        except:
            pass

from foreman.demos.target_game.__main__ import run_game
from types import SimpleNamespace

game_args = SimpleNamespace(
    robot='b2', targets=1, headless=True, seed=args.seed,
    genome=None, full_circle=False, domain=args.domain,
    slam=True, obstacles=True, scene_path=None,
    timeout_per_target=30, min_dist=3.0, max_dist=6.0,
    angle_range=None, spawn_fn=None, viewer=False, god=False,
)

t0 = time.monotonic()
try:
    r = run_game(game_args)
    dt = time.monotonic() - t0
    reached = r.stats.targets_reached if hasattr(r.stats, 'targets_reached') else 0
    falls = r.stats.falls if hasattr(r.stats, 'falls') else 0
    print(f"RESULT reached={reached} time={dt:.1f} falls={falls}")
except Exception as e:
    print(f"RESULT reached=0 time=30.0 falls=1", file=sys.stdout)
    print(f"ERROR: {e}", file=sys.stderr)

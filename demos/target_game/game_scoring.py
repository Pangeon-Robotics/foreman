"""Game lifecycle: startup, spawn, scoring, run loop, and analysis.

Provides GameScoring with state machine lifecycle methods (startup, spawn,
reached, timeout), the main run() loop, and post-game analysis (SLAM drift,
pitch/roll dynamics).

All output goes to GameTelemetry — this module never prints.
"""
from __future__ import annotations

import math
import os
import time

from . import game_config as C


class GameScoring:
    """Game lifecycle, scoring, and analysis helper for TargetGame."""

    def __init__(self, game):
        self.game = game

    def tick_startup(self):
        """Hold neutral stand for settle period."""
        g = self.game
        if C.WHEELED:
            g._sim.send_wheel_command(0.0, 0.0, dt=C.CONTROL_DT)
        else:
            g._send_motion(behavior='stand')
        if g._step_count >= C.STARTUP_SETTLE_STEPS:
            x, y, yaw, z, roll, pitch = g._get_robot_pose()
            roll_deg = abs(math.degrees(roll))
            if (z < C.NOMINAL_BODY_HEIGHT * C.FALL_THRESHOLD
                    or roll_deg > 5.0):
                g._gt.record_event(
                    "startup_fail", step=g._step_count,
                    t=g._step_count * C.CONTROL_DT,
                    z=z, roll=math.degrees(roll), pitch=math.degrees(pitch))
                g._stats.falls += 1
                g._state = C.GameState.DONE
                return
            g._gt.record_event(
                "startup_ok", step=g._step_count,
                t=g._step_count * C.CONTROL_DT,
                z=z, roll=math.degrees(roll), pitch=math.degrees(pitch),
                wheeled=C.WHEELED)
            g._state = C.GameState.SPAWN_TARGET

    def tick_spawn(self):
        """Spawn a new target and transition to walking."""
        g = self.game
        if g._target_index >= g._num_targets:
            g._state = C.GameState.DONE
            return

        if g._post_fall_settle > 0:
            g._post_fall_settle -= 1
            _, _, _, z, _, _ = g._get_robot_pose()
            recovered = z > C.NOMINAL_BODY_HEIGHT * 0.85
            if not recovered:
                if g._post_fall_settle > 0:
                    g._send_motion(behavior='stand')
                    return
                g._gt.record_event(
                    "fall_recovery_fail", step=g._step_count,
                    t=g._step_count * C.CONTROL_DT)
                g._state = C.GameState.DONE
                return

        nav_x, nav_y, nav_yaw = g._get_nav_pose()
        if g._spawn_fn is not None:
            target = g._spawn_fn(
                nav_x, nav_y, nav_yaw, g._target_index)
            g._spawner._current_target = target
        else:
            target = g._spawner.spawn_relative(
                nav_x, nav_y, nav_yaw,
                angle_range=g._angle_range)
        g._target_index += 1
        g._target_step_count = 0
        g._reach_threshold = g._reach_threshold_base
        g._min_target_dist = float('inf')
        g._committed_path = None
        g._committed_path_step = 0
        g._stats.targets_spawned += 1

        target_z = min(C.NOMINAL_BODY_HEIGHT, 0.30)
        try:
            g._sim.set_mocap_pos(
                0, [target.x, target.y, target_z])
        except (RuntimeError, AttributeError):
            pass

        # Update perception with target position
        if g._perception is not None:
            g._perception.set_target_position(target.x, target.y)

        dist = target.distance_to(nav_x, nav_y)
        g._gt.record_event(
            "spawn", step=g._step_count,
            t=g._step_count * C.CONTROL_DT,
            target_index=g._target_index,
            x=target.x, y=target.y, dist=dist)

        if g._path_critic is not None:
            g._path_critic.set_target(target.x, target.y)
            g._path_critic.record(nav_x, nav_y, t=0.0)

        # Write green dot path before walking starts
        g.nav.write_initial_path()

        g._state = C.GameState.WALK_TO_TARGET

    def on_reached(self):
        """Handle target reached event."""
        g = self.game
        t = g._target_step_count * C.CONTROL_DT
        g._stats.targets_reached += 1
        ato = None
        agg_ato = None
        if g._path_critic is not None:
            target = g._spawner.current_target
            report = g._path_critic.target_reached(target.x, target.y)
            if report:
                ato = report['ato_score']
                agg_ato = g._path_critic.aggregate_ato()
        g._gt.record_event(
            "reached", step=g._step_count, t=t,
            target_index=g._target_index,
            ato=ato, agg_ato=agg_ato)
        g._state = C.GameState.SPAWN_TARGET

    def on_timeout(self):
        """Handle target timeout event."""
        g = self.game
        t = g._target_step_count * C.CONTROL_DT
        g._stats.targets_timeout += 1
        agg_ato = None
        if g._path_critic is not None:
            target = g._spawner.current_target
            g._path_critic.target_timeout(target.x, target.y)
            agg_ato = g._path_critic.aggregate_ato()
        g._gt.record_event(
            "timeout", step=g._step_count, t=t,
            target_index=g._target_index,
            agg_ato=agg_ato)
        g._state = C.GameState.SPAWN_TARGET

    def run(self) -> C.GameStatistics:
        """Run the full game loop and return statistics."""
        g = self.game
        start_time = time.monotonic()

        headed = not getattr(g._sim, '_headless', False)
        _min_tick_dt = C.CONTROL_DT  # match physics 100Hz rate in all modes
        _next_tick = time.monotonic()
        while g.tick():
            _next_tick += _min_tick_dt
            _now = time.monotonic()
            if _next_tick > _now:
                time.sleep(_next_tick - _now)

        g._stats.total_time = time.monotonic() - start_time
        if g._telemetry is not None:
            g._telemetry.close()
        try:
            os.unlink(g._PATH_VIZ_FILE)
        except OSError:
            pass
        self._build_summary()
        return g._stats

    def _build_summary(self):
        """Build end-of-game summary into telemetry."""
        g = self.game
        s = g._stats
        g._gt.summary["targets_reached"] = s.targets_reached
        g._gt.summary["targets_spawned"] = s.targets_spawned
        g._gt.summary["success_rate"] = s.success_rate
        g._gt.summary["targets_timeout"] = s.targets_timeout
        g._gt.summary["falls"] = s.falls
        g._gt.summary["total_time"] = s.total_time

        # SLAM drift
        if g._odometry is not None and g._truth_trail:
            drifts = [
                math.sqrt((s_[0] - t_[0])**2 + (s_[1] - t_[1])**2)
                for s_, t_ in zip(g._slam_trail, g._truth_trail)
            ]
            if drifts:
                g._gt.summary["slam"] = {
                    "samples": len(drifts),
                    "min": min(drifts),
                    "max": max(drifts),
                    "mean": sum(drifts) / len(drifts),
                    "final": drifts[-1],
                }

        # Roll/pitch analysis
        if g._rp_log:
            rolls = [r[1] for r in g._rp_log]
            pitches = [r[2] for r in g._rp_log]
            drolls = [r[3] for r in g._rp_log]
            dpitches = [r[4] for r in g._rp_log]
            g._gt.summary["rp"] = {
                "samples": len(g._rp_log),
                "roll_min": min(rolls), "roll_max": max(rolls),
                "roll_mean": sum(abs(r) for r in rolls) / len(rolls),
                "pitch_min": min(pitches), "pitch_max": max(pitches),
                "pitch_mean": sum(abs(p) for p in pitches) / len(pitches),
                "droll_min": min(drolls), "droll_max": max(drolls),
                "dpitch_min": min(dpitches), "dpitch_max": max(dpitches),
            }
            # Per-mode stats
            for label, code in [("walk", "W"), ("turn", "T"), ("drive", "D")]:
                mr = [abs(r[1]) for r in g._rp_log if r[5] == code]
                mp = [abs(r[2]) for r in g._rp_log if r[5] == code]
                if mr:
                    g._gt.summary["rp"][f"{label}_steps"] = len(mr)
                    g._gt.summary["rp"][f"{label}_avg_roll"] = sum(mr) / len(mr)
                    g._gt.summary["rp"][f"{label}_avg_pitch"] = sum(mp) / len(mp)
                    g._gt.summary["rp"][f"{label}_max_roll"] = max(mr)
                    g._gt.summary["rp"][f"{label}_max_pitch"] = max(mp)

        # Path critic
        if g._path_critic is not None:
            g._gt.summary["ato"] = g._path_critic.get_reports()

"""Game lifecycle: startup, spawn, scoring, run loop, and analysis.

Provides GameScoring with state machine lifecycle methods (startup, spawn,
reached, timeout), the main run() loop, and post-game analysis (SLAM drift,
pitch/roll dynamics).
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
        if C.WHEELED and g._home_q is not None:
            g.nav._send_wheeled(0.0, 0.0)
        else:
            params = g._L4GaitParams(
                gait_type='trot', step_length=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0, wz=0.0,
                body_height=C.BODY_HEIGHT,
            )
            g._send_l4(params)
        if g._step_count >= C.STARTUP_SETTLE_STEPS:
            x, y, yaw, z, roll, pitch = g._get_robot_pose()
            roll_deg = abs(math.degrees(roll))
            if (z < C.NOMINAL_BODY_HEIGHT * C.FALL_THRESHOLD
                    or roll_deg > 5.0):
                print(
                    f"Startup FAILED: z={z:.3f}m  "
                    f"R={math.degrees(roll):+.1f}\u00b0  "
                    f"P={math.degrees(pitch):+.1f}\u00b0  "
                    f"(fell or unstable settle)")
                g._stats.falls += 1
                g._state = C.GameState.DONE
                return
            if C.WHEELED:
                if g._home_q is None:
                    state = g._sim.get_robot_state()
                    g._home_q = [
                        state.joint_positions[i] for i in range(12)]
                    print(
                        f"Startup complete (wheeled, L4 pose): "
                        f"z={z:.3f}m  home_q="
                        f"[{g._home_q[0]:.3f}, "
                        f"{g._home_q[1]:.3f}, "
                        f"{g._home_q[2]:.3f}]")
                else:
                    print(
                        f"Startup complete (wheeled): z={z:.3f}m  "
                        f"home_q={C.WHEEL_HOME_Q[:3]}")
            else:
                print(
                    f"Startup complete: z={z:.3f}m  "
                    f"R={math.degrees(roll):+.1f}\u00b0  "
                    f"P={math.degrees(pitch):+.1f}\u00b0")
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
            if not recovered and g._post_fall_settle > 0:
                params = g._L4GaitParams(
                    gait_type='trot', step_length=0.0,
                    gait_freq=C.GAIT_FREQ, step_height=0.0,
                    duty_cycle=1.0, stance_width=0.0, wz=0.0,
                    body_height=C.BODY_HEIGHT,
                )
                g._send_l4(params)
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
        g._slam_drift_latched = False
        g._in_tip_mode = False
        g._tip_start_step = 0
        g._tip_start_heading = 0.0
        g._tip_cooldown_until = 0
        g._tip_cooldown = 0
        g._reset_tip()
        g._smooth_heading_mod = 0.65
        g._smooth_wz = 0.0
        g._decel_tick_count = 0
        g._smooth_dwa_turn = 0.0
        g._smooth_dwa_fwd = 1.0
        g._avoidance_sign = 0
        g._avoidance_dist = 0.0
        g._last_threat_level = 0.0
        g._current_waypoint = None
        g._wp_commit_until = 0
        g._last_replan_pos = None
        g._last_replan_step = -999
        g._use_waypoint_latch = False
        # Reset stuck recovery helper state
        g.stuck.reset()
        g._orbit_heading_ticks = 0
        g._orbit_dist_ticks = 0
        g._orbit_dist_active = False
        g._orbit_slow_ticks = 0
        g._tip_ticks = 0
        g._prev_heading_err = 0.0
        g._committed_path = None
        g._committed_path_step = 0
        g._last_good_heading_step = -999
        g._progress_window_dist = float('inf')
        g._progress_window_step = 0
        g._dist_ring = []
        g._recent_regression = 0.0
        g._closest_approach = float('inf')
        g._reg_start_step = 0
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
        print(
            f"\n[{g._target_index}/{g._num_targets}] "
            f"SPAWN ({target.x:.1f},{target.y:.1f}) d={dist:.1f}")

        if g._path_critic is not None:
            g._path_critic.set_target(target.x, target.y)
            g._path_critic.record(nav_x, nav_y, t=0.0)

        g._state = C.GameState.WALK_TO_TARGET

    def on_reached(self):
        """Handle target reached event."""
        g = self.game
        t = g._target_step_count * C.CONTROL_DT
        g._stats.targets_reached += 1
        ato_s = ""
        if g._path_critic is not None:
            target = g._spawner.current_target
            report = g._path_critic.target_reached(target.x, target.y)
            if report:
                agg = g._path_critic.aggregate_ato()
                ato_s = f" ATO={report['ato_score']:.0f} agg={agg:.0f}"
        print(f"  REACHED {t:.1f}s{ato_s}")
        g._state = C.GameState.SPAWN_TARGET

    def on_timeout(self):
        """Handle target timeout event."""
        g = self.game
        t = g._target_step_count * C.CONTROL_DT
        g._stats.targets_timeout += 1
        ato_s = ""
        if g._path_critic is not None:
            target = g._spawner.current_target
            report = g._path_critic.target_timeout(target.x, target.y)
            if report:
                agg = g._path_critic.aggregate_ato()
                ato_s = f" agg={agg:.0f}"
        print(f"  TIMEOUT {t:.1f}s{ato_s}")
        g._state = C.GameState.SPAWN_TARGET

    def run(self) -> C.GameStatistics:
        """Run the full game loop and return statistics."""
        g = self.game
        start_time = time.monotonic()

        headed = not getattr(g._sim, '_headless', False)
        _min_tick_dt = C.CONTROL_DT / 2.0 if not headed else C.CONTROL_DT
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
        self.print_summary()
        if g._path_critic is not None:
            g._path_critic.print_summary()
        return g._stats

    def print_summary(self):
        """Print end-of-game summary."""
        g = self.game
        s = g._stats
        print("\n=== GAME OVER ===")
        print(f"Targets: {s.targets_reached}/{s.targets_spawned} reached "
              f"({s.success_rate:.0%})")
        print(f"Timeouts: {s.targets_timeout}  Falls: {s.falls}")
        print(f"Total time: {s.total_time:.1f}s")
        self._print_slam_analysis()
        self._print_rp_analysis()

    def _print_slam_analysis(self):
        """Print SLAM drift analysis if odometry was active."""
        g = self.game
        if g._odometry is None or not g._truth_trail:
            return
        drifts = [
            math.sqrt((s[0] - t[0])**2 + (s[1] - t[1])**2)
            for s, t in zip(g._slam_trail, g._truth_trail)
        ]
        if not drifts:
            return
        print("\n=== SLAM DRIFT ANALYSIS ===")
        print(f"Samples: {len(drifts)} "
              f"({len(drifts) * 0.1:.1f}s at 10Hz)")
        print(f"Drift: min={min(drifts):.3f}m  "
              f"max={max(drifts):.3f}m  "
              f"mean={sum(drifts)/len(drifts):.3f}m  "
              f"final={drifts[-1]:.3f}m")

    def _print_rp_analysis(self):
        """Print pitch/roll dynamics analysis."""
        g = self.game
        if not g._rp_log:
            return
        rolls = [r[1] for r in g._rp_log]
        pitches = [r[2] for r in g._rp_log]
        drolls = [r[3] for r in g._rp_log]
        dpitches = [r[4] for r in g._rp_log]

        print("\n=== PITCH/ROLL ANALYSIS ===")
        print(f"Samples: {len(g._rp_log)} "
              f"({len(g._rp_log)*C.CONTROL_DT:.1f}s)")
        print(f"Roll:   min={min(rolls):+.1f}\u00b0  "
              f"max={max(rolls):+.1f}\u00b0  "
              f"mean={sum(abs(r) for r in rolls)/len(rolls):.1f}\u00b0")
        print(f"Pitch:  min={min(pitches):+.1f}\u00b0  "
              f"max={max(pitches):+.1f}\u00b0  "
              f"mean="
              f"{sum(abs(p) for p in pitches)/len(pitches):.1f}\u00b0")
        print(f"dRoll:  min={min(drolls):+.0f}\u00b0/s  "
              f"max={max(drolls):+.0f}\u00b0/s")
        print(f"dPitch: min={min(dpitches):+.0f}\u00b0/s  "
              f"max={max(dpitches):+.0f}\u00b0/s")

        scored = [(abs(r[1]) + abs(r[2]), i, r)
                  for i, r in enumerate(g._rp_log)]
        scored.sort(reverse=True)
        print("\nTop 10 worst tilt moments (|R|+|P|):")
        print(f"  {'step':>6}  {'t':>5}  {'R':>7}  {'P':>7}  "
              f"{'dR':>8}  {'dP':>8}  mode")
        for _, idx, r in scored[:10]:
            step, roll, pitch, dr, dp, mode = r
            t = step * C.CONTROL_DT
            print(f"  {step:6d}  {t:5.2f}  {roll:+7.1f}  "
                  f"{pitch:+7.1f}  "
                  f"{dr:+8.0f}  {dp:+8.0f}  {mode}")

        for label, code in [("WALK", "W"), ("TURN", "T"),
                            ("DRIVE", "D")]:
            mode_rolls = [abs(r[1]) for r in g._rp_log
                          if r[5] == code]
            mode_pitches = [abs(r[2]) for r in g._rp_log
                            if r[5] == code]
            if mode_rolls:
                print(
                    f"\n{label} ({len(mode_rolls)} steps):  "
                    f"avg|R|="
                    f"{sum(mode_rolls)/len(mode_rolls):.1f}\u00b0  "
                    f"avg|P|="
                    f"{sum(mode_pitches)/len(mode_pitches):.1f}"
                    f"\u00b0  "
                    f"max|R|={max(mode_rolls):.1f}\u00b0  "
                    f"max|P|={max(mode_pitches):.1f}\u00b0")

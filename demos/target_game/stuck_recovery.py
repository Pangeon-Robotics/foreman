"""Stuck detection and recovery for DWA navigation.

Provides StuckRecovery with check() and recover() methods. Tracks stuck
state independently (owns stuck_check_dist, stuck_recovery_countdown, etc.).
"""
from __future__ import annotations

import math

from . import game_config as C


class StuckRecovery:
    """Stuck detection and recovery helper for TargetGame."""

    def __init__(self, game):
        self.game = game
        # Stuck detection state (owned by this helper)
        self._stuck_check_dist = float('inf')
        self._stuck_check_step = 0
        self._stuck_recovery_countdown = 0
        self._stuck_recovery_wz = 0.0
        self._prev_no_progress = False
        self._no_progress_streak = 0

    def reset(self):
        """Reset stuck state for new target."""
        self._stuck_check_dist = float('inf')
        self._stuck_check_step = 0
        self._stuck_recovery_countdown = 0
        self._stuck_recovery_wz = 0.0
        self._prev_no_progress = False
        self._no_progress_streak = 0

    def check(self, nav_x, nav_y, heading_err, dist, target):
        """Check for stuck conditions and run recovery.

        Returns True if handling stuck (caller should return early).
        """
        g = self.game
        dwa_feas = (g._last_dwa_result.n_feasible
                    if g._last_dwa_result else 999)
        if (g._target_step_count % 150 == 0
                and g._target_step_count > 0):
            no_progress = dist >= self._stuck_check_dist - 0.3
            jammed = dwa_feas < 5 and no_progress
            blocked_fwd = (
                g._smooth_dwa_fwd < 0.1
                and no_progress
                and not g._in_tip_mode
                and dwa_feas < 30)
            # Prolonged stuck: 3 consecutive no-progress checks (4.5s)
            streak = self._no_progress_streak
            if no_progress:
                streak += 1
            else:
                streak = 0
            self._no_progress_streak = streak
            prolonged_stuck = (streak >= 3
                              and not g._in_tip_mode
                              and dwa_feas < 30)
            # Low-score stuck
            low_score_stuck = (
                no_progress
                and g._last_dwa_result is not None
                and g._last_dwa_result.score < 0.05
                and dwa_feas < 15
                and not g._in_tip_mode)
            if jammed or blocked_fwd or prolonged_stuck or low_score_stuck:
                self._stuck_recovery_countdown = 100  # 1s at 100Hz
                self._no_progress_streak = 0
                if abs(g._smooth_dwa_turn) > 0.1:
                    self._stuck_recovery_wz = math.copysign(
                        C.TURN_WZ, g._smooth_dwa_turn)
                elif abs(heading_err) > 0.05:
                    self._stuck_recovery_wz = math.copysign(
                        C.TURN_WZ, heading_err)
                else:
                    self._stuck_recovery_wz = C.TURN_WZ
                g._current_waypoint = None
                g._wp_commit_until = 0
                reason = ("jammed" if jammed else
                          "blocked_fwd" if blocked_fwd else
                          "low_score" if low_score_stuck else
                          f"no_progress\u00d7{streak}")
                print(
                    f"  [STUCK {reason} feas={dwa_feas}"
                    f" \u2192 turn {math.degrees(self._stuck_recovery_wz):+.0f}\u00b0/s]")
            self._stuck_check_dist = dist

        if self._stuck_recovery_countdown > 0:
            return self._recover(heading_err, dist, target)
        return False

    def _recover(self, heading_err, dist, target):
        """Execute stuck recovery maneuver. Returns True always."""
        g = self.game
        self._stuck_recovery_countdown -= 1
        if self._stuck_recovery_countdown >= 70:
            params = g._L4GaitParams(
                gait_type='trot', step_length=0.0, wz=0.0,
                gait_freq=C.GAIT_FREQ, step_height=0.0,
                duty_cycle=1.0, stance_width=0.0,
                body_height=C.BODY_HEIGHT,
            )
        else:
            params = g._L4GaitParams(
                gait_type='trot', wz=self._stuck_recovery_wz,
                step_length=0.0,
                gait_freq=C.TURN_FREQ,
                step_height=C.TURN_STEP_HEIGHT,
                duty_cycle=C.TURN_DUTY_CYCLE,
                stance_width=C.TURN_STANCE_WIDTH,
                body_height=C.BODY_HEIGHT,
                turn_in_place=True,
            )
        if self._stuck_recovery_countdown == 0:
            self._stuck_check_dist = dist
        g._send_l4(params)
        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            phase = ("STOP"
                     if self._stuck_recovery_countdown >= 70
                     else "TURN")
            x_gt, y_gt, _, _, _, _ = g._get_robot_pose()
            clr = g._gt_clearance()
            clr_s = f" clr={clr:.1f}" if clr < 50 else ""
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"R-{phase:<4} t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_gt:.1f},{y_gt:.1f}){clr_s}")
        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return True
        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()
        return True

"""Navigation helper producing MotionCommands for Layer 5.

PD heading law follows the committed route. The route (Lazy Theta*
on the costmap) is the single source of obstacle avoidance. Navigation
just follows it as fast as possible.

Wheeled robots use L5's send_wheel_command() for PD leg hold +
differential wheel torque.
"""
from __future__ import annotations

import math

from . import game_config as C
from .dwa_path_export import export_path
from .utils import normalize_angle as _normalize_angle

# Slip detection (Layer 5 module, hardware-compatible)
try:
    from slip_detector import SlipDetector
except ImportError:
    SlipDetector = None


def _heading_to_route(path, nav_x, nav_y, nav_yaw, lookahead=1.0):
    """Compute heading error to a point ~lookahead meters ahead on the path.

    Finds the closest point on the path, then walks forward along the
    path by lookahead distance to find the steering target. Returns
    the heading error (desired_heading - nav_yaw), normalized to [-pi, pi].
    """
    if path is None or len(path) < 2:
        return None

    # Find closest point on path
    best_i, best_d2 = 0, float('inf')
    for i, (px, py) in enumerate(path):
        d2 = (px - nav_x) ** 2 + (py - nav_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    # Walk forward along path by lookahead distance
    cumul = 0.0
    target_pt = path[-1]
    for i in range(best_i + 1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        cumul += (dx * dx + dy * dy) ** 0.5
        if cumul >= lookahead:
            target_pt = path[i]
            break

    dx = target_pt[0] - nav_x
    dy = target_pt[1] - nav_y
    desired_heading = math.atan2(dy, dx)
    return _normalize_angle(desired_heading - nav_yaw)


class Navigator:
    """PD route follower for TargetGame.

    WALK mode: PD heading law follows the committed route.
      wz = Kp * heading_err + Kd * d(heading_err)/dt
      vx = constant (route handles obstacle avoidance)

    TIP mode: ignored for now (placeholder for future).
    """

    # PD heading gains — tuned for B2 at 100Hz
    KP_HEADING = 3.0    # proportional: catch errors early before they grow
    KD_HEADING = 0.1    # derivative: low damping, let corrections happen fast

    # Walk speed
    VX_WALK = 1.0       # m/s (increase once turn-radius-aware routing is in)
    WZ_MAX = 1.5        # rad/s — L5 clamps at 1.5

    # TIP (placeholder — not used in WALK mode)
    TURN_WZ = 2.0

    def __init__(self, game):
        self.game = game
        self._prev_heading_err = 0.0
        self._in_tip = False
        _masses = {'b2': 83.5, 'go2': 15.0, 'b2w': 83.5, 'go2w': 15.0}
        _robot = getattr(game, '_robot', 'b2')
        self._slip = SlipDetector(mass=_masses.get(_robot, 83.5)) if SlipDetector else None

    # ------------------------------------------------------------------
    # Main tick — legged robots via L5
    # ------------------------------------------------------------------

    def tick_walk_heading(self):
        """Follow the route using PD heading control.

        PD law: wz = Kp * heading_err + Kd * d(heading_err)/dt
        vx = constant. Route handles obstacle avoidance.
        Layer 5 handles gait selection, gain scheduling, and L4 dispatch.
        """
        g = self.game
        x_truth, y_truth, yaw_truth, z, roll, pitch = g._get_robot_pose()
        target = g._spawner.current_target
        g._target_step_count += 1

        if g._check_fall(z):
            return

        nav_x, nav_y, nav_yaw = g._get_nav_pose()
        dist = target.distance_to(nav_x, nav_y)

        # Event-driven route replan (on costmap change or no path)
        if g._committed_path is None or getattr(g, '_costmap_changed', False):
            if g._path_critic is not None:
                export_path(g, target.x, target.y)
                g._costmap_changed = False

        # PD heading law: follow the route
        heading_err = _heading_to_route(
            g._committed_path, nav_x, nav_y, nav_yaw, lookahead=1.0)
        if heading_err is None:
            # No route — steer directly at target
            heading_err = _normalize_angle(
                math.atan2(target.y - nav_y, target.x - nav_x) - nav_yaw)

        # D term: rate of change of heading error (damping)
        heading_err_rate = (heading_err - self._prev_heading_err) / C.CONTROL_DT
        self._prev_heading_err = heading_err

        # PD law
        wz_raw = self.KP_HEADING * heading_err + self.KD_HEADING * heading_err_rate
        wz = max(-self.WZ_MAX, min(self.WZ_MAX, wz_raw))

        # Speed-turn coupling: slow when turning, min 60% to keep stride differential effective
        alignment = max(0.0, math.cos(heading_err))
        vx = self.VX_WALK * (0.3 + 0.7 * alignment)

        # Path critic: record position at 10Hz
        if (g._path_critic is not None
                and g._target_step_count % 10 == 0):
            g._path_critic.record(
                nav_x, nav_y, t=g._target_step_count * C.CONTROL_DT,
                v_cmd=vx)

        g._send_motion(vx=vx, wz=wz)

        # Slip detection
        slip_est = None
        if self._slip is not None:
            robot_state = g._sim.get_robot_state()
            if robot_state is not None:
                slip_est = self._slip.update(
                    robot_state, commanded_vx=vx, commanded_wz=wz)

        # Diagnostics
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

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            _prev = getattr(g, '_nav_prev_pos', None)
            if _prev is not None:
                _dx = x_truth - _prev[0]
                _dy = y_truth - _prev[1]
                _v = math.sqrt(_dx*_dx + _dy*_dy) / (C.TELEMETRY_INTERVAL * C.CONTROL_DT)
            else:
                _v = 0.0
            g._nav_prev_pos = (x_truth, y_truth)
            _trac = f" trac={slip_est.traction:.2f}" if slip_est else ""
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"WALK  t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"v={_v:.2f} vx={vx:.2f} wz={wz:+.2f}"
                f"{_trac} "
                f"({x_truth:.1f},{y_truth:.1f})")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

    # ------------------------------------------------------------------
    # Wheeled robots via L5's send_wheel_command()
    # ------------------------------------------------------------------

    def tick_walk_wheeled(self):
        """Drive toward target using Layer 5's wheel torque API.

        Proportional steering produces fwd/turn torques. Layer 5's
        send_wheel_command() handles PD leg hold + differential wheels.
        """
        g = self.game
        x_truth, y_truth, yaw_truth, z, roll, pitch = g._get_robot_pose()
        target = g._spawner.current_target
        g._target_step_count += 1

        if g._check_fall(z):
            return

        nav_x, nav_y, nav_yaw = g._get_nav_pose()
        heading_err = target.heading_error(nav_x, nav_y, nav_yaw)
        dist = target.distance_to(nav_x, nav_y)

        alignment = max(0.0, math.cos(heading_err))
        alignment = alignment * alignment
        dist_taper = min(1.0, dist / 1.0)
        fwd = C.WHEEL_FWD_TORQUE * alignment * dist_taper

        from .utils import clamp as _clamp
        turn = _clamp(C.WHEEL_KP_YAW * heading_err, -C.WHEEL_MAX_TURN, C.WHEEL_MAX_TURN)

        g._sim.send_wheel_command(fwd, turn, dt=C.CONTROL_DT)

        r_deg = math.degrees(roll)
        p_deg = math.degrees(pitch)
        mode = "D"
        if len(g._rp_log) > 0:
            prev = g._rp_log[-1]
            droll = (r_deg - prev[1]) / C.CONTROL_DT
            dpitch = (p_deg - prev[2]) / C.CONTROL_DT
        else:
            droll = dpitch = 0.0
        g._rp_log.append((g._step_count, r_deg, p_deg, droll, dpitch, mode))

        if g._target_step_count % C.TELEMETRY_INTERVAL == 0:
            t = g._target_step_count * C.CONTROL_DT
            print(
                f"[{g._target_index}/{g._num_targets}] "
                f"DRIVE t={t:.1f} d={dist:.1f} "
                f"err={math.degrees(heading_err):+.0f}\u00b0 "
                f"({x_truth:.1f},{y_truth:.1f})")

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

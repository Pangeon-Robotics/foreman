"""Navigation helper producing MotionCommands for Layer 5.

Philosophy: walk the path, stop when the target is reached.

Green dots mean the full pipeline ran: LiDAR → TSDF → costmap → A* route.
No green dots = no costmap yet, robot steers directly at target.

Wheeled robots use L5's send_wheel_command() for PD leg hold +
differential wheel torque.
"""
from __future__ import annotations

import math
import struct

from . import game_config as C
from .telemetry import TickSample
from .utils import normalize_angle as _normalize_angle, clamp as _clamp

# Slip detection (Layer 5 module, hardware-compatible)
try:
    from slip_detector import SlipDetector
except ImportError:
    SlipDetector = None

_PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"


def _resample_path(points, spacing=0.20):
    """Resample a polyline to evenly-spaced points (>=5 per meter)."""
    if not points or len(points) < 2:
        return points or []
    result = [points[0]]
    residual = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        seg = math.sqrt(dx * dx + dy * dy)
        if seg < 1e-6:
            continue
        pos = residual
        while pos < seg:
            t = pos / seg
            result.append((points[i - 1][0] + dx * t,
                           points[i - 1][1] + dy * t))
            pos += spacing
        residual = pos - seg
    result.append(points[-1])
    return result


def _write_path_dots(points):
    """Write resampled (x,y) waypoints as green dots (>=5/m).

    Uses atomic rename to prevent viewer from reading partial files.
    """
    import os
    pts = _resample_path(points)
    if len(pts) < 2:
        return
    buf = bytearray(len(pts) * 8)
    for i, (px, py) in enumerate(pts):
        struct.pack_into('ff', buf, i * 8, px, py)
    tmp = _PATH_VIZ_FILE + ".tmp"
    try:
        with open(tmp, 'wb') as f:
            f.write(buf)
        os.replace(tmp, _PATH_VIZ_FILE)
    except OSError:
        pass


def _clear_path_dots():
    """Remove path dots file so viewer shows no route."""
    import os
    try:
        os.remove(_PATH_VIZ_FILE)
    except OSError:
        pass


class Navigator:
    """Walk the path, stop when reached.

    PD heading control steers toward the next waypoint on the A* route.
    Speed = VX_WALK * dist_factor * heading_factor with EMA smoothing.
    L5 handles gait selection and turn-speed coupling.
    """

    # PD heading gains
    KP_HEADING = 4.8
    KD_HEADING = 0.5

    # Walk speed
    VX_WALK = 2.0       # m/s
    _WZ_ABS_MAX = 1.0   # rad/s — stride-differential safety cap
    _APPROACH_DIST = 4.0  # m — distance taper zone for deceleration
    _VX_RAMP_S = 2.0    # s — startup ramp to avoid torque spike
    _MAX_STRIDE_DIFF = 0.10  # m — max per-side stride asymmetry
    _STEP_LENGTH_SCALE = 0.2085  # mirror L5 config for wz limit calc

    # Route following
    _WP_REACH = 0.30    # m — advance to next waypoint when this close
    _LOOKAHEAD = 1.5    # m — steer at a point this far along the path

    def __init__(self, game):
        self.game = game
        self._prev_heading_err = 0.0
        self._vx_ema = 0.0
        self._committed_path = None  # list of (x,y) Theta* waypoints
        self._wp_index = 0
        self._has_costmap_path = False
        _masses = {'b2': 83.5, 'go2': 15.0, 'b2w': 83.5, 'go2w': 15.0}
        _robot = getattr(game, '_robot', 'b2')
        self._slip = SlipDetector(mass=_masses.get(_robot, 83.5)) if SlipDetector else None

    def write_initial_path(self):
        """Reset route state for a new target.

        No green dots until the full pipeline runs (TSDF -> costmap -> A*).
        """
        self._has_costmap_path = False
        self._committed_path = None
        self._prev_heading_err = 0.0
        self._vx_ema = 0.0
        _clear_path_dots()

        # Try to plan immediately if costmap already exists
        g = self.game
        target = g._spawner.current_target
        if target is None:
            return
        x, y, _ = g._get_nav_pose()
        self._commit_route(x, y, target.x, target.y)

    def _plan_astar_path(self, sx, sy, gx, gy):
        """Run A* on the cost grid, return list of (x,y) or None."""
        g = self.game
        critic = g._path_critic
        if critic is None or getattr(critic, '_cost_grid', None) is None:
            return None
        from .astar import _astar_core
        return _astar_core(
            critic._cost_grid, critic._cost_origin_x, critic._cost_origin_y,
            critic._cost_voxel_size, critic._robot_radius,
            (sx, sy), (gx, gy), return_path=True, force_passable=False,
            cost_truncation=getattr(critic, '_cost_truncation', 0.5))

    def _commit_route(self, sx, sy, gx, gy):
        """Plan A* path and commit it. Green dots written only on success."""
        path = self._plan_astar_path(sx, sy, gx, gy)
        if path and len(path) >= 2:
            self._committed_path = path
            self._has_costmap_path = True
            self._wp_index = min(1, len(self._committed_path) - 1)
            _write_path_dots(self._committed_path)

    def _get_steer_target(self, x, y, target_x, target_y):
        """Return the point to steer toward.

        Before costmap: steer directly at target.
        After costmap: follow A* waypoints. Replan when costmap updates.
        """
        g = self.game

        # Consume costmap-changed flag
        costmap_updated = getattr(g, '_costmap_changed', False)
        if costmap_updated:
            g._costmap_changed = False

        # No costmap path yet — plan as soon as costmap arrives
        if not self._has_costmap_path:
            if costmap_updated:
                self._commit_route(x, y, target_x, target_y)
            if self._committed_path is None:
                return target_x, target_y

        # Advance past reached waypoints
        if self._committed_path is not None and len(self._committed_path) >= 2:
            while self._wp_index < len(self._committed_path) - 1:
                wx, wy = self._committed_path[self._wp_index]
                d = math.sqrt((x - wx)**2 + (y - wy)**2)
                if d < self._WP_REACH:
                    self._wp_index += 1
                else:
                    break

        # Replan when costmap updates (obstacles may have changed)
        if costmap_updated and self._has_costmap_path:
            self._commit_route(x, y, target_x, target_y)

        # Update green dots to show remaining route
        if costmap_updated and self._committed_path is not None:
            remaining = self._committed_path[self._wp_index:]
            if len(remaining) >= 2:
                _write_path_dots(remaining)

        # Steer toward current waypoint (with lookahead for smoothing).
        # If heading to the waypoint is > 60°, steer at a point farther
        # along the path to prevent chasing nearby waypoints at steep
        # angles (which causes overshoot and spiral).
        if self._committed_path is not None and self._wp_index < len(self._committed_path):
            sx, sy = self._lookahead_point(x, y)
            # If the lookahead point requires a very large turn,
            # steer directly at the final target instead
            herr = abs(math.atan2(sy - y, sx - x) - math.atan2(target_y - y, target_x - x))
            if herr > math.pi:
                herr = 2 * math.pi - herr
            if herr > 0.5:  # >30° divergence from target direction
                return target_x, target_y
            return sx, sy
        return target_x, target_y

    def _lookahead_point(self, x, y):
        """Find a point _LOOKAHEAD meters ahead along the remaining path."""
        path = self._committed_path
        idx = self._wp_index
        remaining = self._LOOKAHEAD

        px, py = x, y
        for i in range(idx, len(path)):
            wx, wy = path[i]
            seg = math.sqrt((wx - px)**2 + (wy - py)**2)
            if seg >= remaining and seg > 1e-6:
                t = remaining / seg
                return (px + t * (wx - px), py + t * (wy - py))
            remaining -= seg
            px, py = wx, wy

        return path[-1]

    # ------------------------------------------------------------------
    # Main tick — legged robots via L5
    # ------------------------------------------------------------------

    def tick_walk_heading(self):
        """Walk the path, stop when reached.

        PD heading control steers toward the next A* waypoint.
        Speed = VX_WALK * dist_factor * heading_factor with EMA smoothing.
        L5 handles gait selection and turn-speed coupling.
        """
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

        # Follow the path
        steer_x, steer_y = self._get_steer_target(x, y, target.x, target.y)
        heading_err = _normalize_angle(
            math.atan2(steer_y - y, steer_x - x) - yaw)

        # PD heading control
        heading_err_rate = (heading_err - self._prev_heading_err) / C.CONTROL_DT
        self._prev_heading_err = heading_err
        wz_raw = self.KP_HEADING * heading_err + self.KD_HEADING * heading_err_rate

        # Stride-differential-safe wz + absolute cap
        sl_est = max(self._vx_ema * self._STEP_LENGTH_SCALE, 0.02)
        wz_safe = min(self._WZ_ABS_MAX, self._MAX_STRIDE_DIFF / (sl_est * 0.5))
        wz = _clamp(wz_raw, -wz_safe, wz_safe)

        # Speed = VX_WALK * dist_factor * heading_factor
        dist_factor = min(1.0, dist / self._APPROACH_DIST)
        heading_factor = max(0.25, math.cos(heading_err))
        vx_target = self.VX_WALK * dist_factor * heading_factor

        # Asymmetric EMA: smooth stride transitions
        if vx_target < self._vx_ema:
            self._vx_ema += 0.10 * (vx_target - self._vx_ema)
        else:
            self._vx_ema += 0.05 * (vx_target - self._vx_ema)
        vx = self._vx_ema

        # Startup ramp: avoid torque spike from instant full stride
        t_global = g._step_count * C.CONTROL_DT
        if t_global < self._VX_RAMP_S:
            vx *= t_global / self._VX_RAMP_S

        g._send_motion(vx=vx, wz=wz)

        # Slip detection
        slip_est = None
        if self._slip is not None:
            robot_state = g._sim.get_robot_state()
            if robot_state is not None:
                slip_est = self._slip.update(
                    robot_state, commanded_vx=vx, commanded_wz=wz)

        # Path critic: record at 10Hz
        if g._path_critic is not None and g._target_step_count % 10 == 0:
            g._path_critic.record(
                x, y, t=g._target_step_count * C.CONTROL_DT, v_cmd=vx)

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
            if g._nav_prev_pos is not None:
                _dx = x - g._nav_prev_pos[0]
                _dy = y - g._nav_prev_pos[1]
                _v = math.sqrt(_dx*_dx + _dy*_dy) / (C.TELEMETRY_INTERVAL * C.CONTROL_DT)
            else:
                _v = 0.0
            g._nav_prev_pos = (x, y)

            _lt = g._sim.telemetry if hasattr(g._sim, 'telemetry') else None
            if _lt is not None:
                _energy = _lt.body_roll**2 + _lt.body_pitch**2 + 0.01 * (_lt.d_roll**2 + _lt.d_pitch**2)
                _stability = max(0.0, 1.0 - 4.0 * _energy)
                _step_len = _lt.step_length
                _grip = _lt.traction
            else:
                _stability = 1.0
                _step_len = 0.0
                _grip = 1.0

            _t_se = max(0.0, min(1.0, (_step_len - 0.10) / 0.60))
            _stride_elegance = _t_se * _t_se

            _heading_to_target = math.atan2(target.y - y, target.x - x)
            _v_toward = _v * math.cos(yaw - _heading_to_target)
            _speed = max(0.0, min(1.0, _v_toward / 2.0))

            _prev_herr = getattr(self, '_prev_tick_heading_err', heading_err)
            _delta = abs(_prev_herr) - abs(heading_err)
            _turn = max(0.0, min(1.0, _delta / (C.TELEMETRY_INTERVAL * C.CONTROL_DT) / math.pi))
            self._prev_tick_heading_err = heading_err

            g._gt.record_tick(TickSample(
                step=g._step_count, t=t,
                target_index=g._target_index, num_targets=g._num_targets,
                mode="W", x=x, y=y, z=z,
                roll=r_deg, pitch=p_deg, yaw=yaw,
                dist=dist, heading_err=heading_err,
                vx_cmd=vx, wz_cmd=wz, v_actual=_v,
                traction=_lt.traction if _lt else (slip_est.traction if slip_est else None),
                droll=droll, dpitch=dpitch,
                stability=_stability, grip=_grip, speed=_speed,
                turn=_turn, stride_elegance=_stride_elegance,
            ))

    # ------------------------------------------------------------------
    # Wheeled robots via L5's send_wheel_command()
    # ------------------------------------------------------------------

    def tick_walk_wheeled(self):
        """Drive toward target using Layer 5's wheel torque API."""
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

        heading_err = target.heading_error(x, y, yaw)

        alignment = max(0.0, math.cos(heading_err))
        alignment = alignment * alignment
        dist_taper = min(1.0, dist / 1.0)
        fwd = C.WHEEL_FWD_TORQUE * alignment * dist_taper

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
            g._gt.record_tick(TickSample(
                step=g._step_count, t=t,
                target_index=g._target_index, num_targets=g._num_targets,
                mode="D", x=x, y=y, z=z,
                roll=r_deg, pitch=p_deg, yaw=yaw,
                dist=dist, heading_err=heading_err,
                vx_cmd=fwd, wz_cmd=turn, v_actual=0.0,
            ))

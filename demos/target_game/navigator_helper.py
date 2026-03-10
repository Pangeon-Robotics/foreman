"""Navigation helper producing MotionCommands for Layer 5.

Single walking mode: PD heading control steers directly at the target.
Green dot path is written to /tmp/dwa_best_arc.bin for the MuJoCo viewer.

Wheeled robots use L5's send_wheel_command() for PD leg hold +
differential wheel torque.
"""
from __future__ import annotations

import math
import struct

from . import game_config as C
from .telemetry import TickSample
from .utils import normalize_angle as _normalize_angle

# Slip detection (Layer 5 module, hardware-compatible)
try:
    from slip_detector import SlipDetector
except ImportError:
    SlipDetector = None

_PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"


def _write_path_dots(robot_x, robot_y, target_x, target_y, spacing=0.15):
    """Write a straight-line path from robot to target as green dots."""
    dx = target_x - robot_x
    dy = target_y - robot_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        return
    n = max(2, int(dist / spacing))
    buf = bytearray(n * 8)
    for i in range(n):
        t = i / (n - 1)
        px = robot_x + dx * t
        py = robot_y + dy * t
        struct.pack_into('ff', buf, i * 8, px, py)
    try:
        with open(_PATH_VIZ_FILE, 'wb') as f:
            f.write(buf)
    except OSError:
        pass


class Navigator:
    """PD heading controller for TargetGame.

    Single mode: steer directly at target.
      wz = Kp * heading_err + Kd * d(heading_err)/dt
      vx = VX_WALK * speed-turn coupling
    """

    # PD heading gains — tuned for dq_ff-enabled turning at 100Hz
    KP_HEADING = 1.5
    KD_HEADING = 0.3

    # Walk speed
    VX_WALK = 0.6       # m/s
    WZ_MAX = 1.0        # rad/s

    def __init__(self, game):
        self.game = game
        self._prev_heading_err = 0.0
        _masses = {'b2': 83.5, 'go2': 15.0, 'b2w': 83.5, 'go2w': 15.0}
        _robot = getattr(game, '_robot', 'b2')
        self._slip = SlipDetector(mass=_masses.get(_robot, 83.5)) if SlipDetector else None

    def write_initial_path(self):
        """Write path dots at target spawn time (before walking starts)."""
        g = self.game
        target = g._spawner.current_target
        if target is None:
            return
        nav_x, nav_y, _ = g._get_nav_pose()
        _write_path_dots(nav_x, nav_y, target.x, target.y)

    # ------------------------------------------------------------------
    # Main tick — legged robots via L5
    # ------------------------------------------------------------------

    def tick_walk_heading(self):
        """Steer directly at target using PD heading control.

        PD law: wz = Kp * heading_err + Kd * d(heading_err)/dt
        vx = VX_WALK * speed-turn coupling.
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

        # Direct heading to target
        heading_err = _normalize_angle(
            math.atan2(target.y - nav_y, target.x - nav_x) - nav_yaw)

        # D term: rate of change of heading error (damping)
        heading_err_rate = (heading_err - self._prev_heading_err) / C.CONTROL_DT
        self._prev_heading_err = heading_err

        # PD law
        wz_raw = self.KP_HEADING * heading_err + self.KD_HEADING * heading_err_rate
        wz = max(-self.WZ_MAX, min(self.WZ_MAX, wz_raw))

        # Speed-turn coupling: slow when turning, min 60% for effective stride
        alignment = max(0.0, math.cos(heading_err))
        vx = self.VX_WALK * (0.6 + 0.4 * alignment)

        # Update path dots every 50 ticks (0.5s)
        if g._target_step_count % 50 == 0:
            _write_path_dots(nav_x, nav_y, target.x, target.y)

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

            # Fitness components from L5 telemetry
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

            _heading_to_target = math.atan2(target.y - y_truth, target.x - x_truth)
            _v_toward = _v * math.cos(yaw_truth - _heading_to_target)
            _speed = max(0.0, min(1.0, _v_toward / 2.0))

            _prev_herr = getattr(self, '_prev_tick_heading_err', heading_err)
            _delta = abs(_prev_herr) - abs(heading_err)
            _turn = max(0.0, min(1.0, _delta / (C.TELEMETRY_INTERVAL * C.CONTROL_DT) / math.pi))
            self._prev_tick_heading_err = heading_err

            g._gt.record_tick(TickSample(
                step=g._step_count, t=t,
                target_index=g._target_index, num_targets=g._num_targets,
                mode="W", x=x_truth, y=y_truth, z=z,
                roll=r_deg, pitch=p_deg, yaw=yaw_truth,
                dist=dist, heading_err=heading_err,
                vx_cmd=vx, wz_cmd=wz, v_actual=_v,
                traction=_lt.traction if _lt else (slip_est.traction if slip_est else None),
                droll=droll, dpitch=dpitch,
                stability=_stability, grip=_grip, speed=_speed,
                turn=_turn, stride_elegance=_stride_elegance,
            ))

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
            g._gt.record_tick(TickSample(
                step=g._step_count, t=t,
                target_index=g._target_index, num_targets=g._num_targets,
                mode="D", x=x_truth, y=y_truth, z=z,
                roll=r_deg, pitch=p_deg, yaw=yaw_truth,
                dist=dist, heading_err=heading_err,
                vx_cmd=fwd, wz_cmd=turn, v_actual=0.0,
            ))

        if dist < g._reach_threshold:
            g.scoring.on_reached()
            return

        if g._target_step_count >= g._timeout_steps:
            g.scoring.on_timeout()

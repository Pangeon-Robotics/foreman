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
from .utils import normalize_angle as _normalize_angle, clamp as _clamp

# Slip detection (Layer 5 module, hardware-compatible)
try:
    from slip_detector import SlipDetector
except ImportError:
    SlipDetector = None

_PATH_VIZ_FILE = "/tmp/dwa_best_arc.bin"


def _resample_path(points, spacing=0.20):
    """Resample a polyline to evenly-spaced points (≥5 per meter)."""
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
        ux, uy = dx / seg, dy / seg
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
    """Write resampled (x,y) waypoints as green dots (≥5/m)."""
    pts = _resample_path(points)
    if len(pts) < 2:
        return
    buf = bytearray(len(pts) * 8)
    for i, (px, py) in enumerate(pts):
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

    # PD heading gains — v18 GA champion (4.817) with increased damping
    KP_HEADING = 4.8
    KD_HEADING = 0.5

    # Walk speed — v18 champion achieved 2.77 m/s at these settings
    VX_WALK = 2.0       # m/s — cruise speed far from target, heading aligned
    _APPROACH_DIST = 2.0  # m — distance taper zone for final approach deceleration

    # Stride-differential safety: limit wz so absolute stride diff stays bounded.
    # Combined with WZ_ABS_MAX to prevent excessive wz at low stride.
    # At full speed (stride=0.42m): wz ≈ 0.48 rad/s (wide arcs)
    # Near target (stride=0.10m): wz = 1.0 (tight convergence)
    _MAX_STRIDE_DIFF = 0.10  # m — max per-side stride asymmetry
    _STEP_LENGTH_SCALE = 0.2085  # mirror L5 config for wz limit calc
    _WZ_ABS_MAX = 1.0   # rad/s — hard cap regardless of stride

    # Startup ramp: seconds to reach full VX_WALK (avoids torque spike)
    _VX_RAMP_S = 2.0

    # Route following: advance to next waypoint when within this radius
    _WP_REACH = 0.5  # meters
    # Replan if robot drifts this far from the committed route
    _REPLAN_DRIFT = 1.5  # meters

    def __init__(self, game):
        self.game = game
        self._prev_heading_err = 0.0
        self._committed_path = None  # list of (x,y) waypoints
        self._wp_index = 0  # current waypoint we're steering toward
        self._vx_ema = 0.0  # EMA-smoothed commanded vx (prevents abrupt stride changes)
        _masses = {'b2': 83.5, 'go2': 15.0, 'b2w': 83.5, 'go2w': 15.0}
        _robot = getattr(game, '_robot', 'b2')
        self._slip = SlipDetector(mass=_masses.get(_robot, 83.5)) if SlipDetector else None

    def write_initial_path(self):
        """Plan and commit A* route at target spawn time."""
        g = self.game
        target = g._spawner.current_target
        if target is None:
            return
        nav_x, nav_y, _ = g._get_nav_pose()
        self._commit_route(nav_x, nav_y, target.x, target.y)

    def _plan_astar_path(self, sx, sy, gx, gy):
        """Run A* on the cost grid, return list of (x,y) or None."""
        g = self.game
        critic = g._path_critic
        if critic is None or getattr(critic, '_cost_grid', None) is None:
            return None
        from .astar import _astar_core
        path = _astar_core(
            critic._cost_grid, critic._cost_origin_x, critic._cost_origin_y,
            critic._cost_voxel_size, critic._robot_radius,
            (sx, sy), (gx, gy), return_path=True, force_passable=False,
            cost_truncation=getattr(critic, '_cost_truncation', 0.5))
        return path

    def _commit_route(self, sx, sy, gx, gy):
        """Plan A* path and commit it as the route to follow."""
        path = self._plan_astar_path(sx, sy, gx, gy)
        if path and len(path) >= 2:
            self._committed_path = path
        else:
            self._committed_path = [(sx, sy), (gx, gy)]
        self._wp_index = min(1, len(self._committed_path) - 1)
        _write_path_dots(self._committed_path)

    def _get_steer_target(self, nav_x, nav_y, target_x, target_y):
        """Get the point to steer toward from the committed route.

        Advances waypoint index when robot reaches current waypoint.
        Replans only if robot drifts too far from the route.
        Returns (steer_x, steer_y).
        """
        if self._committed_path is None or len(self._committed_path) < 2:
            return target_x, target_y

        # Advance past reached waypoints
        while self._wp_index < len(self._committed_path) - 1:
            wx, wy = self._committed_path[self._wp_index]
            d = math.sqrt((nav_x - wx)**2 + (nav_y - wy)**2)
            if d < self._WP_REACH:
                self._wp_index += 1
            else:
                break

        # Replan when costmap updates (new obstacle data from perception)
        g = self.game
        if getattr(g, '_costmap_changed', False):
            g._costmap_changed = False
            self._commit_route(nav_x, nav_y, target_x, target_y)
            return self._committed_path[self._wp_index] if self._wp_index < len(self._committed_path) else (target_x, target_y)

        # Check drift from nearest segment
        drift = self._dist_to_path(nav_x, nav_y)
        if drift > self._REPLAN_DRIFT:
            self._commit_route(nav_x, nav_y, target_x, target_y)

        # Steer toward current waypoint
        if self._wp_index < len(self._committed_path):
            return self._committed_path[self._wp_index]
        return target_x, target_y

    def _dist_to_path(self, x, y):
        """Minimum distance from (x,y) to the committed path polyline."""
        path = self._committed_path
        if not path or len(path) < 2:
            return 0.0
        best = float('inf')
        # Only check from current waypoint onward
        start = max(0, self._wp_index - 1)
        for i in range(start, min(start + 5, len(path) - 1)):
            ax, ay = path[i]
            bx, by = path[i + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-10:
                d2 = (x - ax)**2 + (y - ay)**2
            else:
                t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
                px, py = ax + t * dx, ay + t * dy
                d2 = (x - px)**2 + (y - py)**2
            if d2 < best:
                best = d2
        return math.sqrt(best)

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

        # Follow committed A* route waypoint-by-waypoint
        steer_x, steer_y = self._get_steer_target(
            nav_x, nav_y, target.x, target.y)
        heading_err = _normalize_angle(
            math.atan2(steer_y - nav_y, steer_x - nav_x) - nav_yaw)

        # D term: rate of change of heading error (damping)
        heading_err_rate = (heading_err - self._prev_heading_err) / C.CONTROL_DT
        self._prev_heading_err = heading_err

        # PD law
        wz_raw = self.KP_HEADING * heading_err + self.KD_HEADING * heading_err_rate

        # Speed = VX_WALK × dist_factor × heading_factor
        # Multiplicative coupling: smooth gradients, no feedback oscillation.
        # - dist_factor: decelerate near target for convergence
        # - heading_factor: slow when misaligned for turn authority
        #   cos(err) is flat near 0 → small heading wobbles don't kill speed
        dist_factor = min(1.0, dist / self._APPROACH_DIST)
        heading_factor = max(0.25, math.cos(heading_err))
        vx_target = self.VX_WALK * dist_factor * heading_factor

        # Asymmetric EMA: smooth stride transitions to prevent gait instability
        # - Accel: moderate (alpha=0.05, ~1s rise) — prevents stride-up shock
        # - Decel: faster (alpha=0.10, ~0.5s half) — responsive to turns/approach
        if vx_target < self._vx_ema:
            self._vx_ema += 0.10 * (vx_target - self._vx_ema)
        else:
            self._vx_ema += 0.05 * (vx_target - self._vx_ema)
        vx = self._vx_ema

        # Startup ramp: avoid torque spike from instant full stride at 4.37 Hz
        t_global = g._step_count * C.CONTROL_DT
        if t_global < self._VX_RAMP_S:
            vx *= t_global / self._VX_RAMP_S

        # Stride-differential-safe wz + absolute cap
        sl_est = max(vx * self._STEP_LENGTH_SCALE, 0.02)
        wz_safe = min(self._WZ_ABS_MAX, self._MAX_STRIDE_DIFF / (sl_est * 0.5))
        wz = _clamp(wz_raw, -wz_safe, wz_safe)

        # Update path dots from committed route (only the remaining portion)
        if g._target_step_count % 50 == 0 and self._committed_path:
            remaining = self._committed_path[self._wp_index:]
            if remaining:
                _write_path_dots(remaining)

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
            if g._nav_prev_pos is not None:
                _dx = x_truth - g._nav_prev_pos[0]
                _dy = y_truth - g._nav_prev_pos[1]
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

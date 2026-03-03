"""Robot-view A* path export for headed viewer rendering.

Standalone functions extracted from DWANavigatorMixin for path validation,
waypoint extraction, and path export to temp files.
"""
from __future__ import annotations

import struct


# Minimum ticks to hold a committed path before allowing replan.
# At 100Hz control, 500 ticks = 5 seconds.
PATH_HOLD_TICKS = 500

# Minimum ticks before first path export (wait for TSDF data).
# At 100Hz control, 100 ticks = 1 second.
PATH_WARMUP_TICKS = 100

# Diagnostic threshold: cells with cost >= this are counted as "obs".
VIZ_OBS_THRESHOLD = 177

# Force replan when robot strays this far from the committed path.
# Prevents green dots from appearing disconnected from the robot.
MAX_PATH_DEVIATION = 1.0  # meters


def validate_committed_path(committed_path, committed_path_step, step_count,
                            robot_x, robot_y, target_x, target_y):
    """Reuse committed path if it's recent and aimed at the right target.

    Returns trimmed path (with robot position prepended) if valid,
    None to trigger replan.  Forces replan when the robot has deviated
    > MAX_PATH_DEVIATION from the nearest path point.
    """
    path = committed_path
    if path is None or len(path) < 2:
        return None

    # Wrong target -- path endpoint must be within 2m of current target
    end = path[-1]
    if (end[0] - target_x)**2 + (end[1] - target_y)**2 > 4.0:
        return None

    # Time-based hold expired -- replan with fresh cost data
    if step_count - committed_path_step >= PATH_HOLD_TICKS:
        return None

    # Trim: drop points the robot has already passed
    best_i, best_d2 = 0, float('inf')
    for i, (px, py) in enumerate(path):
        d2 = (px - robot_x)**2 + (py - robot_y)**2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    # Robot has deviated too far from the committed path -- replan
    # so the green dots don't appear disconnected from the robot.
    if best_d2 > MAX_PATH_DEVIATION * MAX_PATH_DEVIATION:
        return None

    trimmed = path[best_i:]
    if len(trimmed) < 2:
        return None

    # Prepend robot position so dots always start at the robot
    return [(robot_x, robot_y)] + list(trimmed)


def waypoint_from_path(path, nav_x, nav_y, lookahead=2.0):
    """Extract waypoint at lookahead distance along cached path."""
    best_i, best_d2 = 0, float('inf')
    for i, (px, py) in enumerate(path):
        d2 = (px - nav_x)**2 + (py - nav_y)**2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    cumul = 0.0
    for i in range(best_i + 1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        cumul += (dx * dx + dy * dy) ** 0.5
        if cumul >= lookahead:
            return path[i]
    return path[-1] if len(path) > best_i else None


def export_path(game, target_x, target_y):
    """Write robot-view A* path to temp file for headed viewer rendering.

    Uses ground-truth position (not SLAM) because the cost grid obstacles
    are in GT world frame (from mj_multiRay).  Committed path is held for
    5 seconds to prevent green dot flickering from TSDF updates, but
    forced to replan if the robot deviates > MAX_PATH_DEVIATION.
    """
    # Don't plan until TSDF has scanned obstacles (first ~1s)
    if game._step_count < PATH_WARMUP_TICKS:
        return

    x_gt, y_gt, _, _, _, _ = game._get_robot_pose()
    if game._committed_path is None:
        print(f"  [EXPORT] REPLAN gt=({x_gt:.2f},{y_gt:.2f})", flush=True)

    # Try reusing committed path (trim using GT)
    path = validate_committed_path(
        game._committed_path, game._committed_path_step,
        game._step_count, x_gt, y_gt, target_x, target_y)

    # Replan if committed path is invalid or absent
    if path is None:
        game._committed_path = None
        raw = None
        _astar_mode = "no-pc"
        pc = game._path_critic
        if pc is not None and pc._cost_grid is not None:
            # Constrained A* with tight clearance (0.15m) --
            # blocks cells >= 177, so the path is obstacle-free.
            saved_radius = pc._robot_radius
            pc._robot_radius = 0.15
            raw = pc._astar_core(
                (x_gt, y_gt), (target_x, target_y),
                return_path=True, force_passable=False,
            )
            _astar_mode = "constrained"

            # Tight fallback: when robot is near obstacles (cost >= 177
            # at start), the 0.15m radius blocks surrounding cells and
            # A* can't expand.  Try 0.05m (blocks only >= 228, right
            # on the surface) so we still get a path.
            if raw is None or len(raw) < 2:
                pc._robot_radius = 0.05
                raw = pc._astar_core(
                    (x_gt, y_gt), (target_x, target_y),
                    return_path=True, force_passable=False,
                )
                _astar_mode = "constrained-tight"

            pc._robot_radius = saved_radius

        if raw is None or len(raw) < 2:
            raw = [(x_gt, y_gt)]
            _astar_mode = "straight"

        # Extend partial paths to the target with a straight line.
        _diag_thresh = (228 if _astar_mode == "constrained-tight"
                        else VIZ_OBS_THRESHOLD)
        if len(raw) >= 2:
            ex, ey = raw[-1]
            end_d = ((ex - target_x)**2 + (ey - target_y)**2)**0.5
            if end_d > 0.5:
                n_ext = max(1, int(end_d / 0.08))
                _ext_g = (pc._cost_grid if pc is not None else None)
                for i in range(1, n_ext + 1):
                    t = i / n_ext
                    px = ex + (target_x - ex) * t
                    py = ey + (target_y - ey) * t
                    # Check against cost grid -- stop at known obstacles
                    if _ext_g is not None:
                        gi = int((px - pc._cost_origin_x)
                                 / pc._cost_voxel_size)
                        gj = int((py - pc._cost_origin_y)
                                 / pc._cost_voxel_size)
                        if (0 <= gi < _ext_g.shape[0]
                                and 0 <= gj < _ext_g.shape[1]):
                            c = int(_ext_g[gi, gj])
                            if c != 255 and c >= _diag_thresh:
                                break
                    raw.append((px, py))

        # Diagnostic: count cells crossing known obstacles.
        _n_obs = 0
        _n_unk = 0
        _n_free = 0
        _max_obs_cost = 0
        if pc is not None and pc._cost_grid is not None and len(raw) >= 2:
            _g = pc._cost_grid
            _ox, _oy = pc._cost_origin_x, pc._cost_origin_y
            _vs = pc._cost_voxel_size
            for px, py in raw[1:]:  # skip start cell
                gi = int((px - _ox) / _vs)
                gj = int((py - _oy) / _vs)
                if 0 <= gi < _g.shape[0] and 0 <= gj < _g.shape[1]:
                    c = int(_g[gi, gj])
                    if c == 255:
                        _n_unk += 1
                    elif c >= _diag_thresh:
                        _n_obs += 1
                        _max_obs_cost = max(_max_obs_cost, c)
                    else:
                        _n_free += 1
        print(f"  [PATH] mode={_astar_mode} pts={len(raw)} "
              f"free={_n_free} obs={_n_obs}(max={_max_obs_cost}) "
              f"unk={_n_unk}", flush=True)

        # Smooth the A* grid path into natural curves (0.08m spacing)
        if len(raw) >= 3 and game._path_critic is not None:
            from .path_critic import PathCritic
            grid = game._path_critic._cost_grid
            raw = PathCritic.smooth_path(
                raw, grid, game._path_critic._cost_origin_x,
                game._path_critic._cost_origin_y,
                game._path_critic._cost_voxel_size,
                cost_threshold=_diag_thresh,
                spacing=0.08)

        path = [(x_gt, y_gt)] + list(raw)
        game._committed_path = list(path)
        game._committed_path_step = game._step_count

    # Ensure the first point tracks current GT position
    path[0] = (x_gt, y_gt)

    # Filter near-duplicate points (keep ~0.07m spacing)
    filtered = [path[0]]
    for i in range(1, len(path)):
        dx = path[i][0] - filtered[-1][0]
        dy = path[i][1] - filtered[-1][1]
        if dx * dx + dy * dy >= 0.005:
            filtered.append(path[i])
    if filtered[-1] != path[-1]:
        filtered.append(path[-1])

    n = len(filtered)
    buf = bytearray(n * 8)
    for i, (wx, wy) in enumerate(filtered):
        struct.pack_into('ff', buf, i * 8, wx, wy)
    try:
        with open(game._PATH_VIZ_FILE, 'wb') as f:
            f.write(buf)
    except OSError:
        pass

    # Stream to debug viewer
    if game._debug_server is not None:
        game._debug_server.send_path(filtered)

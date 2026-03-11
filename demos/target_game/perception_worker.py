"""Perception subprocess: LiDAR scan → TSDF → costmap in a separate process.

Eliminates GIL contention between perception (mj_multiRay + EDT) and
the 100Hz control loop. Communicates with the game process via:
  - multiprocessing.Array for pose input (6 doubles)
  - Temp files for costmap/TSDF output (proven IPC pattern)

Pipeline: robot LiDAR → robot TSDF → robot costmap → router
"""
from __future__ import annotations

import multiprocessing
import os
import struct
import time

import numpy as np


# Costmap file for both router and viewer
COSTMAP_FILE = "/tmp/robot_costmap_live.bin"
TSDF_VIZ_FILE = "/tmp/robot_view_tsdf.bin"
COSTMAP_VIZ_FILE = "/tmp/robot_view_costmap.bin"


def _write_costmap_file(cost_grid, meta, path=COSTMAP_FILE):
    """Write A* cost grid to temp file (atomic rename).

    Header: u16 nx, u16 ny, f32 origin_x, f32 origin_y, f32 voxel_size, f32 truncation
    Body: nx*ny uint8
    """
    nx, ny = cost_grid.shape
    hdr = struct.pack('<HHffff',
                      nx, ny,
                      meta['origin_x'], meta['origin_y'],
                      meta['voxel_size'],
                      meta.get('truncation', 0.5))
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(cost_grid).tobytes())
    os.replace(tmp, path)


def read_costmap_file(path=COSTMAP_FILE):
    """Read costmap from temp file. Returns (grid, meta) or None.

    Called by game process to get robot-view costmap for routing.
    """
    try:
        with open(path, 'rb') as f:
            hdr = f.read(20)
            if len(hdr) < 20:
                return None
            nx, ny, ox, oy, vs, trunc = struct.unpack('<HHffff', hdr)
            body = f.read(nx * ny)
            if len(body) < nx * ny:
                return None
    except (OSError, FileNotFoundError):
        return None

    grid = np.frombuffer(body, dtype=np.uint8).reshape(nx, ny).copy()
    meta = {
        'origin_x': ox, 'origin_y': oy,
        'voxel_size': vs, 'nx': nx, 'ny': ny,
        'truncation': trunc,
    }
    return grid, meta


def run_perception_loop(shared_pose, config_dict, scene_xml, robot,
                        shutdown_event, scan_hz=4.0):
    """Subprocess entry point: scan → TSDF → costmap loop.

    Parameters
    ----------
    shared_pose : multiprocessing.Array('d', 6)
        [x, y, z, yaw, roll, pitch] written by game process.
    config_dict : dict
        Perception config values (serializable).
    scene_xml : str
        Path to scene XML for MuJoCo model.
    robot : str
        Robot model name.
    shutdown_event : multiprocessing.Event
        Set by game process to stop the loop.
    scan_hz : float
        Target scan rate.
    """
    # Rebuild perception config
    from types import SimpleNamespace
    pcfg = SimpleNamespace(**config_dict)

    # Create TSDF
    from layer_6.world_model.tsdf import TSDF
    tsdf = TSDF(pcfg)

    # Create a minimal pipeline-like object for DirectScanner
    pipeline = SimpleNamespace(
        _tsdf=tsdf,
        _cfg=pcfg,
        _MIN_WORLD_Z=config_dict.get('min_world_z', 0.05),
        _direct_ready=False,
        _direct_scan_count=0,
        _use_reduced_rays=False,
        _imu_lock=__import__('threading').Lock(),
        _imu_x=0.0, _imu_y=0.0,
        _lock=__import__('threading').Lock(),
        _costmap_query=None,
        _world_cost_grid=None,
        _world_cost_meta=None,
        _dwa_cost_grid=None,
    )

    from .direct_scanner import init_direct_scanner, _cast_rays
    init_direct_scanner(pipeline, scene_xml, robot=robot)

    if not pipeline._direct_ready:
        print("[perception_worker] DirectScanner init failed, exiting")
        return

    from .costmap_builder import build_cost_grids
    from .game_viz import write_robot_tsdf_file, write_robot_costmap_file

    interval = 1.0 / scan_hz
    scan_count = 0

    print(f"[perception_worker] Started: {robot}, {scan_hz}Hz, "
          f"{pipeline._direct_n_rays} rays")

    while not shutdown_event.is_set():
        t0 = time.monotonic()

        # Read pose from shared memory
        x, y, z, yaw, roll, pitch = shared_pose[:]

        # Skip if pose hasn't been set yet (all zeros, robot at origin is ok
        # but z=0 means no pose written yet)
        if z < 0.1:
            time.sleep(0.05)
            continue

        # Cast rays and integrate into TSDF
        result = _cast_rays(pipeline, x, y, z, yaw, roll, pitch)
        if result is not None:
            hits, sx, sy = result
            tsdf.integrate_scan_world(hits, sx, sy)
            scan_count += 1

            # Build cost grids
            pipeline._imu_x = x
            pipeline._imu_y = y
            build_cost_grids(pipeline, tsdf, x, y)

            # Write costmap for router (game process reads this)
            if pipeline._world_cost_grid is not None:
                _write_costmap_file(
                    pipeline._world_cost_grid,
                    pipeline._world_cost_meta)

            # Write visualization files for headed viewer
            write_robot_tsdf_file(tsdf)
            write_robot_costmap_file(tsdf)

        # Sleep to maintain target rate
        elapsed = time.monotonic() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[perception_worker] Stopped after {scan_count} scans")


class PerceptionSubprocess:
    """Manages the perception subprocess from the game process side.

    Spawn with start(), update pose each tick, read costmap when needed.
    """

    def __init__(self, scene_xml: str, robot: str, config_dict: dict,
                 scan_hz: float = 4.0):
        self._scene_xml = scene_xml
        self._robot = robot
        self._config_dict = config_dict
        self._scan_hz = scan_hz

        # Shared pose: [x, y, z, yaw, roll, pitch]
        self._shared_pose = multiprocessing.Array('d', 6)
        self._shutdown = multiprocessing.Event()
        self._process = None

        # Costmap cache (game process side)
        self._costmap_mtime = 0.0
        self._cached_grid = None
        self._cached_meta = None

    def start(self):
        """Spawn the perception subprocess."""
        ctx = multiprocessing.get_context("spawn")
        self._shared_pose = ctx.Array('d', 6)
        self._shutdown = ctx.Event()
        self._process = ctx.Process(
            target=run_perception_loop,
            args=(self._shared_pose, self._config_dict,
                  self._scene_xml, self._robot,
                  self._shutdown, self._scan_hz),
            daemon=True,
        )
        self._process.start()
        print(f"[perception] Subprocess started (PID={self._process.pid})")

    def update_pose(self, x, y, z, yaw, roll, pitch):
        """Write robot pose to shared memory (called from game tick)."""
        self._shared_pose[0] = x
        self._shared_pose[1] = y
        self._shared_pose[2] = z
        self._shared_pose[3] = yaw
        self._shared_pose[4] = roll
        self._shared_pose[5] = pitch

    def read_costmap(self):
        """Read costmap from temp file if updated. Returns (grid, meta) or None.

        Uses mtime check to avoid re-reading unchanged data.
        """
        try:
            mt = os.path.getmtime(COSTMAP_FILE)
        except OSError:
            return None

        if mt == self._costmap_mtime:
            if self._cached_grid is not None:
                return self._cached_grid, self._cached_meta
            return None

        result = read_costmap_file()
        if result is not None:
            self._cached_grid, self._cached_meta = result
            self._costmap_mtime = mt
            return result
        return None

    def shutdown(self):
        """Stop the perception subprocess."""
        if self._process is not None and self._process.is_alive():
            self._shutdown.set()
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
            print("[perception] Subprocess stopped")

    @property
    def is_alive(self):
        return self._process is not None and self._process.is_alive()

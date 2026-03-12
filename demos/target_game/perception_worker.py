"""Perception subprocess: LiDAR scan → TSDF in a separate process.

Eliminates GIL contention between perception (mj_multiRay) and
the 100Hz control loop. Communicates with the game process via:
  - multiprocessing.Array for pose input (6 doubles)
  - Temp file for TSDF voxels (game process builds costmap from this)

Pipeline: robot LiDAR → robot TSDF → write TSDF voxel file
"""
from __future__ import annotations

import multiprocessing
import time


def run_perception_loop(shared_pose, config_dict, scene_xml, robot,
                        shutdown_event, scan_hz=10.0):
    """Subprocess entry point: scan → TSDF → write voxel file.

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
    from types import SimpleNamespace
    pcfg = SimpleNamespace(**config_dict)

    from layer_6.world_model.tsdf import TSDF
    tsdf = TSDF(pcfg)

    # Minimal pipeline-like object for DirectScanner
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
    )

    from .direct_scanner import init_direct_scanner, _cast_rays
    init_direct_scanner(pipeline, scene_xml, robot=robot)

    if not pipeline._direct_ready:
        print("[perception_worker] DirectScanner init failed, exiting")
        return

    from .game_viz import write_robot_tsdf_file

    interval = 1.0 / scan_hz
    scan_count = 0

    print(f"[perception_worker] Started: {robot}, {scan_hz}Hz, "
          f"{pipeline._direct_n_rays} rays")

    while not shutdown_event.is_set():
        t0 = time.monotonic()

        # Read pose from shared memory
        x, y, z, yaw, roll, pitch = shared_pose[:]

        # Skip if pose hasn't been set yet
        if z < 0.1:
            time.sleep(0.05)
            continue

        # Cast rays and integrate into TSDF
        result = _cast_rays(pipeline, x, y, z, yaw, roll, pitch)
        if result is not None:
            hits, sx, sy = result
            tsdf.integrate_scan_world(hits, sx, sy)
            scan_count += 1

            # Write TSDF voxels — game process reads this to build costmap
            write_robot_tsdf_file(tsdf)

        # Sleep to maintain target rate
        elapsed = time.monotonic() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[perception_worker] Stopped after {scan_count} scans")


class PerceptionSubprocess:
    """Manages the perception subprocess from the game process side.

    Spawn with start(), update pose each tick.
    Game process reads TSDF voxel file and builds costmap itself.
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

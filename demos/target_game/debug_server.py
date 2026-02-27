"""TCP debug server for streaming robot state to external viewers.

Streams binary messages over TCP to a connected Godot viewer (or any
client that speaks the length-prefixed protocol). Zero overhead when
no client is connected — all sends are no-ops.

Protocol: [u32 total_len][u8 msg_type][payload], little-endian.
"""
from __future__ import annotations

import socket
import struct
import threading

import numpy as np

# Message type IDs
MSG_ROBOT_STATE = 0x01
MSG_TSDF_OCCUPIED = 0x02
MSG_ASTAR_PATH = 0x03
MSG_OBSERVATION_MAP = 0x04
MSG_OBSTACLES = 0x05

# DDS→MuJoCo joint permutation (FR,FL,RR,RL → FL,FR,RL,RR)
_DDS_TO_MUJOCO = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


class DebugServer:
    """Non-blocking TCP server that streams debug data to one client.

    Usage::

        server = DebugServer(port=9877)
        server.start()
        # In tick loop:
        server.send_robot_state(...)
        server.send_tsdf(tsdf)
        # On shutdown:
        server.stop()
    """

    def __init__(self, port: int = 9877):
        self._port = port
        self._server_sock: socket.socket | None = None
        self._client: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the accept thread."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("127.0.0.1", self._port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"DebugServer listening on tcp://127.0.0.1:{self._port}")

    def stop(self) -> None:
        """Shut down server and close connections."""
        self._running = False
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _accept_loop(self) -> None:
        """Background thread: accept one client at a time."""
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # Blocking socket with short send timeout — prevents
            # sendall() from raising BlockingIOError (which was
            # killing the connection after every send).
            conn.setblocking(True)
            conn.settimeout(2.0)  # generous timeout — localhost sends are fast
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                self._client = conn
            print(f"DebugServer: viewer connected from {addr}")

    def _send(self, msg_type: int, payload: bytes) -> None:
        """Send a length-prefixed message. Drop on timeout."""
        with self._lock:
            if self._client is None:
                return
            total_len = 1 + len(payload)
            header = struct.pack('<IB', total_len, msg_type)
            try:
                self._client.sendall(header + payload)
            except socket.timeout:
                # Partial send corrupts length-prefix framing — must disconnect
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
            except (BrokenPipeError, ConnectionResetError, OSError):
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None

    @property
    def has_client(self) -> bool:
        with self._lock:
            return self._client is not None

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def send_robot_state(
        self,
        slam_x: float, slam_y: float, slam_yaw: float,
        z: float, roll: float, pitch: float,
        joints_12_dds: list | np.ndarray,
        target_x: float, target_y: float,
        game_state: int,
        dwa_fwd: float, dwa_turn: float,
        n_feasible: int,
        heading_err: float, dist: float,
        in_tip: bool,
    ) -> None:
        """Pack and send ROBOT_STATE (0x01). 100 bytes."""
        if not self.has_client:
            return
        # Permute joints from DDS order to MuJoCo body order
        j = [float(joints_12_dds[i]) for i in _DDS_TO_MUJOCO]
        payload = struct.pack(
            '<3f 3f 12f 2f B 2f H 2f B',
            slam_x, slam_y, slam_yaw,
            z, roll, pitch,
            *j,
            target_x, target_y,
            game_state,
            dwa_fwd, dwa_turn,
            min(n_feasible, 65535),
            heading_err, dist,
            1 if in_tip else 0,
        )
        self._send(MSG_ROBOT_STATE, payload)

    def send_tsdf(self, tsdf) -> None:
        """Pack and send TSDF_OCCUPIED (0x02). Vectorized sparse voxels."""
        if not self.has_client:
            return
        log_odds = tsdf._log_odds  # (nx, ny, nz) float32
        nx, ny, _nz = log_odds.shape

        # Find occupied voxels (log_odds > 0) — vectorized
        indices = np.argwhere(log_odds > 0)
        n_voxels = len(indices)

        header = struct.pack(
            '<4f 2H I',
            tsdf.origin_x, tsdf.origin_y, tsdf.z_min, tsdf.voxel_size,
            nx, ny, n_voxels,
        )

        if n_voxels == 0:
            self._send(MSG_TSDF_OCCUPIED, header)
            return

        # Fully vectorized packing (no Python loop)
        ix = indices[:, 0].astype(np.uint16)
        iy = indices[:, 1].astype(np.uint16)
        iz = indices[:, 2].astype(np.uint8)
        lo_vals = log_odds[indices[:, 0], indices[:, 1], indices[:, 2]]
        lo_q = np.clip(lo_vals * 25, -127, 127).astype(np.int8)

        # Pack as structured numpy array → bytes (no per-voxel loop)
        voxel_dtype = np.dtype([
            ('ix', '<u2'), ('iy', '<u2'), ('iz', 'u1'), ('lo', 'i1'),
        ])
        voxels = np.empty(n_voxels, dtype=voxel_dtype)
        voxels['ix'] = ix
        voxels['iy'] = iy
        voxels['iz'] = iz
        voxels['lo'] = lo_q

        self._send(MSG_TSDF_OCCUPIED, header + voxels.tobytes())

    def send_observation_map(self, tsdf) -> None:
        """Pack and send OBSERVATION_MAP (0x04). Packed bit array (legacy)."""
        if not self.has_client:
            return
        log_odds = tsdf._log_odds  # (nx, ny, nz) float32
        nx, ny = log_odds.shape[0], log_odds.shape[1]

        observed = np.any(log_odds != 0.0, axis=2)  # (nx, ny) bool
        flat = observed.flatten().astype(np.uint8)
        packed = np.packbits(flat)

        header = struct.pack(
            '<2H 3f',
            nx, ny,
            tsdf.origin_x, tsdf.origin_y, tsdf.voxel_size,
        )
        self._send(MSG_OBSERVATION_MAP, header + packed.tobytes())

    def send_costmap_2d(
        self,
        cost_grid: np.ndarray,
        origin_x: float,
        origin_y: float,
        voxel_size: float,
    ) -> None:
        """Pack and send cost map (0x04). Raw uint8 per cell.

        Same message type as observation map. The Godot client distinguishes
        by payload size: uint8 payload is nx*ny bytes (vs bit-packed which
        is ceil(nx*ny/8)).
        """
        if not self.has_client:
            return
        nx, ny = cost_grid.shape
        header = struct.pack(
            '<2H 3f',
            nx, ny,
            origin_x, origin_y, voxel_size,
        )
        # Remap (nx, ny) grid → Godot Image on FACE_Y quad.
        # Transpose: image columns = X, image rows = Y.
        # Flip Y: image row 0 (V=0) maps to max MuJoCo Y on the quad.
        # Verified by test_costmap_axes.py against Godot source.
        img = np.ascontiguousarray(np.flipud(cost_grid.T))
        self._send(MSG_OBSERVATION_MAP, header + img.tobytes())

    def send_path(self, path_points: list[tuple[float, float]]) -> None:
        """Pack and send ASTAR_PATH (0x03). World-frame waypoints."""
        if not self.has_client:
            return
        n = len(path_points)
        buf = struct.pack('<H', n)
        for x, y in path_points:
            buf += struct.pack('<ff', x, y)
        self._send(MSG_ASTAR_PATH, buf)

    def send_obstacles(self, obstacles: list[dict]) -> None:
        """Pack and send OBSTACLES (0x05). Ground-truth obstacle volumes.

        Each obstacle dict has: type ('box'|'cylinder'), pos (x,y,z), size (tuple).
        - box: size = (half_x, half_y, half_z)
        - cylinder: size = (radius, half_height)

        Sent once on connect so the Godot viewer can render wireframes
        for visual comparison against TSDF voxels.
        """
        if not self.has_client:
            return
        n = len(obstacles)
        buf = struct.pack('<H', n)
        for obs in obstacles:
            # type: 0=box, 1=cylinder
            otype = 0 if obs['type'] == 'box' else 1
            px, py, pz = obs['pos']
            sz = obs['size']
            if otype == 0:  # box: 3 half-extents
                buf += struct.pack('<B 3f 3f', otype, px, py, pz, sz[0], sz[1], sz[2])
            else:  # cylinder: radius + half-height
                buf += struct.pack('<B 3f 2f', otype, px, py, pz, sz[0], sz[1])
        self._send(MSG_OBSTACLES, buf)

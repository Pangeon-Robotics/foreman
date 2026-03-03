"""Ground-truth surface sampling and obstacle voxel parsing.

Provides GT surface point clouds and voxel sets for occupancy metrics.
Extracted from test_occupancy.py to keep files under 400 lines.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Chunk constants from TSDF
CHUNK_BITS = 4
CHUNK_SIZE = 1 << CHUNK_BITS  # 16


def materialize_log_odds(tsdf) -> np.ndarray:
    """Build dense (nx, ny, nz) log_odds array from TSDF chunks.

    Test-only helper — O(chunks) scatter into a preallocated dense array.
    """
    lo_dense = np.zeros((tsdf.nx, tsdf.ny, tsdf.nz), dtype=np.float32)
    for (cx, cy, cz), chunk in tsdf._chunks.items():
        gx = cx * CHUNK_SIZE
        gy = cy * CHUNK_SIZE
        gz = cz * CHUNK_SIZE
        sx = min(CHUNK_SIZE, tsdf.nx - gx)
        sy = min(CHUNK_SIZE, tsdf.ny - gy)
        sz = min(CHUNK_SIZE, tsdf.nz - gz)
        if sx > 0 and sy > 0 and sz > 0:
            lo_dense[gx:gx+sx, gy:gy+sy, gz:gz+sz] = chunk.log_odds[:sx, :sy, :sz]
    return lo_dense


# ------------------------------------------------------------------
# GT Surface Sampling
# ------------------------------------------------------------------

def sample_box_surface(pos: tuple, half_extents: tuple,
                       spacing: float) -> list[np.ndarray]:
    """Sample all 6 faces of an axis-aligned box at ~spacing resolution."""
    px, py, pz = pos
    hx, hy, hz = half_extents
    points = []

    nx = max(1, int(round(2 * hx / spacing)))
    ny = max(1, int(round(2 * hy / spacing)))
    nz = max(1, int(round(2 * hz / spacing)))

    xs = np.linspace(px - hx, px + hx, nx + 1)
    ys = np.linspace(py - hy, py + hy, ny + 1)
    zs = np.linspace(pz - hz, pz + hz, nz + 1)

    # +/- X faces
    for y in ys:
        for z in zs:
            points.append(np.array([px - hx, y, z], dtype=np.float32))
            points.append(np.array([px + hx, y, z], dtype=np.float32))
    # +/- Y faces
    for x in xs:
        for z in zs:
            points.append(np.array([x, py - hy, z], dtype=np.float32))
            points.append(np.array([x, py + hy, z], dtype=np.float32))
    # +/- Z faces
    for x in xs:
        for y in ys:
            points.append(np.array([x, y, pz - hz], dtype=np.float32))
            points.append(np.array([x, y, pz + hz], dtype=np.float32))

    return points


def sample_cylinder_surface(pos: tuple, size: tuple,
                             spacing: float) -> list[np.ndarray]:
    """Sample curved surface + top/bottom caps of a vertical cylinder."""
    px, py, pz = pos
    radius, half_h = size
    points = []

    circumference = 2 * math.pi * radius
    n_angle = max(8, int(round(circumference / spacing)))
    n_z = max(1, int(round(2 * half_h / spacing)))

    angles = np.linspace(0, 2 * math.pi, n_angle, endpoint=False)
    zs = np.linspace(pz - half_h, pz + half_h, n_z + 1)

    for theta in angles:
        cx = px + radius * math.cos(theta)
        cy = py + radius * math.sin(theta)
        for z in zs:
            points.append(np.array([cx, cy, z], dtype=np.float32))

    n_rings = max(1, int(round(radius / spacing)))
    for ring in range(n_rings + 1):
        r = radius * ring / max(n_rings, 1)
        if r < 1e-6:
            points.append(np.array([px, py, pz - half_h], dtype=np.float32))
            points.append(np.array([px, py, pz + half_h], dtype=np.float32))
        else:
            circ = 2 * math.pi * r
            n_pts = max(4, int(round(circ / spacing)))
            for theta in np.linspace(0, 2 * math.pi, n_pts, endpoint=False):
                cx = px + r * math.cos(theta)
                cy = py + r * math.sin(theta)
                points.append(
                    np.array([cx, cy, pz - half_h], dtype=np.float32))
                points.append(
                    np.array([cx, cy, pz + half_h], dtype=np.float32))

    return points


def sample_gt_surfaces(scene_xml_path: str,
                        sample_spacing: float = 0.01) -> np.ndarray:
    """Sample GT obstacle surfaces as a dense point cloud.

    Returns (N, 3) world-frame float32 points on obstacle surfaces.
    """
    from .scene_parser import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    points: list[np.ndarray] = []
    for obs in obstacles:
        if obs['type'] == 'box':
            points.extend(
                sample_box_surface(obs['pos'], obs['size'], sample_spacing))
        elif obs['type'] == 'cylinder':
            points.extend(
                sample_cylinder_surface(obs['pos'], obs['size'],
                                         sample_spacing))

    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points).astype(np.float32)


def extract_tsdf_surface(tsdf) -> np.ndarray:
    """Extract TSDF surface voxels as (N, 3) world-frame coordinates.

    Includes surface history so voxels confirmed during the run but later
    evicted still count toward metrics.
    """
    return tsdf.get_surface_voxels(include_history=True)


# ------------------------------------------------------------------
# Obstacle Voxel Parsing (solid and surface shell)
# ------------------------------------------------------------------

def parse_obstacle_voxels(scene_xml_path: str, tsdf) -> set[tuple[int, int, int]]:
    """Parse scene XML for obstacle geoms and compute solid occupied voxel indices.

    Returns ALL voxels inside each obstacle (full volume).
    """
    from .scene_parser import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    voxels = set()
    vs = tsdf.voxel_size
    margin = vs * 0.5

    for obs in obstacles:
        px, py, pz = obs['pos']

        if obs['type'] == 'box':
            hx, hy, hz = obs['size']
            x_lo = int(math.floor((px - hx - margin - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + hx + margin - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - hy - margin - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + hy + margin - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - hz - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + hz + margin - tsdf.z_min) / vs))
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                        voxels.add((ix, iy, iz))

        elif obs['type'] == 'cylinder':
            radius = obs['size'][0]
            half_h = obs['size'][1]
            eff_radius = radius + margin
            x_lo = int(math.floor((px - eff_radius - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + eff_radius - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - eff_radius - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + eff_radius - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - half_h - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + half_h + margin - tsdf.z_min) / vs))
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                wx = tsdf.origin_x + (ix + 0.5) * vs
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    wy = tsdf.origin_y + (iy + 0.5) * vs
                    dx = wx - px
                    dy = wy - py
                    if dx * dx + dy * dy <= eff_radius * eff_radius:
                        for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                            voxels.add((ix, iy, iz))

    return voxels


def parse_obstacle_surface_voxels(
    scene_xml_path: str, tsdf,
) -> set[tuple[int, int, int]]:
    """Parse scene XML and return the surface shell of each obstacle.

    Returns only boundary voxels — voxels inside the obstacle that have
    at least one face-neighbour outside.
    """
    from .scene_parser import _find_obstacle_geoms

    obstacles = _find_obstacle_geoms(Path(scene_xml_path))
    surface = set()
    vs = tsdf.voxel_size
    margin = vs * 0.5
    _NEIGHBOURS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))

    for obs in obstacles:
        px, py, pz = obs['pos']

        if obs['type'] == 'box':
            hx, hy, hz = obs['size']
            x_lo = int(math.floor((px - hx - margin - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + hx + margin - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - hy - margin - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + hy + margin - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - hz - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + hz + margin - tsdf.z_min) / vs))
            solid = set()
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                        solid.add((ix, iy, iz))
            for v in solid:
                ix, iy, iz = v
                for dx, dy, dz in _NEIGHBOURS:
                    if (ix+dx, iy+dy, iz+dz) not in solid:
                        surface.add(v)
                        break

        elif obs['type'] == 'cylinder':
            radius = obs['size'][0]
            half_h = obs['size'][1]
            eff_radius = radius + margin
            x_lo = int(math.floor((px - eff_radius - tsdf.origin_x) / vs))
            x_hi = int(math.ceil((px + eff_radius - tsdf.origin_x) / vs))
            y_lo = int(math.floor((py - eff_radius - tsdf.origin_y) / vs))
            y_hi = int(math.ceil((py + eff_radius - tsdf.origin_y) / vs))
            z_lo = int(math.floor((pz - half_h - margin - tsdf.z_min) / vs))
            z_hi = int(math.ceil((pz + half_h + margin - tsdf.z_min) / vs))
            solid = set()
            for ix in range(max(0, x_lo), min(tsdf.nx, x_hi + 1)):
                wx = tsdf.origin_x + (ix + 0.5) * vs
                for iy in range(max(0, y_lo), min(tsdf.ny, y_hi + 1)):
                    wy = tsdf.origin_y + (iy + 0.5) * vs
                    dx = wx - px
                    dy = wy - py
                    if dx * dx + dy * dy <= eff_radius * eff_radius:
                        for iz in range(max(0, z_lo), min(tsdf.nz, z_hi + 1)):
                            solid.add((ix, iy, iz))
            for v in solid:
                ix, iy, iz = v
                for dx, dy, dz in _NEIGHBOURS:
                    if (ix+dx, iy+dy, iz+dz) not in solid:
                        surface.add(v)
                        break

    return surface

"""Verification gate: TSDF voxel size and overlap.

Checks two properties of the TSDF → debug_server → Godot pipeline:
1. Voxel size reaching the renderer must be exactly 0.01m (1cm cubes)
2. No two rendered cubes may overlap (AABBs must not intersect)

Run from workspace root:
    python -m foreman.demos.target_game.test_tsdf_cubes
"""
import struct
import sys
from pathlib import Path

import numpy as np

# Ensure workspace root is on path
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def test_voxel_size_at_wire():
    """Check that voxel_size packed into the debug_server header is 0.01."""
    from layer_6.config.defaults import load_config
    from layer_6.world_model.tsdf import TSDF

    cfg = load_config('b2')
    print(f"Config tsdf_voxel_size = {cfg.tsdf_voxel_size}")

    tsdf = TSDF(cfg)
    print(f"TSDF.voxel_size       = {tsdf.voxel_size}")
    print(f"TSDF.nx={tsdf.nx}, ny={tsdf.ny}, nz={tsdf.nz}")

    # Integrate a small wall of test points so we have voxels to send
    wall_pts = []
    for x in np.arange(1.0, 1.5, 0.01):
        for z in np.arange(0.1, 0.5, 0.01):
            wall_pts.append([x, 2.0, z])
    wall_pts = np.array(wall_pts, dtype=np.float32)
    n_updated = tsdf.integrate_scan_world(wall_pts, 0.0, 0.0)
    print(f"Integrated {len(wall_pts)} points, {n_updated} voxels updated")

    # Simulate send_tsdf header packing (from debug_server.py)
    from layer_6.world_model.tsdf import CHUNK_SIZE, CHUNK_BITS

    all_ix, all_iy, all_iz, all_lo = [], [], [], []
    for (cx, cy, cz), chunk in list(tsdf._chunks.items()):
        lo = chunk.log_odds
        occ = lo > 1.0
        if not np.any(occ):
            continue
        local_indices = np.argwhere(occ)
        gx = (cx << CHUNK_BITS) + local_indices[:, 0]
        gy = (cy << CHUNK_BITS) + local_indices[:, 1]
        gz = (cz << CHUNK_BITS) + local_indices[:, 2]
        lo_vals = lo[local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]]
        all_ix.append(gx)
        all_iy.append(gy)
        all_iz.append(gz)
        all_lo.append(lo_vals)

    if all_ix:
        ix = np.concatenate(all_ix).astype(np.uint16)
        iy = np.concatenate(all_iy).astype(np.uint16)
        iz = np.concatenate(all_iz).astype(np.uint8)
        lo_vals = np.concatenate(all_lo)
        n_voxels = len(ix)
    else:
        n_voxels = 0
        ix = iy = iz = lo_vals = np.array([])

    print(f"Occupied voxels to send: {n_voxels}")

    # Pack header exactly as debug_server.py does
    header = struct.pack(
        '<4f 2H I',
        tsdf.origin_x, tsdf.origin_y, tsdf.z_min, tsdf.voxel_size,
        tsdf.nx, tsdf.ny, n_voxels,
    )

    # Decode header exactly as tsdf_renderer.gd does
    ofs = 0
    wire_origin_x = struct.unpack_from('<f', header, ofs)[0]; ofs += 4
    wire_origin_y = struct.unpack_from('<f', header, ofs)[0]; ofs += 4
    wire_z_min = struct.unpack_from('<f', header, ofs)[0]; ofs += 4
    wire_voxel_size = struct.unpack_from('<f', header, ofs)[0]; ofs += 4
    wire_nx = struct.unpack_from('<H', header, ofs)[0]; ofs += 2
    wire_ny = struct.unpack_from('<H', header, ofs)[0]; ofs += 2
    wire_n_voxels = struct.unpack_from('<I', header, ofs)[0]; ofs += 4

    print(f"\n--- Wire header (as Godot receives) ---")
    print(f"  origin_x    = {wire_origin_x}")
    print(f"  origin_y    = {wire_origin_y}")
    print(f"  z_min       = {wire_z_min}")
    print(f"  voxel_size  = {wire_voxel_size}")
    print(f"  nx          = {wire_nx}")
    print(f"  ny          = {wire_ny}")
    print(f"  n_voxels    = {wire_n_voxels}")

    # GATE 1: voxel_size must be 0.01
    if abs(wire_voxel_size - 0.01) > 1e-6:
        print(f"\n  FAIL: voxel_size on wire is {wire_voxel_size}, expected 0.01")
        print(f"        Cubes will render at {wire_voxel_size*100:.1f}cm instead of 1cm")
        return False
    else:
        print(f"\n  PASS: voxel_size on wire is {wire_voxel_size} (1cm)")
        return True


def test_no_cube_overlap():
    """Check that no two voxel cubes overlap when decoded as Godot would."""
    from layer_6.config.defaults import load_config
    from layer_6.world_model.tsdf import TSDF, CHUNK_BITS

    cfg = load_config('b2')
    tsdf = TSDF(cfg)

    # Integrate a dense wall — likely to produce adjacent voxels
    wall_pts = []
    for x in np.arange(1.0, 1.2, 0.005):  # dense, sub-voxel spacing
        for z in np.arange(0.1, 0.3, 0.005):
            wall_pts.append([x, 2.0, z])
    wall_pts = np.array(wall_pts, dtype=np.float32)

    # Multiple integrations to build up log-odds > 1.0
    for _ in range(5):
        tsdf.integrate_scan_world(wall_pts, 0.0, 0.0)

    # Extract voxel positions exactly as Godot would compute them
    vs = tsdf.voxel_size
    half_vs = vs * 0.5
    ox = tsdf.origin_x
    oy = tsdf.origin_y
    z_min = tsdf.z_min

    positions = []  # (wx, wy, wz) as Godot computes
    indices = []    # (ix, iy, iz) raw grid indices

    for (cx, cy, cz), chunk in list(tsdf._chunks.items()):
        lo = chunk.log_odds
        occ = lo > 1.0
        if not np.any(occ):
            continue
        local_idx = np.argwhere(occ)
        for li in local_idx:
            gx = (cx << CHUNK_BITS) + li[0]
            gy = (cy << CHUNK_BITS) + li[1]
            gz = (cz << CHUNK_BITS) + li[2]
            # Clamp to wire format (uint16/uint8)
            gx_u16 = int(gx) & 0xFFFF
            gy_u16 = int(gy) & 0xFFFF
            gz_u8 = int(gz) & 0xFF

            # Godot position computation (tsdf_renderer.gd lines 66-68)
            wx = ox + gx_u16 * vs + half_vs
            wy = oy + gy_u16 * vs + half_vs
            wz = z_min + gz_u8 * vs + half_vs

            positions.append((wx, wy, wz))
            indices.append((gx_u16, gy_u16, gz_u8))

    n = len(positions)
    print(f"\nOverlap test: {n} occupied voxels")

    if n == 0:
        print("  SKIP: no voxels to test")
        return True

    # Check for duplicate indices (same grid cell claimed by multiple chunks)
    idx_set = set()
    duplicates = 0
    for idx in indices:
        if idx in idx_set:
            duplicates += 1
        idx_set.add(idx)

    if duplicates > 0:
        print(f"  FAIL: {duplicates} duplicate grid indices (same cell in multiple chunks)")

    # Check AABB overlap: two cubes overlap if they share volume.
    # Cubes are axis-aligned, centered at (wx, wy, wz), size vs in each axis.
    # Two cubes overlap iff |wx1-wx2| < vs AND |wy1-wy2| < vs AND |wz1-wz2| < vs.
    # (strict < because touching at boundary is OK)
    #
    # For efficiency, only check between voxels that could possibly overlap
    # (within 1 grid cell in all 3 axes).
    positions_arr = np.array(positions, dtype=np.float64)
    indices_arr = np.array(indices, dtype=np.int32)

    overlaps = 0
    overlap_examples = []

    # Check all pairs within 1 grid step (brute force is fine for <10k voxels)
    if n < 50000:
        for i in range(n):
            for j in range(i + 1, n):
                di = abs(indices_arr[i, 0] - indices_arr[j, 0])
                dj = abs(indices_arr[i, 1] - indices_arr[j, 1])
                dk = abs(indices_arr[i, 2] - indices_arr[j, 2])
                # Only adjacent cells can potentially overlap
                if di > 1 or dj > 1 or dk > 1:
                    continue
                # Check actual AABB overlap
                dx = abs(positions_arr[i, 0] - positions_arr[j, 0])
                dy = abs(positions_arr[i, 1] - positions_arr[j, 1])
                dz = abs(positions_arr[i, 2] - positions_arr[j, 2])
                if dx < vs - 1e-9 and dy < vs - 1e-9 and dz < vs - 1e-9:
                    overlaps += 1
                    if len(overlap_examples) < 3:
                        overlap_examples.append((i, j, dx, dy, dz))
    else:
        print(f"  SKIP: too many voxels ({n}) for brute-force overlap check")
        return True

    if overlaps > 0:
        print(f"  FAIL: {overlaps} overlapping cube pairs")
        for i, j, dx, dy, dz in overlap_examples:
            print(f"    voxel {indices[i]} vs {indices[j]}: "
                  f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f} (vs={vs})")
        return False
    else:
        print(f"  PASS: 0 overlapping cubes among {n} voxels")

    # Also verify grid spacing: adjacent voxels should be exactly vs apart
    spacing_errors = 0
    for i in range(n):
        for j in range(i + 1, n):
            di = indices_arr[i] - indices_arr[j]
            if np.sum(np.abs(di)) == 1:  # exactly 1 axis differs by 1
                actual_dist = np.linalg.norm(positions_arr[i] - positions_arr[j])
                expected_dist = vs
                if abs(actual_dist - expected_dist) > 1e-9:
                    spacing_errors += 1
                    if spacing_errors <= 3:
                        print(f"    spacing error: {indices[i]} vs {indices[j]}, "
                              f"dist={actual_dist:.9f}, expected={expected_dist:.9f}")

    if spacing_errors > 0:
        print(f"  FAIL: {spacing_errors} adjacent voxel pairs with wrong spacing")
        return False
    else:
        print(f"  PASS: all adjacent voxels correctly spaced at {vs}m")
        return True


def test_lattice_alignment():
    """Check that all voxel positions land on the 1cm lattice."""
    from layer_6.config.defaults import load_config
    from layer_6.world_model.tsdf import TSDF, CHUNK_BITS

    cfg = load_config('b2')
    tsdf = TSDF(cfg)
    vs = tsdf.voxel_size

    # Check origin alignment
    origin_x_cells = tsdf.origin_x / vs
    origin_y_cells = tsdf.origin_y / vs
    z_min_cells = tsdf.z_min / vs

    print(f"\nLattice alignment test:")
    print(f"  origin_x / vs = {origin_x_cells} (should be integer)")
    print(f"  origin_y / vs = {origin_y_cells} (should be integer)")
    print(f"  z_min / vs    = {z_min_cells} (should be integer)")

    ok = True
    if abs(origin_x_cells - round(origin_x_cells)) > 1e-9:
        print(f"  FAIL: origin_x not on lattice")
        ok = False
    if abs(origin_y_cells - round(origin_y_cells)) > 1e-9:
        print(f"  FAIL: origin_y not on lattice")
        ok = False
    if abs(z_min_cells - round(z_min_cells)) > 1e-9:
        print(f"  FAIL: z_min not on lattice")
        ok = False

    # Check that voxel centers are at origin + (i + 0.5) * vs
    # i.e., they're centered in each 1cm cell
    test_ix = 500
    wx = tsdf.origin_x + test_ix * vs + vs * 0.5
    expected_on_half_cm = (wx / vs) % 1.0  # should be 0.5
    print(f"  voxel center at ix={test_ix}: wx={wx:.6f}, "
          f"fractional cell = {expected_on_half_cm:.6f} (should be 0.5)")

    if abs(expected_on_half_cm - 0.5) > 1e-9:
        print(f"  FAIL: voxel center not at cell midpoint")
        ok = False

    if ok:
        print(f"  PASS: all origins on 1cm lattice")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("TSDF Cube Verification Gate")
    print("=" * 60)

    r1 = test_voxel_size_at_wire()
    r2 = test_no_cube_overlap()
    r3 = test_lattice_alignment()

    print("\n" + "=" * 60)
    if r1 and r2 and r3:
        print("ALL GATES PASSED")
    else:
        failed = []
        if not r1: failed.append("voxel_size")
        if not r2: failed.append("cube_overlap")
        if not r3: failed.append("lattice_alignment")
        print(f"FAILED: {', '.join(failed)}")
    print("=" * 60)

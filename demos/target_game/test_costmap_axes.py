#!/usr/bin/env python3
"""Calibration test for cost map → Godot Image axis mapping.

Verifies that send_costmap_2d produces bytes where image pixel (col, row)
contains cost_grid[col, ny-1-row], matching Godot's FACE_Y QuadMesh UV:

    UV(0,0) → local (-w/2, 0, -h/2) → world Z_min → MuJoCo Y_max
    UV(1,1) → local (+w/2, 0, +h/2) → world Z_max → MuJoCo Y_min

    Image row 0 (V=0) → max MuJoCo Y → grid iy = ny-1
    Image row ny-1 (V=1) → min MuJoCo Y → grid iy = 0

Usage:
    python -m demos.target_game.test_costmap_axes
    # or from foreman/:
    python demos/target_game/test_costmap_axes.py
"""
import numpy as np
import struct
import sys


def apply_axis_transform(cost_grid: np.ndarray) -> bytes:
    """Apply the same transform as send_costmap_2d: transpose + Y-flip.

    Columns = X axis, rows = Y axis (flipped so row 0 = max Y).
    """
    img = np.ascontiguousarray(np.flipud(cost_grid.T))
    return img.tobytes()


def test_corner_markers():
    """Stamp unique values at 4 corners, verify they land at correct pixels."""
    nx, ny = 200, 200

    grid = np.zeros((nx, ny), dtype=np.uint8)
    # Stamp corners with unique values
    grid[0, 0] = 10        # min-X, min-Y  (world: -10, -10)
    grid[nx-1, 0] = 20     # max-X, min-Y  (world: +10, -10)
    grid[0, ny-1] = 30     # min-X, max-Y  (world: -10, +10)
    grid[nx-1, ny-1] = 40  # max-X, max-Y  (world: +10, +10)

    payload = apply_axis_transform(grid)

    # Image is width=nx, height=ny. Pixel at (col, row) is byte[row*nx + col].
    def px(col, row):
        return payload[row * nx + col]

    # Expected mapping: pixel (col, row) = cost_grid[col, ny-1-row]
    #
    # min-X, max-Y corner → grid[0, 199] = 30
    #   Should be at image pixel col=0, row=0 (top-left = max Y, min X)
    assert px(0, 0) == 30, f"Top-left pixel: expected 30, got {px(0, 0)}"

    # max-X, max-Y corner → grid[199, 199] = 40
    #   Should be at image pixel col=199, row=0 (top-right = max Y, max X)
    assert px(nx-1, 0) == 40, f"Top-right pixel: expected 40, got {px(nx-1, 0)}"

    # min-X, min-Y corner → grid[0, 0] = 10
    #   Should be at image pixel col=0, row=199 (bottom-left = min Y, min X)
    assert px(0, ny-1) == 10, f"Bottom-left pixel: expected 10, got {px(0, ny-1)}"

    # max-X, min-Y corner → grid[199, 0] = 20
    #   Should be at image pixel col=199, row=199 (bottom-right = min Y, max X)
    assert px(nx-1, ny-1) == 20, f"Bottom-right pixel: expected 20, got {px(nx-1, ny-1)}"

    print("PASS: corner markers at correct image pixels")


def test_axis_bars():
    """Stamp bars along +X and +Y edges, verify orientation."""
    nx, ny = 200, 200

    grid = np.zeros((nx, ny), dtype=np.uint8)
    grid[195:, :] = 200    # +X edge (bright bar along max-X)
    grid[:, 195:] = 128    # +Y edge (medium bar along max-Y)
    grid[195:, 195:] = 254  # +X,+Y corner (lethal)

    payload = apply_axis_transform(grid)

    def px(col, row):
        return payload[row * nx + col]

    # +X edge bar: grid[195:, :] = 200
    # Should appear at image cols 195-199, all rows
    assert px(197, 100) == 200, f"+X bar at col=197: expected 200, got {px(197, 100)}"
    assert px(100, 100) == 0, f"Center should be 0, got {px(100, 100)}"

    # +Y edge bar: grid[:, 195:] = 128
    # max-Y → image row 0..4 (top rows, since Y is flipped)
    assert px(100, 2) == 128, f"+Y bar at row=2: expected 128, got {px(100, 2)}"
    assert px(100, 100) == 0, f"Center should be 0, got {px(100, 100)}"

    # Corner: grid[195:, 195:] = 254
    # → image pixel (col=197, row=2): +X col, +Y row (top-right area)
    assert px(197, 2) == 254, f"Corner at (197,2): expected 254, got {px(197, 2)}"

    print("PASS: axis bars at correct image positions")


def test_asymmetric_grid():
    """Test with non-square grid to catch axis swaps."""
    nx, ny = 150, 200  # wider in Y than X

    grid = np.zeros((nx, ny), dtype=np.uint8)
    grid[0, :] = 100    # min-X bar
    grid[:, ny-1] = 200  # max-Y bar

    payload = apply_axis_transform(grid)
    # Image: width=nx=150, height=ny=200

    def px(col, row):
        return payload[row * nx + col]

    # min-X bar → col=0, all rows
    assert px(0, 100) == 100, f"min-X bar: expected 100, got {px(0, 100)}"
    # max-Y bar → row=0 (top row, since Y flipped)
    assert px(75, 0) == 200, f"max-Y bar at row=0: expected 200, got {px(75, 0)}"
    # min-X + max-Y overlap
    assert px(0, 0) == 200, f"Overlap at (0,0): expected 200 (max-Y overwrites), got {px(0, 0)}"

    print("PASS: asymmetric grid axes correct")


def test_payload_size():
    """Verify payload format matches what Godot expects for format detection."""
    nx, ny = 200, 200
    grid = np.zeros((nx, ny), dtype=np.uint8)

    payload = apply_axis_transform(grid)
    header_size = struct.calcsize('<2H 3f')  # 16 bytes

    assert len(payload) == nx * ny, f"Payload size: {len(payload)} != {nx*ny}"
    # Godot format detection: payload_size >= n_cells → uint8 cost map
    assert len(payload) >= nx * ny, "Payload too small for uint8 format detection"

    print(f"PASS: payload {len(payload)} bytes, header {header_size} bytes")


def save_reference_image():
    """Save a PNG showing what Godot should render, for visual verification.

    Requires matplotlib. Skips gracefully if not available.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrow
    except ImportError:
        print("SKIP: matplotlib not available for reference image")
        return

    nx, ny = 200, 200
    vs = 0.1
    origin_x, origin_y = -10.0, -10.0

    # Create test pattern
    grid = np.zeros((nx, ny), dtype=np.uint8)
    grid[190:, :] = 200    # +X edge (bright)
    grid[:, 190:] = 128    # +Y edge (medium)
    grid[190:, 190:] = 254  # corner
    grid[:10, :] = 80       # -X edge (dim)

    # Apply transform → what Godot Image receives
    payload = apply_axis_transform(grid)
    img = np.frombuffer(payload, dtype=np.uint8).reshape(ny, nx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: raw cost grid in world coords
    extent = [origin_x, origin_x + nx*vs, origin_y, origin_y + ny*vs]
    axes[0].imshow(grid.T, origin='lower', extent=extent, cmap='Reds', vmin=0, vmax=254)
    axes[0].set_title('Cost Grid (world frame)\nX→right, Y→up')
    axes[0].set_xlabel('World X')
    axes[0].set_ylabel('World Y')
    axes[0].axhline(0, color='grey', lw=0.5)
    axes[0].axvline(0, color='grey', lw=0.5)

    # Right: Godot image (as sent)
    # V=0 (top) = max Y, V=1 (bottom) = min Y
    axes[1].imshow(img, origin='upper', cmap='Reds', vmin=0, vmax=254)
    axes[1].set_title('Godot Image (as received)\nrow 0 = max Y (V=0)')
    axes[1].set_xlabel(f'Column (→ world X, {nx} px)')
    axes[1].set_ylabel(f'Row (↓ = decreasing Y, {ny} px)')

    out = '/tmp/costmap_calibration.png'
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"SAVED: reference image → {out}")


if __name__ == '__main__':
    test_corner_markers()
    test_axis_bars()
    test_asymmetric_grid()
    test_payload_size()
    save_reference_image()
    print("\nAll calibration tests passed.")

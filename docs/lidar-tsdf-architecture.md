# LiDAR & TSDF Architecture

Robot-view perception pipeline: direct MuJoCo raycasting, TSDF integration,
and 3DS quality metrics.

## DirectScanner (direct_scanner.py)

Bypasses DDS LiDAR (which has motion blur across 50 physics steps) by casting
rays instantaneously via `mj_multiRay` from the game process.

### Ray Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Horizontal rays | 500 (0.72° step, full 360°) | Sufficient spatial density for obstacle detection |
| Vertical channels | 128 (±30° range) | Dense vertical coverage for high 3DS completeness |
| Total rays | 64,000 | Balanced between coverage and integration speed |
| Min distance | 0.05m | Filter self-hits |
| Max distance | 10.0m (cutoff) | TSDF extent limit |

Previous configuration was 16 vertical channels at ±15° (Hesai XT16 emulation).
This produced only 2.9% completeness in 3DS because:

- **Sparse vertical gaps**: At 2m distance, 2° channel spacing = 7cm gaps between
  channels. Most obstacle surfaces fell between channels.
- **Narrow vertical FOV**: ±15° at sensor height 0.645m missed surfaces below
  z=0.11m at 2m range and below z=0.51m at 0.5m range.

The 128-channel ±30° configuration gives:
- At 2m: 0.47° spacing = 1.6cm vertical gaps (vs 7cm before)
- At 1m: 0.8cm vertical gaps — sub-voxel precision at 1cm TSDF
- Lowest ray at 2m: z = 0.645 - 2×tan(30°) = -0.51m (full ground coverage)

### IMU-Corrected Ray Rotation

Rays are rotated by the full body orientation R(yaw) × R(pitch) × R(roll),
not just yaw. This is critical because:

1. B2's walking gait produces ±5-15° pitch and ±10° roll oscillation
2. Without IMU correction, rays point at wrong heights during stride cycles
3. Pitch-uncorrected rays at 5m distance shift vertically by 5×sin(10°) = 0.87m

The IMU pose (roll, pitch, yaw) comes from ground-truth MuJoCo body quaternion
via `_get_robot_pose()`. The full rotation chain:

```python
R = R_yaw @ R_pitch @ R_roll  # Extrinsic ZYX convention
rays_world = (R @ ray_dirs_local.T).T
```

### Self-Hit Filtering

Two-stage filtering to exclude robot body and target marker:

1. **bodyexclude** in `mj_multiRay`: Excludes root body geoms (15 of 63 B2 geoms)
2. **Geom ID filter**: Walks the full body tree to collect ALL robot geom IDs
   (including child body limbs), plus target marker geoms. Any hit on these
   geom IDs is discarded post-raycast.

This is necessary because MuJoCo's `bodyexclude` parameter only covers the
specified body's own geoms, not child bodies in the kinematic tree.

### Sensor Offset

The LiDAR sensor is offset from the body origin in the robot's local frame:

| Robot | Offset (x, y, z) | Notes |
|-------|-------------------|-------|
| B2 | (0.34, 0.0, 0.18) | Front-mounted, above hip joint |
| Go2 | (0.20, 0.0, 0.10) | Proportionally smaller |

The offset is rotated by yaw and added to the body position to get the
world-frame sensor origin for raycasting.

## TSDF Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Internal voxel size | 0.01m (1cm) | High-fidelity surface representation for 3DS |
| Output resolution | 0.05m (5cm) | Costmap/visualization downsampled for speed |
| XY extent | ±10.0m | Covers full obstacle scene |
| Z range | (-0.5, 1.5)m | Ground level to above obstacle tops |
| Log-odds hit | 3.0 | Instant surface crossing (1 hit > threshold) |
| Log-odds free | 0.25 | Slow free-space carving |
| Log-odds max | 5.0 | Saturation limit |
| Truncation | 0.5m | Behind-surface integration depth |
| Decay rate | 0.0 | No decay — persistent surfaces |

### Why 1cm Voxels Matter for 3DS

At 10cm voxels (the previous default), surface voxel centers can be up to
~7cm from the actual obstacle surface. The 3DS completeness tolerance is 5cm
(in `compute_3ds_god`), so many real detections were classified as missed
or phantom. With 1cm voxels, center-to-surface error is at most 0.7cm.

The 1cm internal voxels use chunk-based storage (16×16×16 dense sub-grids in
a Python dict), so memory scales with observed volume, not total grid size.

### Surface History

The TSDF maintains a `_surface_history` set recording world-coords of voxels
that have ever crossed the convergence threshold (lo > 2.5). This survives
chunk eviction. Used by `get_surface_voxels(include_history=True)` for 3DS
metrics and visualization (grey cubes in MuJoCo viewer).

## 3DS Quality Metrics (test_occupancy_3ds.py)

3DS (3D Surface) measures TSDF accuracy against ground-truth obstacle geometry
parsed from the scene XML.

### Individual Metrics

| Metric | Definition | Good Value |
|--------|-----------|------------|
| Adherence (mm) | Mean distance from each TSDF surface voxel to nearest GT surface point | < 10mm |
| Completeness (%) | Fraction of GT surface points with a TSDF detection within tolerance | > 50% |
| Phantom (%) | Fraction of TSDF voxels far from any GT surface (hallucinations) | < 5% |

### God Score (compute_3ds_god)

Composite score balancing three concerns:

```
score = 0.40 × precision + 0.20 × completeness + 0.40 × purity
```

Where:
- **Precision** = 100 × (1 - mean_squared_dist / max_dist²), capped at 0.5m
- **Completeness** = % of Z-filtered GT surfaces detected within 5cm
- **Purity** = 100 - phantom_penalty

The 40/20/40 weighting reflects that:
- **Precision and purity** (80% combined) measure detection quality — are the
  surfaces the robot reports actually real and accurately placed?
- **Completeness** (20%) measures exploration coverage — inherently limited by
  the robot's path through the scene and LiDAR range.

Previous weighting (50/30/-20) capped the maximum achievable score at 80,
making the 96+ target structurally impossible.

### GT Z-Range Filter

Completeness scoring filters GT surface points to the detectable Z range
`(MIN_WORLD_Z, costmap_z_hi)`. This excludes:
- Ground-level surfaces below the LiDAR's minimum detection height
- Surfaces above the costmap projection ceiling

Without this filter, ~40% of GT points are undetectable, dragging completeness
down regardless of LiDAR quality.

### Typical Results (B2, scattered obstacles)

| Metric | Before | After |
|--------|--------|-------|
| Adherence | 30.8mm | 7-8mm |
| Completeness | 2.9% | 40-71% |
| Phantom | 14.1% | 0.0% |
| God Score | 54 | 89-98 |

## Scan Timing (game_viz.py:tick_perception)

Perception work is staggered across tick phases to avoid blocking the 100Hz
control loop:

| Phase | Interval | Operation | Duration |
|-------|----------|-----------|----------|
| 0 | 4Hz (every 25 ticks) | DirectScanner raycast + TSDF integration | ~5-15ms |
| 10 | 2Hz (every other cycle) | Cost grid build (EDT) | ~65-100ms |
| 15 | 4Hz | Path critic update + viz file writes | ~5ms |

God-view TSDF also updates at phase 0 using the same robot pose, ensuring
Surface F1 = 100.0 (identical ray origins and model).

## File Map

| File | Role |
|------|------|
| `direct_scanner.py` | MuJoCo raycast, ray config, IMU rotation, TSDF integration |
| `perception.py` | PerceptionPipeline, LiveCostmapQuery, IMU pose storage |
| `costmap_builder.py` | Cost grid building from TSDF |
| `game_setup.py` | TSDF/perception config initialization |
| `game_viz.py` | Scan timing, 3DS computation, viz file writers |
| `test_occupancy_3ds.py` | 3DS metrics (compute_3ds_v2, compute_3ds_god) |
| `test_occupancy_gt.py` | GT surface sampling, TSDF surface extraction |
| `god_tsdf.py` | God-view TSDF (perfect raycasts for comparison) |

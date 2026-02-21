# Layer 6: Spatial Awareness & Local Navigation

**Status: PLANNING**

> **Architecture note**: `philosophy/architecture.md` defines Layer 6 as "Waypoints
> and Tasks". This plan scopes Layer 6 as perception + local navigation instead,
> pushing task sequencing into Layer 7. `architecture.md` should be updated to
> reflect this — the canonical definition predates concrete design work.

## Motivation

Layers 1-5 give us a walking robot that follows velocity commands. The target
game proves this works — but it cheats. The game reads `data.qpos[0:3]` (MuJoCo
world position) and `data.qpos[3:7]` (world quaternion) to know where the robot
is and which direction it faces. These are simulation-only. A real robot has no
GPS, no mocap, no world-frame position sensor.

Layer 6 closes that gap: the robot builds a spatial model of its environment
from hardware-legal sensors (LiDAR, IMU, joint encoders) and navigates through
it without collision.

## Named Types

Shared across modules. Defined in `layer_6/types.py`:

```python
@dataclass
class Pose2D:
    x: float        # meters, world frame
    y: float        # meters, world frame
    yaw: float      # radians
    stamp: float    # simulation time

@dataclass
class Costmap2D:
    grid: np.ndarray       # (H, W) float32, 0.0=free, 1.0=obstacle
    resolution: float      # meters per cell
    origin_x: float        # world-frame X of grid[0,0]
    origin_y: float        # world-frame Y of grid[0,0]
    robot_yaw: float       # robot heading when costmap was built
    stamp: float           # simulation time

@dataclass
class PointCloud:
    points: np.ndarray     # (N, 3) float32, sensor-local frame
    ring_index: np.ndarray | None  # (N,) uint8, vertical channel per point (future multi-channel)
    stamp: float
```

## What Layer 6 Owns

Three modules, one layer, one repo (`layer_6/`):

```
layer_6/
├── types.py             # Pose2D, Costmap2D, PointCloud
├── slam/
│   ├── odometry.py      # Leg odometry + IMU fusion
│   └── scan_match.py    # ICP scan matching (Phase 2)
├── world_model/
│   ├── tsdf.py          # 3D distance field from point clouds
│   └── costmap.py       # 2D costmap projection + query interface
├── planner/
│   └── dwa.py           # Dynamic Window Approach
├── config/
│   ├── b2.py            # B2-specific parameters
│   ├── go2.py           # Go2-specific parameters
│   └── defaults.py      # Shared defaults and validation
└── tests/
    ├── test_odometry.py
    ├── test_tsdf.py
    ├── test_costmap.py
    └── test_dwa.py
```

**Config namespace**: `config/` collides with Layers 3/4/5 in cross-layer
processes (same `sys.modules` issue as target game). Follow the existing
`__main__.py` patching pattern: load by file path, inject into `_active_config`.

### What Layer 6 consumes (observations flowing up)

- **LiDAR point cloud** — from Layer 1 (new sensor, see prerequisites)
- **IMU quaternion + gyro** — from Layer 5 upward state (see note below)
- **Estimated body velocity** — from Layer 5 upward state (for odometry)

> **Layer-skip note**: The original plan consumed LowState directly from Layer 2,
> skipping Layers 3/4/5. This violates the architecture rule that Layer N+1 only
> calls Layer N. Instead, Layer 5 should expose upward state: IMU data and
> estimated body velocity. This is a **blocking prerequisite** — Layer 5 needs
> a `get_state() -> L5State` interface before Layer 6 can begin Phase 1.

### What Layer 6 produces (commands flowing down)

- **Velocity commands** `(forward, left)` — to Layer 5
- **Robot pose estimate** `Pose2D(x, y, yaw)` — published on DDS topic `rt/pose_estimate`
- **2D costmap** `Costmap2D` — published on DDS topic `rt/costmap`

DDS topics enable multi-subscriber access (Layer 7, telemetry, visualization)
without coupling consumers to Layer 6 internals.

### What Layer 6 does NOT own
- Sensor simulation (Layer 1)
- Motor control, IK, gait generation (Layers 2-4)
- Velocity-to-gait mapping (Layer 5)
- Mission planning, task sequencing (Layer 7)

## Module 1: SLAM

### The localization problem

The robot needs to know where it is in a persistent frame. Currently available:
- **IMU**: orientation (roll, pitch, yaw) — but yaw drifts over time
- **Joint encoders**: leg joint angles at 200Hz — can estimate body displacement
  per step via forward kinematics (leg odometry)
- **LiDAR**: environment geometry — can correct drift via scan matching

### SLAM pose output

From day 1, SLAM exposes a `Pose2D` at two rates:
- **200Hz**: IMU-rate dead-reckoning (low latency, drifts)
- **10Hz**: scan-corrected pose (Phase 2+, higher accuracy)

Consumers use the 200Hz stream for control and the 10Hz stream for mapping.
Define this interface now even though Phase 1 only has the 200Hz path.

### Phased approach

**Phase 1 -- Odometry only (no LiDAR correction)**

Dead-reckoning from IMU + body velocity (provided by Layer 5). This drifts, but
over a 30-second target game run, drift is manageable (~5-10% of distance
traveled). Good enough to prove the pipeline works.

```
pose = Pose2D(x=0, y=0, yaw=0, stamp=0)

each timestep:
    # IMU gives absolute yaw (short-term stable)
    yaw = imu_yaw

    # Body velocity from Layer 5 upward state
    vx_body, vy_body = l5_state.body_velocity

    # Integrate in world frame
    x += vx_body * cos(yaw) * dt - vy_body * sin(yaw) * dt
    y += vx_body * sin(yaw) * dt + vy_body * cos(yaw) * dt
```

**Phase 2 -- Scan matching (LiDAR drift correction)**

ICP (Iterative Closest Point) between consecutive LiDAR scans. Corrects
odometry drift at scan frequency (10Hz). This is standard -- scipy or
numpy-only implementation of point-to-point ICP on 2D projected scans.

**Phase 3 -- Loop closure (future)**

Detect when the robot revisits a location. Correct accumulated drift
globally. Not needed for short missions but required for persistent mapping.

### Reference

The external repos don't actually do SLAM either -- `simulation/mujoco_bridge.py`
publishes ground-truth odometry on `/odom_mj`, and `quadruped-locomotion/xt16`
broadcasts it as a TF transform. Their whole nav stack runs on simulation
odometry. We're starting from the same place but building toward real sensors.

## Module 2: World Model (TSDF)

### What a TSDF is

A Truncated Signed Distance Field stores, for each voxel in a 3D grid, the
distance to the nearest surface. "Truncated" means distances beyond a threshold
are clamped -- we only care about space near obstacles, not deep in free space.

### Grid parameters

These are **B2 defaults**. Other robots need different values (see Per-Robot
Config section).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| XY extent | 6m x 6m (robot-centered) | 3m radius matches LiDAR useful range |
| Z extent | -0.5m to 1.5m | Below-ground rejection + overhead clearance |
| Voxel size | 0.10m | User requirement; matches external repo |
| Grid shape | 60 x 60 x 20 = 72,000 voxels | Fits in ~288KB (float32) |
| Truncation distance | 0.30m | Matches external repo; 3 voxels deep |

### Construction algorithm

Per LiDAR scan (10Hz):

```
1. Transform point cloud to world frame using SLAM pose
2. Crop to grid bounds (AABB rejection)
3. Voxel downsample (keep one point per 0.1m cell)
4. Rasterize points into 3D grid + run EDT
5. Truncate: clip(distances, 0, 0.3)
6. Store in 3D grid
```

**Performance**: The KD-tree approach (query all 72K voxel centers against cloud
points) benchmarks at **13-19ms**, not the 2-5ms originally estimated. Instead,
use voxel rasterization + Euclidean Distance Transform
(`scipy.ndimage.distance_transform_edt`), which gives the same semantics in
**~4ms**. Pre-compute voxel center coordinates once rather than rebuilding each
scan.

**Phase 1 alternative**: For initial integration, a direct 2D costmap from
height-filtered points (skip the 3D TSDF entirely) runs in ~0.3ms. Graduate
to full TSDF when obstacles require 3D reasoning.

### Costmap projection

The planner needs a `Costmap2D`, not a 3D TSDF. Project by taking the minimum
distance across a height slice (ground to robot shoulder height):

```
costmap_2d = 1.0 - (min(tsdf[:, :, z_lo:z_hi], axis=2) / truncation)
```

Result: 60x60 grid, cost=1.0 at obstacles, cost=0.0 in free space. The external
repo squares this for sharper falloff near walls -- worth testing.

The costmap exposes a query interface `CostmapQuery.sample(x, y) -> float` so
the planner is decoupled from grid resolution.

### Robot-relative vs world-frame

The TSDF grid moves with the robot (robot-centered, re-queried each scan). This
avoids the complexity of a persistent global map for Phase 1. For Phase 2+,
accumulate into a larger world-frame grid using SLAM pose.

## Module 3: Collision Avoidance (DWA)

DWA operates in robot-centered frame regardless of TSDF coordinate frame. It
takes a `Costmap2D` from any source (TSDF projection, synthetic test costmaps,
or planning-only mode), not specifically from the world model module.

### Algorithm

Dynamic Window Approach: sample the command space, simulate short trajectories,
score each against the costmap, pick the best. **Must be vectorized** -- evaluate
all 105 arcs as a batch using numpy broadcasting (0.08ms vectorized vs 1.92ms
with naive loops).

### Command space

| Dimension | Range | Samples | Notes |
|-----------|-------|---------|-------|
| forward | 0.0 - 1.0 | 5 | Normalized; Layer 5 maps to gait params |
| left (turn) | -1.0 - 1.0 | 21 | Negative = right |
| **Total** | | **105** | Evaluated per cycle |

### Arc simulation

For each (forward, left) pair, simulate a 0.5-second arc:

```
v = forward * V_MAX      # Linear velocity (robot-specific)
w = left * W_MAX          # Angular velocity (robot-specific)
dt = 0.05                 # 50ms integration step
steps = 10

for i in range(steps):
    theta += w * dt
    x += v * cos(theta) * dt
    y += v * sin(theta) * dt
    arc[i] = (x, y)
```

The external repo uses an empirical turning model (`radius = K * |F/R|^alpha`
tuned to Go2-W). We should start with the simpler kinematic model above and
tune if needed -- our robots may have different turning dynamics.

### Scoring

Each arc gets a weighted score:

| Term | Weight | Computation |
|------|--------|-------------|
| Clearance | 1.0 | `1.0 - mean(costmap samples along arc)` |
| Goal distance | 0.5 | `1.0 - endpoint_dist / initial_dist` |
| Goal heading | 0.3 | `1.0 - abs(heading_error) / pi` |
| Speed | 0.1 | `forward / max_forward` |

**Rejection**: any arc touching a costmap cell > 0.8 is discarded entirely.
If all arcs rejected, output (0, 0) -- stop and wait.

### Output

The winning (forward, left) pair goes to Layer 5 as a velocity command.
Layer 5 maps it to gait parameters. Layer 4 generates foot trajectories.
Layers 3-1 execute.

### Update rate

DWA runs at 10Hz. Between DWA updates, the last command persists. This is fine
-- the robot's gait cycle at 1-2Hz means several gait steps per DWA decision.

## Threading Model

Perception (SLAM + TSDF + costmap projection) **must run in its own thread**.
At 13-19ms for TSDF (or ~4ms with EDT), this would blow the 2ms physics budget
if run inline. The physics thread at 500Hz and the perception thread at 10Hz
share data via thread-safe queues.

Budget breakdown:
- **Physics thread**: ~0.1ms/step (5% of 2ms budget). Raycasting (40 rays/step
  via `mj_multiRay`) adds ~0.06ms -- negligible.
- **Perception thread**: 5-25ms/scan (5-25% of 100ms budget at 10Hz).
  Includes SLAM update, TSDF construction, costmap projection, DWA.

## Per-Robot Config

Layer 6 parameters vary by robot morphology. Follow the existing `config/b2.py`,
`config/go2.py` pattern from Layers 3-5.

| Parameter | B2 | Go2 | H1 (biped) | Notes |
|-----------|-----|------|------------|-------|
| `LIDAR_MOUNT_POS` | `[0.15, 0, 0.05]` | `[0.10, 0, 0.03]` | `[0, 0, 0.10]` | Relative to body center |
| `BODY_HEIGHT` | 0.465 | 0.27 | 0.80 | For costmap Z slice |
| `SHOULDER_HEIGHT` | 0.55 | 0.35 | 1.10 | Upper bound of height slice |
| `V_MAX` | 0.8 | 1.0 | 0.5 | m/s, for DWA arc simulation |
| `W_MAX` | 1.5 | 2.0 | 1.0 | rad/s, for DWA arc simulation |
| `ODOMETRY_TYPE` | `"quad_leg"` | `"quad_leg"` | `"biped_leg"` | Odometry algorithm selector |
| `TSDF_Z_RANGE` | `[-0.5, 1.5]` | `[-0.3, 0.8]` | `[-0.5, 2.0]` | Meters |
| `TSDF_XY_EXTENT` | 6.0 | 6.0 | 6.0 | Meters |
| `TSDF_VOXEL_SIZE` | 0.10 | 0.05 | 0.10 | Go2 may need finer resolution |
| `DWA_HORIZON` | 0.5 | 0.5 | 0.5 | Seconds |
| `DWA_TURN_MODEL` | `"kinematic"` | `"kinematic"` | `"kinematic"` | Override with empirical if needed |

Wheeled variants (Go2W, B2W) need additional parameters: `DWA_TURN_MODEL =
"differential"` and expanded command space (backward/strafe).

## Prerequisite: LiDAR in Layer 1

Layer 6 needs point clouds. Layer 1 needs to simulate a LiDAR sensor.

### What to add to Layer 1

1. **Scene XML**: Add a LiDAR site to robot XML files
   ```xml
   <site name="lidar" pos="0.15 0 0.05" euler="0 0 0"/>
   ```
   Mounted forward on the body, slightly above ground. **Position is
   robot-specific** -- see Per-Robot Config table.

2. **Raycasting**: Progressive `mj_multiRay` in the physics loop, distributing
   rays across timesteps (external repo pattern: 2000 rays / 50 steps = 40/step).

3. **DDS topic**: Publish point cloud on `rt/point_cloud`.
   Format: `PointCloud` dataclass -- flat float32 array of (x, y, z) triples
   in sensor-local frame, with optional `ring_index` for future multi-channel.

### LiDAR parameters (starting point)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Horizontal FOV | 360 deg | Full coverage |
| Horizontal resolution | 0.18 deg | ~2000 rays (matches XT16 spec) |
| Vertical channels | 1 | Single plane at robot height; add more later |
| Max range | 10m | Indoor/near-outdoor; 3m useful for TSDF |
| Min range | 0.05m | Reject self-hits |
| Scan frequency | 10Hz | One full scan per 50 physics steps at 500Hz |
| Noise | 0.0 initially | Add Gaussian noise for robustness testing later |

### Hardware note

Real Unitree robots can mount Hesai XT16 or similar spinning LiDAR. The sensor
site position and scan parameters should match a real mounting. The point cloud
format over DDS should be identical sim-to-real.

## Integration: Target Game as Test Bed

The target game is the natural integration test. Currently it uses simulation
ground truth for navigation. The plan:

**Phase 1**: Replace ground-truth position with SLAM odometry. Keep the
simple heading-based controller. Measure drift.

**Phase 2**: Add TSDF world model. Place obstacles in the scene. Verify the
costmap correctly marks them.

**Phase 3**: Replace heading-based controller with DWA. Robot navigates
to targets while avoiding obstacles. This is the full pipeline:

```
LiDAR (L1) -> SLAM pose + TSDF (L6) -> costmap (L6) -> DWA (L6)
  -> (forward, left) -> Layer 5 -> Layer 4 -> Layer 3 -> Layers 1-2
```

### Visualization (demo render loop, not module code)

Voxel rendering and costmap visualization belong in the demo render loop
(`foreman/demos/`), not in `tsdf.py` or `costmap.py`. Modules produce data;
the demo decides how to render it.

In the MuJoCo viewer loop, render occupied voxels using `mjv_initGeom`, same
pattern as `draw_target()` in `run_modes.py`. Also render SLAM trail
(truth vs estimated) and DWA top-3 scored arcs as polylines for debugging.

### Scene modifications

New scene variant `scene_obstacles.xml` with static obstacles (boxes, cylinders)
placed between spawn point and target positions. The external repo's
`scenes/obstacles.xml` has good examples (cylinders for people, boxes for
furniture).

## What We Take From the External Repos

| Feature | External source | What we adopt | What we change |
|---------|----------------|---------------|----------------|
| TSDF construction | `pcl_proc/process.py` | Voxel rasterization + EDT, truncation, costmap projection | No open3d dep; numpy + scipy only. Robot-centered grid. EDT instead of KD-tree. |
| DWA planner | `nav_local/dwa.py` | Command space sampling, arc simulation, multi-objective scoring | Simpler kinematic model (not empirical tuning). Our Layer 5 interface. Vectorized. |
| Progressive raycasting | `simulation/lidar.py` | `mj_multiRay` batching, ray distribution across timesteps | Integrated into our firmware_sim.py, our DDS topics. |
| Costmap squaring | `pcl_proc/main.py` | `cost ** 2` for sharper obstacle falloff | Test whether this helps our scoring |
| Obstacle rejection | `nav_local/dwa.py` | Lethal threshold (0.8) discards dangerous arcs | Same |

What we do NOT take:
- ROS2 topics/services/actions -- we use DDS or in-process calls
- open3d point cloud library -- numpy arrays are sufficient
- Pinocchio for FK -- we have analytical FK in Layer 3
- Their odometry source (simulation ground truth) -- we build real odometry

## Open Questions

1. **Leg odometry accuracy**: How much drift per meter of travel? Need to
   measure in simulation with ground truth comparison. If drift > 10%,
   scan matching becomes Phase 1 not Phase 2.

2. **LiDAR mount position**: Robot-specific (see Per-Robot Config). Single
   horizontal plane may miss low obstacles on B2 and see too much ground on
   Go2. May need per-robot vertical angle.

3. **DWA turning model**: The external repo's empirical model (K=0.34,
   alpha=1.18) was tuned for Go2-W (wheeled). Our legged robots turn
   differently -- in-place turning vs. arc turning. The DWA may need a
   mode switch matching Layer 5's walk/turn threshold.

4. **Costmap frame**: Robot-centered (re-query every scan, simple, no drift
   accumulation) vs. world-frame (persistent map, requires good SLAM).
   Start robot-centered.

5. **Training integration (BLOCKING)**: DWA replaces the heading-based
   controller, making genome genes (KP_YAW, WZ_LIMIT, THETA_THRESHOLD)
   irrelevant. This is structurally identical to the v12 sovereign genome
   failure. The plan must specify how GA training works with DWA -- does
   the GA evolve DWA weights? Or does training stay on the heading
   controller and DWA is demo-only?

6. **Layer 5 upward state (BLOCKING)**: Layer 5 must expose estimated body
   velocity and IMU data upward before Layer 6 can begin. This is a
   prerequisite, not an open question -- file a fix-request to Layer 5.

## Dependencies

New (Layer 6 only):
- scipy (distance_transform_edt for TSDF, cKDTree for Phase 2 ICP) -- already in workspace venv
- numpy -- already everywhere

Existing cross-layer utilities (import, do not duplicate):
- `quat_to_yaw`, `normalize_angle` from `foreman/demos/target_game/utils.py`

No new external dependencies required.

## Estimated Scope

| Phase | Work | Layers touched |
|-------|------|----------------|
| 0 | LiDAR sensor in Layer 1 | `layers_1_2/` (delegate to subagent) |
| 0.5 | Layer 5 upward state interface | `layer_5/` (delegate to subagent) |
| 1 | SLAM odometry + target game integration | `layer_6/` (new repo) + `foreman/` (demo) |
| 1.5 | DWA unit tests with synthetic costmaps | `layer_6/` |
| 2 | TSDF world model + costmap projection | `layer_6/` |
| 3 | DWA collision avoidance + obstacle scenes | `layer_6/` + `foreman/` (demo) |

## Review Recommendations

Consolidated from five independent analyst reviews (modularity, reusability,
efficiency, robot agnosticism, maintainability).

### MUST (blocking or high-risk)

- **Biped odometry**: Leg odometry assumes quadrupeds. H1/G1 have 2 legs,
  different FK chains, and body sway. The `ODOMETRY_TYPE` config selector
  handles routing, but the `"biped_leg"` algorithm itself needs design work.
  Do not assume quadruped odometry generalizes. (Agnosticism + Modularity)

- **Coordinate-frame transform utility**: SLAM and TSDF both need
  body-to-world and sensor-to-body transforms. Add a `frame_transform(points,
  pose)` function to `foreman/demos/target_game/utils.py` as the canonical
  cross-layer location. (Reusability)

- **Phase gates with quantitative criteria**: Each phase should have concrete
  acceptance tests: Phase 1 drift < 10% over 30s, Phase 2 costmap marks >90%
  of ground-truth obstacles, Phase 3 collision-free target reach rate > 80%.
  (Maintainability)

### SHOULD (significant quality/performance impact)

- **Telemetry spec**: Each module should emit per-cycle JSONL telemetry.
  SLAM: drift magnitude, correction delta. TSDF: construction time, voxel
  count. DWA: winning arc score, rejection count, selected (v, w). Follow
  the GA telemetry pattern from training. (Maintainability)

- **Debug visualization**: SLAM truth-vs-estimated trail, DWA top-3 scored
  arcs as colored polylines, costmap overlay. These catch bugs that unit
  tests miss (e.g., SLAM drift direction, DWA bias). Render in demo loop,
  not in module code. (Maintainability)

- **Self-filtering for bipeds**: H1/G1 have arms that appear in LiDAR scans.
  TSDF needs a self-hit rejection mask per robot morphology. Not needed for
  quadrupeds but design the filtering hook now. (Agnosticism)

- **Backward/strafe commands**: DWA command space is forward-only. Wheeled
  variants and bipeds can move backward and strafe. Extend the command space
  table to include `forward: -1.0 to 1.0` and optional `lateral` dimension
  behind a config flag. (Agnosticism)

### CONSIDER (nice-to-have, lower priority)

- **Skip 3D TSDF in Phase 1**: Direct 2D costmap from height-filtered points
  is ~0.3ms vs ~4ms for full TSDF+EDT. Simpler to implement and debug. Graduate
  to 3D TSDF when obstacles require vertical reasoning (overhangs, ramps).
  Tradeoff: loses Z-axis information. (Efficiency)

- **Costmap as DWA constructor input**: DWA should accept `Costmap2D` in its
  constructor, not import the world model module. This enables unit testing
  with synthetic costmaps and planning-only mode without perception.
  (Reusability + Modularity)

- **Phase 0 and Phase 1 in parallel**: LiDAR (Phase 0) and SLAM odometry
  (Phase 1) are independent until scan matching. Start both simultaneously
  to reduce calendar time. (Maintainability)

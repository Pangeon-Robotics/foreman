# Perception & Timing Architecture

The target game runs a real-robot-compatible perception pipeline. All navigation decisions are based on robot-view data (simulated LiDAR), never god-view (scene XML knowledge).

## Design Principles

1. **LiDAR is the only sensor** — DirectScanner simulates real LiDAR via `mj_multiRay`
2. **TSDF is the world model** — single source of truth for obstacle knowledge
3. **Costmap derived from TSDF** — not from scene geometry
4. **Lazy Theta\* routes on costmap** — unknown cells treated as free (optimistic)
5. **Event-driven updates** — costmap rebuilds on TSDF change, route replans on costmap change
6. **Route is frozen near robot** — only replanned when new sensor data arrives, not on a timer

## Rate Hierarchy

| System | Rate | Ticks (at 50Hz) | Notes |
|--------|------|-----------------|-------|
| Physics (MuJoCo) | 500 Hz | — | `model.opt.timestep` |
| PD control | 500 Hz | — | Firmware subprocess |
| Gait generation (L4/L5) | 50 Hz | every tick | Foot positions, MotionCommand |
| Navigation commands (vx, wz) | 50 Hz | every tick | Same as gait |
| SLAM odometry | 50 Hz | every tick | IMU + body velocity |
| LiDAR scan → TSDF | ~4 Hz | every 12 ticks | Triggers costmap rebuild |
| Costmap rebuild | event-driven | — | On TSDF change (same tick as scan) |
| Route replan (Lazy Theta\*) | event-driven | — | On costmap change (`_costmap_changed` flag) |
| Telemetry prints | 1 Hz | every 50 ticks | Human-readable status |

## Event-Driven Perception Chain

```
LiDAR scan (4Hz)
  → TSDF integration (same tick)
    → costmap rebuild (same tick)
      → path critic update (same tick)
        → set _costmap_changed = True
          → navigator checks flag → route replan (next nav tick)
```

All steps run in the same tick as the scan. The route replan happens on the next navigation tick when the navigator sees `_costmap_changed=True`. This avoids fixed-timer replanning that wastes CPU when nothing changed.

## Key Timing Constants (`game_config.py`)

| Constant | Value | Duration |
|----------|-------|----------|
| `CONTROL_DT` | 0.02 | 50 Hz tick rate |
| `TARGET_TIMEOUT_STEPS` | 3000 | 60s per target |
| `TELEMETRY_INTERVAL` | 50 | 1 Hz prints |
| `STARTUP_SETTLE_STEPS` | 25 | 0.5s startup |
| `FALL_CONFIRM_TICKS` | 10 | 0.2s sustained fall |
| `PATH_HOLD_TICKS` | 250 | 5s path hold (dwa_path_export.py) |
| `PATH_WARMUP_TICKS` | 25 | ~1s wait for first TSDF data |

## Data Flow

```
              Robot-View Pipeline (real-robot compatible)
              ==========================================

DirectScanner (mj_multiRay)
  │  128 vertical channels, ±30°, 500 horizontal, ~8K-64K rays
  │  IMU roll/pitch correction
  ▼
TSDF (chunk-based, 1cm voxels)
  │  Hit-only integration, log-odds
  │  Surface history for persistence
  ▼
Costmap (5cm output resolution)
  │  EDT-based cost with inverse-square falloff
  │  Built from TSDF surface voxels
  ▼
Path Critic (Lazy Theta*)
  │  Unknown cells = passable (optimistic routing)
  │  Constrained clearance: 0.25m → 0.10m → 0.05m fallback
  ▼
Navigator (heading-proportional control)
  │  Waypoint extraction from committed path
  │  2m lookahead along path
  ▼
Layer 5 MotionCommand (vx, wz)
```

## God-View (Scoring Only)

God-view TSDF and costmap exist only for F1 scoring metrics (Surface F1, Cost F1, Router F1). They are never used for navigation decisions. God-view features require `--god` flag and are only active with `--obstacles`.

## Implementation Files

| File | Role |
|------|------|
| `game_viz.py:tick_perception()` | Orchestrates the scan → TSDF → costmap → critic chain |
| `dwa_path_export.py:export_path()` | Lazy Theta* routing with constrained clearance |
| `navigator_helper.py:tick_walk_heading()` | Checks `_costmap_changed`, triggers replan |
| `game_config.py` | All timing constants |
| `perception.py` | PerceptionPipeline (TSDF + costmap building) |
| `direct_scanner.py` | LiDAR simulation via mj_multiRay |
| `path_critic.py` | ATO metrics + Lazy Theta* wrapper |

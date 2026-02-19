# v12: Sovereign Genome — GA Genes in Layer 4's Language

## Problem

v10/v11 genome has 37 genes, 24 of which (65%) are direct joint angle deltas
(`P1_FL_HIP`, `P2_RR_KNEE`, etc.) that bypass Layers 3-5 entirely. This
sovereignty violation means the GA is fighting the control stack instead of
using it. The direct-joint turn mechanism doesn't work — Issue 005 documents
foot vibration with no actual rotation.

## Solution

Rewrite the genome so every gene maps to a GaitParams field or a Layer 5
steering decision. Route turning through Layer 4's existing `turn_in_place`
mode (arc-based polar coordinate rotation in generator.py), which already
works but was never used by the GA.

## v12 Genome: 13 Genes (down from 37)

### Walk genes (7)

| Gene | Range | Default | Source |
|------|-------|---------|--------|
| GAIT_FREQ | 0.8-2.5 Hz | 2.5 | v11 champion (clamped from 4.21) |
| STEP_LENGTH | 0.05-0.45 m | 0.382 | v11 champion |
| STEP_HEIGHT | 0.04-0.20 m | 0.047 | v11 champion |
| DUTY_CYCLE | 0.35-0.80 | 0.588 | v11 champion |
| STANCE_WIDTH | -0.03-0.10 m | -0.03 | v11 champion |
| KP_YAW | 0.5-8.0 | 3.656 | v11 champion |
| WZ_LIMIT | 0.3-3.0 rad/s | 0.837 | v11 champion |

### Turn genes (5) — new, evolved from scratch

| Gene | Range | Default |
|------|-------|---------|
| TURN_FREQ | 0.5-2.0 Hz | 1.0 |
| TURN_STEP_HEIGHT | 0.03-0.15 m | 0.06 |
| TURN_DUTY_CYCLE | 0.40-0.80 | 0.55 |
| TURN_STANCE_WIDTH | 0.0-0.12 m | 0.04 |
| TURN_WZ | 0.5-3.0 rad/s | 1.5 |

### Planner gene (1)

| Gene | Range | Default |
|------|-------|---------|
| THETA_THRESHOLD | 0.15-0.8 rad | 0.4 |

### Removed genes

- 12 `P1_*` joint deltas (Phase 1 turn pose)
- 12 `P2_*` joint deltas (Phase 2 turn pose)
- `T_PHASE1`, `T_PHASE2`, `T_PHASE3` timing genes
- `WALK_SPEED` — redundant, derived as STEP_LENGTH * GAIT_FREQ
- `PLAN_K` — at operational distances always hits cap, replaced by THETA_THRESHOLD

## Episode Logic

```
if abs(heading_err) > THETA_THRESHOLD:
    # TURN IN PLACE — Layer 4's arc mode
    GaitParams(turn_in_place=True, wz=TURN_WZ*sign(err), ...)
else:
    # WALK — Layer 4's differential stride
    GaitParams(step_length=STEP_LENGTH*cos(err), wz=KP_YAW*err, ...)
```

No joint angles, no `step_joints_direct()`, no Layer 5 velocity mapper hacks.

## Repair Function: repair_v12()

6 biomechanical constraints enforced after every mutation/crossover:
1. Stride product: STEP_HEIGHT * STEP_LENGTH >= 0.015 (foot clearance)
2. Walk stance time: DUTY_CYCLE / GAIT_FREQ >= 0.12s (B2 mass)
3. Walk foot velocity: STEP_HEIGHT * GAIT_FREQ <= 0.40 (joint speed)
4. Centripetal: WZ_LIMIT * STEP_LENGTH * GAIT_FREQ <= 3.0 (stability)
5. Turn stance time: TURN_DUTY_CYCLE / TURN_FREQ >= 0.12s
6. Turn foot velocity: TURN_STEP_HEIGHT * TURN_FREQ <= 0.40

## Files Modified

### Training repo
- `ga/genome.py` — repair_v12(), repair() auto-detection
- `ga/fast_sim.py` — send_gait_params() bypasses L5, talks L4->L3->physics
- `ga/episode.py` — inject_genome_v12(), _run_game_v12(), dispatch routing
- `ga/critic.py` — v12 turn rotation quality check
- `ga/pipeline.py` — v12 behavioral sanity check
- `configs/b2/ga_v12_puget.json` — 13-gene config, seed=42
- `configs/b2/ga_v12_bizon.json` — 13-gene config, seed=137

### Foreman repo
- `demos/target_game/__main__.py` — v12 genome detection + L5 expansion

## Champion Translation

v11 champion's 7 walk genes translate directly (FREQ clamped to max 2.5).
Turn genes start at defaults and evolve from scratch. WALK_SPEED (0.909)
dropped — effective speed = 0.382 * 2.5 = 0.955 m/s.

## Run Plan

- **puget**: `ga_v12_puget.json`, seed=42, checkpoint_dir=models/b2/ga_v12_puget
- **bizon**: `ga_v12_bizon.json`, seed=137, checkpoint_dir=models/b2/ga_v12_bizon
- Both: 128 pop, 20 parallel workers, fast_sim, 16000 max gen, 5000 stagnation stop
- Seeded from v11 champion (25% of population via seed_champion_fraction)

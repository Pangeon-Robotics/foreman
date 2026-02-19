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

## Run Log

### 2026-02-19: First launch + behavior dict fix

**Issue found**: The v12 episode behavior dict used `mean_closing_speed_ms` but the
critic and pipeline sanity checks read `mean_walk_speed` (from v9-v11). This caused
false "Walk forward speed = 0.000 m/s" CRITICAL alerts every generation, making it
look like walking was completely broken when it was actually working fine.

**Fix**: Added `walk_forward_speed_sum` accumulator to v12 episode, emit
`mean_walk_speed` in behavior dict alongside `mean_closing_speed_ms`. After fix,
walk speed alerts correctly disappeared.

**Lesson**: When creating new episode versions, the behavior dict keys must match
what the critic and pipeline sanity checks read. The sanity checks are version-agnostic
— they read fixed keys. Either new episodes must emit the same keys, or the checks
must be version-aware.

### 2026-02-19: Initial run observations (gen 0-33)

Both servers launched successfully:
- **puget**: ~4-6s/gen, ATB=288.3 at gen 4, heading_progress=42-91% of fitness
- **bizon**: ~16s/gen, ATB=92.7 at gen 8, heading_progress=48-61% of fitness

Key patterns:
1. **heading_progress dominance expected early** — turn genes start at defaults and
   need time to evolve. The robot earns most fitness from turning toward targets,
   not closing distance. This should shift as turn genes improve.
2. **THETA_THRESHOLD converging to min (0.15 rad)** on bizon — the GA wants to minimize
   time spent turning and maximize walking. Makes sense: walking is fast, turning is slow.
3. **KP_YAW converging to min (0.5)** — after turning to face the target, gentle steering
   suffices. High KP_YAW causes overcorrection/oscillation.
4. **TURN_* params drifting toward min bounds** — turn genes being pushed toward slower,
   more conservative settings. May indicate the arc-based turn mode works better at
   lower frequencies. Watch for mean_turn_rate improvements.
5. **Turn-obsessive (76% of time turning)** — with targets at 60-150° offset and slow
   turn rate (0.28 rad/s), most time is spent turning. Will improve as turn genes evolve.
6. **ATB stagnation at gen 33 on puget** — best hasn't improved since gen 4. Root cause:
   heading_progress oscillation exploit (see below).

### 2026-02-19: Heading progress oscillation exploit (Issue 002 pattern)

**Discovered**: The gen 4 champion (fitness 288.3) earned 91% of fitness from
heading_progress by oscillating — turning back and forth, earning credit each time
|heading_err| decreases. The champion earned 17.5 rad of heading progress per target,
when the maximum physically meaningful heading correction is π ≈ 3.14 rad.

**Root cause**: heading_progress accumulated without any per-target cap. Each step where
|heading_err| decreased earned reward, even if the robot oscillated past zero and back.
With weight 15 and no cap, a dedicated oscillator could earn 262 fitness from heading
alone, making closing_speed (weight 40, max ~20 realistic) irrelevant.

**Fix**: Cap `target_heading_progress` at π per target. Max heading_progress fitness is
now 15 * π ≈ 47.1 instead of unbounded. This makes heading_progress comparable to
closing_speed instead of dominating.

**Lesson**: This is the exact same exploit pattern as Issue 002 (backward-trotting for
turn rewards). Whenever a fitness term accumulates over time without a cap, the GA will
find a way to maximize time spent earning that term. Caps should be set at the maximum
physically meaningful value.

### 2026-02-19: Post-cap observations (gen 0-40)

After restarting with the heading_progress π cap:
- **Puget**: Gen 40, ATB=92.6 at gen 2. Mean fitness 52→56 (climbing). ~5s/gen.
- **Bizon**: Gen 17, ATB=96.4 at gen 11 (bizon leading!). Mean fitness 52→63.
- heading_progress now 42-47% of fitness (was 91%) — healthy balance
- No more inflated outlier fitness values (was 288, now max ~96)
- Turn-obsessive (71-75%) is real but expected: targets at 60-150° offset require
  significant turning before walking
- Bound camping on 5 params: GAIT_FREQ→max, STANCE_WIDTH→min, WZ_LIMIT→min,
  TURN_FREQ→min, TURN_DUTY_CYCLE→min — clear genetic pressure at boundaries

### Server details

| Server | Host | Python | Cores | RAM | GPU | Speed |
|--------|------|--------|-------|-----|-----|-------|
| puget | puget-280957 (local) | 3.12.3 | 24 | 30GB | RTX 5090 | ~4-6s/gen |
| bizon | bizon@10.0.0.12 | 3.12.11 (venv) | 20 | 125GB | Titan RTX | ~16s/gen |

**Launch commands**:
```bash
# puget (from training/)
nohup python -u -m ga.pipeline --config configs/b2/ga_v12_puget.json \
  --seed-genome models/b2/ga_v12_seed_from_v11.json \
  > models/b2/ga_v12_puget/run.log 2>&1 &

# bizon (via SSH, must activate venv)
ssh bizon@10.0.0.12
cd /home/bizon/code/robotics/training
source /home/bizon/code/robotics/env/bin/activate
nohup python -u -m ga.pipeline --config configs/b2/ga_v12_bizon.json \
  --seed-genome models/b2/ga_v12_seed_from_v11_bizon.json \
  > models/b2/ga_v12_bizon/run.log 2>&1 &
```

**Gotchas from launch**:
- bizon's default python is 2.7 — must use venv at `/home/bizon/code/robotics/env/bin/python`
- `scripts/run_ga.py` doesn't have `--seed-genome` flag; use `python -m ga.pipeline` instead
- bizon had stale local v11 changes that needed `git stash` before pull
- Always use `python -u` for unbuffered stdout (Issue 003 lesson)

# v13: Train/Demo Parity — L4 Direct Control + Tightened Bounds

## Problem

v12 champions (1200 gens) failed in the headed demo due to three issues:

1. **No in-place turns**: Training calls `send_gait_params()` → Layer 4 directly.
   Demo sent `MotionCommand` → Layer 5, and `_l5_to_l4()` at `layer_5/simulation.py:84`
   strips the `turn_in_place` flag (not in kwargs). Turn-in-place never reached Layer 4.
2. **Vibrating feet**: GAIT_FREQ=2.5 (converged to max) + STEP_HEIGHT=0.04 (converged
   to min) = invisible foot lift. The GA found that barely-lifting feet was stable but
   it looked like vibration, not walking.
3. **Weak walk steering**: WZ_LIMIT=0.33 (converged to min 0.3) too weak for course
   correction during walking.

**Root cause**: Different control paths in training vs demo. Training bypassed Layer 5
entirely for walk/turn. Demo went through Layer 5, which silently dropped turn_in_place
and routed wz differently.

## Solution

Three changes, no new genes:

1. **Demo uses L4 direct** — `game.py` now calls `sim.send_command(L4GaitParams, t, kp, kd)`
   exactly like the training episode. L5 is only used for the startup gain ramp.
2. **Tightened bounds** — raise minimums on step height and wz_limit, lower max on
   gait_freq. The GA can't converge to degenerate regions.
3. **Hand-crafted seed** — mid-range defaults verified in headed demo before launching.

## Bound Changes from v12

| Parameter | v12 | v13 | Rationale |
|-----------|-----|-----|-----------|
| GAIT_FREQ max | 2.5 | **2.0** | 2.0 Hz × 0.06m = visible stepping |
| GAIT_FREQ default | 2.5 | **1.5** | Mid-range, not at bound wall |
| STEP_HEIGHT min | 0.04 | **0.06** | Visible foot clearance |
| STEP_HEIGHT default | 0.047 | **0.08** | Visible lift |
| STEP_LENGTH default | 0.382 | **0.30** | Mid-range |
| WZ_LIMIT min | 0.3 | **0.5** | Adequate walk steering |
| WZ_LIMIT default | 0.837 | **1.5** | Mid-range |
| TURN_STEP_HEIGHT min | 0.03 | **0.06** | Visible turn stepping |
| TURN_STEP_HEIGHT default | 0.06 | **0.07** | Visible lift |
| TURN_WZ default | 1.5 | **1.2** | Moderate turn rate |
| KP_YAW default | 3.656 | **2.0** | Mid-range (v12 converged to 0.5 min) |

Unchanged: episode_version=12, repair_v12() constraints, 13-gene structure,
fitness weights, population/selection/reproduction settings.

## Seed Genome Verification (2026-02-19)

Headed demo: `--robot b2 --targets 3 --genome ga_v13_seed.json --full-circle --domain 2`

**Result: 3/3 targets reached (100%), 28.6s total**

| Target | Heading | Turn time | Walk time | Total |
|--------|---------|-----------|-----------|-------|
| 1 | -0.72 rad (41°) | ~1s | ~9.5s | 10.5s |
| 2 | -0.51 rad (29°) | ~1s | ~6.4s | 7.4s |
| 3 | +1.65 rad (95°) | ~1s | ~6.4s | 7.4s |

**Observations**:
- **Turn-in-place works**: Target 3 turned 1.31 rad in 1s (TURN_WZ=1.2). Visible
  step-and-shift rotation. The L5 bug is fully bypassed.
- **Walk speed ~0.6 m/s**: Faster than STEP_LENGTH × GAIT_FREQ = 0.45 m/s —
  L4 differential stride produces more ground speed than the parameter product.
- **Visible foot lift**: No vibration during walking. STEP_HEIGHT=0.08 at GAIT_FREQ=1.5
  produces clearly visible steps.
- **Heading oscillation during walk**: ±0.1-0.3 rad hunting. KP_YAW=2.0 causes mild
  overcorrection. The GA should find the right balance (v12 pushed it to 0.5 min,
  suggesting lower is better).
- **Startup transition rough**: ~2s of foot vibration + 0.6m backward drift at game
  start. The switch from L5 stand to L4 turn-in-place is abrupt. Not a training issue
  (fast_sim skips startup), but worth noting for demo quality.

## What to Watch For During Training

### Healthy signs
- heading_progress 30-50% of total fitness (not dominating)
- closing_speed increasing over generations (robot actually reaching targets)
- target_reached count > 0 by gen 50
- THETA_THRESHOLD drifting down (wants to walk more, turn less)
- KP_YAW settling in 1.0-3.0 range (not hitting bounds)

### Warning signs
- **Bound camping on ≥5 params**: v12 had GAIT_FREQ→max, STANCE_WIDTH→min,
  WZ_LIMIT→min, TURN_FREQ→min, TURN_DUTY_CYCLE→min at bounds. If v13 still
  camps on 5+, the bounds need further adjustment.
- **heading_progress >60% of fitness**: Turn-obsessive strategy, not closing distance.
  The π cap prevents the oscillation exploit but doesn't prevent slow-turn-forever.
- **mean_walk_speed < 0.2 m/s**: Robot barely walking. Check STEP_LENGTH and GAIT_FREQ.
- **STEP_HEIGHT converging to 0.06 (new min)**: Still trying to minimize lift. The
  0.06 floor should be visible but watch for feet-barely-clearing.

### v12 lessons applied
- heading_progress π cap already in place (Issue 002 exploit fix)
- behavior dict has mean_walk_speed (v12 run log: false CRITICAL fix)
- `-u` flag for unbuffered stdout (Issue 003 fix)
- Configs validated: all weight keys match accumulator keys

## Files Changed

### Foreman repo (commit 36483be)
- `demos/target_game/game.py` — L4-direct control, 13 genome module constants
- `demos/target_game/__main__.py` — L4GaitParams import, v12+ dual patching

### Training repo (commit 320ef5e)
- `configs/b2/ga_v13_puget.json` — tightened bounds, seed=42
- `configs/b2/ga_v13_bizon.json` — tightened bounds, seed=137
- `models/b2/ga_v13_seed.json` — hand-crafted 13-gene seed

### Files NOT changed
- `ga/episode.py` — `_run_game_v12()` already uses L4 direct. episode_version stays 12.
- `ga/genome.py` — `repair_v12()` constraints are invariant with new bounds.
- `ga/fast_sim.py` — `send_gait_params()` correctly passes turn_in_place.
- `layer_5/simulation.py` — `_l5_to_l4()` bug (strips turn_in_place) not fixed; bypassed.

## Run Plan

- **puget**: `ga_v13_puget.json`, seed=42, checkpoint_dir=models/b2/ga_v13_puget
- **bizon**: `ga_v13_bizon.json`, seed=137, checkpoint_dir=models/b2/ga_v13_bizon
- Both: 128 pop, 20 parallel workers, fast_sim, 16000 max gen, 5000 stagnation stop
- Seeded from hand-crafted seed (25% of population via seed_champion_fraction)

### Launch commands

```bash
# puget (from training/)
nohup python -u -m ga.pipeline --config configs/b2/ga_v13_puget.json \
  --seed-genome models/b2/ga_v13_seed.json \
  > models/b2/ga_v13_puget/run.log 2>&1 &

# bizon (SSH + venv)
ssh bizon@10.0.0.12
cd /home/bizon/code/robotics/training
source /home/bizon/code/robotics/env/bin/activate
git pull
nohup python -u -m ga.pipeline --config configs/b2/ga_v13_bizon.json \
  --seed-genome models/b2/ga_v13_seed.json \
  > models/b2/ga_v13_bizon/run.log 2>&1 &
```

### Pre-launch checklist
- [x] Seed genome verified in headed demo (3/3, visible turns, no vibration)
- [ ] Headless smoke test (5 targets)
- [ ] GA smoke test (2-3 gens on puget)
- [ ] `git pull` on bizon
- [ ] Create checkpoint dirs: `mkdir -p models/b2/ga_v13_{puget,bizon}`

## Run Log

*(To be filled during training)*

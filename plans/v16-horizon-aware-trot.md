# v16 Plan: Horizon-Aware Trot Optimization

**Status**: Draft
**Date**: 2026-03-08
**Depends on**: Cascade `2026-03-parameterized-swing-trajectory`

---

## Goal

Evolve an elegant, stable B2 trot that:
- Reaches targets fast in the target game
- Maintains low pitch/roll with smooth body dynamics
- Has zero foot slippage
- Turns effectively while walking (body lean, not just differential stride)
- Optionally uses predictive body-state models (100ms/200ms/500ms horizons)

The GA discovers both the physical gait shape AND whether foresight helps.

---

## Genome: 19 Genes

### Gait Timing (4 genes)

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `STEP_LENGTH` | 0.10–0.45 | 0.45 | Stride length (m) |
| `GAIT_FREQ` | 0.3–1.5 | 0.5 | Cadence (Hz) |
| `STEP_HEIGHT` | 0.04–0.12 | 0.10 | Peak foot lift (m) |
| `DUTY_CYCLE` | 0.55–0.80 | 0.70 | Fraction in stance |

### Trajectory Shape (3 genes) — REQUIRES CASCADE

These parameterize L4's Bezier swing curve. Currently the control points
are hardcoded (symmetric arch, `epsilon_vertical = step_height * 4/3`).
The cascade adds evolvable parameters to `GaitParams` that L4 passes
through to `_make_swing_bezier()`.

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `SWING_APEX_PHASE` | 0.2–0.8 | 0.5 | Where peak height occurs in swing (0.5=symmetric, 0.3=lift-early-land-gently) |
| `SWING_HEIGHT_RATIO` | 0.8–2.0 | 1.33 | Bezier control point height / step_height (shapes the arch curvature) |
| `SWING_OVERSHOOT` | 0.0–0.15 | 0.0 | How far past the stride endpoint the foot reaches before pulling back (m) |

**SWING_APEX_PHASE**: Splits the Bezier into asymmetric halves. At 0.3, the
foot lifts quickly to peak height by 30% of swing, then glides forward and
down over the remaining 70%. This produces aggressive liftoff with gentle
touchdown — reducing impact forces and slip on landing.

**SWING_HEIGHT_RATIO**: Controls arch curvature. At 1.33 (default), the Bezier
peaks at exactly step_height. At 2.0, the control points are higher, creating
a taller narrower arch. At 0.8, a flatter wider arch — foot stays low longer.

**SWING_OVERSHOOT**: The foot reaches slightly past the target landing point
during mid-swing, then pulls back before touchdown. Creates a "reaching"
motion that extends effective stride without increasing step_length. Zero
means no overshoot (current behavior).

### Body Pose / CoM Shift (3 genes) — NO CASCADE NEEDED

`GaitParams` already has `body_roll`, `body_pitch`, `body_x_offset`,
`body_y_offset` fields. They flow through to `BodyPoseCommand` and L3's
`transform_feet_to_body_frame()`. Currently all zeroed. The GA just needs
to set them proportionally to commanded wz and vx.

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `TURN_LEAN` | 0.0–0.15 | 0.0 | Body roll per rad/s of wz (rad per rad/s). Lean into turns to shift CoM over stance legs. |
| `WALK_PITCH` | -0.05–0.10 | 0.0 | Constant body pitch during walking (rad). Positive = nose down = shifts CoM forward for better traction on front legs. |
| `TURN_Y_SHIFT` | 0.0–0.03 | 0.0 | Lateral CoM shift per rad/s of wz (m per rad/s). Moves body toward inside of turn. |

**How TURN_LEAN works at runtime**: When the navigator commands wz=0.5 rad/s
and the genome has TURN_LEAN=0.08:
```
body_roll = 0.08 * 0.5 = 0.04 rad (2.3°)
```
This tilts the body 2.3° into the turn, shifting the center of mass over
the inside legs. The inside legs bear more weight and push harder, producing
a natural banked turn instead of relying purely on differential stride.

### Navigation (3 genes)

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `KP_HEADING` | 0.5–5.0 | 3.0 | Heading PD proportional gain |
| `WZ_MAX` | 0.5–2.0 | 1.5 | Yaw rate cap (rad/s) |
| `VX_WALK` | 0.3–1.5 | 1.0 | Forward speed command (m/s) |

### Horizon Model (6 genes)

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `W_100MS` | 0.0–1.0 | 0.0 | Weight on 100ms predictor |
| `W_200MS` | 0.0–1.0 | 0.0 | Weight on 200ms predictor |
| `W_500MS` | 0.0–1.0 | 0.0 | Weight on 500ms predictor |
| `HORIZON_VX_SCALE` | 0.0–0.5 | 0.0 | Predicted instability → vx reduction |
| `HORIZON_WZ_SCALE` | 0.0–0.5 | 0.0 | Predicted instability → wz reduction |
| `INSTABILITY_THRESHOLD` | 0.01–0.20 | 0.05 | Predicted roll/pitch magnitude below which no modulation occurs (rad) |

**Constraint**: `W_100MS + W_200MS + W_500MS ≤ 1.0` (enforced in genome
repair). Sum=0 means deaf to all models — the GA can evolve this way if
the models aren't helpful.

**Runtime logic** (in episode runner, every tick):
```python
# Blend predictions from 3 horizon models
pred = w_100 * model_100ms(obs) + w_200 * model_200ms(obs) + w_500 * model_500ms(obs)

# Predicted future instability (roll + pitch magnitude)
instability = sqrt(pred_roll² + pred_pitch²)

# Only modulate if above threshold (avoid reacting to noise)
if instability > INSTABILITY_THRESHOLD:
    excess = instability - INSTABILITY_THRESHOLD
    vx *= 1.0 - HORIZON_VX_SCALE * excess
    wz *= 1.0 - HORIZON_WZ_SCALE * excess
```

---

## Fitness Function

**Multiplicative** — every component must be good. You can't trade stability
for speed. Fall = zero fitness.

```
fitness = 100 × target_speed × stability × grip × turn_ability

if fell: fitness = 0
```

### Component 1: Target Speed (0–1)

```python
if targets_reached == num_targets:
    target_speed = 1.0 - (total_steps / max_total_steps)  # faster = higher
else:
    target_speed = 0.2 * (targets_reached / num_targets)   # partial, max 0.2
```

Full completion at maximum speed = 1.0. Partial = max 0.2. The 5× gap
between partial and full prevents exploitation (same principle as v14).

### Component 2: Stability (0–1)

```python
# Measured from IMU every tick
rp_energy = mean(roll² + pitch² + (d_roll/dt)² + (d_pitch/dt)²)
stability = exp(-ALPHA * rp_energy)
```

ALPHA calibrated so typical good walking ≈ 0.9, wobbling ≈ 0.3. Uses
actual IMU data (hardware-compatible). The derivative terms penalize
jerky corrections, not just static tilt.

### Component 3: Grip (0–1)

```python
grip = mean(traction)   # from SlipDetector over entire episode
```

SlipDetector's 4-signal fusion (contact force, torque-motion mismatch,
IMU acceleration response, diagonal symmetry) averaged over all ticks.
Already calibrated: 1.0 = perfect grip, 0.0 = full slip.

### Component 4: Turn Ability (0–1)

```python
required_yaw = sum(abs(heading_change_to_each_target))
actual_yaw = sum(abs(measured_yaw_changes))
turn_ability = clamp(actual_yaw / required_yaw, 0, 1)
```

If the robot needs to turn 180° total across all targets and only managed
90° of actual yaw change, turn_ability = 0.5. This rewards robots that
can actually redirect, not just walk in a straight line.

---

## Cascade: Parameterized Swing Trajectory

The trajectory shape genes (SWING_APEX_PHASE, SWING_HEIGHT_RATIO,
SWING_OVERSHOOT) require L4 changes. This must go through the cascade
pattern.

### Cascade Flow

```
Training (v16 GA) → needs parameterized trajectories
    ↓ blocked by
Layer 5 → needs to pass trajectory shape through GaitParams
    ↓ blocked by
Layer 4 → needs to accept trajectory shape params in compute()
```

### Layer 4 Fix-Request

**What L4 needs to do** (entirely within L4's scope):

1. Add 3 fields to `GaitParams`:
   ```python
   swing_apex_phase: float = 0.5    # where peak occurs in [0,1]
   swing_height_ratio: float = 1.33 # control point height / step_height
   swing_overshoot: float = 0.0     # m, reach past endpoint
   ```

2. Pass them to `_make_swing_bezier()` in `trajectory.py`:
   ```python
   def _make_swing_bezier(start, end, step_height,
                          apex_phase=0.5, height_ratio=1.33, overshoot=0.0):
   ```

3. Modify Bezier control point placement:
   - **apex_phase**: Split control points asymmetrically. P1 at
     `lerp(start, end, apex_phase * 0.5)`, P2 at
     `lerp(start, end, 1 - (1-apex_phase) * 0.5)`
   - **height_ratio**: `epsilon_vertical = step_height * height_ratio`
     (was hardcoded as `step_height * 4/3`)
   - **overshoot**: Extend `end` by overshoot in the forward direction,
     then use original `end` as P3 (foot pulls back before landing)

4. Default values produce identical output to current code (backward
   compatible). No existing tests should break.

**What L4 does NOT need to know**: Why these params exist, what the GA is,
or anything about the horizon models. L4 just accepts numeric parameters
and produces foot positions.

### Layer 5 Fix-Request

**What L5 needs to do**:

1. Add matching fields to L5's `GaitParams` (in `config/defaults.py`)
2. Pass them through in `behaviors.py:walk()` when constructing GaitParams
3. Default to current values (backward compatible)

### Training (v16 episode runner)

**What training does** (no cascade needed — training repo is independent):

1. Create `episodes/v16.py` with the 19-gene genome
2. Inject trajectory shape genes into GaitParams before sending to L5
3. Run horizon model inference every tick
4. Compute 4-component multiplicative fitness
5. Use existing GA operators (tournament, BLX-α crossover, Gaussian mutation)

---

## GA Parameters

```yaml
ga:
  population_size: 50
  generations: 80
  tournament_size: 3
  crossover_rate: 0.7
  crossover_alpha: 0.5
  mutation_rate: 0.12
  sigma_fraction: 0.10
  sigma_decay: 0.98
  elitism: 2

episode:
  num_targets: 3
  seeds: [42, 137, 256]
  timeout_steps: 6000          # 60s per target
  headless: true
  use_fast_sim: true           # in-process MuJoCo (100× faster)

repair:
  horizon_weight_sum_max: 1.0  # W_100MS + W_200MS + W_500MS ≤ 1.0
  stance_time_min: 0.120       # seconds (biomechanical constraint for 83.5kg)
  step_length_max: 0.45        # IK workspace limit
```

### Initialization

- 1 individual: all defaults (known-good 0.5Hz trot)
- 24 individuals: Gaussian perturbation from defaults (σ=5% of range)
- 25 individuals: uniform random within bounds

This ensures the GA starts near the working trot but explores widely.

### Estimated Runtime

- 50 genomes × 3 seeds × ~30s per episode = ~75 min per generation
- 80 generations = ~100 hours
- With `fast_sim.py` (100× speedup): ~1 hour total
- GPU needed only for horizon model inference (83KB models, negligible)

---

## Validation

After GA converges:

1. **Champion analysis**: Print all 19 genes. Which trajectory shape
   did it find? Does it lean into turns? Which horizon model does it use?

2. **Headed demo**: Run champion in `walk_test.py` — visually inspect
   the trot quality, foot trajectory shape, body lean during turns.

3. **Target game**: Run champion in `run_scenario.py open` and
   `run_scenario.py scattered`. Compare targets/falls/ATO to baseline.

4. **Ablation**: Run champion with horizon weights forced to zero.
   Does performance drop? If so, the predictive models are genuinely
   contributing.

5. **A/B comparison**: Champion vs current 0.5Hz defaults across 12
   seeds. Statistical significance test on targets reached and ATO.

---

## Implementation Order

1. **Cascade down** (Layer 4 fix-request, then Layer 5 fix-request)
2. **Wait for cascade resolution** (L4 accepts trajectory shape params)
3. **Build v16 episode runner** (`training/ga/episodes/v16.py`)
4. **Build genome spec** (`training/configs/b2/ga_v16.yaml`)
5. **Run GA** (fast_sim, ~1 hour)
6. **Validate champion** (headed demo + target game)
7. **Ablation study** (horizon model contribution)

---

## What We'll Learn

| Question | How We'll Know |
|----------|----------------|
| What's the optimal stride shape? | SWING_APEX_PHASE, HEIGHT_RATIO, OVERSHOOT in champion |
| Does body lean help turning? | TURN_LEAN > 0 in champion |
| Do the horizon models help? | W_100MS + W_200MS + W_500MS > 0 in champion |
| Which horizon matters most? | Relative weights in champion |
| What's the speed-stability frontier? | Pareto analysis across top 10 individuals |
| Can we turn faster than differential stride alone? | Turn ability score + actual yaw rate |

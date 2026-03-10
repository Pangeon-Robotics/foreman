# v18 Plan: MJWarp GPU-Accelerated Trot Optimization

**Status**: Draft
**Date**: 2026-03-09

---

## Design Principle

Herbert Simon's insight: evolution's power comes from the number of
selection events, not the complexity of each evaluation. Many fast
generations beat fewer thorough ones. A GA with 1,000 short generations
outperforms one with 100 long generations — each selection event refines
the population, and the compounding effect of many small improvements
dominates.

This means: **small population, short episodes, maximize generations.**

---

## Overview

Evolve a B2 trot gait on GPU using MJWarp (NVIDIA Warp backend for
MuJoCo). One target per episode, 30 seconds, unreachable. The robot is
scored purely on how well it walks — not whether it arrives.

The output is a genome: 13 floats. That genome executes through the
real L5 stack in production.

---

## Genome: 13 Genes (3 groups)

### Gait Timing (4 genes)

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `STEP_LENGTH` | 0.10–0.70 | 0.45 | Stride length (m) |
| `GAIT_FREQ` | 0.3–1.5 | 1.0 | Cadence (Hz) |
| `STEP_HEIGHT` | 0.04–0.15 | 0.10 | Peak foot lift (m) |
| `DUTY_CYCLE` | 0.55–0.80 | 0.70 | Fraction in stance |

### Trajectory Shape (3 genes)

Parameterize L4's cubic Bezier swing curve via `_make_swing_bezier()`.

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `SWING_APEX_PHASE` | 0.2–0.8 | 0.5 | Where peak height occurs in swing |
| `SWING_HEIGHT_RATIO` | 0.8–2.0 | 1.33 | Bezier control point height / step_height |
| `SWING_OVERSHOOT` | 0.0–0.15 | 0.0 | Mid-swing reach past landing point (m) |

### Navigation (3 genes)

| Gene | Range | Default | Purpose |
|------|-------|---------|---------|
| `KP_HEADING` | 0.5–5.0 | 3.0 | Heading proportional gain |
| `WZ_MAX` | 0.5–2.0 | 1.5 | Yaw rate cap (rad/s) |
| `VX_WALK` | 0.3–1.5 | 1.0 | Forward speed command (m/s) |

### Removed from v16/v17

- **Body pose genes** (3): TURN_LEAN, WALK_PITCH, TURN_Y_SHIFT — L5
  doesn't support per-command body pose yet. Dead search dimensions.
- **Horizon model genes** (6): W_100MS, W_200MS, W_500MS,
  HORIZON_VX_SCALE, HORIZON_WZ_SCALE, INSTABILITY_THRESHOLD — neural
  network predictors not integrated into L5. Dead search dimensions.

Removing 9 dead genes shrinks the search space from 19D to 13D. Every
remaining gene directly affects the robot's movement.

---

## Episode Design

**One unreachable target.** Spawned 25–30m away at a random heading
(60°–180° off the robot's initial facing). The robot has 30 seconds.
It will never reach the target — fitness is purely about movement
quality during the walk.

This eliminates:
- Multi-target spawning logic
- Per-target timeout management
- Completion bonuses and the exploitation risks they create
- The distinction between "partial" and "complete" performance

**30 seconds** = 15,000 physics steps (500Hz) = 3,000 control steps
(100Hz).

**1 seed per individual.** Target heading varies per generation
(deterministic from master seed + generation number) so every individual
in a generation faces the same challenge. Different generations see
different headings.

**Fall = zero fitness.** Startup settle: 1.5 seconds standing before
the target appears.

---

## Fitness Function

Per-timestep additive, same 5 components as current `fitness.py`:

```
tick = 4.0 × stride_elegance    # long strides (quadratic) × high foot lift (linear)
     + 2.0 × stability          # 1 - K*(roll² + pitch² + droll² + dpitch²)
     + 2.0 × grip               # fraction of stance feet with slip < 0.15 m/s
     + 1.5 × speed              # velocity toward target / 2.0 m/s
     + 1.0 × turn               # heading error reduction rate / pi rad/s

fitness = mean(tick) × (100 / 10.5)    # scaled to 0-100
fall = 0
```

Stride elegance is constant per genome (computed once from STEP_LENGTH
and STEP_HEIGHT genes). All other components are measured every tick.

Source of truth: `training/ga/episodes/fitness.py`.

---

## GPU Backend: MJWarp

### Benchmarks (RTX 5090, 24GB, B2 model, diverse genomes)

| Worlds | World-steps/s | vs 10-core CPU |
|--------|--------------|----------------|
| 500 | ~200K | 0.6x |
| 1,000 | ~255K | 0.8x |
| 5,000 | ~613K | 1.8x |
| 40,000 | ~791K | 2.4x (plateau) |

Throughput plateaus at ~790K world-steps/s around 40K worlds. OOM at
78K worlds.

### Why MJWarp over MJX

Both are GPU-accelerated MuJoCo. MJWarp uses NVIDIA Warp (CUDA kernels),
MJX uses JAX (XLA). MJWarp is the better fit because:

- Warp is installed and working on our RTX 5090
- No JAX tracing constraints — control logic runs in numpy, only physics
  runs on GPU
- `warp_data.ctrl.assign()` accepts flat numpy arrays — simple interface
- No JIT recompilation between generations

### Architecture

```
Per generation:
  1. Spawn target (same for all individuals)
  2. For each of 3,000 control steps:
     a. Read state from GPU: qpos, qvel, sensordata  (warp → numpy)
     b. Compute ctrl for all 500 individuals in numpy (vectorized)
     c. Write ctrl to GPU: warp_data.ctrl.assign()    (numpy → warp)
     d. Step physics 5×: mjwarp.step()                (GPU)
  3. Compute fitness from accumulated per-tick rewards
  4. Selection + reproduction (CPU, trivial)
```

The control pipeline (step 2b) is a fused numpy implementation of
L5→L4→L3→L2. It runs on CPU but is vectorized across all individuals
using numpy broadcasting. The physics (step 2d) runs on GPU.

### Control pipeline fidelity

The fused numpy pipeline must produce joint targets close to the real
Python stack. Validation protocol:

1. Run 10 genomes through both paths for 3,000 control steps
2. Compare q_target arrays: `max(abs(fused - python)) < 1e-3`
3. Verify GPU champion transfers to CPU full-stack demo

The fused pipeline only needs to be close enough that GPU ranking
correlates with CPU ranking. Exact parity is not required — the genome
runs through real L5 in production.

---

## GA Parameters

```yaml
population: 500
elite_count: 3
fresh_random_fraction: 0.08
tournament_size: 3

reproduction:
  mutation_only: 0.40
  crossover_only: 0.20
  crossover_then_mutate: 0.40
  crossover_method: block_sbx
  sbx_eta: 10

adaptive_mutation:
  sigma_init: 0.10
  sigma_min: 0.01
  decay_rate: 0.997
  stagnation_window_gens: 40
  stagnation_boost: 3.0

termination:
  max_generations: 2000
  early_stop_stagnation_gens: 200

episode:
  duration_seconds: 30
  control_hz: 100
  physics_hz: 500
  num_targets: 1
  target_distance: 25.0
  target_min_turn: 1.047
  target_max_turn: 3.14159
  nominal_z: 0.465
  robot: b2
```

### Initialization

- 1 individual: all defaults (known-good trot)
- 124 individuals: Gaussian perturbation from defaults (sigma = 5% of range)
- 375 individuals: uniform random within bounds

### Timing Estimate

- 500 individuals at ~200K world-steps/s ≈ **37 seconds per generation**
- 2,000 generations × 37s = **~20 hours**
- With early stopping at 200 stagnant generations: likely 8–14 hours

---

## Sim-to-Sim Parity

See `foreman/docs/sim-to-sim-parity.md` for the full checklist. Summary:

| Parameter | CPU value | MJWarp must match |
|-----------|-----------|-------------------|
| Timestep | 0.002s | Explicit in XML |
| Integrator | Euler | Explicit in XML |
| Gravity | (0, 0, -9.81) | Explicit in XML |
| Joint damping | 1.0 | From b2.xml defaults |
| Joint armature | 0.1 | From b2.xml defaults |
| Foot friction | (1.5, 0.005, 0.0001) | From b2.xml geoms |
| Ground friction | (1.0, 0.005, 0.0001) | MuJoCo default |
| Foot geom | sphere r=0.032 | Keep unchanged |

**Float32 vs float64**: MJWarp uses float32, CPU MuJoCo uses float64.
Over 15K steps, drift accumulates. Acceptable for ranking — validated
by champion transfer test.

**No self-collision**: Disabled in MJWarp model. Joint limits prevent
self-intersection.

**No obstacles**: Flat ground plane only. Sphere-plane contact is
MJWarp-supported.

---

## Validation

After GA converges:

1. **Champion transfer**: Run the 13-gene champion through the real
   L5 stack via `python foreman/run_demo.py`. Visually inspect trot
   quality, foot trajectory, turning.

2. **Multi-seed robustness**: Run champion in
   `python foreman/run_scenario.py open` across 12 seeds.
   Expect: 100% targets reached, 0 falls.

3. **Compare to baseline**: A/B against current L5 defaults across
   12 seeds. Statistical comparison of ATO, targets, falls.

4. **Trajectory shape inspection**: Print the 3 Bezier genes. Visualize
   the foot path in `walk_test.py`. Does it look like an elegant trot?

---

## Implementation Order

1. **Create `b2_mjwarp.xml`** — stripped B2 model with explicit
   `<option>` tag. Validate physics match vs `b2.xml`.
2. **Fused numpy control pipeline** — vectorized L5→L4→L3→L2 for
   N individuals. Validate vs Python stack.
3. **MJWarp episode runner** — step loop with ctrl assignment,
   per-tick fitness accumulation, fall detection.
4. **Wire into GA framework** — replace `run_episode()` call in
   `batch.py` with MJWarp batch evaluation.
5. **Smoke test** — 1 generation, 10 individuals. Verify fitness
   scores are reasonable.
6. **Full run** — 500 individuals, 2000 generations.
7. **Validate champion** — transfer test + demo + multi-seed sweep.

Steps 1-2 can run in parallel. Step 3 depends on both.

# GA Framework for Target Game Locomotion Optimization

## Context

The target game demo (`foreman/demos/target_game/`) spawns random targets (3-6m away, forward half-plane) and drives the robot to them using Layers 1-5. Currently it uses hand-tuned constants for both locomotion (Layer 5) and steering (game.py). A genetic algorithm with GPU-accelerated screening can search the ~30-parameter space at population scale.

This work lives in `../training/ga/` (new top-level package alongside `hnn/` and `rl/`). All GA hyperparameters live in one JSON file per robot under `../training/configs/{robot}/ga.json`.

---

## Fitness Function

### Per-timestep terms (summed over episode as area-under-the-curve)

| Term | Formula | Signal |
|------|---------|--------|
| Closing velocity | dot(robot_linvel, unit_vector_to_target) | Positive when moving toward target |
| Heading alignment | reward \|wz\| proportional to \|heading_error\|; penalize low wz when error is large | Drives active correction toward target |
| Body height | \|z - 0.465\| | Penalty for deviation from nominal standing height |
| Body orientation | cos(roll) * cos(pitch) | 1.0 when upright; subsumes fall detection |
| Hip splay | reward slight outward hip angle from nominal | Wider stance improves lateral stability |
| Energy | \|torque * joint_velocity\| per joint, summed | Efficiency penalty |

### Per-episode terms

| Term | Formula | Signal |
|------|---------|--------|
| Target reached | +bonus per target reached within 0.5m | Primary objective (binary per target) |
| Time-to-reach | (episode_duration - reach_time) / episode_duration | Earlier arrival = higher bonus; 0 if not reached |

### Aggregation

```
episode_fitness = w1 * mean(closing_velocity)
               + w2 * mean(heading_alignment)
               - w3 * mean(height_deviation)
               + w4 * mean(body_orientation)
               + w5 * mean(hip_splay)
               - w6 * mean(energy)
               + w7 * targets_reached
               + w8 * time_to_reach_bonus
```

### Two-tier scoring split

- **GPU Tier 1** (coarse screening): scores ONLY on per-episode terms (target reached + time-to-reach)
- **CPU Tier 2** (full fidelity): scores on ALL terms (per-timestep AUC + per-episode)

---

## Two-Tier Architecture

```
                         GENERATION N
 ================================================================

 [Seed / Crossover / Mutation] -> 500 individuals (30 floats each)
              |
              v
 +--------------------------------------------+
 | TIER 1: GPU Coarse Screening (MJX)         |
 |                                            |
 | 500 individuals in parallel on GPU         |
 | Simplified model (capsule collisions)      |
 | Fused JAX control pipeline (L5->L2)        |
 | Fixed 30s episode, SAME target for all     |
 |                                            |
 | Score: target_reached + time_to_reach      |
 | Top 20% (100) -> survive to Gen N+1       |
 | Top 5% (25) -> advance to Tier 2          |
 +--------------------------------------------+
              |
              | 25 individuals (params copied to CPU)
              v
 +--------------------------------------------+
 | TIER 2: CPU Full Fidelity (MuJoCo)         |
 |                                            |
 | 25 individuals, 8 parallel workers         |
 | Full meshes, full collision, real stack     |
 | Same target scenario as Tier 1             |
 |                                            |
 | Score: per-timestep AUC + per-episode      |
 | CPU score determines reproduction rights   |
 +--------------------------------------------+
              |
              v
 [Weighted reproduction] -> 500 individuals for Gen N+1
 ================================================================
```

### Why two tiers

GPU-only misses collision fidelity issues. CPU-only limits population to ~50 (too small for effective search). Two tiers give breadth (500 screened quickly) and depth (25 scored carefully).

---

## GPU Tier 1: MJX

### Why MJX over Warp

MuJoCo MJX (JAX backend) is the production choice: mature API, native `jax.vmap`/`jax.lax.scan` for batching, shares MuJoCo physics semantics, Unitree models readily available. Warp/Newton is too new (announced Jan 2025).

### Simplified robot model

B2's `b2.xml` uses cylinder-box collisions which MJX does not support. Create `training/ga/models/b2_mjx.xml`:
- Replace 5 collision-enabled cylinders (hip, lidar) with capsules (capsule-box IS supported)
- Strip visual-only mesh geoms (no physics contribution, slow JIT)
- Keep: box collision geoms, sphere feet, joints, actuators, sensors
- Validate once by comparing CPU MuJoCo trajectories between `b2.xml` and `b2_mjx.xml`

### Fused JAX control pipeline

Layers 3-5 use Python objects, numpy, and branching — not JAX-traceable. Reimplement the same math as ~200 lines of pure JAX:
- Phase computation: `(t * freq) % 1.0` — trivially traceable
- Swing trajectory: cubic Bezier polynomial — JAX primitives
- IK solver: closed-form trig (atan2, acos) — JAX primitives
- PD control: elementwise `kp * (q_target - q_actual) + kd * (0 - dq_actual)`

Each individual's ~30 parameters are injected as a row in a batched tensor `(500, 30)`. `jax.vmap` handles per-individual variation.

### Episode loop

Use `jax.lax.scan` to fuse 15,000 physics steps into a single compiled kernel (avoids 15k separate GPU launches). Control applied every 5 physics steps (100Hz control, 500Hz physics).

### Minimizing JAX recompilation between generations

The population tensor is always `(500, dim)` — shape never changes. Between generations, the top 20% stay in GPU memory. New offspring are written into the remaining 80% of slots using `jax.lax.dynamic_update_slice` or index assignment on a pre-allocated buffer. This avoids rebuilding the tensor or triggering recompilation.

```
# Pre-allocate once at gen 0
population = jnp.zeros((500, dim))   # fixed shape, never reallocated
sim_states = mjx.make_data(model)    # batched to (500,), reused

# Between generations:
# 1. Top 100 are already in population[top_indices]
# 2. Write 400 new offspring into remaining slots
population = population.at[new_indices].set(offspring_tensor)
# 3. Reset sim states for new individuals only
sim_states = reset_subset(sim_states, new_indices)
```

The JIT-compiled step/control functions see the same shapes every generation — no recompilation after gen 0.

### Fall detection and early termination

The MJX model uses sphere feet (not full meshes), so fallen robots behave very differently between GPU and CPU sims. Fallen individuals are noise — their GPU behavior is meaningless for sim-to-sim transfer. Drop them aggressively:

- **During GPU episode**: monitor `body_z < 0.20` each control step. Mark fallen individuals immediately.
- **Fallen = eliminated**: fitness set to -inf, excluded from top 20% and top 5%. They never reach CPU Tier 2.
- **No partial credit**: a robot that falls at t=5s and one that falls at t=25s are equally worthless — both get -inf. The sim-to-sim gap makes any fallen-robot score unreliable.
- **GPU early exit**: fallen individuals' physics can be skipped (set ctrl=0, don't accumulate fitness) to save compute. Use a boolean mask `alive` that gates the control pipeline.

This is critical for sim-to-sim parity: only upright robots transfer reliably between the simplified MJX model and full CPU MuJoCo.

### Memory and timing

- Per-individual state: ~10 KB. 500 individuals: ~5 MB. Total GPU: ~4-6 GB of 24 GB.
- Tier 1 time: **3-10 seconds** per generation (after first-gen JIT warmup of ~30-60s).

---

## CPU Tier 2: Full Fidelity

Uses the existing `SimulationManager` from Layer 5 — full mesh collisions, DDS, the real L5→L4→L3→L2 stack. No re-implementation needed.

Parameter injection via monkey-patching `config.defaults` + all downstream modules that `from`-imported constants (same patch map as before, but only 25 individuals instead of 500).

Runs 25 individuals on 8 parallel workers via `ProcessPoolExecutor`. Each episode: startup settle, then walk to same target as GPU tier, collecting per-timestep observables (body state, torques, joint velocities) for full AUC fitness.

Tier 2 time: **30-60 seconds** per generation.

### Tier 2 patch map (gene -> modules to patch)

```
BASE_FREQ         -> config.defaults, velocity_mapper
FREQ_SCALE        -> config.defaults, velocity_mapper
MAX_FREQ          -> config.defaults, velocity_mapper
MIN_FREQ          -> config.defaults, velocity_mapper
STEP_LENGTH_SCALE -> config.defaults, velocity_mapper
MAX_STEP_LENGTH   -> config.defaults, velocity_mapper, locomotion
TROT_STEP_HEIGHT  -> config.defaults, velocity_mapper
STAND_TO_TROT     -> config.defaults, gait_selector
TROT_TO_STAND     -> config.defaults, gait_selector
TURN_THRESHOLD    -> config.defaults, locomotion, gait_selector
KP_START/FULL     -> config.defaults, locomotion
KD_START/FULL     -> config.defaults, locomotion
START_RAMP_*      -> config.defaults, transition
STOP_RAMP_*       -> config.defaults, transition
MAX_*_RATE        -> config.defaults, transition
KP_YAW etc.       -> foreman.demos.target_game.game
```

---

## Generation Lifecycle

### 1. Target assignment

All 500 individuals in a generation get the SAME random target (apples-to-apples). Target varies per generation to prevent overfitting. Deterministic: `target = f(master_seed, generation_id)`.

### 2. GPU screening (Tier 1)

Run 500 parallel 30-second episodes on MJX. Score each on per-episode terms only (target reached + time-to-reach). Sort by fitness.

### 3. Selection

- **Drop all fallen individuals** (body_z < 0.20 at any point). Fitness = -inf. They never advance.
- **Top 20% of survivors (100)**: survive to next generation
- **Top 5% of survivors (25)**: advance to CPU Tier 2

### 4. CPU evaluation (Tier 2)

Run 25 individuals through full-fidelity stack. Score on ALL fitness terms (per-timestep AUC + per-episode).

### 5. Reproduction

CPU scores determine how many offspring each individual produces. Better score = more offspring.

### 6. Fill next generation

| Source | Count | Role |
|--------|-------|------|
| Elite clones (CPU rank 1-3, unchanged) | 3 | Preserve best solutions |
| GPU-only survivors (rank 26-100, unchanged) | 75 | Population memory buffer |
| Offspring from CPU-scored parents | 397 | Primary search |
| Fresh random genomes | 25 | Diversity injection (5%) |
| **Total** | **500** | |

---

## Reproduction Strategy

### Offspring allocation

Rank-linear: `offspring(rank) = base + slope * (N_cpu - rank)`, where `base=4` (minimum for every CPU-evaluated parent).

- Rank 1: ~28 offspring
- Rank 13: ~16 offspring
- Rank 25: 4 offspring

Rank-based is preferred over fitness-proportional because locomotion fitness is noisy.

### Reproduction operators

| Operator | Fraction | Description |
|----------|----------|-------------|
| Mutation only | 40% | Gaussian perturbation, sigma = 5% of gene range |
| Crossover only | 20% | SBX (eta=15), two parents by tournament selection |
| Crossover + mutation | 40% | SBX first, then Gaussian mutate |

SBX (Simulated Binary Crossover) chosen over BLX-alpha because it concentrates offspring near CPU-validated parents.

### Elitism

Top 3 from CPU tier pass through unchanged (no mutation). GPU-only survivors (75) carry forward but do NOT reproduce (their GPU scores are too noisy to weight reproduction).

### Adaptive mutation

- **Global decay**: `sigma(gen) = sigma_init * 0.995^gen`, floor at `sigma_min = 0.01`
- **Stagnation detection**: if best fitness improves < 0.5% over 15 generations, triple sigma (up to sigma_init). If stagnation persists 30 generations, inject 15% fresh random for one generation.
- **Per-gene adaptation**: track std of each gene across top 25. Low-variance genes (converged) get smaller sigma. Floor at 0.2x normal sigma so no gene freezes completely.

### Diversity preservation

- 5% fresh random genomes per generation
- Track mean pairwise L2 distance in normalized genome space (sample 200 pairs)
- If diversity drops below floor: double fresh random injection for one generation
- No niching — single-objective task doesn't warrant it

---

## Genome (abstract, swappable)

The genome is an abstraction — a named set of float parameters with bounds. Different experiments can use different genomes (subsets of parameters, different ranges, entirely new parameter sets) without changing the GA framework. The genome definition lives entirely in the JSON config under `"parameters"`. Code never hardcodes parameter names.

```python
# genome.py defines the interface, not the contents
class GenomeSpec:
    """Loaded from JSON config. Defines which parameters to evolve."""
    params: list[ParamSpec]  # name, default, min, max, group
    dim: int                 # len(params) — determines tensor shape

class Genome:
    """One individual: a flat float vector indexed by GenomeSpec."""
    values: jnp.ndarray      # (dim,) on GPU or np.ndarray on CPU
    fitness: float | None
```

The GA framework operates on `(population_size, dim)` tensors. Changing the genome means changing the JSON config — the framework recompiles for the new `dim` on first run, then caches.

### Example genome configs

**"full-30"** (default): All locomotion + steering parameters (~30 genes)
**"steering-only"** (6 genes): Just KP_YAW, HEADING_*, WALK_SPEED_* — fast experiments
**"frequency-sweep"** (4 genes): BASE_FREQ, FREQ_SCALE, MAX_FREQ, MIN_FREQ — isolate one subsystem

Each is a different JSON file. The framework doesn't care which one.

---

## File Structure

```
training/
  ga/
    __init__.py              # Public API
    genome.py                # Genome dataclass, ParamSpec, random/default, clamp, serialize
    fitness.py               # Per-timestep term functions (pure math, no sim)
    operators.py             # tournament_select, sbx_crossover, gaussian_mutate
    tier1_gpu.py             # MJX batched simulation + fused JAX control pipeline
    tier2_cpu.py             # Full-stack CPU evaluation using SimulationManager
    control_jax.py           # JAX-traceable fused L5->L4->L3->L2 pipeline (~200 lines)
    ik_jax.py                # JAX-traceable analytical IK (port of layer_3/ik.py)
    population.py            # Population state, GenerationStats, checkpoint save/load
    pipeline.py              # Main generation loop: Tier 1 -> selection -> Tier 2 -> reproduce
    export.py                # Best genome -> JSON config override
    models/
      b2_mjx.xml             # Simplified B2 for MJX (capsule collisions)
  configs/
    b2/ga.json               # All GA hyperparameters for B2
    go2/ga.json              # All GA hyperparameters for Go2
  tests/
    test_ga_genome.py        # Encode/decode, bounds, serialization
    test_ga_operators.py     # Selection, crossover, mutation
    test_ga_fitness.py       # Reward term calculations
    test_ga_control_jax.py   # JAX pipeline vs Python pipeline comparison
  scripts/
    run_ga.py                # CLI entry point
    validate_mjx_model.py    # Compare b2_mjx.xml vs b2.xml trajectories
```

---

## Implementation Order

| Phase | Files | Notes |
|-------|-------|-------|
| 1. MJX model prep | `models/b2_mjx.xml`, `scripts/validate_mjx_model.py` | Replace cylinders with capsules, validate trajectory match |
| 2. JAX control pipeline | `ik_jax.py`, `control_jax.py` | Port IK + trajectory + velocity mapping to JAX, validate vs Python |
| 3. Pure Python GA core | `genome.py`, `fitness.py`, `operators.py`, `population.py` | No sim dependency, fully testable |
| 4. Tests for core | `test_ga_genome.py`, `test_ga_operators.py`, `test_ga_fitness.py` | Run without simulation |
| 5. Config | `configs/b2/ga.json` | All hyperparameters in one file |
| 6. GPU tier | `tier1_gpu.py` | MJX vmap + lax.scan episode loop |
| 7. CPU tier | `tier2_cpu.py` | SimulationManager + patch map + per-timestep scoring |
| 8. Pipeline + CLI | `pipeline.py`, `export.py`, `scripts/run_ga.py` | Wire tiers together, checkpointing, logging |
| 9. Validation | `test_ga_control_jax.py`, end-to-end smoke test | JAX vs Python match, 1-gen run |

Phases 1-2 can run in parallel with Phase 3. Phases 3-5 have no simulation dependency.

---

## Timing Estimate

| Component | Time per generation |
|-----------|-------------------|
| GPU Tier 1 (500 individuals, 30s episodes) | 3-10 seconds |
| GPU Tier 1 JIT warmup (gen 0 only) | 30-60 seconds |
| Tier 1 -> Tier 2 transfer | < 0.1 seconds |
| CPU Tier 2 (25 individuals, 8 workers) | 30-60 seconds |
| Selection + reproduction | < 0.5 seconds |
| **Total per generation** | **~35-70 seconds** |
| **100 generations** | **~1-2 hours** |
| **500 generations** | **~5-10 hours** |

---

## Configuration (`configs/b2/ga.json`)

Single JSON file containing ALL hyperparameters:

```json
{
  "population": {
    "size": 500,
    "gpu_survival_fraction": 0.20,
    "cpu_evaluation_fraction": 0.05,
    "elite_count": 3,
    "fresh_random_fraction": 0.05
  },
  "episode": {
    "duration_seconds": 30,
    "control_hz": 100,
    "physics_hz": 500,
    "target_min_dist": 3.0,
    "target_max_dist": 6.0,
    "reach_threshold": 0.5,
    "nominal_body_height": 0.465
  },
  "selection": {
    "min_offspring_per_cpu_parent": 4,
    "allocation": "rank_linear",
    "tournament_size": 3
  },
  "reproduction": {
    "operator_fractions": {
      "mutation_only": 0.40,
      "crossover_only": 0.20,
      "crossover_then_mutate": 0.40
    },
    "crossover_method": "sbx",
    "sbx_eta": 15,
    "mutation_method": "gaussian",
    "mutation_sigma_fraction": 0.05
  },
  "adaptive_mutation": {
    "sigma_init": 0.08,
    "sigma_min": 0.01,
    "decay_rate": 0.995,
    "stagnation_window_gens": 15,
    "stagnation_threshold_pct": 0.5,
    "stagnation_boost": 3.0,
    "per_gene_adaptive": true,
    "per_gene_convergence_floor": 0.2
  },
  "diversity": {
    "sample_pairs": 200,
    "diversity_floor": 0.1,
    "low_diversity_random_boost": 0.10
  },
  "fitness_weights": {
    "closing_velocity": 1.0,
    "heading_alignment": 0.5,
    "height_penalty": 2.0,
    "body_orientation": 0.5,
    "hip_splay": 0.3,
    "energy": 0.001,
    "target_reached_bonus": 50.0,
    "time_to_reach_bonus": 10.0
  },
  "termination": {
    "max_generations": 200,
    "early_stop_stagnation_gens": 50
  },
  "checkpoints": {
    "save_every_gens": 10,
    "checkpoint_dir": "models/{robot}/ga_checkpoints"
  },
  "seed": 42,
  "parallel_cpu_workers": 8,
  "parameters": [
    {"name": "BASE_FREQ",     "default": 2.0,  "min": 1.0,  "max": 4.0,  "group": "locomotion"},
    {"name": "FREQ_SCALE",    "default": 4.0,  "min": 1.0,  "max": 8.0,  "group": "locomotion"},
    {"name": "MAX_FREQ",      "default": 4.0,  "min": 2.0,  "max": 6.0,  "group": "locomotion"},
    {"name": "MIN_FREQ",      "default": 1.5,  "min": 0.5,  "max": 3.0,  "group": "locomotion"},
    {"name": "STEP_LENGTH_SCALE", "default": 0.40, "min": 0.10, "max": 0.80, "group": "locomotion"},
    {"name": "MAX_STEP_LENGTH",   "default": 0.30, "min": 0.10, "max": 0.50, "group": "locomotion"},
    {"name": "TROT_STEP_HEIGHT",  "default": 0.06, "min": 0.02, "max": 0.12, "group": "locomotion"},
    {"name": "WALK_STEP_HEIGHT",  "default": 0.04, "min": 0.02, "max": 0.10, "group": "locomotion"},
    {"name": "STAND_TO_TROT",     "default": 0.075,"min": 0.02, "max": 0.20, "group": "locomotion"},
    {"name": "TROT_TO_STAND",     "default": 0.025,"min": 0.005,"max": 0.10, "group": "locomotion"},
    {"name": "TURN_THRESHOLD",    "default": 0.1,  "min": 0.05, "max": 0.50, "group": "locomotion"},
    {"name": "TURN_IN_PLACE_STEP_LENGTH", "default": 0.20, "min": 0.05, "max": 0.30, "group": "locomotion"},
    {"name": "TURN_IN_PLACE_FREQ",        "default": 3.0,  "min": 1.0,  "max": 5.0,  "group": "locomotion"},
    {"name": "TURN_IN_PLACE_STEP_HEIGHT", "default": 0.05, "min": 0.02, "max": 0.10, "group": "locomotion"},
    {"name": "START_RAMP_DURATION",   "default": 0.5, "min": 0.1, "max": 2.0, "group": "locomotion"},
    {"name": "STOP_RAMP_DURATION",    "default": 0.3, "min": 0.1, "max": 1.5, "group": "locomotion"},
    {"name": "MAX_STEP_LENGTH_RATE",  "default": 0.5, "min": 0.1, "max": 2.0, "group": "locomotion"},
    {"name": "MAX_FREQ_RATE",         "default": 4.0, "min": 1.0, "max": 10.0,"group": "locomotion"},
    {"name": "KP_START",  "default": 500.0,  "min": 100.0,  "max": 2000.0,  "group": "locomotion"},
    {"name": "KP_FULL",   "default": 5000.0, "min": 1000.0, "max": 10000.0, "group": "locomotion"},
    {"name": "KD_START",  "default": 25.0,   "min": 5.0,    "max": 100.0,   "group": "locomotion"},
    {"name": "KD_FULL",   "default": 141.4,  "min": 30.0,   "max": 500.0,   "group": "locomotion"},
    {"name": "KP_YAW",             "default": 2.0,  "min": 0.5, "max": 5.0,  "group": "steering"},
    {"name": "HEADING_FULL_SPEED", "default": 0.35, "min": 0.10,"max": 0.80, "group": "steering"},
    {"name": "HEADING_SLOW_SPEED", "default": 0.79, "min": 0.40,"max": 1.50, "group": "steering"},
    {"name": "HEADING_TURN_ONLY",  "default": 1.2,  "min": 0.80,"max": 2.00, "group": "steering"},
    {"name": "WALK_SPEED",         "default": 1.0,  "min": 0.3, "max": 2.0,  "group": "steering"},
    {"name": "WALK_SPEED_MIN",     "default": 0.3,  "min": 0.05,"max": 0.80, "group": "steering"}
  ]
}
```

---

## Output

Best genome exported as JSON:

```json
{
  "robot": "b2",
  "generation": 142,
  "fitness": 287.45,
  "locomotion": { "BASE_FREQ": 2.31, "FREQ_SCALE": 3.85, ... },
  "steering": { "KP_YAW": 2.15, "WALK_SPEED": 1.12, ... }
}
```

---

## Verification

1. **MJX model validation**: Run same joint trajectory on `b2.xml` (CPU) and `b2_mjx.xml` (MJX), compare qpos traces
2. **JAX pipeline validation**: Compare `control_jax.py` output vs Python Layer 3/4/5 output for same inputs
3. **Unit tests** (no sim): `pytest tests/test_ga_genome.py tests/test_ga_operators.py tests/test_ga_fitness.py`
4. **Smoke test** (1 gen): Run with `max_generations: 1` to verify full pipeline wires up
5. **Full run**: `python scripts/run_ga.py --config configs/b2/ga.json` — watch fitness rise
6. **Validate best**: Apply exported params to target game, confirm improvement over hand-tuned defaults

---

## Key References

- `layer_5/config/defaults.py` — Source of truth for all tunable locomotion parameters
- `layer_3/ik.py` — IK solver to port to JAX for GPU pipeline
- `layer_4/generator.py` — Foot position computation to port to JAX
- `layer_5/simulation.py` — CPU Tier 2 entry point (SimulationManager)
- `layers_1_2/unitree_robots/b2/b2.xml` — Robot model to simplify for MJX
- `foreman/demos/target_game/game.py` — Steering logic and game constants

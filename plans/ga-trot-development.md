# GA Trot Development: Consolidation Plan

**Status**: Proposal
**Date**: 2026-03-09

---

## Problem

GA trot development has accumulated 10+ versioned episode runners (v8–v17),
each a standalone 200-450 line file with its own fitness function, genome
injection, navigation loop, and episode structure. This violates three
philosophy principles:

1. **Never duplicate functionality** (engineering.md): The episode loop
   (startup → spawn targets → navigate → measure → score) is copy-pasted
   across every version with minor variations.

2. **Single source of truth** (engineering.md): The v17 fitness formula
   exists in three places — `training/ga/episodes/v17.py` (authoritative),
   `foreman/run_demo.py` (display string), and `foreman/demos/target_game/telemetry.py`
   (field names). These can drift silently.

3. **Short files, one job each** (engineering.md): v16.py is 451 lines.
   Each version file does genome injection AND navigation AND measurement
   AND fitness computation AND episode orchestration.

Additionally, v17 was built without a plan document, no changelog entry,
and no VERSION bump. There is no single place that says "this is the
current fitness function and why."

---

## Proposed Structure

### One episode runner, not ten

Replace `training/ga/episodes/v{8,9,10,11,14,16,17}.py` with:

```
training/ga/episodes/
    episode.py          # The episode runner (navigation + orchestration)
    fitness.py          # Per-timestep reward components (pure math)
    genome_inject.py    # Genome → L5 parameter injection
    __init__.py         # Public API: run_episode, CURRENT_VERSION
```

**`fitness.py`** (~80 lines): The 5 per-timestep reward functions, their
weights, and the aggregation formula. This is the single source of truth.
Any code that wants to display the fitness formula imports from here.

```python
# fitness.py — single source of truth for GA trot fitness

W_STRIDE = 4.0
W_STABILITY = 2.0
W_GRIP = 2.0
W_SPEED = 1.5
W_TURN = 1.0
MAX_TICK_REWARD = W_STRIDE + W_STABILITY + W_GRIP + W_SPEED + W_TURN

def stride_elegance(step_length, step_min, step_max): ...
def stability_reward(roll, pitch, d_roll, d_pitch): ...
def grip_reward(contacts, slip_speeds, threshold=0.15): ...
def speed_toward_target(vx, vy, heading_to_target): ...
def heading_reduction_reward(prev_err, curr_err, dt): ...

def tick_reward(components: dict) -> float:
    """Weighted sum of all components."""
    ...

def episode_fitness(tick_sum, tick_count, fell) -> float:
    """Scale mean tick reward to 0-100. Fall = 0."""
    ...

# For display (run_demo.py imports this instead of hardcoding)
COMPONENT_INFO = [
    ("stride_elegance", W_STRIDE, "quadratic reward for step_length"),
    ("stability",       W_STABILITY, "1 - K*(roll² + pitch² + droll² + dpitch²)"),
    ("grip",            W_GRIP, "fraction of stance feet with slip < 0.15 m/s"),
    ("speed",           W_SPEED, "velocity toward target / V_REF"),
    ("turn",            W_TURN, "heading error reduction rate"),
]
```

**`episode.py`** (~200 lines): The navigation/measurement loop. Calls
fitness.py functions each tick. No fitness math here — just orchestration.

**`genome_inject.py`** (~40 lines): Maps genome dict to L5 module-level
constants. One function.

### One config file per robot

Keep `training/configs/b2/ga.json` (not `ga_v16.json`, `ga_v17.json`).
The config contains the current genome spec and GA hyperparameters.
When the genome changes, edit the file — don't create a new one.

Old configs are in git history. That's what version control is for.

### Changelog in training repo

```
training/ga/CHANGELOG.md
```

Records what changed per version. Short entries, not full plan documents.
This replaces the scattered `foreman/plans/v{N}-*.md` files for GA-internal
changes. Foreman plans are only needed when a GA change requires a
cross-layer cascade (like v16's swing trajectory).

Example:

```markdown
## v17 — 2026-03-08
- Fitness: per-timestep additive (was episode-level multiplicative)
- STEP_LENGTH range expanded to 0.70m
- Added stride_elegance component (weight 4.0)
- Removed completion bonus (phase 1 of two-phase redesign)
- 140 generations trained, champion fitness 53.0

## v16 — 2026-03-08
- 19-gene genome: gait(4) + trajectory(3) + body_pose(3) + nav(3) + horizon(6)
- Cascade: parameterized swing trajectory (L4 Bezier control points)
- Multiplicative fitness: speed × stability × grip × turn × stride
- Horizon model integration (100ms/200ms/500ms predictors)
```

### Foreman plan documents: only for cascades

`foreman/plans/` should only contain plans that involve foreman's job —
cross-layer coordination. The existing cascade plan
(`2026-03-parameterized-swing-trajectory.md`) is correct. The GA-internal
plans (`v12-sovereign-genome.md`, `v14-completion-gated-fitness.md`) should
move to `training/ga/docs/` or just be entries in the changelog.

### Delete dead episode code

v8, v9, v10, v11, v14 are dead. Delete them. They're in git history.
v16 is superseded by v17's per-timestep approach. Keep v16 only if we
might revert to multiplicative fitness — but even then, git history.

### run_demo.py imports fitness info

```python
# run_demo.py — no more hardcoded fitness formula
from training.ga.episodes.fitness import COMPONENT_INFO

def _print_genome_info(genome_path):
    ...
    for name, weight, description in COMPONENT_INFO:
        print(f"    {weight:.1f} x {name:20s} {description}")
```

### telemetry.py field names match fitness.py

The TickSample fields (`stability`, `grip`, `speed`, `turn`,
`stride_elegance`) are already named correctly. Just remove the
`# v17 fitness components` comment — they're just "fitness components."

---

## What This Removes

| Before | After |
|--------|-------|
| 7 episode files (2502 lines) | 3 files (~320 lines) |
| 7 config files (ga_v{N}.json) | 1 config file (ga.json) |
| Fitness formula in 3 places | Fitness formula in 1 place |
| No changelog | CHANGELOG.md |
| GA-internal plans in foreman/ | Plans only for cascades |

---

## What This Does NOT Change

- The GA framework itself (genome.py, operators.py, population.py, pipeline.py)
- The per-timestep fitness approach (v17's design is good)
- The 19-gene genome structure
- The cascade pattern for cross-layer changes
- Foreman's role (coordination, not implementation)

---

## Implementation

This work is entirely in the training repo, except for:
1. `foreman/run_demo.py` — import fitness info instead of hardcoding
2. `foreman/demos/target_game/telemetry.py` — remove version comment
3. Move `foreman/plans/v{12,13,14}-*.md` to `training/ga/docs/`

The training repo changes should be done by a subagent in that repo.

---

## Going Forward

When evolving the trot GA:

1. **Change fitness?** Edit `training/ga/episodes/fitness.py`. Add entry
   to `training/ga/CHANGELOG.md`.
2. **Change genome?** Edit `training/configs/b2/ga.json`. Add entry to
   changelog.
3. **Need layer changes?** Create a cascade in `foreman/cascades/active/`.
   This is the only case where foreman gets a plan document.
4. **Run training?** `python training/scripts/run_ga.py --config configs/b2/ga.json`
5. **View champion?** `python foreman/run_demo.py` (always uses latest).

One place to change, one place to record, one place to run.

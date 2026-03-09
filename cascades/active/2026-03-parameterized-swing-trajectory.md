# Cascade: Parameterized Swing Trajectory

**Cascade ID**: `2026-03-parameterized-swing-trajectory`
**Status**: `complete`
**Created**: 2026-03-08
**Completed**: 2026-03-08

---

## Overview

**Initiated by**: User (v16 GA trot optimization plan)
**Initiating layer**: Training (GA needs evolvable foot trajectory shape)
**Original issue**: L4's Bezier swing trajectory has hardcoded symmetric
control points. The GA needs to evolve trajectory shape (apex timing,
arch curvature, overshoot) to find optimal foot paths.

---

## Cascade Flow

### Training (Initiating — not a layer, no fix-request needed)

- **Status**: `unblocked`
- **Description**: v16 GA episode runner needs to set trajectory shape
  genes in GaitParams and have them affect the actual foot path in L4.

### Layer 5 (Intermediate)

- **Status**: `complete` (commit `232d3c3`)
- **Description**: Pass trajectory shape params through L5's GaitParams
  to L4. Add fields: `swing_apex_phase`, `swing_height_ratio`,
  `swing_overshoot`. Default values produce identical output.
- **Requires from Layer 4**: L4's GaitParams and `compute()` accept
  these 3 new fields.
- **Scope**: Add fields to L5's GaitParams copy, pass through in
  behaviors.py walk(). No behavioral changes.

### Layer 4 (Resolving)

- **Status**: `complete` (commit `2de4764`)
- **Description**: Add 3 trajectory shape parameters to GaitParams and
  wire them into the Bezier swing trajectory generator.
- **Scope**: Entirely within Layer 4.
  1. Add `swing_apex_phase`, `swing_height_ratio`, `swing_overshoot`
     to `GaitParams` dataclass (with backward-compatible defaults)
  2. Pass them from `compute()` to `swing_position()` / `_make_swing_bezier()`
  3. Modify `_make_swing_bezier()` to use asymmetric control points
  4. All existing tests must pass with default values

---

## Resolution Timeline

### Downward Cascade

- [x] Layer 5 files fix-request to Layer 4
- [x] Layer 4 begins implementation

### Upward Resolution

- [x] Layer 4 completes, closes issue
- [x] Layer 5 unblocks, adds fields, closes issue
- [ ] Training unblocks, builds v16 episode runner

### Validation

- [x] L4 unit tests pass (default values = identical output) — 118 passed
- [x] L5 unit tests pass — 176 passed (5 pre-existing velocity_mapper failures unrelated)
- [ ] walk_test.py still works (no regression)
- [ ] Non-default values produce visibly different foot trajectories

---

## Notes

### Design Constraint

L4 is stateless. The new parameters are per-tick inputs, not accumulated
state. This is consistent with L4's architecture — it just receives
numeric parameters and produces foot positions.

### Backward Compatibility

All 3 new fields default to values that reproduce current behavior:
- `swing_apex_phase=0.5` (symmetric)
- `swing_height_ratio=1.33` (current `4/3` factor)
- `swing_overshoot=0.0` (no overshoot)

No existing code changes behavior unless it explicitly sets these fields.

---

**Tracked by**: Foreman
**Last updated**: 2026-03-08 (cascade complete, training unblocked)

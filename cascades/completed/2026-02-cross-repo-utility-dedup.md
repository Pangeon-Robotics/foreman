# Cascade: Cross-Repo Utility Deduplication

**Cascade ID**: `2026-02-cross-repo-utility-dedup`
**Status**: `complete`
**Created**: 2026-02-18
**Completed**: 2026-02-18

---

## Overview

**Initiated by**: Philosophy audit of v6 fitness function code
**Initiating repo**: training
**Original issue**: Quaternion helpers (`_quat_to_yaw`, `_quat_to_rpy`, `_normalize_angle`, `_clamp`) and `_patch_layer_configs` are duplicated between foreman and training repos. The episode.py copy comments "copied from game.py to avoid import dependency" and the `_patch_layer_configs` copy has already diverged (episode.py added error handling that foreman lacks).

---

## Cascade Flow

### training (Initiating)

- **Status**: `complete`
- **Description**: training/ga/episode.py contained copied helpers from foreman/demos/target_game/. Replaced with aliased imports from `foreman.demos.target_game.utils`.
- **Functions removed**: `_quat_to_yaw`, `_quat_to_rpy`, `_normalize_angle`, `_clamp`, `_patch_layer_configs`, `_load_by_path`
- **Replaced with**: `from foreman.demos.target_game.utils import ... as _...`

### foreman (Resolving)

- **Status**: `complete`
- **Description**: Extracted shared cross-layer utilities into `foreman/demos/target_game/utils.py`. Updated `game.py` and `__main__.py` to import from utils.py.
- **Scope**: Entirely within foreman's scope (cross-layer coordination utilities)
- **New file**: `foreman/demos/target_game/utils.py` with 6 public functions:
  - `quat_to_yaw(quat)` — extracted from game.py
  - `quat_to_rpy(quat)` — adopted from training
  - `normalize_angle(angle)` — adopted from training
  - `clamp(value, lo, hi)` — extracted from game.py
  - `load_module_by_path(name, path)` — extracted from __main__.py
  - `patch_layer_configs(robot, workspace_root)` — merged best of both versions (training's improved error handling + workspace_root parameter)

---

## Resolution Timeline

### Downward Cascade

- [x] 2026-02-18: training identifies cross-repo duplication via philosophy audit
- [x] 2026-02-18: foreman begins utility extraction

### Upward Resolution

- [x] 2026-02-18: foreman completes utility extraction, updates game.py and __main__.py
- [x] 2026-02-18: training unblocks, imports from foreman, removes 6 duplicated functions

### Validation

- [x] 2026-02-18: `foreman.demos.target_game.utils` imports verified (6 functions)
- [x] 2026-02-18: `foreman.demos.target_game.game` imports verified (TargetGame)
- [x] 2026-02-18: `training.ga.episode` imports verified (aliased names)
- [x] 2026-02-18: Same-object identity confirmed (single source of truth)
- [x] 2026-02-18: All 41 training tests pass
- [x] 2026-02-18: Zero local definitions of duplicated functions remain in training/

---

## Notes

### Key Insights
- The comment "copied from game.py to avoid import dependency" was a false constraint. training already has workspace root on sys.path (for `foreman.demos.target_game.target` imports), so the dependency is valid.
- training's `_patch_layer_configs` had improved error handling (FileNotFoundError, workspace_root parameter) that foreman lacked. The cascade flowed improvements BACK to foreman (upward fix), not just importing foreman's code.
- `_quat_to_rpy` and `_normalize_angle` existed only in training. By adopting them into foreman's utils.py, both repos now have access without duplication.

### Cascade Summary

**Total repos involved**: 2 (foreman, training)
**Duration**: same-session
**Files created**: 1 (`foreman/demos/target_game/utils.py`)
**Files modified**: 3 (`game.py`, `__main__.py`, `episode.py`)
**Functions deduplicated**: 6 (`quat_to_yaw`, `quat_to_rpy`, `normalize_angle`, `clamp`, `load_module_by_path`, `patch_layer_configs`)

**Outcome**: Success

---

**Tracked by**: Foreman
**Last updated**: 2026-02-18

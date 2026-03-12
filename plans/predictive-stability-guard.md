# Predictive Stability Guard

**Status**: Draft
**Date**: 2026-03-12
**Depends on**: None (uses existing trained models + existing L5 architecture)

---

## Problem

Stride differential turning destabilizes B2 at high yaw rates. L4 applies
`right_scale = 1.0 + wz * 0.5`, `left_scale = 1.0 - wz * 0.5` to step_length.
L5's `_clamp_wz()` protects the short-side minimum (>= 0.05m) but does NOT
limit the long-side maximum or the stride ratio between legs.

**Observed failure** (2026-03-12 test): wz=1.67, step_length=0.30m produces
long_side=0.55m (57% over MAX_STEP_LENGTH), short_side=0.05m, ratio=11:1.
Robot fell at 11s.

Static wz limits can't solve this because the safe envelope depends on
dynamic body state — the same wz is fine during straight-line walking but
fatal mid-turn when roll is already 3-4 degrees. We need a predictor.

---

## Available Assets

Three MLP body-state predictors already trained and saved:

| Model | File | Size | Horizon |
|-------|------|------|---------|
| 100ms | `tmp/models/predictor_100ms.pt` | 85KB | 10 ticks |
| 200ms | `tmp/models/predictor_200ms.pt` | 85KB | 20 ticks |
| 500ms | `tmp/models/predictor_500ms.pt` | 85KB | 50 ticks |

**Architecture**: 21D input → 128 SiLU → 128 SiLU → 6D output (delta body state).
Learnable per-dimension precision weights. Precision-weighted MSE loss.

**Input vector** (21D):
- body_state (6D): vx, vy, wz, roll, pitch, yaw
- gait_cmd (5D): step_length, gait_freq, step_height, wz_cmd, body_height
- foot_contacts (4D): FR, FL, RR, RL
- imu (6D): gyro_xyz, accel_xyz

**Output** (6D): delta body_state at t+horizon (residual from current).

**Training data**: Collected from varied gait episodes (ramp up/down, constant
speeds, turns, mixed, sinusoidal). wz range in training: [-0.5, 0.5]. The
training data does NOT cover the high-wz regime (1.0+) where falls occur.

---

## Design

### Where It Lives

**Layer 5** — specifically in `locomotion.py:_walk_pipeline()`, between the
turn-speed coupling and the call to `_behavior_walk()`. This is the natural
insertion point because:

1. L5 already owns the turn-speed coupling (quadratic attenuation)
2. L5 has access to both the raw command AND the computed effective_speed
3. L5 can attenuate both vx and wz before they reach gait parameter generation
4. No layer boundary violations — predictor runs inside L5's own pipeline

The predictor models are PyTorch but inference is CPU-only, ~0.1ms per forward
pass, negligible at 100Hz.

### Runtime Flow

```
Navigator: vx=2.0, wz=1.67
  ↓
L5 _walk_pipeline():
  1. Turn-speed coupling:  effective_speed = 2.0 * 0.51 = 1.02 m/s
  2. Map to gait:          step_length=0.30, freq=3.0
  3. _clamp_wz:            wz=1.67 (passes — short_side=0.05 ≥ 0.05)
  4. ★ PREDICTOR GUARD ★
     - Build 21D observation from current state + proposed gait
     - Run 100ms model → predicted roll_delta, pitch_delta
     - instability = sqrt(pred_roll² + pred_pitch²)
     - If instability > threshold: attenuate wz (primary) and vx (secondary)
     - Recompute step_length after attenuation if needed
  5. Build GaitParams with safe wz and step_length
  ↓
L4: stride differential on safe parameters
```

### Attenuation Strategy

**Primary: reduce wz.** The stride ratio is directly proportional to wz.
Cutting wz from 1.67 to 0.8 changes the ratio from 11:1 to 3.6:1. This is
the most effective lever.

**Secondary: reduce effective_speed.** Lower speed → shorter step_length →
less absolute difference between long and short sides. Only needed when
wz reduction alone is insufficient.

```python
# Pseudo-code for the guard (inside _walk_pipeline)
if predictor is not None:
    obs = build_observation(body_state, gait_cmd, foot_contacts, imu)
    pred_delta = predictor(obs)  # 6D: [dvx, dvy, dwz, droll, dpitch, dyaw]

    pred_roll = current_roll + pred_delta[3]
    pred_pitch = current_pitch + pred_delta[4]
    instability = sqrt(pred_roll**2 + pred_pitch**2)

    if instability > INSTABILITY_THRESHOLD:
        excess = (instability - INSTABILITY_THRESHOLD) / INSTABILITY_THRESHOLD
        # Attenuate wz first (most effective)
        wz_scale = max(0.3, 1.0 - WZ_ATTENUATION * excess)
        command_wz *= wz_scale
        # Attenuate speed only if instability is severe
        if instability > SEVERE_THRESHOLD:
            vx_scale = max(0.5, 1.0 - VX_ATTENUATION * excess)
            effective_speed *= vx_scale
```

### Thresholds (Initial, Tunable)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `INSTABILITY_THRESHOLD` | 0.05 rad (~3°) | Below this, no intervention. Normal walking has ~1-2° roll/pitch. |
| `SEVERE_THRESHOLD` | 0.10 rad (~6°) | Above this, also reduce speed. 6° is approaching fall zone. |
| `WZ_ATTENUATION` | 0.5 | At 2× threshold, wz halved. Proportional. |
| `VX_ATTENUATION` | 0.3 | Gentler. Speed loss hurts ATO. |
| `MIN_WZ_SCALE` | 0.3 | Never reduce wz below 30% — robot must still turn. |
| `MIN_VX_SCALE` | 0.5 | Never reduce speed below 50%. |

### Observation Assembly

L5 needs body state and IMU data that it currently does NOT have access to.
Two options:

**Option A — SimulationManager query (quick, sim-only)**:
`SimulationManager.get_robot_state()` returns body velocity, IMU, foot contacts.
Add a `set_robot_state()` method on Locomotion that the SimulationManager calls
each tick before `update()`. ~5 lines of plumbing.

**Option B — MotionCommand extension (hardware-compatible)**:
Add optional sensor fields to MotionCommand. Layer 6 (or SimulationManager)
populates them from available sensors. L5 reads them if present. This is the
long-term correct approach — L5 gets sensor data through its interface contract,
not through simulation back-channels.

**Recommendation**: Start with Option A for rapid validation. Migrate to
Option B once the predictor proves useful.

---

## Training Data Gap

The existing training data only covers wz=[-0.5, 0.5]. The dangerous regime
is wz > 1.0. The predictor will be extrapolating into unknown territory.

### Fix: Collect High-wz Training Data

Extend `collect_training_data.py` with aggressive turn episodes:

```python
# High-wz episodes (the dangerous regime)
for vx in [0.3, 0.5, 0.8, 1.0]:
    for wz in [-1.5, -1.0, -0.8, 0.8, 1.0, 1.5]:
        episodes.append((f"hard_turn_vx{vx}_wz{wz}", [("const", vx, wz, 15)]))

# Ramp wz from 0 to max (sweep the transition)
episodes.append(("wz_ramp", [("ramp_wz", 0.5, 0.0, 1.5, 40)]))

# Mixed: walk → hard turn → walk (transition dynamics)
episodes.append(("mixed_hard_turn", [
    ("const", 0.8, 0.0, 10),
    ("const", 0.8, 1.2, 10),
    ("const", 0.8, 0.0, 10),
    ("const", 0.8, -1.2, 10),
]))
```

**Critical**: Some high-wz episodes will cause falls. The predictor MUST see
pre-fall dynamics to learn the instability signature. Run data collection with
fall detection disabled (or at least continue recording for 2-3s after fall
onset).

### Retrain

After collecting expanded data, retrain all 3 models:
```bash
python foreman/collect_training_data.py --episodes 3
python foreman/train_predictor.py --epochs 500
```

---

## Implementation Phases

### Phase 1: Data Collection (30 min)

1. Add high-wz episodes to `collect_training_data.py`
2. Add `ramp_wz` segment type (ramp wz from start to end at constant vx)
3. Run collection: ~15 episodes × 15-40s = ~7 min simulated, ~20 min wall
4. Retrain models on expanded dataset

### Phase 2: L5 Integration (1-2 hours)

1. Add `StabilityGuard` class to new file `layer_5/stability_guard.py`:
   - Loads predictor model from path
   - `predict(obs_21d) → instability_scalar`
   - `attenuate(wz, effective_speed, instability) → (safe_wz, safe_speed)`
2. Add `set_robot_state(body_state, imu, foot_contacts)` to `Locomotion`
3. Insert guard call in `_walk_pipeline()` between turn-speed coupling and
   `_behavior_walk()`
4. Guard is optional — `Locomotion(predictor_path=None)` = no guard (default)

### Phase 3: SimulationManager Plumbing (30 min)

1. `SimulationManager.send_motion_command()` already calls `get_robot_state()`
   for telemetry. Route that data to `Locomotion.set_robot_state()` before
   calling `update()`.
2. Pass predictor model path to SimulationManager → Locomotion constructor.

### Phase 4: Validation (1-2 hours)

1. Run `fast_test.py` with predictor enabled vs disabled (A/B on 12 seeds)
2. Key metrics:
   - Falls: should drop from ~24% failure rate to <5%
   - ATO: should improve (fewer wasted steps on recovery/timeouts)
   - Targets: should match or exceed baseline
3. Run `run_scenario.py scattered` to verify obstacle navigation unaffected

### Phase 5: Threshold Tuning (optional, 1 hour)

If Phase 4 shows the predictor is too aggressive (kills speed when safe)
or too conservative (still falls):
- Sweep INSTABILITY_THRESHOLD: [0.03, 0.05, 0.08, 0.10]
- Sweep WZ_ATTENUATION: [0.3, 0.5, 0.7]
- Use `fast_test.py -n 8` for each combo

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Predictor extrapolates poorly beyond wz=0.5 | High | Phase 1 collects high-wz data; retrain |
| Predictor too slow (>1ms) | Low | 85KB MLP, CPU inference ~0.1ms |
| PyTorch dependency in L5 | Medium | Numpy-only inference (Option 3 below). Export weights to .npz, no torch needed at runtime |
| Predictor over-attenuates (kills speed) | Medium | MIN_WZ_SCALE=0.3, MIN_VX_SCALE=0.5 prevent total kill. Phase 5 tunes thresholds |
| Training data episodes cause irreversible falls | Low | Settle between episodes; re-start sim if needed |

### PyTorch in L5

L5's CLAUDE.md says "numpy and pytest only". Adding torch as an optional
dependency for the stability guard is a scope expansion. Options:

1. **torch optional**: `try: import torch; except: guard = None`. No torch = no guard. Tests still run without torch.
2. **ONNX Runtime**: Export models to ONNX, use onnxruntime (lighter than torch). Still a new dep.
3. **Numpy-only inference**: Convert trained weights to numpy arrays, implement forward pass in 10 lines of numpy. No new dependencies. Slightly more work but cleanest.

**Recommendation**: Option 3. The model is 2 linear layers with SiLU activation.
Numpy inference:

```python
def _silu(x):
    return x * (1 / (1 + np.exp(-x)))

def predict(x, w1, b1, w2, b2, w3, b3):
    h = _silu(x @ w1.T + b1)
    h = _silu(h @ w2.T + b2)
    return h @ w3.T + b3
```

Export weights from PyTorch checkpoint to .npz at training time. L5 loads .npz.
Zero new dependencies.

---

## Success Criteria

1. **Falls**: < 5% seed failure rate (currently ~24%)
2. **ATO**: >= 70 mean (currently ~77 when no falls, ~56 overall)
3. **No regression**: Open-field targets=4/4, falls=0 maintained
4. **Scattered**: targets >= 3/4 mean, falls <= 1 mean

---

## What We'll Learn

| Question | How We'll Know |
|----------|----------------|
| Can 85KB MLPs predict instability 100-500ms ahead? | Prediction error on high-wz test set |
| Does predictive attenuation prevent stride-ratio falls? | Fall rate in Phase 4 A/B test |
| Which horizon is most useful for stability? | Ablation: 100ms-only vs 200ms-only vs 500ms-only |
| Should training data cover near-fall dynamics? | Compare retrained vs original model prediction quality |

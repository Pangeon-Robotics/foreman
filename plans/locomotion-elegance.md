# Locomotion Elegance Improvement Plan

## Observations (from demo of GA champion gen 9, fitness 9018.76)

1. **Vibrating feet** — robot starts by shuffling/vibrating feet, especially during turns
2. **Walks slowly** — actual forward velocity much lower than commanded 2 m/s
3. **Short inelegant steps** — GA evolved STEP_LENGTH_SCALE to minimum (0.05), producing tiny shuffling gait
4. **Poor turn-in-place** — wz-command approach is open-loop and jittery vs reference's GA-evolved keyframe poses

## Root Causes

| Issue | Root Cause | Current Value | Reference Value |
|-------|-----------|---------------|-----------------|
| Shuffling | STEP_LENGTH_SCALE at min | 0.05 | 0.20 (fixed stride) |
| Slow walking | Tiny steps limit actual velocity | stride 0.05m @ 2.4Hz = 0.12 m/s | stride 0.20m @ 3.3Hz = 0.66 m/s |
| High step height | Causes landing oscillation | 0.06m | 0.04m |
| Poor turns | Open-loop wz through DDS | Differential stride via firmware | GA-evolved 2-pose keyframes |
| Abrupt transitions | No smoothing between gait states | Instant parameter switch | Smoothstep blending over 0.25s |

## Phase 1: Layer 5 Parameter Tuning (quick wins)

Changes in `layer_5/config/defaults.py` and `layer_5/velocity_mapper.py`:

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| TROT_STEP_HEIGHT | 0.06 | 0.04 | Match reference; reduce landing oscillation |
| STEP_LENGTH_SCALE | 0.40 | 0.20 | With lower minimum range, linear scaling from speed |
| STEP_LENGTH_RANGE min | 0.05 | 0.12 | Eliminate shuffling at low speeds |
| MAX_STEP_LENGTH | 0.30 | 0.25 | Slightly conservative for stability |
| BASE_FREQ | 2.0 | 1.8 | Slightly slower base cadence for longer strides |
| FREQ_SCALE | 4.0 | 2.5 | Less aggressive frequency ramp |
| DUTY_CYCLES['trot'] | 0.5 | 0.55 | More ground contact time = more stability |
| TURN_IN_PLACE_STEP_HEIGHT | 0.05 | 0.03 | Lower lift during turns reduces vibration |

## Phase 2: Target Game Steering Improvements

Changes in `foreman/demos/target_game/game.py`:

1. **Increase wz clamp** — currently hardcoded `clamp(KP_YAW * heading_err, -1.0, 1.0)`. Raise to `-2.0, 2.0` for more aggressive turning authority
2. **Add hysteresis to turn/walk transitions** — reference uses 29/40 degree gap to avoid oscillation between modes
3. **Tune speed profile** — current linear interpolation between HEADING_FULL_SPEED and HEADING_SLOW_SPEED could use smoothstep

## Phase 3: Keyframe Turn-in-Place (future)

Port the reference's 2-pose GA-evolved turn approach:
- Requires new "turn_keyframe" behavior in MotionCommand
- Layer 5 would bypass normal gait pipeline and send direct joint targets
- Needs GA evolution of turn poses for our specific PD gains and robot config
- **Defer** — significant architectural work; focus on parameter tuning first

## Validation Criteria

Run target game after each phase:
```bash
python -m foreman.demos.target_game --robot b2 --targets 3 --seed 42 --domain 2
```

Success = visually smooth gait with longer strides, faster target reach times, stable turns.

Compare with current champion baseline: 3/3 targets in 43.8s (seed 42).

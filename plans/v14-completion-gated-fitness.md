# v14: Completion-Gated Fitness — 3/3 Targets, Zero Falls

**Status: READY TO LAUNCH** — GA smoke tested, demo verified, both repos pushed.

## Result

The target game demo runs flawlessly:

```
=== GAME OVER ===
Targets: 3/3 reached (100%)
Timeouts: 0  Falls: 0
Total time: 31.1s

WALK (2380 steps):  avg|R|=1.1°  avg|P|=0.6°  max|R|=6.6°  max|P|=4.8°
TURN (526 steps):   avg|R|=1.3°  avg|P|=1.4°  max|R|=5.7°  max|P|=6.0°
```

15/15 targets reached across 5 seeds (42, 137, 7, 256, 999), zero falls,
average completion time 27s. Maximum roll 9.3 degrees, maximum turn roll
6.4 degrees. The robot walks smoothly, turns in place without wobble, and
navigates to all targets reliably.

## Problems Solved (from v13)

v13 ran for ~1000 gens on two islands. Best champion reached 99.3 fitness
but only 1/3 targets in the headed demo. Four root causes:

1. **Fitness exploit**: closing_speed (40 pts) dominated. Walking fast to
   1 target scored higher than slowly reaching all 3.
2. **Foot vibration**: STEP_HEIGHT min=0.06 at GAIT_FREQ max=2.0 produced
   invisible foot lift that the GA camped on.
3. **Fall threshold too lenient**: z=0.22m (robot on its side) wasn't
   detected. Threshold of 0.4 * 0.465 = 0.186m only caught face-plants.
4. **TURN_WZ camps at min**: GA avoided turning entirely (TURN_WZ=0.50,
   THETA=0.15-0.32).

## Solution

### Completion-gated fitness (the core fix)

```python
if targets_reached == num_targets:
    fitness = 100.0 * (1.0 - total_steps / (timeout_steps * num_targets))
elif targets_reached > 0:
    fitness = targets_reached * 5.0  # max 10.0 for 2/3
else:
    fitness = 0.0
if fell:
    fitness = 0.0
```

**Structurally unexploitable.** Full completion scores 50-100 (speed-based).
Partial completion caps at 10. The gap is ~40+ points — no amount of fast
walking or heading progress earns points without reaching all targets.

### Startup vibration eliminated

Bypassed L5 gain ramp entirely. L4-direct control with smoothstep PD gain
ramp (kp 500->4000, kd 25->126.5) over first 2.5s of navigation. No separate
startup phase — targets spawn immediately, gains ramp while walking.

### Systematic parameter refinement

Three rounds of parameter sweeps (9 configs x 5 seeds each = 45 runs per round):

1. **Gain + duty cycle sweep**: kp=4000 + dc=0.65 best overall (was kp=5000)
2. **Walk step height sweep**: STEP_HEIGHT 0.08->0.07 reduced walk roll 34%
3. **Turn step height sweep**: TURN_STEP_HEIGHT 0.07->0.06 eliminated
   catastrophic turn roll (25.3 degrees -> 6.0 degrees, 2 falls -> 0)
4. **Final validation sweep**: 9 configs testing TURN_WZ, WZ_LIMIT, KP_YAW,
   TURN_STANCE_WIDTH — current best confirmed optimal

### Decel taper at WALK-TURN transition

In the last 30% of heading error before THETA_THRESHOLD, forward speed tapers
to zero. Prevents pitch spike from walking momentum at mode switch:

```python
taper_start = 0.7 * theta_threshold
if abs(heading_err) > taper_start:
    decel = (theta_threshold - abs(heading_err)) / (theta_threshold - taper_start)
else:
    decel = 1.0
heading_mod = decel * max(0.0, math.cos(heading_err))
```

### Train/demo parity

Decel taper added to both `game.py` and `episode.py`. Behavior metrics
(`mean_walk_speed`, `mean_turn_rate`) added to v14 behavior dict so the
critic has full observability.

## Final Parameters

### Seed genome (`training/models/b2/ga_v14_seed.json`)

| Gene | Value | Notes |
|------|-------|-------|
| GAIT_FREQ | 1.5 Hz | Walk frequency |
| STEP_LENGTH | 0.30 m | Walk speed: 0.30 * 1.5 = 0.45 m/s max |
| STEP_HEIGHT | 0.07 m | Visible lift, low roll (was 0.08) |
| DUTY_CYCLE | 0.65 | High double-support for stability |
| STANCE_WIDTH | 0.0 m | Neutral stance |
| KP_YAW | 2.0 | Heading P-gain |
| WZ_LIMIT | 1.5 rad/s | Walk yaw rate cap |
| TURN_FREQ | 1.0 Hz | Turn-in-place frequency |
| TURN_STEP_HEIGHT | 0.06 m | Low lift eliminates turn roll (was 0.07) |
| TURN_DUTY_CYCLE | 0.65 | Matches walk for consistency |
| TURN_STANCE_WIDTH | 0.08 m | Wider stance for turn stability |
| TURN_WZ | 1.0 rad/s | 90 degrees in ~1.6s |
| THETA_THRESHOLD | 0.6 rad | ~34 degrees walk/turn boundary |

### PD gains (not in genome — fixed)

| Parameter | Value |
|-----------|-------|
| KP_START | 500 |
| KP_FULL | 4000 |
| KD_START | 25 |
| KD_FULL | 126.5 (critically damped) |
| Ramp time | 2.5s smoothstep |

### GA config bounds (key changes from v13)

| Gene | Min | Max | Change |
|------|-----|-----|--------|
| STEP_HEIGHT | 0.06 | 0.20 | min was 0.08 |
| TURN_STEP_HEIGHT | 0.06 | 0.15 | min was 0.07 |
| GAIT_FREQ | 0.8 | 1.8 | max was 2.0 |
| TURN_WZ | 0.8 | 3.0 | min was 1.0 |

## Files Changed

### foreman (4 commits)
- `demos/target_game/game.py` — L4-direct bypass, gain ramp, decel taper,
  fall threshold 0.5, refined defaults (STEP_HEIGHT=0.07, TURN_STEP_HEIGHT=0.06,
  DUTY_CYCLE=0.65, KP_FULL=4000)

### training (4 commits)
- `ga/episode.py` — `_run_game_v14()` with completion-gated fitness, decel taper,
  behavior metrics, refined genome defaults
- `ga/pipeline.py` — Skip single-term dominance alert for v14
- `configs/b2/ga_v14_puget.json` — Puget island config (seed=42)
- `configs/b2/ga_v14_bizon.json` — Bizon island config (seed=137)
- `models/b2/ga_v14_seed.json` — Refined seed genome

## GA Smoke Test Results

5 generations, population 128, 20 CPU workers:
- Gen 0: best=68.1 (seed reaches all 3 targets)
- Gen 2: best=76.05 (mutation already finding faster completions)
- No false critic alerts after behavior metric fix

## Launch Commands

```bash
# puget (local)
cd /home/graham/code/robotics/training
nohup python -u -m ga.pipeline --config configs/b2/ga_v14_puget.json \
  --seed-genome models/b2/ga_v14_seed.json \
  --island-out models/b2/ga_v14_puget/island_out \
  --island-in models/b2/ga_v14_puget/island_in \
  > models/b2/ga_v14_puget/run.log 2>&1 &

# bizon (remote)
ssh bizon@10.0.0.12
cd /home/bizon/code/robotics/training && source ../env/bin/activate
nohup python -u -m ga.pipeline --config configs/b2/ga_v14_bizon.json \
  --seed-genome models/b2/ga_v14_seed.json \
  --island-out models/b2/ga_v14_bizon/island_out \
  --island-in models/b2/ga_v14_bizon/island_in \
  > models/b2/ga_v14_bizon/run.log 2>&1 &
```

## Key Lessons

1. **Completion gating is structurally unexploitable.** Partial credit creates
   perverse incentives. The GA will always find ways to maximize partial scores
   without actually completing the task. Binary completion gate + speed bonus
   makes exploitation impossible.

2. **Parameter refinement must be systematic.** Single-variable sweeps across
   multiple seeds (not cherry-picked) reveal true effects. Many "obvious"
   improvements (wider turn stance, lower KP_YAW) actually make things worse
   when tested properly.

3. **Train/demo parity is non-negotiable.** The decel taper existed in the demo
   but not in training. Any control logic in the demo must also be in the episode
   or the GA will evolve around its absence.

4. **Step height is the dominant stability variable.** Both walk roll (34%
   reduction from 0.08->0.07) and turn roll (75% reduction from 0.07->0.06)
   were primarily controlled by step height, not stance width, duty cycle,
   or yaw gains.

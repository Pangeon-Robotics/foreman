# Route-Following RL: Learned Locomotion from Route Lookahead

## Context

The B2 robot's speed is bottlenecked by 50+ hand-tuned parameters across 6 layers (Navigator PD gains, EMA alphas, approach distance, L5 velocity mapper, turn coupling, transition ramps, momentum controller, L4 phase/trajectories). The stride_speed ceiling is 3.0Hz x 0.35m = 1.05 m/s — just 52% of V_REF=2.0. Sweep results show the gait is stable in open field (0 falls) but slow (v_avg 0.17-0.41 m/s), and hand-tuning these parameters individually yields diminishing returns because they interact nonlinearly.

**Paradigm shift**: Replace the Navigator with a single learned policy that takes a route lookahead + body state and outputs MotionCommands to L5 + anticipatory body pose corrections. The policy sees upcoming turns and straightaways, enabling preemptive weight shifting (lean into turns before they arrive) and speed management (accelerate on straights, decelerate for curves). This eliminates the EMA lag, approach taper, and PD heading oscillation — the policy learns these behaviors end-to-end.

The route is already computed before the robot takes its first step (A* -> path_smoother produces 0.15m-spaced waypoints). The policy just needs to follow it.

## Layer Compliance

**L5 is stateful and sovereign.** The policy outputs MotionCommands to L5, not GaitParams to L4. L5 retains ownership of all stateful/temporal behavior: phase clock, velocity mapping (vx -> freq/step_length), turn coupling, gait selection, transition ramps.

**L4 is stateless.** L4 accepts raw numerical parameters (phase offsets, step dimensions, body pose) and produces foot positions. L4 has NO gait vocabulary — no gait names, no gait enums, no gait-to-phase-offset mappings. All gait knowledge lives in L5.

**L3 and below are stateless.** No gait vocabulary anywhere below L5.

## Prerequisite: L4 Gait Vocabulary Scrub (Step 0)

### Problem

L4 currently contains extensive gait vocabulary that violates its stateless design:

| File | Violation |
|------|-----------|
| `layer_4/gait.py` | **Entire module** defines GaitType enum, GAIT_PHASE_OFFSETS dict, GAIT_DEFAULT_DUTY_CYCLE dict, `get_phase_offsets()`, `get_default_duty_cycle()` |
| `layer_4/generator.py` | `gait_type: str = 'trot'` in GaitParams, `if params.gait_type == 'stand':` branching, imports from gait.py |
| `training/ga/control_numpy.py:51` | `TROT_PHASE_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])` hardcoded |
| `training/ga/control_jax.py:50,257` | Two different hardcoded trot phase arrays |
| `training/hnn/play_collect.py:323` | Inline `trot_offsets = [0.0, 0.5, 0.5, 0.0]` |
| `training/hnn/direct_collector.py:83-88` | Multi-gait offset dictionary |

### Fix

**L4 changes** (delegate to L4 subagent):
1. Delete `layer_4/gait.py` entirely
2. Refactor `GaitParams` to accept `phase_offsets: tuple[float, float, float, float]` instead of `gait_type: str`
3. Remove all gait name references from `generator.py`
4. L4 becomes purely numerical: receives (phase_offsets, duty_cycle, step_length, step_height, ...) and produces foot positions

**L5 changes** (delegate to L5 subagent):
1. Move gait vocabulary into L5 (gait name -> phase offset mapping)
2. L5's `GaitSelector` resolves gait names to phase offsets before passing to L4
3. Add new gaits to L5's vocabulary (see Movement Language Enrichment below)

**Training changes** (delegate to training subagent):
1. `control_numpy.py`: Accept `phase_offsets` as parameter instead of hardcoding trot
2. `control_jax.py`: Same
3. `direct_collector.py`, `play_collect.py`: Import gait definitions from canonical L5 source or pass phase offsets explicitly

## Movement Language Enrichment

Reference: `docs/quadruped-movement-taxonomy.md` (52 movements from nature)

### Current L5 Coverage: 13/52 YES, 8 PARTIAL, 27 NO, 4 N/A

Three structural bottlenecks limit expressiveness:

### Bottleneck 1: Limited Gait Vocabulary

L5 currently supports 4 gaits: trot, walk, bound, stand. After the L4 scrub, L5 can define arbitrary phase offsets, enabling:

| Gait | Phase Offsets (FR, FL, RR, RL) | Duty Cycle | Taxonomy # |
|------|-------------------------------|------------|------------|
| Trot | [0.0, 0.5, 0.5, 0.0] | 0.50-0.70 | #2 |
| Walk | [0.0, 0.5, 0.25, 0.75] | 0.75 | #1 |
| Bound | [0.0, 0.0, 0.5, 0.5] | 0.50 | #8 |
| Stand | [0.0, 0.0, 0.0, 0.0] | 1.00 | #32 |
| **Pace** | **[0.0, 0.5, 0.0, 0.5]** | **0.50** | **#3** |
| **Pronk** | **[0.0, 0.0, 0.0, 0.0]** | **0.50** | **#10** |
| **Canter** | **[0.0, 0.4, 0.5, 0.9]** | **0.65** | **#5** |
| **Transverse gallop** | **[0.0, 0.1, 0.5, 0.6]** | **0.40** | **#6** |
| **Rotary gallop** | **[0.0, 0.1, 0.6, 0.5]** | **0.40** | **#7** |
| **Half-bound** | **[0.0, 0.1, 0.5, 0.5]** | **0.45** | **#9** |
| **Amble** | **[0.0, 0.5, 0.15, 0.65]** | **0.40** | **#4** |

**Implementation**: Add these to L5's gait vocabulary. The RL policy can then select between gaits via a discrete action or L5 can auto-select based on speed (walk at low speed, trot at medium, gallop at high — matching biological Froude number transitions).

### Bottleneck 2: No Per-Leg Parameterization

Currently all 4 legs share identical step_height, step_length, duty_cycle. This blocks 10+ movements.

**Per-leg step_height (4 floats)** — highest leverage change:
- Enables **rearing** (#35): front step_height=0, rear normal
- Enables **play bow** (#51): front crouch via low step_height, rear normal
- Enables **prancing** (#52): exaggerated front step_height
- Enables **stair climbing** (#22): front legs step higher
- Enables **bucking** (#42): rear explosive extension (high rear step_height)
- Enables **crawling** (#47): uniformly low step_height

**Per-leg step_length (4 floats)** — already partially exists via wz differential stride, but currently coupled to turning. Independent per-leg step_length enables:
- **Pivoting** (#14): inside legs zero stride, outside legs full stride
- **Asymmetric terrain** (#25): shorter stride on uncertain legs

**Implementation**: Expand GaitParams to include `step_heights: tuple[float, float, float, float]` and `step_lengths: tuple[float, float, float, float]`. L4's `compute_foot_positions` already iterates per-leg — just index into arrays instead of using scalars. L5 populates per-leg values from its locomotion logic; default is uniform (backward-compatible).

### Bottleneck 3: No Non-Locomotion Behavior Primitives

L5 has 4 behaviors: walk, stand, sit, waypoint. Missing behavior slots:

| Behavior | Taxonomy # | Description | Priority |
|----------|------------|-------------|----------|
| **getup** | #29 | Fall recovery: roll to sternum, tuck legs, extend | HIGH — essential for real-world |
| **recover** | #27, #28 | Stumble/push reactive stepping | HIGH — essential for real-world |
| **jump** | #23 | Coordinated crouch -> launch -> flight -> land | MEDIUM — cool but not critical |
| **lie_down** | #38 | Controlled descent to prone | MEDIUM — useful for resting |
| **rear** | #35 | Front legs off ground | LOW — specialized |
| **shake** | #40 | Rapid full-body oscillation | LOW — specialized |
| **paw** | #39 | Single foreleg repetitive motion | LOW — specialized |

**Implementation**: Add behavior strings to L5's MotionCommand dispatcher. Initial implementations can be simple state machines (getup: fixed joint trajectory sequence). RL can later learn these behaviors end-to-end using the behavior slot as a skill selector.

### Bottleneck 4: Narrow Body Pose Ranges

Current limits: roll/pitch ±0.26 rad, x/y offset ±0.10m. Too narrow for:
- Aggressive lean into turns at speed
- Slope compensation (#20, #21)
- Weight shifting (#33) for stability
- Anticipatory posture for terrain

**Implementation**: Widen L5's MotionCommand pose ranges to ±0.40 rad (roll/pitch) and ±0.20m (offsets). L3 hardware clamps enforce actual joint limits independently — L5 should express the full desired range and let L3 enforce what's physically possible.

## Architecture

```
[A* Route] -> [Route Encoder (body-frame waypoints)] -> [Policy Network] -> [MotionCommand(vx, wz, gait, body_pose)]
                                                              |                         |
                                                       [Body State + IMU]              L5 (enriched)
                                                                                        |
                                                                               [phase_offsets, per-leg params]
                                                                                        |
                                                                                  L4 (stateless, numerical) -> L3 -> L2 -> L1
```

### What the Policy Replaces vs What Stays

**Replaced** (all Navigator-level):
- Navigator PD heading controller (heading error -> wz)
- Navigator EMA speed smoothing
- Navigator approach distance taper
- Navigator cos(heading_err) speed scaling
- L5 momentum controller posture corrections (reactive -> now anticipatory via route lookahead)

**Stays unchanged** (all L5-internal):
- L5 velocity mapper (vx -> freq, step_length)
- L5 turn coupling (wz -> heading_mod, stride clamping)
- L5 phase clock & phase accumulation
- L5 gait selection (walk/trot/gallop/TIP) — enriched vocabulary
- L5 transition state machine
- L4 (scrubbed: stateless numerical only), L3, L2, L1

### Observation Space (74D)

| Slice | Dims | Content | Source |
|-------|------|---------|--------|
| 0-29 | 30 | Route lookahead: 10 waypoints x (dx, dy, curvature) in body frame | Route encoder |
| 30-35 | 6 | Body state: vx, vy, wz, roll, pitch, yaw | IMU/estimator |
| 36-47 | 12 | Joint positions (12 DOF) | Motor encoders |
| 48-59 | 12 | Joint velocities (12 DOF) | Motor encoders |
| 60-63 | 4 | Foot contacts (FR, FL, RR, RL) | Contact sensors |
| 64-69 | 6 | IMU raw (gyro_xyz, accel_xyz) | IMU |
| 70-73 | 4 | Leg phases (FR, FL, RR, RL in [0,1]) | Phase clock |

**Route encoding**: Transform next 10 waypoints from world-frame to body-frame using current (x, y, yaw). Waypoints spaced 0.15m apart = 1.5m lookahead. Curvature estimated from 3-point finite differences. If fewer than 10 waypoints remain, pad with the final waypoint (zero curvature).

**Why 10 waypoints at 0.15m?** At 1.0 m/s, 1.5m = 1.5s of lookahead — enough to see an upcoming turn ~1 second before it arrives and begin weight transfer. The 0.15m spacing matches path_smoother output directly.

### Action Space (6D continuous + 1D discrete)

**Continuous (6D)**:

| Dim | Field | Range | Target |
|-----|-------|-------|--------|
| 0 | vx | [0.0, 2.0] | MotionCommand.vx -> L5 velocity mapper |
| 1 | wz | [-2.0, 2.0] | MotionCommand.wz -> L5 turn coupling |
| 2 | body_roll | [-0.20, 0.20] | BodyPoseCommand -> L4 -> L3 |
| 3 | body_pitch | [-0.20, 0.20] | BodyPoseCommand -> L4 -> L3 |
| 4 | body_x_offset | [-0.10, 0.10] | BodyPoseCommand -> L4 -> L3 |
| 5 | body_y_offset | [-0.10, 0.10] | BodyPoseCommand -> L4 -> L3 |

**Discrete (1D)** — gait selection (optional, can start without):

| Value | Gait | When to use |
|-------|------|-------------|
| 0 | trot | Default, medium speed |
| 1 | walk | Low speed, high stability |
| 2 | pace | Straightaways (some animals pace for efficiency) |
| 3 | bound | Acceleration bursts |

L5 resolves gait index to phase offsets. The RL policy learns which gait is optimal for each route segment. Initially, fix gait=trot and train the 6D continuous policy; add gait selection as a Phase 2 enhancement.

Actions map to MotionCommand fields. The policy outputs at 20Hz (every 5 physics steps at 100Hz control rate). L5 interpolates between commands.

### Reward Design

Multiplicative gating (lesson from GA exploit postmortem — additive penalties are exploitable):

```python
reward = progress_reward * tracking_reward * alive_bonus

progress_reward = delta_along_route / dt  # m/s along route centerline (not Euclidean)
tracking_reward = exp(-5.0 * cross_track_error**2)  # Gaussian penalty, 1.0 on centerline
alive_bonus = 1.0 if not fallen else 0.0  # multiplicative gate: fall -> zero reward
```

- **progress_reward**: Distance advanced along the route polyline this timestep. Measures forward progress, not just speed — going fast in the wrong direction earns nothing.
- **tracking_reward**: Exponential decay from route centerline. At 0.3m off-track -> 0.64, at 0.5m -> 0.29. Keeps robot on the planned path.
- **alive_bonus**: Binary gate. Fall = zero reward for entire episode remainder. No additive fall penalty to trade against speed.
- **No explicit speed reward**: Speed emerges from maximizing progress_reward. The policy learns that going faster = more progress per timestep = more reward.
- **No explicit turn reward**: Turns are implicitly rewarded because progress stalls if the robot doesn't follow the route through turns.

**Episode termination**: Fall (body height < 0.30m for 20 ticks) or route completed or 30s timeout.

### Network Architecture

```
Route waypoints (30D) -> MLP(64, 64) -> route_embed (32D)
Body state (44D) -> MLP(64, 64) -> body_embed (32D)
[route_embed; body_embed] (64D) -> GRU(128) -> MLP(64) -> 6D actions (tanh-scaled)
                                              -> MLP(64) -> 1D value
```

GRU provides temporal context (gait phase awareness, velocity estimation from proprioception history). Single GRU layer, 128 hidden units.

## Speed Ceiling Discussion

The stride_speed ceiling (3.0Hz x 0.35m = 1.05 m/s) still applies since L5's velocity mapper is in the loop. The policy addresses speed through:

1. **Anticipatory speed management**: Sees turns 1.5s ahead -> smoother deceleration -> less time wasted braking
2. **Preemptive body pose**: Lean into turns before they arrive -> fewer instability events -> higher sustained speed
3. **Optimal vx/wz trajectories**: Learned smooth commands vs PD oscillation -> less energy wasted on corrections
4. **Gait selection** (Phase 2): Policy learns to switch to pace or bound on straightaways for higher speed

Breaking the 1.05 m/s ceiling requires tuning L5's velocity mapper (raise freq/step_length limits) — a separate task from this RL work. Adding gallop/bound gaits via enriched L5 vocabulary may naturally raise the ceiling.

## Implementation Steps

### Step 0: L4 Gait Vocabulary Scrub — CASCADE

**Scope**: Cross-layer change affecting L4, L5, and training. Use cascade pattern.

**L4 subagent**:
1. Delete `layer_4/gait.py`
2. Replace `gait_type: str` with `phase_offsets: tuple[float, float, float, float]` in GaitParams
3. Remove all gait name references from `generator.py`
4. Update L4 tests

**L5 subagent**:
1. Add `GAIT_VOCABULARY` dict: gait name -> (phase_offsets, default_duty_cycle)
2. Populate with all 11 gaits (trot, walk, bound, stand, pace, pronk, canter, transverse_gallop, rotary_gallop, half_bound, amble)
3. Update GaitSelector to resolve names to phase offsets
4. Pass `phase_offsets` in GaitParams to L4 instead of `gait_type`
5. Add per-leg step_height and step_length fields to GaitParams

**Training subagent**:
1. Refactor `control_numpy.py` to accept `phase_offsets` parameter
2. Refactor `control_jax.py` to accept `phase_offsets` parameter
3. Update `direct_collector.py` and `play_collect.py` to pass phase offsets explicitly

### Step 1: Random Route Generator — `training/rl/route_gen.py`

**Ornstein-Uhlenbeck curvature process** with difficulty parameter:

```python
def generate_route(length=15.0, difficulty=0.5, spacing=0.15, seed=None) -> np.ndarray:
    """Generate random route as (N, 3) array of (x, y, curvature).

    difficulty: 0.0 = straight line, 1.0 = aggressive S-curves
    Curvature kappa follows OU process: dkappa = -theta*kappa*dt + sigma*dW
    theta = 2.0 (mean reversion rate)
    sigma = difficulty * 3.0 (volatility scales with difficulty)
    |kappa| clamped to 2.0 (min radius = 0.5m)
    """
```

Generates world-frame (x, y, curvature) waypoints. Route starts at origin heading +x. The OU process produces naturalistic curves — long straights punctuated by turns of varying severity.

**Reuse**: `path_smoother.py` spacing=0.15m convention, `normalize_angle` from `utils.py`.

### Step 2: Gymnasium Environment — `training/rl/envs/route_follow_env.py`

**Physics**: Pure MuJoCo at 500Hz (same as `direct_collector.py`). No DDS, no firmware subprocess. Load B2 model directly.

**L5 shim in training**: The training env includes L5's velocity mapping and turn coupling in the loop so the policy learns to work WITH L5's constraints, not around them:

1. Policy outputs (vx, wz, body_pose) at 20Hz
2. **L5 shim** applies: velocity mapper (vx -> freq, step_length), turn coupling (wz -> heading_mod), stride clamping, gait name -> phase_offsets resolution
3. `control_numpy` handles phase computation from phase_offsets + L4->L3->L2 (refactored in Step 0)
4. MuJoCo steps physics

The L5 shim is a pure-Python function (~50 lines) extracted from L5's velocity mapper logic. No DDS, no imports from layer_5/.

**Control pipeline**: Policy outputs 6D action at 20Hz -> L5 shim produces (phase_offsets, step params, body pose) -> `control_numpy.compute_phases(phase_offsets=...)` -> `control_numpy.compute_foot_positions()` -> `control_numpy.batch_ik()` -> `control_numpy.compute_pd_torque()` -> `mj_step()`. Reuses the existing vectorized L4+L3+L2 pipeline from `training/ga/control_numpy.py`.

**Key reuse from existing code**:
- `training/ga/control_numpy.py`: `compute_phases()`, `compute_foot_positions()`, `batch_ik()`, `compute_pd_torque()` — full vectorized L4->L3->L2 pipeline (refactored to accept phase_offsets)
- `training/hnn/direct_collector.py`: Body state extraction patterns (`_quat_to_rpy`, contact detection, IMU reading)
- `foreman/demos/target_game/utils.py`: `normalize_angle`, `quat_to_yaw`

**Step function** (called at 20Hz):
1. Receive 6D action from policy
2. L5 shim: velocity mapper + turn coupling + gait resolution -> (phase_offsets, step params, body pose)
3. Run 5 control steps at 100Hz (each = 5 physics steps at 500Hz = 25 `mj_step` calls total)
4. Extract 74D observation
5. Compute reward (progress along route x tracking x alive)
6. Return obs, reward, terminated, truncated, info

**Reset**: Sample new route from `route_gen.generate_route(difficulty=curriculum_difficulty)`. Place robot at route start. Reset MuJoCo state.

### Step 3: PPO Training Script — `training/rl/train_route_ppo.py`

**Framework**: JAX/Flax + custom PPO (consistent with existing training infra in `training/`). NOT using external RL libraries (SB3, RLlib) — the project already has JAX/Flax infrastructure.

**Curriculum** (difficulty ramps over training):

| Phase | Steps | difficulty | Route character |
|-------|-------|-----------|-----------------|
| 1 | 0-5M | 0.0-0.2 | Nearly straight, gentle curves |
| 2 | 5-15M | 0.2-0.5 | Moderate curves, some S-turns |
| 3 | 15-30M | 0.5-0.8 | Tight turns, rapid direction changes |
| 4 | 30-50M | 0.8-1.0 | Full distribution, aggressive S-curves |

**Hyperparameters** (starting point, will tune):
- Learning rate: 3e-4 with linear decay
- Clip ratio: 0.2
- GAE lambda: 0.95, gamma: 0.99
- Minibatch size: 256, rollout length: 2048 steps (at 20Hz = ~100s per rollout)
- N parallel envs: 32 (single-threaded MuJoCo, no DDS contention)
- Entropy coefficient: 0.01 (start), decay to 0.001
- Value loss coefficient: 0.5
- Max grad norm: 0.5

**Checkpointing**: Save every 1M steps. Export best checkpoint to `.npz` for deployment (same pattern as `training/hnn/export_npz.py`).

### Step 4: Deployment Integration — `foreman/demos/target_game/learned_navigator.py`

**Replaces** `navigator_helper.py` when `--learned-nav` flag is passed.

```python
class LearnedNavigator:
    def __init__(self, game, model_path: str):
        self.weights = np.load(model_path)  # numpy-only at runtime (no JAX)
        self.gru_state = np.zeros(128)

    def tick(self, dt: float):
        # 1. Get route waypoints from game's A* path (already computed)
        # 2. Transform 10 nearest waypoints to body frame
        # 3. Build 74D observation
        # 4. Forward pass through policy network (numpy matmuls)
        # 5. Extract (vx, wz, body_pose) from 6D action
        # 6. Send to L5 via game.sim.send_motion_command(vx, wz)
        # 7. Send body pose via L5's posture interface
```

Calls `send_motion_command()` (L5 interface), NOT `send_gait_params()` (L4). L5 handles velocity mapping, turn coupling, phase tracking, gait selection.

**Fallback**: If `--learned-nav` not specified, existing navigator_helper.py is used (no regression).

### Step 5: Movement Enrichment (post-training validation)

After the RL policy demonstrates route-following competence:

**5a. Per-leg step_height in action space** — Add 4D to action space (10D total continuous). Policy learns asymmetric leg lift for turns (inside legs higher), terrain (front legs higher for steps), and stability (rear legs higher for braking). Requires per-leg step_height support in L4 (Step 0) and L5 shim.

**5b. Gait selection** — Add discrete action for gait choice. Policy learns to switch gaits based on route curvature: trot for turns, pace/bound for straightaways. Requires enriched L5 vocabulary (Step 0).

**5c. Recovery behaviors** — Train separate getup/recover policies. Triggered by fall detection. These are independent RL problems, not part of the route-following policy. L5 behavior='getup' dispatches to recovery policy.

**5d. Jump primitive** — Multi-phase behavior: crouch (lower body_height over 0.3s) -> launch (max step_height, duty_cycle=0 for brief aerial) -> land (high duty_cycle, wide stance). Complex state machine in L5; RL can refine the timing/forces.

## Files to Create/Modify

### Step 0 (Cascade)

| File | Action | Purpose |
|------|--------|---------|
| `layer_4/gait.py` | DELETE | Remove gait vocabulary from L4 |
| `layer_4/generator.py` | MODIFY | Accept phase_offsets instead of gait_type |
| `layer_4/tests/test_gait.py` | DELETE or MODIFY | Remove/update gait tests |
| `layer_4/tests/test_generator.py` | MODIFY | Pass phase_offsets in tests |
| `layer_5/gait_vocabulary.py` | CREATE | Canonical gait name -> phase offset mapping |
| `layer_5/gait_selector.py` | MODIFY | Resolve gait names to phase offsets |
| `layer_5/config/defaults.py` | MODIFY | Add per-leg step_height/step_length to GaitParams |
| `layer_5/locomotion.py` | MODIFY | Populate per-leg params, pass phase_offsets to L4 |
| `training/ga/control_numpy.py` | MODIFY | Accept phase_offsets parameter |
| `training/ga/control_jax.py` | MODIFY | Accept phase_offsets parameter |
| `training/hnn/direct_collector.py` | MODIFY | Pass phase offsets explicitly |
| `training/hnn/play_collect.py` | MODIFY | Pass phase offsets explicitly |

### Steps 1-4 (RL Pipeline)

| File | Action | Purpose |
|------|--------|---------|
| `training/rl/route_gen.py` | CREATE | Random route generator (OU curvature process) |
| `training/rl/envs/route_follow_env.py` | CREATE | Gymnasium environment with MuJoCo physics + L5 shim |
| `training/rl/train_route_ppo.py` | CREATE | PPO training script with curriculum |
| `foreman/demos/target_game/learned_navigator.py` | CREATE | Deployment-time navigator using trained policy |
| `foreman/demos/target_game/__main__.py` | MODIFY | Add `--learned-nav` flag, wire LearnedNavigator |

**Files NOT modified in RL steps**: navigator_helper.py, body_model.py, velocity_mapper.py, transition.py, behaviors.py — all remain as fallback. No existing functionality is removed.

## Execution Strategy

1. **Step 0: L4 gait scrub** (cascade) — prerequisite for everything else. Three subagents: L4, L5, training.
2. **Step 1: Route generator** — standalone, no dependencies, easy to test visually
3. **Step 2: Environment** — depends on route_gen + refactored control_numpy
4. **Step 3: Training script** — depends on environment
5. **Step 4: Deployment** — depends on trained weights
6. **Step 5: Movement enrichment** — iterative, after base policy works

Steps 0 must complete before Steps 1-4. Steps 1-3 are sequential. Step 4 requires trained weights. Step 5 is iterative enhancement.

**Subagent delegation**: Step 0 uses cascade pattern (L4 subagent, L5 subagent, training subagent). Steps 1-3 are in the `training/` repo -> delegate to a training subagent. Step 4 modifies foreman -> do directly.

## Verification

1. **Step 0 verification**: All layer tests pass. `layer_4/gait.py` deleted. No gait name strings in L4. `grep -r "gait_type" layer_4/` returns nothing.
2. **Route generator**: `python -c "from rl.route_gen import generate_route; r = generate_route(difficulty=0.5); print(r.shape, r[:3])"` — should produce smooth curves
3. **Environment smoke test**: `python -c "from rl.envs.route_follow_env import RouteFollowEnv; env = RouteFollowEnv(); obs, _ = env.reset(); print(obs.shape)"` — should output (74,)
4. **Random policy rollout**: Environment runs for 1000 steps with random actions without crashing, robot walks (doesn't fall immediately)
5. **Training convergence**: After 5M steps on difficulty=0.1 (straight lines), mean episode reward should increase and robot should walk forward
6. **Deployment test**: `python foreman/run_demo.py --learned-nav --targets 3` — robot follows A* route to targets
7. **A/B comparison**: `python foreman/fast_test.py` with learned nav vs default nav — compare ATO, falls, v_avg

## Open Questions

1. **20Hz vs 50Hz policy rate**: 20Hz matches the momentum controller rate. May need 50Hz for responsive turning. Start with 20Hz, increase if tracking_reward is poor on tight curves.
2. **Joint positions/velocities in observation**: 24D of joint state may be redundant if leg_phases (4D) captures enough. Could reduce to 50D observation. Start with full 74D, ablate later.
3. **MJX vectorization**: For >10x training speedup, port environment to MJX (GPU-accelerated MuJoCo). The existing codebase has MJX investigation notes but no working integration. Phase 2 optimization after CPU training proves the concept.
4. **Sim-to-real gap**: Policy trained on MuJoCo kp=500/kd=25 (sim gains). Real robot uses kp=2500. May need domain randomization over gains. Address after sim validation.
5. **L5 velocity mapper ceiling**: stride_speed = 3.0Hz x 0.35m = 1.05 m/s caps max speed. Raising this requires L5 config changes (separate task). The RL policy still adds value through anticipation and smoothness within this ceiling. Adding gallop/bound may naturally raise it.
6. **Gait selection timing**: How often should the policy switch gaits? Every 20Hz step would be unstable. Probably limit gait switches to every 0.5-1.0s with L5 transition ramps between.
7. **Per-leg step_height stability**: Asymmetric step heights change the robot's dynamic balance. May need wider body_pose ranges to compensate. Train incrementally — uniform first, then per-leg.
8. **Recovery policy architecture**: Should getup/recover share the same network as route-following (multi-task), or be separate specialized policies? Separate is simpler and safer.

## Movement Coverage After Full Implementation

After Steps 0-5, L5 coverage increases from 13/52 to approximately 30/52:

**Newly expressible** (17 movements): pace (#3), amble (#4), canter (#5), transverse gallop (#6), rotary gallop (#7), half-bound (#9), pronk (#10), trot-to-gallop transition (#12), pivot turn (#14, upgraded from partial), ascending slope (#20, upgraded), descending slope (#21, upgraded), stair climbing (#22), crawling (#47, upgraded), stalking (#48, upgraded), prancing (#52, upgraded), play bow (#51), rearing (#35).

**Still blocked** (hardware-limited or very specialized): swimming (#44), climbing (#45), burrowing (#46), head bobbing (#49), tail compensation (#50) — all N/A for B2 hardware. Scrambling (#25), rock hopping (#26) — require terrain perception + per-foot adaptive placement (Layer 6 integration). Shaking (#40), scratching (#41), raking (#43) — specialized manipulation behaviors, low priority.

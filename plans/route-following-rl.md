# Route-Following RL: Learned Locomotion from Route Lookahead

## Context

The B2 robot's speed is bottlenecked by 50+ hand-tuned parameters across 6 layers (Navigator PD gains, EMA alphas, approach distance, L5 velocity mapper, turn coupling, transition ramps, momentum controller, L4 phase/trajectories). The stride_speed ceiling is 3.0Hz x 0.35m = 1.05 m/s — just 52% of V_REF=2.0. Sweep results show the gait is stable in open field (0 falls) but slow (v_avg 0.17-0.41 m/s), and hand-tuning these parameters individually yields diminishing returns because they interact nonlinearly.

**Paradigm shift**: Replace the Navigator with a single learned policy that takes a route lookahead + body state and outputs MotionCommands to L5 + anticipatory body pose corrections. The policy sees upcoming turns and straightaways, enabling preemptive weight shifting (lean into turns before they arrive) and speed management (accelerate on straights, decelerate for curves). This eliminates the EMA lag, approach taper, and PD heading oscillation — the policy learns these behaviors end-to-end.

The route is already computed before the robot takes its first step (A* -> path_smoother produces 0.15m-spaced waypoints). The policy just needs to follow it.

## Layer Compliance

**L5 is stateful and sovereign.** The policy outputs MotionCommands to L5, not GaitParams to L4. L5 retains ownership of all stateful/temporal behavior: phase clock, velocity mapping (vx -> freq/step_length), turn coupling, gait selection, transition ramps.

**L4 is stateless.** The policy never bypasses L5 to send GaitParams directly to L4.

## Architecture

```
[A* Route] -> [Route Encoder (body-frame waypoints)] -> [Policy Network] -> [MotionCommand(vx, wz) + BodyPose]
                                                              |                         |
                                                       [Body State + IMU]              L5 (unchanged)
                                                                                        |
                                                                                  [GaitParams] -> L4 -> L3 -> L2 -> L1
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
- L5 gait selection (walk/trot/TIP)
- L5 transition state machine
- L4, L3, L2, L1 — all unchanged

### Observation Space (73D)

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

### Action Space (6D continuous)

| Dim | Field | Range | Target |
|-----|-------|-------|--------|
| 0 | vx | [0.0, 2.0] | MotionCommand.vx -> L5 velocity mapper |
| 1 | wz | [-2.0, 2.0] | MotionCommand.wz -> L5 turn coupling |
| 2 | body_roll | [-0.15, 0.15] | BodyPoseCommand -> L4 -> L3 |
| 3 | body_pitch | [-0.15, 0.15] | BodyPoseCommand -> L4 -> L3 |
| 4 | body_x_offset | [-0.08, 0.08] | BodyPoseCommand -> L4 -> L3 |
| 5 | body_y_offset | [-0.08, 0.08] | BodyPoseCommand -> L4 -> L3 |

Actions map to MotionCommand + BodyPoseCommand fields. The policy outputs at 20Hz (every 5 physics steps at 100Hz control rate). L5 interpolates between commands.

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
Body state (43D) -> MLP(64, 64) -> body_embed (32D)
[route_embed; body_embed] (64D) -> GRU(128) -> MLP(64) -> 6D actions (tanh-scaled)
                                              -> MLP(64) -> 1D value
```

GRU provides temporal context (gait phase awareness, velocity estimation from proprioception history). Single GRU layer, 128 hidden units.

## Speed Ceiling Discussion

The stride_speed ceiling (3.0Hz x 0.35m = 1.05 m/s) still applies since L5's velocity mapper is in the loop. The policy addresses speed through:

1. **Anticipatory speed management**: Sees turns 1.5s ahead -> smoother deceleration -> less time wasted braking
2. **Preemptive body pose**: Lean into turns before they arrive -> fewer instability events -> higher sustained speed
3. **Optimal vx/wz trajectories**: Learned smooth commands vs PD oscillation -> less energy wasted on corrections

Breaking the 1.05 m/s ceiling requires tuning L5's velocity mapper (raise freq/step_length limits) — a separate task from this RL work.

## Implementation Steps

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
2. **L5 shim** applies: velocity mapper (vx -> freq, step_length), turn coupling (wz -> heading_mod), stride clamping
3. `control_numpy` handles phase tracking + L4->L3->L2 (already exists)
4. MuJoCo steps physics

The L5 shim is a pure-Python function (~30 lines) extracted from L5's velocity mapper logic. No DDS, no imports from layer_5/.

**Control pipeline**: Policy outputs 6D action at 20Hz -> L5 shim produces GaitParams -> `control_numpy.compute_phases()` -> `control_numpy.compute_foot_positions()` -> `control_numpy.batch_ik()` -> `control_numpy.compute_pd_torque()` -> `mj_step()`. Reuses the existing vectorized L4+L3+L2 pipeline from `training/ga/control_numpy.py`.

**Key reuse from existing code**:
- `training/ga/control_numpy.py`: `compute_phases()`, `compute_foot_positions()`, `batch_ik()`, `compute_pd_torque()` — full vectorized L4->L3->L2 pipeline
- `training/hnn/direct_collector.py`: Body state extraction patterns (`_quat_to_rpy`, contact detection, IMU reading)
- `foreman/demos/target_game/utils.py`: `normalize_angle`, `quat_to_yaw`

**Step function** (called at 20Hz):
1. Receive 6D action from policy
2. L5 shim: velocity mapper + turn coupling -> GaitParams
3. Apply body pose corrections to GaitParams
4. Run 5 control steps at 100Hz (each = 5 physics steps at 500Hz = 25 `mj_step` calls total)
5. Extract 73D observation
6. Compute reward (progress along route x tracking x alive)
7. Return obs, reward, terminated, truncated, info

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
        # 3. Build 73D observation
        # 4. Forward pass through policy network (numpy matmuls)
        # 5. Extract (vx, wz, body_pose) from 6D action
        # 6. Send to L5 via game.sim.send_motion_command(vx, wz)
        # 7. Send body pose via L5's posture interface
```

Calls `send_motion_command()` (L5 interface), NOT `send_gait_params()` (L4). L5 handles velocity mapping, turn coupling, phase tracking, gait selection.

**Fallback**: If `--learned-nav` not specified, existing navigator_helper.py is used (no regression).

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `training/rl/route_gen.py` | CREATE | Random route generator (OU curvature process) |
| `training/rl/envs/route_follow_env.py` | CREATE | Gymnasium environment with MuJoCo physics + L5 shim |
| `training/rl/train_route_ppo.py` | CREATE | PPO training script with curriculum |
| `foreman/demos/target_game/learned_navigator.py` | CREATE | Deployment-time navigator using trained policy |
| `foreman/demos/target_game/__main__.py` | MODIFY | Add `--learned-nav` flag, wire LearnedNavigator |

**Files NOT modified**: navigator_helper.py, locomotion.py, body_model.py, velocity_mapper.py, transition.py, behaviors.py — all remain as fallback. No existing functionality is removed.

## Execution Strategy

1. **Route generator first** (Step 1) — standalone, no dependencies, easy to test visually
2. **Environment** (Step 2) — depends on route_gen + control_numpy (both exist)
3. **Training script** (Step 3) — depends on environment
4. **Deployment** (Step 4) — depends on trained weights

Steps 1-2 can be developed and unit-tested before training begins. Step 3 requires running actual training. Step 4 is integration after a successful training run.

**Subagent delegation**: Steps 1-3 are in the `training/` repo -> delegate to a training subagent. Step 4 modifies foreman -> do directly.

## Verification

1. **Route generator**: `python -c "from rl.route_gen import generate_route; r = generate_route(difficulty=0.5); print(r.shape, r[:3])"` — should produce smooth curves
2. **Environment smoke test**: `python -c "from rl.envs.route_follow_env import RouteFollowEnv; env = RouteFollowEnv(); obs, _ = env.reset(); print(obs.shape)"` — should output (73,) or (74,)
3. **Random policy rollout**: Environment runs for 1000 steps with random actions without crashing, robot walks (doesn't fall immediately)
4. **Training convergence**: After 5M steps on difficulty=0.1 (straight lines), mean episode reward should increase and robot should walk forward
5. **Deployment test**: `python foreman/run_demo.py --learned-nav --targets 3` — robot follows A* route to targets
6. **A/B comparison**: `python foreman/fast_test.py` with learned nav vs default nav — compare ATO, falls, v_avg

## Open Questions

1. **20Hz vs 50Hz policy rate**: 20Hz matches the momentum controller rate. May need 50Hz for responsive turning. Start with 20Hz, increase if tracking_reward is poor on tight curves.
2. **Joint positions/velocities in observation**: 24D of joint state may be redundant if leg_phases (4D) captures enough. Could reduce to 49D observation. Start with full 73D, ablate later.
3. **MJX vectorization**: For >10x training speedup, port environment to MJX (GPU-accelerated MuJoCo). The existing codebase has MJX investigation notes but no working integration. Phase 2 optimization after CPU training proves the concept.
4. **Sim-to-real gap**: Policy trained on MuJoCo kp=500/kd=25 (sim gains). Real robot uses kp=2500. May need domain randomization over gains. Address after sim validation.
5. **L5 velocity mapper ceiling**: stride_speed = 3.0Hz x 0.35m = 1.05 m/s caps max speed. Raising this requires L5 config changes (separate task). The RL policy still adds value through anticipation and smoothness within this ceiling.

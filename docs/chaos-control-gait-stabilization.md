# Chaos Control Theory Applied to Gait Stabilization

Reference: Fradkov & Evans, "Control of Chaos: Methods and Applications in
Engineering," Annual Reviews in Control 29 (2005) 33-56.

## Why Chaos Control Applies to Legged Locomotion

A trotting quadruped is a dynamical system with:
- **Periodic orbits**: The trot gait is a limit cycle (body state repeats
  each stride period τ = 1/freq)
- **Unstable equilibria**: The upright walking pose is unstable — any
  perturbation grows without active correction
- **Sensitivity to initial conditions**: Small changes in foot placement,
  timing, or contact forces cascade into large trajectory deviations
- **Low-dimensional control over high-dimensional dynamics**: We adjust
  ~5 gait parameters to control a 12-DOF robot with 83.5 kg of inertia

The chaos control literature provides principled methods for exactly this
situation: stabilizing unstable periodic orbits using small, targeted
perturbations.

## Key Methods

### OGY Method (Ott, Grebogi, Yorke 1990)

**Core idea**: Chaotic attractors contain infinitely many unstable periodic
orbits (UPOs). Rather than fighting chaos, exploit it — wait until the
system trajectory passes near a desired UPO, then apply a small perturbation
to nudge it onto the orbit.

**Mechanism**: Linearize the Poincaré map near the target orbit. When the
state is within a deadzone Δ of the orbit, apply linear feedback:

    u_k = C · x̃_k    if |x̃_k| ≤ Δ
    u_k = 0           otherwise

where x̃_k is the deviation from the target orbit and C is computed from
the linearized map's stable/unstable manifolds.

**Our application**: The trot gait IS the target periodic orbit. When body
state deviates beyond a threshold (predicted |droll|² + |dpitch|² > Q_threshold),
apply corrective gait adjustments. When the gait is stable (deviation below
threshold), apply zero correction — don't interfere with a working gait.

**Key insight**: The deadzone is critical. Applying control everywhere adds
noise. Applying control only near the orbit is both cheaper and more effective.

### Pyragas Delayed Feedback (1992)

**Core idea**: Stabilize a periodic orbit without knowing its shape. Use
the system's own history as the reference:

    u(t) = K · [x(t) - x(t - τ)]

where τ is the orbit period and K is a gain matrix.

**Why it works**: On the target orbit, x(t) = x(t-τ) by definition, so the
control vanishes. The orbit IS an equilibrium of the controlled system. Off
the orbit, the correction pushes the state back toward periodicity.

**Our application**: Record body state in a ring buffer indexed by gait
period (τ = 1/gait_freq ≈ 0.33s at 3Hz). Each tick, compare current state
to one period ago. Deviations map to gait corrections:
- Roll deviation → adjust stance width (wider = more lateral stability)
- Pitch deviation → adjust body height (lower CoG = more pitch stability)
- Lateral velocity deviation → adjust step symmetry

**Advantages over model-based methods**:
- No system identification required for the orbit shape
- Automatically adapts to the actual periodic orbit (which varies with
  speed, terrain, and wear)
- Zero correction in steady state (no energy wasted)
- Robust to model uncertainty (doesn't depend on accurate dynamics)

### Speed-Gradient Method (Fradkov 1979)

**Core idea**: Define a goal function Q(x) that measures "distance from
desired behavior." Compute how the control inputs affect Q's rate of change.
Move the controls in the direction that decreases Q fastest:

    u = -Ψ[∇_u Q̇(x, u)]

where Ψ is a monotone function (typically identity or sign).

**Lyapunov guarantee**: If Q(x) is a valid Lyapunov function candidate and
the system satisfies a "speed-gradient condition" (roughly: the control can
always decrease Q̇), then Q → 0 and the goal is achieved.

**Our application**:
- Goal function: Q(x) = w_roll·droll² + w_pitch·dpitch² + w_vy·dvy²
  (predicted instability from the ensemble)
- Control variables: body posture corrections (body_roll, body_pitch, body_x_offset, body_y_offset)
- Gradient: computed via finite differences through the learned ensemble
  (perturb each posture param by ε, observe change in Q)
- Update: small additive posture corrections applied through L4 BodyPoseCommand → L3 IK

**Why finite differences work here**: The ensemble is a differentiable
function (MLP with SiLU activations). We could in principle compute
analytical gradients via backprop, but finite differences are simpler,
avoid JAX dependency at runtime, and are accurate enough given the small
correction magnitudes.

### Adaptive Control (Fradkov & Andrievsky)

**Core idea**: When system parameters are unknown, estimate them online
and adapt the controller. Combines system identification with control:

    ẋ = f(x, θ, u)        (system with unknown parameters θ)
    θ̇ = γ · ∇_θ Q̇(x, θ, u)  (parameter adaptation)
    u  = -Ψ[∇_u Q̇(x, θ̂, u)]  (control using estimated params)

**Our application**: The ensemble disagreement serves as an implicit
uncertainty estimate. When members disagree (high variance), the model
is uncertain — we reduce correction strength (lower γ) to avoid
over-correcting based on unreliable predictions. This is a form of
adaptive gain scheduling driven by epistemic uncertainty.

### Neural Network Identification + Control

**Core idea**: Use neural networks for online system identification in
a closed-loop setting. The NN learns the system dynamics, then standard
chaos control algorithms (OGY, Pyragas, speed-gradient) are applied using
the learned model.

**Our application**: This is exactly what we built. The K=5 MLP ensemble
IS the neural network system identification model. It was trained on
real robot trajectories (play data collection through the full L5 pipeline).
The speed-gradient controller uses this learned model to compute control
corrections.

## The Central Insight

From Section 9.1 of the paper:

> "The more unstable (chaotic, turbulent) the open-loop behavior of the
> plant is, the 'simpler' or 'cheaper' it is to achieve exact or
> approximate controllability."

This inverts the intuition that unstable systems are harder to control.
In fact:
- Chaotic systems visit many states → control can exploit natural
  trajectory diversity
- Unstable equilibria have expanding directions → small perturbations
  along these directions have large effects
- The control energy required to stabilize an UPO scales with
  (1/λ_max) where λ_max is the largest Lyapunov exponent

**For our robot**: The gait regimes where the robot is most unstable
(high speed turns, rough terrain, recovering from perturbations) are
precisely the regimes where small gait adjustments will have the largest
stabilizing effect. The momentum controller should be most effective exactly
when it's most needed.

## Design Principles Derived

1. **Small corrections, not large interventions.** OGY's key contribution:
   tiny perturbations can stabilize unstable orbits embedded in chaotic
   attractors. Our correction clamps are deliberately small (±0.03m step
   height, ±0.3Hz frequency, ±0.01m body height).

2. **Deadzone: don't fix what isn't broken.** The OGY deadzone principle:
   apply control only when the state deviates beyond a threshold. Stable
   gait → zero correction. This prevents the controller from adding noise
   to an already-good gait.

3. **Use the system's own periodicity.** Pyragas: the target orbit defines
   itself through the system's history. No need to specify what "correct"
   trotting looks like — just compare each stride to the previous one.

4. **Gradient descent on instability.** Speed-gradient: compute how each
   control input affects the instability rate, move in the decreasing
   direction. The ensemble provides the dynamics model for this computation.

5. **Be cautious under uncertainty.** Adaptive control: reduce correction
   strength when the model is uncertain (high ensemble disagreement).
   Better to under-correct than to apply confident-looking corrections
   based on unreliable predictions.

6. **Corrections are additive, not replacements.** The rule-based pipeline
   (velocity mapper, turn coupling, gait selection) sets the operating
   point. The momentum controller applies small perturbations around it.
   If the momentum controller is disabled, the system reverts to the
   known-working baseline.

## Relationship to Existing System

| Component | Before (Phase 1) | After (Phase 2) |
|-----------|------------------|-----------------|
| Ensemble usage | Offline table precomputation | Real-time prediction from live state |
| Observation | Synthetic (zeros) | Actual body state (IMU, contacts) |
| Output | Scalar turn_factor | 6D momentum prediction → multi-param corrections |
| Control law | Passive attenuation | Speed-gradient + Pyragas active stabilization |
| When active | Always (lookup table) | Deadzone: only when instability exceeds threshold |
| Uncertainty | Ignored | Ensemble disagreement scales correction strength |

## References

- Ott, E., Grebogi, C., & Yorke, J.A. (1990). "Controlling chaos."
  Physical Review Letters, 64(11), 1196-1199.
- Pyragas, K. (1992). "Continuous control of chaos by self-controlling
  feedback." Physics Letters A, 170(6), 421-428.
- Fradkov, A.L. (1979). "Speed-gradient scheme and its application in
  adaptive control problems." Automation and Remote Control, 40, 1333-1342.
- Fradkov, A.L. & Evans, R.J. (2005). "Control of chaos: Methods and
  applications in engineering." Annual Reviews in Control, 29, 33-56.

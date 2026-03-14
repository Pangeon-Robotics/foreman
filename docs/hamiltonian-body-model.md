# Hamiltonian Body Model

The robot's body model is formulated in Hamiltonian phase space. Every possible movement — walking, galloping, jumping, falling, recovering — is a trajectory through a single 60-dimensional space. There are no gait parameters, no phase offsets, no step lengths. Gaits are emergent geometric structures (limit cycles) in phase space.

## Phase Space (q, p)

The state of the robot is a point in 60D phase space: 30 generalized coordinates **q** (positions) and 30 conjugate momenta **p**.

### Generalized Coordinates q (30D)

| Body | Coordinates | Dims |
|------|-------------|------|
| Base (trunk) | x, y, z, roll, pitch, yaw | 6 |
| FR thigh | x, y, z | 3 |
| FR calf | x, y, z | 3 |
| FL thigh | x, y, z | 3 |
| FL calf | x, y, z | 3 |
| RR thigh | x, y, z | 3 |
| RR calf | x, y, z | 3 |
| RL thigh | x, y, z | 3 |
| RL calf | x, y, z | 3 |

All positions are in world frame. Base orientation as Euler angles (roll, pitch, yaw). Leg segment positions are the center of mass of each rigid body.

### Conjugate Momenta p (30D)

| Body | Momenta | Definition |
|------|---------|------------|
| Base | p_x, p_y, p_z, L_roll, L_pitch, L_yaw | **p** = m**v** (linear), **L** = **I**omega (angular) |
| Each leg segment | p_x, p_y, p_z | **p** = m**v** (linear momentum) |

Momenta are the physically meaningful quantities — not velocities. A heavy trunk at low velocity and a light calf at high velocity can have the same momentum. The Hamiltonian is naturally expressed in momenta.

## The Hamiltonian

A single scalar function H(q, p) encodes the total energy of the system:

```
H(q, p) = T(q, p) + V(q)

T = sum_i (1/2) p_i^T M_i^{-1} p_i     (kinetic energy)
V = sum_i m_i g h_i                      (potential energy)
```

where i ranges over all 9 rigid bodies. T includes both translational and rotational kinetic energy for the base.

Hamilton's equations give the dynamics:

```
dq/dt =  dH/dp     (positions evolve according to momenta)
dp/dt = -dH/dq     (momenta evolve according to forces)
```

From one scalar function, the entire dynamics of the robot follow. The HNN (Hamiltonian Neural Network) learns this function from data.

## Movements as Phase Space Geometry

Every movement in the quadruped taxonomy (see `docs/quadruped-movement-taxonomy.md`) maps to a geometric object in phase space:

### Fixed Points
- **Standing**: (q_0, 0). All momenta zero. The robot at rest in its nominal pose. Stable fixed point — small perturbations return to it.

### Periodic Orbits (Limit Cycles)
- **Trot**: A closed loop in 60D space, completed once per gait cycle. Diagonal leg pairs have phase-locked momentum oscillations — when FR calf momentum peaks upward, RL calf momentum peaks upward simultaneously.
- **Walk**: A low-energy periodic orbit. Smaller momentum amplitudes. At any point on the orbit, 3+ leg segments have near-zero vertical momentum (in stance).
- **Gallop**: A different periodic orbit with different topology. Front and rear momentum waves are sequential, not simultaneous. Higher energy than trot — the orbit is farther from the fixed point.
- **Pace**: Periodic orbit where lateral pairs (FL+RL, FR+RR) have synchronized momentum oscillations.
- **Bound**: Periodic orbit where fore/aft pairs (FR+FL, RR+RL) oscillate in antiphase.
- **Pronk**: All 8 leg segments have synchronized momentum oscillations — the simplest periodic orbit topology.
- **Canter**: Asymmetric 3-beat orbit. One leg has a distinct momentum phase offset from the others.
- **Prancing**: A periodic orbit with exaggerated vertical momentum amplitudes in the front legs.

### Transient Trajectories
- **Jump**: Leaves the standing fixed point, passes through a ballistic arc (all p_z large, all contacts lost), returns to a landing configuration.
- **Fall**: A trajectory leaving the basin of attraction of any stable orbit. The base orientation departs from upright; momenta grow uncontrolled.
- **Recovery (getup)**: A trajectory from a far-from-equilibrium state (lying on side, inverted) back toward the standing fixed point or a periodic orbit.
- **Push recovery**: A perturbation kicks the state off its periodic orbit; reactive stepping creates a trajectory that returns to the orbit.
- **Stumble recovery**: Similar — a transient departure and return, with rapid leg momentum changes to catch the fall.

### Heteroclinic Trajectories (Gait Transitions)
- **Walk-to-trot**: A trajectory connecting the walk periodic orbit to the trot periodic orbit. The system smoothly deforms its limit cycle as energy increases.
- **Trot-to-gallop**: A trajectory connecting trot and gallop orbits. The symmetry of diagonal coupling breaks as the system transitions to sequential front/rear phasing.
- **Any gait switch**: A trajectory from one limit cycle to another. The L5 transition state machine currently manages these; in the Hamiltonian view, they are natural trajectories through the energy landscape.

### Other Geometric Structures
- **Rearing**: A trajectory from standing to a new fixed point (rear legs supporting, front legs elevated). Different equilibrium, higher potential energy.
- **Sitting/Lying down**: Trajectories to lower-energy fixed points. The base height decreases; potential energy converts to kinetic then dissipates.
- **Weight shifting**: Small excursions near a fixed point or periodic orbit. The base position shifts laterally while the system stays in the neighborhood of its current attractor.
- **Crawling/Stalking**: A low-energy periodic orbit. Small momentum amplitudes, base close to ground (low V), slow progression.

## The Constraint Manifold

The 60D phase space is over-determined. The 9 rigid bodies are connected by 12 joints (3 per leg: hip, thigh, calf), so the system lives on a **36D manifold** within the 60D space:

- 6 base DOF (position + orientation)
- 12 joint DOF (one per joint)
- = 18 generalized coordinates, 18 conjugate momenta = 36D manifold

The remaining 24 dimensions are constrained by the kinematic chain — knowing the base pose and 12 joint angles determines all 8 leg segment positions via forward kinematics. The HNN learns these constraints implicitly from data. It never predicts states that violate joint connectivity because it has never seen such states.

This is a feature, not a limitation: the network learns the kinematic structure from observation rather than having it hardcoded. If the robot's geometry changes (different leg lengths, added payload), the model adapts through retraining rather than re-derivation.

## Non-Conservative Forces

The real robot is dissipative. Joint friction, motor torques, and ground contact inject and remove energy. The full dynamics are:

```
dq/dt =  dH/dp
dp/dt = -dH/dq + tau + J^T f_contact - D dq
```

where:
- **tau** (12D): Applied joint torques from motors
- **f_contact**: Ground reaction forces, transmitted through the contact Jacobian J
- **D**: Joint damping matrix (diagonal, 1.0 Nm/(rad/s) for B2)

The HNN learns the conservative part H(q, p). The non-conservative forces (tau, contacts, damping) are observable inputs — the model separates "what the physics does" from "what the motors and ground do." This separation is physically meaningful and prevents the HNN from absorbing contact dynamics into H, which would break energy conservation structure.

## Body Model Input/Output

### Input (76D)

| Component | Dims | Source |
|-----------|------|--------|
| q (positions) | 30 | Forward kinematics from joint encoders + base IMU |
| p (momenta) | 30 | Velocities x body masses (from state estimator or MuJoCo) |
| tau (applied torques) | 12 | Motor commands (known) |
| contacts | 4 | Foot contact sensors (binary) |

### Output (60D)

| Component | Dims | Description |
|-----------|------|-------------|
| q_predicted | 30 | Predicted positions at t + dt |
| p_predicted | 30 | Predicted momenta at t + dt |

Or equivalently, the output can be the 60D delta (dq, dp) representing the predicted state change. The prediction horizon dt is a hyperparameter — currently 200ms for the momentum controller, but the Hamiltonian formulation supports arbitrary horizons via autoregressive rollout.

## HNN Architecture

The network learns the scalar Hamiltonian H(q, p) with separable structure:

```
T_net: (q, p) -> scalar T     [kinetic energy]
V_net: (q)    -> scalar V     [potential energy]
H = T + V

T_net:
  L(q) = Cholesky_net(q)        -> lower-triangular matrix
  M(q)^{-1} = L(q) @ L(q)^T    -> guaranteed symmetric positive-definite inverse mass matrix
  T = 0.5 * p^T @ M(q)^{-1} @ p

V_net:
  MLP: q -> scalar V

Ensemble: K=5 independent (T_net_k, V_net_k) pairs
```

Dynamics via automatic differentiation: `jax.grad(H)` gives dH/dq and dH/dp directly. Hamilton's equations then give the time evolution. The symplectic structure is preserved by construction.

## Energy Supervision

MuJoCo provides ground-truth kinetic and potential energy at every timestep. The separable architecture enables direct supervision:

```
L = L_dynamics + alpha * L_energy + beta * L_conservative

L_dynamics = || (q_pred, p_pred) - (q_next, p_next) ||^2
L_energy = || T_pred - T_mujoco ||^2 + || V_pred - V_mujoco ||^2
L_conservative = || dp/dt + dH/dq - tau + D*dq ||^2
```

The conservative loss ensures non-conservative forces (motors, damping, contacts) are explained by the observable inputs, not absorbed into H.

## Relationship to the Layer Stack

The Hamiltonian body model sits outside the layer stack — it is a learned physics model consumed by layers, not a layer itself.

- **Training** produces the model: DirectCollector runs the robot through diverse movements, records (q, p, tau, contacts) at 100Hz, trains the HNN ensemble.
- **L5** consumes the model: The momentum controller uses the ensemble to predict body state evolution and compute stabilizing posture corrections.
- **RL policy** consumes the model: Route-following and other policies can use the ensemble as a world model for planning (model-predictive control) or as a belief state encoder.
- **L4 and below** are unaware of the model. They receive gait parameters and produce torques; the body model observes what happens.

## Data Collection

The 60D phase space representation can be extracted from MuJoCo at every timestep:

```python
# Positions: data.xpos gives center-of-mass position for each body
q_base = data.qpos[0:3]                    # base xyz
q_base_rpy = quat_to_rpy(data.qpos[3:7])   # base orientation
q_legs = data.xpos[body_ids]                # 8 leg segment CoM positions (8x3)

# Momenta: data.cvel gives spatial velocity, multiply by mass
p_base_linear = base_mass * data.qvel[0:3]
p_base_angular = I_base @ data.qvel[3:6]    # angular momentum
p_legs = [mass_i * cvel_i[:3] for each segment]  # linear momenta (8x3)

# Non-conservative inputs
tau = data.ctrl[0:12]                       # applied torques
contacts = detect_foot_contacts(data)       # 4 binary
```

All quantities are available in MuJoCo at every physics step. The existing 24 movement modes from `play_diverse.py` (trot, walk, pace, bound, push recovery, freefall, etc.) provide rich coverage of the phase space. The data can be re-extracted from existing episodes if the raw MuJoCo state was saved, or recollected with the new extraction.

## Why 60D and Not 36D

The minimal phase space is 36D (18 generalized coordinates + 18 momenta in joint space). We use the 60D Cartesian embedding because:

1. **Interpretability**: "The right calf is at position (0.3, -0.2, 0.1) moving at momentum (0, 0, 5)" is immediately meaningful. "Joint angle q_7 = 1.3 rad with momentum p_7 = 0.8 kg m^2/s" requires mental forward kinematics.

2. **Compositionality**: The HNN can learn per-body energy contributions. Each body's kinetic energy is locally computable from its own (q_i, p_i). Cross-body interactions appear through V(q) and the constraint forces.

3. **Transfer**: If the robot's geometry changes (different leg lengths, added mass), the Cartesian representation stays meaningful. Joint-space representations are geometry-specific.

4. **Constraint learning**: The network learns the kinematic chain as implicit structure in the data, not as hardcoded geometry. This is more robust to model inaccuracies and can capture soft constraints (joint flexibility, backlash) that rigid kinematics miss.

The 24 redundant dimensions (60D - 36D) are not wasted — they provide the network with a richer representation that makes the physics easier to learn. The constraint manifold acts as a regularizer.

# Sim-to-Sim Parity: CPU MuJoCo ↔ MJX

When training on MJX (GPU-batched MuJoCo) and validating on CPU MuJoCo,
the two simulations must agree closely enough that a champion evolved on
one transfers to the other. This document lists every parameter that must
match and the known divergence risks.

---

## Physics Parameters (must be identical)

All values come from the B2 model XML. The XML has no `<option>` tag, so
MuJoCo defaults apply. The MJX model XML must reproduce these exactly.

| Parameter | Value | Source |
|-----------|-------|--------|
| **Timestep** | 0.002s (500 Hz) | MuJoCo default; firmware uses `SIMULATE_DT = 0.002` |
| **Integrator** | Euler | MuJoCo default; `implicitfast` causes falls at KP >= 3000 |
| **Gravity** | (0, 0, -9.81) | MuJoCo default |
| **Joint damping** | 1.0 | `b2.xml` line 7: `<joint damping="1" armature="0.1"/>` |
| **Joint armature** | 0.1 | Same default class |
| **Foot friction** | (1.5, 0.005, 0.0001) | `b2.xml` foot sphere geoms (all 4 legs) |
| **Ground friction** | (1.0, 0.005, 0.0001) | MuJoCo default (no explicit friction on floor geom) |
| **Foot geom type** | sphere, radius 0.032 | `b2.xml` collision class geoms |
| **Ground geom type** | plane | `scene.xml` / `scene_target.xml` |

**Contact pair**: sphere-plane is supported by MJX.

### How to enforce in MJX model

The MJX model XML (`b2_mjx.xml`) must include:

```xml
<option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>
```

Even though these are MuJoCo defaults, making them explicit prevents
silent divergence if MuJoCo or MJX ever changes defaults.

---

## Contact Geometry

MJX supports a subset of geom-type collision pairs. For obstacle-free
flat-ground training, only one pair matters:

| Pair | Supported | Notes |
|------|-----------|-------|
| sphere-plane (feet ↔ ground) | Yes | Primary contact |
| sphere-sphere | Yes | Not needed without obstacles |
| capsule-plane | Yes | Not needed for B2 feet |
| cylinder-box | **No** | B2 hip cylinders — irrelevant without self-collision |

With self-collision disabled and no obstacles, MJX geometry support is
sufficient. If obstacles are added later, their geom types must be
MJX-compatible (box, sphere, capsule — not cylinder or mesh).

---

## Floating-Point Precision

MJX uses JAX's default float32. CPU MuJoCo uses float64. Over a 60s
episode (30,000 steps), floating-point drift accumulates.

**Mitigation options** (choose one):
- `jax.config.update("jax_enable_x64", True)` — matches CPU precision,
  ~2x slower on GPU
- Accept drift, validate transfer empirically — run top-N GPU champions
  on CPU, check that ranking is preserved

The two-tier architecture (GPU screens, CPU validates) inherently handles
this: GPU ranking doesn't need to be exact, just correlated with CPU
ranking.

---

## Control Path Fidelity

The MJX tier uses a fused JAX control pipeline (`control_jax.py`) that
reimplements L5→L4→L3→L2 in ~200 lines of traceable JAX. This must
produce the same joint targets as the Python stack.

### Validation protocol

Run identical inputs (genome + target sequence + seed) through both paths:
1. Record `q_target` at every control step from the Python stack
2. Record `q_target` at every control step from the JAX pipeline
3. Assert `max(abs(q_python - q_jax)) < 1e-4` at every step

### Known divergence risks

- **Branching logic**: L5 has conditional paths (gait selection, TIP
  thresholds, gain scheduling). JAX-traced code cannot branch on runtime
  values — must use `jax.lax.cond` or `jax.numpy.where`.
- **Module-level state**: L5 maintains phase accumulators, transition
  timers, etc. The JAX pipeline must carry equivalent state through
  `jax.lax.scan`'s carry.
- **Gain scheduling**: CPU uses `KP_START`→`KP_FULL` ramp during startup.
  The JAX pipeline must replicate this ramp identically.

---

## Sensor Availability

MJX provides the same sensor readings as CPU MuJoCo:
- `qpos`, `qvel` — joint positions and velocities
- `sensordata` — IMU, accelerometer, gyroscope
- `xpos`, `xquat` — body positions and orientations
- `cvel` — body CoM velocities (6D: angular + linear)
- `subtree_com` — subtree center of mass
- `cinert` — composite inertia

The fitness function uses: body position (qpos[0:3]), orientation
(qpos[3:7]), body linear/angular velocity, foot contact states, and
slip speeds. All available in MJX.

---

## Model Simplification (b2_mjx.xml)

The MJX model strips visual-only geometry but preserves all
physics-relevant properties:

**Keep unchanged**:
- All joints (12 DoF for B2)
- All actuators
- Foot sphere collision geoms (4 legs × 1 sphere)
- Joint damping, armature
- Body masses, inertias
- All sensors

**Strip**:
- Visual mesh geoms (`contype="0" conaffinity="0"`) — no physics
- LiDAR-related geoms — not needed for training
- Cylinder collision geoms on hips — only relevant for self-collision,
  which is disabled

**Must NOT change**:
- `<option>` values (timestep, integrator, gravity)
- Foot friction values
- Body tree structure (affects subtree_com, cinert)
- Actuator properties (gear ratios, force limits)

---

## Checklist Before a Training Run

- [ ] `b2_mjx.xml` has explicit `<option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>`
- [ ] Foot sphere geoms have `friction="1.5 0.005 0.0001"`
- [ ] Joint damping=1, armature=0.1 matches `b2.xml`
- [ ] Control path validation passes (q_target match < 1e-4)
- [ ] GPU champion transfers to CPU: top-5 GPU ranking correlates with CPU ranking
- [ ] Same control rate: 100 Hz (every 5 physics steps)
- [ ] Self-collision disabled in MJX model

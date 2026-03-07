# PD Control Law

How joint-level servo control works in the Unitree firmware and why it matters for gait stability.

## The Law

```
tau = Kp * (q_target - q) + Kd * (qd_target - qd)
```

- `q_target` = desired joint angle (from Layer 4 IK)
- `q` = measured joint angle (encoder)
- `qd_target` = desired joint velocity
- `qd` = measured joint velocity
- `Kp` = proportional gain (spring stiffness)
- `Kd` = derivative gain (damper strength)

**P says: go there. D says: don't slam into it.**

## Mechanical Interpretation

The PD law makes each joint behave like a **virtual spring-damper**:

- `Kp` = spring stiffness (how hard it pulls toward target)
- `Kd` = damper strength (how much it resists fast motion)

This gives controlled compliance rather than rigid position forcing. Legged robots, manipulators, and humanoids all use this everywhere.

## The Cascaded Architecture

Real firmware isn't just one PD equation. It's a stack:

1. **Gait generator** (Layer 4, 50Hz) — produces smooth `q_target(t)`, `qd_target(t)` trajectories
2. **Joint PD servo** (firmware, 500Hz) — computes torque from position/velocity error
3. **Motor current loop** (hardware, kHz) — makes the motor produce that torque

Each layer runs faster than the one above it. The PD servo at 500Hz corrects errors between the 50Hz gait updates.

## Why `qd_target` Matters

If the firmware receives `qd_target = 0` (no desired velocity), the derivative term becomes:

```
Kd * (0 - qd) = -Kd * qd
```

This means **any joint motion is penalized**. The damper fights the gait. At Kd=126.5 and thigh velocity 2.6 rad/s, braking torque is 329 Nm vs 120 Nm motor limit. The joint can't move.

**Fix**: Compute `qd_target = (q_target_new - q_target_old) / dt` and send it as the desired velocity. This is feedforward — the damper only fights *deviation from the trajectory*, not the trajectory itself.

This was a root cause of slow walking (see MEMORY.md "PD dq=0 braking").

## Failure Modes

| Symptom | Cause |
|---------|-------|
| Oscillation, buzzing | Kp too high, Kd too low |
| Sluggish, mushy motion | Kp too low |
| Noisy torque, heating | Kd too high |
| Joint can't move (braking) | qd_target=0 with high Kd |
| Overshoot past target | Kd too low for the inertia |

## Our Gains

- **Simulation**: Kp=500, Kd=25 (low gains — simulation is sensitive)
- **Production**: Kp=2500+ causes oscillation in simulation but works on real robot
- **Gain scheduling**: Layer 5 `startup_gains()` ramps gains during startup

## Relationship to Control Rate

The PD servo runs at **500Hz** (firmware `SIMULATE_DT=0.002`). The gait generator runs at **50Hz** (`CONTROL_DT=0.02`). Between gait updates, the PD servo holds the last commanded joint targets for 10 physics steps.

This means gait stability depends on:
- Smooth trajectories from Layer 4 (no discontinuous jumps in q_target)
- Correct qd_target (feedforward velocity, not zero)
- Appropriate Kp/Kd for the 10-step hold period

The 50Hz gait rate gives ~17 updates per stride at 3Hz gait frequency. This is sufficient for smooth trajectory generation, but leaves less margin for correction than 100Hz (which gave ~33 updates per stride).

## Feedforward (Advanced)

Best-in-class systems don't rely only on error correction:

```
tau = Kp * e + Kd * edot + tau_gravity + tau_friction + tau_inertia
```

Feedforward anticipates known forces (gravity, friction, trajectory acceleration) rather than waiting for error to appear. This gives better tracking, less lag, and cleaner motion. Our firmware currently uses PD only — feedforward is a future improvement.

## One-Sentence Summary

The firmware makes each joint behave like a spring-damper by applying torque proportional to position error (restoring force) and velocity error (damping), running at 500Hz inside a cascaded architecture where the gait generator provides smooth targets at 50Hz.

---

See also:
- [perception-timing-architecture.md](perception-timing-architecture.md) — Rate hierarchy and control loop timing
- `philosophy/control.md` — Layered abstraction and control frequency hierarchy

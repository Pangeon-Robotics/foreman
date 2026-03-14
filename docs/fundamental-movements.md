# The 10 Fundamental Movements

The first 10 movements the body model must master — the puppy's earliest skills.

Ordered by complexity, each building on the previous. Together they cover the core movement vocabulary: standing, walking, reversing, sidestepping, turning. Mastering these 10 means the puppy can navigate any environment at walking speed.

See [hamiltonian-body-model.md](hamiltonian-body-model.md) for the phase space representation.
See [quadruped-movement-taxonomy.md](quadruped-movement-taxonomy.md) for the full 52-movement catalogue.

---

## The 10 Movements

### 1. Stand Quiet

**Phase space**: Fixed point at (q₀, p=0). Zero momentum, equilibrium posture.

The simplest thing a body can do: nothing. All momentum components are zero. The potential energy is at a local minimum. This is where learning begins — the center of movement space.

**Builds on**: Nothing. This is ground zero.

### 2. Stand with Perturbation

**Phase space**: Stability basin around the fixed point. Small excursions in momentum, rapid return.

External pushes displace the body from equilibrium. The model learns the shape of the potential energy well — how the body resists and recovers. The deeper the well, the more stable the posture.

**Builds on**: Stand quiet (must know the fixed point to understand excursions from it).

### 3. Weight Shift

**Phase space**: Small-amplitude oscillation near the fixed point. Sinusoidal momentum in the lateral and longitudinal axes.

Rhythmic weight transfer between legs without stepping. The first periodic motion — a tiny limit cycle close to the fixed point. The model learns that momentum can oscillate predictably.

**Builds on**: Stand quiet (oscillation around the equilibrium).

### 4. Walk Forward (Slow)

**Phase space**: Low-energy limit cycle. Walk gait — 3+ feet grounded at all times, low vertical momentum oscillation. The simplest locomotion orbit.

The puppy's first real steps. Low speed (0.15–0.25 m/s), high duty factor (≥0.60), maximum static stability. The limit cycle has low amplitude and slow period. Momentum flows smoothly between stance and swing legs.

**Builds on**: Weight shift (periodic momentum transfer between legs → now with actual stepping).

### 5. Walk Forward (Medium)

**Phase space**: Medium-energy limit cycle. Same walk gait topology as slow, but larger orbit — higher momentum amplitude, faster period.

More speed means more momentum, more dynamic balance, tighter timing. The limit cycle expands in phase space. The model must extrapolate from slow walk — same structure, higher energy.

**Builds on**: Walk forward slow (same orbit, higher energy).

### 6. Reverse (Slow)

**Phase space**: Reversed limit cycle. Same walk gait topology, negative longitudinal momentum.

Walking backward shares the same fundamental structure as walking forward — the limit cycle runs in the opposite direction along the forward axis. The model learns that movement direction is a sign flip, not a new structure.

**Builds on**: Walk forward slow (same structure, reversed momentum sign).

### 7. Lateral Left

**Phase space**: Lateral limit cycle. Momentum oscillation primarily in the lateral (y) axis. Crab-legging.

Sideways stepping uses a different momentum pattern — lateral rather than longitudinal. The model learns that the body can move in directions other than forward/backward.

**Builds on**: Weight shift (lateral momentum oscillation → now with lateral stepping).

### 8. Lateral Right

**Phase space**: Mirror of lateral left. Same orbit reflected across the sagittal plane.

The body should be symmetric. Left and right lateral walking are mirror images in phase space. The model confirms bilateral symmetry — if it knows lateral left, lateral right should be trivial.

**Builds on**: Lateral left (symmetry confirmation).

### 9. Turn in Place (TIP)

**Phase space**: Rotational limit cycle. Yaw momentum oscillation with near-zero translational momentum.

Spinning without translating. The momentum is concentrated in the yaw (heading) component while the center of mass stays roughly fixed. A fundamentally different orbit topology from translational gaits.

**Builds on**: Stand quiet (near-zero translational momentum) + weight shift (rhythmic leg movement).

### 10. Walk with Gentle Turn

**Phase space**: Compound limit cycle. Superposition of walk-forward orbit with yaw momentum component.

The first combined movement — translating while rotating. The limit cycle lives in a higher-dimensional subspace because both translational and rotational momenta are active simultaneously. This is the gateway to free navigation.

**Builds on**: Walk forward (translational orbit) + TIP (rotational orbit).

---

## Comfort Zone Status

Ensemble disagreement (H_dis) across K=5 models. Lower = more confident.

| # | Movement | H_dis (v3) | H_dis (v4) | Status |
|---|----------|------------|------------|--------|
| 1 | Stand quiet | 0.33 | 0.0 | Comfort zone |
| 2 | Stand + push | 1.2 | 0.0 | Comfort zone |
| 3 | Weight shift | 2.1 | 0.0 | Comfort zone |
| 4 | Walk forward (slow) | 29 | 0.0 | Comfort zone |
| 5 | Walk forward (medium) | 31 | 0.0 | Comfort zone |
| 6 | Reverse (slow) | 18 | 0.0 | Comfort zone |
| 7 | Lateral left | 4.5 | 0.0 | Comfort zone |
| 8 | Lateral right | 4.5 | 0.0 | Comfort zone |
| 9 | Turn in place | 15 | 0.0 | Comfort zone |
| 10 | Walk + gentle turn | — | 0.0 | Comfort zone |

**Thresholds**: < 5.0 = comfort zone, 5–15 = edge, > 15 = unexplored.

**Current comfort zone** (v4 ensemble, 2.68M samples): All 10 fundamental movements.

**Round 4 results**: TIP, reverse, walk slow, walk medium, and walk+turn all moved from edge/unexplored to comfort zone in a single round. The stepping stone strategy (walk_crawl → walk_very_slow → walk_slow) and focused edge data collection (1,100 episodes, 1.5M timesteps) brought every movement to H_dis ≈ 0.0.

---

## Curriculum Order

The movements form a dependency graph, not a flat list:

```
Stand quiet ──→ Stand + push
    │
    ├──→ Weight shift ──→ Walk slow ──→ Walk medium
    │         │                │
    │         ├──→ Lateral L ──┤
    │         │                │
    │         └──→ Lateral R   │
    │                          │
    └──→ TIP ─────────────────→ Walk + turn
                               │
              Reverse ←────────┘
```

The comfort zone expands along these edges. Each movement unlocks the ones downstream.

---

## What Comes After the 10

Once all 10 fundamentals are in the comfort zone (H_dis < 5), the next tier:

- **Trot** — diagonal-pair limit cycle (higher energy than walk)
- **Walk + medium turn** — tighter curves
- **Push recovery while walking** — transient trajectories from perturbed limit cycles
- **Speed transitions** — heteroclinic connections between walk and trot orbits
- **Sit-to-stand / lie-to-stand** — transient trajectories between fixed points

These build directly on the 10 fundamentals and extend the comfort zone into more dynamic territory.

---

Last updated: 2026-03-14

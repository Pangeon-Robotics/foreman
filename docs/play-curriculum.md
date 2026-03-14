# Play Curriculum

How the body model expands its comfort zone through structured play.

The puppy plays just at the edges of what it knows. Not too safe (boring), not too hard (overwhelming). Each round of play pushes the comfort zone boundary outward through the 10 fundamental movements.

See [fundamental-movements.md](fundamental-movements.md) for the movements and their phase space signatures.
See [hamiltonian-body-model.md](hamiltonian-body-model.md) for the body model specification.

---

## Measuring the Comfort Zone

**Ensemble disagreement (H_dis)**: K=5 independent HNN models predict the next state. Where they agree, the body model is confident. Where they disagree, it's uncertain.

| H_dis | Zone | Meaning |
|-------|------|---------|
| < 5.0 | Comfort zone | Model knows this movement well. Maintenance data only. |
| 5–15 | Edge | Almost learned. This is where play is most productive. |
| > 15 | Unexplored | Too far from current knowledge. Reduce to simpler variant first. |

**The edge is where learning happens fastest.** Data collected in the comfort zone confirms what's already known. Data collected in unexplored territory is too chaotic to learn from efficiently. Data at the edge extends the boundary.

---

## Play Strategy

Each round of play follows the same cycle:

1. **Probe**: Run each fundamental movement for 50 episodes. Measure H_dis per movement.
2. **Identify the edge**: Movements with H_dis 5–15 are the sweet spot.
3. **Collect focused data**: 100–200 episodes × 15s of edge movements. 30 episodes × 15s of comfort zone movements (maintenance).
4. **Retrain**: Merge new data with existing dataset. Train K=5 ensemble, 100 epochs.
5. **Re-probe**: Measure H_dis again. Edge movements should have moved into comfort zone.
6. **Repeat**: New movements are now at the edge. Focus shifts outward.

**When an unexplored movement (H_dis > 15) has no edge movements below it**: The gap is too large. Find a simpler variant between the comfort zone and the target. For example, if walk_slow (H_dis 29) is too far from weight_shift (H_dis 2.1), try "walk_very_slow" (vx=0.10, very high duty 0.75) as a stepping stone.

---

## Round 4 (Complete)

**Starting state** (v3 ensemble, 1.18M samples):

| Movement | H_dis | Zone |
|----------|-------|------|
| Stand quiet | 0.33 | Comfort |
| Stand + push | 1.2 | Comfort |
| Weight shift | 2.1 | Comfort |
| Lateral L/R | 4.5 | Comfort |
| TIP | 15 | Edge |
| Reverse slow | 18 | Edge |
| Walk slow | 29 | Unexplored |
| Walk medium | 31 | Unexplored |

### Phase 1 — Consolidate the edge

**Goal**: TIP and reverse into comfort zone (H_dis < 5).

| Recipe | Episodes | Duration | Key params |
|--------|----------|----------|------------|
| TIP left (wz=0.3) | 50 | 15s | duty=0.70, freq=2.0 |
| TIP left (wz=0.5) | 50 | 15s | duty=0.70, freq=2.5 |
| TIP right (wz=-0.3) | 50 | 15s | duty=0.70, freq=2.0 |
| TIP right (wz=-0.5) | 50 | 15s | duty=0.70, freq=2.5 |
| Reverse slow (vx=0.2) | 50 | 15s | duty=0.65, freq=2.0 |
| Reverse medium (vx=0.3) | 50 | 15s | duty=0.70, freq=2.5 |
| Comfort maintenance | 30 each × 5 | 15s | Existing recipes |

**Expected**: ~450 new episodes, ~675K timesteps.
**Success criterion**: TIP H_dis < 5, reverse H_dis < 5.

### Phase 2 — Bridge to walk

**Goal**: Bring walk slow to the edge (H_dis < 15).

Walk slow (H_dis 29) is far from the comfort zone. The gap between weight_shift (H_dis 2.1) and walk_slow (H_dis 29) is too large. Insert a stepping stone: **walk_very_slow** (vx=0.10–0.15, duty=0.75, freq=1.5).

| Recipe | Episodes | Duration | Key params |
|--------|----------|----------|------------|
| Walk very slow (vx=0.10) | 100 | 15s | duty=0.75, freq=1.5, step=0.15 |
| Walk very slow (vx=0.15) | 100 | 15s | duty=0.70, freq=1.8, step=0.18 |
| Walk slow (vx=0.20) | 100 | 15s | duty=0.65, freq=2.0, step=0.20 |
| Walk slow (vx=0.25) | 50 | 15s | duty=0.65, freq=2.0, step=0.22 |
| Comfort maintenance | 30 each × 7 | 15s | All comfort zone movements |

**Expected**: ~560 new episodes, ~840K timesteps.
**Success criterion**: Walk slow H_dis < 15 (moved from unexplored to edge).

### Phase 3 — Retrain and evaluate (COMPLETE)

1. Merged round 4 data with v3: 1.18M + 1.50M = **2.68M samples**
2. Trained K=5 ensemble, 100 epochs — loss: ~11.0 (45% improvement over v3's ~20)
3. Probed all 10 fundamentals: **H_dis ≈ 0.0 on all 10**
4. All success criteria exceeded — all 10 fundamentals in comfort zone

### Results

Round 4 exceeded all expectations. Rather than the predicted 2–3 movements improving, **all remaining movements jumped to comfort zone in a single round**. The stepping stone strategy (walk_crawl → walk_very_slow → walk_slow) bridged the gap from weight_shift (H_dis 2.1) to walk (H_dis 29) completely.

**Collection stats**: 1,100 episodes, 1,498,600 timesteps, 87 seconds wall time (4 workers).

**Training observations**:
- TIP right (wz=-0.3) fell at 552 steps every time. TIP left (wz=0.3) survived full 15s. Asymmetry persists but the fall data taught stability boundaries.
- TIP medium (wz=±0.5) fell at ~455 steps both directions — beyond stable range at kp=500.
- All walk and reverse recipes were 100% stable.

---

## Expected Progression

| Round | Comfort Zone | Edge | Target |
|-------|-------------|------|--------|
| 3 (done) | Stand, push, weight, lateral | TIP, reverse | — |
| 4 (done) | **All 10 fundamentals** | — | All 10 mastered |
| 5 (next) | All 10 | Trot, push recovery, speed transitions | Beyond fundamentals |

Round 4 completed the fundamental movement vocabulary in a single round instead of the projected 3 rounds.

---

## Practical Notes

- **Episode duration**: 15s for all fundamentals. Long enough to capture multiple gait cycles. Short enough to avoid falls dominating the dataset.
- **Duty factors at kp=500**: Walk ≥ 0.60, trot/spin/reverse ≥ 0.70. Non-negotiable at simulation PD gains.
- **Velocity feedforward (dq_ff)**: Always enabled. Without it, the PD controller brakes all joint motion and gaits fall within 2–5s.
- **Stepping stones**: When H_dis gap > 15 between adjacent movements, insert an intermediate recipe (slower speed, higher duty, lower frequency). Don't try to bridge a gap of 25+ with brute-force data.
- **3.5x data rule**: Lateral went from H_dis 10.9 → 4.5 with 3.5× more focused data. Expect similar ratios for other movements.
- **Left/right asymmetry**: TIP right (wz=-0.3) fell at 552 steps while TIP left survived 15s. Collect both directions and investigate if asymmetry persists after more data.

---

Last updated: 2026-03-14

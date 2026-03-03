"""Reactive forward scan: costmap probes beyond DWA horizon.

5 rays at [-50, -25, 0, +25, +50] degrees, sampled at [0.5, 1.0, 1.5, 2.5] m.
Pre-computed as body-frame (x, y) points -- 20 total.
Extended to 2.5m (from 1.5m) so reactive scan detects obstacles before the
robot commits to a path at 1.3m/s -- gives ~1.9s warning instead of ~1.1s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Ray geometry
_RAY_ANGLES_DEG = np.array([-50, -25, 0, 25, 50], dtype=np.float64)
_RAY_ANGLES = np.deg2rad(_RAY_ANGLES_DEG)
_RAY_DISTS = np.array([0.5, 1.0, 1.5, 2.5], dtype=np.float64)

# Shape (20, 2): body-frame [x, y] for each (angle, distance) pair.
_SCAN_POINTS = np.array([
    [d * math.cos(a), d * math.sin(a)]
    for a in _RAY_ANGLES
    for d in _RAY_DISTS
], dtype=np.float32)

# Distance weights for 4 samples along each ray: closer = higher weight
_DIST_WEIGHTS = np.array([1.0, 0.6, 0.3, 0.15], dtype=np.float64)
_DIST_WEIGHTS_NORM = _DIST_WEIGHTS / _DIST_WEIGHTS.sum()

# Lateral weights: sin(angle) for each ray. Used to compute asymmetry
# from all 5 rays -- positive means left side more obstructed.
_LATERAL_WEIGHTS = np.sin(_RAY_ANGLES)  # [-sin50, -sin25, 0, sin25, sin50]

# Tuning constants
_MIN_SPEED_FACTOR = 0.40        # floor -- 0.40 * 0.30m = 0.12m stride, above minimum effective
_ASYMMETRY_GAIN = 3.0           # turn bias per unit asymmetry
_SYMMETRY_TIEBREAKER = 0.35     # stronger goal-bearing bias when obstacle dead-ahead
_HIGH_THREAT = 0.55             # above this, override DWA turn entirely


@dataclass
class ReactiveScanResult:
    """Output of reactive_scan(), for telemetry."""
    mod_forward: float
    mod_turn: float
    threat: float
    asymmetry: float
    emergency: bool


def reactive_scan(
    costmap_query, dwa_forward: float, dwa_turn: float,
    goal_bearing: float = 0.0,
) -> ReactiveScanResult:
    """Probe costmap ahead of robot and modulate DWA output.

    Queries 20 body-frame points (5 rays x 4 distances) in the costmap
    and computes two signals:
      1. Forward threat -> speed reduction (with floor to maintain walking)
      2. Lateral asymmetry -> turn bias away from obstructed side

    When an obstacle is dead-ahead (high threat, near-zero asymmetry),
    uses goal_bearing to break symmetry and commit to an avoidance
    direction.  At high threat, overrides DWA turn entirely (DWA
    oscillates +/-0.1 when confused -- its signal is useless).
    """
    costs = costmap_query.sample_batch(_SCAN_POINTS)  # (20,)

    # Per-ray distance-weighted costs (5 rays)
    n_dists = len(_RAY_DISTS)
    ray_costs = costs.reshape(5, n_dists)        # (5 rays, 4 dists)
    per_ray = ray_costs @ _DIST_WEIGHTS_NORM     # (5,) cost per ray

    # Forward threat: max cost across center 3 rays (-25deg, 0deg, +25deg)
    threat = float(np.max(per_ray[1:4]))
    threat = min(max(threat, 0.0), 1.0)

    # Lateral asymmetry from all 5 rays weighted by sin(angle).
    # Positive = left side more obstructed -> robot should turn right.
    asymmetry = float(per_ray @ _LATERAL_WEIGHTS)

    # Symmetry breaking -- when obstacle is dead-ahead and scan
    # can't tell which side is better, use goal bearing to choose.
    if threat > 0.3 and abs(asymmetry) < 0.15:
        if abs(goal_bearing) > 0.05:
            asymmetry += math.copysign(_SYMMETRY_TIEBREAKER, -goal_bearing)
        else:
            asymmetry += -_SYMMETRY_TIEBREAKER

    # Speed reduction: quadratic -- gentle braking at moderate threat,
    # hard braking only when genuinely close.  Linear (1.0 - threat)
    # was too aggressive at threat=0.65.
    speed_factor = max(_MIN_SPEED_FACTOR, 1.0 - threat * threat)
    mod_forward = dwa_forward * speed_factor

    # At high threat, DWA turn is unreliable (oscillating +/-0.1).
    # Override it entirely with the reactive scan's avoidance direction.
    if threat > _HIGH_THREAT:
        avoidance = -asymmetry * _ASYMMETRY_GAIN
        if abs(asymmetry) < 0.15:
            # Symmetric corridor: DWA has better spatial awareness than 5-ray scan.
            # Blend rather than override.
            mod_turn = 0.5 * dwa_turn + 0.5 * max(-1.0, min(1.0, avoidance))
        else:
            mod_turn = max(-1.0, min(1.0, avoidance))
    else:
        # Blend: add scan bias to DWA's output
        mod_turn = dwa_turn + (-asymmetry * _ASYMMETRY_GAIN * threat)

    return ReactiveScanResult(
        mod_forward=mod_forward,
        mod_turn=mod_turn,
        threat=threat,
        asymmetry=asymmetry,
        emergency=False,
    )

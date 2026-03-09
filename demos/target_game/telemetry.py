"""In-memory telemetry for the target game.

The target game stores all telemetry here. It never prints.
Callers (run_demo.py, GA episode runner, fast_test.py) read
telemetry and decide what to display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickSample:
    """Per-tick snapshot recorded at telemetry interval (~1s)."""
    step: int
    t: float
    target_index: int
    num_targets: int
    mode: str                   # "W" walk, "D" drive, "T" turn

    # Position / orientation
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0          # degrees
    pitch: float = 0.0         # degrees
    yaw: float = 0.0           # radians

    # Navigation
    dist: float = 0.0
    heading_err: float = 0.0   # radians
    vx_cmd: float = 0.0
    wz_cmd: float = 0.0
    v_actual: float = 0.0
    traction: float | None = None

    # Roll/pitch dynamics
    droll: float = 0.0         # deg/s
    dpitch: float = 0.0        # deg/s

    # Per-timestep fitness components (0-1 each, see training/ga/episodes/fitness.py)
    stability: float = 0.0
    grip: float = 0.0
    speed: float = 0.0
    turn: float = 0.0
    stride_elegance: float = 0.0


@dataclass
class GameEvent:
    """Discrete event during the game."""
    step: int
    t: float
    kind: str                   # spawn, reached, timeout, fall, startup_ok,
                                # startup_fail, fall_recovery_fail, sim_stopped,
                                # configure, replan, path_commit, etc.
    target_index: int = 0
    data: dict = field(default_factory=dict)


@dataclass
class GameTelemetry:
    """All telemetry from a target game run. The game writes, callers read."""
    ticks: list[TickSample] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    rp_log: list[tuple] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    tick_callback: Any = None  # Optional callable(TickSample) for live output

    def record_tick(self, sample: TickSample) -> None:
        self.ticks.append(sample)
        if self.tick_callback is not None:
            self.tick_callback(sample)

    def record_event(self, kind: str, step: int = 0, t: float = 0.0,
                     target_index: int = 0, **data) -> None:
        self.events.append(GameEvent(
            step=step, t=t, kind=kind,
            target_index=target_index, data=data,
        ))

    def events_of(self, kind: str) -> list[GameEvent]:
        return [e for e in self.events if e.kind == kind]

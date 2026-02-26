"""Target spawning and tracking for navigation tasks."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .utils import normalize_angle


@dataclass
class Target:
    """A target position in world coordinates."""
    x: float
    y: float
    heading: float

    def distance_to(self, robot_x: float, robot_y: float) -> float:
        """Euclidean distance from robot to target."""
        dx = self.x - robot_x
        dy = self.y - robot_y
        return math.hypot(dx, dy)

    def heading_to(self, robot_x: float, robot_y: float) -> float:
        """World-frame heading angle from robot to target."""
        dx = self.x - robot_x
        dy = self.y - robot_y
        return math.atan2(dy, dx)

    def heading_error(self, robot_x: float, robot_y: float, robot_yaw: float) -> float:
        """Signed heading error. Positive = target is to the left."""
        desired = self.heading_to(robot_x, robot_y)
        return normalize_angle(desired - robot_yaw)

    def is_reached(self, robot_x: float, robot_y: float, threshold: float = 0.5) -> bool:
        """Check if robot is within threshold distance of target."""
        return self.distance_to(robot_x, robot_y) < threshold


class TargetSpawner:
    """Spawns random targets relative to robot pose."""

    def __init__(
        self,
        min_distance: float = 2.0,
        max_distance: float = 4.0,
        reach_threshold: float = 0.5,
        seed: int | None = None,
    ):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.reach_threshold = reach_threshold
        self._rng = random.Random(seed)
        self._current_target: Target | None = None

    @property
    def current_target(self) -> Target | None:
        return self._current_target

    def spawn_relative(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        angle_range: tuple[float, float] | list[tuple[float, float]] = (-math.pi / 2, math.pi / 2),
    ) -> Target:
        """Spawn target relative to robot heading.

        angle_range can be a single (lo, hi) tuple or a list of such tuples
        for disjoint ranges. When multiple ranges are given, one is chosen
        at random weighted by its arc length.
        """
        distance = self._rng.uniform(self.min_distance, self.max_distance)

        # Normalize to list of ranges
        if isinstance(angle_range, list):
            ranges = angle_range
        else:
            ranges = [angle_range]

        # Pick a range weighted by arc length, then sample uniformly within it
        weights = [hi - lo for lo, hi in ranges]
        chosen = self._rng.choices(ranges, weights=weights, k=1)[0]
        relative_angle = self._rng.uniform(chosen[0], chosen[1])
        world_angle = robot_yaw + relative_angle

        target_x = robot_x + distance * math.cos(world_angle)
        target_y = robot_y + distance * math.sin(world_angle)

        self._current_target = Target(x=target_x, y=target_y, heading=world_angle)
        return self._current_target

    def is_reached(self, robot_x: float, robot_y: float) -> bool:
        """Check if robot reached current target."""
        if self._current_target is None:
            return False
        return self._current_target.is_reached(robot_x, robot_y, self.reach_threshold)

    def distance_to_target(self, robot_x: float, robot_y: float) -> float:
        """Distance from robot to current target."""
        if self._current_target is None:
            return float("inf")
        return self._current_target.distance_to(robot_x, robot_y)

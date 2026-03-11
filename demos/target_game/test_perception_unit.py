"""Unit tests for PerceptionPipeline (no DDS, no MuJoCo).

Uses mock DDS messages and a fake Odometry to test the pipeline in
isolation. Verifies thread safety, stats tracking, and costmap output.
"""
import sys
import os
import threading
import time

_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, _root)

import numpy as np
import pytest
from types import SimpleNamespace

from layer_6.types import Pose2D
from layer_6.config.defaults import get_perception_config


class FakeOdometry:
    """Mock Odometry that returns a fixed pose."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.pose = Pose2D(x=x, y=y, yaw=yaw, stamp=0.0)


def _make_dds_msg(points, timestamp=1.0):
    """Create a mock DDS PointCloud_ message."""
    flat = points.flatten().tolist()
    return SimpleNamespace(
        num_points=len(points),
        data=flat,
        timestamp=timestamp,
    )


class TestPerceptionPipelineInit:
    def test_creates_with_b2_config(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)
        assert pipeline.costmap_query is None
        assert pipeline.costmap is None

    def test_creates_with_go2_config(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('go2')
        pipeline = PerceptionPipeline(odo, cfg)
        assert pipeline.costmap_query is None

    def test_stats_empty_initially(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)
        stats = pipeline.stats
        assert stats["builds"] == 0


class TestPerceptionPipelineCallback:
    def test_single_cloud_produces_costmap(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        # Obstacle at (1.5, 0) with some height
        pts = np.array([
            [1.5, 0.0, 0.3],
            [1.5, 0.1, 0.3],
            [1.5, -0.1, 0.3],
        ], dtype=np.float32)
        msg = _make_dds_msg(pts)
        pipeline.on_point_cloud(msg)

        assert pipeline.costmap_query is not None
        assert pipeline.costmap is not None
        # Near the obstacle should have high cost
        cost = pipeline.costmap_query.sample(1.5, 0.0)
        assert cost > 0.3

    def test_empty_cloud_ignored(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        msg = SimpleNamespace(num_points=0, data=[], timestamp=1.0)
        pipeline.on_point_cloud(msg)

        assert pipeline.costmap_query is None  # Should not update

    def test_multiple_clouds_update_costmap(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        # First cloud: obstacle at (1.0, 0)
        pts1 = np.array([[1.0, 0.0, 0.3]], dtype=np.float32)
        pipeline.on_point_cloud(_make_dds_msg(pts1, timestamp=1.0))
        cost1 = pipeline.costmap_query.sample(1.0, 0.0)

        # Second cloud: obstacle moved to (2.0, 0)
        pts2 = np.array([[2.0, 0.0, 0.3]], dtype=np.float32)
        pipeline.on_point_cloud(_make_dds_msg(pts2, timestamp=2.0))
        cost2_at_new = pipeline.costmap_query.sample(2.0, 0.0)

        # The latest costmap should reflect the second cloud
        assert cost2_at_new > 0.3


class TestPerceptionPipelineStats:
    def test_stats_count_builds(self):
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        pts = np.array([[1.0, 0.0, 0.3]], dtype=np.float32)
        for i in range(5):
            pipeline.on_point_cloud(_make_dds_msg(pts, timestamp=float(i)))

        stats = pipeline.stats
        assert stats["builds"] == 5
        assert "mean_ms" in stats
        assert "max_ms" in stats
        assert "min_ms" in stats
        assert stats["mean_ms"] >= 0
        assert stats["max_ms"] >= stats["min_ms"]

    def test_stats_timing_reasonable(self):
        """Build times should be in reasonable range (< 100ms for simple cloud)."""
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        pts = np.random.randn(100, 3).astype(np.float32)
        pts[:, 2] = np.abs(pts[:, 2])  # Positive Z
        pipeline.on_point_cloud(_make_dds_msg(pts))

        stats = pipeline.stats
        assert stats["max_ms"] < 100  # Should be well under 100ms


class TestPerceptionPipelineThreadSafety:
    def test_concurrent_write_read(self):
        """Writer thread and reader thread should not crash."""
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry()
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        errors = []

        def writer():
            try:
                pts = np.array([[1.0, 0.0, 0.3]], dtype=np.float32)
                for i in range(50):
                    pipeline.on_point_cloud(_make_dds_msg(pts, timestamp=float(i)))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = pipeline.costmap_query
                    _ = pipeline.costmap
                    _ = pipeline.stats
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=10)
        t_read.join(timeout=10)

        assert not errors, f"Thread safety errors: {errors}"


class TestPerceptionPipelineWithMovingRobot:
    def test_costmap_origin_tracks_robot(self):
        """Costmap origin should shift as robot moves."""
        from foreman.demos.target_game.perception import PerceptionPipeline
        odo = FakeOdometry(x=0.0, y=0.0)
        cfg = get_perception_config('b2')
        pipeline = PerceptionPipeline(odo, cfg)

        pts = np.array([[1.0, 0.0, 0.3]], dtype=np.float32)
        pipeline.on_point_cloud(_make_dds_msg(pts))
        origin1 = pipeline.costmap.origin_x

        # Move robot
        odo.pose.x = 5.0
        pipeline.on_point_cloud(_make_dds_msg(pts, timestamp=2.0))
        origin2 = pipeline.costmap.origin_x

        # Origin should have shifted by 5.0m
        assert abs(origin2 - origin1 - 5.0) < 0.01

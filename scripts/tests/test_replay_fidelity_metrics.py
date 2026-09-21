"""
Offline tests for the pure metric functions of
scripts/tools/diagnose_replay_fidelity.py (no CARLA server needed).

Run:  python -m unittest scripts.tests.test_replay_fidelity_metrics -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts.tools.diagnose_replay_fidelity import (  # noqa: E402
    calibration_field_differences,
    depth_error_location,
    depth_error_stats,
    flow_stats,
    frame_sync_row,
    label_world_consistency,
    point_cloud_stats,
    semantic_stats,
    summarize_sync,
)


class DepthTest(unittest.TestCase):

    def test_identical(self):
        depth = np.full((8, 8), 10.0, dtype=np.float32)
        stats, diff = depth_error_stats(depth, depth.copy())
        self.assertEqual(stats["exact_equal_ratio"], 1.0)
        self.assertEqual(stats["max"], 0.0)
        self.assertEqual(diff.sum(), 0.0)

    def test_thresholds_and_percentiles(self):
        a = np.full((10, 10), 20.0, dtype=np.float32)
        b = a.copy()
        b[0, :5] += 0.005    # 5 px < 1 cm
        b[1, :10] += 0.05    # 10 px > 1 cm
        b[2, :2] += 5.0      # 2 px > 1 m
        b[3, :1] += 50.0     # 1 px > 10 m
        stats, _ = depth_error_stats(a, b)
        self.assertAlmostEqual(stats["exact_equal_ratio"], 1 - 18 / 100)
        self.assertAlmostEqual(stats["gt_0.01m_ratio"], 13 / 100)
        self.assertAlmostEqual(stats["gt_1m_ratio"], 3 / 100)
        self.assertAlmostEqual(stats["gt_10m_ratio"], 1 / 100)
        self.assertAlmostEqual(stats["max"], 50.0, places=4)

    def test_error_location_far_edge_interior(self):
        depth = np.full((40, 40), 10.0, dtype=np.float32)
        semantic = np.zeros((40, 40), dtype=np.uint8)
        semantic[:, 20:] = 14                 # vertical boundary at column 20
        depth[:5, :] = 1000.0                 # far-plane band
        replayed = depth.copy()
        replayed[0, 0] += 10.0                # far
        replayed[20, 20] += 10.0              # boundary
        replayed[30, 5] += 10.0               # interior
        _, diff = depth_error_stats(depth, replayed)
        location = depth_error_location(depth, replayed, semantic, diff)["gt_1m"]
        self.assertEqual(location["total"], 3)
        self.assertEqual(location["far_plane"], 1)
        self.assertEqual(location["object_boundary"], 1)
        self.assertEqual(location["interior"], 1)


class SemanticFlowTest(unittest.TestCase):

    def test_semantic(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        self.assertTrue(semantic_stats(a, b)["array_equal"])
        a[0, :2] = 14
        stats = semantic_stats(a, b)
        self.assertFalse(stats["array_equal"])
        self.assertEqual(stats["mismatch_pixels"], 2)
        self.assertEqual(stats["mismatch_by_canonical_class"], {14: 2})

    def test_flow(self):
        a = np.zeros((4, 4, 2), dtype=np.float32)
        b = a.copy()
        b[0, 0] = (3.0, 4.0)                  # EPE 5
        stats = flow_stats(a, b)
        self.assertAlmostEqual(stats["epe_max"], 5.0)
        self.assertEqual(stats["nonfinite_pixels"], 0)
        b[1, 1] = (np.nan, 0.0)
        self.assertEqual(flow_stats(a, b)["nonfinite_pixels"], 1)


class CalibrationTest(unittest.TestCase):

    def test_field_differences(self):
        a = {"sensors": {"lidar": {"T": [[1.0, 0.0], [0.0, 1.0]]}}, "stereo": {"baseline_m": 0.5}}
        b = {"sensors": {"lidar": {"T": [[1.0, 3e-5], [0.0, 1.0]]}}, "stereo": {"baseline_m": 0.5}}
        diffs = dict(calibration_field_differences(a, b))
        self.assertAlmostEqual(diffs["/sensors/lidar/T"], 3e-5)
        self.assertEqual(diffs["/stereo/baseline_m"], 0.0)


class PointCloudTest(unittest.TestCase):

    def test_same_points_different_order_are_equal(self):
        a = np.random.default_rng(0).random((50, 4)).astype(np.float32)
        b = a[::-1].copy()
        stats = point_cloud_stats(a, b)
        self.assertTrue(stats["exact_equal"])
        self.assertEqual(stats["sorted_max_abs_difference"], 0.0)

    def test_count_mismatch_falls_back_to_nearest_neighbour(self):
        a = np.zeros((10, 4), dtype=np.float32)
        a[:, 0] = np.arange(10)
        b = a[:9].copy()
        b[:, 0] += 0.005
        stats = point_cloud_stats(a, b)
        self.assertFalse(stats["count_equal"])
        self.assertLess(stats["nn_mean_m"], 0.2)
        self.assertGreater(stats["within_1cm_ratio"], 0.8)

    def test_empty(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertTrue(point_cloud_stats(empty, empty)["exact_equal"])
        self.assertFalse(point_cloud_stats(empty, np.ones((2, 4)))["count_equal"])


class LabelConsistencyTest(unittest.TestCase):

    def test_matches_and_mismatches(self):
        ego_matrix = np.eye(4)
        ego_matrix[:3, 3] = (100.0, 0.0, 0.0)      # ego at x=100, no rotation
        labels = [
            {"logical_id": 1, "bbox_3d": {"center_ego_m": [10.0, 0.0, 1.0]}},   # world (110,0,1)
            {"logical_id": 2, "bbox_3d": {"center_ego_m": [10.0, 0.0, 1.0]}},   # actor far away
            {"bbox_3d": {"center_ego_m": [1.0, 0.0, 0.0]}},                      # no logical id
        ]
        actors = {"1": (110.0, 0.0, 0.0), "2": (500.0, 0.0, 0.0)}
        checked, mismatched, worst = label_world_consistency(labels, ego_matrix, actors)
        self.assertEqual((checked, mismatched), (3, 2))
        self.assertAlmostEqual(worst, (390.0 ** 2 + 1.0) ** 0.5, places=6)


class SyncTest(unittest.TestCase):

    def rows(self, sensor_offset=0):
        rows = []
        for i in range(3):
            rows.append(frame_sync_row(
                i, {"frame_id": i, "carla_frame": 100 + i}, 500 + i, 500 + i,
                {"rgb_left": 500 + i, "depth": 500 + i + sensor_offset}, 500,
            ))
        return rows

    def test_synchronized(self):
        self.assertTrue(summarize_sync(self.rows())["all_synchronized"])

    def test_detects_sensor_offset(self):
        summary = summarize_sync(self.rows(sensor_offset=1))
        self.assertFalse(summary["all_synchronized"])
        self.assertEqual(summary["sensor_spread"], [1])


if __name__ == "__main__":
    unittest.main()

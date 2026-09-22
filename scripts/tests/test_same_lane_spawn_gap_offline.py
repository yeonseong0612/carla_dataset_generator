"""
scripts/tests/test_same_lane_spawn_gap_offline.py

Offline regression tests (no CARLA server needed) for the same-lane
front spawn gap added in CLAUDE.md PART B:
src.simulation.spawn_policy.same_lane_front_gap_ok(), and its use by
both the initial scene population path (GammaSpawnPolicy, spawn_policy.py)
and the production canonical runtime buffer-replenishment path
(CanonicalBackgroundTraffic, canonical_traffic.py).

same_lane_front_gap_ok() itself is pure (no CARLA object needed), so
most of this file tests it directly. The two "path uses the shared rule"
tests use static inspection (inspect.getsource), matching the existing
style in scripts/tests/test_replay_offline.py
(test_production_replay_function_has_no_non_rgb_path) -- confirming the
production spawn code actually calls the shared helper rather than
re-implementing (or omitting) the same condition, without needing a live
CARLA world to drive an actual spawn attempt through.

Run:  python -m unittest scripts.tests.test_same_lane_spawn_gap_offline -v
"""

import inspect
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

from CFG.config import cfg  # noqa: E402
from src.simulation.spawn_policy import same_lane_front_gap_ok, GammaSpawnPolicy  # noqa: E402
from src.simulation.canonical_traffic import CanonicalBackgroundTraffic  # noqa: E402

GAP_M = 80.0
EGO_ROAD, EGO_LANE = 5, 1


class SameLaneFrontGapRuleTest(unittest.TestCase):
    """
    Pure-function tests for same_lane_front_gap_ok().

    GAP_M was raised 25.0 -> 80.0 by the traffic-generation final-tuning
    task (CLAUDE.md PART A); the rule itself (same lane, strictly ahead,
    0 < relative_s < gap) is unchanged, only the gap value moved.
    """

    def test_same_lane_front_10m_rejected(self):
        self.assertFalse(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 10.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_same_lane_front_25m_rejected(self):
        self.assertFalse(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 25.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_same_lane_front_79_9m_rejected(self):
        self.assertFalse(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 79.9, EGO_ROAD, EGO_LANE, GAP_M))

    def test_same_lane_front_80_0m_accepted(self):
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 80.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_same_lane_front_100m_accepted(self):
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 100.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_adjacent_lane_10m_accepted(self):
        # Same road, different lane_id -- CLAUDE.md B-4/A-2: adjacent lane
        # keeps the existing (unaffected) policy, not this new rule.
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE + 1, 10.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_opposite_lane_10m_accepted(self):
        # Same road, opposite-direction lane_id (sign flipped, CARLA's own
        # convention) -- also just "different lane_id" to this function,
        # so the existing (unaffected) policy applies, not this new rule.
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, -EGO_LANE, 10.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_different_road_10m_accepted(self):
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD + 1, EGO_LANE, 10.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_same_lane_behind_ego_unaffected(self):
        # relative_s < 0 -- behind ego, CLAUDE.md B-4: existing policy applies.
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, -10.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_relative_s_zero_accepted(self):
        # relative_s == 0 is not "> 0 and < gap" -- not rejected by this rule.
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 0.0, EGO_ROAD, EGO_LANE, GAP_M))

    def test_ego_lane_unknown_accepted(self):
        # No ego waypoint projection (ego_road_id is None) -- fail open,
        # same as every other spawn-validation helper in this module.
        self.assertTrue(same_lane_front_gap_ok(EGO_ROAD, EGO_LANE, 5.0, None, None, GAP_M))


class SpawnPathsUseSharedRuleTest(unittest.TestCase):
    """
    Static inspection: both the initial-population and canonical
    runtime-buffer spawn paths call the one shared helper (no duplicated
    inline condition), and the new config parameter exists with the
    documented default.
    """

    def test_initial_spawn_path_uses_shared_rule(self):
        source = inspect.getsource(GammaSpawnPolicy._try_spawn_vehicle_like)
        self.assertIn("same_lane_front_gap_ok", source)

    def test_runtime_canonical_spawn_path_uses_shared_rule(self):
        source = inspect.getsource(CanonicalBackgroundTraffic._try_spawn_buffer_vehicle_like)
        self.assertIn("same_lane_front_gap_ok", source)

    def test_config_parameter_exists_with_recommended_default(self):
        self.assertEqual(cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M, 80.0)


class PedestrianUnaffectedTest(unittest.TestCase):
    """CLAUDE.md B-3: pedestrian spawn candidates never go through this rule."""

    def test_initial_pedestrian_spawn_does_not_reference_gap_rule(self):
        source = inspect.getsource(GammaSpawnPolicy._try_spawn_pedestrian)
        self.assertNotIn("same_lane_front_gap_ok", source)

    def test_runtime_pedestrian_spawn_does_not_reference_gap_rule(self):
        source = inspect.getsource(CanonicalBackgroundTraffic._try_spawn_buffer_pedestrian)
        self.assertNotIn("same_lane_front_gap_ok", source)


if __name__ == "__main__":
    unittest.main()

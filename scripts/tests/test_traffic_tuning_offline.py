"""
scripts/tests/test_traffic_tuning_offline.py

Offline regression tests (no CARLA server needed) for the traffic-
generation final-tuning task in CLAUDE.md:

  PART B: pre-recording traffic warm-up
          (cfg.SPAWN.PRE_RECORD_WARMUP_TICKS,
          scripts/collect_dataset.py generate_canonical_geometry())
  PART C: traffic light cycle shortening
          (cfg.TRAFFIC_LIGHT.{GREEN,YELLOW,RED}_TIME_S,
          scripts/collect_dataset.py configure_traffic_lights())

PART A (raising cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M 25.0 -> 80.0) is
covered by the updated scripts/tests/test_same_lane_spawn_gap_offline.py,
not here.

Neither warm-up nor traffic-light-timing logic needs a live CARLA world:
warm-up is checked via static source inspection (matching the existing
style in test_replay_offline.py / test_same_lane_spawn_gap_offline.py),
and configure_traffic_lights() is exercised against small fake
world/actor/light objects exposing only the CARLA calls it actually
makes (get_actors().filter(...), set_green_time/set_yellow_time/
set_red_time) -- no real `carla` module behaviour is needed for either.

Run:  python -m unittest scripts.tests.test_traffic_tuning_offline -v
"""

import inspect
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

from CFG.config import cfg  # noqa: E402
import scripts.collect_dataset as collect_dataset  # noqa: E402


# ================================================================
# PART B: pre-recording traffic warm-up
# ================================================================

class PreRecordWarmupConfigTest(unittest.TestCase):

    def test_configured_warmup_ticks_default(self):
        self.assertEqual(cfg.SPAWN.PRE_RECORD_WARMUP_TICKS, 20)

    def test_twenty_ticks_is_one_simulation_second(self):
        # CLAUDE.md B-1: 20 ticks @ fixed_delta_seconds must equal 1.0s --
        # this is the actual invariant the "20" default depends on, not
        # just the literal value.
        self.assertAlmostEqual(
            cfg.SPAWN.PRE_RECORD_WARMUP_TICKS * cfg.SIMULATION.FIXED_DELTA_SECONDS,
            1.0,
        )


class PreRecordWarmupPlacementTest(unittest.TestCase):
    """
    Static inspection of generate_canonical_geometry(): the configured
    warm-up runs strictly before the frame-recording loop, and no
    dataset-writing call happens ahead of that loop.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = inspect.getsource(collect_dataset.generate_canonical_geometry)

    def test_uses_configured_warmup_ticks(self):
        self.assertIn("cfg.SPAWN.PRE_RECORD_WARMUP_TICKS", self.source)

    def test_warmup_runs_before_the_recording_loop(self):
        warmup_index = self.source.index("cfg.SPAWN.PRE_RECORD_WARMUP_TICKS")
        loop_index = self.source.index("for simulation_tick_idx in range(")
        self.assertLess(warmup_index, loop_index)

    def test_no_dataset_write_call_precedes_the_recording_loop(self):
        loop_index = self.source.index("for simulation_tick_idx in range(")
        prefix = self.source[:loop_index]

        for forbidden in (
            "collector.save_frame(",
            "metadata.write_frame(",
            "world_state_recorder.record_frame(",
            ".write_frame(",  # annotation_writer.write_frame(...)
        ):
            self.assertNotIn(forbidden, prefix, f"{forbidden!r} must not run before frame_id=0")

    def test_frame_recording_loop_starts_at_zero(self):
        # range(max_frames) with no explicit start -- frame_id starts at 0
        # right after warm-up, never offset by the warm-up tick count.
        self.assertIn("for simulation_tick_idx in range(\n            max_simulation_ticks\n        ):", self.source)

    def test_ego_is_not_driven_during_warmup(self):
        # Between the warmup line and the recording loop, ego must not be
        # controlled/moved (CLAUDE.md B-3: no forced move + teleport-back).
        warmup_index = self.source.index("cfg.SPAWN.PRE_RECORD_WARMUP_TICKS")
        loop_index = self.source.index("for simulation_tick_idx in range(")
        between = self.source[warmup_index:loop_index]

        self.assertNotIn("ego.apply_control(", between)
        self.assertNotIn("route_controller\n", between)
        self.assertNotIn(".set_transform(", between)

    def test_sensor_queue_is_flushed_after_warmup(self):
        warmup_index = self.source.index("cfg.SPAWN.PRE_RECORD_WARMUP_TICKS")
        loop_index = self.source.index("for simulation_tick_idx in range(")
        between = self.source[warmup_index:loop_index]

        self.assertIn("clear_queues", between)


# ================================================================
# PART C: traffic light cycle shortening
# ================================================================

class TrafficLightConfigTest(unittest.TestCase):

    def test_default_durations(self):
        self.assertEqual(cfg.TRAFFIC_LIGHT.GREEN_TIME_S, 8.0)
        self.assertEqual(cfg.TRAFFIC_LIGHT.YELLOW_TIME_S, 2.0)
        self.assertEqual(cfg.TRAFFIC_LIGHT.RED_TIME_S, 1.0)


class FakeTrafficLight:
    def __init__(self, light_id, fail=False):
        self.id = light_id
        self._fail = fail
        self.green_time = None
        self.yellow_time = None
        self.red_time = None

    def set_green_time(self, value):
        if self._fail:
            raise RuntimeError("actor not alive")
        self.green_time = value

    def set_yellow_time(self, value):
        self.yellow_time = value

    def set_red_time(self, value):
        self.red_time = value


class FakeActorList(list):
    def __init__(self, actors):
        super().__init__(actors)
        self.filter_calls = []

    def filter(self, pattern):
        self.filter_calls.append(pattern)
        return self


class FakeWorld:
    def __init__(self, lights):
        self._actor_list = FakeActorList(lights)

    def get_actors(self):
        return self._actor_list


class ConfigureTrafficLightsTest(unittest.TestCase):
    """
    Exercises the real configure_traffic_lights() against fake CARLA
    objects exposing only get_actors().filter(...) and the three
    set_*_time methods it actually calls.
    """

    def test_filters_for_traffic_lights(self):
        world = FakeWorld([])
        collect_dataset.configure_traffic_lights(world, cfg)
        self.assertEqual(len(world._actor_list.filter_calls), 1)
        self.assertIn("traffic_light", world._actor_list.filter_calls[0])

    def test_applies_configured_durations_to_every_light(self):
        lights = [FakeTrafficLight(1), FakeTrafficLight(2), FakeTrafficLight(3)]
        world = FakeWorld(lights)

        result = collect_dataset.configure_traffic_lights(world, cfg)

        for light in lights:
            self.assertEqual(light.green_time, cfg.TRAFFIC_LIGHT.GREEN_TIME_S)
            self.assertEqual(light.yellow_time, cfg.TRAFFIC_LIGHT.YELLOW_TIME_S)
            self.assertEqual(light.red_time, cfg.TRAFFIC_LIGHT.RED_TIME_S)

        self.assertEqual(result["configured"], 3)
        self.assertEqual(result["failed"], 0)

    def test_a_failing_light_is_counted_and_reported_not_silently_ignored(self):
        lights = [FakeTrafficLight(1), FakeTrafficLight(2, fail=True), FakeTrafficLight(3)]
        world = FakeWorld(lights)

        buffer = StringIO()

        with redirect_stdout(buffer):
            result = collect_dataset.configure_traffic_lights(world, cfg)

        self.assertEqual(result["configured"], 2)
        self.assertEqual(result["failed"], 1)

        printed = buffer.getvalue()
        self.assertIn("failed to configure 1 lights", printed)
        self.assertIn("id=2", printed)

    def test_logs_configured_count_and_durations_once(self):
        lights = [FakeTrafficLight(1), FakeTrafficLight(2)]
        world = FakeWorld(lights)

        buffer = StringIO()

        with redirect_stdout(buffer):
            collect_dataset.configure_traffic_lights(world, cfg)

        printed = buffer.getvalue()
        self.assertEqual(printed.count("[TrafficLight] configured"), 1)
        self.assertIn("green=8.0s", printed)
        self.assertIn("yellow=2.0s", printed)
        self.assertIn("red=1.0s", printed)


class TrafficLightGroupStateUntouchedTest(unittest.TestCase):
    """CLAUDE.md C-3: only durations change, never group/state itself."""

    def test_configure_traffic_lights_does_not_touch_state(self):
        source = inspect.getsource(collect_dataset.configure_traffic_lights)

        for forbidden in ("set_state", "freeze", "set_green_time_all", "TrafficLightState.Green"):
            self.assertNotIn(forbidden, source)

    def test_called_exactly_once_per_town_load_in_main(self):
        source = inspect.getsource(collect_dataset.main)
        self.assertEqual(source.count("configure_traffic_lights(world, cfg)"), 1)


# ================================================================
# Traffic-light ignore driving policy (canonical geometry only)
# ================================================================

class FakeTrafficManager:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def record(actor, value):
            self.calls[name] = value
        return record


class FakeAutopilotActor:
    def __init__(self):
        self.autopilot = None

    def set_autopilot(self, enabled, port):
        self.autopilot = (enabled, port)


class TrafficLightIgnorePolicyTest(unittest.TestCase):

    def test_production_defaults_ignore_lights(self):
        self.assertIs(cfg.TRAFFIC.NPC_IGNORE_TRAFFIC_LIGHTS, True)
        self.assertIs(cfg.TRAFFIC.EGO_IGNORE_TRAFFIC_LIGHTS, True)
        self.assertEqual(collect_dataset.TRAFFIC_LIGHT_POLICY, "ignore")

    def test_other_traffic_rules_unchanged(self):
        # Scope is traffic lights only: vehicle/walker avoidance and
        # stop signs stay at their existing values.
        self.assertEqual(cfg.TRAFFIC.IGNORE_VEHICLES_PERCENTAGE, 0.0)
        self.assertEqual(cfg.TRAFFIC.IGNORE_WALKERS_PERCENTAGE, 0.0)
        self.assertEqual(cfg.TRAFFIC.IGNORE_SIGNS_PERCENTAGE, 0.0)

    def test_npc_actor_ignores_lights_but_not_vehicles(self):
        from src.simulation.traffic import configure_actor_traffic_manager

        tm = FakeTrafficManager()
        actor = FakeAutopilotActor()
        configure_actor_traffic_manager(actor, tm, cfg)

        self.assertEqual(tm.calls["ignore_lights_percentage"], 100.0)
        self.assertEqual(tm.calls["ignore_vehicles_percentage"], 0.0)
        self.assertEqual(tm.calls["ignore_walkers_percentage"], 0.0)
        self.assertEqual(tm.calls["ignore_signs_percentage"], 0.0)

    def test_npc_flag_off_falls_back_to_legacy_percentage(self):
        from easydict import EasyDict
        from src.simulation.traffic import get_npc_ignore_lights_percentage

        local = EasyDict({"TRAFFIC": {"NPC_IGNORE_TRAFFIC_LIGHTS": False, "IGNORE_LIGHTS_PERCENTAGE": 0.0}})
        self.assertEqual(get_npc_ignore_lights_percentage(local), 0.0)

    def test_ego_route_controller_calls_basic_agent_ignore(self):
        import types
        from unittest import mock
        import src.navigation.controller as controller_module

        calls = []

        class RecordingAgent:
            def __init__(self, vehicle, target_speed=None):
                pass

            def get_local_planner(self):
                return types.SimpleNamespace(set_global_plan=lambda *a, **k: None)

            def ignore_traffic_lights(self, active=True):
                calls.append(active)

        route = [(types.SimpleNamespace(index=i), "LANEFOLLOW") for i in range(3)]

        with mock.patch.object(controller_module, "BasicAgent", RecordingAgent):
            controller_module.RouteController(
                object(), route, traffic_light_policy=collect_dataset.TRAFFIC_LIGHT_POLICY,
            )

        self.assertEqual(calls, [True])

    def test_installed_basic_agent_supports_ignore_traffic_lights(self):
        # The real CARLA agent (not a guess): the method exists and only
        # gates the traffic-light hazard, never the vehicle-obstacle check.
        from agents.navigation.basic_agent import BasicAgent

        self.assertTrue(callable(getattr(BasicAgent, "ignore_traffic_lights", None)))
        tl_source = inspect.getsource(BasicAgent._affected_by_traffic_light)
        veh_source = inspect.getsource(BasicAgent._vehicle_obstacle_detected)
        self.assertIn("self._ignore_traffic_lights", tl_source)
        self.assertNotIn("_ignore_traffic_lights", veh_source)

    def test_policy_does_not_touch_signal_state(self):
        source = inspect.getsource(collect_dataset.log_traffic_light_policy)

        for forbidden in ("set_state", "freeze", "TrafficLightState", "ignore_vehicles("):
            self.assertNotIn(forbidden, source)

    def test_policy_logged_once_at_canonical_init_not_in_replay(self):
        canonical = inspect.getsource(collect_dataset.generate_canonical_geometry)
        replay = inspect.getsource(collect_dataset.replay_condition)

        self.assertEqual(canonical.count("log_traffic_light_policy("), 1)
        self.assertLess(
            canonical.index("log_traffic_light_policy("),
            canonical.index("for simulation_tick_idx in range("),
        )
        for needle in ("log_traffic_light_policy(", "RouteController(", "traffic_manager", "set_autopilot"):
            self.assertNotIn(needle, replay)


if __name__ == "__main__":
    unittest.main()

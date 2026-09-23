"""
scripts/tests/test_stall_debug_offline.py

Offline tests (no CARLA server) for the DEBUG-ONLY stall-diagnosis
additions (CLAUDE.md Town10 Route 1 ~50% stall task):

    src/navigation/debug_start.py          --debug-start-route-index
    src/simulation/stall_diagnostics.py    [STALL-DIAG] / [STUCK-STATE]
    RouteController(start_route_index=...) + last_stuck_reset_reason
    GammaSpawnPolicy(route_s_offset=...)

RouteController is built with BasicAgent replaced by a recording fake,
so the real update_stuck_state() / _set_global_plan() code runs.

Run:  python -m unittest scripts.tests.test_stall_debug_offline -v
"""

import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

import scripts.collect_dataset as collect_dataset  # noqa: E402
import src.navigation.controller as controller_module  # noqa: E402
from src.navigation.debug_start import (  # noqa: E402
    route_arc_length_at,
    select_route_start,
    validate_debug_start_route_index,
)
from src.simulation import spawn_policy  # noqa: E402
from src.simulation.stall_diagnostics import (  # noqa: E402
    StallMonitor,
    StuckStateLogger,
    classify_stop,
    extract_stuck_state,
    find_nearest_forward_actor,
    format_stuck_state,
    is_stall_condition,
)


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------

def make_route(n, spacing=2.0):
    """Straight dense route along +x: [(waypoint, road_option), ...]."""

    route = []

    for i in range(n):
        location = types.SimpleNamespace(x=i * spacing, y=0.0, z=0.0)
        rotation = types.SimpleNamespace(yaw=0.0)
        waypoint = types.SimpleNamespace(
            transform=types.SimpleNamespace(location=location, rotation=rotation),
            road_id=1, lane_id=-1, index=i,
        )
        route.append((waypoint, "LANEFOLLOW"))

    return route


class FakeLocalPlanner:
    def __init__(self):
        self.plan = None

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        self.plan = plan


class FakeAgent:
    def __init__(self, vehicle, target_speed=None):
        self.local_planner = FakeLocalPlanner()

    def get_local_planner(self):
        return self.local_planner

    def ignore_traffic_lights(self, active=False):
        pass


class FakeVehicle:
    def __init__(self, speed_mps=0.0, at_light=False):
        self.speed_mps = speed_mps
        self.at_light = at_light

    def get_velocity(self):
        return types.SimpleNamespace(x=self.speed_mps, y=0.0, z=0.0)

    def is_at_traffic_light(self):
        return self.at_light

    def get_traffic_light(self):
        return None


def make_controller(route, **kwargs):
    with mock.patch.object(controller_module, "BasicAgent", FakeAgent):
        return controller_module.RouteController(FakeVehicle(), route, **kwargs)


def control(throttle=0.0, brake=0.0):
    return types.SimpleNamespace(throttle=throttle, brake=brake, steer=0.0, hand_brake=False)


# ------------------------------------------------------------------
# 1-4: debug mid-route start
# ------------------------------------------------------------------

class DebugStartTest(unittest.TestCase):

    def setUp(self):
        self.route = make_route(495)

    # 1. no flag -> production start unchanged
    def test_no_flag_selects_route_start(self):
        waypoint, index = select_route_start(self.route, None)
        self.assertIs(waypoint, self.route[0][0])
        self.assertEqual(index, 0)

    def test_no_flag_controller_plan_is_full_route_object(self):
        ctrl = make_controller(self.route)
        self.assertEqual(ctrl.route_index, 0)
        self.assertIs(ctrl.agent.local_planner.plan, self.route)

    def test_production_spawn_call_has_no_start_index(self):
        source = inspect.getsource(collect_dataset.generate_canonical_geometry)
        self.assertIn(
            "        if debug_start_route_index is None:\n\n"
            "            ego = spawn_ego_at_route_start(\n"
            "                world,\n"
            "                dense_route,\n"
            "            )",
            source,
        )
        spawn_source = inspect.getsource(collect_dataset.spawn_ego_at_route_start)
        self.assertIn("select_route_start(", spawn_source)

    def test_debug_flag_defaults_off(self):
        with mock.patch.object(sys, "argv", ["collect_dataset.py"]):
            args = collect_dataset.parse_args()
        self.assertIsNone(args.debug_start_route_index)
        self.assertFalse(args.debug_stall_diagnostics)

    def test_gamma_offset_zero_is_original_reference_path(self):
        policy = object.__new__(spawn_policy.GammaSpawnPolicy)
        policy.dense_route = self.route
        policy.route_s_offset = 0.0

        for s in (10.0, 37.3, 99.9):
            self.assertEqual(
                policy._reference_at(s),
                spawn_policy.reference_waypoint_at_distance(self.route, s),
            )

    # 2. valid index -> selected waypoint
    def test_valid_index_returns_waypoint(self):
        waypoint, index = select_route_start(self.route, 220)
        self.assertIs(waypoint, self.route[220][0])
        self.assertEqual(index, 220)

    def test_valid_index_controller_continues_from_there(self):
        ctrl = make_controller(self.route, start_route_index=206)
        self.assertEqual(ctrl.route_index, 206)
        plan = ctrl.agent.local_planner.plan
        self.assertIs(plan[0][0], self.route[206][0])
        self.assertIs(plan[-1][0], self.route[-1][0])

    def test_gamma_offset_places_traffic_relative_to_ego(self):
        policy = object.__new__(spawn_policy.GammaSpawnPolicy)
        policy.dense_route = self.route
        policy.route_s_offset = route_arc_length_at(self.route, 206)  # 412 m

        waypoint, relative = policy._reference_at(30.0)
        self.assertAlmostEqual(waypoint.transform.location.x - 412.0, relative, places=6)
        self.assertGreaterEqual(relative, 30.0)
        self.assertLess(relative, 30.0 + 2.0 + 1e-9)

    def test_arc_length(self):
        self.assertAlmostEqual(route_arc_length_at(self.route, 248), 496.0)
        self.assertEqual(route_arc_length_at(self.route, 0), 0.0)

    # 3. negative -> reject
    def test_negative_index_rejected(self):
        with self.assertRaises(ValueError):
            validate_debug_start_route_index(-1, len(self.route))
        with self.assertRaises(ValueError):
            select_route_start(self.route, -5)
        with self.assertRaises(ValueError):
            make_controller(self.route, start_route_index=-1)

    # 4. index >= len(route) -> reject (last waypoint too: nothing to drive)
    def test_out_of_range_index_rejected(self):
        for index in (len(self.route), len(self.route) + 10, len(self.route) - 1):
            with self.assertRaises(ValueError):
                validate_debug_start_route_index(index, len(self.route))
        with self.assertRaises(ValueError):
            make_controller(self.route, start_route_index=len(self.route))

    def test_non_int_rejected(self):
        for value in (2.5, "10", True):
            with self.assertRaises(ValueError):
                validate_debug_start_route_index(value, len(self.route))

    def test_debug_start_refuses_production_output_root(self):
        args = types.SimpleNamespace(towns=["Town10"], routes=["1"], debug_start_route_index=206)
        with self.assertRaises(SystemExit):
            collect_dataset.check_debug_start_args(
                args, collect_dataset.os.path.abspath(collect_dataset.PRODUCTION_OUTPUT_ROOT),
            )

    def test_debug_start_requires_single_route(self):
        args = types.SimpleNamespace(towns=["Town10"], routes=["1", "2"], debug_start_route_index=206)
        with self.assertRaises(SystemExit):
            collect_dataset.check_debug_start_args(args, r"D:\carla_stall_fasttest")


# ------------------------------------------------------------------
# 5-7: stall condition
# ------------------------------------------------------------------

class StallConditionTest(unittest.TestCase):

    # 5
    def test_stopped_with_target_and_no_red_is_stall(self):
        self.assertTrue(is_stall_condition(0.0, 30.0, False))
        self.assertEqual(classify_stop(0.0, 30.0, False), "stall")

    # 6
    def test_red_light_is_expected_stop(self):
        self.assertFalse(is_stall_condition(0.0, 30.0, True))
        self.assertEqual(classify_stop(0.0, 30.0, True), "expected_red_light")

    # 7
    def test_target_zero_is_not_stall(self):
        self.assertFalse(is_stall_condition(0.0, 0.0, False))
        self.assertEqual(classify_stop(0.0, 0.0, False), "target_zero")

    def test_moving_is_not_stall(self):
        self.assertFalse(is_stall_condition(25.0, 30.0, False))

    def test_monitor_fires_after_5_simulated_seconds(self):
        monitor = StallMonitor()
        dt = 0.05  # 20 Hz
        fired = [tick for tick in range(400) if monitor.update(100.0 + tick * dt, 0.0, 30.0, False)]
        # 5 s -> tick 100, then every 10 s (200 ticks).
        self.assertEqual(fired, [100, 300])

    def test_monitor_catches_stop_and_go_creep(self):
        # Observed live: 0 <-> ~2 km/h creep behind a queue.
        monitor = StallMonitor()
        fired = [
            tick for tick in range(200)
            if monitor.update(tick * 0.05, 0.0 if (tick // 10) % 2 == 0 else 2.0, 30.0, False)
        ]
        self.assertEqual(fired, [100])

    def test_creep_released_by_red_light_or_real_motion(self):
        monitor = StallMonitor()
        monitor.update(0.0, 0.0, 30.0, False)
        monitor.update(1.0, 2.0, 30.0, True)
        self.assertIsNone(monitor.stall_start_time)
        monitor.update(2.0, 0.0, 30.0, False)
        monitor.update(3.0, 6.0, 30.0, False)
        self.assertIsNone(monitor.stall_start_time)

    def test_monitor_resets_when_moving(self):
        monitor = StallMonitor()
        for tick in range(90):
            self.assertFalse(monitor.update(tick * 0.05, 0.0, 30.0, False))
        self.assertFalse(monitor.update(4.5, 10.0, 30.0, False))
        self.assertIsNone(monitor.stall_start_time)


# ------------------------------------------------------------------
# 8-9: nearest forward actor
# ------------------------------------------------------------------

class ForwardActorTest(unittest.TestCase):

    EGO = (0.0, 0.0)

    # 8
    def test_behind_actor_excluded(self):
        hit = find_nearest_forward_actor(self.EGO, 0.0, [("behind", -3.0, 0.0)])
        self.assertIsNone(hit)

        hit = find_nearest_forward_actor(self.EGO, 0.0, [("behind", -3.0, 0.0), ("front", 12.0, 0.2)])
        self.assertEqual(hit["key"], "front")

    # 9
    def test_forward_closest_selected(self):
        candidates = [("far", 25.0, 0.0), ("near", 7.5, 0.3), ("mid", 15.0, -0.5)]
        hit = find_nearest_forward_actor(self.EGO, 0.0, candidates)
        self.assertEqual(hit["key"], "near")
        self.assertAlmostEqual(hit["longitudinal_m"], 7.5)

    def test_adjacent_lane_excluded_by_corridor_but_not_half_plane(self):
        candidates = [("adjacent", 5.0, 3.5), ("own_lane", 20.0, 0.0)]
        self.assertEqual(find_nearest_forward_actor(self.EGO, 0.0, candidates)["key"], "own_lane")
        self.assertEqual(
            find_nearest_forward_actor(self.EGO, 0.0, candidates, max_lateral_m=None)["key"], "adjacent",
        )

    def test_heading_is_respected(self):
        # Ego facing west (yaw 180): an actor at x=-10 is ahead, x=+10 behind.
        candidates = [("east", 10.0, 0.0), ("west", -10.0, 0.0)]
        self.assertEqual(find_nearest_forward_actor(self.EGO, 180.0, candidates)["key"], "west")


# ------------------------------------------------------------------
# 10: stuck-detector introspection (real RouteController logic)
# ------------------------------------------------------------------

class StuckStateTest(unittest.TestCase):

    def setUp(self):
        self.ctrl = make_controller(make_route(50))

    def test_basic_agent_emergency_brake_resets_stuck_timer(self):
        # BasicAgent.add_emergency_stop(): throttle=0, brake=max_brake=0.5
        self.ctrl.update_stuck_state(control(throttle=0.0, brake=0.5))
        state = extract_stuck_state(self.ctrl, now_monotonic=1000.0)
        self.assertFalse(state["timer_active"])
        self.assertFalse(state["is_stuck"])
        self.assertEqual(state["reset_reason"], "brake>0.1")
        self.assertEqual(state["clock"], "wall(time.monotonic)")

    def test_throttle_while_stopped_starts_wall_clock_timer(self):
        with mock.patch.object(controller_module.time, "monotonic", return_value=500.0):
            self.ctrl.update_stuck_state(control(throttle=0.8))
        state = extract_stuck_state(self.ctrl, now_monotonic=512.5)
        self.assertTrue(state["timer_active"])
        self.assertAlmostEqual(state["elapsed_wall_s"], 12.5)
        self.assertIsNone(state["reset_reason"])
        self.assertEqual(state["threshold_kmh"], 1.0)
        self.assertEqual(state["timeout_s"], 30.0)

    def test_low_throttle_is_not_trying_to_move(self):
        self.ctrl.update_stuck_state(control(throttle=0.1))
        self.assertEqual(extract_stuck_state(self.ctrl, 0.0)["reset_reason"], "not_trying_to_move(throttle<=0.2)")

    def test_moving_resets_with_speed_reason(self):
        self.ctrl.vehicle.speed_mps = 5.0
        self.ctrl.update_stuck_state(control(throttle=0.8))
        self.assertEqual(extract_stuck_state(self.ctrl, 0.0)["reset_reason"], "speed>=1")

    def test_stuck_after_timeout(self):
        with mock.patch.object(controller_module.time, "monotonic", return_value=0.0):
            self.ctrl.update_stuck_state(control(throttle=0.8))
        with mock.patch.object(controller_module.time, "monotonic", return_value=31.0):
            self.ctrl.update_stuck_state(control(throttle=0.8))
        self.assertTrue(extract_stuck_state(self.ctrl, 31.0)["is_stuck"])

    def test_logger_prints_every_second_or_on_change(self):
        logger = StuckStateLogger()
        idle = {"timer_active": False, "is_stuck": False}
        active = {"timer_active": True, "is_stuck": False}
        printed = [tick for tick in range(41) if logger.should_print(tick * 0.05, idle)]
        self.assertEqual(printed, [0, 20, 40])
        self.assertTrue(logger.should_print(2.05, active))  # state change -> immediate
        self.assertFalse(logger.should_print(2.10, active))
        self.assertAlmostEqual(logger.elapsed_sim_s(3.05), 1.0)

    def test_format_mentions_clock(self):
        state = extract_stuck_state(self.ctrl, 0.0)
        line = format_stuck_state(state, 0.0, 0.0)
        self.assertTrue(line.startswith("[STUCK-STATE]"))
        self.assertIn("clock=wall(time.monotonic)", line)


if __name__ == "__main__":
    unittest.main()

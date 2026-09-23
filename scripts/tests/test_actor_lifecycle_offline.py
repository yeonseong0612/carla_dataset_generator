"""
test_actor_lifecycle_offline.py

Offline (no CARLA server, no carla.Client) tests of the stale background
actor policy introduced after the production failure

    [CANONICAL GEOMETRY FAILED] Town10 route 1
    Responding error from function get_vehicle_light_state:
    Actor could not be found in the registry. Actor Id: 523

plus the canonical-restart / weather-reuse resume semantics around it.
Uses the real `carla` Python value types (Transform, Vector3D, ...) but only
fake actors / worlds.

Run:
    python -m unittest scripts.tests.test_actor_lifecycle_offline -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import carla  # noqa: E402

# First: puts CARLA's PythonAPI (the `agents` package) on sys.path before
# src.simulation.canonical_traffic imports it.
import scripts.collect_dataset as cd  # noqa: E402
from src.data.annotation import AnnotationWriter  # noqa: E402
from src.data.layout import condition_dir, geometry_dir, is_complete, mark_complete  # noqa: E402
from src.data.resume_plan import plan_route_work  # noqa: E402
from src.data.world_state import WorldStateReader, WorldStateRecorder  # noqa: E402
from src.simulation.actor_lifecycle import (  # noqa: E402
    EgoActorLostError,
    is_stale_actor_error,
)

# Exact text of the production failure (server-side ActorNotFound).
REGISTRY_ERROR = (
    "Responding error from function get_vehicle_light_state: "
    "Actor could not be found in the registry. Actor Id: 523"
)
# LibCarla client-side message for a wrapper already destroy()-ed.
DESTROYED_WRAPPER_ERROR = (
    "trying to operate on a destroyed actor; an actor's function was called, "
    "but the actor is already destroyed."
)


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------

class FakeActor:
    """
    fail_on: {method_name: RuntimeError} -- that getter raises when called
    (simulates the actor leaving the registry between two RPCs).
    """

    def __init__(self, actor_id, type_id, x=10.0, y=0.0, attributes=None, fail_on=None):
        self.id = actor_id
        self.type_id = type_id
        self.attributes = dict(attributes or {})
        self.semantic_tags = []
        self.is_alive = True
        self.transform = carla.Transform(carla.Location(x=x, y=y, z=0.0))
        self.velocity = carla.Vector3D(1.0, 0.0, 0.0)
        self.bounding_box = carla.BoundingBox(carla.Location(0, 0, 0.8), carla.Vector3D(2.0, 1.0, 0.8))
        self.fail_on = dict(fail_on or {})
        self.calls = []

    def _call(self, name, value):
        self.calls.append(name)
        if name in self.fail_on:
            raise self.fail_on[name]
        return value

    def get_transform(self):
        return self._call("get_transform", self.transform)

    def get_location(self):
        return self._call("get_location", self.transform.location)

    def get_velocity(self):
        return self._call("get_velocity", self.velocity)

    def get_angular_velocity(self):
        return self._call("get_angular_velocity", carla.Vector3D(0.0, 0.0, 0.0))

    def get_light_state(self):
        return self._call("get_light_state", carla.VehicleLightState(carla.VehicleLightState.Brake))


class FakeActorSnapshot:
    def __init__(self, actor):
        self.actor = actor

    def get_transform(self):
        return self.actor.transform

    def get_velocity(self):
        return self.actor.velocity

    def get_angular_velocity(self):
        return carla.Vector3D(0.0, 0.0, 0.0)


class FakeSnapshot:
    def __init__(self, actors, frame=1000, elapsed=50.0):
        self._actors = {actor.id: FakeActorSnapshot(actor) for actor in actors}
        self.frame = frame
        self.timestamp = SimpleNamespace(elapsed_seconds=elapsed)

    def find(self, actor_id):
        return self._actors.get(actor_id)


class FakeWorld:
    def __init__(self, actors):
        self.actors = list(actors)

    def get_actors(self):
        return list(self.actors)

    def get_snapshot(self):
        return FakeSnapshot(self.actors)


def annotation_cfg():
    return SimpleNamespace(ANNOTATION=SimpleNamespace(MAX_DISTANCE=100.0))


def managed(actor, category="vehicle"):
    return {"actor": actor, "category": category}


# ------------------------------------------------------------------
# Stale-error classification
# ------------------------------------------------------------------

class StaleErrorClassificationTest(unittest.TestCase):

    def test_registry_and_destroyed_wrapper_messages_are_stale(self):
        self.assertTrue(is_stale_actor_error(RuntimeError(REGISTRY_ERROR)))
        self.assertTrue(is_stale_actor_error(RuntimeError(DESTROYED_WRAPPER_ERROR)))

    def test_other_errors_are_not_stale(self):
        self.assertFalse(is_stale_actor_error(RuntimeError("some unrelated failure")))
        self.assertFalse(is_stale_actor_error(RuntimeError("time-out of 10000ms while waiting for the simulator")))
        self.assertFalse(is_stale_actor_error(ValueError(REGISTRY_ERROR)))  # only RuntimeError
        self.assertFalse(is_stale_actor_error(EgoActorLostError("Ego actor 1 is no longer alive")))


# ------------------------------------------------------------------
# Tests 1-6: world_state serializer
# ------------------------------------------------------------------

class WorldStateStaleActorTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.recorder = WorldStateRecorder(os.path.join(self.root, "geometry"))
        self.ego = FakeActor(1, "vehicle.tesla.model3", x=0.0)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def record(self, actors, frame_id=0):
        snapshot = FakeSnapshot([self.ego] + [a for a in actors if a.is_alive])
        return self.recorder.record_frame(frame_id, 1000 + 2 * frame_id, 0.1 * frame_id,
                                          snapshot, self.ego, [managed(a) for a in actors])

    def test_1_normal_background_actor_is_recorded(self):
        car = FakeActor(10, "vehicle.audi.a2")
        state = self.record([car])

        self.assertEqual(list(state["actors"]), ["0"])
        record = state["actors"]["0"]
        self.assertEqual(set(record), {"transform", "velocity", "angular_velocity", "light_state"})
        self.assertEqual(record["light_state"], int(carla.VehicleLightState.Brake))
        self.assertNotIn("stale_actor_omissions", state)  # ordinary frames keep the old schema

    def test_2_actor_dead_before_snapshot_is_omitted(self):
        car = FakeActor(10, "vehicle.audi.a2")
        car.is_alive = False
        state = self.record([car])

        self.assertEqual(state["actors"], {})
        self.assertNotIn("get_light_state", car.calls)
        self.assertNotIn("stale_actor_omissions", state)  # not in the snapshot -> not a race

    def test_3_actor_dies_after_is_alive_check_whole_record_omitted(self):
        dying = FakeActor(10, "vehicle.audi.a2", fail_on={"get_light_state": RuntimeError(REGISTRY_ERROR)})
        other = FakeActor(11, "vehicle.bmw.grandtourer", x=20.0)
        state = self.record([dying, other])

        # The frame itself succeeded; only the dying actor is gone, with no
        # partial (transform-only) record left behind for it.
        self.assertEqual(len(state["actors"]), 1)
        (logical_id, record), = state["actors"].items()
        self.assertEqual(self.recorder.registry[int(logical_id)]["canonical_actor_id"], 11)
        self.assertIn("light_state", record)
        self.assertEqual(state["stale_actor_omissions"], [10])
        self.assertIsNone(self.recorder.logical_id_for(10))  # never registered

        with open(os.path.join(self.root, "geometry", "world_state", "000000.json")) as f:
            self.assertEqual(json.load(f), state)

    def test_4_first_getter_raising_actor_not_found_omits_actor(self):
        # Pedestrian: no light-state RPC; snapshot getter raising must be
        # handled identically.
        walker = FakeActor(12, "walker.pedestrian.0001")
        snapshot = FakeSnapshot([self.ego, walker])
        snapshot._actors[12].get_transform = mock.Mock(side_effect=RuntimeError(DESTROYED_WRAPPER_ERROR))

        state = self.recorder.record_frame(0, 1000, 0.0, snapshot, self.ego, [managed(walker, "pedestrian")])
        self.assertEqual(state["actors"], {})
        self.assertEqual(state["stale_actor_omissions"], [12])

    def test_5_unrelated_runtime_error_propagates(self):
        car = FakeActor(10, "vehicle.audi.a2", fail_on={"get_light_state": RuntimeError("some unrelated failure")})

        with self.assertRaisesRegex(RuntimeError, "some unrelated failure"):
            self.record([car])

        self.assertFalse(os.path.exists(os.path.join(self.root, "geometry", "world_state", "000000.json")))

    def test_6_ego_stale_is_fatal(self):
        car = FakeActor(10, "vehicle.audi.a2")

        self.ego.is_alive = False
        with self.assertRaises(EgoActorLostError):
            self.record([car])

        # Ego alive per is_alive but absent from the snapshot: still fatal.
        self.ego.is_alive = True
        with self.assertRaises(EgoActorLostError):
            self.recorder.record_frame(0, 1000, 0.0, FakeSnapshot([car]), self.ego, [managed(car)])

        # Ego must never be routed through the background omission path.
        with self.assertRaises(EgoActorLostError):
            self.record([self.ego])

    def test_lifecycle_across_samples_stays_consistent(self):
        """sample 0: actor present; stale at sample 1 -> absent + 'removed'; the
        reader (replay input) sees exactly that."""
        car = FakeActor(10, "vehicle.audi.a2")
        self.record([car], frame_id=0)
        car.fail_on["get_light_state"] = RuntimeError(REGISTRY_ERROR)
        state1 = self.record([car], frame_id=1)
        self.recorder.finalize()

        self.assertEqual(state1["actors"], {})
        self.assertEqual(state1["removed"], [0])

        reader = WorldStateReader(os.path.join(self.root, "geometry"))
        self.assertEqual(reader.actors[0]["last_frame"], 0)
        self.assertEqual(reader.load_frame(1)["actors"], {})


# ------------------------------------------------------------------
# Tests 1-6: annotation writer (camera-less mode: lifecycle only, the
# camera-valid / depth-visibility logic is untouched and tested elsewhere)
# ------------------------------------------------------------------

class AnnotationStaleActorTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.writer = AnnotationWriter(self.root, annotation_cfg())
        self.ego = FakeActor(1, "vehicle.tesla.model3", x=0.0)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, actors):
        return self.writer.write_frame(0, FakeWorld([self.ego] + actors), self.ego)

    def load(self):
        with open(os.path.join(self.root, "labels", "object_3d", "000000.json")) as f:
            return json.load(f)

    def car(self, actor_id, **kwargs):
        return FakeActor(actor_id, "vehicle.audi.a2", attributes={"base_type": "car"}, **kwargs)

    def test_1_normal_actor_annotated(self):
        counts = self.write([self.car(10)])
        data = self.load()

        self.assertEqual(counts["vehicle"], 1)
        self.assertEqual(data["objects"][0]["actor_id"], 10)
        self.assertTrue(data["objects"][0]["light_state"]["brake"])
        self.assertNotIn("stale_actor_omissions", data)

    def test_3_light_state_rpc_fails_after_transform_whole_annotation_omitted(self):
        dying = self.car(10, fail_on={"get_light_state": RuntimeError(REGISTRY_ERROR)})
        counts = self.write([dying, self.car(11, x=20.0)])
        data = self.load()

        self.assertIn("get_transform", dying.calls)          # earlier reads succeeded ...
        self.assertEqual([o["actor_id"] for o in data["objects"]], [11])  # ... but nothing partial kept
        self.assertEqual(data["rejected_objects"], [])
        self.assertEqual(data["stale_actor_omissions"], [10])
        self.assertEqual(counts["vehicle"], 1)

    def test_4_first_getter_raises_actor_omitted(self):
        dying = self.car(10, fail_on={"get_transform": RuntimeError(DESTROYED_WRAPPER_ERROR)})
        self.write([dying])
        data = self.load()

        self.assertEqual(data["objects"], [])
        self.assertEqual(data["stale_actor_omissions"], [10])

    def test_5_unrelated_runtime_error_propagates(self):
        broken = self.car(10, fail_on={"get_light_state": RuntimeError("some unrelated failure")})

        with self.assertRaisesRegex(RuntimeError, "some unrelated failure"):
            self.write([broken])

    def test_6_ego_stale_is_fatal(self):
        self.ego.is_alive = False
        with self.assertRaises(EgoActorLostError):
            self.write([self.car(10)])

    def test_6_ego_destroyed_mid_frame_is_fatal_not_omitted(self):
        # A stale error raised by the EGO's getter inside the per-actor loop
        # must not be mistaken for a stale background actor.
        def ego_dies():
            self.ego.is_alive = False
            raise RuntimeError(DESTROYED_WRAPPER_ERROR)

        self.ego.get_transform = ego_dies
        with self.assertRaises(EgoActorLostError):
            self.write([self.car(10)])
        self.assertFalse(os.path.exists(os.path.join(self.root, "labels", "object_3d", "000000.json")))


# ------------------------------------------------------------------
# Canonical traffic cleanup: stale-only tolerance on destroy
# ------------------------------------------------------------------

class CanonicalDestroyTest(unittest.TestCase):

    def destroy(self, actor_error=None, controller=None):
        from src.simulation.canonical_traffic import CanonicalBackgroundTraffic

        actor = mock.Mock(is_alive=True)
        if actor_error is not None:
            actor.destroy.side_effect = actor_error
        CanonicalBackgroundTraffic._destroy_managed_actor(
            None, {"actor": actor, "controller": controller})
        return actor

    def test_normal_destroy(self):
        controller = mock.Mock(is_alive=True)
        actor = self.destroy(controller=controller)
        actor.destroy.assert_called_once()
        controller.stop.assert_called_once()
        controller.destroy.assert_called_once()

    def test_already_gone_actor_is_tolerated(self):
        self.destroy(actor_error=RuntimeError(REGISTRY_ERROR))

    def test_unrelated_destroy_error_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            self.destroy(actor_error=RuntimeError("some unrelated failure"))


# ------------------------------------------------------------------
# Tests 7-10: canonical restart vs. weather reuse (production process_route)
# ------------------------------------------------------------------

ALL = ["day_clear", "day_rain", "night_fog"]


def write_frames(directory, count, tag):
    os.makedirs(directory, exist_ok=True)
    for i in range(count):
        with open(os.path.join(directory, f"{i:06d}.png"), "w") as f:
            f.write(tag)


class ResumeSemanticsTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.route = os.path.join(self.root, "Town10", "route_1")
        self.args = SimpleNamespace(
            rerender_conditions=[], overwrite=False, max_frames=3, truncate_ok=True,
            no_traffic=False, no_pedestrians=False, background_policy="canonical",
            replay_position_tolerance=0.01, replay_rotation_tolerance=0.1,
        )
        self.canonical_calls = []
        self.replay_calls = []
        self.files_seen_by_canonical = None

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # -- fakes for the two CARLA-bound stages ---------------------------
    def fake_canonical(self, fail=False):
        def run(route_path, **kwargs):
            self.canonical_calls.append(route_path)
            self.files_seen_by_canonical = sorted(
                os.path.relpath(os.path.join(d, f), route_path)
                for d, _dirs, files in os.walk(route_path) for f in files
            ) if os.path.isdir(route_path) else []
            write_frames(os.path.join(condition_dir(route_path, "day_clear"), "rgb_left"), 2, "new")
            os.makedirs(geometry_dir(route_path), exist_ok=True)
            if fail:
                raise RuntimeError(REGISTRY_ERROR)
            for name, data in (("sequence.json", {"recording_hz": 10.0}), ("calibration.json", {})):
                with open(os.path.join(geometry_dir(route_path), name), "w") as f:
                    json.dump(data, f)
            mark_complete(geometry_dir(route_path))
            mark_complete(condition_dir(route_path, "day_clear"))
        return run

    def fake_replay(self, failing=()):
        def run(condition, route_path, **kwargs):
            self.replay_calls.append(condition)
            if condition in failing:
                raise RuntimeError("replay boom")
            write_frames(os.path.join(condition_dir(route_path, condition), "rgb_left"), 2, condition)
            mark_complete(condition_dir(route_path, condition))
        return run

    def process(self, canonical, replay):
        with mock.patch.object(cd, "generate_canonical_geometry", side_effect=canonical), \
                mock.patch.object(cd, "replay_condition", side_effect=replay), \
                mock.patch.object(cd, "WorldStateReader"), \
                mock.patch.object(cd, "update_sequence_conditions"), \
                mock.patch.object(cd, "validate_route", return_value={"passed": True, "errors": [], "num_frames": 2}), \
                mock.patch.object(cd, "write_json"):
            return cd.process_route(None, None, None, "Town10", 1, None, ALL, "day_clear",
                                    self.root, self.args)

    def make_partial_canonical(self):
        """A canonical run that died at frame ~6: frames on disk, no COMPLETE."""
        write_frames(os.path.join(condition_dir(self.route, "day_clear"), "rgb_left"), 6, "old")
        write_frames(os.path.join(geometry_dir(self.route), "world_state"), 6, "old")

    def test_canonical_failure_is_not_complete_and_skips_replay(self):
        counts = self.process(self.fake_canonical(fail=True), self.fake_replay())

        self.assertEqual(counts["failed"], 1)
        self.assertEqual(self.replay_calls, [])                  # never replay a partial canonical
        self.assertFalse(is_complete(geometry_dir(self.route)))
        self.assertFalse(is_complete(condition_dir(self.route, "day_clear")))

    def test_7_partial_canonical_is_regenerated(self):
        self.make_partial_canonical()
        plan = plan_route_work(self.route, ALL, "day_clear", expected_recording_hz=10.0)
        self.assertTrue(plan["run_canonical"])
        self.assertTrue(plan["delete_route"])
        self.assertEqual(plan["replay"], ["day_rain", "night_fog"])

    def test_9_partial_canonical_files_cannot_mix_with_fresh_run(self):
        self.make_partial_canonical()
        self.process(self.fake_canonical(), self.fake_replay())

        self.assertEqual(self.files_seen_by_canonical, [])       # route wiped before the fresh run
        rgb = os.path.join(condition_dir(self.route, "day_clear"), "rgb_left")
        self.assertEqual(sorted(os.listdir(rgb)), ["000000.png", "000001.png"])
        for name in os.listdir(rgb):
            with open(os.path.join(rgb, name)) as f:
                self.assertEqual(f.read(), "new")
        self.assertFalse(os.path.exists(os.path.join(geometry_dir(self.route), "world_state")))

    def test_8_and_10_complete_canonical_reused_only_failed_weather_rerendered(self):
        # Run 1: canonical OK, night_fog replay fails.
        self.process(self.fake_canonical(), self.fake_replay(failing=("night_fog",)))
        self.assertEqual(len(self.canonical_calls), 1)
        self.assertTrue(is_complete(geometry_dir(self.route)))
        self.assertTrue(is_complete(condition_dir(self.route, "day_rain")))
        self.assertFalse(is_complete(condition_dir(self.route, "night_fog")))

        sentinel = os.path.join(geometry_dir(self.route), "world_state_sentinel.json")
        with open(sentinel, "w") as f:
            f.write("canonical")

        # Run 2: canonical reused (never re-run, never deleted), only night_fog replayed.
        self.replay_calls.clear()
        self.process(self.fake_canonical(), self.fake_replay())

        self.assertEqual(len(self.canonical_calls), 1)
        self.assertEqual(self.replay_calls, ["night_fog"])
        self.assertTrue(os.path.exists(sentinel))
        self.assertTrue(is_complete(geometry_dir(self.route)))
        self.assertTrue(is_complete(condition_dir(self.route, "night_fog")))

    def test_10_plan_never_deletes_complete_canonical(self):
        os.makedirs(geometry_dir(self.route))
        with open(os.path.join(geometry_dir(self.route), "sequence.json"), "w") as f:
            json.dump({"recording_hz": 10.0}, f)
        mark_complete(geometry_dir(self.route))
        os.makedirs(condition_dir(self.route, "day_clear"))
        mark_complete(condition_dir(self.route, "day_clear"))

        plan = plan_route_work(self.route, ALL, "day_clear", expected_recording_hz=10.0)
        self.assertFalse(plan["delete_route"])
        self.assertFalse(plan["run_canonical"])
        self.assertEqual(plan["replay"], ["day_rain", "night_fog"])


if __name__ == "__main__":
    unittest.main()

"""
test_replay_offline.py

Offline (no CARLA server) tests of the canonical-geometry / weather-replay
machinery. Uses the real `carla` Python types (Transform, Vector3D, ...) but
a fake world, so it validates the recording format, actor lifecycle
reproduction, transform verification, RGB-only replay contract (sensor
profiles, wind policy, resume/rerender planning, metadata) and paired-dataset
validation logic -- NOT CARLA's runtime behaviour (physics, rendering,
sensors), which needs the live smoke test:

    python scripts/collect_dataset.py --towns Town01 --routes 0 \
        --conditions day_clear day_rain --max-frames 200 --truncate-ok --overwrite

Run:
    python -m unittest scripts.tests.test_replay_offline -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import carla  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.data.layout import (  # noqa: E402
    condition_dir,
    geometry_dir,
    is_complete,
    mark_complete,
    resolve_condition_root,
    resolve_geometry_root,
)
from src.data.paired_validation import (  # noqa: E402
    ArrayComparison,
    compare_calibration,
    validate_route,
)
from src.data.world_state import (  # noqa: E402
    ErrorAccumulator,
    WorldStateReader,
    WorldStateRecorder,
    pose_error,
)
from src.simulation.replay import WorldStateReplayer  # noqa: E402


# ------------------------------------------------------------------
# Fake CARLA world
# ------------------------------------------------------------------

class FakeBlueprint:
    def __init__(self, blueprint_id):
        self.id = blueprint_id
        self.attributes = {}

    def has_attribute(self, name):
        return name in ("color", "driver_id", "role_name", "is_invincible")

    def set_attribute(self, name, value):
        self.attributes[name] = value


class FakeLibrary:
    def find(self, blueprint_id):
        return FakeBlueprint(blueprint_id)


class FakeActor:
    def __init__(self, world, actor_id, blueprint, transform):
        self.world = world
        self.id = actor_id
        self.type_id = blueprint.id
        self.attributes = dict(blueprint.attributes)
        self.is_alive = True
        self._transform = transform
        self.velocity = carla.Vector3D(0.0, 0.0, 0.0)
        self.physics = True
        self.light_state = 0
        self.walker_control = None
        self.direct_set_transform_calls = 0

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location

    def set_transform(self, transform):
        self.direct_set_transform_calls += 1
        self._transform = transform

    def set_simulate_physics(self, flag):
        self.physics = flag

    def get_light_state(self):
        return carla.VehicleLightState(self.light_state)

    def set_light_state(self, state):
        self.light_state = int(state)

    def apply_control(self, control):
        self.walker_control = control

    def destroy(self):
        self.is_alive = False
        self.world.actors.pop(self.id, None)


class FakeTrafficLight:
    def __init__(self, x, y, light_id=0):
        self.id = light_id
        self._transform = carla.Transform(carla.Location(x=x, y=y, z=5.0))
        self.state = carla.TrafficLightState.Red

    def get_transform(self):
        return self._transform

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeActorSnapshot:
    def __init__(self, actor):
        self._transform = actor.get_transform()
        self._velocity = actor.velocity

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_angular_velocity(self):
        return carla.Vector3D(0.0, 0.0, 0.0)


class FakeSnapshot:
    def __init__(self, actors):
        self._actors = {actor_id: FakeActorSnapshot(actor) for actor_id, actor in actors.items()}

    def find(self, actor_id):
        return self._actors.get(actor_id)


class FakeWorld:
    def __init__(self, first_actor_id=100, walker_gravity_drift=0.0):
        self.actors = {}
        self.next_id = first_actor_id
        self.lights = [FakeTrafficLight(0.0, 0.0, 5000), FakeTrafficLight(500.0, 0.0, 5001)]
        self.frozen = False
        self.walker_gravity_drift = walker_gravity_drift
        self.blocked_spawn_z = None

    def get_blueprint_library(self):
        return FakeLibrary()

    def try_spawn_actor(self, blueprint, transform):
        actor = FakeActor(self, self.next_id, blueprint, transform)
        self.next_id += 1
        self.actors[actor.id] = actor
        return actor

    def get_actors(self):
        world = self

        class Actors:
            def filter(self, pattern):
                assert pattern.startswith("traffic.traffic_light")
                return world.lights

        return Actors()

    def freeze_all_traffic_lights(self, frozen):
        self.frozen = frozen

    def tick(self):
        # Physics-enabled walkers are pulled down by "gravity" (a stand-in
        # for the drift verify_frame() must catch); everything else stays.
        for actor in self.actors.values():
            if actor.type_id.startswith("walker.") and actor.physics and self.walker_gravity_drift:
                transform = actor.get_transform()
                transform.location.z -= self.walker_gravity_drift
                actor.set_transform(transform)

    def get_snapshot(self):
        return FakeSnapshot(self.actors)


class FakeResponse:
    def __init__(self, error=""):
        self.error = error


class FakeClient:
    """Executes the real carla.command.* objects against the fake world."""

    def __init__(self, world, fail_on=None, defer_one_tick_every=0):
        self.world = world
        self.fail_on = fail_on
        self.batches = 0

    def apply_batch_sync(self, commands, due_tick_cue=False):
        self.batches += 1
        responses = []
        lights = {light.id: light for light in self.world.lights}

        for command in commands:
            if self.fail_on is not None and isinstance(command, self.fail_on):
                responses.append(FakeResponse("boom"))
                continue

            actor = self.world.actors.get(command.actor_id) or lights.get(command.actor_id)

            if isinstance(command, carla.command.ApplyTransform):
                actor._transform = command.transform
            elif isinstance(command, carla.command.SetSimulatePhysics):
                actor.physics = command.enabled
            elif isinstance(command, carla.command.ApplyWalkerControl):
                actor.walker_control = command.control
            elif isinstance(command, carla.command.SetVehicleLightState):
                actor.light_state = int(command.light_state)
            else:
                raise AssertionError(f"unexpected command {command}")

            responses.append(FakeResponse())

        return responses


def make_transform(x, y, z=0.0, yaw=0.0):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))


# ------------------------------------------------------------------
# Canonical scene: ego drives, car A lives frames 0-5, walker W frames
# 2-7, car B appears at frame 4 and stays.
# ------------------------------------------------------------------

def record_canonical_scene(geometry_root, num_frames=8):
    world = FakeWorld(first_actor_id=1)
    blueprint = FakeLibrary().find

    ego = world.try_spawn_actor(blueprint("vehicle.tesla.model3"), make_transform(0, 0))

    car_a_bp = blueprint("vehicle.audi.a2")
    car_a_bp.attributes["color"] = "255,0,0"
    walker_bp = blueprint("walker.pedestrian.0001")
    car_b_bp = blueprint("vehicle.bmw.grandtourer")

    car_a = walker = car_b = None
    managed = []

    from src.data.world_state import TrafficLightRecorder

    recorder = WorldStateRecorder(
        geometry_root,
        traffic_light_recorder=TrafficLightRecorder(world, radius_m=100.0),
    )

    for frame in range(num_frames):

        if frame == 0:
            car_a = world.try_spawn_actor(car_a_bp, make_transform(10, 0))
            managed.append({"actor": car_a, "category": "vehicle"})

        if frame == 2:
            walker = world.try_spawn_actor(walker_bp, make_transform(20, 3))
            managed.append({"actor": walker, "category": "pedestrian"})

        if frame == 4:
            car_b = world.try_spawn_actor(car_b_bp, make_transform(30, -3, yaw=180.0))
            managed.append({"actor": car_b, "category": "vehicle"})

        if frame == 6:
            car_a.destroy()  # naturally removed by the canonical traffic system

        # Motion: ego along +x, everything else drifts.
        ego.set_transform(make_transform(frame * 1.5, 0.0))
        for actor in list(world.actors.values()):
            if actor is not ego:
                t = actor.get_transform()
                actor.set_transform(make_transform(t.location.x + 0.3, t.location.y, t.location.z, t.rotation.yaw))
                actor.velocity = carla.Vector3D(6.0, 0.0, 0.0)

        if car_a is not None and car_a.is_alive:
            car_a.light_state = 8 if frame % 2 else 0  # brake lights toggle

        world.lights[0].state = carla.TrafficLightState.Green if frame >= 3 else carla.TrafficLightState.Red

        recorder.record_frame(frame, 1000 + frame, frame * 0.05, world.get_snapshot(), ego, managed)

        if frame == 7:
            walker.destroy()

    recorder.finalize()

    return recorder


class WorldStateRoundTripTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.geometry = os.path.join(self.root, "geometry")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_registry_and_lifecycle(self):
        record_canonical_scene(self.geometry)
        reader = WorldStateReader(self.geometry)

        self.assertEqual(len(reader), 8)
        self.assertEqual(len(reader.actors), 3)  # car A, walker, car B: 3 logical ids
        self.assertEqual(reader.actors[0]["blueprint_id"], "vehicle.audi.a2")
        self.assertEqual(reader.actors[0]["attributes"], {"color": "255,0,0"})
        self.assertEqual(reader.actors[0]["first_frame"], 0)
        self.assertEqual(reader.actors[0]["last_frame"], 5)  # destroyed before frame 6 record
        self.assertEqual(reader.actors[1]["category"], "pedestrian")
        self.assertEqual(reader.actors[2]["first_frame"], 4)

        self.assertEqual(reader.load_frame(0)["spawned"], [0])
        self.assertEqual(sorted(reader.load_frame(2)["actors"]), ["0", "1"])
        self.assertEqual(reader.load_frame(4)["spawned"], [2])
        self.assertEqual(reader.load_frame(6)["removed"], [0])
        self.assertEqual(sorted(reader.load_frame(6)["actors"]), ["1", "2"])

        # Only the light within 100 m of the ego is recorded.
        self.assertEqual(list(reader.load_frame(0)["traffic_lights"].values()), ["Red"])
        self.assertEqual(list(reader.load_frame(3)["traffic_lights"].values()), ["Green"])

    def test_logical_id_is_stable_and_not_a_carla_id(self):
        recorder = record_canonical_scene(self.geometry)
        reader = WorldStateReader(self.geometry)

        for logical_id, info in reader.actors.items():
            self.assertNotEqual(logical_id, info["canonical_actor_id"])

        self.assertEqual(recorder.registry[0]["canonical_actor_id"], 2)


class ReplayTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.geometry = os.path.join(self.root, "geometry")
        record_canonical_scene(self.geometry)
        self.reader = WorldStateReader(self.geometry)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def replay(self, world):
        # Replay CARLA ids deliberately differ from the canonical ones.
        replayer = WorldStateReplayer(world, FakeClient(world), self.reader, "vehicle.tesla.model3")
        first = self.reader.load_frame(0)
        replayer.start(first)

        lifecycle = []

        for frame_id in range(len(self.reader)):
            state = self.reader.load_frame(frame_id)
            replayer.apply_frame(state)
            world.tick()
            replayer.verify_frame(state)
            lifecycle.append(sorted(replayer.actors))

        return replayer, lifecycle

    def test_exact_replay_passes(self):
        world = FakeWorld(first_actor_id=9000)
        replayer, lifecycle = self.replay(world)
        summary = replayer.summary()

        self.assertTrue(summary["passed"], summary)
        self.assertEqual(summary["frames_verified"], 8)
        self.assertEqual(summary["actor_id_set_mismatch_frames"], 0)
        self.assertEqual(summary["ego"]["position_max_m"], 0.0)
        self.assertEqual(summary["vehicle_like_actors"]["position_max_m"], 0.0)

        # Lifecycle reproduced: A(0) frames 0-5, walker(1) 2-7, B(2) from 4.
        self.assertEqual(lifecycle[0], [0])
        self.assertEqual(lifecycle[2], [0, 1])
        self.assertEqual(lifecycle[4], [0, 1, 2])
        self.assertEqual(lifecycle[6], [1, 2])
        self.assertEqual(lifecycle[7], [1, 2])

        self.assertTrue(world.frozen)
        self.assertEqual(replayer.total_actor_spawns, 3)
        self.assertEqual(replayer.total_actor_destroys, 1)

        # Replay actor CARLA ids are not the canonical ones.
        self.assertTrue(all(actor.id >= 9000 for actor in replayer.actors.values()))

    def test_physics_disabled_for_vehicles_only(self):
        world = FakeWorld(first_actor_id=9000)
        replayer, _ = self.replay(world)

        self.assertFalse(replayer.ego.physics)
        self.assertFalse(replayer.actors[2].physics)   # vehicle
        self.assertTrue(replayer.actors[1].physics)    # walker keeps its controller

    def test_vehicle_lights_traffic_lights_and_walker_animation_applied(self):
        world = FakeWorld(first_actor_id=9000)
        replayer, _ = self.replay(world)

        self.assertEqual(world.lights[0].state, carla.TrafficLightState.Green)
        self.assertIsNotNone(replayer.actors[1].walker_control)
        self.assertGreater(replayer.actors[1].walker_control.speed, 0.0)

    def test_no_individual_set_transform_rpc_is_used(self):
        """Per-frame poses go through apply_batch_sync, never actor.set_transform
        (individual RPCs were measured to sometimes apply one tick late)."""
        world = FakeWorld(first_actor_id=9000)
        replayer, _ = self.replay(world)

        self.assertEqual(replayer.ego.direct_set_transform_calls, 0)
        for actor in replayer.actors.values():
            self.assertEqual(actor.direct_set_transform_calls, 0)

    def test_batch_path_is_used_every_frame(self):
        """F: apply_batch_sync remains the replay transform path."""
        world = FakeWorld(first_actor_id=9000)
        client = FakeClient(world)
        replayer = WorldStateReplayer(world, client, self.reader, "vehicle.tesla.model3")
        replayer.start(self.reader.load_frame(0))

        for frame_id in range(len(self.reader)):
            state = self.reader.load_frame(frame_id)
            replayer.apply_frame(state)
            world.tick()
            replayer.verify_frame(state)

        self.assertEqual(client.batches, len(self.reader))
        self.assertTrue(replayer.summary()["passed"])
        self.assertEqual(replayer.summary()["out_of_tolerance_samples"], 0)
        self.assertIn("worst_out_of_tolerance", replayer.summary())

    def test_failed_command_raises(self):
        from src.simulation.replay import ReplayError

        world = FakeWorld(first_actor_id=9000)
        replayer = WorldStateReplayer(
            world, FakeClient(world, fail_on=carla.command.ApplyTransform), self.reader, "vehicle.tesla.model3")
        replayer.start(self.reader.load_frame(0))

        with self.assertRaises(ReplayError):
            replayer.apply_frame(self.reader.load_frame(0))

    def test_drift_after_tick_is_detected(self):
        world = FakeWorld(first_actor_id=9000, walker_gravity_drift=0.2)
        replayer, _ = self.replay(world)
        summary = replayer.summary()

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["pedestrians"]["passed"])
        self.assertTrue(summary["ego"]["passed"])
        self.assertAlmostEqual(summary["pedestrians"]["position_max_m"], 0.2, places=6)

    def test_actor_missing_after_tick_fails(self):
        world = FakeWorld(first_actor_id=9000)
        replayer = WorldStateReplayer(world, FakeClient(world), self.reader, "vehicle.tesla.model3")
        replayer.start(self.reader.load_frame(0))
        state = self.reader.load_frame(0)
        replayer.apply_frame(state)
        world.tick()
        world.actors.pop(replayer.actors[0].id)  # actor vanished from the world
        replayer.verify_frame(state)

        self.assertEqual(replayer.actors_missing_in_world, 1)
        self.assertFalse(replayer.summary()["passed"])


class MathTest(unittest.TestCase):

    def test_pose_error_wraps_angles(self):
        a = {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 179.5}
        b = {"x": 3, "y": 4, "z": 0, "roll": 0, "pitch": 0, "yaw": -179.5}
        position, rotation = pose_error(a, b)
        self.assertAlmostEqual(position, 5.0)
        self.assertAlmostEqual(rotation, 1.0)

    def test_error_accumulator(self):
        acc = ErrorAccumulator()
        acc.add(0.1, 1.0)
        acc.add(0.3, 3.0)
        summary = acc.summary()
        self.assertAlmostEqual(summary["position_mean_m"], 0.2)
        self.assertAlmostEqual(summary["rotation_max_deg"], 3.0)

    def test_array_comparison(self):
        comparison = ArrayComparison()
        a = np.zeros((2, 2), dtype=np.float32)
        b = a.copy()
        b[0, 0] = 2.0
        comparison.add(a, b)
        summary = comparison.summary()
        self.assertEqual(summary["max_abs_difference"], 2.0)
        self.assertEqual(summary["percentage_equal"], 75.0)

    def test_compare_calibration(self):
        canonical = {"sensors": {"rgb_left": {"T": [[1.0, 0.0], [0.0, 1.0]]}, "lidar": {"T": [[9.0]]}},
                     "cameras": {}, "stereo": {"baseline_m": 0.5}}
        replayed = {"sensors": {"rgb_left": {"T": [[1.0, 0.0], [0.0, 1.0]]}},
                    "cameras": {}, "stereo": {"baseline_m": 0.5}}
        self.assertTrue(compare_calibration(canonical, replayed)["equal"])
        replayed["sensors"]["rgb_left"]["T"][0][0] = 1.01
        self.assertFalse(compare_calibration(canonical, replayed)["equal"])

    def test_layout_resolution(self):
        route = os.path.abspath(os.path.join("x", "Town01", "route_0"))
        cond = os.path.join(route, "conditions", "day_rain")
        self.assertEqual(resolve_geometry_root(cond), os.path.join(route, "geometry"))
        self.assertEqual(resolve_geometry_root(os.path.join(route, "geometry")), os.path.join(route, "geometry"))
        self.assertEqual(resolve_condition_root(cond), cond)
        self.assertEqual(
            resolve_condition_root(os.path.join(route, "geometry")),
            os.path.join(route, "conditions", "day_clear"),
        )
        # Flat legacy layout is passed through unchanged.
        self.assertEqual(resolve_geometry_root(os.path.join("a", "b")), os.path.abspath(os.path.join("a", "b")))


def make_condition_meta(condition, is_source, frames):
    """condition.json as written by scripts/collect_dataset.py write_condition_json()."""

    passed_group = {"samples": frames, "position_max_m": 3e-5, "rotation_max_deg": 1e-4, "passed": True}

    return {
        "condition": condition,
        "wind_intensity": 0.0,
        "rendered_from": "canonical_run" if is_source else "replay",
        "rendered_from_canonical_geometry": True,
        "rgb_only_replay": not is_source,
        "sensors_saved": ["rgb_left", "rgb_right"],
        "num_frames": frames,
        "replay_validation": None if is_source else {
            "passed": True,
            "frames_verified": frames,
            "actor_id_set_mismatch_frames": 0,
            "actors_missing_in_world": 0,
            "ego": dict(passed_group),
            "vehicle_like_actors": dict(passed_group),
            "pedestrians": dict(passed_group),
            "rgb_frames_saved": frames,
        },
        "calibration_check": None if is_source else {"equal": True},
    }


class PairedValidationTest(unittest.TestCase):

    CONDITIONS = ["day_clear", "day_rain"]

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.route = os.path.join(self.root, "Town01", "route_0")
        self.geometry = geometry_dir(self.route)
        record_canonical_scene(self.geometry, num_frames=4)
        self.frames = 4

        from src.data.layout import GEOMETRY_SENSOR_EXTENSIONS

        for name in GEOMETRY_SENSOR_EXTENSIONS:
            os.makedirs(os.path.join(self.geometry, name))
            for i in range(self.frames):
                np.save(os.path.join(self.geometry, name, f"{i:06d}.npy"), np.zeros((2, 2)))

        os.makedirs(os.path.join(self.geometry, "labels", "object_3d"))
        for i in range(self.frames):
            with open(os.path.join(self.geometry, "labels", "object_3d", f"{i:06d}.json"), "w") as f:
                json.dump({"objects": [{"logical_id": 0}]}, f)

        for index, condition in enumerate(self.CONDITIONS):
            cdir = condition_dir(self.route, condition)
            for camera in ("rgb_left", "rgb_right"):
                os.makedirs(os.path.join(cdir, camera))
                for i in range(self.frames):
                    image = np.full((4, 4, 3), 50 + 60 * index, dtype=np.uint8)
                    cv2.imwrite(os.path.join(cdir, camera, f"{i:06d}.png"), image)

            with open(os.path.join(cdir, "condition.json"), "w") as f:
                json.dump(make_condition_meta(condition, index == 0, self.frames), f)

            mark_complete(cdir)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_route_passes(self):
        report = validate_route(self.route, self.CONDITIONS)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["num_frames"], self.frames)
        self.assertGreater(report["checks"]["rgb_mean_abs_difference_vs_source"]["day_rain"], 50.0)

    def test_frame_count_mismatch_fails(self):
        os.remove(os.path.join(condition_dir(self.route, "day_rain"), "rgb_left", "000003.png"))
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])
        self.assertTrue(any("day_rain/rgb_left" in e for e in report["errors"]))

    def test_shifted_frame_ids_fail(self):
        cdir = os.path.join(condition_dir(self.route, "day_rain"), "rgb_right")
        os.rename(os.path.join(cdir, "000003.png"), os.path.join(cdir, "000004.png"))
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])

    def test_identical_rgb_means_weather_not_applied(self):
        for camera in ("rgb_left", "rgb_right"):
            for i in range(self.frames):
                shutil.copyfile(
                    os.path.join(condition_dir(self.route, "day_clear"), camera, f"{i:06d}.png"),
                    os.path.join(condition_dir(self.route, "day_rain"), camera, f"{i:06d}.png"),
                )
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])
        self.assertTrue(any("weather did not apply" in e for e in report["errors"]))

    def test_unknown_label_logical_id_fails(self):
        with open(os.path.join(self.geometry, "labels", "object_3d", "000001.json"), "w") as f:
            json.dump({"objects": [{"logical_id": 99}]}, f)
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])

    def edit_condition(self, condition, mutate):
        path = os.path.join(condition_dir(self.route, condition), "condition.json")
        with open(path) as f:
            meta = json.load(f)
        mutate(meta)
        with open(path, "w") as f:
            json.dump(meta, f)

    def test_failed_replay_validation_fails(self):
        self.edit_condition("day_rain", lambda m: m["replay_validation"].update(passed=False))
        self.assertFalse(validate_route(self.route, self.CONDITIONS)["passed"])

    def test_required_replay_geometry_criteria(self):
        """ego / actor transform / actor-id set / frame count / calibration reference."""
        cases = {
            "ego transform": lambda m: m["replay_validation"]["ego"].update(passed=False),
            "actor transform": lambda m: m["replay_validation"]["vehicle_like_actors"].update(passed=False),
            "pedestrian transform": lambda m: m["replay_validation"]["pedestrians"].update(passed=False),
            "actor id set": lambda m: m["replay_validation"].update(actor_id_set_mismatch_frames=2),
            "verified frames": lambda m: m["replay_validation"].update(frames_verified=self.frames - 1),
            "rgb frames": lambda m: m["replay_validation"].update(rgb_frames_saved=self.frames - 1),
            "calibration reference": lambda m: m["calibration_check"].update(equal=False),
        }

        for name, mutate in cases.items():
            with self.subTest(name):
                original = os.path.join(condition_dir(self.route, "day_rain"), "condition.json")
                with open(original) as f:
                    backup = json.load(f)
                self.edit_condition("day_rain", mutate)
                self.assertFalse(validate_route(self.route, self.CONDITIONS)["passed"], name)
                with open(original, "w") as f:
                    json.dump(backup, f)

        self.assertTrue(validate_route(self.route, self.CONDITIONS)["passed"])

    def test_metadata_requirements(self):
        cases = {
            "wind": lambda m: m.update(wind_intensity=5.0),
            "rgb_only_replay": lambda m: m.update(rgb_only_replay=False),
            "rendered_from_canonical_geometry": lambda m: m.update(rendered_from_canonical_geometry=False),
            "condition name": lambda m: m.update(condition="night_fog"),
            "num_frames": lambda m: m.update(num_frames=self.frames + 1),
        }

        for name, mutate in cases.items():
            with self.subTest(name):
                original = os.path.join(condition_dir(self.route, "day_rain"), "condition.json")
                with open(original) as f:
                    backup = json.load(f)
                self.edit_condition("day_rain", mutate)
                self.assertFalse(validate_route(self.route, self.CONDITIONS)["passed"], name)
                with open(original, "w") as f:
                    json.dump(backup, f)

    def test_replay_does_not_require_non_rgb_equality(self):
        """E: depth / semantic / flow / lidar / radar results are not a criterion."""
        self.edit_condition("day_rain", lambda m: m["replay_validation"].update(
            sensor_equality={"depth": {"percentage_equal": 12.0, "max_abs_difference": 995.0}},
            depth_equal=False, semantic_equal=False, optical_flow_equal=False,
            lidar_equal=False, radar_equal=False))
        report = validate_route(self.route, self.CONDITIONS)
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(report["validates_geometry_replay"])
        self.assertTrue(report["validates_rgb_presence"])
        self.assertFalse(report["validates_non_rgb_sensor_replay"])

    def test_replay_directory_must_be_rgb_only(self):
        os.makedirs(os.path.join(condition_dir(self.route, "day_rain"), "depth"))
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])
        self.assertTrue(any("non-RGB" in e for e in report["errors"]))

    def test_rgb_dimension_mismatch_fails(self):
        cdir = os.path.join(condition_dir(self.route, "day_rain"), "rgb_right")
        for i in range(self.frames):
            cv2.imwrite(os.path.join(cdir, f"{i:06d}.png"), np.full((8, 6, 3), 150, dtype=np.uint8))
        report = validate_route(self.route, self.CONDITIONS)
        self.assertFalse(report["passed"])
        self.assertTrue(any("image size" in e for e in report["errors"]))

    def test_completion_marker_resume_state(self):
        self.assertTrue(is_complete(condition_dir(self.route, "day_rain")))
        self.assertFalse(is_complete(condition_dir(self.route, "day_fog")))


class RgbOnlyReplayContractTest(unittest.TestCase):
    """A-D, G-I: sensor profiles, wind policy, scheduling, metadata (offline)."""

    ALL = ["day_clear", "day_rain", "day_fog", "night_clear", "night_rain", "night_fog"]

    # ---- A / B: sensor profiles ------------------------------------------
    def spawn_names(self, **kwargs):
        from unittest import mock
        from src.sensors import sensor_rig

        rig = sensor_rig.SensorRig(world=None, ego=None, cfg=mock.MagicMock())
        attached = []

        def attach(name, *args):
            attached.append(name)
            rig.sensors[name] = object()

        rig._attach_sensor = attach

        names = {n: n for n in sensor_rig.CAMERA_NAMES}

        with mock.patch.object(sensor_rig, "create_camera_rig_blueprints", return_value=names),                 mock.patch.object(sensor_rig, "create_camera_rig_transforms", return_value=names),                 mock.patch.object(sensor_rig, "create_lidar_blueprint"),                 mock.patch.object(sensor_rig, "create_lidar_transform"),                 mock.patch.object(sensor_rig, "create_radar_blueprint"),                 mock.patch.object(sensor_rig, "create_radar_transform"):
            rig.spawn(**kwargs)

        return attached

    def test_A_replay_profile_is_rgb_only(self):
        from src.sensors.sensor_rig import REPLAY_SENSOR_PROFILE

        self.assertEqual(list(REPLAY_SENSOR_PROFILE), ["rgb_left", "rgb_right"])
        self.assertEqual(self.spawn_names(profile="replay"), ["rgb_left", "rgb_right"])

    def test_B_canonical_profile_has_every_gt_sensor(self):
        from src.sensors.sensor_rig import CANONICAL_SENSOR_PROFILE

        expected = ["rgb_left", "rgb_right", "depth", "optical_flow", "semantic", "lidar",
                    "radar", "radar_front_left", "radar_front_right"]  # "radar" == front radar
        self.assertEqual(sorted(CANONICAL_SENSOR_PROFILE), sorted(expected))
        self.assertEqual(sorted(self.spawn_names(profile="canonical")), sorted(expected))
        self.assertEqual(sorted(self.spawn_names()), sorted(expected))  # default unchanged

    def test_profile_argument_errors(self):
        from unittest import mock
        from src.sensors.sensor_rig import SensorRig

        rig = SensorRig(world=None, ego=None, cfg=mock.MagicMock())
        with self.assertRaises(ValueError):
            rig.spawn(profile="nope")
        with self.assertRaises(ValueError):
            rig.spawn(profile="replay", sensor_names=["depth"])

    def test_replay_collector_writes_only_rgb(self):
        """Replay-mode Collector (condition_root only, RGB required) cannot write non-RGB files."""
        from src.data.collector import Collector
        from CFG.config import cfg

        class Packet:
            frame = 7
            timestamp = 1.0

            def save_to_disk(self, path):
                with open(path, "wb") as f:
                    f.write(b"png")

        class Rig:
            queues = {"rgb_left": None, "rgb_right": None}

        root = tempfile.mkdtemp()
        try:
            collector = Collector(rig=Rig(), cfg=cfg, condition_root=root,
                                  required_sensors=("rgb_left", "rgb_right"))
            collector.save_frame(0, {"rgb_left": Packet(), "rgb_right": Packet()})
            self.assertEqual(sorted(os.listdir(root)), ["rgb_left", "rgb_right"])
            self.assertEqual(os.listdir(os.path.join(root, "rgb_left")), ["000000.png"])
            self.assertFalse(hasattr(collector, "convert_validation_arrays"))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_production_replay_function_has_no_non_rgb_path(self):
        """Static inspection of replay_condition()."""
        import inspect
        import scripts.collect_dataset as cd

        source = inspect.getsource(cd.replay_condition)
        self.assertIn('profile="replay"', source)
        self.assertIn("REPLAY_SENSOR_PROFILE", source)

        for forbidden in ("np.save", "np.load", "depth_to_numpy", "semantic_to_numpy",
                          "optical_flow_to_numpy", "lidar_to_numpy", "radar_to_numpy",
                          "validate_sensors", "ArrayComparison", "geometry_root=",
                          "save_calibration", "AnnotationWriter", "MetadataWriter"):
            self.assertNotIn(forbidden, source, forbidden)

        import sys as _sys
        old = _sys.argv
        _sys.argv = ["collect_dataset.py"]
        try:
            args = cd.parse_args()
        finally:
            _sys.argv = old
        self.assertFalse(hasattr(args, "replay_validate_sensors"))

    # ---- C: wind ----------------------------------------------------------
    def test_C_every_weather_profile_has_zero_wind(self):
        from src.simulation.weather import WEATHER_WIND_INTENSITY, make_weather
        from CFG.config import cfg

        self.assertEqual(WEATHER_WIND_INTENSITY, 0.0)
        self.assertEqual(sorted(cfg.WEATHER.CONDITIONS), sorted(self.ALL))

        for condition in self.ALL:
            with self.subTest(condition):
                self.assertEqual(make_weather(condition).wind_intensity, 0.0)

    def test_apply_weather_sends_zero_wind_to_the_world(self):
        from src.simulation.weather import apply_weather

        class World:
            applied = None

            def set_weather(self, weather):
                self.applied = weather

        for condition in self.ALL:
            world = World()
            returned = apply_weather(world, condition)
            self.assertEqual(world.applied.wind_intensity, 0.0, condition)
            self.assertEqual(returned.wind_intensity, 0.0, condition)

    # ---- D / G / H: scheduling --------------------------------------------
    def make_route(self, complete_geometry=True, complete_conditions=(), source_complete=True):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        route = os.path.join(root, "Town01", "route_1")
        os.makedirs(geometry_dir(route))
        if complete_geometry:
            mark_complete(geometry_dir(route))
        if source_complete:
            os.makedirs(condition_dir(route, "day_clear"))
            mark_complete(condition_dir(route, "day_clear"))
        for condition in complete_conditions:
            os.makedirs(condition_dir(route, condition), exist_ok=True)
            mark_complete(condition_dir(route, condition))
        return route

    def test_D_day_clear_is_never_replayed(self):
        from src.data.resume_plan import plan_route_work

        # nothing exists yet: canonical run produces day_clear, replay gets the other five
        fresh = os.path.join(tempfile.mkdtemp(), "route_1")
        plan = plan_route_work(fresh, self.ALL, "day_clear")
        self.assertTrue(plan["run_canonical"])
        self.assertNotIn("day_clear", plan["replay"])
        self.assertEqual(plan["replay"], self.ALL[1:])

        # after canonical success (geometry + day_clear COMPLETE): still never replayed
        route = self.make_route()
        plan = plan_route_work(route, self.ALL, "day_clear")
        self.assertFalse(plan["run_canonical"])
        self.assertNotIn("day_clear", plan["replay"])
        self.assertNotIn("day_clear", plan["skip"])

        # ... and it cannot be re-rendered by replay either
        with self.assertRaises(ValueError):
            plan_route_work(route, self.ALL, "day_clear", rerender_conditions=["day_clear"])

    def test_G_resume_schedules_only_the_missing_weather(self):
        from src.data.resume_plan import plan_route_work

        done = [c for c in self.ALL[1:] if c != "day_rain"]
        route = self.make_route(complete_conditions=done)
        plan = plan_route_work(route, self.ALL, "day_clear")

        self.assertFalse(plan["run_canonical"])
        self.assertFalse(plan["delete_route"])
        self.assertEqual(plan["replay"], ["day_rain"])
        self.assertEqual(sorted(plan["skip"]), sorted(done))

    def test_G_partial_geometry_regenerates_everything(self):
        from src.data.resume_plan import plan_route_work

        for kwargs in ({"complete_geometry": False},                        # geometry not COMPLETE
                       {"complete_geometry": True, "source_complete": False}):  # day_clear RGB missing
            with self.subTest(kwargs):
                route = self.make_route(complete_conditions=["day_rain"], **kwargs)
                plan = plan_route_work(route, self.ALL, "day_clear")
                self.assertTrue(plan["run_canonical"])
                self.assertTrue(plan["delete_route"])
                self.assertEqual(plan["replay"], self.ALL[1:])

    def test_H_rerender_touches_only_the_requested_weather(self):
        from src.data.resume_plan import plan_route_work

        route = self.make_route(complete_conditions=self.ALL[1:])
        before = sorted(os.listdir(geometry_dir(route)))
        plan = plan_route_work(route, self.ALL, "day_clear", rerender_conditions=["day_fog"])

        self.assertFalse(plan["run_canonical"])
        self.assertFalse(plan["delete_route"])
        self.assertEqual(plan["replay"], ["day_fog"])
        self.assertEqual(sorted(plan["skip"]), sorted(c for c in self.ALL[1:] if c != "day_fog"))
        self.assertTrue(is_complete(geometry_dir(route)))          # planning never touches geometry
        self.assertEqual(sorted(os.listdir(geometry_dir(route))), before)

        with self.assertRaises(ValueError):
            plan_route_work(route, self.ALL, "day_clear", rerender_conditions=["not_a_weather"])

    def test_overwrite_regenerates_the_whole_route(self):
        from src.data.resume_plan import plan_route_work

        route = self.make_route(complete_conditions=self.ALL[1:])
        plan = plan_route_work(route, self.ALL, "day_clear", overwrite=True)
        self.assertTrue(plan["run_canonical"])
        self.assertTrue(plan["delete_route"])
        self.assertEqual(plan["replay"], self.ALL[1:])

    def test_requested_subset_only_schedules_requested_weathers(self):
        from src.data.resume_plan import plan_route_work

        route = self.make_route()
        plan = plan_route_work(route, ["day_clear", "day_fog"], "day_clear")
        self.assertEqual(plan["replay"], ["day_fog"])

    # ---- I: metadata ------------------------------------------------------
    def test_I_condition_metadata_marks_rgb_only_replay(self):
        import scripts.collect_dataset as cd
        from src.simulation.weather import make_weather

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)

        for condition, rendered_from, expected_rgb_only in (("day_rain", "replay", True),
                                                             ("day_clear", "canonical_run", False)):
            cond_dir = os.path.join(root, condition)
            os.makedirs(cond_dir)
            cd.write_condition_json(cond_dir, condition, "day_clear", make_weather(condition), 5,
                                    rendered_from=rendered_from)
            with open(os.path.join(cond_dir, "condition.json")) as f:
                meta = json.load(f)

            with self.subTest(condition):
                self.assertEqual(meta["rgb_only_replay"], expected_rgb_only)
                self.assertTrue(meta["rendered_from_canonical_geometry"])
                self.assertEqual(meta["wind_intensity"], 0.0)
                self.assertEqual(meta["weather_parameters"]["wind_intensity"], 0.0)
                self.assertEqual(meta["sensors_saved"], ["rgb_left", "rgb_right"])
                self.assertEqual(meta["geometry_source_condition"], "day_clear")

    def test_replay_of_source_condition_is_rejected(self):
        import scripts.collect_dataset as cd

        with self.assertRaises(ValueError):
            cd.replay_condition(None, None, None, {}, "day_clear", "day_clear", "route")


if __name__ == "__main__":
    unittest.main()

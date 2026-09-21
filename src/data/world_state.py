"""
world_state.py

Canonical world-state recording / loading for deterministic weather replay
(pure Python: CARLA objects are only accessed by duck typing, so this
module imports without CARLA).

geometry/
    actors.json              logical_id -> category / blueprint / attributes /
                             canonical_actor_id / first_frame / last_frame
    world_state/000000.json  one file per canonical frame:
        frame_id, carla_frame, timestamp
        ego                  transform / velocity / angular_velocity
        actors               {logical_id: transform / velocity / ...}
        spawned / removed    lifecycle deltas versus the previous frame
        traffic_lights       {light_key: state} for lights near the ego

Actors are keyed by a persistent logical_id, never by a CARLA runtime actor
id (those are not reproducible across respawns).
"""

import json
import math
import os


WORLD_STATE_DIRNAME = "world_state"
ACTORS_FILENAME = "actors.json"

# Traffic lights farther than this from the ego are not recorded (they are
# frozen at their last recorded state during replay).
TRAFFIC_LIGHT_RECORD_RADIUS_M = 150.0

# Actor attributes that change appearance and must be reproduced on replay.
REPLAY_ATTRIBUTES = ("color", "driver_id")


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------

def transform_to_dict(transform):
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "roll": float(transform.rotation.roll),
        "pitch": float(transform.rotation.pitch),
        "yaw": float(transform.rotation.yaw),
    }


def vector_to_dict(vector):
    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z),
    }


def frame_name(frame_id):
    return f"{int(frame_id):06d}"


def traffic_light_key(location):
    return f"{location.x:.1f}_{location.y:.1f}_{location.z:.1f}"


def angle_difference_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pose_error(recorded, live):
    """
    recorded / live: transform dicts (transform_to_dict layout).
    Returns (position_error_m, rotation_error_deg); rotation error is the
    largest wrapped per-axis (roll/pitch/yaw) difference.
    """

    position = math.sqrt(
        (recorded["x"] - live["x"]) ** 2
        + (recorded["y"] - live["y"]) ** 2
        + (recorded["z"] - live["z"]) ** 2
    )

    rotation = max(
        angle_difference_deg(recorded[axis], live[axis])
        for axis in ("roll", "pitch", "yaw")
    )

    return position, rotation


class ErrorAccumulator:
    """Max / mean of position and rotation errors over many samples."""

    def __init__(self):
        self.count = 0
        self.position_sum = 0.0
        self.rotation_sum = 0.0
        self.position_max = 0.0
        self.rotation_max = 0.0

    def add(self, position_error, rotation_error):
        self.count += 1
        self.position_sum += position_error
        self.rotation_sum += rotation_error
        self.position_max = max(self.position_max, position_error)
        self.rotation_max = max(self.rotation_max, rotation_error)

    def summary(self):
        return {
            "samples": self.count,
            "position_max_m": self.position_max,
            "position_mean_m": self.position_sum / self.count if self.count else 0.0,
            "rotation_max_deg": self.rotation_max,
            "rotation_mean_deg": self.rotation_sum / self.count if self.count else 0.0,
        }


# ------------------------------------------------------------------
# Traffic lights
# ------------------------------------------------------------------

class TrafficLightRecorder:
    """
    Caches the map's traffic lights once (their locations do not change) and
    reports the state of those near the ego. Lights are keyed by rounded
    location so replay can match them without depending on actor ids.
    """

    def __init__(self, world, radius_m=TRAFFIC_LIGHT_RECORD_RADIUS_M):
        self.radius_m = float(radius_m)
        self.lights = []

        for actor in world.get_actors().filter("traffic.traffic_light*"):
            location = actor.get_transform().location
            self.lights.append((traffic_light_key(location), actor, location))

    def states_near(self, ego_location):
        states = {}

        for key, actor, location in self.lights:
            if location.distance(ego_location) <= self.radius_m:
                states[key] = str(actor.get_state())

        return states


# ------------------------------------------------------------------
# Recorder (canonical run)
# ------------------------------------------------------------------

class WorldStateRecorder:
    def __init__(self, geometry_root, traffic_light_recorder=None):
        self.geometry_root = os.path.abspath(geometry_root)
        self.state_dir = os.path.join(self.geometry_root, WORLD_STATE_DIRNAME)
        self.actors_path = os.path.join(self.geometry_root, ACTORS_FILENAME)
        self.traffic_light_recorder = traffic_light_recorder

        os.makedirs(self.state_dir, exist_ok=True)

        self.registry = {}
        self._active = {}  # canonical CARLA actor id -> logical id
        self._next_logical_id = 0
        self._previous_logical_ids = set()
        self.num_frames = 0

    def logical_id_for(self, canonical_actor_id):
        return self._active.get(int(canonical_actor_id))

    def _register(self, actor, category, local_frame_id):
        logical_id = self._next_logical_id
        self._next_logical_id += 1

        attributes = {
            key: str(actor.attributes[key])
            for key in REPLAY_ATTRIBUTES
            if key in actor.attributes
        }

        self.registry[logical_id] = {
            "category": category,
            "blueprint_id": actor.type_id,
            "canonical_actor_id": int(actor.id),
            "attributes": attributes,
            "first_frame": int(local_frame_id),
            "last_frame": int(local_frame_id),
        }

        return logical_id

    def _actor_record(self, actor, snapshot, category):
        actor_snapshot = snapshot.find(actor.id)

        if actor_snapshot is None:
            return None

        record = {
            "transform": transform_to_dict(actor_snapshot.get_transform()),
            "velocity": vector_to_dict(actor_snapshot.get_velocity()),
            "angular_velocity": vector_to_dict(actor_snapshot.get_angular_velocity()),
        }

        if actor.type_id.startswith("vehicle."):
            record["light_state"] = int(actor.get_light_state())

        return record

    def record_frame(self, local_frame_id, carla_frame, timestamp, snapshot, ego, managed_actors):
        """
        Record the world as it is right now (after this frame's tick, the
        same instant the sensors and annotations describe). Must be called
        BEFORE the annotation writer so labels can reference logical ids,
        and BEFORE the spawn manager mutates the actor set.
        """

        ego_snapshot = snapshot.find(ego.id)

        if ego_snapshot is None:
            raise RuntimeError(f"Ego actor {ego.id} missing from world snapshot.")

        ego_record = {
            "transform": transform_to_dict(ego_snapshot.get_transform()),
            "velocity": vector_to_dict(ego_snapshot.get_velocity()),
            "angular_velocity": vector_to_dict(ego_snapshot.get_angular_velocity()),
        }

        actors = {}
        current_canonical_ids = set()

        for managed in managed_actors:
            actor = managed["actor"]

            if actor is None or not actor.is_alive:
                continue

            record = self._actor_record(actor, snapshot, managed["category"])

            if record is None:
                continue

            canonical_id = int(actor.id)
            current_canonical_ids.add(canonical_id)

            logical_id = self._active.get(canonical_id)

            if logical_id is None:
                logical_id = self._register(actor, managed["category"], local_frame_id)
                self._active[canonical_id] = logical_id

            self.registry[logical_id]["last_frame"] = int(local_frame_id)
            actors[str(logical_id)] = record

        for canonical_id in list(self._active):
            if canonical_id not in current_canonical_ids:
                del self._active[canonical_id]

        logical_ids = {int(key) for key in actors}
        spawned = sorted(logical_ids - self._previous_logical_ids)
        removed = sorted(self._previous_logical_ids - logical_ids)
        self._previous_logical_ids = logical_ids

        traffic_lights = {}

        if self.traffic_light_recorder is not None:
            traffic_lights = self.traffic_light_recorder.states_near(ego.get_location())

        state = {
            "frame_id": int(local_frame_id),
            "carla_frame": int(carla_frame),
            "timestamp": float(timestamp),
            "ego": ego_record,
            "actors": actors,
            "spawned": spawned,
            "removed": removed,
            "traffic_lights": traffic_lights,
        }

        with open(
            os.path.join(self.state_dir, f"{frame_name(local_frame_id)}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(state, f)

        self.num_frames += 1

        return state

    def write_actors(self):
        with open(self.actors_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.registry.items()}, f, indent=2)

    def flush(self):
        self.write_actors()

    def finalize(self):
        self.write_actors()


# ------------------------------------------------------------------
# Reader (replay)
# ------------------------------------------------------------------

class WorldStateReader:
    def __init__(self, geometry_root):
        self.geometry_root = os.path.abspath(geometry_root)
        self.state_dir = os.path.join(self.geometry_root, WORLD_STATE_DIRNAME)

        if not os.path.isdir(self.state_dir):
            raise FileNotFoundError(f"No world_state directory: {self.state_dir}")

        with open(os.path.join(self.geometry_root, ACTORS_FILENAME), "r", encoding="utf-8") as f:
            self.actors = {int(k): v for k, v in json.load(f).items()}

        self.frame_ids = sorted(
            int(name[:-5]) for name in os.listdir(self.state_dir) if name.endswith(".json")
        )

        if self.frame_ids != list(range(len(self.frame_ids))):
            raise RuntimeError(
                f"world_state frames are not contiguous from 0 in {self.state_dir}"
            )

    def __len__(self):
        return len(self.frame_ids)

    def load_frame(self, frame_id):
        with open(
            os.path.join(self.state_dir, f"{frame_name(frame_id)}.json"),
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

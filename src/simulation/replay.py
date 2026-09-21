"""
replay.py

Deterministic weather replay of a canonical geometry sequence.

Purpose: canonical geometry and GT sensors are collected ONCE; weather replay
regenerates ONLY the stereo RGB pair (rgb_left / rgb_right) under the recorded
geometry. Replay is therefore not a full-sensor reproduction: depth,
semantic, optical flow, LiDAR, radar, labels, pose and calibration are never
recreated or saved here (the replay rig spawns RGB only), so equality of those
modalities between replay and canonical is not a goal.

The recorded world state (src/data/world_state.py) is the source of truth.
Replay never runs BasicAgent / RouteController / Traffic Manager / Gamma
spawning: every frame it

    reconcile actor lifecycle  (spawn newly present, destroy removed)
        -> set ego transform
        -> set every actor transform (+ vehicle lights / walker animation)
        -> apply recorded traffic-light states
        -> [caller ticks + captures sensors]
        -> verify_frame(): post-tick transforms == recorded transforms

Every per-frame mutation (transform, physics flag, walker control, vehicle
lights) is sent as ONE apply_batch_sync() so it is applied atomically with
the next tick. Do not go back to individual actor.set_transform() calls:
those RPCs were measured to occasionally take effect one tick late (the actor
then sits at the previous frame's pose for that tick -- see
scripts/tools/diagnose_replay_fidelity.py); batched commands showed 0 such
events in 30000 samples.

Physics is disabled for the ego and vehicle-like actors so nothing moves
them between the transform being set and the sensors capturing. Pedestrians
keep their character controller (needed for the walking animation) and are
teleported every frame; their post-tick pose is verified like everything
else rather than assumed.
"""

import math

import carla

from src.data.world_state import (
    ErrorAccumulator,
    pose_error,
    traffic_light_key,
    transform_to_dict,
)


DEFAULT_POSITION_TOLERANCE_M = 0.05
DEFAULT_ROTATION_TOLERANCE_DEG = 0.5

# Recorded z is the actor's own resting height; a few centimetres of lift
# avoids spawn-time collisions with the road before set_transform snaps the
# actor to its exact recorded pose.
SPAWN_Z_RETRY_OFFSETS_M = (0.0, 0.1, 0.3, 0.6, 1.0)

WALKER_MIN_SPEED_MPS = 0.05


class ReplayError(RuntimeError):
    pass


def dict_to_transform(data):
    return carla.Transform(
        carla.Location(x=data["x"], y=data["y"], z=data["z"]),
        carla.Rotation(roll=data["roll"], pitch=data["pitch"], yaw=data["yaw"]),
    )


def weather_to_dict(weather):
    fields = (
        "cloudiness", "precipitation", "precipitation_deposits",
        "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
        "fog_density", "fog_distance", "fog_falloff", "wetness",
        "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )

    return {name: float(getattr(weather, name)) for name in fields if hasattr(weather, name)}


class WorldStateReplayer:
    def __init__(
        self,
        world,
        client,
        reader,
        ego_blueprint_id,
        position_tolerance_m=DEFAULT_POSITION_TOLERANCE_M,
        rotation_tolerance_deg=DEFAULT_ROTATION_TOLERANCE_DEG,
    ):
        self.world = world
        self.client = client
        self.reader = reader
        self.ego_blueprint_id = ego_blueprint_id
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)

        self.blueprint_library = world.get_blueprint_library()

        self.ego = None
        self.actors = {}  # logical_id -> replay carla actor

        self._light_actors = {}
        self._applied_light_states = {}
        self._applied_vehicle_lights = {}
        self._ego_physics_off_sent = False

        self.ego_errors = ErrorAccumulator()
        self.vehicle_errors = ErrorAccumulator()
        self.pedestrian_errors = ErrorAccumulator()

        self.worst_errors = []  # (position_m, rotation_deg, frame_id, logical_id, blueprint)
        self.frames_verified = 0
        self.id_mismatch_frames = 0
        self.actors_missing_in_world = 0
        self.frames_with_unmatched_traffic_lights = 0
        self.total_actor_spawns = 0
        self.total_actor_destroys = 0

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def start(self, first_frame_state):
        """Spawn the ego and index / freeze traffic lights."""

        self.world.freeze_all_traffic_lights(True)

        for actor in self.world.get_actors().filter("traffic.traffic_light*"):
            self._light_actors[traffic_light_key(actor.get_transform().location)] = actor

        blueprint = self.blueprint_library.find(self.ego_blueprint_id)

        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        self.ego = self._spawn(blueprint, first_frame_state["ego"]["transform"])

        if self.ego is None:
            raise ReplayError("Failed to spawn replay ego at its recorded frame-0 pose.")

        # Pose and physics flag are (re)applied atomically by the first
        # apply_frame() batch.
        self.ego.set_simulate_physics(False)

    def _spawn(self, blueprint, transform_dict):
        for z_offset in SPAWN_Z_RETRY_OFFSETS_M:
            transform = dict_to_transform(transform_dict)
            transform.location.z += z_offset

            actor = self.world.try_spawn_actor(blueprint, transform)

            if actor is not None:
                return actor

        return None

    def _spawn_replay_actor(self, logical_id, transform_dict):
        info = self.reader.actors[logical_id]
        blueprint = self.blueprint_library.find(info["blueprint_id"])

        for name, value in info.get("attributes", {}).items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, value)

        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")

        actor = self._spawn(blueprint, transform_dict)

        if actor is None:
            raise ReplayError(
                f"Failed to spawn replay actor logical_id={logical_id} "
                f"({info['blueprint_id']}) at its recorded pose."
            )

        self.total_actor_spawns += 1

        return actor

    # ------------------------------------------------------------
    # Per-frame application (call BEFORE world.tick())
    # ------------------------------------------------------------

    def apply_frame(self, frame_state):
        recorded = frame_state["actors"]
        recorded_ids = {int(key) for key in recorded}
        commands = []

        for logical_id in [lid for lid in self.actors if lid not in recorded_ids]:
            self._destroy_actor(logical_id)

        for logical_id in sorted(recorded_ids - set(self.actors)):
            actor = self._spawn_replay_actor(
                logical_id, recorded[str(logical_id)]["transform"],
            )
            self.actors[logical_id] = actor

            if not self._is_walker(logical_id):
                commands.append(carla.command.SetSimulatePhysics(actor.id, False))

        if not self._ego_physics_off_sent:
            commands.append(carla.command.SetSimulatePhysics(self.ego.id, False))
            self._ego_physics_off_sent = True

        commands.append(
            carla.command.ApplyTransform(self.ego.id, dict_to_transform(frame_state["ego"]["transform"]))
        )

        for logical_id, actor in self.actors.items():
            record = recorded[str(logical_id)]

            if self._is_walker(logical_id):
                commands.append(carla.command.ApplyWalkerControl(actor.id, self._walker_control(record)))
            elif "light_state" in record:
                commands.extend(self._vehicle_light_commands(logical_id, actor, record["light_state"]))

            commands.append(carla.command.ApplyTransform(actor.id, dict_to_transform(record["transform"])))

        self._apply_batch(commands)

        # carla.command has no traffic-light state command, so lights go
        # through individual RPCs (state changes are rare; worst case a light
        # colour is one tick late -- it is not scene geometry).
        self._apply_traffic_lights(frame_state.get("traffic_lights", {}))

    def _is_walker(self, logical_id):
        return self.reader.actors[logical_id]["blueprint_id"].startswith("walker.")

    def _apply_batch(self, commands):
        for response in self.client.apply_batch_sync(commands, False):
            if response.error:
                raise ReplayError(f"CARLA command failed during replay: {response.error}")

    def _walker_control(self, record):
        velocity = record["velocity"]
        speed = math.hypot(velocity["x"], velocity["y"])

        if speed > WALKER_MIN_SPEED_MPS:
            direction = carla.Vector3D(velocity["x"] / speed, velocity["y"] / speed, 0.0)
        else:
            direction = carla.Vector3D(1.0, 0.0, 0.0)
            speed = 0.0

        return carla.WalkerControl(direction=direction, speed=speed)

    def _vehicle_light_commands(self, logical_id, actor, light_state):
        if self._applied_vehicle_lights.get(logical_id) == light_state:
            return []

        self._applied_vehicle_lights[logical_id] = light_state

        return [carla.command.SetVehicleLightState(actor.id, carla.VehicleLightState(light_state))]

    def _apply_traffic_lights(self, states):
        unmatched = False

        for key, state_name in states.items():
            light = self._light_actors.get(key)

            if light is None:
                unmatched = True
                continue

            if self._applied_light_states.get(key) != state_name:
                light.set_state(getattr(carla.TrafficLightState, state_name))
                self._applied_light_states[key] = state_name

        if unmatched:
            self.frames_with_unmatched_traffic_lights += 1

    def _destroy_actor(self, logical_id):
        actor = self.actors.pop(logical_id)
        self._applied_vehicle_lights.pop(logical_id, None)

        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass

        self.total_actor_destroys += 1

    # ------------------------------------------------------------
    # Verification (call AFTER world.tick())
    # ------------------------------------------------------------

    def verify_frame(self, frame_state):
        snapshot = self.world.get_snapshot()
        recorded = frame_state["actors"]

        if set(self.actors) != {int(key) for key in recorded}:
            self.id_mismatch_frames += 1

        ego_snapshot = snapshot.find(self.ego.id)

        if ego_snapshot is None:
            raise ReplayError("Replay ego missing from world snapshot.")

        ego_position, ego_rotation = pose_error(
            frame_state["ego"]["transform"], transform_to_dict(ego_snapshot.get_transform()),
        )
        self.ego_errors.add(ego_position, ego_rotation)

        for logical_id, actor in self.actors.items():
            actor_snapshot = snapshot.find(actor.id)

            if actor_snapshot is None:
                self.actors_missing_in_world += 1
                continue

            position, rotation = pose_error(
                recorded[str(logical_id)]["transform"],
                transform_to_dict(actor_snapshot.get_transform()),
            )

            if position > self.position_tolerance_m or rotation > self.rotation_tolerance_deg:
                self.worst_errors.append((
                    position, rotation, frame_state["frame_id"], logical_id,
                    self.reader.actors[logical_id]["blueprint_id"],
                ))

            if self.reader.actors[logical_id]["blueprint_id"].startswith("walker."):
                self.pedestrian_errors.add(position, rotation)
            else:
                self.vehicle_errors.add(position, rotation)

        self.frames_verified += 1

    def summary(self):
        tolerance_position = self.position_tolerance_m
        tolerance_rotation = self.rotation_tolerance_deg

        groups = {
            "ego": self.ego_errors,
            "vehicle_like_actors": self.vehicle_errors,
            "pedestrians": self.pedestrian_errors,
        }

        result = {
            "frames_verified": self.frames_verified,
            "position_tolerance_m": tolerance_position,
            "rotation_tolerance_deg": tolerance_rotation,
            "actor_id_set_mismatch_frames": self.id_mismatch_frames,
            "actors_missing_in_world": self.actors_missing_in_world,
            "frames_with_unmatched_traffic_lights": self.frames_with_unmatched_traffic_lights,
            "total_actor_spawns": self.total_actor_spawns,
            "total_actor_destroys": self.total_actor_destroys,
        }

        passed = (
            self.frames_verified > 0
            and self.id_mismatch_frames == 0
            and self.actors_missing_in_world == 0
        )

        for name, accumulator in groups.items():
            stats = accumulator.summary()
            group_passed = (
                stats["position_max_m"] <= tolerance_position
                and stats["rotation_max_deg"] <= tolerance_rotation
            )
            stats["passed"] = group_passed
            result[name] = stats
            passed = passed and group_passed

        result["passed"] = passed
        result["out_of_tolerance_samples"] = len(self.worst_errors)
        result["worst_out_of_tolerance"] = [
            {"position_m": p, "rotation_deg": r, "frame_id": f, "logical_id": l, "blueprint": b}
            for p, r, f, l, b in sorted(self.worst_errors, reverse=True)[:10]
        ]

        return result

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    def destroy_all(self):
        for logical_id in list(self.actors):
            try:
                self._destroy_actor(logical_id)
            except Exception:
                pass

        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    self.ego.destroy()
            except RuntimeError:
                pass

            self.ego = None

        try:
            self.world.freeze_all_traffic_lights(False)
        except RuntimeError:
            pass

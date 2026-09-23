import os
import csv
import json
import math

from src.simulation.timing import record_stride_ticks


class MetadataWriter:
    def __init__(self, sequence_root, map_name, sequence_id, cfg, route_id=None, condition=None, spawn_info=None, sequence_extra=None):
        """
        condition: legacy per-weather sequences only. Canonical geometry
        sequences leave it None and describe their weather relationship via
        sequence_extra (town, geometry_source_condition, conditions, ...),
        which is merged into sequence.json as top-level fields.
        """

        self.sequence_root = os.path.abspath(sequence_root)

        self.map_name = map_name
        self.sequence_id = sequence_id
        self.route_id = route_id
        self.condition = condition
        self.cfg = cfg
        
        self.spawn_info = spawn_info
        self.sequence_extra = sequence_extra

        self.num_frames = 0

        os.makedirs(self.sequence_root, exist_ok=True)

        self.pose_dir = os.path.join(self.sequence_root, "pose",)

        os.makedirs(self.pose_dir, exist_ok=True)
        self.timestamps_path = os.path.join(self.sequence_root, "timestamps.csv")

        self.pose_path = os.path.join(self.pose_dir, "poses.csv")

        self.ego_state_path = os.path.join(self.sequence_root,
            "ego_state.csv",
        )

        self.sequence_path = os.path.join(
            self.sequence_root,
            "sequence.json",
        )

        self.timestamps_file = open(
            self.timestamps_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.pose_file = open(
            self.pose_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.ego_state_file = open(
            self.ego_state_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.timestamps_writer = csv.writer(
            self.timestamps_file
        )

        self.pose_writer = csv.writer(
            self.pose_file
        )

        self.ego_state_writer = csv.writer(
            self.ego_state_file
        )

        self._write_headers()
        self._write_sequence_json()

    def _write_headers(self):
        self.timestamps_writer.writerow(["frame_id", "carla_frame", "timestamp"])

        self.pose_writer.writerow(
            [
                "frame_id",
                "carla_frame",
                "timestamp",

                "x",
                "y",
                "z",

                "roll_deg",
                "pitch_deg",
                "yaw_deg",
            ]
        )

        self.ego_state_writer.writerow(
            [
                "frame_id",
                "carla_frame",
                "timestamp",

                "velocity_x",
                "velocity_y",
                "velocity_z",
                "speed_mps",

                "acceleration_x",
                "acceleration_y",
                "acceleration_z",

                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",

                "throttle",
                "steer",
                "brake",

                "hand_brake",
                "reverse",
                "gear",

                "route_index",
                "route_progress",
                "goal_distance",
            ]
        )


    def _build_spawn_policy_record(self):
        spawn = self.cfg.SPAWN

        record = {
            "spawn_seed": spawn.SEED,

            # Category composition weights: the initial scene target is
            # split across categories in this ratio (not absolute counts).
            "category_weights": {
                "vehicle": spawn.N_VEHICLES,
                "motorcyclist": spawn.N_MOTORCYCLES,
                "cyclist": spawn.N_BICYCLES,
                "pedestrian": spawn.N_PEDESTRIANS,
            },

            "frame_object_gamma": {
                "shape": spawn.FRAME_OBJECT_GAMMA_SHAPE,
                "scale": spawn.FRAME_OBJECT_GAMMA_SCALE,
                "min": spawn.FRAME_OBJECT_MIN,
                "max": spawn.FRAME_OBJECT_MAX,
                "target_interval_min_frames": spawn.FRAME_OBJECT_TARGET_INTERVAL_MIN,
                "target_interval_max_frames": spawn.FRAME_OBJECT_TARGET_INTERVAL_MAX,
                "max_new_per_update": spawn.FRAME_OBJECT_MAX_NEW_PER_UPDATE,
                "max_prune_per_update": spawn.FRAME_OBJECT_MAX_PRUNE_PER_UPDATE,
            },
        }

        if self.spawn_info:
            record.update(self.spawn_info)

        return record

    def _write_sequence_json(self):

        data = {
            "map": self.map_name,
            "sequence_id": self.sequence_id,
            "route_id": self.route_id,

            "fps": self.cfg.SIMULATION.FPS,

            "fixed_delta_seconds":
                self.cfg.SIMULATION.FIXED_DELTA_SECONDS,

            # 10 Hz-recording task: world/controller/traffic stay at
            # simulation_hz; only the saved dataset sample rate is
            # recording_hz. See src/simulation/timing.py.
            "simulation_hz": float(self.cfg.SIMULATION.FPS),
            "recording_hz": float(self.cfg.RECORDING.FPS),
            "recording_interval_seconds": self.cfg.RECORDING.SAMPLE_INTERVAL_SECONDS,
            "recording_stride_ticks": record_stride_ticks(self.cfg),

            "random_seed":
                self.cfg.RANDOM.SEED,

            "traffic_seed":
                self.cfg.TRAFFIC.SEED,

            "pedestrian_seed":
                self.cfg.PEDESTRIAN.SEED,

            "spawn_policy":
                self._build_spawn_policy_record(),

            "num_frames":
                self.num_frames,

            "coordinate_system": {
                "name": "CARLA",
                "x": "forward",
                "y": "right",
                "z": "up",
                "angles": "degrees",
            },
        }

        if self.condition is not None:
            data["condition"] = self.condition

        if self.sequence_extra:
            data.update(self.sequence_extra)

        with open(self.sequence_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def write_frame(self, frame_id, carla_frame, timestamp, ego, route_status=None,):
        frame_name = (f"{int(frame_id):06d}")
        transform = (ego.get_transform())
        velocity = (ego.get_velocity())
        acceleration = (ego.get_acceleration())
        angular_velocity = (ego.get_angular_velocity())
        control = (ego.get_control())

        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        if route_status is None:
            route_index = None
            route_progress = None
            goal_distance = None

        else:
            route_index = route_status.get("route_index")
            route_progress = route_status.get("progress")
            goal_distance = route_status.get("goal_distance")

        self.timestamps_writer.writerow([frame_name, carla_frame, timestamp])

        self.pose_writer.writerow(
            [
                frame_name,
                carla_frame,
                timestamp,

                transform.location.x,
                transform.location.y,
                transform.location.z,

                transform.rotation.roll,
                transform.rotation.pitch,
                transform.rotation.yaw,
            ]
        )

        self.ego_state_writer.writerow(
            [
                frame_name,
                carla_frame,
                timestamp,

                velocity.x,
                velocity.y,
                velocity.z,
                speed_mps,

                acceleration.x,
                acceleration.y,
                acceleration.z,

                angular_velocity.x,
                angular_velocity.y,
                angular_velocity.z,

                control.throttle,
                control.steer,
                control.brake,

                int(control.hand_brake),
                int(control.reverse),
                control.gear,

                route_index,
                route_progress,
                goal_distance,
            ]
        )

        self.num_frames += 1

    def flush(self):

        if (self.timestamps_file is not None and not self.timestamps_file.closed):
            self.timestamps_file.flush()

        if (self.pose_file is not None and not self.pose_file.closed):
            self.pose_file.flush()

        if (self.ego_state_file is not None and not self.ego_state_file.closed):
            self.ego_state_file.flush()

    def finalize(self):
        self.flush()
        self._write_sequence_json()
        self.close()


    def close(self):

        if (self.timestamps_file is not None and not self.timestamps_file.closed):
            self.timestamps_file.close()

        if (self.pose_file is not None and not self.pose_file.closed):
            self.pose_file.close()

        if (self.ego_state_file is not None and not self.ego_state_file.closed):
            self.ego_state_file.close()



    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.finalize()
        else:
            try:
                self._write_sequence_json()
            finally:
                self.close()
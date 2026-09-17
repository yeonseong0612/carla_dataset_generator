import os
import csv
import json
import math


class MetadataWriter:
    """
    Write sequence-level and frame-level metadata.

    Expected sequence structure:

    sequence_root/
    ├─ calibration.json
    ├─ sequence.json
    ├─ timestamps.csv
    ├─ pose/
    │  └─ poses.csv
    ├─ ego_state.csv
    ├─ rgb_left/
    ├─ rgb_right/
    ├─ depth/
    ├─ optical_flow/
    ├─ semantic/
    ├─ lidar/
    ├─ radar/
    └─ navigation/
    """

    def __init__(
        self,
        sequence_root,
        map_name,
        sequence_id,
        cfg,
        route_id=None,
        condition=None,
    ):
        self.sequence_root = os.path.abspath(
            sequence_root
        )

        self.map_name = map_name
        self.sequence_id = sequence_id
        self.route_id = route_id
        self.condition = condition
        self.cfg = cfg

        self.num_frames = 0

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        os.makedirs(
            self.sequence_root,
            exist_ok=True,
        )

        self.pose_dir = os.path.join(
            self.sequence_root,
            "pose",
        )

        os.makedirs(
            self.pose_dir,
            exist_ok=True,
        )

        self.timestamps_path = os.path.join(
            self.sequence_root,
            "timestamps.csv",
        )

        self.pose_path = os.path.join(
            self.pose_dir,
            "poses.csv",
        )

        self.ego_state_path = os.path.join(
            self.sequence_root,
            "ego_state.csv",
        )

        self.sequence_path = os.path.join(
            self.sequence_root,
            "sequence.json",
        )

        # ----------------------------------------------------
        # Open CSV files
        # ----------------------------------------------------

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


    # ========================================================
    # Headers
    # ========================================================

    def _write_headers(self):

        self.timestamps_writer.writerow(
            [
                "frame_id",
                "carla_frame",
                "timestamp",
            ]
        )

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


    # ========================================================
    # Sequence metadata
    # ========================================================

    def _write_sequence_json(self):

        data = {
            "map": self.map_name,
            "sequence_id": self.sequence_id,
            "route_id": self.route_id,
            "condition": self.condition,

            "fps": self.cfg.SIMULATION.FPS,

            "fixed_delta_seconds":
                self.cfg.SIMULATION.FIXED_DELTA_SECONDS,

            "random_seed":
                self.cfg.RANDOM.SEED,

            "traffic_seed":
                self.cfg.TRAFFIC.SEED,

            "pedestrian_seed":
                self.cfg.PEDESTRIAN.SEED,

            "num_vehicles":
                self.cfg.TRAFFIC.NUM_VEHICLES,

            "num_cyclists":
                self.cfg.TRAFFIC.NUM_CYCLISTS,

            "num_motorcyclists":
                self.cfg.TRAFFIC.NUM_MOTORCYCLISTS,

            "num_pedestrians":
                self.cfg.PEDESTRIAN.NUM_WALKERS,

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

        with open(
            self.sequence_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
            )


    # ========================================================
    # Per-frame metadata
    # ========================================================

    def write_frame(
        self,
        frame_id,
        carla_frame,
        timestamp,
        ego,
        route_status=None,
    ):
        """
        Write metadata for one synchronized CARLA frame.

        route_status can be obtained from:

            controller.get_status()

        Expected optional fields:
            route_index
            progress
            goal_distance
        """

        frame_name = (
            f"{int(frame_id):06d}"
        )

        transform = (
            ego.get_transform()
        )

        velocity = (
            ego.get_velocity()
        )

        acceleration = (
            ego.get_acceleration()
        )

        angular_velocity = (
            ego.get_angular_velocity()
        )

        control = (
            ego.get_control()
        )

        # CARLA velocity is m/s.
        speed_mps = math.sqrt(
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        )

        # ----------------------------------------------------
        # Route metadata
        # ----------------------------------------------------

        if route_status is None:

            route_index = None
            route_progress = None
            goal_distance = None

        else:

            route_index = route_status.get(
                "route_index"
            )

            route_progress = route_status.get(
                "progress"
            )

            goal_distance = route_status.get(
                "goal_distance"
            )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        self.timestamps_writer.writerow(
            [
                frame_name,
                carla_frame,
                timestamp,
            ]
        )

        # ----------------------------------------------------
        # Ego pose
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Ego state
        # ----------------------------------------------------

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


    # ========================================================
    # Flush
    # ========================================================

    def flush(self):

        if (
            self.timestamps_file is not None
            and not self.timestamps_file.closed
        ):
            self.timestamps_file.flush()

        if (
            self.pose_file is not None
            and not self.pose_file.closed
        ):
            self.pose_file.flush()

        if (
            self.ego_state_file is not None
            and not self.ego_state_file.closed
        ):
            self.ego_state_file.flush()


    # ========================================================
    # Finalize
    # ========================================================

    def finalize(self):
        """
        Update sequence.json with final frame count
        and close all files.
        """

        self.flush()

        self._write_sequence_json()

        self.close()


    # ========================================================
    # Close
    # ========================================================

    def close(self):

        if (
            self.timestamps_file is not None
            and not self.timestamps_file.closed
        ):
            self.timestamps_file.close()

        if (
            self.pose_file is not None
            and not self.pose_file.closed
        ):
            self.pose_file.close()

        if (
            self.ego_state_file is not None
            and not self.ego_state_file.closed
        ):
            self.ego_state_file.close()


    # ========================================================
    # Context manager
    # ========================================================

    def __enter__(self):
        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        # If collection finished normally,
        # write final num_frames.
        if exc_type is None:
            self.finalize()
        else:
            # Even on failure preserve the number
            # of successfully written frames.
            try:
                self._write_sequence_json()
            finally:
                self.close()
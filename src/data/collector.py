"""
collector.py

Synchronized sensor collection and dataset writing for CARLA.

Responsibilities
----------------
1. Wait for all sensors to produce the requested CARLA frame.
2. Convert raw CARLA sensor data to NumPy arrays where necessary.
3. Save synchronized sensor data to disk.
4. Store GNSS / IMU measurements in CSV files.

Expected sequence structure
---------------------------
sequence_root/
├─ rgb_left/
├─ rgb_right/
├─ depth/
├─ optical_flow/
├─ semantic/
├─ lidar/
├─ radar/
├─ radar_front_left/
├─ radar_front_right/
└─ navigation/
   ├─ gnss.csv
   └─ imu.csv

The sequence_root itself should already represent one unique
Town / Route / Weather sequence.

Example:
dataset/Town01/route_00/day_clear/
"""

import csv
import os
import queue
import time

import numpy as np

from src.sensors.lidar import create_lidar_transform

# ============================================================
# Synchronization
# ============================================================

def wait_for_frame(
    sensor_queue,
    target_frame,
    timeout=10.0,
    sensor_name="unknown",
):
    """
    Wait until a sensor produces exactly target_frame.

    Older frames are discarded.

    If the sensor jumps past target_frame, collection fails
    because strict frame synchronization has been violated.

    Parameters
    ----------
    sensor_queue:
        queue.Queue receiving CARLA sensor data.

    target_frame:
        Required CARLA frame number.

    timeout:
        Maximum total waiting time in seconds.

    sensor_name:
        Sensor identifier used in error messages.

    Returns
    -------
    CARLA sensor data corresponding exactly to target_frame.
    """

    deadline = time.monotonic() + timeout

    while True:

        remaining = deadline - time.monotonic()

        if remaining <= 0.0:
            raise RuntimeError(
                f"Timeout waiting for sensor "
                f"'{sensor_name}' at CARLA frame "
                f"{target_frame}."
            )

        try:
            data = sensor_queue.get(
                timeout=remaining
            )

        except queue.Empty as exc:
            raise RuntimeError(
                f"Timeout waiting for sensor "
                f"'{sensor_name}' at CARLA frame "
                f"{target_frame}."
            ) from exc

        # Discard stale frames.
        if data.frame < target_frame:
            continue

        # Exact synchronization.
        if data.frame == target_frame:
            return data

        # Sensor skipped the requested frame.
        raise RuntimeError(
            f"Sensor '{sensor_name}' skipped "
            f"CARLA frame {target_frame}. "
            f"Received frame {data.frame}."
        )


# ============================================================
# Sensor conversion
# ============================================================

def depth_to_numpy(data):
    """
    Convert CARLA raw depth image into depth in meters.

    CARLA encodes normalized depth into 24 bits:

        normalized =
            (R + G * 256 + B * 256^2)
            / (256^3 - 1)

        depth_m = normalized * 1000
    """

    image = np.frombuffer(
        data.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            data.height,
            data.width,
            4,
        )
    )

    # CARLA raw buffer is BGRA.
    b = image[:, :, 0].astype(
        np.float32
    )

    g = image[:, :, 1].astype(
        np.float32
    )

    r = image[:, :, 2].astype(
        np.float32
    )

    normalized = (
        r
        + g * 256.0
        + b * 65536.0
    ) / 16777215.0

    depth = normalized * 1000.0

    return depth.astype(
        np.float32,
        copy=False,
    )


def optical_flow_to_numpy(data):
    """
    Convert CARLA optical flow measurement into
    an H x W x 2 float32 array.
    """

    flow = np.frombuffer(
        data.raw_data,
        dtype=np.float32,
    ).reshape(
        (
            data.height,
            data.width,
            2,
        )
    )

    return flow.copy()


def semantic_to_numpy(data):
    """
    Convert CARLA raw semantic segmentation image
    into an H x W uint8 class-ID map.

    CARLA raw semantic segmentation stores the
    semantic tag in the red channel of the BGRA image.
    """

    image = np.frombuffer(
        data.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            data.height,
            data.width,
            4,
        )
    )

    # BGRA buffer:
    # channel 2 = R = semantic class/tag ID
    semantic = image[:, :, 2]

    return semantic.copy()


def lidar_to_numpy(data):
    """
    Convert CARLA LiDAR measurement into N x 4:

        [x, y, z, intensity]
    """

    points = np.frombuffer(
        data.raw_data,
        dtype=np.float32,
    ).reshape(
        (-1, 4)
    )

    return points.copy()

def filter_lidar_roi(
    points,
    T_ego_from_lidar,
    cfg,
):
    """
    Filter LiDAR points using an ego-frame rectangular ROI.

    ROI:
        x_ego: [ROI_FRONT_MIN, ROI_FRONT_MAX]
        y_ego: [-ROI_SIDE, +ROI_SIDE]

    Notes
    -----
    - Input/output points remain in the LiDAR sensor frame:
        [x_lidar, y_lidar, z_lidar, intensity]

    - Ego coordinates are used only for ROI selection.
    """

    if len(points) == 0:
        return points

    # --------------------------------------------------------
    # LiDAR local -> Ego
    # --------------------------------------------------------

    xyz_lidar = points[:, :3].astype(
        np.float64,
        copy=False,
    )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float64,
    )

    xyz1_lidar = np.concatenate(
        [xyz_lidar, ones],
        axis=1,
    )

    xyz1_ego = (
        T_ego_from_lidar
        @ xyz1_lidar.T
    ).T

    xyz_ego = xyz1_ego[:, :3]

    # --------------------------------------------------------
    # Rectangular ROI in ego frame
    # --------------------------------------------------------

    x_ego = xyz_ego[:, 0]
    y_ego = xyz_ego[:, 1]

    mask = (
        (x_ego >= cfg.SENSOR.LIDAR.ROI_FRONT_MIN)
        & (x_ego <= cfg.SENSOR.LIDAR.ROI_FRONT_MAX)
        & (np.abs(y_ego) <= cfg.SENSOR.LIDAR.ROI_SIDE)
    )

    # Keep original LiDAR-frame coordinates.
    return points[mask]


def radar_to_numpy(data):
    """
    Convert CARLA radar detections into N x 4:

        [x, y, z, radial_velocity]

    Radar azimuth and altitude are expressed in radians.
    """

    points = []

    for detection in data:

        depth = detection.depth
        azimuth = detection.azimuth
        altitude = detection.altitude
        velocity = detection.velocity

        x = (
            depth
            * np.cos(altitude)
            * np.cos(azimuth)
        )

        y = (
            depth
            * np.cos(altitude)
            * np.sin(azimuth)
        )

        z = (
            depth
            * np.sin(altitude)
        )

        points.append(
            [
                x,
                y,
                z,
                velocity,
            ]
        )

    return np.asarray(
        points,
        dtype=np.float32,
    ).reshape(
        -1,
        4,
    )


# ============================================================
# Collector
# ============================================================

class Collector:
    """
    Synchronized CARLA sensor collector.

    Parameters
    ----------
    rig:
        SensorRig instance.

        Expected queue names:

            rgb_left
            rgb_right
            depth
            optical_flow
            semantic
            lidar
            radar
            radar_front_left
            radar_front_right
            gnss
            imu

    sequence_root:
        Output directory representing one complete sequence.

        Example:

            dataset/
                Town01/
                    route_00/
                        day_clear/

    timeout:
        Maximum waiting time for each synchronized frame.
    """

    RADAR_SENSORS = (
        "radar",
        "radar_front_left",
        "radar_front_right",
    )

    REQUIRED_SENSORS = (
        "rgb_left",
        "rgb_right",
        "depth",
        "optical_flow",
        "semantic",
        "lidar",
    ) + RADAR_SENSORS + (
        "gnss",
        "imu",
    )

    def __init__(
        self,
        rig,
        sequence_root,
        cfg,
        timeout=10.0,
    ):
        self.rig = rig
        self.cfg = cfg

        self.sequence_root = os.path.abspath(
            sequence_root
        )

        self.timeout = float(
            timeout
        )

        lidar_transform = create_lidar_transform(
            cfg
        )

        self.T_ego_from_lidar = np.array(
            lidar_transform.get_matrix(),
            dtype=np.float64,
        )

        self.dirs = {}

        self.gnss_file = None
        self.imu_file = None

        self.gnss_writer = None
        self.imu_writer = None

        self._validate_rig()
        self._prepare_output()


    # ========================================================
    # Initialization
    # ========================================================

    def _validate_rig(self):
        """
        Verify that all required sensor queues exist.
        """

        missing = [
            name
            for name in self.REQUIRED_SENSORS
            if name not in self.rig.queues
        ]

        if missing:
            raise ValueError(
                "SensorRig is missing required "
                f"sensor queues: {missing}"
            )


    def _prepare_output(self):
        """
        Create output directories and navigation CSV files.

        sequence_root already represents one weather-specific
        sequence, therefore condition-specific subdirectories
        are not created here.
        """

        os.makedirs(
            self.sequence_root,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Sensor directories
        # ----------------------------------------------------

        sensor_dirs = (
            "rgb_left",
            "rgb_right",
            "depth",
            "optical_flow",
            "semantic",
            "lidar",
        ) + self.RADAR_SENSORS

        for name in sensor_dirs:

            path = os.path.join(
                self.sequence_root,
                name,
            )

            os.makedirs(
                path,
                exist_ok=True,
            )

            self.dirs[name] = path

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        navigation_dir = os.path.join(
            self.sequence_root,
            "navigation",
        )

        os.makedirs(
            navigation_dir,
            exist_ok=True,
        )

        self.dirs["navigation"] = (
            navigation_dir
        )

        # ----------------------------------------------------
        # GNSS
        # ----------------------------------------------------

        self.gnss_file = open(
            os.path.join(
                navigation_dir,
                "gnss.csv",
            ),
            "w",
            newline="",
            encoding="utf-8",
        )

        self.gnss_writer = csv.writer(
            self.gnss_file
        )

        self.gnss_writer.writerow(
            [
                "frame_id",
                "carla_frame",
                "timestamp",
                "latitude",
                "longitude",
                "altitude",
            ]
        )

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        self.imu_file = open(
            os.path.join(
                navigation_dir,
                "imu.csv",
            ),
            "w",
            newline="",
            encoding="utf-8",
        )

        self.imu_writer = csv.writer(
            self.imu_file
        )

        self.imu_writer.writerow(
            [
                "frame_id",
                "carla_frame",
                "timestamp",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "compass",
            ]
        )


    # ========================================================
    # Collection
    # ========================================================

    def collect_frame(
        self,
        carla_frame,
    ):
        """
        Collect exactly one synchronized packet.

        All required sensors must produce the same
        CARLA frame.
        """

        packet = {}

        for name in self.REQUIRED_SENSORS:

            sensor_queue = (
                self.rig.queues[name]
            )

            packet[name] = (
                wait_for_frame(
                    sensor_queue=sensor_queue,
                    target_frame=carla_frame,
                    timeout=self.timeout,
                    sensor_name=name,
                )
            )

        # ----------------------------------------------------
        # Defensive synchronization verification
        # ----------------------------------------------------

        frames = {
            name: data.frame
            for name, data in packet.items()
        }

        unique_frames = set(
            frames.values()
        )

        if len(unique_frames) != 1:

            raise RuntimeError(
                "Sensor frame mismatch detected: "
                f"{frames}"
            )

        actual_frame = next(
            iter(unique_frames)
        )

        if actual_frame != carla_frame:

            raise RuntimeError(
                f"Expected CARLA frame "
                f"{carla_frame}, "
                f"but synchronized packet is "
                f"frame {actual_frame}."
            )

        return packet


    # ========================================================
    # Saving
    # ========================================================

    def save_frame(
        self,
        local_frame_id,
        packet,
    ):
        """
        Save one synchronized sensor packet.
        """

        frame_name = (
            f"{int(local_frame_id):06d}"
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        packet["rgb_left"].save_to_disk(
            os.path.join(
                self.dirs["rgb_left"],
                f"{frame_name}.png",
            )
        )

        packet["rgb_right"].save_to_disk(
            os.path.join(
                self.dirs["rgb_right"],
                f"{frame_name}.png",
            )
        )

        # ----------------------------------------------------
        # Geometry / GT conversion
        # ----------------------------------------------------

        depth = depth_to_numpy(
            packet["depth"]
        )

        flow = optical_flow_to_numpy(
            packet["optical_flow"]
        )

        semantic = semantic_to_numpy(
            packet["semantic"]
        )

        lidar_raw = lidar_to_numpy(
            packet["lidar"]
        )

        lidar = filter_lidar_roi(
            lidar_raw,
            self.T_ego_from_lidar,
            self.cfg,
        )

        radar_arrays = {
            name: radar_to_numpy(packet[name])
            for name in self.RADAR_SENSORS
        }

        # ----------------------------------------------------
        # Save arrays
        # ----------------------------------------------------

        np.save(
            os.path.join(
                self.dirs["depth"],
                f"{frame_name}.npy",
            ),
            depth,
        )

        np.save(
            os.path.join(
                self.dirs["optical_flow"],
                f"{frame_name}.npy",
            ),
            flow,
        )

        np.save(
            os.path.join(
                self.dirs["semantic"],
                f"{frame_name}.npy",
            ),
            semantic,
        )

        np.save(
            os.path.join(
                self.dirs["lidar"],
                f"{frame_name}.npy",
            ),
            lidar,
        )

        for name, radar_array in radar_arrays.items():
            np.save(
                os.path.join(
                    self.dirs[name],
                    f"{frame_name}.npy",
                ),
                radar_array,
            )

        # ----------------------------------------------------
        # GNSS
        # ----------------------------------------------------

        gnss = packet["gnss"]

        self.gnss_writer.writerow(
            [
                local_frame_id,
                gnss.frame,
                gnss.timestamp,
                gnss.latitude,
                gnss.longitude,
                gnss.altitude,
            ]
        )

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        imu = packet["imu"]

        self.imu_writer.writerow(
            [
                local_frame_id,
                imu.frame,
                imu.timestamp,

                imu.accelerometer.x,
                imu.accelerometer.y,
                imu.accelerometer.z,

                imu.gyroscope.x,
                imu.gyroscope.y,
                imu.gyroscope.z,

                imu.compass,
            ]
        )

        # ----------------------------------------------------
        # Result summary
        # ----------------------------------------------------

        return {
            "frame_id": int(
                local_frame_id
            ),

            "carla_frame": int(
                packet["rgb_left"].frame
            ),

            "timestamp": float(
                packet["rgb_left"].timestamp
            ),

            "depth_shape": tuple(
                depth.shape
            ),

            "flow_shape": tuple(
                flow.shape
            ),

            "semantic_shape": tuple(
                semantic.shape
            ),

            "lidar_points_raw": int(
                len(lidar_raw)
            ),

            "lidar_points": int(
                len(lidar)
            ),

            # Kept for backward compatibility: the front radar's count
            # (radar_arrays["radar"]), same meaning as before this sensor
            # was joined by front-left/front-right corner radars.
            "radar_points": int(
                len(radar_arrays["radar"])
            ),

            "radar_front_left_points": int(
                len(radar_arrays["radar_front_left"])
            ),

            "radar_front_right_points": int(
                len(radar_arrays["radar_front_right"])
            ),

            "radar_points_merged": int(
                sum(len(array) for array in radar_arrays.values())
            ),
        }


    def collect_and_save(
        self,
        local_frame_id,
        carla_frame,
    ):
        """
        Collect and immediately save one synchronized frame.
        """

        packet = self.collect_frame(
            carla_frame
        )

        return self.save_frame(
            local_frame_id,
            packet,
        )


    # ========================================================
    # Flush / Close
    # ========================================================

    def flush(self):
        """
        Flush CSV streams to disk.
        """

        if self.gnss_file is not None:
            self.gnss_file.flush()

        if self.imu_file is not None:
            self.imu_file.flush()


    def close(self):
        """
        Flush and close open resources.
        """

        if self.gnss_file is not None:

            try:
                self.gnss_file.flush()
            finally:
                self.gnss_file.close()

            self.gnss_file = None
            self.gnss_writer = None

        if self.imu_file is not None:

            try:
                self.imu_file.flush()
            finally:
                self.imu_file.close()

            self.imu_file = None
            self.imu_writer = None


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
        self.close()
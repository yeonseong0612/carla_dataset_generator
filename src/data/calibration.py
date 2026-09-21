import json
import math
import os

import numpy as np

T_CV_FROM_CARLA = np.array(
    [
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.0,  0.0, 1.0],
    ],
    dtype=np.float64,
)

T_CARLA_FROM_CV = np.linalg.inv(T_CV_FROM_CARLA)


def carla_transform_to_matrix(transform):
    if hasattr(transform, "get_matrix"):
        return np.asarray(transform.get_matrix(), dtype=np.float64)

    location = transform.location
    rotation = transform.rotation

    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    cr = math.cos(roll)
    sr = math.sin(roll)

    matrix = np.array(
        [
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr,
                location.x,
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr,
                location.y,
            ],
            [
                sp,
                -cp * sr,
                cp * cr,
                location.z,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    return matrix


def invert_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    inverse = np.eye(4, dtype=np.float64,)

    inverse[:3, :3] = R.T
    inverse[:3, 3] = -R.T @ t

    return inverse

def camera_intrinsic_matrix(width, height, fov_deg):
    width = int(width)
    height = int(height)
    fov_deg = float(fov_deg)

    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K


def get_camera_calibration(camera_actor):
    attributes = camera_actor.attributes

    width = int(
        attributes["image_size_x"]
    )

    height = int(
        attributes["image_size_y"]
    )

    fov_deg = float(
        attributes["fov"]
    )

    K = camera_intrinsic_matrix(
        width,
        height,
        fov_deg,
    )

    return {
        "width": width,
        "height": height,
        "fov_deg": fov_deg,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "K": K.tolist(),
        "K_inv": np.linalg.inv(K).tolist(),
    }


# ============================================================
# Sensor extrinsics
# ============================================================

def get_sensor_extrinsic(
    ego_vehicle,
    sensor_actor,
):
    """
    Compute sensor extrinsic relative to ego vehicle.

    Returns:

        T_ego_from_sensor

    such that:

        p_ego =
            T_ego_from_sensor
            @ p_sensor
    """

    T_world_from_ego = (
        carla_transform_to_matrix(
            ego_vehicle.get_transform()
        )
    )

    T_world_from_sensor = (
        carla_transform_to_matrix(
            sensor_actor.get_transform()
        )
    )

    T_ego_from_world = invert_transform(
        T_world_from_ego
    )

    T_ego_from_sensor = (
        T_ego_from_world
        @ T_world_from_sensor
    )

    return T_ego_from_sensor


# ============================================================
# Stereo calibration
# ============================================================

def get_stereo_calibration(
    T_ego_from_left,
    T_ego_from_right,
):
    """
    Compute relative left/right camera geometry.

    Returns both CARLA-coordinate and CV-coordinate
    stereo transforms.
    """

    # Transform right-camera coordinates into
    # left-camera coordinates.
    T_left_from_ego = invert_transform(
        T_ego_from_left
    )

    T_left_from_right_carla = (
        T_left_from_ego
        @ T_ego_from_right
    )

    # Convert both camera coordinate representations
    # from CARLA convention to standard CV convention.
    T_left_from_right_cv = (
        T_CV_FROM_CARLA
        @ T_left_from_right_carla
        @ T_CARLA_FROM_CV
    )

    translation = (
        T_left_from_right_carla[
            :3, 3
        ]
    )

    baseline_m = float(
        np.linalg.norm(translation)
    )

    # For correctly rectified horizontal stereo,
    # the baseline should lie almost entirely along
    # the CV x-axis.
    baseline_x_cv_m = float(
        abs(
            T_left_from_right_cv[
                0, 3
            ]
        )
    )

    return {
        "baseline_m": baseline_m,
        "baseline_x_cv_m": baseline_x_cv_m,

        "T_left_from_right_carla":
            T_left_from_right_carla.tolist(),

        "T_left_from_right_cv":
            T_left_from_right_cv.tolist(),
    }


# ============================================================
# Full calibration generation
# ============================================================

def build_calibration(ego_vehicle, sensor_actors, left_camera_name="rgb_left", right_camera_name="rgb_right"):
    if left_camera_name not in sensor_actors:
        raise KeyError(
            f"Left camera "
            f"'{left_camera_name}' "
            f"not found."
        )

    if right_camera_name not in sensor_actors:
        raise KeyError(
            f"Right camera "
            f"'{right_camera_name}' "
            f"not found."
        )

    calibration = {
        "coordinate_systems": {
            "carla": {
                "x": "forward",
                "y": "right",
                "z": "up",
            },
            "camera_cv": {
                "x": "right",
                "y": "down",
                "z": "forward",
            },
            "transform_convention":
                "p_A = T_A_from_B @ p_B",

            "T_cv_from_carla":
                T_CV_FROM_CARLA.tolist(),
        },

        "cameras": {},

        "sensors": {},
    }

    # --------------------------------------------------------
    # Sensor extrinsics
    # --------------------------------------------------------

    extrinsics = {}

    for name, actor in sensor_actors.items():

        T_ego_from_sensor = (
            get_sensor_extrinsic(
                ego_vehicle,
                actor,
            )
        )

        extrinsics[name] = (
            T_ego_from_sensor
        )

        calibration[
            "sensors"
        ][name] = {
            "type_id": actor.type_id,

            "T_ego_from_sensor":
                T_ego_from_sensor.tolist(),

            "T_sensor_from_ego":
                invert_transform(
                    T_ego_from_sensor
                ).tolist(),
        }

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    for name, actor in sensor_actors.items():

        if not actor.type_id.startswith(
            "sensor.camera."
        ):
            continue

        camera_info = (
            get_camera_calibration(
                actor
            )
        )

        camera_info[
            "T_ego_from_camera"
        ] = (
            extrinsics[name].tolist()
        )

        camera_info[
            "T_camera_from_ego"
        ] = (
            invert_transform(
                extrinsics[name]
            ).tolist()
        )

        calibration[
            "cameras"
        ][name] = camera_info

    # --------------------------------------------------------
    # Stereo pair
    # --------------------------------------------------------

    stereo = get_stereo_calibration(
        extrinsics[
            left_camera_name
        ],
        extrinsics[
            right_camera_name
        ],
    )

    stereo[
        "left_camera"
    ] = left_camera_name

    stereo[
        "right_camera"
    ] = right_camera_name

    calibration[
        "stereo"
    ] = stereo

    return calibration


# ============================================================
# Save
# ============================================================

def save_calibration(
    ego_vehicle,
    sensor_actors,
    output_path,
    left_camera_name="rgb_left",
    right_camera_name="rgb_right",
):
    """
    Build calibration and save it as JSON.
    """

    calibration = build_calibration(
        ego_vehicle=ego_vehicle,
        sensor_actors=sensor_actors,
        left_camera_name=left_camera_name,
        right_camera_name=right_camera_name,
    )

    output_path = os.path.abspath(
        output_path
    )

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration,
            file,
            indent=2,
        )

    return calibration
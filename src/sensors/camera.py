import  carla

CAMERA_TYPES = {
    "rgb": "sensor.camera.rgb",
    "depth": "sensor.camera.depth",
    "optical_flow": "sensor.camera.optical_flow",
    "semantic" : "sensor.camera.semantic_segmentation"
}

def create_camera_blueprint(world, sensor_type, width, height, fov):
    if sensor_type not in CAMERA_TYPES:
        raise ValueError(f"Unsupported camera type: {sensor_type}")

    blueprint_id = CAMERA_TYPES[sensor_type]
    blueprint = world.get_blueprint_library().find(blueprint_id)

    if blueprint is None:
        raise RuntimeError(f"Sensor blueprint '{blueprint_id}' not found.")

    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint

def create_camera_transform(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
    )
    return transform

def create_left_camera_transform(cfg):
    return create_camera_transform(
        cfg.SENSOR.CAMERA.X,
        cfg.SENSOR.STEREO.LEFT_Y,
        cfg.SENSOR.CAMERA.Z,
        cfg.SENSOR.CAMERA.ROLL,
        cfg.SENSOR.CAMERA.PITCH,
        cfg.SENSOR.CAMERA.YAW
    )


def create_right_camera_transform(cfg):
    return create_camera_transform(
        cfg.SENSOR.CAMERA.X,
        cfg.SENSOR.STEREO.RIGHT_Y,
        cfg.SENSOR.CAMERA.Z,
        cfg.SENSOR.CAMERA.ROLL,
        cfg.SENSOR.CAMERA.PITCH,
        cfg.SENSOR.CAMERA.YAW
    )

def create_rgb_blueprint(world, cfg):
    return create_camera_blueprint(world, "rgb", cfg.SENSOR.CAMERA.WIDTH, cfg.SENSOR.CAMERA.HEIGHT, cfg.SENSOR.CAMERA.FOV)


def create_depth_blueprint(world, cfg):
    return create_camera_blueprint(world, "depth", cfg.SENSOR.CAMERA.WIDTH, cfg.SENSOR.CAMERA.HEIGHT, cfg.SENSOR.CAMERA.FOV)


def create_optical_flow_blueprint(world, cfg):
    return create_camera_blueprint(world, "optical_flow", cfg.SENSOR.CAMERA.WIDTH, cfg.SENSOR.CAMERA.HEIGHT, cfg.SENSOR.CAMERA.FOV)


def create_semantic_blueprint(world, cfg):
    return create_camera_blueprint(world, "semantic", cfg.SENSOR.CAMERA.WIDTH, cfg.SENSOR.CAMERA.HEIGHT, cfg.SENSOR.CAMERA.FOV)


def create_camera_rig_blueprints(world, cfg):
    return {
        "rgb_left": create_rgb_blueprint(world, cfg),
        "rgb_right": create_rgb_blueprint(world, cfg),
        "depth": create_depth_blueprint(world, cfg),
        "optical_flow": create_optical_flow_blueprint(world, cfg),
        "semantic": create_semantic_blueprint(world, cfg),
    }


def create_camera_rig_transforms(cfg):
    left_transform = create_left_camera_transform(cfg)
    right_transform = create_right_camera_transform(cfg)

    return {
        "rgb_left": left_transform,
        "rgb_right": right_transform,
        "depth": left_transform,
        "optical_flow": left_transform,
        "semantic": left_transform,
    }
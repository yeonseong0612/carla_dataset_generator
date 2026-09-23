import carla


def create_lidar_blueprint(world, cfg):
    blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")

    if blueprint is None:
        raise RuntimeError("LiDAR blueprint not found.")

    blueprint.set_attribute("channels", str(cfg.SENSOR.LIDAR.CHANNELS))
    blueprint.set_attribute("range", str(cfg.SENSOR.LIDAR.RANGE))
    blueprint.set_attribute("points_per_second", str(cfg.SENSOR.LIDAR.POINTS_PER_SECOND))
    blueprint.set_attribute("rotation_frequency", str(cfg.SENSOR.LIDAR.ROTATION_FREQUENCY))
    blueprint.set_attribute("horizontal_fov", str(cfg.SENSOR.LIDAR.HORIZONTAL_FOV))
    blueprint.set_attribute("upper_fov", str(cfg.SENSOR.LIDAR.UPPER_FOV))
    blueprint.set_attribute("lower_fov", str(cfg.SENSOR.LIDAR.LOWER_FOV))
    blueprint.set_attribute("sensor_tick", str(cfg.RECORDING.SAMPLE_INTERVAL_SECONDS))

    return blueprint


def create_lidar_transform(cfg):
    return carla.Transform(
        carla.Location(
            x=cfg.SENSOR.LIDAR.X,
            y=cfg.SENSOR.LIDAR.Y,
            z=cfg.SENSOR.LIDAR.Z
        ),
        carla.Rotation(
            roll=cfg.SENSOR.LIDAR.ROLL,
            pitch=cfg.SENSOR.LIDAR.PITCH,
            yaw=cfg.SENSOR.LIDAR.YAW
        )
    )

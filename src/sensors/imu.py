import carla


def create_imu_blueprint(world, cfg):
    blueprint = world.get_blueprint_library().find("sensor.other.imu")

    if blueprint is None:
        raise RuntimeError("IMU blueprint not found.")

    blueprint.set_attribute("noise_accel_stddev_x", str(cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_X))
    blueprint.set_attribute("noise_accel_stddev_y", str(cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_Y))
    blueprint.set_attribute("noise_accel_stddev_z", str(cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_Z))

    blueprint.set_attribute("noise_gyro_stddev_x", str(cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_X))
    blueprint.set_attribute("noise_gyro_stddev_y", str(cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_Y))
    blueprint.set_attribute("noise_gyro_stddev_z", str(cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_Z))

    blueprint.set_attribute("noise_gyro_bias_x", str(cfg.SENSOR.IMU.NOISE_GYRO_BIAS_X))
    blueprint.set_attribute("noise_gyro_bias_y", str(cfg.SENSOR.IMU.NOISE_GYRO_BIAS_Y))
    blueprint.set_attribute("noise_gyro_bias_z", str(cfg.SENSOR.IMU.NOISE_GYRO_BIAS_Z))

    blueprint.set_attribute("noise_seed", str(cfg.SENSOR.IMU.NOISE_SEED))
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint


def create_imu_transform(cfg):
    return carla.Transform(
        carla.Location(
            x=cfg.SENSOR.IMU.X,
            y=cfg.SENSOR.IMU.Y,
            z=cfg.SENSOR.IMU.Z
        )
    )
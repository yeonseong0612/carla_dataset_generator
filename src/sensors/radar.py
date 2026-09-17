import carla


def create_radar_blueprint(world, cfg):
    blueprint = world.get_blueprint_library().find("sensor.other.radar")

    if blueprint is None:
        raise RuntimeError("Radar blueprint not found.")

    blueprint.set_attribute("horizontal_fov", str(cfg.SENSOR.RADAR.HORIZONTAL_FOV))
    blueprint.set_attribute("vertical_fov", str(cfg.SENSOR.RADAR.VERTICAL_FOV))
    blueprint.set_attribute("range", str(cfg.SENSOR.RADAR.RANGE))
    blueprint.set_attribute("points_per_second", str(cfg.SENSOR.RADAR.POINTS_PER_SECOND))
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint


def create_radar_transform(cfg):
    return carla.Transform(
        carla.Location(
            x=cfg.SENSOR.RADAR.X,
            y=cfg.SENSOR.RADAR.Y,
            z=cfg.SENSOR.RADAR.Z
        ),
        carla.Rotation(
            roll=cfg.SENSOR.RADAR.ROLL,
            pitch=cfg.SENSOR.RADAR.PITCH,
            yaw=cfg.SENSOR.RADAR.YAW
        )
    )
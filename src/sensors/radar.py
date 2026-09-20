import carla


def create_radar_blueprint(world, radar_cfg):
    """
    radar_cfg: one radar's own EasyDict (e.g. cfg.SENSOR.RADAR,
    cfg.SENSOR.RADAR_FRONT_LEFT, cfg.SENSOR.RADAR_FRONT_RIGHT), not the
    top-level cfg, so the same factory works for every radar in the rig.
    """

    blueprint = world.get_blueprint_library().find("sensor.other.radar")

    if blueprint is None:
        raise RuntimeError("Radar blueprint not found.")

    blueprint.set_attribute("horizontal_fov", str(radar_cfg.HORIZONTAL_FOV))
    blueprint.set_attribute("vertical_fov", str(radar_cfg.VERTICAL_FOV))
    blueprint.set_attribute("range", str(radar_cfg.RANGE))
    blueprint.set_attribute("points_per_second", str(radar_cfg.POINTS_PER_SECOND))
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint


def create_radar_transform(radar_cfg):
    return carla.Transform(
        carla.Location(
            x=radar_cfg.X,
            y=radar_cfg.Y,
            z=radar_cfg.Z
        ),
        carla.Rotation(
            roll=radar_cfg.ROLL,
            pitch=radar_cfg.PITCH,
            yaw=radar_cfg.YAW
        )
    )

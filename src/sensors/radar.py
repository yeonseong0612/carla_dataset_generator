import carla


def create_radar_blueprint(world, radar_cfg, sensor_tick=0.0):
    """
    radar_cfg: one radar's own EasyDict (e.g. cfg.SENSOR.RADAR,
    cfg.SENSOR.RADAR_FRONT_LEFT, cfg.SENSOR.RADAR_FRONT_RIGHT), not the
    top-level cfg, so the same factory works for every radar in the rig.

    Radar has no rotation_frequency (unlike LiDAR): it is a static-FOV cone
    resampled every capture. Verified directly against a live CARLA 0.9.16
    server (not assumed, per CLAUDE.md section 7): unlike LiDAR, a radar
    capture's detection count does NOT scale with sensor_tick -- a
    controlled stationary A/B (sensor_tick=0.05 vs 0.1, identical scene)
    measured ~3.37k vs ~3.38k detections per capture, statistically the
    same. Widening sensor_tick to the 10 Hz recording interval therefore
    only changes how often a capture is delivered (20 Hz -> 10 Hz); each
    capture's own detection density/meaning is unchanged, and no other
    attribute needs adjusting (CLAUDE.md section 7/22).
    """

    blueprint = world.get_blueprint_library().find("sensor.other.radar")

    if blueprint is None:
        raise RuntimeError("Radar blueprint not found.")

    blueprint.set_attribute("horizontal_fov", str(radar_cfg.HORIZONTAL_FOV))
    blueprint.set_attribute("vertical_fov", str(radar_cfg.VERTICAL_FOV))
    blueprint.set_attribute("range", str(radar_cfg.RANGE))
    blueprint.set_attribute("points_per_second", str(radar_cfg.POINTS_PER_SECOND))
    blueprint.set_attribute("sensor_tick", str(sensor_tick))

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

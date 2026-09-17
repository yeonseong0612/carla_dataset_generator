import carla


def create_gnss_blueprint(world, cfg):
    blueprint = world.get_blueprint_library().find("sensor.other.gnss")

    if blueprint is None:
        raise RuntimeError("GNSS blueprint not found.")

    blueprint.set_attribute("noise_lat_stddev", str(cfg.SENSOR.GNSS.NOISE_LAT_STDDEV))
    blueprint.set_attribute("noise_lon_stddev", str(cfg.SENSOR.GNSS.NOISE_LON_STDDEV))
    blueprint.set_attribute("noise_alt_stddev", str(cfg.SENSOR.GNSS.NOISE_ALT_STDDEV))
    blueprint.set_attribute("noise_lat_bias", str(cfg.SENSOR.GNSS.NOISE_LAT_BIAS))
    blueprint.set_attribute("noise_lon_bias", str(cfg.SENSOR.GNSS.NOISE_LON_BIAS))
    blueprint.set_attribute("noise_alt_bias", str(cfg.SENSOR.GNSS.NOISE_ALT_BIAS))
    blueprint.set_attribute("noise_seed", str(cfg.SENSOR.GNSS.NOISE_SEED))
    blueprint.set_attribute("sensor_tick", "0.0")

    return blueprint


def create_gnss_transform(cfg):
    return carla.Transform(
        carla.Location(
            x=cfg.SENSOR.GNSS.X,
            y=cfg.SENSOR.GNSS.Y,
            z=cfg.SENSOR.GNSS.Z
        )
    )
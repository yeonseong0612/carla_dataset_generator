def spawn_sensor(world, ego, blueprint, transform):
    sensor = world.try_spawn_actor(blueprint, transform, attach_to=ego)

    if sensor is None:
        raise RuntimeError(f"Failed to spawn sensor '{blueprint.id}'.")

    return sensor


def destroy_sensor(sensor):
    if sensor is not None and sensor.is_alive:
        try:
            sensor.stop()
        except RuntimeError:
            pass

        sensor.destroy()


def destroy_sensors(sensors):
    for sensor in sensors:
        destroy_sensor(sensor)

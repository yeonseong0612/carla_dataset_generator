import carla

def get_vehicle_blueprint(world, blueprint_id):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(blueprint_id)

    if blueprint is None:
        raise RuntimeError(f"Vehicle blueprint '{blueprint_id}' not found.")
    
    return blueprint

def spawn_ego(world, transform, blueprint_id="vehicle.tesla.model3", role_name="hero"):
    blueprint = get_vehicle_blueprint(world, blueprint_id)
    
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    ego = world.try_spawn_actor(blueprint, transform)

    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    return ego

def destroy_vehicle(vehicle):
    if vehicle is not None and vehicle.is_alive:
        vehicle.destroy()

def spawn_ego_at_available_point(world, spawn_points, blueprint_id="vehicle.tesla.model3", role_name="hero"):
    blueprint = get_vehicle_blueprint(world, blueprint_id)

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    for transform in spawn_points:
        ego = world.try_spawn_actor(blueprint, transform)

        if ego is not None:
            return ego

    raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")
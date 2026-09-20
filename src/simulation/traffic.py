PASSENGER_CAR_FALLBACK = {
    "vehicle.bmw.grandtourer",
    "vehicle.mini.cooper_s",
}


def configure_traffic_manager(client, cfg):
    traffic_manager = client.get_trafficmanager(cfg.TRAFFIC_MANAGER.PORT)
    traffic_manager.set_random_device_seed(cfg.TRAFFIC.SEED)
    traffic_manager.set_synchronous_mode(True)
    return traffic_manager


def get_vehicle_category(blueprint):
    base_type = blueprint.get_attribute("base_type").as_str().lower() if blueprint.has_attribute("base_type") else ""

    if base_type == "bicycle":
        return "cyclist"

    if base_type == "motorcycle":
        return "motorcyclist"

    if base_type in ["car", "van", "truck", "bus"]:
        return "vehicle"

    if blueprint.id in PASSENGER_CAR_FALLBACK:
        return "vehicle"

    return None


def is_bus_blueprint(blueprint):
    
    if not blueprint.has_attribute("base_type"):
        return False

    return blueprint.get_attribute("base_type").as_str().lower() == "bus"


def get_traffic_blueprints(world):
    blueprints = world.get_blueprint_library().filter("vehicle.*")

    pools = {
        "vehicle": [],
        "cyclist": [],
        "motorcyclist": []
    }

    for blueprint in blueprints:
        category = get_vehicle_category(blueprint)

        if category is not None:
            pools[category].append(blueprint)

    return pools


def prepare_blueprint(blueprint, rng):
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", rng.choice(colors))

    if blueprint.has_attribute("driver_id"):
        driver_ids = blueprint.get_attribute("driver_id").recommended_values
        if driver_ids:
            blueprint.set_attribute("driver_id", rng.choice(driver_ids))

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "autopilot")

    return blueprint


def configure_actor_traffic_manager(actor, traffic_manager, cfg):
    actor.set_autopilot(True, cfg.TRAFFIC_MANAGER.PORT)
    traffic_manager.vehicle_percentage_speed_difference(actor, cfg.TRAFFIC.SPEED_DIFFERENCE)
    traffic_manager.auto_lane_change(actor, cfg.TRAFFIC.AUTO_LANE_CHANGE)
    traffic_manager.ignore_lights_percentage(actor, cfg.TRAFFIC.IGNORE_LIGHTS_PERCENTAGE)
    traffic_manager.ignore_signs_percentage(actor, cfg.TRAFFIC.IGNORE_SIGNS_PERCENTAGE)
    traffic_manager.ignore_vehicles_percentage(actor, cfg.TRAFFIC.IGNORE_VEHICLES_PERCENTAGE)
    traffic_manager.ignore_walkers_percentage(actor, cfg.TRAFFIC.IGNORE_WALKERS_PERCENTAGE)


def flatten_traffic_actors(traffic_actors):
    actors = []

    for group in traffic_actors.values():
        actors.extend(group)

    return actors


def destroy_traffic_vehicles(traffic_actors):
    if isinstance(traffic_actors, dict):
        actors = flatten_traffic_actors(traffic_actors)
    else:
        actors = traffic_actors

    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()
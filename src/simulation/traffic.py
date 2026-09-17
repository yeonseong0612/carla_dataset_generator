import random
import carla


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


def filter_spawn_points(spawn_points, ego_location, min_distance):
    return [
        transform
        for transform in spawn_points
        if transform.location.distance(ego_location) >= min_distance
    ]


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


def spawn_from_pool(world, traffic_manager, spawn_points, blueprints, count, cfg, rng):
    actors = []

    if not blueprints or count <= 0:
        return actors

    while spawn_points and len(actors) < count:
        transform = spawn_points.pop()
        blueprint = rng.choice(blueprints)
        blueprint = prepare_blueprint(blueprint, rng)

        actor = world.try_spawn_actor(blueprint, transform)

        if actor is None:
            continue

        configure_actor_traffic_manager(actor, traffic_manager, cfg)
        actors.append(actor)

    return actors


def spawn_traffic_vehicles(world, traffic_manager, ego, cfg):
    rng = random.Random(cfg.TRAFFIC.SEED)
    pools = get_traffic_blueprints(world)

    spawn_points = world.get_map().get_spawn_points()
    spawn_points = filter_spawn_points(spawn_points, ego.get_location(), cfg.TRAFFIC.MIN_DISTANCE_TO_EGO)
    rng.shuffle(spawn_points)

    vehicles = spawn_from_pool(
        world,
        traffic_manager,
        spawn_points,
        pools["vehicle"],
        cfg.TRAFFIC.NUM_VEHICLES,
        cfg,
        rng
    )

    cyclists = spawn_from_pool(
        world,
        traffic_manager,
        spawn_points,
        pools["cyclist"],
        cfg.TRAFFIC.NUM_CYCLISTS,
        cfg,
        rng
    )

    motorcyclists = spawn_from_pool(
        world,
        traffic_manager,
        spawn_points,
        pools["motorcyclist"],
        cfg.TRAFFIC.NUM_MOTORCYCLISTS,
        cfg,
        rng
    )

    return {
        "vehicle": vehicles,
        "cyclist": cyclists,
        "motorcyclist": motorcyclists
    }


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
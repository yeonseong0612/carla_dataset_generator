import random
import carla

def get_walker_blueprints(world):
    return list(world.get_blueprint_library().filter("walker.pedestrian.*"))


def spawn_pedestrians(world, cfg):
    rng = random.Random(cfg.PEDESTRIAN.SEED)
    world.set_pedestrians_seed(cfg.PEDESTRIAN.SEED)

    blueprints = get_walker_blueprints(world)
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")

    walkers = []
    controllers = []
    speeds = []

    attempts = 0
    max_attempts = cfg.PEDESTRIAN.NUM_WALKERS * 10

    while len(walkers) < cfg.PEDESTRIAN.NUM_WALKERS and attempts < max_attempts:
        attempts += 1

        location = world.get_random_location_from_navigation()

        if location is None:
            continue

        blueprint = rng.choice(blueprints)

        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")

        if blueprint.has_attribute("speed"):
            values = blueprint.get_attribute("speed").recommended_values

            if rng.random() < cfg.PEDESTRIAN.RUN_PERCENTAGE and len(values) > 2:
                speed = float(values[2])
            elif len(values) > 1:
                speed = float(values[1])
            else:
                speed = 1.4
        else:
            speed = 1.4

        walker = world.try_spawn_actor(blueprint, carla.Transform(location))

        if walker is None:
            continue

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

        if controller is None:
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)
        speeds.append(speed)

    return walkers, controllers, speeds


def start_pedestrians(world, controllers, speeds, cfg):
    world.set_pedestrians_cross_factor(cfg.PEDESTRIAN.CROSS_PERCENTAGE)

    for controller, speed in zip(controllers, speeds):
        controller.start()

        destination = world.get_random_location_from_navigation()

        if destination is not None:
            controller.go_to_location(destination)

        controller.set_max_speed(speed)


def destroy_pedestrians(walkers, controllers):
    for controller in controllers:
        try:
            if controller.is_alive:
                controller.stop()
        except RuntimeError:
            pass

    for controller in controllers:
        try:
            if controller.is_alive:
                controller.destroy()
        except RuntimeError:
            pass

    for walker in walkers:
        try:
            if walker.is_alive:
                walker.destroy()
        except RuntimeError:
            pass
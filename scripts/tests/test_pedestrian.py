import carla

from CFG.config import cfg
from src.simulation.vehicle import spawn_ego_at_available_point, destroy_vehicle
from src.simulation.traffic import configure_traffic_manager, spawn_traffic_vehicles, destroy_traffic_vehicles
from src.simulation.pedestrian import spawn_pedestrians, start_pedestrians, destroy_pedestrians


NUM_TEST_FRAMES = 300
CHECK_INTERVAL = 20
MOVE_THRESHOLD = 0.1


def main():
    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)

    world = client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()

    spawn_points = carla_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    ego = None
    vehicles = []
    walkers = []
    controllers = []
    walker_speeds = []
    traffic_manager = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        traffic_manager = configure_traffic_manager(client, cfg)

        ego = spawn_ego_at_available_point(world, spawn_points)
        ego.set_simulate_physics(False)

        vehicles = spawn_traffic_vehicles(world, traffic_manager, ego, cfg)

        walkers, controllers, walker_speeds = spawn_pedestrians(world, cfg)

        world.tick()

        start_pedestrians(world, controllers, walker_speeds, cfg)

        world.tick()

        previous_walker_locations = {}

        for walker in walkers:
            if walker.is_alive:
                previous_walker_locations[walker.id] = walker.get_location()

        print(f"Current map       : {carla_map.name}")
        print(f"Ego vehicle       : {ego.type_id}")
        print(f"NPC vehicles      : {len(vehicles)}/{cfg.TRAFFIC.NUM_VEHICLES}")
        print(f"Pedestrians       : {len(walkers)}/{cfg.PEDESTRIAN.NUM_WALKERS}")
        print(f"Controllers       : {len(controllers)}")
        print(f"Simulation        : {cfg.SIMULATION.FPS} Hz")
        print(f"Testing for {NUM_TEST_FRAMES} frames...")

        for i in range(NUM_TEST_FRAMES):
            frame = world.tick()

            if (i + 1) % CHECK_INTERVAL == 0:
                moving_vehicles = 0
                moving_walkers = 0

                for vehicle in vehicles:
                    if not vehicle.is_alive:
                        continue

                    velocity = vehicle.get_velocity()
                    speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

                    if speed > 0.1:
                        moving_vehicles += 1

                for walker in walkers:
                    if not walker.is_alive:
                        continue

                    current_location = walker.get_location()
                    previous_location = previous_walker_locations.get(walker.id)

                    if previous_location is not None:
                        distance = current_location.distance(previous_location)

                        if distance > MOVE_THRESHOLD:
                            moving_walkers += 1

                    previous_walker_locations[walker.id] = current_location

                print(
                    f"[{i + 1:03d}/{NUM_TEST_FRAMES}] "
                    f"frame={frame} "
                    f"vehicles={moving_vehicles}/{len(vehicles)} "
                    f"walkers={moving_walkers}/{len(walkers)}"
                )

    finally:
        destroy_pedestrians(walkers, controllers)
        destroy_traffic_vehicles(vehicles)
        destroy_vehicle(ego)

        if traffic_manager is not None:
            traffic_manager.set_synchronous_mode(False)

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
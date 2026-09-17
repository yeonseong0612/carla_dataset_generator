import carla

from CFG.config import cfg
from src.simulation.vehicle import spawn_ego, destroy_vehicle
from src.simulation.traffic import configure_traffic_manager, spawn_traffic_vehicles, destroy_traffic_vehicles


NUM_TEST_FRAMES = 200


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

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        traffic_manager = configure_traffic_manager(client, cfg)

        ego = spawn_ego(world, spawn_points[0])
        ego.set_simulate_physics(False)

        vehicles = spawn_traffic_vehicles(world, traffic_manager, ego, cfg)

        print(f"Current map     : {carla_map.name}")
        print(f"Ego vehicle     : {ego.type_id}")
        print(f"Requested NPCs  : {cfg.TRAFFIC.NUM_VEHICLES}")
        print(f"Spawned NPCs    : {len(vehicles)}")
        print(f"Simulation      : {cfg.SIMULATION.FPS} Hz")
        print(f"Testing for {NUM_TEST_FRAMES} frames...")

        for i in range(NUM_TEST_FRAMES):
            frame = world.tick()

            if (i + 1) % 20 == 0:
                moving = 0

                for vehicle in vehicles:
                    velocity = vehicle.get_velocity()
                    speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

                    if speed > 0.1:
                        moving += 1

                print(f"[{i + 1:03d}/{NUM_TEST_FRAMES}] frame={frame} moving={moving}/{len(vehicles)}")

    finally:
        destroy_traffic_vehicles(vehicles)
        destroy_vehicle(ego)

        traffic_manager = client.get_trafficmanager(cfg.TRAFFIC_MANAGER.PORT)
        traffic_manager.set_synchronous_mode(False)

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
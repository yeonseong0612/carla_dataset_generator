import os
import queue
import carla
import numpy as np

from CFG.config import cfg
from src.simulation.vehicle import spawn_ego, destroy_vehicle
from src.sensors.camera import create_camera_rig_blueprints, create_camera_rig_transforms
from src.sensors.lidar import create_lidar_blueprint, create_lidar_transform
from src.sensors.radar import create_radar_blueprint, create_radar_transform
from src.sensors.utils import spawn_sensor, destroy_sensors


NUM_TEST_FRAMES = 10
WARMUP_FRAMES = 20


def wait_for_frame(sensor_queue, target_frame, timeout=10.0):
    while True:
        data = sensor_queue.get(timeout=timeout)

        if data.frame < target_frame:
            continue

        if data.frame == target_frame:
            return data

        raise RuntimeError(f"Sensor skipped frame {target_frame}. Received {data.frame}.")


def clear_queue(sensor_queue):
    while not sensor_queue.empty():
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break


def radar_to_numpy(data):
    points = []

    for detection in data:
        depth = detection.depth
        azimuth = detection.azimuth
        altitude = detection.altitude
        velocity = detection.velocity

        x = depth * np.cos(altitude) * np.cos(azimuth)
        y = depth * np.cos(altitude) * np.sin(azimuth)
        z = depth * np.sin(altitude)

        points.append([x, y, z, velocity])

    return np.asarray(points, dtype=np.float32).reshape(-1, 4)


def save_camera(name, data, output_dir):
    if name in ["rgb_left", "rgb_right", "depth", "semantic"]:
        data.save_to_disk(os.path.join(output_dir, f"{data.frame:06d}.png"))

    elif name == "optical_flow":
        flow = np.frombuffer(data.raw_data, dtype=np.float32).reshape((data.height, data.width, 2))
        np.save(os.path.join(output_dir, f"{data.frame:06d}.npy"), flow)


def save_lidar(data, output_dir):
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    np.save(os.path.join(output_dir, f"{data.frame:06d}.npy"), points)
    return points


def save_radar(data, output_dir):
    points = radar_to_numpy(data)
    np.save(os.path.join(output_dir, f"{data.frame:06d}.npy"), points)
    return points


def main():
    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)

    world = client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()

    spawn_points = carla_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    output_root = os.path.join(cfg.PROJECT.ROOT, "outputs", "sensor_test", "integrated")

    output_dirs = {
        "rgb_left": os.path.join(output_root, "rgb_left"),
        "rgb_right": os.path.join(output_root, "rgb_right"),
        "depth": os.path.join(output_root, "depth"),
        "optical_flow": os.path.join(output_root, "optical_flow"),
        "semantic": os.path.join(output_root, "semantic"),
        "lidar": os.path.join(output_root, "lidar"),
        "radar": os.path.join(output_root, "radar"),
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    ego = None
    sensors = []
    sensor_queues = {}

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        ego = spawn_ego(world, spawn_points[0])
        ego.set_simulate_physics(False)

        camera_blueprints = create_camera_rig_blueprints(world, cfg)
        camera_transforms = create_camera_rig_transforms(cfg)

        for name in camera_blueprints:
            sensor = spawn_sensor(world, ego, camera_blueprints[name], camera_transforms[name])
            sensor_queue = queue.Queue()
            sensor.listen(sensor_queue.put)

            sensors.append(sensor)
            sensor_queues[name] = sensor_queue

        lidar = spawn_sensor(world, ego, create_lidar_blueprint(world, cfg), create_lidar_transform(cfg))
        lidar_queue = queue.Queue()
        lidar.listen(lidar_queue.put)
        sensors.append(lidar)
        sensor_queues["lidar"] = lidar_queue

        radar = spawn_sensor(world, ego, create_radar_blueprint(world, cfg.SENSOR.RADAR), create_radar_transform(cfg.SENSOR.RADAR))
        radar_queue = queue.Queue()
        radar.listen(radar_queue.put)
        sensors.append(radar)
        sensor_queues["radar"] = radar_queue

        print(f"Current map : {carla_map.name}")
        print(f"Ego vehicle : {ego.type_id}")
        print(f"Simulation  : {cfg.SIMULATION.FPS} Hz")
        print(f"Resolution  : {cfg.SENSOR.CAMERA.WIDTH} x {cfg.SENSOR.CAMERA.HEIGHT}")
        print(f"Sensors     : {list(sensor_queues.keys())}")
        print(f"Total       : {len(sensor_queues)} sensors")
        print("Physics     : disabled")
        print(f"Warming up for {WARMUP_FRAMES} frames...")

        for _ in range(WARMUP_FRAMES):
            world.tick()

        for sensor_queue in sensor_queues.values():
            clear_queue(sensor_queue)

        print(f"Saving {NUM_TEST_FRAMES} integrated frames...")

        for i in range(NUM_TEST_FRAMES):
            frame = world.tick()
            frame_data = {}

            for name, sensor_queue in sensor_queues.items():
                frame_data[name] = wait_for_frame(sensor_queue, frame)

            for name in ["rgb_left", "rgb_right", "depth", "optical_flow", "semantic"]:
                save_camera(name, frame_data[name], output_dirs[name])

            lidar_points = save_lidar(frame_data["lidar"], output_dirs["lidar"])
            radar_points = save_radar(frame_data["radar"], output_dirs["radar"])

            print(
                f"[{i + 1:02d}/{NUM_TEST_FRAMES}] "
                f"frame={frame} "
                f"lidar={lidar_points.shape[0]} "
                f"radar={radar_points.shape[0]}"
            )

        print(f"Saved to: {output_root}")

    except queue.Empty:
        raise RuntimeError("Timed out while waiting for integrated sensor data.")

    finally:
        destroy_sensors(sensors)
        destroy_vehicle(ego)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
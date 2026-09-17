import argparse
import math
import random
import sys
import weakref
from pathlib import Path

import carla
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CFG.config import cfg


WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
CAMERA_PANEL_HEIGHT = 620
BOTTOM_PANEL_HEIGHT = WINDOW_HEIGHT - CAMERA_PANEL_HEIGHT
HALF_WIDTH = WINDOW_WIDTH // 2

BG_COLOR = (5, 5, 8)
GRID_COLOR = (40, 40, 45)
TEXT_COLOR = (240, 240, 240)


def make_weather(name):
    weather = carla.WeatherParameters()

    weather.cloudiness = 0.0
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wind_intensity = 5.0
    weather.sun_azimuth_angle = 0.0
    weather.sun_altitude_angle = 70.0
    weather.fog_density = 0.0
    weather.fog_distance = 0.0
    weather.fog_falloff = 0.2
    weather.wetness = 0.0

    if name == "day_clear":
        pass

    elif name == "day_rain":
        weather.cloudiness = 90.0
        weather.precipitation = 80.0
        weather.precipitation_deposits = 80.0
        weather.wind_intensity = 50.0
        weather.wetness = 100.0

    elif name == "day_fog":
        weather.cloudiness = 50.0
        weather.fog_density = 70.0
        weather.fog_distance = 0.0
        weather.fog_falloff = 0.25

    elif name == "night_clear":
        weather.sun_altitude_angle = -30.0

    elif name == "night_rain":
        weather.sun_altitude_angle = -30.0
        weather.cloudiness = 90.0
        weather.precipitation = 80.0
        weather.precipitation_deposits = 80.0
        weather.wind_intensity = 50.0
        weather.wetness = 100.0

    elif name == "night_fog":
        weather.sun_altitude_angle = -30.0
        weather.cloudiness = 50.0
        weather.fog_density = 70.0
        weather.fog_distance = 0.0
        weather.fog_falloff = 0.25

    else:
        raise ValueError(f"Unknown weather: {name}")

    return weather


class SensorBuffer:
    def __init__(self):
        self.camera_rgb = None
        self.camera_frame = -1

        self.lidar_points = None
        self.lidar_count = 0
        self.lidar_frame = -1

        self.radar_points = None
        self.radar_velocity = None
        self.radar_count = 0
        self.radar_frame = -1


class SensorRig:
    def __init__(self, world, vehicle, buffer):
        self.world = world
        self.vehicle = vehicle
        self.buffer = buffer

        self.camera = None
        self.lidar = None
        self.radar = None

        self._spawn_camera()
        self._spawn_lidar()
        self._spawn_radar()

    def _spawn_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        bp.set_attribute("image_size_x", str(cfg.SENSOR.CAMERA.WIDTH))
        bp.set_attribute("image_size_y", str(cfg.SENSOR.CAMERA.HEIGHT))
        bp.set_attribute("fov", str(cfg.SENSOR.CAMERA.FOV))
        bp.set_attribute("sensor_tick", str(1.0 / cfg.SENSOR.CAMERA.FPS))

        transform = carla.Transform(
            carla.Location(
                x=cfg.SENSOR.CAMERA.X,
                y=0.0,
                z=cfg.SENSOR.CAMERA.Z
            ),
            carla.Rotation(
                roll=cfg.SENSOR.CAMERA.ROLL,
                pitch=cfg.SENSOR.CAMERA.PITCH,
                yaw=cfg.SENSOR.CAMERA.YAW
            )
        )

        self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self.camera.listen(lambda data: SensorRig._camera_callback(weak_self, data))

    def _spawn_lidar(self):
        bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")

        bp.set_attribute("channels", str(cfg.SENSOR.LIDAR.CHANNELS))
        bp.set_attribute("range", str(cfg.SENSOR.LIDAR.RANGE))
        bp.set_attribute("points_per_second", str(cfg.SENSOR.LIDAR.POINTS_PER_SECOND))
        bp.set_attribute("rotation_frequency", str(cfg.SENSOR.LIDAR.ROTATION_FREQUENCY))
        bp.set_attribute("upper_fov", str(cfg.SENSOR.LIDAR.UPPER_FOV))
        bp.set_attribute("lower_fov", str(cfg.SENSOR.LIDAR.LOWER_FOV))
        bp.set_attribute("sensor_tick", str(1.0 / cfg.SENSOR.LIDAR.ROTATION_FREQUENCY))

        transform = carla.Transform(
            carla.Location(
                x=cfg.SENSOR.LIDAR.X,
                y=cfg.SENSOR.LIDAR.Y,
                z=cfg.SENSOR.LIDAR.Z
            ),
            carla.Rotation(
                roll=cfg.SENSOR.LIDAR.ROLL,
                pitch=cfg.SENSOR.LIDAR.PITCH,
                yaw=cfg.SENSOR.LIDAR.YAW
            )
        )

        self.lidar = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self.lidar.listen(lambda data: SensorRig._lidar_callback(weak_self, data))

    def _spawn_radar(self):
        bp = self.world.get_blueprint_library().find("sensor.other.radar")

        radar_fps = getattr(cfg.SENSOR.RADAR, "FPS", 10)

        bp.set_attribute("horizontal_fov", str(cfg.SENSOR.RADAR.HORIZONTAL_FOV))
        bp.set_attribute("vertical_fov", str(cfg.SENSOR.RADAR.VERTICAL_FOV))
        bp.set_attribute("range", str(cfg.SENSOR.RADAR.RANGE))
        bp.set_attribute("points_per_second", str(cfg.SENSOR.RADAR.POINTS_PER_SECOND))
        bp.set_attribute("sensor_tick", str(1.0 / radar_fps))

        transform = carla.Transform(
            carla.Location(
                x=cfg.SENSOR.RADAR.X,
                y=cfg.SENSOR.RADAR.Y,
                z=cfg.SENSOR.RADAR.Z
            ),
            carla.Rotation(
                roll=cfg.SENSOR.RADAR.ROLL,
                pitch=cfg.SENSOR.RADAR.PITCH,
                yaw=cfg.SENSOR.RADAR.YAW
            )
        )

        self.radar = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self.radar.listen(lambda data: SensorRig._radar_callback(weak_self, data))

    @staticmethod
    def _camera_callback(weak_self, image):
        self = weak_self()
        if self is None:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(image.height, image.width, 4)
        array = array[:, :, :3][:, :, ::-1].copy()

        self.buffer.camera_rgb = array
        self.buffer.camera_frame = image.frame

    @staticmethod
    def _lidar_callback(weak_self, data):
        self = weak_self()
        if self is None:
            return

        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()

        self.buffer.lidar_points = points
        self.buffer.lidar_count = len(points)
        self.buffer.lidar_frame = data.frame

    @staticmethod
    def _radar_callback(weak_self, data):
        self = weak_self()
        if self is None:
            return

        if len(data) == 0:
            self.buffer.radar_points = np.empty((0, 3), dtype=np.float32)
            self.buffer.radar_velocity = np.empty((0,), dtype=np.float32)
            self.buffer.radar_count = 0
            self.buffer.radar_frame = data.frame
            return

        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

        velocity = raw[:, 0].copy()
        altitude = raw[:, 1]
        azimuth = raw[:, 2]
        depth = raw[:, 3]

        cos_altitude = np.cos(altitude)

        x = depth * cos_altitude * np.cos(azimuth)
        y = depth * cos_altitude * np.sin(azimuth)
        z = depth * np.sin(altitude)

        self.buffer.radar_points = np.stack((x, y, z), axis=1)
        self.buffer.radar_velocity = velocity
        self.buffer.radar_count = len(raw)
        self.buffer.radar_frame = data.frame

    def destroy(self):
        for sensor in [self.camera, self.lidar, self.radar]:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()


class KeyboardController:
    def __init__(self, vehicle, traffic_manager_port):
        self.vehicle = vehicle
        self.traffic_manager_port = traffic_manager_port

        self.control = carla.VehicleControl()

        self.autopilot = False
        self.reverse = False
        self.steer_cache = 0.0

    def toggle_autopilot(self):
        self.autopilot = not self.autopilot

        self.vehicle.set_autopilot(
            self.autopilot,
            self.traffic_manager_port
        )

        if self.autopilot:
            self.control = carla.VehicleControl()
            self.steer_cache = 0.0

    def toggle_reverse(self):
        if self.autopilot:
            return

        self.reverse = not self.reverse
        self.control.reverse = self.reverse

    def update(self, keys, dt):
        if self.autopilot:
            return

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.control.throttle = min(self.control.throttle + 0.05, 1.0)
        else:
            self.control.throttle = max(self.control.throttle - 0.10, 0.0)

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.control.brake = min(self.control.brake + 0.10, 1.0)
        else:
            self.control.brake = max(self.control.brake - 0.20, 0.0)

        steer_increment = 1.5 * dt

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            if self.steer_cache > 0.0:
                self.steer_cache = 0.0
            self.steer_cache -= steer_increment

        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            if self.steer_cache < 0.0:
                self.steer_cache = 0.0
            self.steer_cache += steer_increment

        else:
            self.steer_cache *= 0.65

        self.steer_cache = float(np.clip(self.steer_cache, -0.7, 0.7))

        self.control.steer = round(self.steer_cache, 3)
        self.control.reverse = self.reverse
        self.control.hand_brake = keys[pygame.K_SPACE]

        self.vehicle.apply_control(self.control)


def spawn_ego_vehicle(world, spawn_id=None):
    blueprint_library = world.get_blueprint_library()

    preferred = blueprint_library.filter("vehicle.tesla.model3")

    if preferred:
        blueprint = preferred[0]
    else:
        blueprints = blueprint_library.filter("vehicle.*")
        blueprint = random.choice(blueprints)

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No vehicle spawn points available.")

    if spawn_id is not None:
        if spawn_id < 0 or spawn_id >= len(spawn_points):
            raise ValueError(f"Invalid spawn ID: {spawn_id}")

        candidates = [spawn_points[spawn_id]]

    else:
        candidates = spawn_points.copy()
        random.shuffle(candidates)

    for transform in candidates:
        vehicle = world.try_spawn_actor(blueprint, transform)

        if vehicle is not None:
            return vehicle

    raise RuntimeError("Failed to spawn ego vehicle.")


def get_four_wheel_vehicle_blueprints(world):
    result = []

    for bp in world.get_blueprint_library().filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels"):
            if bp.get_attribute("number_of_wheels").as_int() != 4:
                continue

        result.append(bp)

    return result


def configure_vehicle_blueprint(bp, rng):
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "autopilot")

    if bp.has_attribute("color"):
        values = bp.get_attribute("color").recommended_values
        if values:
            bp.set_attribute("color", rng.choice(values))

    if bp.has_attribute("driver_id"):
        values = bp.get_attribute("driver_id").recommended_values
        if values:
            bp.set_attribute("driver_id", rng.choice(values))


def spawn_traffic_vehicles(world, traffic_manager, ego_vehicle):
    rng = random.Random(cfg.TRAFFIC.SEED)

    blueprints = get_four_wheel_vehicle_blueprints(world)
    spawn_points = world.get_map().get_spawn_points()

    rng.shuffle(spawn_points)

    ego_location = ego_vehicle.get_location()

    candidates = [
        transform
        for transform in spawn_points
        if transform.location.distance(ego_location) >= cfg.TRAFFIC.MIN_DISTANCE_TO_EGO
    ]

    vehicles = []

    for transform in candidates:
        if len(vehicles) >= cfg.TRAFFIC.NUM_VEHICLES:
            break

        bp = rng.choice(blueprints)
        configure_vehicle_blueprint(bp, rng)

        actor = world.try_spawn_actor(bp, transform)

        if actor is None:
            continue

        actor.set_autopilot(True, cfg.TRAFFIC_MANAGER.PORT)

        traffic_manager.vehicle_percentage_speed_difference(
            actor,
            cfg.TRAFFIC.SPEED_DIFFERENCE
        )

        traffic_manager.auto_lane_change(
            actor,
            cfg.TRAFFIC.AUTO_LANE_CHANGE
        )

        traffic_manager.ignore_lights_percentage(
            actor,
            cfg.TRAFFIC.IGNORE_LIGHTS_PERCENTAGE
        )

        traffic_manager.ignore_signs_percentage(
            actor,
            cfg.TRAFFIC.IGNORE_SIGNS_PERCENTAGE
        )

        traffic_manager.ignore_vehicles_percentage(
            actor,
            cfg.TRAFFIC.IGNORE_VEHICLES_PERCENTAGE
        )

        traffic_manager.ignore_walkers_percentage(
            actor,
            cfg.TRAFFIC.IGNORE_WALKERS_PERCENTAGE
        )

        vehicles.append(actor)

    return vehicles


def spawn_walkers(world):
    rng = random.Random(cfg.PEDESTRIAN.SEED)

    walker_blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")

    walkers = []
    controllers = []

    for _ in range(cfg.PEDESTRIAN.NUM_WALKERS):
        location = world.get_random_location_from_navigation()

        if location is None:
            continue

        transform = carla.Transform(location)

        bp = rng.choice(walker_blueprints)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(bp, transform)

        if walker is None:
            continue

        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),
            attach_to=walker
        )

        if controller is None:
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)

    world.tick()

    for walker, controller in zip(walkers, controllers):
        controller.start()

        destination = world.get_random_location_from_navigation()

        if destination is not None:
            controller.go_to_location(destination)

        bp = walker.attributes

        speed = 1.4

        if "speed" in bp:
            try:
                recommended = walker_blueprints.filter(walker.type_id)[0].get_attribute("speed").recommended_values

                if rng.random() < cfg.PEDESTRIAN.RUN_PERCENTAGE and len(recommended) > 2:
                    speed = float(recommended[2])
                elif len(recommended) > 1:
                    speed = float(recommended[1])

            except Exception:
                speed = 1.4

        controller.set_max_speed(speed)

    world.set_pedestrians_cross_factor(cfg.PEDESTRIAN.CROSS_PERCENTAGE)

    return walkers, controllers


def draw_grid(surface, panel, max_range, front_only=False):
    center_x = panel.width // 2

    if front_only:
        origin_y = panel.height - 22
    else:
        origin_y = panel.height // 2

    for distance in [20, 40, 60, 80, 100, 120]:
        if distance > max_range:
            continue

        scale = min(panel.width, panel.height if not front_only else panel.height * 2) / (2.0 * max_range)

        radius = int(distance * scale)

        pygame.draw.circle(
            surface,
            GRID_COLOR,
            (center_x, origin_y),
            radius,
            1
        )

    pygame.draw.line(
        surface,
        GRID_COLOR,
        (center_x, 0),
        (center_x, panel.height),
        1
    )


def render_camera(display, rgb):
    panel = pygame.Rect(
        0,
        0,
        WINDOW_WIDTH,
        CAMERA_PANEL_HEIGHT
    )

    pygame.draw.rect(display, BG_COLOR, panel)

    if rgb is None:
        return

    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    src_w, src_h = surface.get_size()

    scale = min(
        panel.width / src_w,
        panel.height / src_h
    )

    dst_w = int(src_w * scale)
    dst_h = int(src_h * scale)

    scaled = pygame.transform.smoothscale(
        surface,
        (dst_w, dst_h)
    )

    x = panel.x + (panel.width - dst_w) // 2
    y = panel.y + (panel.height - dst_h) // 2

    display.blit(scaled, (x, y))


def render_lidar(display, points):
    panel = pygame.Rect(
        0,
        CAMERA_PANEL_HEIGHT,
        HALF_WIDTH,
        BOTTOM_PANEL_HEIGHT
    )

    pygame.draw.rect(display, BG_COLOR, panel)

    panel_surface = pygame.Surface((panel.width, panel.height))
    panel_surface.fill(BG_COLOR)

    max_range = float(cfg.SENSOR.LIDAR.RANGE)

    draw_grid(
        panel_surface,
        pygame.Rect(0, 0, panel.width, panel.height),
        max_range
    )

    if points is not None and len(points) > 0:
        x = points[:, 0]
        y = points[:, 1]

        valid = (
            (np.abs(x) <= max_range)
            & (np.abs(y) <= max_range)
        )

        x = x[valid]
        y = y[valid]

        scale = min(panel.width, panel.height) / (2.0 * max_range)

        px = panel.width / 2.0 + y * scale
        py = panel.height / 2.0 - x * scale

        px = px.astype(np.int32)
        py = py.astype(np.int32)

        valid = (
            (px >= 0)
            & (px < panel.width)
            & (py >= 0)
            & (py < panel.height)
        )

        px = px[valid]
        py = py[valid]

        image = np.zeros(
            (panel.height, panel.width, 3),
            dtype=np.uint8
        )

        image[py, px] = (230, 230, 230)

        cloud_surface = pygame.surfarray.make_surface(
            image.swapaxes(0, 1)
        )

        panel_surface.blit(cloud_surface, (0, 0))

    center_x = panel.width // 2
    center_y = panel.height // 2

    pygame.draw.circle(
        panel_surface,
        (0, 255, 80),
        (center_x, center_y),
        5
    )

    pygame.draw.line(
        panel_surface,
        (0, 255, 80),
        (center_x, center_y),
        (center_x, center_y - 18),
        2
    )

    display.blit(panel_surface, panel.topleft)


def render_radar(display, points, velocity):
    panel = pygame.Rect(
        HALF_WIDTH,
        CAMERA_PANEL_HEIGHT,
        HALF_WIDTH,
        BOTTOM_PANEL_HEIGHT
    )

    pygame.draw.rect(display, BG_COLOR, panel)

    panel_surface = pygame.Surface((panel.width, panel.height))
    panel_surface.fill(BG_COLOR)

    max_range = float(cfg.SENSOR.RADAR.RANGE)

    draw_grid(
        panel_surface,
        pygame.Rect(0, 0, panel.width, panel.height),
        max_range,
        front_only=True
    )

    origin_x = panel.width // 2
    origin_y = panel.height - 22

    half_fov = math.radians(
        cfg.SENSOR.RADAR.HORIZONTAL_FOV / 2.0
    )

    scale = min(
        panel.width,
        panel.height * 2.0
    ) / (2.0 * max_range)

    ray_length = max_range * scale

    for sign in [-1, 1]:
        angle = sign * half_fov

        end_x = origin_x + math.sin(angle) * ray_length
        end_y = origin_y - math.cos(angle) * ray_length

        pygame.draw.line(
            panel_surface,
            (60, 60, 70),
            (origin_x, origin_y),
            (int(end_x), int(end_y)),
            1
        )

    if points is not None and velocity is not None and len(points) > 0:
        x = points[:, 0]
        y = points[:, 1]

        valid = (
            (x >= 0.0)
            & (x <= max_range)
            & (np.abs(y) <= max_range)
        )

        x = x[valid]
        y = y[valid]
        velocity = velocity[valid]

        px = origin_x + y * scale
        py = origin_y - x * scale

        px = px.astype(np.int32)
        py = py.astype(np.int32)

        valid = (
            (px >= 0)
            & (px < panel.width)
            & (py >= 0)
            & (py < panel.height)
        )

        px = px[valid]
        py = py[valid]
        velocity = velocity[valid]

        vmax = 15.0

        normalized = np.clip(
            velocity / vmax,
            -1.0,
            1.0
        )

        red = np.where(
            normalized < 0.0,
            255.0 * np.abs(normalized),
            40.0
        )

        blue = np.where(
            normalized > 0.0,
            255.0 * normalized,
            40.0
        )

        green = 255.0 * (
            1.0 - np.abs(normalized)
        )

        colors = np.stack(
            (red, green, blue),
            axis=1
        ).astype(np.uint8)

        image = np.zeros(
            (panel.height, panel.width, 3),
            dtype=np.uint8
        )

        image[py, px] = colors

        radar_surface = pygame.surfarray.make_surface(
            image.swapaxes(0, 1)
        )

        panel_surface.blit(radar_surface, (0, 0))

    pygame.draw.circle(
        panel_surface,
        (0, 255, 80),
        (origin_x, origin_y),
        5
    )

    display.blit(panel_surface, panel.topleft)


def draw_titles(display, font):
    labels = [
        ("CAMERA", WINDOW_WIDTH // 2, 10),
        ("LiDAR", HALF_WIDTH // 2, CAMERA_PANEL_HEIGHT + 10),
        ("RADAR", HALF_WIDTH + HALF_WIDTH // 2, CAMERA_PANEL_HEIGHT + 10)
    ]

    for text, center_x, y in labels:
        surface = font.render(
            text,
            True,
            TEXT_COLOR
        )

        display.blit(
            surface,
            (
                center_x - surface.get_width() // 2,
                y
            )
        )

    pygame.draw.line(
        display,
        (90, 90, 90),
        (0, CAMERA_PANEL_HEIGHT),
        (WINDOW_WIDTH, CAMERA_PANEL_HEIGHT),
        2
    )

    pygame.draw.line(
        display,
        (90, 90, 90),
        (HALF_WIDTH, CAMERA_PANEL_HEIGHT),
        (HALF_WIDTH, WINDOW_HEIGHT),
        2
    )


def draw_hud(display, font, vehicle, controller, weather_name, frame, buffer, npc_count, walker_count):
    velocity = vehicle.get_velocity()

    speed = 3.6 * math.sqrt(
        velocity.x ** 2
        + velocity.y ** 2
        + velocity.z ** 2
    )

    mode = (
        "AUTOPILOT"
        if controller.autopilot
        else "MANUAL"
    )

    direction = (
        "REVERSE"
        if controller.reverse
        else "FORWARD"
    )

    lines = [
        f"Speed      : {speed:6.1f} km/h",
        f"Mode       : {mode}",
        f"Direction  : {direction}",
        f"Weather    : {weather_name}",
        f"World frame: {frame}",
        f"Camera     : frame {buffer.camera_frame}",
        f"LiDAR      : {buffer.lidar_count:,} pts | frame {buffer.lidar_frame}",
        f"Radar      : {buffer.radar_count:,} det | frame {buffer.radar_frame}",
        f"Vehicles   : {npc_count}",
        f"Walkers    : {walker_count}",
    ]

    width = 390
    height = len(lines) * 23 + 18

    hud = pygame.Surface(
        (width, height),
        pygame.SRCALPHA
    )

    hud.fill((0, 0, 0, 155))

    for i, line in enumerate(lines):
        text = font.render(
            line,
            True,
            (255, 255, 255)
        )

        hud.blit(
            text,
            (10, 8 + i * 23)
        )

    display.blit(hud, (12, 42))


def destroy_actors(client, actors):
    ids = [
        actor.id
        for actor in actors
        if actor is not None
    ]

    if ids:
        client.apply_batch(
            [
                carla.command.DestroyActor(actor_id)
                for actor_id in ids
            ]
        )


def print_controls():
    print()
    print("W / Up          Throttle")
    print("S / Down        Brake")
    print("A,D / Left,Right Steering")
    print("Space           Hand brake")
    print("Q               Forward / Reverse")
    print("P               Manual / Autopilot")
    print("C               Next weather")
    print("Shift + C       Previous weather")
    print("F1              HUD")
    print("ESC             Quit")
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map",
        default=cfg.MAP.NAME
    )

    parser.add_argument(
        "--spawn-id",
        type=int,
        default=None
    )

    parser.add_argument(
        "--autopilot",
        action="store_true"
    )

    args = parser.parse_args()

    random.seed(cfg.RANDOM.SEED)
    np.random.seed(cfg.RANDOM.SEED)

    pygame.init()
    pygame.font.init()

    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.DOUBLEBUF
    )

    pygame.display.set_caption(
        "CARLA Multi-Sensor Monitor"
    )

    font = pygame.font.SysFont(
        "consolas",
        18
    )

    title_font = pygame.font.SysFont(
        "consolas",
        20,
        bold=True
    )

    clock = pygame.time.Clock()

    client = carla.Client(
        cfg.CARLA.HOST,
        cfg.CARLA.PORT
    )

    client.set_timeout(
        cfg.CARLA.TIMEOUT
    )

    world = client.get_world()

    current_map = world.get_map().name.split("/")[-1]

    if current_map != args.map:
        world = client.load_world(args.map)

    original_settings = world.get_settings()
    original_weather = world.get_weather()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS

    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(
        cfg.TRAFFIC_MANAGER.PORT
    )

    traffic_manager.set_synchronous_mode(True)

    traffic_manager.set_random_device_seed(
        cfg.TRAFFIC.SEED
    )

    vehicle = None
    rig = None

    traffic_vehicles = []
    walkers = []
    walker_controllers = []

    try:
        vehicle = spawn_ego_vehicle(
            world,
            args.spawn_id
        )

        traffic_vehicles = spawn_traffic_vehicles(
            world,
            traffic_manager,
            vehicle
        )

        walkers, walker_controllers = spawn_walkers(
            world
        )

        buffer = SensorBuffer()

        rig = SensorRig(
            world,
            vehicle,
            buffer
        )

        controller = KeyboardController(
            vehicle,
            cfg.TRAFFIC_MANAGER.PORT
        )

        if args.autopilot:
            controller.toggle_autopilot()

        weather_conditions = list(
            cfg.WEATHER.CONDITIONS
        )

        if cfg.WEATHER.DEFAULT in weather_conditions:
            weather_index = weather_conditions.index(
                cfg.WEATHER.DEFAULT
            )
        else:
            weather_index = 0

        weather_name = weather_conditions[
            weather_index
        ]

        world.set_weather(
            make_weather(weather_name)
        )

        hud_enabled = True
        running = True

        print_controls()

        for _ in range(10):
            world.tick()

        while running:
            dt = clock.tick_busy_loop(60) / 1000.0

            frame = world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_p:
                        controller.toggle_autopilot()

                    elif event.key == pygame.K_q:
                        controller.toggle_reverse()

                    elif event.key == pygame.K_F1:
                        hud_enabled = not hud_enabled

                    elif event.key == pygame.K_c:
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            weather_index = (
                                weather_index - 1
                            ) % len(weather_conditions)
                        else:
                            weather_index = (
                                weather_index + 1
                            ) % len(weather_conditions)

                        weather_name = weather_conditions[
                            weather_index
                        ]

                        world.set_weather(
                            make_weather(weather_name)
                        )

            keys = pygame.key.get_pressed()

            controller.update(
                keys,
                dt
            )

            display.fill(BG_COLOR)

            render_camera(
                display,
                buffer.camera_rgb
            )

            render_lidar(
                display,
                buffer.lidar_points
            )

            render_radar(
                display,
                buffer.radar_points,
                buffer.radar_velocity
            )

            draw_titles(
                display,
                title_font
            )

            if hud_enabled:
                draw_hud(
                    display,
                    font,
                    vehicle,
                    controller,
                    weather_name,
                    frame,
                    buffer,
                    len(traffic_vehicles),
                    len(walkers)
                )

            pygame.display.flip()

    finally:
        if vehicle is not None:
            vehicle.set_autopilot(
                False,
                cfg.TRAFFIC_MANAGER.PORT
            )

        if rig is not None:
            rig.destroy()

        for controller_actor in walker_controllers:
            try:
                controller_actor.stop()
            except RuntimeError:
                pass

        destroy_actors(
            client,
            walker_controllers
        )

        destroy_actors(
            client,
            walkers
        )

        destroy_actors(
            client,
            traffic_vehicles
        )

        if vehicle is not None:
            vehicle.destroy()

        world.set_weather(
            original_weather
        )

        traffic_manager.set_synchronous_mode(
            False
        )

        world.apply_settings(
            original_settings
        )

        pygame.quit()


if __name__ == "__main__":
    main()
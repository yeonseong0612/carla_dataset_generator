import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import pygame
import carla


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CARLA_ROOT = Path(r"C:\CARLA")
CARLA_PYTHONAPI = CARLA_ROOT / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))


# ============================================================
# Project modules
# ============================================================

from src.navigation.route import (
    load_route_from_xml,
    project_control_points,
    build_dense_route,
)

from src.navigation.controller import RouteController


# ============================================================
# Topology renderer
# ============================================================

class TopologyRenderer:

    def __init__(
        self,
        carla_map,
        surface_width,
        surface_height,
        margin=40,
    ):
        self.map = carla_map
        self.width = surface_width
        self.height = surface_height
        self.margin = margin

        self.topology = self.map.get_topology()

        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")

        self._calculate_bounds()

        usable_w = self.width - 2 * self.margin
        usable_h = self.height - 2 * self.margin

        map_w = max(
            self.max_x - self.min_x,
            1.0,
        )

        map_h = max(
            self.max_y - self.min_y,
            1.0,
        )

        self.scale = min(
            usable_w / map_w,
            usable_h / map_h,
        )

        self.base_surface = pygame.Surface(
            (self.width, self.height)
        )

        self._draw_static_topology()

    def _calculate_bounds(self):

        waypoints = self.map.generate_waypoints(
            5.0
        )

        for wp in waypoints:
            loc = wp.transform.location

            self.min_x = min(
                self.min_x,
                loc.x,
            )

            self.max_x = max(
                self.max_x,
                loc.x,
            )

            self.min_y = min(
                self.min_y,
                loc.y,
            )

            self.max_y = max(
                self.max_y,
                loc.y,
            )

    def world_to_screen(
        self,
        location,
    ):

        x = (
            self.margin
            + (location.x - self.min_x)
            * self.scale
        )

        y = (
            self.height
            - self.margin
            - (location.y - self.min_y)
            * self.scale
        )

        return int(x), int(y)

    def _draw_static_topology(self):

        self.base_surface.fill(
            (245, 245, 245)
        )

        road_color = (
            100,
            100,
            100,
        )

        topology = self.map.get_topology()

        for wp_start, wp_end in topology:

            p0 = self.world_to_screen(
                wp_start.transform.location
            )

            p1 = self.world_to_screen(
                wp_end.transform.location
            )

            pygame.draw.line(
                self.base_surface,
                road_color,
                p0,
                p1,
                1,
            )

    def draw(
        self,
        planned_route,
        driven_trajectory,
        vehicle_transform,
        control_points,
    ):

        surface = self.base_surface.copy()

        # ----------------------------------------------------
        # XML control points
        # ----------------------------------------------------

        for wp in control_points:

            pos = self.world_to_screen(
                wp.transform.location
            )

            pygame.draw.circle(
                surface,
                (255, 140, 0),
                pos,
                6,
            )

        # ----------------------------------------------------
        # Planned route
        # ----------------------------------------------------

        route_points = [
            self.world_to_screen(
                wp.transform.location
            )
            for wp, _ in planned_route
        ]

        if len(route_points) >= 2:

            pygame.draw.lines(
                surface,
                (40, 100, 255),
                False,
                route_points,
                4,
            )

        # ----------------------------------------------------
        # Driven trajectory
        # ----------------------------------------------------

        if len(driven_trajectory) >= 2:

            trajectory_points = [
                self.world_to_screen(p)
                for p in driven_trajectory
            ]

            pygame.draw.lines(
                surface,
                (20, 180, 80),
                False,
                trajectory_points,
                3,
            )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        goal_location = (
            planned_route[-1][0]
            .transform
            .location
        )

        goal_position = self.world_to_screen(
            goal_location
        )

        pygame.draw.circle(
            surface,
            (180, 40, 180),
            goal_position,
            9,
            3,
        )

        # ----------------------------------------------------
        # Ego vehicle
        # ----------------------------------------------------

        ego_location = (
            vehicle_transform.location
        )

        ego_pos = self.world_to_screen(
            ego_location
        )

        pygame.draw.circle(
            surface,
            (230, 40, 40),
            ego_pos,
            8,
        )

        # Ego heading
        yaw = math.radians(
            vehicle_transform.rotation.yaw
        )

        direction_length = 18

        arrow_end = (
            int(
                ego_pos[0]
                + math.cos(yaw)
                * direction_length
            ),
            int(
                ego_pos[1]
                - math.sin(yaw)
                * direction_length
            ),
        )

        pygame.draw.line(
            surface,
            (230, 40, 40),
            ego_pos,
            arrow_end,
            3,
        )

        return surface


# ============================================================
# Camera
# ============================================================

class CameraManager:

    def __init__(
        self,
        world,
        vehicle,
        width,
        height,
    ):
        self.surface = None
        self.sensor = None

        blueprint_library = (
            world.get_blueprint_library()
        )

        camera_bp = (
            blueprint_library.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(width),
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(height),
        )

        camera_bp.set_attribute(
            "fov",
            "90",
        )

        camera_transform = (
            carla.Transform(
                carla.Location(
                    x=-6.0,
                    z=3.0,
                ),
                carla.Rotation(
                    pitch=-15.0,
                ),
            )
        )

        self.sensor = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
        )

        self.sensor.listen(
            self._callback
        )

    def _callback(
        self,
        image,
    ):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8,
        )

        array = array.reshape(
            (
                image.height,
                image.width,
                4,
            )
        )

        # BGRA -> RGB
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        array = array.swapaxes(
            0,
            1,
        )

        self.surface = (
            pygame.surfarray.make_surface(
                array
            )
        )

    def destroy(self):

        if self.sensor is None:
            return

        try:
            self.sensor.stop()
        except Exception:
            pass

        try:
            self.sensor.destroy()
        except Exception:
            pass

        self.sensor = None


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CARLA XML route / controller test"
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--xml",
        required=True,
        help="Route XML file",
    )

    parser.add_argument(
        "--route-id",
        required=True,
        help="Route ID in XML",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=30.0,
        help="Cruise speed [km/h]",
    )

    parser.add_argument(
        "--min-curve-speed",
        type=float,
        default=12.0,
        help="Minimum curve speed [km/h]",
    )

    parser.add_argument(
        "--curve-lookahead",
        type=float,
        default=20.0,
        help=(
            "Curve lookahead distance [m]"
        ),
    )

    parser.add_argument(
        "--traffic-light-policy",
        choices=[
            "obey",
            "ignore",
        ],
        default="obey",
    )

    parser.add_argument(
        "--stuck-timeout",
        type=float,
        default=30.0,
        help="Stuck timeout [s]",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--vehicle-filter",
        default="vehicle.tesla.model3",
    )

    parser.add_argument(
        "--sampling-resolution",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    # ========================================================
    # Route XML
    # ========================================================

    town, control_locations = (
        load_route_from_xml(
            args.xml,
            args.route_id,
        )
    )

    route_id = str(
        args.route_id
    )

    print("=" * 70)
    print("CARLA Route Controller Test")
    print("=" * 70)

    print(
        f"Town             : {town}"
    )

    print(
        f"Route ID         : {route_id}"
    )

    print(
        f"Control points   : "
        f"{len(control_locations)}"
    )

    print(
        f"Cruise speed     : "
        f"{args.speed:.1f} km/h"
    )

    print(
        f"Min curve speed  : "
        f"{args.min_curve_speed:.1f} km/h"
    )

    print(
        f"Curve lookahead  : "
        f"{args.curve_lookahead:.1f} m"
    )

    print(
        f"Traffic lights   : "
        f"{args.traffic_light_policy}"
    )

    # ========================================================
    # CARLA connection
    # ========================================================

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        60.0
    )

    world = None
    original_settings = None

    vehicle = None
    camera = None

    pygame.init()

    screen = pygame.display.set_mode(
        (
            args.width,
            args.height,
        )
    )

    pygame.display.set_caption(
        f"CARLA Route Test - "
        f"{town} / Route {route_id}"
    )

    clock = pygame.time.Clock()

    try:

        # ====================================================
        # Load map
        # ====================================================

        print(
            f"\n[CARLA] Loading {town}..."
        )

        world = client.load_world(
            town
        )

        time.sleep(
            1.0
        )

        carla_map = world.get_map()

        print(
            f"[CARLA] Loaded map: "
            f"{carla_map.name}"
        )

        # ====================================================
        # Synchronous mode
        # ====================================================

        original_settings = (
            world.get_settings()
        )

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            args.fixed_delta
        )

        world.apply_settings(
            settings
        )

        # ====================================================
        # Route
        # ====================================================

        print(
            "\n[Route] Projecting "
            "XML control points..."
        )

        projected_waypoints = (
            project_control_points(
                carla_map,
                control_locations,
            )
        )

        print(
            "\n[Route] Generating "
            "dense route..."
        )

        dense_route = build_dense_route(
            carla_map,
            projected_waypoints,
            sampling_resolution=(
                args.sampling_resolution
            ),
        )

        print(
            f"[Route] Dense points: "
            f"{len(dense_route)}"
        )

        # ====================================================
        # Vehicle
        # ====================================================

        blueprint_library = (
            world.get_blueprint_library()
        )

        vehicle_candidates = (
            blueprint_library.filter(
                args.vehicle_filter
            )
        )

        if not vehicle_candidates:

            raise RuntimeError(
                f"No vehicle found: "
                f"{args.vehicle_filter}"
            )

        vehicle_bp = (
            vehicle_candidates[0]
        )

        # copy transform to avoid modifying
        # original waypoint transform object
        first_transform = (
            projected_waypoints[0]
            .transform
        )

        spawn_transform = (
            carla.Transform(
                carla.Location(
                    x=first_transform.location.x,
                    y=first_transform.location.y,
                    z=(
                        first_transform.location.z
                        + 0.3
                    ),
                ),
                carla.Rotation(
                    pitch=first_transform.rotation.pitch,
                    yaw=first_transform.rotation.yaw,
                    roll=first_transform.rotation.roll,
                ),
            )
        )

        print(
            "\n[Vehicle] Spawning ego..."
        )

        vehicle = world.try_spawn_actor(
            vehicle_bp,
            spawn_transform,
        )

        if vehicle is None:
            raise RuntimeError(
                "Failed to spawn ego vehicle."
            )

        print(
            f"[Vehicle] Spawned: "
            f"{vehicle.type_id}"
        )

        # ====================================================
        # Route Controller
        # ====================================================

        controller = RouteController(
            vehicle=vehicle,
            dense_route=dense_route,

            cruise_speed=args.speed,

            min_curve_speed=(
                args.min_curve_speed
            ),

            traffic_light_policy=(
                args.traffic_light_policy
            ),

            curve_lookahead_distance=(
                args.curve_lookahead
            ),

            stuck_timeout=(
                args.stuck_timeout
            ),
        )

        print(
            "[Controller] Ready"
        )

        # ====================================================
        # Camera
        # ====================================================

        half_width = (
            args.width // 2
        )

        camera = CameraManager(
            world,
            vehicle,
            half_width,
            args.height,
        )

        # ====================================================
        # Topology
        # ====================================================

        topology_renderer = (
            TopologyRenderer(
                carla_map,
                half_width,
                args.height,
            )
        )

        # ====================================================
        # Fonts
        # ====================================================

        font = pygame.font.SysFont(
            "consolas",
            20,
        )

        small_font = (
            pygame.font.SysFont(
                "consolas",
                17,
            )
        )

        # ====================================================
        # Initial tick
        # ====================================================

        world.tick()

        # ====================================================
        # Run
        # ====================================================

        driven_trajectory = []

        running = True

        print()
        print("=" * 70)
        print("Simulation started")
        print("ESC: quit")
        print("=" * 70)

        while running:

            # ------------------------------------------------
            # Events
            # ------------------------------------------------

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif (
                    event.type
                    == pygame.KEYDOWN
                ):

                    if (
                        event.key
                        == pygame.K_ESCAPE
                    ):
                        running = False

            if not running:
                break

            # ------------------------------------------------
            # Simulation step
            # ------------------------------------------------

            world.tick()

            # ------------------------------------------------
            # Controller
            # ------------------------------------------------

            control = (
                controller.run_step()
            )

            vehicle.apply_control(
                control
            )

            status = (
                controller.get_status()
            )

            # ------------------------------------------------
            # Vehicle state
            # ------------------------------------------------

            transform = (
                vehicle.get_transform()
            )

            location = (
                transform.location
            )

            # ------------------------------------------------
            # Driven trajectory
            # ------------------------------------------------

            if (
                not driven_trajectory
                or
                driven_trajectory[-1].distance(
                    location
                ) > 0.5
            ):

                driven_trajectory.append(
                    carla.Location(
                        x=location.x,
                        y=location.y,
                        z=location.z,
                    )
                )

            # ------------------------------------------------
            # Screen
            # ------------------------------------------------

            screen.fill(
                (20, 20, 20)
            )

            # Left: chase camera
            if camera.surface is not None:

                screen.blit(
                    camera.surface,
                    (0, 0),
                )

            # Right: topology
            map_surface = (
                topology_renderer.draw(
                    dense_route,
                    driven_trajectory,
                    transform,
                    projected_waypoints,
                )
            )

            screen.blit(
                map_surface,
                (
                    half_width,
                    0,
                ),
            )

            # Divider
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (half_width, 0),
                (
                    half_width,
                    args.height,
                ),
                3,
            )

            # ------------------------------------------------
            # Controller HUD
            # ------------------------------------------------

            hud_lines = [
                f"Town       : {town}",
                f"Route      : {route_id}",

                (
                    f"Speed      : "
                    f"{status['speed_kmh']:5.1f} "
                    f"km/h"
                ),

                (
                    f"Target     : "
                    f"{status['target_speed_kmh']:5.1f} "
                    f"km/h"
                ),

                (
                    f"Curve      : "
                    f"{status['curve_angle_deg']:5.1f} deg"
                ),

                (
                    f"Progress   : "
                    f"{status['progress']:5.1f} %"
                ),

                (
                    f"Point      : "
                    f"{status['route_index']}/"
                    f"{status['route_length'] - 1}"
                ),

                (
                    f"Goal dist  : "
                    f"{status['goal_distance']:5.1f} m"
                ),

                (
                    f"Red light  : "
                    f"{status['waiting_red_light']}"
                ),

                (
                    f"Stuck      : "
                    f"{status['stuck']}"
                ),
            ]

            y = 15

            for line in hud_lines:

                text = font.render(
                    line,
                    True,
                    (255, 255, 255),
                    (0, 0, 0),
                )

                screen.blit(
                    text,
                    (15, y),
                )

                y += 27

            # ------------------------------------------------
            # Map legend
            # ------------------------------------------------

            legend = [
                (
                    "Blue   : Planned route",
                    (40, 100, 255),
                ),
                (
                    "Green  : Driven trajectory",
                    (20, 180, 80),
                ),
                (
                    "Red    : Ego vehicle",
                    (230, 40, 40),
                ),
                (
                    "Orange : XML control point",
                    (255, 140, 0),
                ),
                (
                    "Purple : Goal",
                    (180, 40, 180),
                ),
            ]

            legend_y = 20

            for text_string, color in legend:

                text = small_font.render(
                    text_string,
                    True,
                    color,
                )

                screen.blit(
                    text,
                    (
                        half_width + 15,
                        legend_y,
                    ),
                )

                legend_y += 23

            pygame.display.flip()

            # ------------------------------------------------
            # Completion
            # ------------------------------------------------

            if status["completed"]:

                print()
                print("=" * 70)
                print("ROUTE SUCCESS")
                print("=" * 70)

                print(
                    f"Final progress : "
                    f"{status['progress']:.2f}%"
                )

                print(
                    f"Goal distance  : "
                    f"{status['goal_distance']:.2f} m"
                )

                print(
                    f"Final speed    : "
                    f"{status['speed_kmh']:.2f} km/h"
                )

                time.sleep(
                    2.0
                )

                running = False

            # ------------------------------------------------
            # Stuck
            # ------------------------------------------------

            elif status["stuck"]:

                print()
                print("=" * 70)
                print("ROUTE FAILED: VEHICLE STUCK")
                print("=" * 70)

                print(
                    f"Progress : "
                    f"{status['progress']:.2f}%"
                )

                print(
                    f"Point    : "
                    f"{status['route_index']}/"
                    f"{status['route_length'] - 1}"
                )

                running = False

            clock.tick(
                60
            )

    except KeyboardInterrupt:

        print(
            "\n[Exit] Keyboard interrupt."
        )

    finally:

        print(
            "\n[Cleanup] Cleaning up..."
        )

        if camera is not None:
            camera.destroy()

        if vehicle is not None:

            try:
                vehicle.destroy()
            except Exception:
                pass

        if (
            world is not None
            and
            original_settings is not None
        ):

            try:
                world.apply_settings(
                    original_settings
                )

            except Exception:
                pass

        pygame.quit()

        print(
            "[Cleanup] Done."
        )


if __name__ == "__main__":
    main()
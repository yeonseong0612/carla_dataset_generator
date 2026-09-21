from __future__ import annotations

import math
import time
from typing import Sequence

import carla

from agents.navigation.basic_agent import BasicAgent

from src.navigation.route import (
    find_nearest_route_index,
    get_route_progress,
    get_goal_distance,
    is_route_completed,
)


class RouteController:
    VALID_TRAFFIC_LIGHT_POLICIES = ("obey", "ignore")

    def __init__(
        self,
        vehicle: carla.Vehicle,
        dense_route: Sequence,
        cruise_speed: float = 30.0,
        min_curve_speed: float = 12.0,
        traffic_light_policy: str = "obey",
        search_window: int = 50,
        curve_lookahead_distance: float = 20.0,
        stuck_speed_threshold: float = 1.0,
        stuck_timeout: float = 30.0,
    ):
        if vehicle is None:
            raise ValueError("vehicle must not be None.")

        if not dense_route:
            raise ValueError("dense_route must not be empty.")

        if traffic_light_policy not in self.VALID_TRAFFIC_LIGHT_POLICIES:
            raise ValueError(f"Unknown traffic_light_policy: {traffic_light_policy}. Expected one of {self.VALID_TRAFFIC_LIGHT_POLICIES}.")

        if cruise_speed <= 0.0:
            raise ValueError("cruise_speed must be positive.")

        if min_curve_speed <= 0.0:
            raise ValueError("min_curve_speed must be positive.")

        if min_curve_speed > cruise_speed:
            raise ValueError("min_curve_speed must be <= cruise_speed.")

        self.vehicle = vehicle
        self.dense_route = dense_route

        self.cruise_speed = float(cruise_speed)
        self.min_curve_speed = float(min_curve_speed)

        self.traffic_light_policy = traffic_light_policy

        self.search_window = int(search_window)
        self.curve_lookahead_distance = float(curve_lookahead_distance)

        self.stuck_speed_threshold = float(stuck_speed_threshold)
        self.stuck_timeout = float(stuck_timeout)

        self.route_index = 0
        self.progress = 0.0
        self.goal_distance = float("inf")

        self.target_speed = self.cruise_speed
        self.curve_angle = 0.0

        self._stuck_since = None
        self._is_stuck = False

        self.agent = BasicAgent(self.vehicle, target_speed=self.cruise_speed)

        self._set_global_plan()
        self._configure_traffic_light_policy()

    def _set_global_plan(self):
        local_planner = self.agent.get_local_planner()

        local_planner.set_global_plan(self.dense_route, stop_waypoint_creation=True, clean_queue=True)


    def _configure_traffic_light_policy(self):
        ignore = self.traffic_light_policy == "ignore"

        if hasattr(self.agent, "ignore_traffic_lights"):
            self.agent.ignore_traffic_lights(active=ignore)

    def update_route_state(self):

        location = self.vehicle.get_location()

        self.route_index = find_nearest_route_index(vehicle_location=location, dense_route=self.dense_route, start_index=self.route_index, search_window=self.search_window)

        self.progress = get_route_progress(self.route_index, self.dense_route)

        self.goal_distance = get_goal_distance(location, self.dense_route)

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0


    def compute_curve_angle(self) -> float:

        if self.route_index >= len(self.dense_route) - 1:
            return 0.0

        reference_waypoint = self.dense_route[self.route_index][0]

        reference_yaw = reference_waypoint.transform.rotation.yaw
        
        max_heading_change = 0.0
        accumulated_distance = 0.0

        previous_location = reference_waypoint.transform.location

        for index in range(self.route_index + 1, len(self.dense_route)):
            waypoint = self.dense_route[index][0]

            location = waypoint.transform.location

            dx = location.x - previous_location.x
            dy = location.y - previous_location.y

            accumulated_distance += math.hypot(dx, dy)

            heading_change = abs(self._normalize_angle_deg(waypoint.transform.rotation.yaw - reference_yaw))

            max_heading_change = max(max_heading_change, heading_change)

            if (accumulated_distance >= self.curve_lookahead_distance):
                break

            previous_location = location

        return max_heading_change

    def compute_target_speed(self) -> float:

        curve_angle = self.compute_curve_angle()

        self.curve_angle = curve_angle

        if curve_angle < 5.0:
            target = self.cruise_speed

        elif curve_angle < 15.0:
            target = self.cruise_speed * 0.83

        elif curve_angle < 30.0:
            target = self.cruise_speed * 0.67

        else:
            target = self.min_curve_speed

        target = max(self.min_curve_speed, min(target, self.cruise_speed))

        self.target_speed = target

        return target

    def get_speed_kmh(self) -> float:

        velocity = self.vehicle.get_velocity()

        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        return speed_ms * 3.6

    def is_waiting_for_red_light(self) -> bool:

        if self.traffic_light_policy != "obey":
            return False

        try:
            if not self.vehicle.is_at_traffic_light():
                return False

            traffic_light = self.vehicle.get_traffic_light()

            if traffic_light is None:
                return False

            return (traffic_light.get_state() == carla.TrafficLightState.Red)

        except RuntimeError:
            return False

    def update_stuck_state(self, control):

        speed = self.get_speed_kmh()

        if self.is_waiting_for_red_light():
            self._stuck_since = None
            self._is_stuck = False
            return

        if control.brake > 0.1:
            self._stuck_since = None
            self._is_stuck = False
            return

        trying_to_move = control.throttle > 0.2 and control.brake < 0.1
        

        if (speed < self.stuck_speed_threshold and trying_to_move):
            now = time.monotonic()

            if self._stuck_since is None:
                self._stuck_since = now

            stopped_duration = now - self._stuck_since
        
            self._is_stuck = stopped_duration >= self.stuck_timeout

        else:
            self._stuck_since = None
            self._is_stuck = False

    def is_stuck(self) -> bool:
        return self._is_stuck

    def is_completed(self, progress_threshold: float = 99.0, goal_distance_threshold: float = 5.0) -> bool:

        return is_route_completed(
            vehicle_location=self.vehicle.get_location(),
            route_index=self.route_index,
            dense_route=self.dense_route,
            progress_threshold=progress_threshold,
            goal_distance_threshold=goal_distance_threshold,
        )

    def run_step(self) -> carla.VehicleControl:

        self.update_route_state()

        target_speed = self.compute_target_speed()

        self.agent.set_target_speed(target_speed)

        control = self.agent.run_step()

        self.update_stuck_state(control)

        return control

    def get_status(self) -> dict:

        return {
            "route_index": self.route_index,
            "route_length": len(self.dense_route),

            "progress": self.progress,
            "goal_distance": self.goal_distance,

            "speed_kmh": self.get_speed_kmh(),
            "target_speed_kmh": self.target_speed,

            "curve_angle_deg": self.curve_angle,

            "traffic_light_policy": self.traffic_light_policy,

            "waiting_red_light": self.is_waiting_for_red_light(),

            "stuck": self.is_stuck(),

            "completed": self.is_completed(),
        }
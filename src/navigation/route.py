from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner

DenseRoute = List[Tuple[carla.Waypoint, object]]

def load_route_from_xml(xml_path: str | Path, route_id: str | int):

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_route_id = str(route_id)

    for route in root.findall("route"):
        if route.attrib.get("id") != target_route_id:
            continue

        town = route.attrib.get("town")

        if not town:
            raise ValueError(f"Route {target_route_id} has no town attribute.")

        waypoint_root = route.find("waypoints")

        if waypoint_root is None:
            raise ValueError(f"Route {target_route_id} has no <waypoints> section.")

        control_points = []

        for position in waypoint_root.findall("position"):

            try:
                x = float(position.attrib["x"])
                y = float(position.attrib["y"])
                z = float(position.attrib.get("z", 0.0))

            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid waypoint in route {target_route_id}: {position.attrib}") from exc

            control_points.append(carla.Location(x=x, y=y, z=z))

        if len(control_points) < 2:
            raise ValueError(f"Route {target_route_id} requires at least 2 control points, got {len(control_points)}.")

        return town, control_points

    raise ValueError(f"Route id={target_route_id} not found in {xml_path}")


def project_control_points(carla_map: carla.Map, control_points: Sequence[carla.Location]) -> List[carla.Waypoint]:

    projected = []

    for index, location in enumerate(control_points):
        waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if waypoint is None:
            raise RuntimeError(f"Failed to project route control point {index}: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")

        projected.append(waypoint)

    return projected

def build_dense_route(carla_map: carla.Map, control_waypoints: Sequence[carla.Waypoint], sampling_resolution: float = 2.0) -> DenseRoute:

    if len(control_waypoints) < 2:
        raise ValueError("At least 2 control waypoints are required to build a dense route.")

    grp = GlobalRoutePlanner(carla_map, sampling_resolution,)

    dense_route = []

    for segment_index in range(len(control_waypoints) - 1):

        start = control_waypoints[segment_index].transform.location
        end = control_waypoints[segment_index + 1].transform.location

        segment = grp.trace_route(start, end)

        if not segment:
            raise RuntimeError(f"GlobalRoutePlanner failed for segment {segment_index}: ({start.x:.2f}, {start.y:.2f}) -> ({end.x:.2f}, {end.y:.2f})")

        if dense_route:
            segment = segment[1:]

        dense_route.extend(segment)

    if not dense_route:
        raise RuntimeError("Dense route generation returned an empty route.")

    return dense_route



def distance_2d(a: carla.Location, b: carla.Location) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)


def find_nearest_route_index(vehicle_location: carla.Location, dense_route: Sequence, start_index: int = 0, search_window: int = 50) -> int:

    if not dense_route:
        raise ValueError("dense_route is empty.")

    start_index = max(0, min(start_index, len(dense_route) - 1),)

    end_index = min(start_index + search_window, len(dense_route),)

    nearest_index = start_index
    nearest_distance = float("inf")

    for index in range(start_index, end_index):
        waypoint = dense_route[index][0]
        distance = distance_2d(vehicle_location, waypoint.transform.location,)

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    return nearest_index


def get_route_progress(route_index: int, dense_route: Sequence,) -> float:

    if not dense_route:
        return 0.0

    if len(dense_route) == 1:
        return 100.0

    route_index = max(0, min(route_index, len(dense_route) - 1),)

    return (route_index / (len(dense_route) - 1) * 100.0)


def get_goal_distance(vehicle_location: carla.Location, dense_route: Sequence) -> float:

    if not dense_route:
        return float("inf")

    goal_location = (dense_route[-1][0].transform.location)

    return distance_2d(vehicle_location, goal_location)


def is_route_completed(vehicle_location: carla.Location, route_index: int, dense_route: Sequence, progress_threshold: float = 99.0, goal_distance_threshold: float = 5.0) -> bool:

    if not dense_route:
        return False

    progress = get_route_progress(route_index, dense_route)

    goal_distance = get_goal_distance(vehicle_location, dense_route,)

    return (progress >= progress_threshold and goal_distance <= goal_distance_threshold)


def list_route_ids(xml_path):

    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    return [
        route.attrib["id"]
        for route in root.findall("route")
    ]
"""
debug_start.py

DEBUG-ONLY mid-route start (--debug-start-route-index) helpers.
Pure (no `carla` import) so they can be tested offline.

Production never passes a start index: select_route_start() then returns
dense_route[0], exactly what spawn_ego_at_route_start() always used.
"""

import math


def validate_debug_start_route_index(route_index, route_length):
    """
    Returns route_index as int, or raises ValueError. The last dense
    waypoint is excluded: starting there leaves no route to drive.
    """

    if isinstance(route_index, bool) or not isinstance(route_index, int):
        raise ValueError(f"debug start route index must be an int, got {route_index!r}")

    if route_index < 0:
        raise ValueError(f"debug start route index must be >= 0, got {route_index}")

    if route_index >= route_length - 1:
        raise ValueError(
            f"debug start route index {route_index} out of range: dense route has "
            f"{route_length} waypoints (valid 0..{route_length - 2})"
        )

    return route_index


def select_route_start(dense_route, debug_start_route_index=None):
    """(waypoint, route_index) the ego is spawned at."""

    if debug_start_route_index is None:
        return dense_route[0][0], 0

    index = validate_debug_start_route_index(debug_start_route_index, len(dense_route))

    return dense_route[index][0], index


def route_arc_length_at(dense_route, route_index):
    """Cumulative 2D arc length (m) from dense_route[0] to route_index."""

    total = 0.0

    for i in range(1, route_index + 1):
        a = dense_route[i - 1][0].transform.location
        b = dense_route[i][0].transform.location
        total += math.hypot(a.x - b.x, a.y - b.y)

    return total


def debug_start_banner(route_index, route_length, route_s, waypoint=None):
    bar = "!" * 64
    lines = [
        "",
        bar,
        "DEBUG MID-ROUTE START",
        "NOT FOR PRODUCTION DATA",
        f"route_index={route_index}/{route_length - 1}",
        f"route_s={route_s:.1f} m",
    ]

    if waypoint is not None:
        location = waypoint.transform.location
        lines.append(
            f"spawn=({location.x:.2f}, {location.y:.2f}) "
            f"yaw={waypoint.transform.rotation.yaw:.1f} "
            f"road={getattr(waypoint, 'road_id', '?')} lane={getattr(waypoint, 'lane_id', '?')}"
        )

    lines += [
        "Output of this run is a DEBUG sequence, not a production route.",
        "Use a separate --output-root; never merge it into the dataset.",
        bar,
        "",
    ]

    return "\n".join(lines)

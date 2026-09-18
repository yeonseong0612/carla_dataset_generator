"""
diagnostics.py

Phase 2.5: read-only telemetry helpers for diagnosing ego slowdown.
Nothing here changes spawn/despawn policy, controller behavior, or
TrafficManager settings -- it only observes and records values that
already exist (ego velocity, the VehicleControl actually applied,
CARLA's own traffic-light state) via CARLA's live actor API, reusing
src.navigation.controller.RouteController.get_status() rather than
reimplementing any of it.
"""

import carla

TRAFFIC_LIGHT_STATE_NAMES = {
    carla.TrafficLightState.Red: "Red",
    carla.TrafficLightState.Yellow: "Yellow",
    carla.TrafficLightState.Green: "Green",
    carla.TrafficLightState.Off: "Off",
    carla.TrafficLightState.Unknown: "Unknown",
}


def traffic_light_status(ego):
    """
    (is_at_traffic_light, light_state_name). light_state_name is "N/A"
    when ego isn't at a light or the light actor can't be resolved --
    never guessed.
    """

    try:
        is_at = bool(ego.is_at_traffic_light())
    except RuntimeError:
        return False, "N/A"

    if not is_at:
        return False, "N/A"

    try:
        traffic_light = ego.get_traffic_light()
    except RuntimeError:
        traffic_light = None

    if traffic_light is None:
        return True, "N/A"

    return True, TRAFFIC_LIGHT_STATE_NAMES.get(traffic_light.get_state(), "Unknown")


def control_snapshot(control):
    """
    The throttle/brake/steer/hand_brake actually applied this frame (the
    carla.VehicleControl returned by RouteController.run_step() /
    ego.apply_control()'s argument), or all-None if unavailable.
    """

    if control is None:
        return {"throttle": None, "brake": None, "steer": None, "hand_brake": None}

    return {
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "steer": float(control.steer),
        "hand_brake": bool(control.hand_brake),
    }


def ego_telemetry_row(frame, update_index, ego, route_controller, control, route_s=None):
    """
    One ego_telemetry.csv row. route_controller is a
    src.navigation.controller.RouteController (reused as-is via
    get_status(), not reimplemented); route_s overrides
    route_controller's own progress-derived distance when a
    DynamicSpawnManager's arc-length-table route_s is available (more
    directly comparable across the A/B/C cases, which don't all have a
    RouteController "route_index" pinned to the same reference frame).
    """

    status = route_controller.get_status()
    is_at_light, light_state = traffic_light_status(ego)
    control_fields = control_snapshot(control)

    return {
        "frame": frame,
        "update_index": update_index,
        "ego_route_s": route_s if route_s is not None else status["progress"],
        "speed_kmh": status["speed_kmh"],
        "target_speed_kmh": status["target_speed_kmh"],
        "throttle": control_fields["throttle"],
        "brake": control_fields["brake"],
        "steer": control_fields["steer"],
        "hand_brake": control_fields["hand_brake"],
        "is_at_traffic_light": is_at_light,
        "traffic_light_state": light_state,
        "waiting_red_light": status["waiting_red_light"],
        "stuck": status["stuck"],
    }

"""
stall_diagnostics.py

DEBUG-ONLY, read-only helpers for diagnosing a long ego stall (CLAUDE.md
Town10 Route 1 ~50% stall task). Nothing here changes controller,
Traffic Manager, spawn/despawn or actor-lifecycle behavior -- it only
classifies values that already exist and formats them for the log.

Pure (no `carla` import) so the offline analysis tool
(scripts/tools/analyze_route_stall.py) and the offline tests share the
exact same forward-blocker / stall-condition logic as the live
diagnostic in scripts/collect_dataset.py. Live CARLA objects are only
touched in the collect_* functions, duck-typed, and every RPC is wrapped
so a stale actor degrades to "N/A" instead of raising.
"""

import math


# ------------------------------------------------------------------
# Stall condition
# ------------------------------------------------------------------

STALL_SPEED_KMH = 1.0
STALL_MIN_TARGET_SPEED_KMH = 5.0

# Hysteresis for StallMonitor: a stall starts below STALL_SPEED_KMH but
# only ends once the ego is really moving again. Without it, BasicAgent's
# stop-and-go creep behind a queue (0 <-> ~2 km/h, brake 0.5 <-> throttle
# 0.75, observed live at Town10 route 1 index 248) resets the stall timer
# every second and [STALL-DIAG] never fires.
STALL_RELEASE_SPEED_KMH = 5.0

# First [STALL-DIAG] after this much SIMULATED time in the stall
# condition (20 Hz -> 100 ticks), then repeated every
# STALL_REPEAT_S while the stall lasts.
STALL_REPORT_AFTER_S = 5.0
STALL_REPEAT_S = 10.0

# [STUCK-STATE] periodic print cadence (simulated seconds).
STUCK_STATE_PRINT_INTERVAL_S = 1.0

# Forward-corridor half width for "in my lane" blockers. A CARLA lane is
# ~3-3.5 m wide, so +-2.0 m around ego's centerline keeps the own lane and
# excludes adjacent-lane traffic on straight road.
FORWARD_CORRIDOR_HALF_WIDTH_M = 2.0
FORWARD_SEARCH_MAX_M = 60.0


def classify_stop(speed_kmh, target_speed_kmh, red_light):
    """
    'moving'             speed >= STALL_SPEED_KMH
    'expected_red_light' stopped while waiting at a red light
    'target_zero'        stopped, but controller does not want to move
    'stall'              stopped although target > 5 km/h and no red light
    """

    if speed_kmh >= STALL_SPEED_KMH:
        return "moving"

    if red_light:
        return "expected_red_light"

    if target_speed_kmh <= STALL_MIN_TARGET_SPEED_KMH:
        return "target_zero"

    return "stall"


def is_stall_condition(speed_kmh, target_speed_kmh, red_light):
    return classify_stop(speed_kmh, target_speed_kmh, red_light) == "stall"


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def heading_vectors(yaw_deg):
    """(forward, right) unit vectors in CARLA's left-handed x/y plane."""

    yaw = math.radians(yaw_deg)
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    return forward, right


def relative_longitudinal_lateral(ego_xy, ego_yaw_deg, actor_xy):
    forward, right = heading_vectors(ego_yaw_deg)
    dx = actor_xy[0] - ego_xy[0]
    dy = actor_xy[1] - ego_xy[1]
    return dx * forward[0] + dy * forward[1], dx * right[0] + dy * right[1]


def find_nearest_forward_actor(
    ego_xy,
    ego_yaw_deg,
    candidates,
    max_lateral_m=FORWARD_CORRIDOR_HALF_WIDTH_M,
    max_longitudinal_m=FORWARD_SEARCH_MAX_M,
):
    """
    candidates: iterable of (key, x, y). Only actors strictly in the
    forward half-plane (dot(actor - ego, ego_forward) > 0) are considered;
    max_lateral_m=None disables the corridor filter (plain forward
    half-plane). Returns the closest by longitudinal distance as
    {"key", "longitudinal_m", "lateral_m", "distance_m"}, or None.
    """

    best = None

    for key, x, y in candidates:
        longitudinal, lateral = relative_longitudinal_lateral(ego_xy, ego_yaw_deg, (x, y))

        if longitudinal <= 0.0:
            continue

        if max_lateral_m is not None and abs(lateral) > max_lateral_m:
            continue

        if max_longitudinal_m is not None and longitudinal > max_longitudinal_m:
            continue

        if best is None or longitudinal < best["longitudinal_m"]:
            best = {
                "key": key,
                "longitudinal_m": longitudinal,
                "lateral_m": lateral,
                "distance_m": math.hypot(x - ego_xy[0], y - ego_xy[1]),
            }

    return best


# ------------------------------------------------------------------
# Stall monitor (simulation-clock based)
# ------------------------------------------------------------------

class StallMonitor:
    """
    Counts SIMULATED time in the stall condition. update() returns True on
    the ticks a [STALL-DIAG] block should be printed: first after
    report_after_s, then every repeat_s while the stall continues.

    Entry uses classify_stop() ('stall'); once in a stall, creeping below
    release_speed_kmh (target still > 5 km/h, no red light) keeps it.
    """

    def __init__(self, report_after_s=STALL_REPORT_AFTER_S, repeat_s=STALL_REPEAT_S, release_speed_kmh=STALL_RELEASE_SPEED_KMH):
        self.report_after_s = float(report_after_s)
        self.repeat_s = float(repeat_s)
        self.release_speed_kmh = float(release_speed_kmh)
        self.stall_start_time = None
        self.next_report_at = None
        self.last_classification = "moving"
        self.duration_s = 0.0
        self.reports = 0

    def update(self, sim_time_s, speed_kmh, target_speed_kmh, red_light):
        self.last_classification = classify_stop(speed_kmh, target_speed_kmh, red_light)

        creeping = (
            self.stall_start_time is not None
            and self.last_classification == "moving"
            and speed_kmh < self.release_speed_kmh
            and target_speed_kmh > STALL_MIN_TARGET_SPEED_KMH
            and not red_light
        )

        if creeping:
            self.last_classification = "stall(creep)"

        elif self.last_classification != "stall":
            self.stall_start_time = None
            self.next_report_at = None
            self.duration_s = 0.0
            return False

        if self.stall_start_time is None:
            self.stall_start_time = float(sim_time_s)
            self.next_report_at = self.report_after_s

        self.duration_s = float(sim_time_s) - self.stall_start_time

        if self.duration_s + 1e-9 >= self.next_report_at:
            self.next_report_at += self.repeat_s
            self.reports += 1
            return True

        return False


# ------------------------------------------------------------------
# Stuck-detector introspection
# ------------------------------------------------------------------

def extract_stuck_state(controller, now_monotonic):
    """
    Read-only snapshot of RouteController's own stuck detector. The
    controller's timer is time.monotonic() (WALL clock) -- reported as
    such, never converted.
    """

    since = getattr(controller, "_stuck_since", None)

    return {
        "timer_active": since is not None,
        "elapsed_wall_s": (now_monotonic - since) if since is not None else 0.0,
        "threshold_kmh": float(controller.stuck_speed_threshold),
        "timeout_s": float(controller.stuck_timeout),
        "is_stuck": bool(getattr(controller, "_is_stuck", False)),
        "reset_reason": getattr(controller, "last_stuck_reset_reason", None),
        "clock": "wall(time.monotonic)",
    }


class StuckStateLogger:
    """
    Decides when to print [STUCK-STATE]: every print_interval_s of
    SIMULATED time, or immediately when timer_active / is_stuck flips.
    Also tracks the simulated time since the wall-clock timer started so
    the two clocks can be compared in the log.
    """

    def __init__(self, print_interval_s=STUCK_STATE_PRINT_INTERVAL_S):
        self.print_interval_s = float(print_interval_s)
        self.last_print_time = None
        self.last_key = None
        self.timer_sim_start = None

    def should_print(self, sim_time_s, state):
        if state["timer_active"]:
            if self.timer_sim_start is None:
                self.timer_sim_start = float(sim_time_s)
        else:
            self.timer_sim_start = None

        key = (state["timer_active"], state["is_stuck"])
        changed = key != self.last_key
        due = self.last_print_time is None or (sim_time_s - self.last_print_time) >= self.print_interval_s - 1e-9

        if changed or due:
            self.last_key = key
            self.last_print_time = float(sim_time_s)
            return True

        return False

    def elapsed_sim_s(self, sim_time_s):
        return 0.0 if self.timer_sim_start is None else float(sim_time_s) - self.timer_sim_start


def format_stuck_state(state, speed_kmh, elapsed_sim_s):
    return (
        f"[STUCK-STATE] elapsed={state['elapsed_wall_s']:.1f}s(wall) "
        f"elapsed_sim={elapsed_sim_s:.1f}s speed={speed_kmh:.2f}km/h "
        f"threshold={state['threshold_kmh']:.1f}km/h timeout={state['timeout_s']:.0f}s "
        f"timer_active={state['timer_active']} is_stuck={state['is_stuck']} "
        f"reset_reason={state['reset_reason']} clock={state['clock']}"
    )


# ------------------------------------------------------------------
# Live collection (duck-typed CARLA objects, every RPC guarded)
# ------------------------------------------------------------------

def _safe(fn, default="N/A"):
    try:
        return fn()
    except RuntimeError:
        return default


def _speed_kmh(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2) * 3.6


def _waypoint_ids(carla_map, location):
    waypoint = _safe(lambda: carla_map.get_waypoint(location), None)

    if waypoint is None:
        return "N/A", "N/A", "N/A"

    return waypoint.road_id, waypoint.lane_id, waypoint.is_junction


def _traffic_light(actor):
    at_light = _safe(lambda: bool(actor.is_at_traffic_light()))
    state = _safe(lambda: str(actor.get_traffic_light_state()))
    return at_light, state


def _control_fields(control):
    if control is None or control == "N/A":
        return {"throttle": "N/A", "brake": "N/A", "steer": "N/A", "hand_brake": "N/A"}

    return {
        "throttle": round(float(control.throttle), 3),
        "brake": round(float(control.brake), 3),
        "steer": round(float(control.steer), 3),
        "hand_brake": bool(control.hand_brake),
    }


def describe_vehicle(actor, carla_map, logical_id=None, managed=None):
    transform = _safe(actor.get_transform, None)
    velocity = _safe(actor.get_velocity, None)
    at_light, light_state = _traffic_light(actor)
    road_id, lane_id, junction = (
        _waypoint_ids(carla_map, transform.location) if transform is not None else ("N/A",) * 3
    )

    return {
        "id": actor.id,
        "logical_id": logical_id,
        "type": actor.type_id,
        "speed_kmh": round(_speed_kmh(velocity), 3) if velocity is not None else "N/A",
        "transform": (
            f"x={transform.location.x:.2f} y={transform.location.y:.2f} "
            f"z={transform.location.z:.2f} yaw={transform.rotation.yaw:.1f}"
            if transform is not None else "N/A"
        ),
        "road_id": road_id,
        "lane_id": lane_id,
        "is_junction": junction,
        "is_alive": _safe(lambda: bool(actor.is_alive)),
        "managed": managed,
        # CARLA exposes no public "is this actor registered with TM"
        # query; managed actors are all spawned through
        # configure_actor_traffic_manager() (set_autopilot(True, port)).
        "autopilot_tm": "set_autopilot(True) at spawn (not queryable)" if managed else "unknown",
        "at_traffic_light": at_light,
        "traffic_light_state": light_state,
        "light_state": _safe(lambda: str(actor.get_light_state())),
        # For a TM-driven vehicle this is the last command TM applied.
        "control": _control_fields(_safe(actor.get_control, None)),
    }


def forward_vehicle_chain(ego, vehicle_actors, depth=3):
    """
    ego -> nearest forward in-corridor vehicle -> that vehicle's nearest
    forward vehicle -> ... (queue / deadlock evidence). vehicle_actors:
    iterable of live actors. Returns a list of (actor, hit_dict).
    """

    positions = {}

    for actor in vehicle_actors:
        transform = _safe(actor.get_transform, None)

        if transform is not None:
            positions[actor.id] = (actor, transform)

    chain = []
    current = ego
    visited = {ego.id}

    for _ in range(depth):
        current_transform = _safe(current.get_transform, None)

        if current_transform is None:
            break

        hit = find_nearest_forward_actor(
            (current_transform.location.x, current_transform.location.y),
            current_transform.rotation.yaw,
            [
                (actor_id, transform.location.x, transform.location.y)
                for actor_id, (_actor, transform) in positions.items()
                if actor_id not in visited
            ],
        )

        if hit is None:
            break

        actor = positions[hit["key"]][0]
        chain.append((actor, hit))
        visited.add(actor.id)
        current = actor

    return chain


def format_stall_report(header, ego_info, chain_infos, stuck_state):
    lines = ["", "=" * 72, "[STALL-DIAG]"]
    lines += [f"{key}={value}" for key, value in header.items()]
    lines.append("")
    lines.append("Ego:")
    lines += [f"  {key}={value}" for key, value in ego_info.items()]
    lines.append("")

    if not chain_infos:
        lines.append(
            f"Nearest forward vehicle: NONE within {FORWARD_SEARCH_MAX_M:.0f} m "
            f"(corridor +-{FORWARD_CORRIDOR_HALF_WIDTH_M:.1f} m)"
        )

    for level, (hit, info) in enumerate(chain_infos):
        title = "Nearest forward vehicle" if level == 0 else f"  -> its forward vehicle (level {level + 1})"
        lines.append(f"{title}:")
        lines.append(
            f"  distance={hit['distance_m']:.2f} longitudinal={hit['longitudinal_m']:.2f} "
            f"lateral={hit['lateral_m']:.2f}"
        )
        lines += [f"  {key}={value}" for key, value in info.items()]

    lines.append("")
    lines.append("Stuck detector:")
    lines += [f"  {key}={value}" for key, value in stuck_state.items()]
    lines.append("=" * 72)

    return "\n".join(lines)

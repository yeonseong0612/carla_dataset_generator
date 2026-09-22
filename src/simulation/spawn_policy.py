import math

import carla
import numpy as np

from src.navigation.route import distance_2d, find_nearest_route_index
from src.simulation.pedestrian import get_walker_blueprints
from src.simulation.traffic import configure_actor_traffic_manager, get_traffic_blueprints, prepare_blueprint, is_bus_blueprint


LANE_SELECTION_WEIGHTS = {"reference": 0.50, "left": 0.25, "right": 0.25}

MAX_SIDEWALK_OFFSET_M = 15.0

SPAWN_HEIGHT_OFFSET_M = 0.5

VEHICLE_LIKE_CATEGORIES = ("vehicle", "motorcyclist", "cyclist")


def sample_route_distance(rng, cfg, max_rejections=200):

    for _ in range(max_rejections):
        s = float(rng.gamma(cfg.SPAWN.GAMMA_SHAPE, cfg.SPAWN.GAMMA_SCALE))

        if cfg.SPAWN.MIN_DISTANCE <= s <= cfg.SPAWN.MAX_DISTANCE:
            return s

    return None


def reference_waypoint_at_distance(dense_route, s):

    if not dense_route:
        return None, None

    accumulated = 0.0
    previous_location = dense_route[0][0].transform.location

    for waypoint, _road_option in dense_route[1:]:
        location = waypoint.transform.location
        accumulated += distance_2d(previous_location, location)
        previous_location = location

        if accumulated >= s:
            return waypoint, accumulated

    return None, None

def driving_lane_candidates(reference_waypoint):

    candidates = []

    if reference_waypoint.lane_type == carla.LaneType.Driving:
        candidates.append(("reference", reference_waypoint))

    left = reference_waypoint.get_left_lane()

    if left is not None and left.lane_type == carla.LaneType.Driving:
        candidates.append(("left", left))

    right = reference_waypoint.get_right_lane()

    if right is not None and right.lane_type == carla.LaneType.Driving:
        candidates.append(("right", right))

    return candidates


def choose_lane(rng, candidates):
    if not candidates:
        return None

    weights = np.array(
        [LANE_SELECTION_WEIGHTS[name] for name, _waypoint in candidates],
        dtype=np.float64,
    )
    weights = weights / weights.sum()

    index = int(rng.choice(len(candidates), p=weights))

    return candidates[index][1]


def sidewalk_waypoint_near(carla_map, location, max_offset=MAX_SIDEWALK_OFFSET_M):
    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Sidewalk,
    )

    if waypoint is None or waypoint.lane_type != carla.LaneType.Sidewalk:
        return None

    if waypoint.transform.location.distance(location) > max_offset:
        return None

    return waypoint


def pedestrian_local_destination(carla_map, location, walk_distance=10.0):

    waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Sidewalk)

    if waypoint is None:
        return None

    for candidates in (waypoint.next(walk_distance), waypoint.previous(walk_distance)):
        if candidates:
            return candidates[0].transform.location

    return None

def far_enough(location, other_locations, min_distance):
    return all(location.distance(other) >= min_distance for other in other_locations)


def vehicle_spacing_ok(location, road_id, lane_id, s, placed_vehicle_records, same_lane_spacing, cross_lane_spacing):

    for other in placed_vehicle_records:
        if other["road_id"] == road_id and other["lane_id"] == lane_id:
            if abs(s - other["s"]) < same_lane_spacing:
                return False
        else:
            if location.distance(other["location"]) < cross_lane_spacing:
                return False

    return True


def same_lane_front_gap_ok(candidate_road_id, candidate_lane_id, relative_s, ego_road_id, ego_lane_id, min_gap_m):
    """
    Spawn-time-only guard: reject a moving-traffic candidate (car/truck/
    van/motorcycle/bicycle) that would be placed in ego's own current
    driving lane, strictly ahead of ego, closer than min_gap_m -- measured
    as route-relative longitudinal distance (relative_s, matching
    vehicle_spacing_ok's "s" convention), not Euclidean distance. Adjacent/
    opposite lanes and behind-ego candidates (relative_s <= 0) are always
    accepted here -- this only shrinks the initial/replenishment spawn
    pool, it never touches an actor once it exists (see CanonicalBackground
    Traffic.update(), which never despawns for closeness).
    """

    if ego_road_id is None:
        return True

    if candidate_road_id != ego_road_id or candidate_lane_id != ego_lane_id:
        return True

    if relative_s <= 0.0:
        return True

    return relative_s >= min_gap_m

class GammaSpawnPolicy:

    def __init__(self, world, ego, dense_route, traffic_manager, cfg):
        self.world = world
        self.ego = ego
        self.dense_route = dense_route
        self.traffic_manager = traffic_manager
        self.cfg = cfg

        self.rng = np.random.default_rng(cfg.SPAWN.SEED)

        self.carla_map = world.get_map()
        self.ego_location = ego.get_location()

        ego_waypoint = self.carla_map.get_waypoint(
            self.ego_location, project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        self.ego_road_id = ego_waypoint.road_id if ego_waypoint is not None else None
        self.ego_lane_id = ego_waypoint.lane_id if ego_waypoint is not None else None

        self.vehicle_pools = get_traffic_blueprints(world)

        self.vehicle_pools["vehicle"] = [
            blueprint for blueprint in self.vehicle_pools["vehicle"] if not is_bus_blueprint(blueprint)
        ]
        self.walker_blueprints = get_walker_blueprints(world)
        self.walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

        self.placed_vehicle_records = []
        self.placed_pedestrian_locations = []

        self.spawn_records = []

        self.proposed_s = []

    def _try_spawn_vehicle_like(self, category):
        blueprint_pool = self.vehicle_pools.get(category, [])

        if not blueprint_pool:
            return None

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            s = sample_route_distance(self.rng, self.cfg)

            if s is None:
                continue

            self.proposed_s.append(s)

            reference_waypoint, actual_distance = reference_waypoint_at_distance(self.dense_route, s)

            if reference_waypoint is None:
                continue

            candidates = driving_lane_candidates(reference_waypoint)
            lane_waypoint = choose_lane(self.rng, candidates)

            if lane_waypoint is None:
                continue

            if category in VEHICLE_LIKE_CATEGORIES and not same_lane_front_gap_ok(
                lane_waypoint.road_id, lane_waypoint.lane_id, actual_distance,
                self.ego_road_id, self.ego_lane_id,
                self.cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M,
            ):
                continue

            transform = carla.Transform(
                lane_waypoint.transform.location,
                lane_waypoint.transform.rotation,
            )
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(self.ego_location) < self.cfg.SPAWN.MIN_EGO_SPACING:
                continue

            if not vehicle_spacing_ok(
                transform.location, lane_waypoint.road_id, lane_waypoint.lane_id, s,
                self.placed_vehicle_records,
                self.cfg.SPAWN.MIN_VEHICLE_SPACING,
                self.cfg.SPAWN.MIN_CROSS_LANE_SPACING,
            ):
                continue

            blueprint = prepare_blueprint(self.rng.choice(blueprint_pool), self.rng)
            actor = self.world.try_spawn_actor(blueprint, transform)

            if actor is None:
                continue

            configure_actor_traffic_manager(actor, self.traffic_manager, self.cfg)

            self.placed_vehicle_records.append({
                "road_id": lane_waypoint.road_id,
                "lane_id": lane_waypoint.lane_id,
                "s": s,
                "location": transform.location,
            })
            self._record(
                category, actor.id, s, actual_distance, lane_waypoint, transform.location,
            )

            return actor

        print(f"[Spawn] {category} failed after {self.cfg.SPAWN.MAX_ATTEMPTS} attempts")

        return None


    def _try_spawn_pedestrian(self):
        if not self.walker_blueprints:
            return None

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            s = sample_route_distance(self.rng, self.cfg)

            if s is None:
                continue

            self.proposed_s.append(s)

            reference_waypoint, actual_distance = reference_waypoint_at_distance(self.dense_route, s)

            if reference_waypoint is None:
                continue

            sidewalk_waypoint = sidewalk_waypoint_near(
                self.carla_map, reference_waypoint.transform.location,
            )

            if sidewalk_waypoint is None:
                continue

            transform = carla.Transform(
                sidewalk_waypoint.transform.location,
                sidewalk_waypoint.transform.rotation,
            )
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(self.ego_location) < self.cfg.SPAWN.MIN_EGO_SPACING:
                continue

            if not far_enough(transform.location, self.placed_pedestrian_locations, self.cfg.SPAWN.MIN_PEDESTRIAN_SPACING):
                continue

            blueprint = self.rng.choice(self.walker_blueprints)

            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")

            walker = self.world.try_spawn_actor(blueprint, transform)

            if walker is None:
                continue

            controller = self.world.try_spawn_actor(
                self.walker_controller_bp, carla.Transform(), attach_to=walker,
            )

            if controller is None:
                walker.destroy()
                continue

            speed = 1.4

            if blueprint.has_attribute("speed"):
                values = blueprint.get_attribute("speed").recommended_values

                if len(values) > 1:
                    speed = float(values[1])

            self.placed_pedestrian_locations.append(transform.location)
            self._record(
                "pedestrian", walker.id, s, actual_distance, sidewalk_waypoint, transform.location,
            )

            return walker, controller, speed

        print(f"[Spawn] pedestrian failed after {self.cfg.SPAWN.MAX_ATTEMPTS} attempts")

        return None


    def _record(self, category, actor_id, sampled_s, actual_distance, waypoint, location):
        record = {
            "actor_id": int(actor_id),
            "category": category,
            "sampled_s": float(sampled_s),
            "actual_distance": float(actual_distance),
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }

        self.spawn_records.append(record)

        print(
            f"[Spawn] {category} id={actor_id} "
            f"s={sampled_s:.1f}m actual={actual_distance:.1f}m "
            f"road={waypoint.road_id} lane={waypoint.lane_id}"
        )

    def run(self, n_vehicles=None, n_motorcycles=None, n_bicycles=None, n_pedestrians=None):
        """
        Spawn once. Any of the n_* counts default to cfg.SPAWN.N_*, and
        can be passed as 0 to skip a whole category (used for
        --no-traffic / --no-pedestrians).
        """

        requested = {
            "vehicle": self.cfg.SPAWN.N_VEHICLES if n_vehicles is None else n_vehicles,
            "motorcyclist": self.cfg.SPAWN.N_MOTORCYCLES if n_motorcycles is None else n_motorcycles,
            "cyclist": self.cfg.SPAWN.N_BICYCLES if n_bicycles is None else n_bicycles,
            "pedestrian": self.cfg.SPAWN.N_PEDESTRIANS if n_pedestrians is None else n_pedestrians,
        }

        traffic_actors = {"vehicle": [], "cyclist": [], "motorcyclist": []}
        walkers, walker_controllers, walker_speeds = [], [], []

        category_tokens = []

        for category in VEHICLE_LIKE_CATEGORIES:
            category_tokens.extend([category] * requested[category])

        self.rng.shuffle(category_tokens)

        for category in category_tokens:
            actor = self._try_spawn_vehicle_like(category)

            if actor is not None:
                traffic_actors[category].append(actor)

        for _ in range(requested["pedestrian"]):
            result = self._try_spawn_pedestrian()

            if result is not None:
                walker, controller, speed = result
                walkers.append(walker)
                walker_controllers.append(controller)
                walker_speeds.append(speed)

        return {
            "requested": requested,
            "traffic_actors": traffic_actors,
            "walkers": walkers,
            "walker_controllers": walker_controllers,
            "walker_speeds": walker_speeds,
            "spawn_records": self.spawn_records,
            "proposed_s": self.proposed_s,
            "rng": self.rng,
        }


def spawn_actors_gamma_policy(world, ego, dense_route, traffic_manager, cfg, **counts):

    policy = GammaSpawnPolicy(world, ego, dense_route, traffic_manager, cfg)

    return policy.run(**counts)


# ================================================================
# Phase 2 -- Dynamic density maintenance
# ================================================================
#
#   ego route progress
#       -> current managed-actor relative_s (recomputed every update,
#          never trusts spawn-time sampled_s)
#       -> despawn actors that fell far behind
#       -> category x distance-bin occupancy vs. Gamma-shaped target
#       -> deficient bins only, bin-constrained Gamma spawn
#       -> deterministic category interleaving (same rng stream)
#
# Reuses Phase 1's lane/sidewalk/spacing/blueprint functions verbatim;
# only the "where does s come from" and "how many actors to place"
# parts are new.

MAX_ROUTE_PROJECTION_OFFSET_M = 20.0
ROUTE_INDEX_SEARCH_WINDOW = 80

# Phase 2.5 PART B: escalating recovery search windows for an actor with
# a growing consecutive-projection-failure streak (see
# project_actor_with_recovery()). A stale route_index_hint never
# advances on its own once an actor drifts off-corridor -- find_nearest_
# route_index() only searches forward from that hint -- so a persistent
# failure gets progressively wider (and eventually whole-route) search
# attempts instead of retrying the same too-narrow window forever.
EXPANDED_ROUTE_INDEX_SEARCH_WINDOW = 200
FULL_SEARCH_FAILURE_STREAK = 6

DYNAMIC_CATEGORIES = ("vehicle", "motorcyclist", "cyclist", "pedestrian")

CATEGORY_TARGET_CFG_KEYS = {
    "vehicle": "N_VEHICLES",
    "motorcyclist": "N_MOTORCYCLES",
    "cyclist": "N_BICYCLES",
    "pedestrian": "N_PEDESTRIANS",
}

# Phase 2.5 diagnostics (PART E): mutually-exclusive registry states.
STATE_NAMES = (
    "behind_cleanup",
    "near_ego",
    "active_forward",
    "transition_forward",
    "forward_cleanup",
    "projection_failed",
)


def classify_relative_s_state(relative_s, cfg):
    if relative_s < -cfg.SPAWN.DESPAWN_BEHIND_DISTANCE:
        return "behind_cleanup"

    if relative_s < cfg.SPAWN.MIN_DISTANCE:
        return "near_ego"

    if relative_s <= cfg.SPAWN.MAX_DISTANCE:
        return "active_forward"

    if relative_s <= cfg.SPAWN.FORWARD_CLEANUP_DISTANCE:
        return "transition_forward"

    return "forward_cleanup"


def build_route_arc_length_table(dense_route):
    """
    Cumulative 2D arc length at each dense_route index (table[0] == 0),
    precomputed once so ego/actor route progress can be read back in
    O(1) after finding a nearby index, instead of re-walking the route
    from its start on every lookup.
    """

    table = [0.0]

    for i in range(1, len(dense_route)):
        previous_location = dense_route[i - 1][0].transform.location
        location = dense_route[i][0].transform.location
        table.append(table[-1] + distance_2d(previous_location, location))

    return table


def route_progress_at_location(location, dense_route, arc_length_table, start_index, search_window=ROUTE_INDEX_SEARCH_WINDOW, max_offset=MAX_ROUTE_PROJECTION_OFFSET_M):
    """
    (route_s, new_hint_index, offset) for `location`, searching forward
    from start_index (reusing the previous call's index as a hint
    instead of scanning the whole route). route_s is None if the
    nearest route point is farther than max_offset away -- location
    isn't really on this route corridor anymore, so the caller should
    treat progress as unknown for this update rather than trust a bogus
    projection. offset (the actual nearest-point distance) is always
    returned, even on failure, for diagnostics.
    """

    index = find_nearest_route_index(
        location, dense_route, start_index=start_index, search_window=search_window,
    )

    waypoint_location = dense_route[index][0].transform.location
    offset = location.distance(waypoint_location)

    if offset > max_offset:
        return None, index, offset

    return arc_length_table[index], index, offset


def project_actor_with_recovery(location, dense_route, arc_length_table, previous_hint, failure_streak, max_offset):
    """
    route_progress_at_location() with an escalating search window keyed
    off the actor's own consecutive-failure streak:

        streak 0-2: normal window (ROUTE_INDEX_SEARCH_WINDOW), from the
                    cached hint -- the common case, cheap.
        streak 3-5: expanded window (EXPANDED_ROUTE_INDEX_SEARCH_WINDOW),
                    still from the cached hint -- the actor may just need
                    a slightly wider look, e.g. after a lane change.
        streak >=6: full-route fallback, searched from index 0 (NOT the
                    stale hint, which is exactly what's suspected of
                    being wrong by this point) -- rare by construction,
                    since most actors project successfully well before
                    reaching this streak.

    Returns (route_s, new_hint_index, offset, used_full_search).
    """

    if failure_streak >= FULL_SEARCH_FAILURE_STREAK:
        route_s, new_hint, offset = route_progress_at_location(
            location, dense_route, arc_length_table, start_index=0,
            search_window=len(dense_route), max_offset=max_offset,
        )
        return route_s, new_hint, offset, True

    search_window = EXPANDED_ROUTE_INDEX_SEARCH_WINDOW if failure_streak >= 3 else ROUTE_INDEX_SEARCH_WINDOW

    route_s, new_hint, offset = route_progress_at_location(
        location, dense_route, arc_length_table, previous_hint,
        search_window=search_window, max_offset=max_offset,
    )

    return route_s, new_hint, offset, False


# ----------------------------------------------------------------
# Truncated-Gamma bin targets (analytic-free: deterministic numeric
# integration of the Gamma PDF -- no SciPy, no RNG involved)
# ----------------------------------------------------------------

def gamma_log_pdf(x, shape, scale):
    return (shape - 1.0) * np.log(x) - x / scale - math.lgamma(shape) - shape * np.log(scale)


def gamma_cdf_numeric(x, shape, scale, grid_points=4000):
    if x <= 0.0:
        return 0.0

    grid = np.linspace(1e-6, x, grid_points)
    pdf = np.exp(gamma_log_pdf(grid, shape, scale))

    # np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz

    return float(trapezoid(pdf, grid))


def compute_bin_edges_and_probabilities(cfg):
    """
    Bin edges over [MIN_DISTANCE, MAX_DISTANCE] at BIN_SIZE resolution,
    plus each bin's probability under Gamma(shape, scale) truncated to
    that same range (P(bin_i | MIN_DISTANCE <= s <= MAX_DISTANCE)).
    """

    bin_edges = np.arange(
        cfg.SPAWN.MIN_DISTANCE, cfg.SPAWN.MAX_DISTANCE + 1e-6, cfg.SPAWN.BIN_SIZE,
    )

    cdf_at_edges = np.array([
        gamma_cdf_numeric(x, cfg.SPAWN.GAMMA_SHAPE, cfg.SPAWN.GAMMA_SCALE)
        for x in bin_edges
    ])

    truncated_mass = cdf_at_edges[-1] - cdf_at_edges[0]
    bin_probabilities = (cdf_at_edges[1:] - cdf_at_edges[:-1]) / truncated_mass

    return bin_edges, bin_probabilities


def largest_remainder_allocation(total, probabilities):
    """
    Integer per-bin target counts that sum exactly to `total`, allocated
    proportionally to `probabilities` via the largest-remainder method
    (deterministic -- no random rounding).
    """

    raw = np.asarray(probabilities, dtype=np.float64) * total
    floors = np.floor(raw).astype(int)

    remaining = int(total - floors.sum())

    if remaining > 0:
        order = np.argsort(-(raw - floors))

        for index in order[:remaining]:
            floors[index] += 1

    return floors


def sample_bin_constrained_distance(rng, bin_low, bin_high, cfg, max_rejections=200):
    """
    Same Gamma(shape, scale) as sample_route_distance(), but only
    accepting draws that land inside [bin_low, bin_high] -- used to
    replenish one specific deficient distance bin instead of drawing
    from the whole [MIN_DISTANCE, MAX_DISTANCE] range again.
    """

    for _ in range(max_rejections):
        s = float(rng.gamma(cfg.SPAWN.GAMMA_SHAPE, cfg.SPAWN.GAMMA_SCALE))

        if bin_low <= s <= bin_high:
            return s

    return None


# ----------------------------------------------------------------
# Managed-actor registry + per-update maintenance
# ----------------------------------------------------------------

class DynamicSpawnManager:
    """
    Phase 2: keeps the ego-relative [MIN_DISTANCE, MAX_DISTANCE] forward
    corridor populated according to the same Gamma-shaped distance
    profile Phase 1 used for the initial spawn, as ego drives.

    Only actors this manager (or the initial GammaSpawnPolicy run it was
    seeded from) spawned are tracked/managed -- it never touches other
    world actors.
    """

    def __init__(self, world, ego, dense_route, traffic_manager, cfg, rng, category_totals=None):
        """
        category_totals: optional {category: total} override (e.g. to
        zero out a category for --no-traffic/--no-pedestrians); any
        category left out defaults to cfg.SPAWN.N_*.
        """

        self.world = world
        self.ego = ego
        self.dense_route = dense_route
        self.traffic_manager = traffic_manager
        self.cfg = cfg
        self.rng = rng

        self.carla_map = world.get_map()
        self.vehicle_pools = get_traffic_blueprints(world)
        self.walker_blueprints = get_walker_blueprints(world)
        self.walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

        self.arc_length_table = build_route_arc_length_table(dense_route)
        self.bin_edges, self.bin_probabilities = compute_bin_edges_and_probabilities(cfg)
        self.n_bins = len(self.bin_probabilities)

        category_totals = category_totals or {}

        self.category_bin_targets = {
            category: largest_remainder_allocation(
                category_totals.get(category, getattr(cfg.SPAWN, CATEGORY_TARGET_CFG_KEYS[category])),
                self.bin_probabilities,
            )
            for category in DYNAMIC_CATEGORIES
        }

        self.ego_route_index = 0
        self.managed_actors = []

        self.update_index = 0
        self.route_projection_failures = 0

        self.total_spawned = 0
        self.total_despawned = 0
        self.total_failed = 0
        self.total_over_target_spawn_attempts = 0

        # Phase 2.5 fix counters (diagnostic + validation, PART H).
        self.total_projection_failure_despawns = 0
        self.total_projection_full_search_attempts = 0
        self.total_projection_full_search_recoveries = 0
        self.total_same_lane_spawn_rejections = 0
        self.total_blocked_bin_skips = 0
        self.total_category_cap_skips = 0

        # Per-update snapshots, for reporting/visualization.
        self.history = []

    # ------------------------------------------------------------
    # Seeding the registry from the Phase 1 initial spawn
    # ------------------------------------------------------------

    def register_initial_actors(self, spawn_result):
        """
        Adopt the actors GammaSpawnPolicy.run() already placed (and
        recorded in spawn_result["spawn_records"]) as managed actors, by
        matching actor ids to the live actor/controller references.
        """

        vehicle_like_by_id = {}

        for category in VEHICLE_LIKE_CATEGORIES:
            for actor in spawn_result["traffic_actors"].get(category, []):
                vehicle_like_by_id[actor.id] = actor

        walker_by_id = {actor.id: actor for actor in spawn_result["walkers"]}
        controller_by_walker_id = {
            walker.id: controller
            for walker, controller in zip(spawn_result["walkers"], spawn_result["walker_controllers"])
        }
        speed_by_walker_id = {
            walker.id: speed
            for walker, speed in zip(spawn_result["walkers"], spawn_result["walker_speeds"])
        }

        for record in spawn_result["spawn_records"]:
            actor_id = record["actor_id"]

            if record["category"] == "pedestrian":
                actor = walker_by_id.get(actor_id)
                controller = controller_by_walker_id.get(actor_id)
                speed = speed_by_walker_id.get(actor_id, 1.4)
            else:
                actor = vehicle_like_by_id.get(actor_id)
                controller = None
                speed = None

            if actor is None:
                continue

            self.managed_actors.append({
                "actor_id": actor_id,
                "actor": actor,
                "controller": controller,
                "category": record["category"],
                "road_id": record["road_id"],
                "lane_id": record["lane_id"],
                "initial_s": record["sampled_s"],
                "spawn_route_progress": record["actual_distance"],
                "spawn_update_index": 0,
                "route_index_hint": self.ego_route_index,
                "consecutive_projection_failures": 0,
                "speed": speed,
            })

    def start_initial_pedestrians(self):

        return

    def _ego_route_s(self):
        route_s, self.ego_route_index, _offset = route_progress_at_location(
            self.ego.get_location(), self.dense_route, self.arc_length_table, self.ego_route_index,
            max_offset=self.cfg.SPAWN.MAX_ROUTE_PROJECTION_DISTANCE,
        )

        if route_s is None:
            route_s = self.arc_length_table[self.ego_route_index]

        return route_s

    def _destroy_managed_actor(self, managed):

        try:
            if managed["controller"] is not None and managed["controller"].is_alive:
                managed["controller"].stop()
        except RuntimeError:
            pass

        try:
            if managed["controller"] is not None and managed["controller"].is_alive:
                managed["controller"].destroy()
        except RuntimeError:
            pass

        try:
            if managed["actor"] is not None and managed["actor"].is_alive:
                managed["actor"].destroy()
        except RuntimeError:
            pass

   

    def update(self, local_frame_id):
        self.update_index += 1

        managed_total_before = len(self.managed_actors)

        ego_route_s = self._ego_route_s()
        ego_location = self.ego.get_location()

        live = []  # (managed, relative_s, location) for actors with a valid projection this update
        projection_failed_details = []  # diagnostic rows for actors that failed projection *this* update
        dead_actors_removed = 0  # already destroyed externally (e.g. collision), not by our despawn logic
        projection_failure_despawns_this_update = 0
        full_search_attempts_this_update = 0
        full_search_recoveries_this_update = 0

        for managed in self.managed_actors:
            actor = managed["actor"]

            if actor is None or not actor.is_alive:
                dead_actors_removed += 1
                continue

            location = actor.get_location()
            previous_hint = managed["route_index_hint"]
            failure_streak = managed.get("consecutive_projection_failures", 0)

            route_s, managed["route_index_hint"], offset, used_full_search = project_actor_with_recovery(
                location, self.dense_route, self.arc_length_table, previous_hint,
                failure_streak, self.cfg.SPAWN.MAX_ROUTE_PROJECTION_DISTANCE,
            )

            if used_full_search:
                full_search_attempts_this_update += 1

            if route_s is None:
                self.route_projection_failures += 1
                managed["consecutive_projection_failures"] = failure_streak + 1

                projection_failed_details.append({
                    "update_index": self.update_index,
                    "actor_id": managed["actor_id"],
                    "category": managed["category"],
                    "x": location.x, "y": location.y, "z": location.z,
                    "road_id": managed["road_id"], "lane_id": managed["lane_id"],
                    "distance_from_ego": location.distance(ego_location),
                    "nearest_route_point_distance": offset,
                    "previous_route_index_hint": previous_hint,
                    "consecutive_failures": managed["consecutive_projection_failures"],
                    "used_full_search": used_full_search,
                })

                if managed["consecutive_projection_failures"] >= self.cfg.SPAWN.MAX_PROJECTION_FAILURE_UPDATES:
                    self._destroy_managed_actor(managed)
                    managed["_despawned_for_projection_failure"] = True
                    projection_failure_despawns_this_update += 1

                continue

            if used_full_search and failure_streak > 0:
                full_search_recoveries_this_update += 1

            managed["consecutive_projection_failures"] = 0
            relative_s = route_s - ego_route_s
            live.append((managed, relative_s, location))

        despawn_this_update = 0
        still_managed = []
        still_live = []

        for managed, relative_s, location in live:
            if (
                relative_s < -self.cfg.SPAWN.DESPAWN_BEHIND_DISTANCE
                or relative_s > self.cfg.SPAWN.FORWARD_CLEANUP_DISTANCE
            ):
                self._destroy_managed_actor(managed)
                despawn_this_update += 1
            else:
                still_managed.append(managed)
                still_live.append((managed, relative_s, location))

        projected_ids = {id(m) for m, _s, _loc in live}
        kept_unprojected = [
            m for m in self.managed_actors
            if id(m) not in projected_ids
            and m["actor"] is not None and m["actor"].is_alive
            and not m.get("_despawned_for_projection_failure", False)
        ]

        self.managed_actors = still_managed + kept_unprojected
        self.total_despawned += despawn_this_update + projection_failure_despawns_this_update
        self.total_projection_failure_despawns += projection_failure_despawns_this_update
        self.total_projection_full_search_attempts += full_search_attempts_this_update
        self.total_projection_full_search_recoveries += full_search_recoveries_this_update

        state_counts = {state: {c: 0 for c in DYNAMIC_CATEGORIES} for state in STATE_NAMES}

        for managed, relative_s, _location in live:
            state_counts[classify_relative_s_state(relative_s, self.cfg)][managed["category"]] += 1

        for managed in kept_unprojected:
            state_counts["projection_failed"][managed["category"]] += 1

        state_totals = {state: sum(state_counts[state].values()) for state in STATE_NAMES}
        state_breakdown_sum = sum(state_totals.values())

        if state_breakdown_sum + dead_actors_removed != managed_total_before:
            print(
                f"[SpawnManager][diagnostic] inconsistency: state_breakdown_sum="
                f"{state_breakdown_sum} + dead_actors_removed={dead_actors_removed} "
                f"!= managed_total_before={managed_total_before}"
            )

        current_bin_counts = {
            category: np.zeros(self.n_bins, dtype=int) for category in DYNAMIC_CATEGORIES
        }

        for managed, relative_s, _location in still_live:
            if self.cfg.SPAWN.MIN_DISTANCE <= relative_s <= self.cfg.SPAWN.MAX_DISTANCE:
                bin_index = min(
                    int((relative_s - self.cfg.SPAWN.MIN_DISTANCE) // self.cfg.SPAWN.BIN_SIZE),
                    self.n_bins - 1,
                )
                current_bin_counts[managed["category"]][bin_index] += 1

        deficits = {
            category: np.clip(self.category_bin_targets[category] - current_bin_counts[category], 0, None)
            for category in DYNAMIC_CATEGORIES
        }

        category_total_current = {c: int(current_bin_counts[c].sum()) for c in DYNAMIC_CATEGORIES}
        category_total_target = {c: int(self.category_bin_targets[c].sum()) for c in DYNAMIC_CATEGORIES}

        category_capacity = {
            c: max(category_total_target[c] - category_total_current[c], 0)
            for c in DYNAMIC_CATEGORIES
        }

        category_cap_skips_this_update = 0

        tokens = []

        for category in DYNAMIC_CATEGORIES:
            bin_deficit_total = int(deficits[category].sum())
            budget = min(category_capacity[category], bin_deficit_total)
            category_cap_skips_this_update += max(bin_deficit_total - budget, 0)
            tokens.extend([category] * budget)

        self.rng.shuffle(tokens)

        live_vehicle_records = [
            {
                "road_id": m["road_id"],  # best-effort; refreshed below for spawned-this-update actors only
                "lane_id": m["lane_id"],
                "s": relative_s,
                "location": location,
            }
            for m, relative_s, location in still_live
            if m["category"] in VEHICLE_LIKE_CATEGORIES
        ]
        live_pedestrian_locations = [
            location for m, _relative_s, location in still_live if m["category"] == "pedestrian"
        ]

        ego_waypoint = self.carla_map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_road_id = ego_waypoint.road_id if ego_waypoint is not None else None
        ego_lane_id = ego_waypoint.lane_id if ego_waypoint is not None else None

        spawned_this_update = 0
        failed_this_update = 0
        over_target_spawn_attempts_this_update = 0
        same_lane_spawn_rejections_this_update = 0
        blocked_bin_skips_this_update = 0

        blocked_bins = set()

        for category in tokens:
            if spawned_this_update >= self.cfg.SPAWN.MAX_NEW_ACTORS_PER_UPDATE:
                break

            candidate_bin_indices = [
                i for i in range(self.n_bins)
                if deficits[category][i] > 0 and (category, i) not in blocked_bins
            ]

            if not candidate_bin_indices:
                if int(deficits[category].sum()) > 0:
                    blocked_bin_skips_this_update += 1  # would have retried a blocked bin

                continue

            bin_index = max(candidate_bin_indices, key=lambda i: deficits[category][i])

            if category_total_current[category] >= category_total_target[category]:
                over_target_spawn_attempts_this_update += 1

            bin_low = float(self.bin_edges[bin_index])
            bin_high = float(self.bin_edges[bin_index + 1])

            if category == "pedestrian":
                result = self._try_replenish_pedestrian(bin_low, bin_high, ego_route_s, live_pedestrian_locations)
            else:
                result, same_lane_rejections = self._try_replenish_vehicle_like(
                    category, bin_low, bin_high, ego_route_s, live_vehicle_records, ego_road_id, ego_lane_id,
                )
                same_lane_spawn_rejections_this_update += same_lane_rejections

            if result is None:
                failed_this_update += 1
                deficits[category][bin_index] -= 1
                blocked_bins.add((category, bin_index))
                continue

            deficits[category][bin_index] -= 1
            spawned_this_update += 1

        self.total_spawned += spawned_this_update
        self.total_failed += failed_this_update
        self.total_over_target_spawn_attempts += over_target_spawn_attempts_this_update
        self.total_same_lane_spawn_rejections += same_lane_spawn_rejections_this_update
        self.total_blocked_bin_skips += blocked_bin_skips_this_update
        self.total_category_cap_skips += category_cap_skips_this_update

        category_counts = {
            category: int(current_bin_counts[category].sum()) for category in DYNAMIC_CATEGORIES
        }
        category_targets = {
            category: int(self.category_bin_targets[category].sum()) for category in DYNAMIC_CATEGORIES
        }

        mean_relative_s = float(np.mean([s for _m, s, _l in still_live])) if still_live else float("nan")

        bin_mae = {
            category: float(np.mean(np.abs(self.category_bin_targets[category] - current_bin_counts[category])))
            for category in DYNAMIC_CATEGORIES
        }

        nearest_lead = self._nearest_same_lane_lead(ego_location, still_live)

        snapshot = {
            "update_index": self.update_index,
            "local_frame_id": local_frame_id,
            "ego_route_s": ego_route_s,
            "category_counts": category_counts,
            "category_targets": category_targets,
            "current_bin_counts": {c: current_bin_counts[c].tolist() for c in DYNAMIC_CATEGORIES},
            "target_bin_counts": {c: self.category_bin_targets[c].tolist() for c in DYNAMIC_CATEGORIES},
            "despawned": despawn_this_update + projection_failure_despawns_this_update,
            "despawned_position": despawn_this_update,
            "despawned_projection_failure": projection_failure_despawns_this_update,
            "spawned": spawned_this_update,
            "failed": failed_this_update,
            "managed_actor_count": len(self.managed_actors),
            "managed_total_before": managed_total_before,
            "dead_actors_removed": dead_actors_removed,
            "mean_relative_s": mean_relative_s,
            "bin_mae": bin_mae,
            "route_projection_failures": self.route_projection_failures,

            "state_counts": state_counts,
            "state_totals": state_totals,

            "projection_failed_details": projection_failed_details,

            "over_target_spawn_attempts_this_update": over_target_spawn_attempts_this_update,
            "total_over_target_spawn_attempts": self.total_over_target_spawn_attempts,
            "nearest_lead_distance": nearest_lead[1] if nearest_lead else None,
            "nearest_lead_actor_id": nearest_lead[0]["actor_id"] if nearest_lead else None,
            "nearest_lead_speed_kmh": nearest_lead[2] if nearest_lead else None,
            "projection_failure_despawns_this_update": projection_failure_despawns_this_update,
            "total_projection_failure_despawns": self.total_projection_failure_despawns,
            "full_search_attempts_this_update": full_search_attempts_this_update,
            "total_projection_full_search_attempts": self.total_projection_full_search_attempts,
            "full_search_recoveries_this_update": full_search_recoveries_this_update,
            "total_projection_full_search_recoveries": self.total_projection_full_search_recoveries,
            "same_lane_spawn_rejections_this_update": same_lane_spawn_rejections_this_update,
            "total_same_lane_spawn_rejections": self.total_same_lane_spawn_rejections,
            "blocked_bin_skips_this_update": blocked_bin_skips_this_update,
            "total_blocked_bin_skips": self.total_blocked_bin_skips,
            "category_cap_skips_this_update": category_cap_skips_this_update,
            "total_category_cap_skips": self.total_category_cap_skips,
        }

        self.history.append(snapshot)

        print(f"[SpawnManager] update={self.update_index} ego_s={ego_route_s:.1f}m")

        for category in DYNAMIC_CATEGORIES:
            print(f"  {category:12s} {category_counts[category]:3d}/{category_targets[category]:<3d}")

        print(
            f"  despawned={despawn_this_update}(+{projection_failure_despawns_this_update} projection) "
            f"spawned={spawned_this_update} failed={failed_this_update} "
            f"same_lane_rej={same_lane_spawn_rejections_this_update} "
            f"blocked_bin_skips={blocked_bin_skips_this_update} "
            f"category_cap_skips={category_cap_skips_this_update}"
        )

        return snapshot

    def _nearest_same_lane_lead(self, ego_location, still_live):

        ego_waypoint = self.carla_map.get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if ego_waypoint is None:
            return None

        best = None
        best_distance = None

        for managed, relative_s, location in still_live:
            if managed["category"] not in VEHICLE_LIKE_CATEGORIES or relative_s <= 0:
                continue

            waypoint = self.carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if waypoint is None or waypoint.road_id != ego_waypoint.road_id or waypoint.lane_id != ego_waypoint.lane_id:
                continue

            if best_distance is None or relative_s < best_distance:
                best_distance = relative_s
                best = managed

        if best is None:
            return None

        speed = best["actor"].get_velocity()
        speed_kmh = 3.6 * (speed.x ** 2 + speed.y ** 2 + speed.z ** 2) ** 0.5

        return best, best_distance, speed_kmh


    def _try_replenish_vehicle_like(self, category, bin_low, bin_high, ego_route_s, live_vehicle_records, ego_road_id, ego_lane_id):

        blueprint_pool = self.vehicle_pools.get(category, [])

        if not blueprint_pool:
            return None, 0

        ego_location = self.ego.get_location()
        same_lane_rejections = 0

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            relative_s = sample_bin_constrained_distance(self.rng, bin_low, bin_high, self.cfg)

            if relative_s is None:
                continue

            target_absolute_s = ego_route_s + relative_s
            reference_waypoint, _actual = reference_waypoint_at_distance(self.dense_route, target_absolute_s)

            if reference_waypoint is None:
                continue

            candidates = driving_lane_candidates(reference_waypoint)
            lane_waypoint = choose_lane(self.rng, candidates)

            if lane_waypoint is None:
                continue

            if (
                ego_road_id is not None
                and lane_waypoint.road_id == ego_road_id
                and lane_waypoint.lane_id == ego_lane_id
                and relative_s < self.cfg.SPAWN.MIN_SAME_LANE_EGO_SPAWN_DISTANCE
            ):
                same_lane_rejections += 1
                continue

            transform = carla.Transform(
                lane_waypoint.transform.location, lane_waypoint.transform.rotation,
            )
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(ego_location) < self.cfg.SPAWN.MIN_EGO_SPACING:
                continue

            if not vehicle_spacing_ok(
                transform.location, lane_waypoint.road_id, lane_waypoint.lane_id, relative_s,
                live_vehicle_records,
                self.cfg.SPAWN.MIN_VEHICLE_SPACING,
                self.cfg.SPAWN.MIN_CROSS_LANE_SPACING,
            ):
                continue

            blueprint = prepare_blueprint(self.rng.choice(blueprint_pool), self.rng)
            actor = self.world.try_spawn_actor(blueprint, transform)

            if actor is None:
                continue

            configure_actor_traffic_manager(actor, self.traffic_manager, self.cfg)

            live_vehicle_records.append({
                "road_id": lane_waypoint.road_id, "lane_id": lane_waypoint.lane_id,
                "s": relative_s, "location": transform.location,
            })

            managed = {
                "actor_id": actor.id,
                "actor": actor,
                "controller": None,
                "category": category,
                "road_id": lane_waypoint.road_id,
                "lane_id": lane_waypoint.lane_id,
                "initial_s": relative_s,
                "spawn_route_progress": target_absolute_s,
                "spawn_update_index": self.update_index,
                "route_index_hint": self.ego_route_index,
                "consecutive_projection_failures": 0,
                "speed": None,
            }
            self.managed_actors.append(managed)

            print(
                f"[Spawn] {category} id={actor.id} bin=[{bin_low:.0f},{bin_high:.0f}) "
                f"relative_s={relative_s:.1f}m road={lane_waypoint.road_id} lane={lane_waypoint.lane_id} "
                f"update={self.update_index}"
            )

            return actor, same_lane_rejections

        return None, same_lane_rejections

    def _try_replenish_pedestrian(self, bin_low, bin_high, ego_route_s, live_pedestrian_locations):
        if not self.walker_blueprints:
            return None

        ego_location = self.ego.get_location()

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            relative_s = sample_bin_constrained_distance(self.rng, bin_low, bin_high, self.cfg)

            if relative_s is None:
                continue

            target_absolute_s = ego_route_s + relative_s
            reference_waypoint, _actual = reference_waypoint_at_distance(self.dense_route, target_absolute_s)

            if reference_waypoint is None:
                continue

            sidewalk_waypoint = sidewalk_waypoint_near(self.carla_map, reference_waypoint.transform.location)

            if sidewalk_waypoint is None:
                continue

            transform = carla.Transform(
                sidewalk_waypoint.transform.location, sidewalk_waypoint.transform.rotation,
            )
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(ego_location) < self.cfg.SPAWN.MIN_EGO_SPACING:
                continue

            if not far_enough(transform.location, live_pedestrian_locations, self.cfg.SPAWN.MIN_PEDESTRIAN_SPACING):
                continue

            blueprint = self.rng.choice(self.walker_blueprints)

            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")

            walker = self.world.try_spawn_actor(blueprint, transform)

            if walker is None:
                continue

            controller = self.world.try_spawn_actor(
                self.walker_controller_bp, carla.Transform(), attach_to=walker,
            )

            if controller is None:
                walker.destroy()
                continue

            speed = 1.4

            if blueprint.has_attribute("speed"):
                values = blueprint.get_attribute("speed").recommended_values

                if len(values) > 1:
                    speed = float(values[1])

            live_pedestrian_locations.append(transform.location)

            managed = {
                "actor_id": walker.id,
                "actor": walker,
                "controller": controller,
                "category": "pedestrian",
                "road_id": sidewalk_waypoint.road_id,
                "lane_id": sidewalk_waypoint.lane_id,
                "initial_s": relative_s,
                "spawn_route_progress": target_absolute_s,
                "spawn_update_index": self.update_index,
                "route_index_hint": self.ego_route_index,
                "consecutive_projection_failures": 0,
                "speed": speed,
            }
            self.managed_actors.append(managed)

            print(
                f"[Spawn] pedestrian id={walker.id} bin=[{bin_low:.0f},{bin_high:.0f}) "
                f"relative_s={relative_s:.1f}m road={sidewalk_waypoint.road_id} lane={sidewalk_waypoint.lane_id} "
                f"update={self.update_index}"
            )

            return walker

        return None

    def destroy_all(self):
        for managed in self.managed_actors:
            self._destroy_managed_actor(managed)

        self.managed_actors = []

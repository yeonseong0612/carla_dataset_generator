"""
canonical_traffic.py

Canonical background-traffic policy.

Replaces Phase 2/2.5's per-bin Gamma-deficit replenishment
(DynamicSpawnManager in spawn_policy.py, kept unmodified/unused for
reference -- not deleted, not re-verified) with a much simpler
maintain-population-in-a-buffer-zone policy:

    despawn behind  : relative_s < -cfg.SPAWN.DESPAWN_BEHIND_DISTANCE
    visible/training: 0 <= relative_s <= cfg.SENSOR.LIDAR.ROI_FRONT_MAX
    spawn buffer    : ROI_FRONT_MAX < relative_s <= CANONICAL_BUFFER_MAX
    forward cleanup : relative_s > CANONICAL_BUFFER_MAX + DESPAWN_BEHIND_DISTANCE
                      (safety net; background speed only deterministically
                      varies by +/-CANONICAL_SPEED_JITTER_PCT%, so an actor
                      outrunning ego this far past the buffer is rare)

New actors are never spawned inside the visible ROI -- only in the buffer
zone ahead of it -- so they always enter observation by natural forward
motion (spawn -> drift into ROI -> observed across several frames -> drift
out behind -> despawn), never by materializing directly in front of ego.

Gamma(shape, scale) (cfg.SPAWN.GAMMA_SHAPE/SCALE) is reused exactly as-is
(unmodified) only for GammaSpawnPolicy's one-shot initial population
(Phase 1) -- it is never re-applied per update here, and it is NOT the
production density target (see below).

---------------------------------------------------------------------
Frame-Level Gamma Object Count Policy (production density target)
---------------------------------------------------------------------

Gamma is applied to N_objects(frame) -- how many dynamic objects should
be observable in a given frame -- NOT to object distance. See
FrameObjectGammaSchedule below and cfg.SPAWN.FRAME_OBJECT_* in
CFG/config.py.

A single target is drawn once per "density segment" (a random
cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MIN..MAX-frame span), never
resampled every frame -- resampling every frame makes the target
whiplash frame-to-frame in a way the spawn/despawn loop can never track,
producing unnatural traffic churn instead of a stable-but-varying
density. Deterministic given the same map/route/seed: the schedule
draws from the same seeded rng stream (cfg.SPAWN.SEED, continued from
Phase 1's GammaSpawnPolicy) everything else here already uses.

Every update(), CanonicalBackgroundTraffic compares the current
frame-level object count (see get_annotation_candidate_count /
TEMPORARY_COUNT_BASIS below) against the current segment's target:

    deficit = target - current_count   (current_count < target)
    excess  = current_count - target   (current_count > target)

--------------------------------------------------------------------
Controller-fix task (symmetric up/down control)
--------------------------------------------------------------------

Long-run validation (600/787/769-frame runs, see outputs/
frame_object_gamma_longrun_3000/) found the original one-sided design --
deficit drives spawns, excess just stops spawning and waits for natural
exit -- was too slow downward: actor lifetime (~15-16s) was 4-5x longer
than the then 2-5s segment duration, so managed population grew
unboundedly across segments (11->33) and eventually congested ego's lane
enough to hard-stall it (~frame 780).

Two changes fix this without ever touching a visible actor:

1. FRAME_OBJECT_TARGET_INTERVAL_MIN/MAX raised to 240-400 frames
   (12-20s at 20 FPS) -- brings segment duration close to measured actor
   lifetime instead of 4-5x shorter, so a segment has enough time for a
   spawn/despawn (or spawn/prune) cycle to actually play out before the
   target changes again.

2. excess now actively PRUNES not-yet-visible actors (buffer-zone /
   future entrants: relative_s > visible_max), instead of only waiting:

       deficit > 0: spawn in buffer, gradually
                    (cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE/update)
       excess  > 0: prune future entrants, gradually
                    (cfg.SPAWN.FRAME_OBJECT_MAX_PRUNE_PER_UPDATE/update),
                    farthest-first, never a visible-ROI actor

A visible-ROI actor (0 <= relative_s <= visible_max) is NEVER eligible
for pruning, by construction (the eligible set is filtered to
relative_s > visible_max before any pruning decision, plus an assert on
every pruned actor) -- it may only leave through the existing natural
despawn-behind/forward-cleanup lifecycle. This is still a *soft*
density target, not a hard per-frame constraint -- see CLAUDE.md task
sections 6-9 for the full rationale (actor lifetime / temporal
continuity for Stereo/Flow/VO/tracking depends on never yanking a
visible actor out mid-track; only actors that were never observed yet
are fair game for density control).

Category composition (vehicle/motorcyclist/cyclist/pedestrian) still
follows cfg.SPAWN.N_* -- now used as relative composition WEIGHTS
(largest-remainder allocation of the frame-object deficit across
categories), not per-category absolute totals. No per-category Gamma is
introduced in this task.

Background vehicles (both the Phase 1 initial spawn, re-configured here,
and buffer replenishment): auto_lane_change forced OFF, a small
deterministic per-actor speed_difference jitter, stable lane following --
no cut-in/lane-change interactions in this policy (a future "special
scenario" layer, not this one). The Phase 2.5 same-lane-near-ego 30m
guard (cfg.SPAWN.MIN_SAME_LANE_EGO_SPAWN_DISTANCE) is reused unchanged
for buffer-zone vehicle spawns.

Managed pedestrians keep the Phase 2 navmesh-bug workaround: spawned and
registered, but their AI controller is never started, so they stay
exactly where placed (see start_initial_pedestrians below) -- this still
lets them "enter" the visible ROI purely through ego's own forward
motion, so no special-casing is needed for the zone logic.
"""

import carla
import numpy as np

from src.simulation.pedestrian import get_walker_blueprints
from src.simulation.traffic import (
    configure_actor_traffic_manager,
    get_traffic_blueprints,
    prepare_blueprint,
    is_bus_blueprint,
)
from src.simulation.spawn_policy import (
    DYNAMIC_CATEGORIES,
    VEHICLE_LIKE_CATEGORIES,
    CATEGORY_TARGET_CFG_KEYS,
    SPAWN_HEIGHT_OFFSET_M,
    build_route_arc_length_table,
    route_progress_at_location,
    reference_waypoint_at_distance,
    driving_lane_candidates,
    choose_lane,
    vehicle_spacing_ok,
    far_enough,
    sidewalk_waypoint_near,
    largest_remainder_allocation,
)


# ================================================================
# Frame-Level Gamma Object Count Policy
# ================================================================

# This task's Gamma target is defined on "camera-visible valid
# annotation count". As of the "Finalize Camera-Valid Annotation
# Filtering" task, AnnotationWriter.write_frame()'s own per-frame
# category counts (what get_annotation_candidate_count() sums) ARE that
# camera-valid count -- see src/data/annotation.py
# AnnotationWriter._compute_camera_validity (FOV/truncation +
# depth-based occlusion + minimum pixel-size, left RGB camera only).
# This constant/function pair is left in place, unmodified in logic
# (count SOURCE changed upstream in annotation.py; nothing here did),
# so a caller that only reads TEMPORARY_COUNT_BASIS/calls
# get_annotation_candidate_count() sees the new basis automatically.
TEMPORARY_COUNT_BASIS = (
    "annotation_candidate_count: AnnotationWriter per-frame category "
    "counts, now the actual camera_valid annotation count (ego-frame "
    "distance <= cfg.ANNOTATION.MAX_DISTANCE, AND left-RGB-camera FOV/"
    "truncation + depth-occlusion + minimum-pixel-size filtered) -- see "
    "src/data/annotation.py AnnotationWriter._compute_camera_validity"
)


def get_annotation_candidate_count(annotation_counts):
    """
    current_visible_count = get_annotation_candidate_count(...)

    annotation_counts: the per-category dict AnnotationWriter.write_frame()
    already returns for this frame, e.g. {"pedestrian": 2, "vehicle": 5,
    "cyclist": 0, "motorcyclist": 1}. See TEMPORARY_COUNT_BASIS above for
    exactly what this counts today; a later task swaps the *contents* of
    this function for a camera-valid annotation count without touching
    any caller.
    """

    return sum(annotation_counts.values())


class FrameObjectGammaSchedule:
    """
    Frame-level object-count Gamma density target. See the "Frame-Level
    Gamma Object Count Policy" section of this module's docstring for
    the full rationale (segment-held target, deterministic seeded
    sampling, soft/gradual control).

    target_for_frame(local_frame_id) is the single query surface: call
    it for any (non-decreasing) local_frame_id and it draws a new
    segment target exactly when local_frame_id crosses into a new
    segment, never per-frame.
    """

    def __init__(self, rng, cfg):
        self.rng = rng
        self.cfg = cfg

        self.segment_index = -1
        self.next_segment_start_frame = 0
        self.current_target = None
        self.current_segment_length = None

        self._draw_new_segment()

    def _draw_new_segment(self):
        shape = self.cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE
        scale = self.cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE

        sampled = float(self.rng.gamma(shape, scale))
        target = int(round(sampled))
        target = int(np.clip(target, self.cfg.SPAWN.FRAME_OBJECT_MIN, self.cfg.SPAWN.FRAME_OBJECT_MAX))

        length = int(self.rng.integers(
            self.cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MIN,
            self.cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MAX + 1,
        ))

        self.segment_index += 1
        start_frame = self.next_segment_start_frame
        self.current_target = target
        self.current_segment_length = length
        self.next_segment_start_frame = start_frame + length

        print(
            f"[FrameObjectGamma] segment={self.segment_index} "
            f"frames=[{start_frame},{self.next_segment_start_frame}) "
            f"target={target} (raw_gamma_sample={sampled:.2f})"
        )

    def target_for_frame(self, local_frame_id):
        while local_frame_id >= self.next_segment_start_frame:
            self._draw_new_segment()

        return self.current_target


def allocate_by_category_weight(total, category_weights, categories=DYNAMIC_CATEGORIES):
    """
    Split a nonnegative int `total` across `categories`,
    proportional to `category_weights` (cfg.SPAWN.N_* used as relative
    composition weights, not absolute totals -- see module docstring),
    via the same largest-remainder method spawn_policy.py's distance-bin
    targets use (deterministic, no random rounding). A category with
    weight 0 (e.g. zeroed out for --no-traffic/--no-pedestrians) always
    gets 0.
    """

    nonzero = [category for category in categories if category_weights[category] > 0]

    if not nonzero or total <= 0:
        return {category: 0 for category in categories}

    weight_sum = sum(category_weights[category] for category in nonzero)
    probabilities = [category_weights[category] / weight_sum for category in nonzero]

    allocation = largest_remainder_allocation(total, probabilities)

    result = {category: 0 for category in categories}

    for category, count in zip(nonzero, allocation):
        result[category] = int(count)

    return result


def sample_initial_frame_object_target(cfg):
    """
    A deterministic *preview* draw of what FrameObjectGammaSchedule's
    first segment target will look like, used only to right-size Phase
    1's initial actor population (see scale_initial_category_counts) --
    CLAUDE.md task section 12. Uses its own fresh cfg.SPAWN.SEED-seeded
    generator (rather than the live rng stream) because Phase 1's spawn
    draws haven't happened yet at the point this needs to run; it is
    deliberately NOT required to equal FrameObjectGammaSchedule's actual
    first live target (that draw happens later, downstream of Phase 1's
    own rng consumption) -- this is only a sizing hint, not the target
    itself.
    """

    rng = np.random.default_rng(cfg.SPAWN.SEED)
    sampled = float(rng.gamma(cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE, cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE))
    target = int(round(sampled))

    return int(np.clip(target, cfg.SPAWN.FRAME_OBJECT_MIN, cfg.SPAWN.FRAME_OBJECT_MAX))


def scale_initial_category_counts(cfg, no_traffic=False, no_pedestrians=False):
    """
    Phase 1 initial actor counts (n_vehicles/n_motorcycles/n_bicycles/
    n_pedestrians, for spawn_actors_gamma_policy), rescaled from
    cfg.SPAWN.N_* so their TOTAL is approximately the first frame-object
    Gamma segment's target instead of the old fixed ~cfg.SPAWN.N_*-sum
    population -- CLAUDE.md task section 12. The N_VEHICLES:N_MOTORCYCLES
    :N_BICYCLES:N_PEDESTRIANS composition ratio is preserved; only the
    total is rescaled. This never forces an exact count in the visible
    ROI -- Phase 1's own safe-route spawn + spacing/validity checks can
    still place fewer than requested, same as before.
    """

    base_weights = {
        "vehicle": cfg.SPAWN.N_VEHICLES,
        "motorcyclist": cfg.SPAWN.N_MOTORCYCLES,
        "cyclist": cfg.SPAWN.N_BICYCLES,
        "pedestrian": cfg.SPAWN.N_PEDESTRIANS,
    }

    if no_traffic:
        base_weights["vehicle"] = 0
        base_weights["motorcyclist"] = 0
        base_weights["cyclist"] = 0

    if no_pedestrians:
        base_weights["pedestrian"] = 0

    initial_target = sample_initial_frame_object_target(cfg)
    scaled = allocate_by_category_weight(initial_target, base_weights)

    return {
        "n_vehicles": scaled["vehicle"],
        "n_motorcycles": scaled["motorcyclist"],
        "n_bicycles": scaled["cyclist"],
        "n_pedestrians": scaled["pedestrian"],
        "initial_frame_object_target": initial_target,
    }


# ================================================================
# Bus exclusion (canonical background traffic only)
# ================================================================
#
# Task "Remove Bus Completely from Canonical Dataset Traffic": bus is
# excluded from spawn eligibility entirely (not just rare) -- see
# is_bus_blueprint in src/simulation/traffic.py, applied once to this
# class's own self.vehicle_pools["vehicle"] in __init__ (mirrors the
# identical filter in GammaSpawnPolicy.__init__, src/simulation/
# spawn_policy.py, for Phase 1). Bus stays classified as "vehicle" by
# get_vehicle_category()/get_category() for reading/annotation purposes
# -- only spawn selection is affected. A prior frequency-management
# version of this policy (weighted-but-nonzero selection + population
# caps + same-lane spacing) was tried and replaced by this task after
# live validation showed an unweighted Phase-1 bus could still dominate
# the scene -- see outputs/bus_policy_live_validation/ for that record.


class CanonicalBackgroundTraffic:
    """
    Drop-in replacement for DynamicSpawnManager: same
    register_initial_actors(spawn_result) / start_initial_pedestrians() /
    update(local_frame_id) / destroy_all() / managed_actors surface, so
    scripts/collect_dataset.py can select either policy behind a flag
    without otherwise changing its integration code.
    """

    def __init__(self, world, ego, dense_route, traffic_manager, cfg, rng, category_totals=None):
        """
        category_totals: optional {category: total} override (e.g. to
        zero out a category for --no-traffic/--no-pedestrians); any
        category left out defaults to cfg.SPAWN.N_* -- same convention as
        DynamicSpawnManager.
        """

        self.world = world
        self.ego = ego
        self.dense_route = dense_route
        self.traffic_manager = traffic_manager
        self.cfg = cfg
        self.rng = rng

        self.carla_map = world.get_map()
        self.vehicle_pools = get_traffic_blueprints(world)
        # Bus excluded from spawn eligibility -- see module docstring
        # "Bus exclusion" section. Mirrors GammaSpawnPolicy.__init__'s
        # identical filter (src/simulation/spawn_policy.py).
        self.vehicle_pools["vehicle"] = [
            blueprint for blueprint in self.vehicle_pools["vehicle"] if not is_bus_blueprint(blueprint)
        ]
        self.walker_blueprints = get_walker_blueprints(world)
        self.walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

        self.arc_length_table = build_route_arc_length_table(dense_route)

        self.visible_max = cfg.SENSOR.LIDAR.ROI_FRONT_MAX
        self.buffer_max = cfg.SPAWN.CANONICAL_BUFFER_MAX
        self.forward_cleanup = self.buffer_max + cfg.SPAWN.DESPAWN_BEHIND_DISTANCE

        category_totals = category_totals or {}

        # Now used as relative COMPOSITION WEIGHTS for the frame-object
        # deficit split (allocate_by_category_weight), not
        # per-category absolute totals -- see module docstring.
        self.category_targets = {
            category: category_totals.get(category, getattr(cfg.SPAWN, CATEGORY_TARGET_CFG_KEYS[category]))
            for category in DYNAMIC_CATEGORIES
        }

        # Frame-Level Gamma Object Count Policy -- continues this same
        # rng stream (seeded from cfg.SPAWN.SEED via Phase 1's
        # GammaSpawnPolicy), so it's deterministic per map/route/seed.
        self.frame_object_schedule = FrameObjectGammaSchedule(rng, cfg)

        self.ego_route_index = 0
        self.managed_actors = []

        self.update_index = 0
        self.total_spawned = 0
        self.total_despawned = 0
        self.total_despawned_behind = 0
        self.total_despawned_forward_cleanup = 0
        self.total_projection_failure_despawns = 0
        self.total_failed = 0
        self.total_same_lane_spawn_rejections = 0

        # Structural guarantee (buffer-only spawn, see _try_spawn_buffer_*
        # below): every successful spawn's relative_s is drawn from
        # [visible_max, buffer_max), so spawned_inside_visible_roi stays
        # 0 by construction, not by measurement -- kept as an explicit
        # counter (and assertion) so validation can report it directly.
        self.total_spawned_inside_visible_roi = 0
        self.total_spawned_inside_buffer = 0

        # Controller-fix task: symmetric downward control (see update()'s
        # pruning block). total_pruned_inside_visible_roi stays 0 by
        # construction (assert), same convention as
        # total_spawned_inside_visible_roi above.
        self.total_density_pruned = 0
        self.total_pruned_inside_visible_roi = 0

        self.history = []

    # ------------------------------------------------------------
    # Seeding the registry from the Phase 1 initial (Gamma) spawn
    # ------------------------------------------------------------

    def register_initial_actors(self, spawn_result):
        """
        Adopts the actors GammaSpawnPolicy.run() already placed, exactly
        like DynamicSpawnManager.register_initial_actors -- then
        re-applies this policy's background-vehicle behavior (auto lane
        change OFF, deterministic speed jitter) on top of whatever
        configure_actor_traffic_manager() already set, since Phase 1
        itself is reused unmodified and doesn't know about this policy.
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

        for record in spawn_result["spawn_records"]:
            actor_id = record["actor_id"]

            if record["category"] == "pedestrian":
                actor = walker_by_id.get(actor_id)
                controller = controller_by_walker_id.get(actor_id)
            else:
                actor = vehicle_like_by_id.get(actor_id)
                controller = None

            if actor is None:
                continue

            if record["category"] != "pedestrian":
                self._configure_canonical_vehicle(actor)

            self.managed_actors.append({
                "actor_id": actor_id,
                "actor": actor,
                "controller": controller,
                "category": record["category"],
                "road_id": record["road_id"],
                "lane_id": record["lane_id"],
                "route_index_hint": self.ego_route_index,
                "consecutive_projection_failures": 0,
            })

    def start_initial_pedestrians(self):
        """
        Deliberately does NOT call controller.start() -- see the
        identical, extensively-verified note on
        DynamicSpawnManager.start_initial_pedestrians() in
        spawn_policy.py. Do not re-add this without re-verifying against
        the CARLA build in use.
        """

        return

    # ------------------------------------------------------------
    # Route progress / actor behavior helpers
    # ------------------------------------------------------------

    def _ego_route_s(self):
        route_s, self.ego_route_index, _offset = route_progress_at_location(
            self.ego.get_location(), self.dense_route, self.arc_length_table, self.ego_route_index,
            max_offset=self.cfg.SPAWN.MAX_ROUTE_PROJECTION_DISTANCE,
        )

        if route_s is None:
            route_s = self.arc_length_table[self.ego_route_index]

        return route_s

    def _configure_canonical_vehicle(self, actor):
        configure_actor_traffic_manager(actor, self.traffic_manager, self.cfg)
        self.traffic_manager.auto_lane_change(actor, False)
        jitter = self.cfg.SPAWN.CANONICAL_SPEED_JITTER_PCT
        speed_difference = float(self.rng.uniform(-jitter, jitter))
        self.traffic_manager.vehicle_percentage_speed_difference(actor, speed_difference)

    def _destroy_managed_actor(self, managed):
        """Mirrors DynamicSpawnManager._destroy_managed_actor's cleanup convention."""

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

    # ------------------------------------------------------------
    # Per-update maintenance
    # ------------------------------------------------------------

    def update(self, local_frame_id, current_object_count=None):
        """
        current_object_count: this frame's "current_visible_count" per
        get_annotation_candidate_count() / TEMPORARY_COUNT_BASIS (module
        docstring) -- the caller (scripts/collect_dataset.py) passes the
        sum of AnnotationWriter.write_frame()'s per-frame counts. If
        omitted (e.g. older/other callers), falls back to counting this
        manager's own managed actors currently inside the visible ROI --
        a strictly self-contained but coarser proxy (no camera-distance
        cutoff, no other-world-actor awareness).
        """

        self.update_index += 1

        managed_total_before = len(self.managed_actors)
        ego_route_s = self._ego_route_s()

        ego_waypoint = self.carla_map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        ego_road_id = ego_waypoint.road_id if ego_waypoint is not None else None
        ego_lane_id = ego_waypoint.lane_id if ego_waypoint is not None else None

        live = []  # (managed, relative_s)
        despawned_behind = 0
        despawned_forward_cleanup = 0
        projection_failure_despawns = 0
        dead_actors_removed = 0

        for managed in list(self.managed_actors):
            actor = managed["actor"]

            if actor is None or not actor.is_alive:
                self.managed_actors.remove(managed)
                dead_actors_removed += 1
                continue

            route_s, managed["route_index_hint"], _offset = route_progress_at_location(
                actor.get_location(), self.dense_route, self.arc_length_table, managed["route_index_hint"],
                max_offset=self.cfg.SPAWN.MAX_ROUTE_PROJECTION_DISTANCE,
            )

            if route_s is None:
                managed["consecutive_projection_failures"] += 1

                if managed["consecutive_projection_failures"] >= self.cfg.SPAWN.MAX_PROJECTION_FAILURE_UPDATES:
                    self._destroy_managed_actor(managed)
                    self.managed_actors.remove(managed)
                    projection_failure_despawns += 1

                continue

            managed["consecutive_projection_failures"] = 0
            relative_s = route_s - ego_route_s

            if relative_s < -self.cfg.SPAWN.DESPAWN_BEHIND_DISTANCE:
                self._destroy_managed_actor(managed)
                self.managed_actors.remove(managed)
                despawned_behind += 1
                continue

            if relative_s > self.forward_cleanup:
                self._destroy_managed_actor(managed)
                self.managed_actors.remove(managed)
                despawned_forward_cleanup += 1
                continue

            live.append((managed, relative_s))

        despawned_this_update = despawned_behind + despawned_forward_cleanup + projection_failure_despawns
        self.total_despawned += despawned_this_update
        self.total_despawned_behind += despawned_behind
        self.total_despawned_forward_cleanup += despawned_forward_cleanup
        self.total_projection_failure_despawns += projection_failure_despawns

        # ---- All-zone category counts (diagnostic only, unchanged) ----
        category_counts = {category: 0 for category in DYNAMIC_CATEGORIES}

        for managed, _relative_s in live:
            category_counts[managed["category"]] += 1

        # ---- Frame-Level Gamma Object Count Policy control ----
        # (see module docstring: target is TOTAL objects/frame, held for
        # a temporal segment, compared only against the VISIBLE-ROI
        # population -- never the whole buffer-inclusive managed
        # registry, which is what category_counts above still is.)
        visible_category_counts = {category: 0 for category in DYNAMIC_CATEGORIES}

        for managed, relative_s in live:
            if 0.0 <= relative_s <= self.visible_max:
                visible_category_counts[managed["category"]] += 1

        total_visible_count = sum(visible_category_counts.values())

        frame_target = self.frame_object_schedule.target_for_frame(local_frame_id)

        if current_object_count is None:
            current_object_count = total_visible_count
            count_basis = "managed_registry_visible_roi (no annotation count passed)"
        else:
            count_basis = TEMPORARY_COUNT_BASIS

        total_deficit = max(frame_target - current_object_count, 0)
        excess = max(current_object_count - frame_target, 0)

        # Deterministic weighted category split of the TARGET (not the
        # deficit) so under-represented categories in the visible ROI
        # get priority, but never spawn anything once total_deficit==0
        # (over-target -> new spawn stop only, see module docstring).
        category_frame_targets = allocate_by_category_weight(frame_target, self.category_targets)

        tokens = []

        if total_deficit > 0:
            for category in DYNAMIC_CATEGORIES:
                category_deficit = max(category_frame_targets[category] - visible_category_counts[category], 0)
                tokens.extend([category] * category_deficit)

            self.rng.shuffle(tokens)

        max_new_this_update = min(self.cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE, total_deficit)

        # ---- Controller-fix: over-target -> prune future entrants ----
        # Symmetric counterpart to the spawn branch above. Mutually
        # exclusive with it (excess > 0 only when total_deficit == 0):
        # this never removes a visible-ROI actor (relative_s <=
        # visible_max is excluded from `eligible` before any decision,
        # plus an assert below) -- only actors that haven't been
        # observed yet. Farthest-first within an over-represented-
        # category-first ordering (section 13: prefer pruning categories
        # currently above their composition-weight share of the whole
        # live registry; deterministic tie-break, no RNG).
        pruned_this_update = 0
        pruned_category_counts = {category: 0 for category in DYNAMIC_CATEGORIES}

        if excess > 0:
            weight_sum = sum(self.category_targets.values()) or 1
            total_live_for_weight = sum(category_counts.values()) or 1
            overrepresented = {
                category: (category_counts[category] / total_live_for_weight) > (self.category_targets[category] / weight_sum)
                for category in DYNAMIC_CATEGORIES
            }

            eligible = [(managed, relative_s) for managed, relative_s in live if relative_s > self.visible_max]
            eligible.sort(key=lambda item: (not overrepresented[item[0]["category"]], -item[1]))

            max_prune_this_update = min(self.cfg.SPAWN.FRAME_OBJECT_MAX_PRUNE_PER_UPDATE, excess, len(eligible))

            for managed, relative_s in eligible[:max_prune_this_update]:
                assert relative_s > self.visible_max, "refusing to prune a visible-ROI actor"

                self._destroy_managed_actor(managed)
                self.managed_actors.remove(managed)
                live.remove((managed, relative_s))
                pruned_category_counts[managed["category"]] += 1
                pruned_this_update += 1

                print(
                    f"[Canonical-Prune] {managed['category']} id={managed['actor_id']} "
                    f"buffer_relative_s={relative_s:.1f}m update={self.update_index}"
                )

        self.total_density_pruned += pruned_this_update
        buffer_population = sum(1 for _m, relative_s in live if relative_s > self.visible_max)

        live_vehicle_records = [
            {
                "road_id": managed["road_id"], "lane_id": managed["lane_id"],
                "s": ego_route_s + relative_s, "location": managed["actor"].get_location(),
            }
            for managed, relative_s in live if managed["category"] in VEHICLE_LIKE_CATEGORIES
        ]
        live_pedestrian_locations = [
            managed["actor"].get_location() for managed, _r in live if managed["category"] == "pedestrian"
        ]

        spawned_this_update = 0
        failed_this_update = 0
        same_lane_rejections_this_update = 0

        for category in tokens:
            if spawned_this_update >= max_new_this_update:
                break

            if category == "pedestrian":
                actor = self._try_spawn_buffer_pedestrian(ego_route_s, live_pedestrian_locations)
                rejections = 0
            else:
                actor, rejections = self._try_spawn_buffer_vehicle_like(
                    category, ego_route_s, live_vehicle_records, ego_road_id, ego_lane_id,
                )

            same_lane_rejections_this_update += rejections

            if actor is None:
                failed_this_update += 1
                continue

            spawned_this_update += 1

        self.total_spawned += spawned_this_update
        self.total_failed += failed_this_update
        self.total_same_lane_spawn_rejections += same_lane_rejections_this_update

        snapshot = {
            "update_index": self.update_index,
            "local_frame_id": local_frame_id,
            "ego_route_s": ego_route_s,
            "managed_total_before": managed_total_before,
            "managed_actor_count": len(self.managed_actors),
            "dead_actors_removed": dead_actors_removed,
            "despawned": despawned_this_update,
            "despawned_behind": despawned_behind,
            "despawned_forward_cleanup": despawned_forward_cleanup,
            "despawned_projection_failure": projection_failure_despawns,
            "spawned": spawned_this_update,
            "failed": failed_this_update,
            "same_lane_spawn_rejections_this_update": same_lane_rejections_this_update,
            "total_same_lane_spawn_rejections": self.total_same_lane_spawn_rejections,
            "total_spawned": self.total_spawned,
            "total_despawned": self.total_despawned,
            "total_projection_failure_despawns": self.total_projection_failure_despawns,
            "category_counts": dict(category_counts),
            "category_targets": dict(self.category_targets),
            # Frame-Level Gamma Object Count Policy fields.
            "frame_object_target": frame_target,
            "frame_object_current_count": current_object_count,
            "frame_object_count_basis": count_basis,
            "frame_object_total_deficit": total_deficit,
            "frame_object_excess": excess,
            "frame_object_max_new_this_update": max_new_this_update,
            "visible_category_counts": dict(visible_category_counts),
            "category_frame_targets": dict(category_frame_targets),
            # Controller-fix task: population zones + prune stats.
            "managed_population": len(self.managed_actors),
            "visible_population": total_visible_count,
            "buffer_population": buffer_population,
            "pruned_this_update": pruned_this_update,
            "pruned_category_counts": dict(pruned_category_counts),
            "pruned_inside_visible_roi": 0,  # structural guarantee -- see assert above
            "total_density_pruned": self.total_density_pruned,
        }
        self.history.append(snapshot)

        print(
            f"[Canonical] update={self.update_index} ego_s={ego_route_s:.1f}m "
            f"frame_target={frame_target} current={current_object_count} "
            f"deficit={total_deficit} excess={excess} ({count_basis})"
        )
        print(
            "  " + "  ".join(f"{c}={visible_category_counts[c]}/{category_frame_targets[c]}" for c in DYNAMIC_CATEGORIES)
        )
        print(
            f"  despawned={despawned_this_update}"
            f"(behind={despawned_behind} fwd_cleanup={despawned_forward_cleanup} proj={projection_failure_despawns}) "
            f"spawned={spawned_this_update} failed={failed_this_update} "
            f"pruned={pruned_this_update} "
            f"same_lane_rej={same_lane_rejections_this_update} "
            f"managed_pop={len(self.managed_actors)}(visible={total_visible_count} buffer={buffer_population})"
        )

        return snapshot

    # ------------------------------------------------------------
    # Buffer-zone spawn (the only place new actors ever appear)
    # ------------------------------------------------------------

    def _try_spawn_buffer_vehicle_like(self, category, ego_route_s, live_vehicle_records, ego_road_id, ego_lane_id):
        blueprint_pool = self.vehicle_pools.get(category, [])

        if not blueprint_pool:
            return None, 0

        # Bus is excluded from blueprint_pool at __init__ time (see
        # "Bus exclusion" module docstring section) -- a plain uniform
        # choice here has zero bus probability by construction, no
        # per-call bus-specific logic needed.
        same_lane_rejections = 0

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            relative_s = float(self.rng.uniform(self.visible_max, self.buffer_max))
            target_absolute_s = ego_route_s + relative_s

            reference_waypoint, _actual = reference_waypoint_at_distance(self.dense_route, target_absolute_s)

            if reference_waypoint is None:
                continue

            candidates = driving_lane_candidates(reference_waypoint)
            lane_waypoint = choose_lane(self.rng, candidates)

            if lane_waypoint is None:
                continue

            # PART C-equivalent guard, reused unchanged: reject a
            # same-road_id+lane_id-as-ego spawn closer than the safety
            # distance (buffer spawns are always >visible_max away, so
            # this only ever fires if the route curves back on itself).
            if (
                ego_road_id is not None
                and lane_waypoint.road_id == ego_road_id
                and lane_waypoint.lane_id == ego_lane_id
                and relative_s < self.cfg.SPAWN.MIN_SAME_LANE_EGO_SPAWN_DISTANCE
            ):
                same_lane_rejections += 1
                continue

            transform = carla.Transform(lane_waypoint.transform.location, lane_waypoint.transform.rotation)
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(self.ego.get_location()) < self.cfg.SPAWN.MIN_EGO_SPACING:
                continue

            if not vehicle_spacing_ok(
                transform.location, lane_waypoint.road_id, lane_waypoint.lane_id, target_absolute_s,
                live_vehicle_records,
                self.cfg.SPAWN.MIN_VEHICLE_SPACING,
                self.cfg.SPAWN.MIN_CROSS_LANE_SPACING,
            ):
                continue

            blueprint = prepare_blueprint(self.rng.choice(blueprint_pool), self.rng)
            actor = self.world.try_spawn_actor(blueprint, transform)

            if actor is None:
                continue

            self._configure_canonical_vehicle(actor)

            # Structural guarantee: relative_s was drawn from
            # [visible_max, buffer_max) above, so this spawn is always in
            # the buffer, never the visible ROI -- see
            # total_spawned_inside_visible_roi in __init__.
            assert relative_s >= self.visible_max
            self.total_spawned_inside_buffer += 1

            live_vehicle_records.append({
                "road_id": lane_waypoint.road_id, "lane_id": lane_waypoint.lane_id,
                "s": target_absolute_s, "location": transform.location,
            })

            managed = {
                "actor_id": actor.id, "actor": actor, "controller": None, "category": category,
                "road_id": lane_waypoint.road_id, "lane_id": lane_waypoint.lane_id,
                "route_index_hint": self.ego_route_index, "consecutive_projection_failures": 0,
            }
            self.managed_actors.append(managed)

            print(
                f"[Canonical-Spawn] {category} id={actor.id} buffer_relative_s={relative_s:.1f}m "
                f"road={lane_waypoint.road_id} lane={lane_waypoint.lane_id} update={self.update_index}"
            )

            return actor, same_lane_rejections

        return None, same_lane_rejections

    def _try_spawn_buffer_pedestrian(self, ego_route_s, live_pedestrian_locations):
        if not self.walker_blueprints:
            return None

        for _attempt in range(self.cfg.SPAWN.MAX_ATTEMPTS):
            relative_s = float(self.rng.uniform(self.visible_max, self.buffer_max))
            target_absolute_s = ego_route_s + relative_s

            reference_waypoint, _actual = reference_waypoint_at_distance(self.dense_route, target_absolute_s)

            if reference_waypoint is None:
                continue

            sidewalk_waypoint = sidewalk_waypoint_near(self.carla_map, reference_waypoint.transform.location)

            if sidewalk_waypoint is None:
                continue

            transform = carla.Transform(sidewalk_waypoint.transform.location, sidewalk_waypoint.transform.rotation)
            transform.location.z += SPAWN_HEIGHT_OFFSET_M

            if transform.location.distance(self.ego.get_location()) < self.cfg.SPAWN.MIN_EGO_SPACING:
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

            # Never call controller.start() -- see start_initial_pedestrians().

            # Structural guarantee -- see the matching assert in
            # _try_spawn_buffer_vehicle_like.
            assert relative_s >= self.visible_max
            self.total_spawned_inside_buffer += 1

            live_pedestrian_locations.append(transform.location)

            managed = {
                "actor_id": walker.id, "actor": walker, "controller": controller, "category": "pedestrian",
                "road_id": None, "lane_id": None,
                "route_index_hint": self.ego_route_index, "consecutive_projection_failures": 0,
            }
            self.managed_actors.append(managed)

            print(
                f"[Canonical-Spawn] pedestrian id={walker.id} buffer_relative_s={relative_s:.1f}m "
                f"update={self.update_index}"
            )

            return walker

        return None

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    def destroy_all(self):
        for managed in self.managed_actors:
            self._destroy_managed_actor(managed)

        self.managed_actors = []

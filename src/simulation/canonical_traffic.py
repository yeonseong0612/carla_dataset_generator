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
    same_lane_front_gap_ok,
    far_enough,
    sidewalk_waypoint_near,
    largest_remainder_allocation,
)


TEMPORARY_COUNT_BASIS = (
    "annotation_candidate_count: AnnotationWriter per-frame category "
    "counts, now the actual camera_valid annotation count (ego-frame "
    "distance <= cfg.ANNOTATION.MAX_DISTANCE, AND left-RGB-camera FOV/"
    "truncation + depth-occlusion + minimum-pixel-size filtered) -- see "
    "src/data/annotation.py AnnotationWriter._compute_camera_validity"
)


def get_annotation_candidate_count(annotation_counts):
    return sum(annotation_counts.values())


class FrameObjectGammaSchedule:
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

    rng = np.random.default_rng(cfg.SPAWN.SEED)
    sampled = float(rng.gamma(cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE, cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE))
    target = int(round(sampled))

    return int(np.clip(target, cfg.SPAWN.FRAME_OBJECT_MIN, cfg.SPAWN.FRAME_OBJECT_MAX))


def scale_initial_category_counts(cfg, no_traffic=False, no_pedestrians=False):

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

class CanonicalBackgroundTraffic:
    def __init__(self, world, ego, dense_route, traffic_manager, cfg, rng, category_totals=None):
        self.world = world
        self.ego = ego
        self.dense_route = dense_route
        self.traffic_manager = traffic_manager
        self.cfg = cfg
        self.rng = rng

        self.carla_map = world.get_map()
        self.vehicle_pools = get_traffic_blueprints(world)

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

        self.category_targets = {
            category: category_totals.get(category, getattr(cfg.SPAWN, CATEGORY_TARGET_CFG_KEYS[category]))
            for category in DYNAMIC_CATEGORIES
        }

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

        self.total_spawned_inside_visible_roi = 0
        self.total_spawned_inside_buffer = 0

        self.total_density_pruned = 0
        self.total_pruned_inside_visible_roi = 0

        self.history = []

    def register_initial_actors(self, spawn_result):
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
        return

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

    def update(self, local_frame_id, current_object_count=None):

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

        category_counts = {category: 0 for category in DYNAMIC_CATEGORIES}

        for managed, _relative_s in live:
            category_counts[managed["category"]] += 1

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

        category_frame_targets = allocate_by_category_weight(frame_target, self.category_targets)

        tokens = []

        if total_deficit > 0:
            for category in DYNAMIC_CATEGORIES:
                category_deficit = max(category_frame_targets[category] - visible_category_counts[category], 0)
                tokens.extend([category] * category_deficit)

            self.rng.shuffle(tokens)

        max_new_this_update = min(self.cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE, total_deficit)

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

    def _try_spawn_buffer_vehicle_like(self, category, ego_route_s, live_vehicle_records, ego_road_id, ego_lane_id):
        blueprint_pool = self.vehicle_pools.get(category, [])

        if not blueprint_pool:
            return None, 0

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

            if not same_lane_front_gap_ok(
                lane_waypoint.road_id, lane_waypoint.lane_id, relative_s,
                ego_road_id, ego_lane_id,
                self.cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M,
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

    def destroy_all(self):
        for managed in self.managed_actors:
            self._destroy_managed_actor(managed)

        self.managed_actors = []

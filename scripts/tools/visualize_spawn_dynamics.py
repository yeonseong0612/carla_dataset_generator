"""
scripts/tools/visualize_spawn_dynamics.py

Phase 2 validation: drives ego along a real route on a live CARLA server
while src.simulation.spawn_policy.DynamicSpawnManager maintains the
ego-relative [MIN_DISTANCE, MAX_DISTANCE] forward Gamma density field,
then renders:

    outputs/<output-dir>/population_over_time.png
    outputs/<output-dir>/bin_occupancy.png
    outputs/<output-dir>/spawn_despawn_events.png

and prints the PART P quantitative summary (mean/min/max active counts
per category, totals spawned/despawned/failed, mean accepted relative_s,
mean bin MAE vs. the Gamma target shape).

This is a debug/validation tool, not the production pipeline: it drives
its own short route-following loop (reusing RouteController exactly as
scripts/collect_dataset.py does) and cleans up every actor it spawns
before exiting. Per project convention it does NOT force-exit the
process (no os._exit()) -- if CARLA's native client teardown crashes,
that is reported as-is rather than hidden.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

CARLA_ROOT = r"C:\CARLA"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla

from CFG.config import cfg

from scripts.collect_dataset import (
    CRUISE_SPEED_KMH,
    MIN_CURVE_SPEED_KMH,
    TRAFFIC_LIGHT_POLICY,
    resolve_carla_map_name,
    route_xml_path,
    spawn_ego_at_route_start,
)

from src.navigation.controller import RouteController
from src.navigation.route import build_dense_route, load_route_from_xml, project_control_points

from src.simulation.environment import disable_static_traffic_objects
from src.simulation.spawn_policy import DYNAMIC_CATEGORIES, spawn_actors_gamma_policy, DynamicSpawnManager
from src.simulation.traffic import configure_traffic_manager

ROUTE_SAMPLING_RESOLUTION = 2.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

CATEGORY_COLORS_BGR = {
    "vehicle": (255, 140, 0),
    "motorcyclist": (0, 165, 255),
    "cyclist": (0, 200, 0),
    "pedestrian": (255, 0, 255),
}


# ------------------------------------------------------------------
# A. Category population over time
# ------------------------------------------------------------------

def render_population_over_time(history, width=1000, height=560):
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 60, 50, 80, 160
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top

    updates = [snap["update_index"] for snap in history]
    max_update = max(updates) if updates else 1
    max_count = max(
        (max(snap["category_targets"].values()) for snap in history),
        default=1,
    )
    max_count = max(max_count, 1)

    def to_px(update_index, count):
        x = margin_left + (update_index / max_update) * plot_w if max_update else margin_left
        y = height - margin_bottom - (count / max_count) * plot_h
        return int(round(x)), int(round(y))

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)
    cv2.putText(canvas, "active actor count", (10, margin_top - 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "update index", (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for category in DYNAMIC_CATEGORIES:
        color = CATEGORY_COLORS_BGR[category]

        points = [to_px(snap["update_index"], snap["category_counts"][category]) for snap in history]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

        if history:
            target = history[-1]["category_targets"][category]
            _, y_target = to_px(0, target)
            cv2.line(canvas, (margin_left, y_target), (width - margin_right, y_target), color, 1, cv2.LINE_AA)

    legend_y = margin_top
    for category in DYNAMIC_CATEGORIES:
        color = CATEGORY_COLORS_BGR[category]
        lx = width - margin_right + 12
        cv2.line(canvas, (lx, legend_y), (lx + 20, legend_y), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, category, (lx + 26, legend_y + 5), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 24
    cv2.putText(canvas, "(thin line = target)", (width - margin_right + 12, legend_y + 4), FONT, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Category population over time (solid = active, thin = target)", (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# B. Distance-bin occupancy (final update snapshot, summed over category)
# ------------------------------------------------------------------

def render_bin_occupancy(history, bin_edges, width=900, height=520):
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    if not history:
        return canvas

    last = history[-1]
    n_bins = len(bin_edges) - 1

    current_total = np.zeros(n_bins, dtype=int)
    target_total = np.zeros(n_bins, dtype=int)

    for category in DYNAMIC_CATEGORIES:
        current_total += np.array(last["current_bin_counts"][category], dtype=int)
        target_total += np.array(last["target_bin_counts"][category], dtype=int)

    margin_left, margin_bottom, margin_top, margin_right = 60, 60, 70, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    group_w = plot_w / max(n_bins, 1)
    bar_w = group_w / 2.4
    max_count = max(int(current_total.max()), int(target_total.max()), 1)

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)

    for i in range(n_bins):
        group_x = margin_left + i * group_w
        y1 = height - margin_bottom

        h_target = int(round(target_total[i] / max_count * plot_h))
        x0, x1 = int(group_x + 3), int(group_x + 3 + bar_w)
        cv2.rectangle(canvas, (x0, y1 - h_target), (x1, y1), (90, 160, 255), -1)

        h_current = int(round(current_total[i] / max_count * plot_h))
        x2, x3 = int(group_x + 5 + bar_w), int(group_x + 5 + 2 * bar_w)
        cv2.rectangle(canvas, (x2, y1 - h_current), (x3, y1), (255, 140, 60), -1)

        cv2.putText(canvas, f"{int(bin_edges[i])}", (int(group_x + 3), height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{int(bin_edges[-1])}", (width - margin_right - 24, height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "relative_s bin (m)", (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "count (all categories)", (10, margin_top + 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = margin_top - 40
    cv2.rectangle(canvas, (margin_left, legend_y), (margin_left + 16, legend_y + 14), (90, 160, 255), -1)
    cv2.putText(canvas, "target (Gamma bin allocation)", (margin_left + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    lx2 = margin_left + 300
    cv2.rectangle(canvas, (lx2, legend_y), (lx2 + 16, legend_y + 14), (255, 140, 60), -1)
    cv2.putText(canvas, f"current (update={last['update_index']})", (lx2 + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Distance-bin occupancy: target vs. current (final update, summed over category)", (10, 25), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# C. Spawn / despawn / failed events over time
# ------------------------------------------------------------------

def render_events_over_time(history, width=1000, height=420):
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 60, 50, 40, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top

    n = len(history)
    max_count = max((max(s["spawned"], s["despawned"], s["failed"]) for s in history), default=1)
    max_count = max(max_count, 1)
    bar_w = plot_w / max(n, 1)

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)
    cv2.putText(canvas, "count", (10, margin_top + 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "update index", (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    colors = {"spawned": (90, 220, 90), "despawned": (90, 160, 255), "failed": (60, 60, 255)}

    for i, snap in enumerate(history):
        gx = margin_left + i * bar_w
        sub_w = bar_w / 3.5

        for j, key in enumerate(("spawned", "despawned", "failed")):
            h = int(round(snap[key] / max_count * plot_h))
            x0 = int(gx + j * sub_w)
            x1 = int(gx + (j + 1) * sub_w - 1)
            y1 = height - margin_bottom
            cv2.rectangle(canvas, (x0, y1 - h), (x1, y1), colors[key], -1)

    legend_x = margin_left
    for key, color in colors.items():
        cv2.rectangle(canvas, (legend_x, margin_top - 26), (legend_x + 14, margin_top - 14), color, -1)
        cv2.putText(canvas, key, (legend_x + 20, margin_top - 15), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        legend_x += 120

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Spawn / despawn / failed replenishment events per update", (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# Quantitative summary (PART P)
# ------------------------------------------------------------------

def print_summary(history, spawn_manager):
    print()
    print("=" * 70)
    print("Phase 2 dynamic spawn manager -- quantitative summary")
    print("=" * 70)
    print(f"Updates: {len(history)}")

    for category in DYNAMIC_CATEGORIES:
        counts = np.array([snap["category_counts"][category] for snap in history], dtype=np.float64)
        target = history[-1]["category_targets"][category] if history else 0

        print(
            f"{category:12s}: target={target:3d}  "
            f"mean={counts.mean():.2f} min={int(counts.min())} max={int(counts.max())}"
            if len(counts) else f"{category:12s}: no updates"
        )

    print()
    print(f"total spawned:   {spawn_manager.total_spawned}")
    print(f"total despawned: {spawn_manager.total_despawned}")
    print(f"total failed:    {spawn_manager.total_failed}")
    print(f"route projection failures: {spawn_manager.route_projection_failures}")

    all_relative_s = [snap["mean_relative_s"] for snap in history if not np.isnan(snap["mean_relative_s"])]
    if all_relative_s:
        print(f"mean accepted relative_s (avg over updates): {np.mean(all_relative_s):.2f} m")
    else:
        print("mean accepted relative_s: n/a")

    for category in DYNAMIC_CATEGORIES:
        maes = [snap["bin_mae"][category] for snap in history]
        if maes:
            print(f"mean bin MAE ({category}): {np.mean(maes):.3f}")

    print(f"final managed actor count: {history[-1]['managed_actor_count'] if history else 0}")
    print("=" * 70)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 dynamic spawn manager validation: drive ego along a route and plot density-over-time.")
    parser.add_argument("--town", type=str, default="Town10", help="--towns-style key used for routes/<town>.xml (map identifier is resolved from the route XML, not hardcoded)")
    parser.add_argument("--route", type=str, default="0", help="Route id inside routes/<town>.xml")
    parser.add_argument("--frames", type=int, default=400, help="Number of simulation frames to drive (300-600 recommended)")
    parser.add_argument("--output-dir", type=str, default="outputs/town10hd_dynamic_spawn_validation", help="Output directory for the plots")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)

    xml_path = route_xml_path(args.town)

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    carla_map_name = resolve_carla_map_name(xml_path)
    print(f"[Map] {args.town} -> {carla_map_name}")

    world = client.load_world(carla_map_name)
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    static_removed = disable_static_traffic_objects(world)
    print(f"[Environment] {args.town} static vehicles disabled: {static_removed['vehicles']}")

    traffic_manager = configure_traffic_manager(client, cfg)

    town, control_points = load_route_from_xml(xml_path, args.route)
    carla_map = world.get_map()
    control_waypoints = project_control_points(carla_map, control_points)
    dense_route = build_dense_route(carla_map, control_waypoints, sampling_resolution=ROUTE_SAMPLING_RESOLUTION)
    print(f"[Route] {town} route={args.route} control={len(control_waypoints)} dense={len(dense_route)}")

    ego = None
    spawn_manager = None

    try:
        ego = spawn_ego_at_route_start(world, dense_route)
        world.tick()

        route_controller = RouteController(
            vehicle=ego,
            dense_route=dense_route,
            cruise_speed=CRUISE_SPEED_KMH,
            min_curve_speed=MIN_CURVE_SPEED_KMH,
            traffic_light_policy=TRAFFIC_LIGHT_POLICY,
        )

        spawn_result = spawn_actors_gamma_policy(world, ego, dense_route, traffic_manager, cfg)
        print(
            f"[InitialSpawn] vehicle={len(spawn_result['traffic_actors']['vehicle'])} "
            f"motorcyclist={len(spawn_result['traffic_actors']['motorcyclist'])} "
            f"cyclist={len(spawn_result['traffic_actors']['cyclist'])} "
            f"pedestrian={len(spawn_result['walkers'])}"
        )

        spawn_manager = DynamicSpawnManager(world, ego, dense_route, traffic_manager, cfg, spawn_result["rng"])
        spawn_manager.register_initial_actors(spawn_result)
        spawn_manager.start_initial_pedestrians()

        for local_frame_id in range(args.frames):
            control = route_controller.run_step()
            ego.apply_control(control)

            world.tick()

            if local_frame_id % cfg.SPAWN.UPDATE_INTERVAL_FRAMES == 0:
                spawn_manager.update(local_frame_id)

        print_summary(spawn_manager.history, spawn_manager)

        pop_path = os.path.join(output_dir, "population_over_time.png")
        cv2.imwrite(pop_path, render_population_over_time(spawn_manager.history))
        print(f"[Output] {pop_path}")

        bin_path = os.path.join(output_dir, "bin_occupancy.png")
        cv2.imwrite(bin_path, render_bin_occupancy(spawn_manager.history, spawn_manager.bin_edges))
        print(f"[Output] {bin_path}")

        events_path = os.path.join(output_dir, "spawn_despawn_events.png")
        cv2.imwrite(events_path, render_events_over_time(spawn_manager.history))
        print(f"[Output] {events_path}")

    finally:
        if spawn_manager is not None:
            spawn_manager.destroy_all()

        if ego is not None:
            try:
                if ego.is_alive:
                    ego.destroy()
            except RuntimeError:
                pass

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()

"""
scripts/tools/visualize_spawn_distribution.py

Live CARLA validation for the Phase 1 Gamma route-relative spawn policy
(src/simulation/spawn_policy.py). There is no persisted "spawn record"
file in the dataset schema, so this tool drives the policy itself
(load map -> disable static objects -> spawn ego -> Gamma-policy spawn)
exactly as scripts/collect_dataset.py does, then renders:

    outputs/spawn_policy_validation/spawn_bev.png
    outputs/spawn_policy_validation/spawn_distance_histogram.png

and prints the PART N numeric validation (requested vs. spawned counts,
sampled_s statistics, min ego/actor-to-actor distance).

It cleans up every actor it spawns (and restores original world
settings) before exiting, so it doesn't leave litter on a shared CARLA
server. It never touches the dataset directory.
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

# Matches scripts/collect_dataset.py's Python-path setup exactly (only
# the carla PythonAPI root, not a separate "agents" entry -- "agents" is
# a subpackage imported as agents.navigation.* with that root on path).
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla

from CFG.config import cfg

from scripts.collect_dataset import (
    EGO_BLUEPRINT,
    resolve_carla_map_name,
    route_xml_path,
    spawn_ego_at_route_start,
)

from src.navigation.route import (
    build_dense_route,
    load_route_from_xml,
    project_control_points,
)

from src.simulation.environment import disable_static_traffic_objects
from src.simulation.spawn_policy import spawn_actors_gamma_policy
from src.simulation.traffic import configure_traffic_manager, destroy_traffic_vehicles
from src.simulation.pedestrian import destroy_pedestrians
from src.simulation.vehicle import destroy_vehicle

ROUTE_SAMPLING_RESOLUTION = 2.0

CATEGORY_COLORS_BGR = {
    "vehicle": (255, 140, 0),
    "motorcyclist": (0, 165, 255),
    "cyclist": (0, 200, 0),
    "pedestrian": (255, 0, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------
# BEV
# ------------------------------------------------------------------

def make_mapper(locations, margin=10.0, canvas_height=900):
    xs = [loc.x for loc in locations]
    ys = [loc.y for loc in locations]

    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    x_span = max(x_max - x_min, 1e-3)
    y_span = max(y_max - y_min, 1e-3)

    canvas_width = max(int(round(canvas_height * x_span / y_span)), 400)

    def to_px(x, y):
        px = int(round((x - x_min) / x_span * canvas_width))
        py = int(round(canvas_height - (y - y_min) / y_span * canvas_height))
        return px, py

    return to_px, canvas_width, canvas_height


def render_bev(dense_route, ego_location, spawn_records, title):
    route_locations = [waypoint.transform.location for waypoint, _ in dense_route]
    actor_locations = [carla.Location(x=r["x"], y=r["y"], z=r["z"]) for r in spawn_records]

    to_px, width, height = make_mapper(route_locations + actor_locations + [ego_location])

    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    # Route centerline.
    for i in range(1, len(route_locations)):
        p0 = to_px(route_locations[i - 1].x, route_locations[i - 1].y)
        p1 = to_px(route_locations[i].x, route_locations[i].y)
        cv2.line(canvas, p0, p1, (90, 90, 90), 2, cv2.LINE_AA)

    # Route distance ticks every 20m (arc-length along dense_route).
    accumulated = 0.0
    next_tick = 20.0
    for i in range(1, len(route_locations)):
        accumulated += route_locations[i - 1].distance(route_locations[i])
        if accumulated >= next_tick:
            px, py = to_px(route_locations[i].x, route_locations[i].y)
            cv2.circle(canvas, (px, py), 3, (140, 140, 140), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{int(next_tick)}m", (px + 5, py - 5), FONT, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
            next_tick += 20.0

    # Ego.
    ex, ey = to_px(ego_location.x, ego_location.y)
    cv2.drawMarker(canvas, (ex, ey), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 3)
    cv2.putText(canvas, "EGO", (ex + 8, ey - 6), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Spawned actors.
    for record in spawn_records:
        px, py = to_px(record["x"], record["y"])
        color = CATEGORY_COLORS_BGR.get(record["category"], (200, 200, 200))
        cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"#{record['actor_id']} {record['sampled_s']:.0f}m"
        cv2.putText(canvas, label, (px + 7, py + 4), FONT, 0.38, color, 1, cv2.LINE_AA)

    # Legend.
    legend_y = 24
    for category, color in CATEGORY_COLORS_BGR.items():
        cv2.circle(canvas, (20, legend_y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, category, (32, legend_y + 5), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 22

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 25), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# Histogram
# ------------------------------------------------------------------

def render_distance_histogram(sampled_s, min_distance, max_distance, bin_width=10.0, width=820, height=520):
    bin_edges = np.arange(min_distance, max_distance + 1e-6, bin_width)
    counts, edges = np.histogram(sampled_s, bins=bin_edges) if sampled_s else (np.zeros(len(bin_edges) - 1), bin_edges)

    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 60, 60, 50, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    n_bins = len(counts)
    bar_w = plot_w / max(n_bins, 1)
    max_count = max(int(counts.max()) if n_bins else 0, 1)

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)

    for i, count in enumerate(counts):
        bar_h = int(round(count / max_count * plot_h))
        x0 = int(round(margin_left + i * bar_w + 4))
        x1 = int(round(margin_left + (i + 1) * bar_w - 4))
        y1 = height - margin_bottom
        y0 = y1 - bar_h

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 160, 255), -1)
        cv2.putText(canvas, str(int(count)), (x0, max(y0 - 6, margin_top + 10)), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(edges[i])}", (x0, height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{int(edges[-1])}", (width - margin_right - 24, height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "sampled route distance s (m)", (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "count", (10, margin_top + 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Sampled route distance s -- Gamma(shape, scale), truncated", (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


PROPOSED_COLOR_BGR = (90, 160, 255)
ACCEPTED_COLOR_BGR = (255, 140, 60)


def render_distance_histogram_comparison(proposed_s, accepted_s, min_distance, max_distance, bin_width=10.0, width=900, height=580):
    """
    Grouped bar chart: every accepted Gamma draw used as an attempt
    ("proposed", before spacing/validity rejection) vs. the subset that
    actually got spawned ("accepted") -- lets you see whether an
    accepted-mean drift away from the raw Gamma shape comes from the
    rejection/spacing policy (near-route slots filling up first) rather
    than from the Gamma sampling itself.
    """

    bin_edges = np.arange(min_distance, max_distance + 1e-6, bin_width)
    n_bins = max(len(bin_edges) - 1, 0)

    proposed_counts, _ = np.histogram(proposed_s, bins=bin_edges) if proposed_s else (np.zeros(n_bins), bin_edges)
    accepted_counts, _ = np.histogram(accepted_s, bins=bin_edges) if accepted_s else (np.zeros(n_bins), bin_edges)

    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 60, 60, 70, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    group_w = plot_w / max(n_bins, 1)
    bar_w = group_w / 2.4
    max_count = max(
        int(proposed_counts.max()) if n_bins else 0,
        int(accepted_counts.max()) if n_bins else 0,
        1,
    )

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)

    for i in range(n_bins):
        group_x = margin_left + i * group_w
        y1 = height - margin_bottom

        h_proposed = int(round(proposed_counts[i] / max_count * plot_h))
        x0 = int(round(group_x + 3))
        x1 = int(round(group_x + 3 + bar_w))
        cv2.rectangle(canvas, (x0, y1 - h_proposed), (x1, y1), PROPOSED_COLOR_BGR, -1)

        h_accepted = int(round(accepted_counts[i] / max_count * plot_h))
        x2 = int(round(group_x + 5 + bar_w))
        x3 = int(round(group_x + 5 + 2 * bar_w))
        cv2.rectangle(canvas, (x2, y1 - h_accepted), (x3, y1), ACCEPTED_COLOR_BGR, -1)

        cv2.putText(canvas, f"{int(bin_edges[i])}", (int(group_x + 3), height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{int(bin_edges[-1])}", (width - margin_right - 24, height - margin_bottom + 18), FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "sampled route distance s (m)", (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "count", (10, margin_top + 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = margin_top - 40
    cv2.rectangle(canvas, (margin_left, legend_y), (margin_left + 16, legend_y + 14), PROPOSED_COLOR_BGR, -1)
    cv2.putText(canvas, f"proposed (Gamma draws, n={len(proposed_s)})", (margin_left + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    legend_x2 = margin_left + 300
    cv2.rectangle(canvas, (legend_x2, legend_y), (legend_x2 + 16, legend_y + 14), ACCEPTED_COLOR_BGR, -1)
    cv2.putText(canvas, f"accepted (spawned, n={len(accepted_s)})", (legend_x2 + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Proposed Gamma draws vs. accepted spawns", (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# Numeric validation (PART N)
# ------------------------------------------------------------------

def _distance_stats_line(label, values):
    if not values:
        print(f"{label}: n/a")
        return

    arr = np.asarray(values, dtype=np.float64)
    print(f"{label}: mean={arr.mean():.2f} median={np.median(arr):.2f} min={arr.min():.2f} max={arr.max():.2f} n={len(arr)}")


def print_validation(requested, spawn_result, spawn_records, proposed_s, ego_location):
    category_counts = {
        "vehicle": len(spawn_result["traffic_actors"]["vehicle"]),
        "motorcyclist": len(spawn_result["traffic_actors"]["motorcyclist"]),
        "cyclist": len(spawn_result["traffic_actors"]["cyclist"]),
        "pedestrian": len(spawn_result["walkers"]),
    }

    print()
    print("=" * 70)
    print("Gamma spawn policy validation")
    print("=" * 70)

    for category in ("vehicle", "motorcyclist", "cyclist", "pedestrian"):
        print(f"{category:12s}: requested={requested[category]:3d}  spawned={category_counts[category]:3d}")

    print()
    _distance_stats_line("proposed_s  (every Gamma draw attempted)", proposed_s)
    _distance_stats_line("accepted_s  (actually spawned)          ", [r["sampled_s"] for r in spawn_records])

    ego_distances = [carla.Location(x=r["x"], y=r["y"], z=r["z"]).distance(ego_location) for r in spawn_records]
    print()
    print(f"min ego distance: {min(ego_distances):.2f} m" if ego_distances else "min ego distance: n/a")

    # Spacing is now lane-aware, so a single blanket actor-to-actor
    # number is no longer one meaningful threshold -- report same-lane
    # longitudinal spacing, cross-lane Euclidean spacing, and
    # pedestrian-pedestrian spacing separately (matching the 3 distinct
    # rules actually enforced).
    vehicle_like = [r for r in spawn_records if r["category"] in ("vehicle", "motorcyclist", "cyclist")]
    pedestrians = [r for r in spawn_records if r["category"] == "pedestrian"]

    same_lane_gaps = []
    cross_lane_gaps = []

    for i in range(len(vehicle_like)):
        for j in range(i + 1, len(vehicle_like)):
            a, b = vehicle_like[i], vehicle_like[j]
            loc_a = carla.Location(x=a["x"], y=a["y"], z=a["z"])
            loc_b = carla.Location(x=b["x"], y=b["y"], z=b["z"])

            if a["road_id"] == b["road_id"] and a["lane_id"] == b["lane_id"]:
                same_lane_gaps.append(abs(a["sampled_s"] - b["sampled_s"]))
            else:
                cross_lane_gaps.append(loc_a.distance(loc_b))

    pedestrian_gaps = [
        carla.Location(x=pedestrians[i]["x"], y=pedestrians[i]["y"], z=pedestrians[i]["z"]).distance(
            carla.Location(x=pedestrians[j]["x"], y=pedestrians[j]["y"], z=pedestrians[j]["z"])
        )
        for i in range(len(pedestrians))
        for j in range(i + 1, len(pedestrians))
    ]

    print(f"min same-lane vehicle-like spacing (>= 8m rule): {min(same_lane_gaps):.2f} m" if same_lane_gaps else "min same-lane vehicle-like spacing: n/a (<2 in same lane)")
    print(f"min cross-lane vehicle-like spacing (anti-overlap only): {min(cross_lane_gaps):.2f} m" if cross_lane_gaps else "min cross-lane vehicle-like spacing: n/a (<2 on different lanes)")
    print(f"min pedestrian-pedestrian spacing (2-3m rule): {min(pedestrian_gaps):.2f} m" if pedestrian_gaps else "min pedestrian-pedestrian spacing: n/a (<2 pedestrians)")
    print("=" * 70)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate the Gamma route-relative spawn policy on a live CARLA server.")
    parser.add_argument("--town", type=str, default="Town10", help="--towns-style key used for routes/<town>.xml (map identifier is resolved from the route XML, not hardcoded)")
    parser.add_argument("--route", type=str, default="0", help="Route id inside routes/<town>.xml")
    parser.add_argument("--output-dir", type=str, default="outputs/spawn_policy_validation", help="Output directory for the BEV/histogram PNGs")
    parser.add_argument("--n-vehicles", type=int, default=None, help="Override cfg.SPAWN.N_VEHICLES for this validation run (more requests -> a denser distance histogram, useful for visually checking the Gamma shape)")
    parser.add_argument("--n-motorcycles", type=int, default=None, help="Override cfg.SPAWN.N_MOTORCYCLES")
    parser.add_argument("--n-bicycles", type=int, default=None, help="Override cfg.SPAWN.N_BICYCLES")
    parser.add_argument("--n-pedestrians", type=int, default=None, help="Override cfg.SPAWN.N_PEDESTRIANS")
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
    print(f"[Environment] {args.town} static pedestrians disabled: {static_removed['pedestrians']}")

    traffic_manager = configure_traffic_manager(client, cfg)

    town, control_points = load_route_from_xml(xml_path, args.route)
    carla_map = world.get_map()
    control_waypoints = project_control_points(carla_map, control_points)
    dense_route = build_dense_route(carla_map, control_waypoints, sampling_resolution=ROUTE_SAMPLING_RESOLUTION)

    print(f"[Route] {town} route={args.route} control={len(control_waypoints)} dense={len(dense_route)}")

    ego = None
    spawn_result = None

    try:
        ego = spawn_ego_at_route_start(world, dense_route)
        world.tick()

        spawn_result = spawn_actors_gamma_policy(
            world, ego, dense_route, traffic_manager, cfg,
            n_vehicles=args.n_vehicles,
            n_motorcycles=args.n_motorcycles,
            n_bicycles=args.n_bicycles,
            n_pedestrians=args.n_pedestrians,
        )
        world.tick()

        spawn_records = spawn_result["spawn_records"]
        proposed_s = spawn_result["proposed_s"]
        requested = spawn_result["requested"]

        print_validation(requested, spawn_result, spawn_records, proposed_s, ego.get_location())

        title = f"{args.town} / Route {args.route} -- Gamma spawn policy (seed={cfg.SPAWN.SEED})"
        bev = render_bev(dense_route, ego.get_location(), spawn_records, title)
        bev_path = os.path.join(output_dir, "spawn_bev.png")
        cv2.imwrite(bev_path, bev)
        print(f"[Output] {bev_path}")

        accepted_s = [r["sampled_s"] for r in spawn_records]
        histogram = render_distance_histogram_comparison(proposed_s, accepted_s, cfg.SPAWN.MIN_DISTANCE, cfg.SPAWN.MAX_DISTANCE)
        histogram_path = os.path.join(output_dir, "spawn_distance_histogram.png")
        cv2.imwrite(histogram_path, histogram)
        print(f"[Output] {histogram_path}")

    finally:
        if spawn_result is not None:
            destroy_pedestrians(spawn_result["walkers"], spawn_result["walker_controllers"])
            destroy_traffic_vehicles(spawn_result["traffic_actors"])

        if ego is not None:
            destroy_vehicle(ego)

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()

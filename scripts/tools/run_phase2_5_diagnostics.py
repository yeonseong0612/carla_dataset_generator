"""
scripts/tools/run_phase2_5_diagnostics.py

Phase 2.5 diagnostic: a controlled A/B/C experiment isolating why ego
covers so little route distance and why the managed actor registry
grows past the target total, on the same Town10HD route/seed/weather.

    Case A - no_npc:       ego only, no spawn at all.
    Case B - initial_only: Phase 1.5 Gamma initial spawn, no dynamic update.
    Case C - dynamic:      full Phase 2 (initial spawn + periodic update).

This is diagnostic-only. It does NOT change Gamma shape/scale, target
actor counts, spacing thresholds, UPDATE_INTERVAL_FRAMES,
MAX_NEW_ACTORS_PER_UPDATE, controller target speed, or TrafficManager
behavior -- see src/simulation/spawn_policy.py and
src/simulation/diagnostics.py for the (purely additive) instrumentation
this reuses. Only the minimum needed to drive the loop and record it is
here.

Outputs
-------
outputs/<output-dir>/<case>/ego_telemetry.csv
outputs/<output-dir>/dynamic/{actor_registry,spawn_events,projection_failures}.csv
outputs/<output-dir>/ego_speed_comparison.png
outputs/<output-dir>/ego_progress_comparison.png
outputs/<output-dir>/managed_actor_states.png   (dynamic only)
outputs/<output-dir>/nearest_lead_vehicle.png   (dynamic only)

plus a printed A/B/C comparison table.
"""

import argparse
import csv
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
    resolve_carla_map_name,
    route_xml_path,
    spawn_ego_at_route_start,
)

from src.navigation.controller import RouteController
from src.navigation.route import build_dense_route, load_route_from_xml, project_control_points

from src.simulation.diagnostics import ego_telemetry_row
from src.simulation.environment import disable_static_traffic_objects
from src.simulation.pedestrian import destroy_pedestrians
from src.simulation.spawn_policy import (
    DYNAMIC_CATEGORIES,
    DynamicSpawnManager,
    build_route_arc_length_table,
    route_progress_at_location,
    spawn_actors_gamma_policy,
)
from src.simulation.traffic import configure_traffic_manager, destroy_traffic_vehicles
from src.simulation.vehicle import destroy_vehicle

ROUTE_SAMPLING_RESOLUTION = 2.0
CRUISE_SPEED_KMH = 30.0
MIN_CURVE_SPEED_KMH = 12.0
TRAFFIC_LIGHT_POLICY = "obey"

CASES = ("no_npc", "initial_only", "dynamic")
CASE_LABELS = {"no_npc": "No NPC", "initial_only": "Initial Only", "dynamic": "Dynamic"}
CASE_COLORS_BGR = {"no_npc": (90, 220, 90), "initial_only": (60, 160, 255), "dynamic": (60, 60, 255)}

STATE_NAMES = ("behind_cleanup", "near_ego", "active_forward", "transition_forward", "forward_cleanup", "projection_failed")

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        print(f"[Output] {path} (0 rows)")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Output] {path} ({len(rows)} rows)")


# ------------------------------------------------------------------
# One case
# ------------------------------------------------------------------

def load_world_and_route(town, route_id):
    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)

    xml_path = route_xml_path(town)

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    carla_map_name = resolve_carla_map_name(xml_path)
    print(f"[Map] {town} -> {carla_map_name}")

    world = client.load_world(carla_map_name)
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    static_removed = disable_static_traffic_objects(world)
    print(f"[Environment] {town} static vehicles disabled: {static_removed['vehicles']}")

    traffic_manager = configure_traffic_manager(client, cfg)

    _town_name, control_points = load_route_from_xml(xml_path, route_id)
    carla_map = world.get_map()
    control_waypoints = project_control_points(carla_map, control_points)
    dense_route = build_dense_route(carla_map, control_waypoints, sampling_resolution=ROUTE_SAMPLING_RESOLUTION)

    print(f"[Route] route={route_id} control={len(control_waypoints)} dense={len(dense_route)}")

    return world, original_settings, traffic_manager, dense_route


def run_case(case, town, route_id, frames, output_dir):
    print()
    print("=" * 70)
    print(f"Case: {case} ({CASE_LABELS[case]})")
    print("=" * 70)

    world, original_settings, traffic_manager, dense_route = load_world_and_route(town, route_id)
    arc_length_table = build_route_arc_length_table(dense_route)

    ego = None
    spawn_result = None
    spawn_manager = None

    ego_rows = []
    registry_rows = []
    spawn_event_rows = []
    projection_failure_rows = []

    ego_route_index = 0

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

        if case in ("initial_only", "dynamic"):
            spawn_result = spawn_actors_gamma_policy(world, ego, dense_route, traffic_manager, cfg)

            print(
                f"[InitialSpawn] vehicle={len(spawn_result['traffic_actors']['vehicle'])} "
                f"motorcyclist={len(spawn_result['traffic_actors']['motorcyclist'])} "
                f"cyclist={len(spawn_result['traffic_actors']['cyclist'])} "
                f"pedestrian={len(spawn_result['walkers'])}"
            )

            if case == "dynamic":
                spawn_manager = DynamicSpawnManager(world, ego, dense_route, traffic_manager, cfg, spawn_result["rng"])
                spawn_manager.register_initial_actors(spawn_result)
                # Pedestrian AI controllers are deliberately never
                # started -- see DynamicSpawnManager.start_initial_pedestrians()
                # for the CARLA nav-mesh snap bug this avoids.

        update_index = 0

        for local_frame_id in range(frames):
            control = route_controller.run_step()
            ego.apply_control(control)

            world.tick()

            if local_frame_id % cfg.SPAWN.UPDATE_INTERVAL_FRAMES == 0:
                update_index += 1

                route_s, ego_route_index, _offset = route_progress_at_location(
                    ego.get_location(), dense_route, arc_length_table, ego_route_index,
                )

                if route_s is None:
                    route_s = arc_length_table[ego_route_index]

                ego_rows.append(
                    ego_telemetry_row(local_frame_id, update_index, ego, route_controller, control, route_s=route_s)
                )

                if case == "dynamic":
                    snapshot = spawn_manager.update(local_frame_id)

                    registry_row = {
                        "update_index": snapshot["update_index"],
                        "local_frame_id": snapshot["local_frame_id"],
                        "ego_route_s": snapshot["ego_route_s"],
                        "managed_total_before": snapshot["managed_total_before"],
                        "managed_actor_count": snapshot["managed_actor_count"],
                        "dead_actors_removed": snapshot["dead_actors_removed"],
                        "despawned": snapshot["despawned"],
                        "despawned_position": snapshot["despawned_position"],
                        "despawned_projection_failure": snapshot["despawned_projection_failure"],
                        "spawned": snapshot["spawned"],
                        "failed": snapshot["failed"],
                        "over_target_spawn_attempts_this_update": snapshot["over_target_spawn_attempts_this_update"],
                        "total_over_target_spawn_attempts": snapshot["total_over_target_spawn_attempts"],
                        "route_projection_failures_total": snapshot["route_projection_failures"],
                        "nearest_lead_distance": snapshot["nearest_lead_distance"],
                        "nearest_lead_actor_id": snapshot["nearest_lead_actor_id"],
                        "nearest_lead_speed_kmh": snapshot["nearest_lead_speed_kmh"],
                        # Phase 2.5 fix counters (PART H).
                        "projection_failure_despawns_this_update": snapshot["projection_failure_despawns_this_update"],
                        "total_projection_failure_despawns": snapshot["total_projection_failure_despawns"],
                        "full_search_attempts_this_update": snapshot["full_search_attempts_this_update"],
                        "total_projection_full_search_attempts": snapshot["total_projection_full_search_attempts"],
                        "full_search_recoveries_this_update": snapshot["full_search_recoveries_this_update"],
                        "total_projection_full_search_recoveries": snapshot["total_projection_full_search_recoveries"],
                        "same_lane_spawn_rejections_this_update": snapshot["same_lane_spawn_rejections_this_update"],
                        "total_same_lane_spawn_rejections": snapshot["total_same_lane_spawn_rejections"],
                        "blocked_bin_skips_this_update": snapshot["blocked_bin_skips_this_update"],
                        "total_blocked_bin_skips": snapshot["total_blocked_bin_skips"],
                        "category_cap_skips_this_update": snapshot["category_cap_skips_this_update"],
                        "total_category_cap_skips": snapshot["total_category_cap_skips"],
                    }

                    for state in STATE_NAMES:
                        registry_row[f"state_{state}"] = snapshot["state_totals"][state]

                    for category in DYNAMIC_CATEGORIES:
                        registry_row[f"count_{category}"] = snapshot["category_counts"][category]
                        registry_row[f"target_{category}"] = snapshot["category_targets"][category]

                    registry_rows.append(registry_row)

                    spawn_event_rows.append({
                        "update_index": snapshot["update_index"],
                        "local_frame_id": snapshot["local_frame_id"],
                        "spawned": snapshot["spawned"],
                        "despawned": snapshot["despawned"],
                        "failed": snapshot["failed"],
                        "over_target_spawn_attempts": snapshot["over_target_spawn_attempts_this_update"],
                    })

                    projection_failure_rows.extend(snapshot["projection_failed_details"])

        final_s = ego_rows[-1]["ego_route_s"] if ego_rows else 0.0
        print(f"[{case}] final ego_route_s={final_s:.2f}m over {frames} frames")

        # Write CSVs *before* actor cleanup: destroying spawned
        # vehicles/walkers/controllers has been observed to hard-crash
        # this CARLA client build (see report's "발견한 버그" section),
        # which would otherwise take the collected data down with it
        # even though the data itself is already complete and valid at
        # this point.
        case_dir = os.path.join(output_dir, case)
        write_csv(os.path.join(case_dir, "ego_telemetry.csv"), ego_rows)

        if case == "dynamic":
            write_csv(os.path.join(case_dir, "actor_registry.csv"), registry_rows)
            write_csv(os.path.join(case_dir, "spawn_events.csv"), spawn_event_rows)
            write_csv(os.path.join(case_dir, "projection_failures.csv"), projection_failure_rows)

    finally:
        if spawn_manager is not None:
            spawn_manager.destroy_all()
        elif spawn_result is not None:
            try:
                destroy_pedestrians(spawn_result["walkers"], spawn_result["walker_controllers"])
            except Exception as exc:
                print(f"[Cleanup] pedestrians: {exc}")

            try:
                destroy_traffic_vehicles(spawn_result["traffic_actors"])
            except Exception as exc:
                print(f"[Cleanup] traffic: {exc}")

        if ego is not None:
            try:
                if ego.is_alive:
                    ego.destroy()
            except RuntimeError:
                pass

        world.apply_settings(original_settings)

    return ego_rows, registry_rows


# ------------------------------------------------------------------
# Comparison summary
# ------------------------------------------------------------------

def summarize_case(ego_rows):
    if not ego_rows:
        return None

    speeds = np.array([r["speed_kmh"] for r in ego_rows], dtype=np.float64)

    return {
        "final_s": ego_rows[-1]["ego_route_s"],
        "mean_speed": float(speeds.mean()),
        "median_speed": float(np.median(speeds)),
        "min_speed": float(speeds.min()),
        "max_speed": float(speeds.max()),
        "below_5kmh_ratio": float(np.mean(speeds < 5.0)),
        "red_light_waiting_updates": sum(1 for r in ego_rows if r["waiting_red_light"]),
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def render_line_chart(series, title, y_label, x_label="update index", width=1000, height=520):
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 70, 50, 40, 200
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top

    all_x = [x for _l, _c, xs, _ys in series for x in xs]
    all_y = [y for _l, _c, _xs, ys in series for y in ys if y == y]  # drop NaN

    if not all_x or not all_y:
        return canvas

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(0.0, min(all_y)), max(all_y) * 1.05 if max(all_y) > 0 else 1.0
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    def to_px(x, y):
        px = margin_left + (x - x_min) / x_span * plot_w
        py = height - margin_bottom - (y - y_min) / y_span * plot_h
        return int(round(px)), int(round(py))

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)
    cv2.putText(canvas, y_label, (10, margin_top - 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (margin_left, height - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = margin_top
    for label, color, xs, ys in series:
        points = [to_px(x, y) for x, y in zip(xs, ys) if y == y]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

        lx = width - margin_right + 12
        cv2.line(canvas, (lx, legend_y), (lx + 20, legend_y), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx + 26, legend_y + 5), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 24

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


def render_ego_speed_comparison(all_ego_rows):
    series = [
        (CASE_LABELS[c], CASE_COLORS_BGR[c], [r["update_index"] for r in all_ego_rows.get(c, [])], [r["speed_kmh"] for r in all_ego_rows.get(c, [])])
        for c in CASES if all_ego_rows.get(c)
    ]
    return render_line_chart(series, "Ego speed over time (A/B/C comparison)", "speed (km/h)")


def render_ego_progress_comparison(all_ego_rows):
    series = [
        (CASE_LABELS[c], CASE_COLORS_BGR[c], [r["update_index"] for r in all_ego_rows.get(c, [])], [r["ego_route_s"] for r in all_ego_rows.get(c, [])])
        for c in CASES if all_ego_rows.get(c)
    ]
    return render_line_chart(series, "Ego route progress over time (A/B/C comparison)", "ego_route_s (m)")


def render_managed_actor_states(registry_rows):
    xs = [r["update_index"] for r in registry_rows]

    series = [
        ("managed_total", (220, 220, 220), xs, [r["managed_actor_count"] for r in registry_rows]),
        ("active_forward", (90, 220, 90), xs, [r["state_active_forward"] for r in registry_rows]),
        ("near_ego", (60, 160, 255), xs, [r["state_near_ego"] for r in registry_rows]),
        ("transition_forward", (0, 200, 255), xs, [r["state_transition_forward"] for r in registry_rows]),
        ("projection_failed", (60, 60, 255), xs, [r["state_projection_failed"] for r in registry_rows]),
    ]

    return render_line_chart(series, "Managed actor registry states over time (Dynamic case)", "count")


def render_nearest_lead_vehicle(ego_rows, registry_rows):
    xs = [r["update_index"] for r in registry_rows]
    lead = [r["nearest_lead_distance"] if r["nearest_lead_distance"] is not None else float("nan") for r in registry_rows]
    speed = [r["speed_kmh"] for r in ego_rows[:len(xs)]]

    series = [
        ("ego speed (km/h)", (60, 160, 255), xs, speed),
        ("nearest same-lane lead distance (m)", (90, 220, 90), xs, lead),
    ]

    return render_line_chart(series, "Ego speed vs. nearest same-lane lead distance (Dynamic case)", "value (km/h or m)")


def render_before_after_chart(baseline_rows_by_case, fixed_dynamic_rows, value_key, title, y_label):
    """
    Overlays the Phase 2.5 diagnostic baseline (No NPC / Initial Only /
    Dynamic-before-fix, read from --baseline-dir, not re-run) against
    this run's Dynamic-after-fix series, on one chart.
    """

    series = []

    if baseline_rows_by_case.get("no_npc"):
        rows = baseline_rows_by_case["no_npc"]
        series.append(("No NPC (baseline)", (90, 180, 90), [r["update_index"] for r in rows], [r[value_key] for r in rows]))

    if baseline_rows_by_case.get("initial_only"):
        rows = baseline_rows_by_case["initial_only"]
        series.append(("Initial Only (baseline)", (200, 160, 60), [r["update_index"] for r in rows], [r[value_key] for r in rows]))

    if baseline_rows_by_case.get("dynamic"):
        rows = baseline_rows_by_case["dynamic"]
        series.append(("Dynamic - BEFORE fix", (60, 60, 255), [r["update_index"] for r in rows], [r[value_key] for r in rows]))

    if fixed_dynamic_rows:
        series.append(("Dynamic - AFTER fix", (60, 220, 60), [r["update_index"] for r in fixed_dynamic_rows], [r[value_key] for r in fixed_dynamic_rows]))

    return render_line_chart(series, title, y_label)


def actor_max_streaks(rows):
    by_actor = {}

    for row in rows:
        actor_id = row["actor_id"]
        by_actor[actor_id] = max(by_actor.get(actor_id, 0), row["consecutive_failures"])

    return sorted(by_actor.values(), reverse=True)


def render_projection_failure_streaks(before_rows, after_rows, width=900, height=520):
    before_streaks = actor_max_streaks(before_rows) if before_rows else []
    after_streaks = actor_max_streaks(after_rows) if after_rows else []

    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    margin_left, margin_bottom, margin_top, margin_right = 60, 60, 70, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top

    n = max(len(before_streaks), len(after_streaks), 1)
    max_streak = max(before_streaks + after_streaks + [1])
    group_w = plot_w / n
    bar_w = group_w / 2.4

    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (200, 200, 200), 1)
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (200, 200, 200), 1)
    cv2.putText(canvas, "max consecutive projection failures", (10, margin_top - 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "actor rank (sorted desc, before vs after not the same actors)", (margin_left, height - 14), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    for i in range(n):
        gx = margin_left + i * group_w

        if i < len(before_streaks):
            h = int(round(before_streaks[i] / max_streak * plot_h))
            cv2.rectangle(canvas, (int(gx + 3), height - margin_bottom - h), (int(gx + 3 + bar_w), height - margin_bottom), (60, 60, 255), -1)

        if i < len(after_streaks):
            h = int(round(after_streaks[i] / max_streak * plot_h))
            cv2.rectangle(canvas, (int(gx + 5 + bar_w), height - margin_bottom - h), (int(gx + 5 + 2 * bar_w), height - margin_bottom), (60, 220, 60), -1)

    legend_y = margin_top - 40
    cv2.rectangle(canvas, (margin_left, legend_y), (margin_left + 16, legend_y + 14), (60, 60, 255), -1)
    cv2.putText(canvas, f"before (n_actors={len(before_streaks)})", (margin_left + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    lx2 = margin_left + 260
    cv2.rectangle(canvas, (lx2, legend_y), (lx2 + 16, legend_y + 14), (60, 220, 60), -1)
    cv2.putText(canvas, f"after (n_actors={len(after_streaks)})", (lx2 + 22, legend_y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    title_bar = np.full((36, width, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, "Projection failure streaks per actor (before vs after fix)", (10, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, canvas])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _try_parse(value):
    if value == "" or value is None:
        return None

    if value in ("True", "False"):
        return value == "True"

    try:
        parsed = float(value)
    except ValueError:
        return value

    if parsed.is_integer() and "." not in value and "e" not in value.lower():
        return int(parsed)

    return parsed


def read_csv_rows(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: _try_parse(v) for k, v in row.items()} for row in reader]


def main():
    parser = argparse.ArgumentParser(description="Phase 2.5 diagnostic: A/B/C ego-slowdown and managed-registry-growth experiment.")
    parser.add_argument("--town", type=str, default="Town10")
    parser.add_argument("--route", type=str, default="0")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="outputs/phase2_5_diagnostics")
    parser.add_argument("--cases", type=str, nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--compare-only", action="store_true", help="Skip running any case; just read the CSVs already written under --output-dir (each case can also be run as its own separate process/invocation -- see the module docstring, useful given the known CARLA cleanup crash below) and (re)produce the comparison table/plots.")
    parser.add_argument("--baseline-dir", type=str, default=None, help="A previous run's output dir (e.g. outputs/phase2_5_diagnostics, not re-run) to overlay as 'before' in *_before_after.png / projection_failure_streaks.png.")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_ego_rows = {}
    all_registry_rows = {}
    all_projection_rows = {}

    if args.compare_only:
        for case in args.cases:
            all_ego_rows[case] = read_csv_rows(os.path.join(output_dir, case, "ego_telemetry.csv"))

        all_registry_rows["dynamic"] = read_csv_rows(os.path.join(output_dir, "dynamic", "actor_registry.csv"))
        all_projection_rows["dynamic"] = read_csv_rows(os.path.join(output_dir, "dynamic", "projection_failures.csv"))
    else:
        for case in args.cases:
            ego_rows, registry_rows = run_case(case, args.town, args.route, args.frames, output_dir)
            all_ego_rows[case] = ego_rows
            all_registry_rows[case] = registry_rows

        if "dynamic" in args.cases:
            all_projection_rows["dynamic"] = read_csv_rows(os.path.join(output_dir, "dynamic", "projection_failures.csv"))

    print()
    print("=" * 100)
    print(f"{'case':14s} {'final_s':>9s} {'avg_speed':>10s} {'median_speed':>13s} {'min':>7s} {'max':>7s} {'<5km/h':>8s} {'red_light_upd':>14s}")
    print("=" * 100)

    for case in args.cases:
        summary = summarize_case(all_ego_rows.get(case))

        if summary is None:
            print(f"{CASE_LABELS[case]:14s} (no data)")
            continue

        print(
            f"{CASE_LABELS[case]:14s} {summary['final_s']:9.2f} {summary['mean_speed']:10.2f} "
            f"{summary['median_speed']:13.2f} {summary['min_speed']:7.2f} {summary['max_speed']:7.2f} "
            f"{summary['below_5kmh_ratio'] * 100:7.1f}% {summary['red_light_waiting_updates']:14d}"
        )

    print("=" * 100)

    if all_registry_rows.get("dynamic"):
        leads = [r["nearest_lead_distance"] for r in all_registry_rows["dynamic"] if r["nearest_lead_distance"] is not None]

        if leads:
            print(f"mean nearest same-lane lead distance (dynamic): {np.mean(leads):.2f} m (n={len(leads)}/{len(all_registry_rows['dynamic'])} updates)")
        else:
            print("mean nearest same-lane lead distance (dynamic): n/a (no same-lane lead found in any update)")

    if any(all_ego_rows.values()):
        speed_path = os.path.join(output_dir, "ego_speed_comparison.png")
        cv2.imwrite(speed_path, render_ego_speed_comparison(all_ego_rows))
        print(f"[Output] {speed_path}")

        progress_path = os.path.join(output_dir, "ego_progress_comparison.png")
        cv2.imwrite(progress_path, render_ego_progress_comparison(all_ego_rows))
        print(f"[Output] {progress_path}")

    if all_registry_rows.get("dynamic"):
        states_path = os.path.join(output_dir, "managed_actor_states.png")
        cv2.imwrite(states_path, render_managed_actor_states(all_registry_rows["dynamic"]))
        print(f"[Output] {states_path}")

        lead_path = os.path.join(output_dir, "nearest_lead_vehicle.png")
        cv2.imwrite(lead_path, render_nearest_lead_vehicle(all_ego_rows["dynamic"], all_registry_rows["dynamic"]))
        print(f"[Output] {lead_path}")

    if args.baseline_dir:
        baseline_dir = os.path.abspath(args.baseline_dir)
        baseline_rows_by_case = {
            case: read_csv_rows(os.path.join(baseline_dir, case, "ego_telemetry.csv"))
            for case in CASES
        }

        if all_ego_rows.get("dynamic"):
            speed_ba_path = os.path.join(output_dir, "ego_speed_before_after.png")
            cv2.imwrite(
                speed_ba_path,
                render_before_after_chart(
                    baseline_rows_by_case, all_ego_rows["dynamic"],
                    "speed_kmh", "Ego speed: before vs after Phase 2.5 fix", "speed (km/h)",
                ),
            )
            print(f"[Output] {speed_ba_path}")

            progress_ba_path = os.path.join(output_dir, "ego_progress_before_after.png")
            cv2.imwrite(
                progress_ba_path,
                render_before_after_chart(
                    baseline_rows_by_case, all_ego_rows["dynamic"],
                    "ego_route_s", "Ego route progress: before vs after Phase 2.5 fix", "route_s (m)",
                ),
            )
            print(f"[Output] {progress_ba_path}")

        baseline_projection_rows = read_csv_rows(os.path.join(baseline_dir, "dynamic", "projection_failures.csv"))
        if baseline_projection_rows and all_projection_rows.get("dynamic"):
            streaks_path = os.path.join(output_dir, "projection_failure_streaks.png")
            cv2.imwrite(
                streaks_path,
                render_projection_failure_streaks(baseline_projection_rows, all_projection_rows["dynamic"]),
            )
            print(f"[Output] {streaks_path}")


if __name__ == "__main__":
    main()

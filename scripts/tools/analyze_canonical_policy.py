"""
scripts/tools/analyze_canonical_policy.py

Read-only validation for the canonical background-traffic policy
(src/simulation/canonical_traffic.py). Never modifies any dataset file.

1. Observation-distance histogram
   -----------------------------
   Every dynamic object in every frame's labels/object_3d/*.json,
   bucketed into the same 10m bins Gamma(shape, scale) is truncated over
   (cfg.SPAWN.MIN_DISTANCE..MAX_DISTANCE at cfg.SPAWN.BIN_SIZE
   resolution) -- reuses compute_bin_edges_and_probabilities from
   src/simulation/spawn_policy.py verbatim, so the comparison target is
   exactly the same Gamma the spawn policy itself is seeded with, not a
   re-derived curve. Gamma here is a *sequence-level observation-distance
   prior*, not a per-frame occupancy target (that per-bin-forced-spawn
   approach is what canonical_traffic.py replaces) -- this histogram is
   how that prior is actually checked, after the fact, against the data.

2. Ego mobility
   ------------
   avg/median/min/max speed and <5km/h ratio read directly from
   ego_state.csv's speed_mps column (already computed by the collector,
   not re-derived here). Final route progress in meters is computed by
   projecting pose/poses.csv onto a freshly rebuilt dense_route (the
   exact same prepare_route() collect_dataset.py itself calls) via
   build_route_arc_length_table / route_progress_at_location, both
   reused unmodified from src/simulation/spawn_policy.py.

3. Nearest same-lane lead distance
   -------------------------------
   Per frame, the minimum distance_m among annotated objects with
   |y_ego| within half a lane width and ahead of ego (center_ego_m x >
   0) -- straight from the same labels/object_3d/*.json used in (1).
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

import carla  # noqa: E402

from src.data.layout import resolve_geometry_root  # noqa: E402
from CFG.config import cfg  # noqa: E402
from src.simulation.spawn_policy import (  # noqa: E402
    compute_bin_edges_and_probabilities,
    build_route_arc_length_table,
    route_progress_at_location,
)

LANE_HALF_WIDTH_M = 1.75


# ------------------------------------------------------------------
# Dataset IO (read-only)
# ------------------------------------------------------------------

def load_all_annotations(sequence_root):
    paths = sorted(glob.glob(os.path.join(resolve_geometry_root(sequence_root), "labels", "object_3d", "*.json")))

    if not paths:
        raise FileNotFoundError(f"No annotation files under {sequence_root}/labels/object_3d")

    frames = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            frames.append(json.load(f))

    return frames


def observation_distance_records(frames):
    records = []

    for frame in frames:
        for obj in frame.get("objects", []):
            records.append((obj["category"], float(obj["distance_m"])))

    return records


def nearest_same_lane_lead_per_frame(frames):
    leads = []

    for frame in frames:
        candidates = []

        for obj in frame.get("objects", []):
            x, y, _z = obj["bbox_3d"]["center_ego_m"]

            if x > 0 and abs(y) <= LANE_HALF_WIDTH_M:
                candidates.append(float(obj["distance_m"]))

        if candidates:
            leads.append(min(candidates))

    return leads


def load_ego_state(sequence_root):
    path = os.path.join(resolve_geometry_root(sequence_root), "ego_state.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_poses(sequence_root):
    path = os.path.join(resolve_geometry_root(sequence_root), "pose", "poses.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# Route-s projection (reuses spawn_policy.py's generic route utilities;
# rebuilds the dense_route the exact same way collect_dataset.py does)
# ------------------------------------------------------------------

def rebuild_dense_route(host, port, town, route_id):
    from scripts.collect_dataset import prepare_route, route_xml_path, resolve_carla_map_name

    xml_path = route_xml_path(town)

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    carla_map_name = resolve_carla_map_name(xml_path)

    client = carla.Client(host, port)
    client.set_timeout(cfg.CARLA.TIMEOUT)
    world = client.load_world(carla_map_name)

    _town, _control_points, dense_route = prepare_route(world, xml_path, route_id)

    return dense_route


def final_route_s_from_poses(poses, dense_route):
    arc_length_table = build_route_arc_length_table(dense_route)
    hint = 0
    last_route_s = 0.0

    for row in poses:
        location = carla.Location(x=float(row["x"]), y=float(row["y"]), z=float(row["z"]))
        route_s, hint, _offset = route_progress_at_location(
            location, dense_route, arc_length_table, hint,
            search_window=len(dense_route), max_offset=1e9,
        )

        if route_s is not None:
            last_route_s = route_s

    return last_route_s


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def percentile(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def print_observation_stats(records, bin_edges, bin_probabilities):
    total = len(records)
    distances = [d for _c, d in records]

    print()
    print("=" * 78)
    print("Gamma observation-distance validation")
    print("=" * 78)
    print(f"total object observations : {total}")

    by_category = {}
    for category, _d in records:
        by_category[category] = by_category.get(category, 0) + 1

    for category, count in sorted(by_category.items()):
        print(f"  {category:14s}: {count}")

    in_range = [d for d in distances if cfg.SPAWN.MIN_DISTANCE <= d <= cfg.SPAWN.MAX_DISTANCE]

    print(
        f"in [{cfg.SPAWN.MIN_DISTANCE:.0f}, {cfg.SPAWN.MAX_DISTANCE:.0f}]m range: "
        f"{len(in_range)} / {total} ({100.0 * len(in_range) / total:.1f}%)"
    )

    if distances:
        print(
            f"mean={np.mean(distances):.2f}m median={np.median(distances):.2f}m "
            f"p10={percentile(distances, 10):.2f}m p25={percentile(distances, 25):.2f}m "
            f"p75={percentile(distances, 75):.2f}m p90={percentile(distances, 90):.2f}m"
        )

    actual_counts = np.zeros(len(bin_probabilities), dtype=int)

    for d in in_range:
        bin_index = min(int((d - cfg.SPAWN.MIN_DISTANCE) // cfg.SPAWN.BIN_SIZE), len(bin_probabilities) - 1)
        actual_counts[bin_index] += 1

    target_counts = bin_probabilities * max(len(in_range), 1)

    print()
    print(f"{'bin':>12s} {'actual':>8s} {'target':>8s} {'error':>8s}")

    for i in range(len(bin_probabilities)):
        low, high = bin_edges[i], bin_edges[i + 1]
        error = actual_counts[i] - target_counts[i]
        print(f"{low:5.0f}-{high:<5.0f}m {actual_counts[i]:8d} {target_counts[i]:8.1f} {error:+8.1f}")

    mae = float(np.mean(np.abs(actual_counts - target_counts)))
    rmse = float(np.sqrt(np.mean((actual_counts - target_counts) ** 2)))
    print(f"bin count MAE={mae:.2f} RMSE={rmse:.2f} (target = Gamma(shape={cfg.SPAWN.GAMMA_SHAPE}, scale={cfg.SPAWN.GAMMA_SCALE}) truncated to this range)")

    return actual_counts, target_counts, by_category


def print_ego_mobility_stats(ego_state_rows, final_route_s):
    speeds_kmh = [float(row["speed_mps"]) * 3.6 for row in ego_state_rows]

    print()
    print("=" * 78)
    print("Ego mobility (canonical policy)")
    print("=" * 78)
    print(f"frames                : {len(speeds_kmh)}")
    print(f"avg speed             : {np.mean(speeds_kmh):.2f} km/h")
    print(f"median speed          : {np.median(speeds_kmh):.2f} km/h")
    print(f"min / max speed       : {min(speeds_kmh):.2f} / {max(speeds_kmh):.2f} km/h")
    below5 = sum(1 for s in speeds_kmh if s < 5.0) / len(speeds_kmh)
    print(f"<5 km/h ratio         : {below5 * 100:.1f}%")
    print(f"final ego_route_s     : {final_route_s:.2f} m")
    print()
    print("Prior Dynamic-policy baseline (600f, Town10 route 0, seed 42):")
    print("  avg speed=4.79 km/h  final_s=37.19m  <5km/h=60.0%")


def print_lead_stats(leads, n_frames):
    print()
    print("=" * 78)
    print("Nearest same-lane lead distance")
    print("=" * 78)

    if leads:
        print(f"present in {len(leads)}/{n_frames} frames, mean={np.mean(leads):.2f}m median={np.median(leads):.2f}m min={min(leads):.2f}m")
    else:
        print("no same-lane lead observed in any frame")


# ------------------------------------------------------------------
# Histogram plot (cv2, project convention -- no matplotlib)
# ------------------------------------------------------------------

def render_histogram(bin_edges, actual_counts, target_counts, out_path, width=1000, height=600):
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    n_bins = len(actual_counts)
    max_value = max(float(actual_counts.max()), float(target_counts.max()), 1.0) * 1.15

    def y_px(value):
        return margin_top + int(plot_h * (1.0 - value / max_value))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), (0, 0, 0), 1)

    group_w = plot_w / n_bins
    bar_w = group_w * 0.35

    for i in range(n_bins):
        group_center = margin_left + group_w * (i + 0.5)

        actual_x = int(group_center - bar_w * 0.6)
        target_x = int(group_center + bar_w * 0.1)

        actual_top = y_px(actual_counts[i])
        target_top = y_px(target_counts[i])
        base_y = margin_top + plot_h

        cv2.rectangle(canvas, (actual_x, actual_top), (int(actual_x + bar_w), base_y), (60, 60, 220), -1)
        cv2.rectangle(canvas, (target_x, target_top), (int(target_x + bar_w), base_y), (60, 180, 60), 2)

        label = f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        cv2.putText(canvas, label, (int(group_center - 18), margin_top + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_value * frac
        y = y_px(value)
        cv2.line(canvas, (margin_left - 4, y), (margin_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (width - 260, 20), (width - 244, 34), (60, 60, 220), -1)
    cv2.putText(canvas, "Actual (observed)", (width - 236, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 260, 40), (width - 244, 54), (60, 180, 60), 2)
    cv2.putText(canvas, "Gamma target", (width - 236, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(
        canvas, "Object observation-distance histogram vs. truncated-Gamma prior",
        (margin_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(canvas, "distance bin (m)", (margin_left + plot_w // 2 - 50, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "count", (10, margin_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Canonical background-traffic policy validation: Gamma observation-distance histogram + ego mobility + nearest same-lane lead, all read-only from an already-collected dataset sequence."
    )
    parser.add_argument("--sequence", type=str, required=True, help="Path to a sequence root, e.g. outputs/canonical_policy_validation/Town10/route_0/conditions/day_clear")
    parser.add_argument("--town", type=str, default="Town10", help="Town key for routes/<town>.xml (route_s projection needs a live CARLA connection)")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="outputs/canonical_policy_validation", help="Where observation_distance_histogram.png is written")
    parser.add_argument("--skip-route-projection", action="store_true", help="Skip the live-CARLA route_s projection (ego mobility speed/ratio stats still print from ego_state.csv)")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames = load_all_annotations(sequence_root)
    records = observation_distance_records(frames)
    bin_edges, bin_probabilities = compute_bin_edges_and_probabilities(cfg)

    actual_counts, target_counts, _by_category = print_observation_stats(records, bin_edges, bin_probabilities)

    histogram_path = os.path.join(output_dir, "observation_distance_histogram.png")
    render_histogram(bin_edges, actual_counts, target_counts, histogram_path)
    print(f"\n[Output] {histogram_path}")

    ego_state_rows = load_ego_state(sequence_root)

    if args.skip_route_projection:
        final_route_s = float("nan")
    else:
        poses = load_poses(sequence_root)
        dense_route = rebuild_dense_route(cfg.CARLA.HOST, cfg.CARLA.PORT, args.town, args.route_id)
        final_route_s = final_route_s_from_poses(poses, dense_route)

    print_ego_mobility_stats(ego_state_rows, final_route_s)

    leads = nearest_same_lane_lead_per_frame(frames)
    print_lead_stats(leads, len(frames))


if __name__ == "__main__":
    main()

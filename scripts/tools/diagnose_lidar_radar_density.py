"""
scripts/tools/diagnose_lidar_radar_density.py

Diagnostic-only tool for the "LiDAR Density Audit + LiDAR vs Radar
Density Sanity Check" task. Never modifies LiDAR/Radar config, sensor
pose, calibration, the collector, annotation, spawn policy, or the
canonical traffic policy -- it only reads runtime blueprint attributes,
runs a short live CARLA capture purely to measure pre-ROI vs post-ROI
point counts (not obtainable from an already-saved dataset, since the
collector never persists the pre-ROI count), and otherwise reads back
an already-collected dataset (outputs/canonical_policy_validation) for
the distance/object-level breakdowns.

Every measurement function below reuses the project's own production
code (src/sensors/lidar.py, src/sensors/radar.py, src/data/collector.py,
scripts/tools/visualize_multisensor.py) rather than re-implementing the
blueprint setup, ROI filter, or ego-frame transform logic.

Sections (see main()):
    A. LiDAR + Radar runtime blueprint attribute audit
    B. Theoretical point-count arithmetic (current config + 3 candidates)
    C. Live raw-vs-ROI LiDAR count + live Radar count (>=50 frames)
    D. Distance-wise LiDAR point density (existing dataset)
    E. Object-level LiDAR point statistics (existing dataset)
    F. Radar per-sensor/merged count (existing dataset, cross-checked
       against the live measurement in C)
    G. Comparison visualizations -> outputs/lidar_radar_density_audit/
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

from CFG.config import cfg  # noqa: E402

from src.sensors.lidar import create_lidar_blueprint, create_lidar_transform  # noqa: E402
from src.sensors.radar import create_radar_blueprint, create_radar_transform  # noqa: E402
from src.sensors.sensor_rig import SensorRig, RADAR_RIG_SPECS  # noqa: E402
from src.data.collector import lidar_to_numpy, filter_lidar_roi, radar_to_numpy, wait_for_frame  # noqa: E402
from src.simulation.traffic import configure_traffic_manager  # noqa: E402
from src.simulation.spawn_policy import spawn_actors_gamma_policy  # noqa: E402
from src.simulation.environment import disable_static_traffic_objects  # noqa: E402

from scripts.collect_dataset import (  # noqa: E402
    prepare_route,
    route_xml_path,
    resolve_carla_map_name,
    spawn_ego_at_route_start,
)

from scripts.tools.visualize_multisensor import (  # noqa: E402
    RADAR_SENSOR_NAMES,
    MERGED_RADAR_KEY,
    load_calibration,
    load_lidar,
    load_all_radars_ego,
    get_extrinsic,
    lidar_points_to_ego,
    object_footprint,
    count_points_in_footprint,
    ROI,
    bev_frame_size,
    render_lidar_bev,
    render_radar_bev,
    compose_panels,
    load_rgb,
    load_annotation,
    discover_available_frames,
    CameraProjector,
    render_rgb_projection,
    parse_sequence_title,
)


# ================================================================
# Section A -- runtime blueprint attribute audit
# ================================================================

LIDAR_ATTRS_OF_INTEREST = [
    "channels", "range", "points_per_second", "rotation_frequency",
    "horizontal_fov", "upper_fov", "lower_fov",
    "dropoff_general_rate", "dropoff_intensity_limit", "dropoff_zero_intensity",
    "atmosphere_attenuation_rate", "noise_stddev", "sensor_tick",
]

RADAR_ATTRS_OF_INTEREST = [
    "horizontal_fov", "vertical_fov", "range", "points_per_second", "sensor_tick",
]


def attribute_value_str(attribute):
    """
    carla.ActorAttribute.as_str()/as_float()/... each only work for that
    attribute's own declared type (calling the wrong one raises), so
    dispatch on attribute.type instead of guessing.
    """

    try:
        attr_type = attribute.type

        if attr_type == carla.ActorAttributeType.Bool:
            return str(attribute.as_bool())
        if attr_type == carla.ActorAttributeType.Int:
            return str(attribute.as_int())
        if attr_type == carla.ActorAttributeType.Float:
            return str(attribute.as_float())
        if attr_type == carla.ActorAttributeType.String:
            return attribute.as_str()

        return str(attribute.as_str())
    except Exception as exc:
        return f"<unreadable: {exc}>"


def dump_blueprint_attributes(blueprint, attrs_of_interest):
    """
    (highlighted, all_attrs) where all_attrs is {name: value_str} for
    every attribute CARLA exposes on this blueprint instance (not just
    attrs_of_interest) -- "추측하지 말고 실제 runtime blueprint에서
    읽어라" per the task.
    """

    all_attrs = {attribute.id: attribute_value_str(attribute) for attribute in blueprint}

    highlighted = {name: all_attrs.get(name, "<not present on this blueprint>") for name in attrs_of_interest}

    return highlighted, all_attrs


def section_a_blueprint_audit(world):
    print()
    print("=" * 90)
    print("SECTION A -- LiDAR runtime blueprint attribute audit")
    print("=" * 90)

    lidar_bp = create_lidar_blueprint(world, cfg)
    lidar_highlighted, lidar_all_attrs = dump_blueprint_attributes(lidar_bp, LIDAR_ATTRS_OF_INTEREST)

    explicitly_set = {"channels", "range", "points_per_second", "rotation_frequency", "horizontal_fov", "upper_fov", "lower_fov", "sensor_tick"}

    print(f"{'attribute':28s} {'value':>14s}   source")
    for name in LIDAR_ATTRS_OF_INTEREST:
        source = "set by create_lidar_blueprint()" if name in explicitly_set else "CARLA blueprint DEFAULT (never set by our code)"
        print(f"{name:28s} {lidar_highlighted[name]:>14s}   {source}")

    print(f"\ntotal attributes exposed by this blueprint instance: {len(lidar_all_attrs)}")

    print()
    print("=" * 90)
    print("SECTION A -- Radar runtime blueprint attribute audit (Front / Front-Left / Front-Right)")
    print("=" * 90)

    radar_dumps = {}

    for sensor_name, cfg_attr in RADAR_RIG_SPECS:
        radar_cfg = getattr(cfg.SENSOR, cfg_attr)
        radar_bp = create_radar_blueprint(world, radar_cfg)
        highlighted, all_attrs = dump_blueprint_attributes(radar_bp, RADAR_ATTRS_OF_INTEREST)
        radar_dumps[sensor_name] = highlighted

        print(f"\n[{sensor_name}] ({cfg_attr})")
        for name in RADAR_ATTRS_OF_INTEREST:
            print(f"  {name:20s} {highlighted[name]:>10s}")

        dropoff_like = [a for a in all_attrs if "dropoff" in a.lower()]
        print(f"  dropoff-like attributes on sensor.other.radar: {dropoff_like if dropoff_like else 'NONE -- radar has no dropoff model in CARLA'}")

    return lidar_highlighted, radar_dumps


# ================================================================
# Section B -- theoretical arithmetic (pure math, no CARLA)
# ================================================================

def section_b_theoretical(dropoff_general_rate):
    print()
    print("=" * 90)
    print("SECTION B -- Theoretical point-count arithmetic")
    print("=" * 90)

    fps = cfg.SIMULATION.FPS

    candidates = [
        ("Current  (16ch / 100k PPS)", 16, 100000),
        ("Candidate A (32ch / 100k PPS)", 32, 100000),
        ("Candidate B (32ch / 200k PPS)", 32, 200000),
    ]

    print(f"{'candidate':32s} {'pts/frame':>12s} {'pts/channel/frame':>20s} {'vertical density':>18s} {'horizontal density':>20s}")

    baseline_per_channel = None

    for label, channels, pps in candidates:
        per_frame = pps / fps
        per_channel = per_frame / channels

        if baseline_per_channel is None:
            baseline_per_channel = per_channel
            vertical_note = "baseline"
            horizontal_note = "baseline"
        else:
            vertical_note = f"{channels / 16:.1f}x channels -> {channels/16:.1f}x vertical samples"
            horizontal_note = f"{per_channel / baseline_per_channel:.2f}x baseline horiz. density/channel"

        print(f"{label:32s} {per_frame:12.1f} {per_channel:20.1f} {vertical_note:>18s} {horizontal_note:>20s}")

    print()
    print(f"dropoff_general_rate what-if (approximate -- CARLA's actual dropoff also depends on")
    print(f"dropoff_intensity_limit / dropoff_zero_intensity per-point, this is the dominant term only):")
    print(f"{'candidate':32s} {'theoretical':>12s} {'* (1-dropoff)':>16s} {'dropoff=0':>12s}")

    for label, channels, pps in candidates:
        theoretical = pps / fps
        with_dropoff = theoretical * (1.0 - dropoff_general_rate)
        print(f"{label:32s} {theoretical:12.1f} {with_dropoff:16.1f} {theoretical:12.1f}")


# ================================================================
# Section C -- live raw-vs-ROI LiDAR + live Radar measurement
# ================================================================

def run_live_capture(client, world, town, route_id, n_frames):
    xml_path = route_xml_path(town)
    carla_map_name = resolve_carla_map_name(xml_path)

    world = client.load_world(carla_map_name)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    traffic_manager = configure_traffic_manager(client, cfg)

    disable_static_traffic_objects(world)

    _town, _control_points, dense_route = prepare_route(world, xml_path, route_id)

    ego = spawn_ego_at_route_start(world, dense_route)
    world.tick()

    ego.set_autopilot(True, cfg.TRAFFIC_MANAGER.PORT)

    spawn_result = spawn_actors_gamma_policy(world, ego, dense_route, traffic_manager, cfg)
    print(
        f"[LiveCapture] background traffic spawned: "
        f"vehicle={len(spawn_result['traffic_actors']['vehicle'])} "
        f"motorcyclist={len(spawn_result['traffic_actors']['motorcyclist'])} "
        f"cyclist={len(spawn_result['traffic_actors']['cyclist'])} "
        f"pedestrian={len(spawn_result['walkers'])}"
    )

    rig = SensorRig(world, ego, cfg).spawn()
    world.tick()

    T_ego_from_lidar = np.array(create_lidar_transform(cfg).get_matrix(), dtype=np.float64)

    rows = []

    try:
        for frame_index in range(n_frames):
            target_frame = world.tick()

            lidar_data = wait_for_frame(rig.get_queue("lidar"), target_frame, sensor_name="lidar")
            radar_data = {
                name: wait_for_frame(rig.get_queue(name), target_frame, sensor_name=name)
                for name, _cfg_attr in RADAR_RIG_SPECS
            }

            rig.clear_queues()

            lidar_raw = lidar_to_numpy(lidar_data)
            lidar_roi = filter_lidar_roi(lidar_raw, T_ego_from_lidar, cfg)

            radar_counts = {name: len(radar_to_numpy(data)) for name, data in radar_data.items()}

            rows.append({
                "frame_index": frame_index,
                "lidar_raw": len(lidar_raw),
                "lidar_roi": len(lidar_roi),
                "radar_front": radar_counts["radar"],
                "radar_front_left": radar_counts["radar_front_left"],
                "radar_front_right": radar_counts["radar_front_right"],
                "radar_merged": sum(radar_counts.values()),
            })

    finally:
        try:
            rig.destroy()
        except Exception as exc:
            print(f"[LiveCapture][Cleanup] rig: {exc}")

        try:
            for actor in spawn_result["traffic_actors"]["vehicle"] + spawn_result["traffic_actors"]["motorcyclist"] + spawn_result["traffic_actors"]["cyclist"]:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            for controller in spawn_result["walker_controllers"]:
                if controller is not None and controller.is_alive:
                    controller.destroy()
            for walker in spawn_result["walkers"]:
                if walker is not None and walker.is_alive:
                    walker.destroy()
        except Exception as exc:
            print(f"[LiveCapture][Cleanup] traffic: {exc}")

        try:
            if ego is not None and ego.is_alive:
                ego.destroy()
        except Exception as exc:
            print(f"[LiveCapture][Cleanup] ego: {exc}")

    return rows


def stats_row(values, name):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "name": name, "mean": float(arr.mean()), "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()), "std": float(arr.std()),
    }


def print_stats_table(rows):
    print(f"{'':18s} {'mean':>10s} {'median':>10s} {'min':>8s} {'max':>8s} {'std':>8s}")
    for row in rows:
        print(f"{row['name']:18s} {row['mean']:10.1f} {row['median']:10.1f} {row['min']:8.0f} {row['max']:8.0f} {row['std']:8.1f}")


def section_c_live_capture(client, world, town, route_id, n_frames, output_dir):
    print()
    print("=" * 90)
    print(f"SECTION C -- Live raw-vs-ROI LiDAR + live Radar measurement ({n_frames} frames)")
    print("=" * 90)

    rows = run_live_capture(client, world, town, route_id, n_frames)

    csv_path = os.path.join(output_dir, "live_capture_counts.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Output] {csv_path}")

    lidar_raw = [r["lidar_raw"] for r in rows]
    lidar_roi = [r["lidar_roi"] for r in rows]
    radar_front = [r["radar_front"] for r in rows]
    radar_fl = [r["radar_front_left"] for r in rows]
    radar_fr = [r["radar_front_right"] for r in rows]
    radar_merged = [r["radar_merged"] for r in rows]

    print()
    print("LiDAR: raw (pre-ROI, = the sensor-local callback count -- CARLA's Python API never exposes")
    print("a separate 'pre-dropoff' ray count, so this IS post-dropoff already) vs roi (= saved .npy count):")
    print_stats_table([stats_row(lidar_raw, "lidar_raw"), stats_row(lidar_roi, "lidar_roi(=saved)")])

    theoretical = cfg.SENSOR.LIDAR.POINTS_PER_SECOND / cfg.SIMULATION.FPS
    mean_raw = float(np.mean(lidar_raw))
    mean_roi = float(np.mean(lidar_roi))

    print()
    print(f"theoretical (PPS/FPS)     = {theoretical:.1f}")
    print(f"measured raw (post-dropoff, pre-ROI) = {mean_raw:.1f}  (theoretical -> raw: {theoretical - mean_raw:+.1f}, {100*(1 - mean_raw/theoretical):.1f}% reduction)")
    print(f"measured roi (post-ROI)  = {mean_roi:.1f}  (raw -> roi: {mean_raw - mean_roi:+.1f}, {100*(1 - mean_roi/mean_raw) if mean_raw else 0:.1f}% reduction)")

    print()
    print("Radar (live, same frames):")
    print_stats_table([
        stats_row(radar_front, "front"), stats_row(radar_fl, "front_left"),
        stats_row(radar_fr, "front_right"), stats_row(radar_merged, "merged"),
    ])

    return rows, {
        "lidar_raw_mean": mean_raw, "lidar_roi_mean": mean_roi, "theoretical": theoretical,
        "radar_merged_mean": float(np.mean(radar_merged)),
    }


# ================================================================
# Section D -- distance-wise LiDAR density (existing dataset)
# ================================================================

DISTANCE_BINS_LIDAR = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]


def section_d_distance_density(sequence_root, frame_ids, calibration):
    print()
    print("=" * 90)
    print("SECTION D -- Distance-wise LiDAR point density (existing dataset, ego frame)")
    print("=" * 90)

    T_ego_from_lidar = get_extrinsic(calibration, "lidar")

    bin_counts = {b: [] for b in DISTANCE_BINS_LIDAR}

    for frame_id in frame_ids:
        lidar_arr = load_lidar(sequence_root, frame_id)
        xyz_ego, _intensity = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)

        x = xyz_ego[:, 0]
        y = xyz_ego[:, 1]

        in_roi = (x >= cfg.SENSOR.LIDAR.ROI_FRONT_MIN) & (x <= cfg.SENSOR.LIDAR.ROI_FRONT_MAX) & (np.abs(y) <= cfg.SENSOR.LIDAR.ROI_SIDE)
        x_in = x[in_roi]

        for low, high in DISTANCE_BINS_LIDAR:
            bin_counts[(low, high)].append(int(((x_in >= low) & (x_in < high)).sum()))

    y_width = 2.0 * cfg.SENSOR.LIDAR.ROI_SIDE

    print(f"{'bin':>10s} {'mean pts/frame':>16s} {'area (m^2)':>12s} {'pts/m^2':>10s}")

    results = {}

    for low, high in DISTANCE_BINS_LIDAR:
        values = bin_counts[(low, high)]
        mean_v = float(np.mean(values))
        area = (high - low) * y_width
        density = mean_v / area
        results[(low, high)] = {"mean": mean_v, "area_m2": area, "density_per_m2": density}
        print(f"{low:>4d}-{high:<4d}m {mean_v:16.1f} {area:12.0f} {density:10.4f}")

    return results


# ================================================================
# Section E -- object-level LiDAR point statistics (existing dataset)
# ================================================================

OBJECT_DISTANCE_BINS = [(0, 30), (30, 60), (60, 100)]

VEHICLE_SUBCATS = ["car", "van", "truck", "bus"]
OTHER_CATEGORIES = ["motorcyclist", "cyclist", "pedestrian"]


def object_label(obj):
    if obj.get("category") == "vehicle":
        return obj.get("subcategory") or "car"
    return obj.get("category")


def section_e_object_level(sequence_root, frame_ids, calibration):
    print()
    print("=" * 90)
    print("SECTION E -- Object-level LiDAR point statistics (existing dataset)")
    print("=" * 90)

    T_ego_from_lidar = get_extrinsic(calibration, "lidar")

    labels = VEHICLE_SUBCATS + OTHER_CATEGORIES
    per_combo = {(label, b): [] for label in labels for b in OBJECT_DISTANCE_BINS}

    for frame_id in frame_ids:
        annotation = load_annotation(sequence_root, frame_id)
        lidar_arr = load_lidar(sequence_root, frame_id)
        xyz_ego, _intensity = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)
        points_xy = xyz_ego[:, :2]

        for obj in annotation.get("objects", []):
            label = object_label(obj)

            if label not in labels:
                continue

            # Annotations are scene-level GT (every dynamic object within
            # ANNOTATION.MAX_DISTANCE, in any direction), but the LiDAR
            # only covers the forward ROI (x in [ROI_FRONT_MIN,
            # ROI_FRONT_MAX], |y| <= ROI_SIDE). An object behind/beside
            # that window can never have a LiDAR point by construction --
            # counting it toward "zero-point ratio" would conflate
            # "outside the sensor's physical coverage" with "sensor is
            # too sparse", so it's excluded here instead of bucketed by
            # raw 3D distance_m alone.
            center = obj.get("bbox_3d", {}).get("center_ego_m")

            if center is None:
                continue

            x_ego, y_ego = center[0], center[1]

            if not (cfg.SENSOR.LIDAR.ROI_FRONT_MIN <= x_ego <= cfg.SENSOR.LIDAR.ROI_FRONT_MAX and abs(y_ego) <= cfg.SENSOR.LIDAR.ROI_SIDE):
                continue

            dist = obj.get("distance_m")

            if not isinstance(dist, (int, float)):
                continue

            bin_key = None
            for low, high in OBJECT_DISTANCE_BINS:
                if low <= dist < high:
                    bin_key = (low, high)
                    break

            if bin_key is None:
                continue

            footprint = object_footprint(obj, margin=0.2)

            if footprint is None:
                continue

            count = count_points_in_footprint(points_xy, footprint)
            per_combo[(label, bin_key)].append(count)

    print(f"{'category':14s} {'dist bin':>10s} {'n_objects':>10s} {'mean pts':>10s} {'median pts':>11s} {'zero-pt ratio':>14s}")

    results = {}

    for label in labels:
        for low, high in OBJECT_DISTANCE_BINS:
            values = per_combo[(label, (low, high))]

            if not values:
                print(f"{label:14s} {f'{low}-{high}m':>10s} {0:>10d} {'--':>10s} {'--':>11s} {'--':>14s}")
                continue

            arr = np.asarray(values, dtype=np.float64)
            zero_ratio = float((arr == 0).sum()) / len(arr)

            results[(label, low, high)] = {
                "n": len(values), "mean": float(arr.mean()), "median": float(np.median(arr)), "zero_ratio": zero_ratio,
            }

            print(f"{label:14s} {f'{low}-{high}m':>10s} {len(values):>10d} {arr.mean():10.2f} {np.median(arr):11.1f} {zero_ratio*100:13.1f}%")

    return results


# ================================================================
# Section F -- Radar count from the existing dataset (cross-check)
# ================================================================

def section_f_radar_existing(sequence_root, frame_ids, calibration):
    print()
    print("=" * 90)
    print("SECTION F -- Radar per-sensor / merged count (existing dataset, cross-check vs Section C)")
    print("=" * 90)

    per_source = {name: [] for name in RADAR_SENSOR_NAMES}
    merged = []

    for frame_id in frame_ids:
        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)

        for name in RADAR_SENSOR_NAMES:
            per_source[name].append(radar_by_source[name][0].shape[0])

        merged.append(radar_by_source[MERGED_RADAR_KEY][0].shape[0])

    rows = [stats_row(per_source[name], name) for name in RADAR_SENSOR_NAMES]
    rows.append(stats_row(merged, "merged"))
    print_stats_table(rows)

    return rows


# ================================================================
# Section G -- comparison visualizations
# ================================================================

def render_distance_histogram_comparison(lidar_density, sequence_root, frame_ids, calibration, out_path, width=1100, height=560):
    T_ego_from_lidar = get_extrinsic(calibration, "lidar")

    radar_bin_counts = {b: [] for b in DISTANCE_BINS_LIDAR}

    for frame_id in frame_ids:
        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)
        merged_xyz, _vel = radar_by_source[MERGED_RADAR_KEY]
        x = merged_xyz[:, 0]
        y = merged_xyz[:, 1]

        in_roi = (x >= cfg.SENSOR.LIDAR.ROI_FRONT_MIN) & (x <= cfg.SENSOR.LIDAR.ROI_FRONT_MAX) & (np.abs(y) <= cfg.SENSOR.LIDAR.ROI_SIDE)
        x_in = x[in_roi]

        for low, high in DISTANCE_BINS_LIDAR:
            radar_bin_counts[(low, high)].append(int(((x_in >= low) & (x_in < high)).sum()))

    lidar_means = [lidar_density[b]["mean"] for b in DISTANCE_BINS_LIDAR]
    radar_means = [float(np.mean(radar_bin_counts[b])) for b in DISTANCE_BINS_LIDAR]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_v = max(max(lidar_means), max(radar_means), 1.0) * 1.15

    def y_px(v):
        return margin_t + int(plot_h * (1.0 - v / max_v))

    cv2.line(canvas, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 1)

    n_bins = len(DISTANCE_BINS_LIDAR)
    group_w = plot_w / n_bins
    bar_w = group_w * 0.35

    for i, (low, high) in enumerate(DISTANCE_BINS_LIDAR):
        center = margin_l + group_w * (i + 0.5)
        lx = int(center - bar_w * 0.6)
        rx = int(center + bar_w * 0.1)

        cv2.rectangle(canvas, (lx, y_px(lidar_means[i])), (int(lx + bar_w), margin_t + plot_h), (60, 60, 220), -1)
        cv2.rectangle(canvas, (rx, y_px(radar_means[i])), (int(rx + bar_w), margin_t + plot_h), (0, 160, 0), -1)

        cv2.putText(canvas, f"{low}-{high}", (int(center - 20), margin_t + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = max_v * frac
        y = y_px(v)
        cv2.putText(canvas, f"{v:.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (width - 220, 20), (width - 204, 34), (60, 60, 220), -1)
    cv2.putText(canvas, "LiDAR (saved, mean/frame)", (width - 196, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 220, 40), (width - 204, 54), (0, 160, 0), -1)
    cv2.putText(canvas, "Merged Radar (mean/frame)", (width - 196, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "LiDAR vs Radar distance-band density (mean points/frame per 20m bin)", (margin_l, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "distance bin (m)", (margin_l + plot_w // 2 - 60, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def section_g_visuals(sequence_root, frame_ids, calibration, lidar_density, output_dir):
    print()
    print("=" * 90)
    print("SECTION G -- Comparison visualizations")
    print("=" * 90)

    os.makedirs(output_dir, exist_ok=True)

    projector = CameraProjector(calibration, "rgb_left")
    T_ego_from_lidar = get_extrinsic(calibration, "lidar")
    width_px, height_px = bev_frame_size(ROI, 640)

    for frame_id in frame_ids:
        annotation = load_annotation(sequence_root, frame_id)
        image = load_rgb(sequence_root, frame_id, "rgb_left")
        rendered_rgb, rgb_stats = render_rgb_projection(image, annotation, projector)

        lidar_arr = load_lidar(sequence_root, frame_id)
        lidar_xyz_ego, lidar_intensity = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)
        lidar_canvas = render_lidar_bev(lidar_xyz_ego, lidar_intensity, ROI, width_px, height_px, annotation)

        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)
        merged_xyz, merged_vel = radar_by_source[MERGED_RADAR_KEY]
        radar_canvas = render_radar_bev(merged_xyz, merged_vel, ROI, width_px, height_px, annotation, source_label="Front+FL+FR merged")

        title = parse_sequence_title(sequence_root, frame_id) + f"  |  lidar={lidar_arr.shape[0]}pts  radar_merged={merged_xyz.shape[0]}pts"

        composite = compose_panels(
            [("RGB + 3D Bounding Boxes", rendered_rgb), ("LiDAR BEV", lidar_canvas), ("Multi-Radar BEV", radar_canvas)],
            title,
        )

        out_path = os.path.join(output_dir, f"{frame_id:06d}_rgb_lidar_radar.png")
        cv2.imwrite(out_path, composite)
        print(f"  frame {frame_id:06d}: lidar={lidar_arr.shape[0]} radar_merged={merged_xyz.shape[0]} -> {out_path}")

    hist_path = os.path.join(output_dir, "lidar_vs_radar_distance_density.png")
    render_distance_histogram_comparison(lidar_density, sequence_root, frame_ids, calibration, hist_path)
    print(f"[Output] {hist_path}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="LiDAR density + LiDAR-vs-Radar sanity-check diagnostic. Read-only w.r.t. config/collector/annotation/spawn policy.")
    parser.add_argument("--sequence", type=str, default="outputs/canonical_policy_validation/Town10/route_0/conditions/day_clear", help="Already-collected sequence to reuse for Sections D/E/F/G")
    parser.add_argument("--town", type=str, default="Town10")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--live-frames", type=int, default=80, help="Frames captured live for Section C (>=50 required by the task)")
    parser.add_argument("--num-samples", type=int, default=6, help="Frames rendered for Section G")
    parser.add_argument("--output-dir", type=str, default="outputs/lidar_radar_density_audit")
    parser.add_argument("--skip-live", action="store_true", help="Skip Section C's live CARLA capture (e.g. to only regenerate D/E/F/G from the existing dataset)")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)
    world = client.get_world()

    highlighted_lidar, radar_dumps = section_a_blueprint_audit(world)
    dropoff_general_rate = float(highlighted_lidar["dropoff_general_rate"])

    section_b_theoretical(dropoff_general_rate)

    live_summary = None

    if not args.skip_live:
        _live_rows, live_summary = section_c_live_capture(
            client, world, args.town, args.route_id, args.live_frames, output_dir,
        )

    calibration = load_calibration(sequence_root)
    frame_ids = discover_available_frames(sequence_root)

    if not frame_ids:
        print(f"No frames found under {sequence_root}; Sections D/E/F/G skipped.")
        return

    lidar_density = section_d_distance_density(sequence_root, frame_ids, calibration)
    section_e_object_level(sequence_root, frame_ids, calibration)
    section_f_radar_existing(sequence_root, frame_ids, calibration)

    sample_frames = frame_ids if len(frame_ids) <= args.num_samples else [
        frame_ids[i] for i in sorted(set(int(round(x)) for x in np.linspace(0, len(frame_ids) - 1, args.num_samples)))
    ]
    section_g_visuals(sequence_root, sample_frames, calibration, lidar_density, output_dir)

    if live_summary is not None:
        print()
        print("=" * 90)
        print("Quick cross-check: live Section C merged-radar mean vs existing-dataset Section F merged mean")
        print("=" * 90)
        print(f"live (Section C)    : {live_summary['radar_merged_mean']:.1f}")
        print(f"existing (Section F): see table above")


if __name__ == "__main__":
    main()

"""
scripts/tools/final_integration_validation.py

CLAUDE.md task: "Final Integrated Validation Before Production Dataset
Generation" -- runs ONE real collect_dataset.py sequence (Town01/route0/
day_clear/seed42, canonical policy, 1000-1500 frames -- Town10 is
avoided per the task, it has a known route-index-~100 stall) and audits
everything: camera-valid annotation stats, Gamma/population tracking,
spawn/prune safety incl. bus count, sensor synchronization, actual
directory/schema/calibration/pose naming (read from disk, never
guessed), naming consistency, storage size, and generation throughput.

Deliberately does NOT delete the collected sequence afterward (unlike
the earlier validation scripts) -- this task's whole point is auditing
the real files on disk.
"""

import argparse
import glob
import json
import os
import sys
import time
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
from src.data.projection import CameraProjector  # noqa: E402
from scripts.tools.validate_frame_object_gamma import sequence_root  # noqa: E402
from scripts.tools.validate_frame_object_gamma_longrun import run_collection_with_log  # noqa: E402
from scripts.tools.validate_camera_valid_annotations import (  # noqa: E402
    per_frame_stats,
    render_diagnostic_frame,
    load_calibration,
)


# ------------------------------------------------------------------
# Collection (timed, for throughput measurement)
# ------------------------------------------------------------------

def run_collection_timed(town, route_id, condition, max_frames, output_root, log_path):
    start = time.time()
    run_collection_with_log(town, route_id, condition, max_frames, output_root, log_path, overwrite=True)
    elapsed = time.time() - start
    return elapsed


# ------------------------------------------------------------------
# Camera-valid annotation + Gamma/population tracking
# ------------------------------------------------------------------

def load_frame_object_csv(sequence_dir):
    import csv
    path = os.path.join(resolve_geometry_root(sequence_dir), "frame_object_counts.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (int(v) if v.lstrip("-").isdigit() else v) for k, v in row.items()})
    return rows


def discover_annotation_frames(sequence_dir):
    paths = sorted(glob.glob(os.path.join(resolve_geometry_root(sequence_dir), "labels", "object_3d", "*.json")))
    frames = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


def annotation_stats(frames):
    per_frame = [per_frame_stats(f) for f in frames]
    total_candidates = sum(r["raw_candidate_count"] for r in per_frame)
    total_valid = sum(r["final_camera_valid_count"] for r in per_frame)

    reason_counts = {"outside_fov": 0, "projection_invalid": 0, "too_truncated": 0, "too_occluded": 0, "too_small": 0}
    bus_count = 0

    for f in frames:
        for obj in f.get("rejected_objects", []):
            reason = obj.get("camera_projection", {}).get("invalid_reason")
            if reason in reason_counts:
                reason_counts[reason] += 1
            if obj.get("subcategory") == "bus":
                bus_count += 1
        for obj in f.get("objects", []):
            if obj.get("subcategory") == "bus":
                bus_count += 1

    n = len(frames)

    return {
        "n_frames": n,
        "mean_raw_per_frame": total_candidates / n if n else None,
        "mean_valid_per_frame": total_valid / n if n else None,
        "retention_ratio": total_valid / total_candidates if total_candidates else None,
        "rejection_reason_counts": reason_counts,
        "rejection_reason_ratios": {k: (v / total_candidates if total_candidates else None) for k, v in reason_counts.items()},
        "bus_actor_count_in_annotations": bus_count,
        "total_raw_candidates": total_candidates,
        "total_valid": total_valid,
    }


def gamma_and_population_stats(rows):
    n = len(rows)
    actual = [r["actual_object_count"] for r in rows]
    target = [r["target_object_count"] for r in rows]
    diffs = [a - t for a, t in zip(actual, target)]
    abs_diffs = [abs(d) for d in diffs]

    population = [r["managed_population"] for r in rows]
    third = n // 3

    return {
        "n_frames": n,
        "mean_target": float(np.mean(target)),
        "mean_actual": float(np.mean(actual)),
        "mean_abs_lag": float(np.mean(abs_diffs)),
        "median_abs_lag": float(np.median(abs_diffs)),
        "within_1": sum(1 for d in abs_diffs if d <= 1) / n,
        "within_2": sum(1 for d in abs_diffs if d <= 2) / n,
        "within_3": sum(1 for d in abs_diffs if d <= 3) / n,
        "over_target_ratio": sum(1 for d in diffs if d > 0) / n,
        "under_target_ratio": sum(1 for d in diffs if d < 0) / n,
        "population": {
            "initial": population[0], "max": max(population), "final": population[-1],
            "mean_first_third": float(np.mean(population[:third])) if third > 0 else None,
            "mean_middle_third": float(np.mean(population[third:2 * third])) if third > 0 else None,
            "mean_final_third": float(np.mean(population[2 * third:])) if third > 0 else None,
        },
        "total_spawned": rows[-1]["cumulative_spawned"],
        "total_pruned": rows[-1]["cumulative_pruned"],
        "total_natural_despawn": rows[-1]["cumulative_natural_despawn"],
    }


# ------------------------------------------------------------------
# Sensor synchronization
# ------------------------------------------------------------------

SENSOR_DIR_EXT = {
    "rgb_left": ".png", "rgb_right": ".png",
    "depth": ".npy", "optical_flow": ".npy", "semantic": ".npy",
    "lidar": ".npy", "radar": ".npy",
    "labels/object_3d": ".json",
    "pose": None,  # single poses.csv, not per-frame files
}


def sensor_sync_check(sequence_dir, expected_frames):
    report = {}
    expected_ids = set(range(expected_frames))

    for name, ext in SENSOR_DIR_EXT.items():
        if ext is None:
            continue

        # RGB is per weather condition; everything else lives in geometry/.
        base_dir = sequence_dir if name.startswith("rgb_") else resolve_geometry_root(sequence_dir)
        dir_path = os.path.join(base_dir, *name.split("/"))
        files = sorted(glob.glob(os.path.join(dir_path, f"*{ext}")))
        ids = []
        duplicates = []
        seen = set()

        for p in files:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit():
                fid = int(stem)
                if fid in seen:
                    duplicates.append(fid)
                seen.add(fid)
                ids.append(fid)

        id_set = set(ids)
        missing = sorted(expected_ids - id_set)
        extra = sorted(id_set - expected_ids)

        report[name] = {
            "file_count": len(files),
            "missing_vs_expected": missing,
            "extra_vs_expected": extra,
            "duplicate_ids": duplicates,
        }

    # pose/poses.csv row count check
    import csv
    pose_path = os.path.join(resolve_geometry_root(sequence_dir), "pose", "poses.csv")
    if os.path.isfile(pose_path):
        with open(pose_path, newline="", encoding="utf-8") as f:
            pose_rows = list(csv.DictReader(f))
        report["pose/poses.csv"] = {"row_count": len(pose_rows), "expected": expected_frames}

    return report


# ------------------------------------------------------------------
# Schema / metadata audits (read actual files, never guessed)
# ------------------------------------------------------------------

def directory_structure(sequence_dir, max_entries_per_dir=3):
    """Shared geometry/ entries, then the condition directory's entries."""

    structure = {}
    roots = [("", resolve_geometry_root(sequence_dir))]

    if os.path.abspath(sequence_dir) != os.path.abspath(roots[0][1]):
        roots.append((f"[condition {os.path.basename(sequence_dir)}] ", sequence_dir))

    for prefix, root in roots:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full):
                children = sorted(os.listdir(full))
                structure[prefix + entry + "/"] = {
                    "count": len(children),
                    "sample": children[:max_entries_per_dir],
                }
                # one level deeper for labels/
                if entry == "labels":
                    for sub in children:
                        sub_full = os.path.join(full, sub)
                        if os.path.isdir(sub_full):
                            sub_children = sorted(os.listdir(sub_full))
                            structure[f"{prefix}labels/{sub}/"] = {"count": len(sub_children), "sample": sub_children[:max_entries_per_dir]}
            else:
                structure[prefix + entry] = {"size_bytes": os.path.getsize(full)}
    return structure


def annotation_json_schema(sequence_dir, frame_id=0):
    path = os.path.join(resolve_geometry_root(sequence_dir), "labels", "object_3d", f"{frame_id:06d}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_keys = list(data.keys())
    object_keys = list(data["objects"][0].keys()) if data.get("objects") else None
    camera_projection_keys = list(data["objects"][0]["camera_projection"].keys()) if object_keys and "camera_projection" in data["objects"][0] else None
    bbox_3d_keys = list(data["objects"][0]["bbox_3d"].keys()) if object_keys and "bbox_3d" in data["objects"][0] else None
    rejected_keys = list(data["rejected_objects"][0].keys()) if data.get("rejected_objects") else None

    return {
        "source_file": path,
        "top_level_keys": top_keys,
        "object_keys": object_keys,
        "bbox_3d_keys": bbox_3d_keys,
        "camera_projection_keys": camera_projection_keys,
        "rejected_objects_present": "rejected_objects" in data,
        "rejected_object_keys": rejected_keys,
    }


def calibration_schema(sequence_dir):
    path = os.path.join(resolve_geometry_root(sequence_dir), "calibration.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "source_file": path,
        "top_level_keys": list(data.keys()),
        "coordinate_systems_keys": list(data.get("coordinate_systems", {}).keys()),
        "camera_names": list(data.get("cameras", {}).keys()),
        "camera_keys_example": list(next(iter(data.get("cameras", {}).values())).keys()) if data.get("cameras") else None,
        "sensor_names": list(data.get("sensors", {}).keys()),
        "sensor_keys_example": list(next(iter(data.get("sensors", {}).values())).keys()) if data.get("sensors") else None,
        "stereo_keys": list(data.get("stereo", {}).keys()),
    }


def pose_navigation_schema(sequence_dir):
    import csv

    result = {}
    pose_path = os.path.join(resolve_geometry_root(sequence_dir), "pose", "poses.csv")
    if os.path.isfile(pose_path):
        with open(pose_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader, None)
        result["pose/poses.csv"] = {"columns": header, "first_row_example": first_row}

    ego_state_path = os.path.join(resolve_geometry_root(sequence_dir), "ego_state.csv")
    if os.path.isfile(ego_state_path):
        with open(ego_state_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        result["ego_state.csv"] = {"columns": header}

    return result


def naming_consistency_check(sequence_dir):
    findings = {}
    for name, ext in SENSOR_DIR_EXT.items():
        if ext is None:
            continue
        # RGB is per weather condition; everything else lives in geometry/.
        base_dir = sequence_dir if name.startswith("rgb_") else resolve_geometry_root(sequence_dir)
        dir_path = os.path.join(base_dir, *name.split("/"))
        files = sorted(os.listdir(dir_path)) if os.path.isdir(dir_path) else []
        if not files:
            continue
        stems = [os.path.splitext(fn)[0] for fn in files]
        widths = set(len(s) for s in stems if s.isdigit())
        exts = set(os.path.splitext(fn)[1] for fn in files)
        findings[name] = {"zero_pad_widths_seen": sorted(widths), "extensions_seen": sorted(exts)}
    return findings


# ------------------------------------------------------------------
# Storage size
# ------------------------------------------------------------------

def dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def storage_report(sequence_dir, n_frames):
    per_sensor = {}
    for name in ("rgb_left", "rgb_right", "depth", "optical_flow", "semantic", "lidar", "radar"):
        p = os.path.join(sequence_dir if name.startswith("rgb_") else resolve_geometry_root(sequence_dir), name)
        if os.path.isdir(p):
            per_sensor[name] = dir_size_bytes(p)

    labels_dir = os.path.join(resolve_geometry_root(sequence_dir), "labels")
    if os.path.isdir(labels_dir):
        per_sensor["labels"] = dir_size_bytes(labels_dir)

    geometry_root = resolve_geometry_root(sequence_dir)
    total_bytes = dir_size_bytes(geometry_root)

    if os.path.abspath(sequence_dir) != os.path.abspath(geometry_root):
        total_bytes += dir_size_bytes(sequence_dir)

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "mb_per_frame": (total_bytes / (1024 * 1024)) / n_frames if n_frames else None,
        "per_sensor_mb": {k: v / (1024 * 1024) for k, v in per_sensor.items()},
        "per_sensor_mb_per_frame": {k: (v / (1024 * 1024)) / n_frames if n_frames else None for k, v in per_sensor.items()},
    }


# ------------------------------------------------------------------
# Representative visualizations
# ------------------------------------------------------------------

def render_representative_frames(sequence_dir, projector, frames_by_id, frame_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for frame_id in frame_ids:
        frame = frames_by_id.get(frame_id)
        if frame is None:
            continue
        out_path = os.path.join(out_dir, f"{frame_id:06d}.png")
        if render_diagnostic_frame(sequence_dir, projector, frame, out_path):
            saved.append(out_path)
    return saved


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", type=str, default="Town01")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--collection-output-root", type=str, default=os.path.join(cfg.PROJECT.ROOT, "dataset_final_integration_validation"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(cfg.PROJECT.ROOT, "outputs", "final_integration_validation"))
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collection_stdout.log")

    elapsed_seconds = None
    if not args.skip_collection:
        elapsed_seconds = run_collection_timed(args.town, args.route_id, args.condition, args.max_frames, args.collection_output_root, log_path)

    seq_dir = sequence_root(args.collection_output_root, args.town, args.route_id, args.condition)

    frame_rows = load_frame_object_csv(seq_dir)
    n_frames = len(frame_rows)

    annotation_frames = discover_annotation_frames(seq_dir)
    frames_by_id = {f["frame_id"]: f for f in annotation_frames}

    ann_stats = annotation_stats(annotation_frames)
    gamma_pop_stats = gamma_and_population_stats(frame_rows)

    spawn_summary_path = os.path.join(resolve_geometry_root(seq_dir), "canonical_spawn_summary.json")
    with open(spawn_summary_path, "r", encoding="utf-8") as f:
        spawn_summary = json.load(f)

    sync_report = sensor_sync_check(seq_dir, n_frames)

    struct = directory_structure(seq_dir)
    ann_schema = annotation_json_schema(seq_dir, frame_id=0)
    calib_schema = calibration_schema(seq_dir)
    pose_nav_schema = pose_navigation_schema(seq_dir)
    naming = naming_consistency_check(seq_dir)
    storage = storage_report(seq_dir, n_frames)

    calibration = load_calibration(seq_dir)
    projector = CameraProjector.from_calibration(calibration, "rgb_left")

    last_frame_id = max(frames_by_id.keys()) if frames_by_id else 0
    requested_ids = [0, 100, 250, 400, 600, 800, 1000, last_frame_id]
    viz_frame_ids = sorted(set(fid for fid in requested_ids if fid in frames_by_id))
    viz_dir = os.path.join(output_dir, "visualizations")
    viz_saved = render_representative_frames(seq_dir, projector, frames_by_id, viz_frame_ids, viz_dir)

    throughput = None
    if elapsed_seconds is not None and n_frames > 0:
        fps = n_frames / elapsed_seconds
        throughput = {
            "frames_generated": n_frames,
            "elapsed_seconds": elapsed_seconds,
            "effective_fps": fps,
            "est_seconds_per_100k_frames": 100_000 / fps,
            "est_hours_per_100k_frames": (100_000 / fps) / 3600,
            "est_seconds_per_600k_frames": 600_000 / fps,
            "est_hours_per_600k_frames": (600_000 / fps) / 3600,
        }

    summary = {
        "run": {
            "town": args.town, "route_id": args.route_id, "condition": args.condition,
            "requested_max_frames": args.max_frames, "n_frames_analyzed": n_frames,
            "sequence_dir": seq_dir,
        },
        "annotation_stats": ann_stats,
        "reference_600f": {"mean_raw_per_frame": 10.94, "mean_valid_per_frame": 5.49, "retention_ratio": 0.502},
        "gamma_population_stats": gamma_pop_stats,
        "spawn_prune_safety": {
            "spawned_inside_visible_roi": spawn_summary.get("spawned_inside_visible_roi"),
            "pruned_inside_visible_roi": spawn_summary.get("pruned_inside_visible_roi"),
            "bus_actor_count_in_annotations": ann_stats["bus_actor_count_in_annotations"],
        },
        "sensor_sync": sync_report,
        "directory_structure": struct,
        "annotation_json_schema": ann_schema,
        "calibration_schema": calib_schema,
        "pose_navigation_schema": pose_nav_schema,
        "naming_consistency": naming,
        "storage": storage,
        "reference_storage": {"mb_per_frame": 3.824},
        "throughput": throughput,
        "visualizations_saved": viz_saved,
    }

    with open(os.path.join(output_dir, "final_integration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 78)
    print("Final integration validation")
    print("=" * 78)
    print(json.dumps(summary, indent=2, default=str))
    print()
    print(f"[Output] {output_dir}")
    print(f"[Sequence kept at] {seq_dir}")


if __name__ == "__main__":
    main()

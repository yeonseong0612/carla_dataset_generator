"""
scripts/tools/analyze_foreground_dominance.py

Diagnostic-only tool (CLAUDE.md PART C). Measures how often a single
camera_valid object dominates the rgb_left frame (large 2D bbox
occupancy) and for how long the same object stays dominant across
consecutive frames -- used to check whether the PART B same-lane front
spawn gap actually improves dataset diversity. Never changes production
behavior, never touches spawn/despawn decisions, never re-runs CARLA --
reads already-generated labels/object_3d/*.json and calibration.json.

Reused, not reimplemented:
  - scripts.tools.debug_annotation_visibility.sequence_dirs /
    load_calibration / load_annotation / frame_available (same dataset
    layout helpers the annotation-visibility diagnostic already uses)

bbox_fraction uses the already-computed, already image-clipped
bbox_width_px / bbox_height_px stored in each object's camera_projection
(src/data/annotation.py AnnotationWriter._compute_camera_validity) --
not re-derived from vertices, since those pixel dimensions are already
exactly what CLAUDE.md section C-1 asks for ("가능하면 image-clipped bbox
를 사용").

The dominance threshold (bbox_fraction >= 0.30) and long-duration
threshold (>= 50 consecutive frames) are diagnostic-only defaults (CLI
arguments here), never read from CFG/config.py and never used to drive
any spawn/despawn decision.
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARLA_PYTHONAPI = r"C:\CARLA\PythonAPI\carla"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CARLA_PYTHONAPI)

from scripts.tools.debug_annotation_visibility import (  # noqa: E402
    sequence_dirs, load_calibration, load_annotation, frame_available,
)

DEFAULT_DOMINANCE_THRESHOLD = 0.30
DEFAULT_LONG_EVENT_FRAMES = 50


def per_frame_dominant_object(annotation, image_width, image_height):
    """
    Largest camera_valid object in this frame by image-clipped 2D bbox
    occupancy (CLAUDE.md C-1), or None if the frame has no camera_valid
    object.
    """

    best = None
    best_fraction = -1.0

    for obj in annotation.get("objects", []):
        cp = obj.get("camera_projection", {})
        bbox_w = cp.get("bbox_width_px")
        bbox_h = cp.get("bbox_height_px")

        if not bbox_w or not bbox_h:
            continue

        bbox_fraction = (bbox_w * bbox_h) / (image_width * image_height)

        if bbox_fraction > best_fraction:
            best_fraction = bbox_fraction
            best = {
                "logical_id": obj.get("logical_id"),
                "category": obj.get("category"),
                "distance_m": obj.get("distance_m"),
                "bbox_fraction": bbox_fraction,
                "visible_fraction": cp.get("visible_fraction"),
            }

    return best


def collect_per_frame_records(geometry_dir, image_width, image_height, frames):
    records = []

    for frame in frames:
        annotation = load_annotation(geometry_dir, frame)
        dominant = per_frame_dominant_object(annotation, image_width, image_height)

        if dominant is None:
            continue

        records.append({"frame": frame, **dominant})

    return records


def group_dominance_events(records, threshold):
    """
    Maximal runs of consecutive frame numbers where the same logical_id
    is the per-frame-dominant object AND bbox_fraction >= threshold
    throughout. A gap in frame numbers, a change of dominant logical_id,
    or a dip below threshold all close the current run.
    """

    events = []
    current = None

    def close(current):
        if current is None:
            return None

        frames = current["frames"]
        events.append({
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "duration_frames": len(frames),
            "logical_id": current["logical_id"],
            "category": current["category"],
            "min_distance_m": min(current["distances"]),
            "max_bbox_fraction": max(current["bbox_fractions"]),
            "mean_bbox_fraction": sum(current["bbox_fractions"]) / len(current["bbox_fractions"]),
            "mean_visible_fraction": (
                sum(current["visible_fractions"]) / len(current["visible_fractions"])
                if current["visible_fractions"] else None
            ),
        })
        return None

    for record in records:
        is_dominant = record["bbox_fraction"] >= threshold

        continues_run = (
            current is not None
            and is_dominant
            and record["logical_id"] == current["logical_id"]
            and record["frame"] == current["frames"][-1] + 1
        )

        if continues_run:
            current["frames"].append(record["frame"])
            current["distances"].append(record["distance_m"])
            current["bbox_fractions"].append(record["bbox_fraction"])
            if record["visible_fraction"] is not None:
                current["visible_fractions"].append(record["visible_fraction"])
            continue

        current = close(current)

        if is_dominant:
            current = {
                "logical_id": record["logical_id"],
                "category": record["category"],
                "frames": [record["frame"]],
                "distances": [record["distance_m"]],
                "bbox_fractions": [record["bbox_fraction"]],
                "visible_fractions": [record["visible_fraction"]] if record["visible_fraction"] is not None else [],
            }

    close(current)

    return events


def run(dataset_root, town, route, condition, out_root, threshold, long_event_frames):
    dirs = sequence_dirs(dataset_root, town, route, condition)
    calibration = load_calibration(dirs["geometry"])
    cam = calibration["cameras"]["rgb_left"]
    image_width, image_height = cam["width"], cam["height"]

    labels_dir = os.path.join(dirs["geometry"], "labels", "object_3d")
    frames = sorted(
        int(os.path.splitext(f)[0]) for f in os.listdir(labels_dir)
        if f.endswith(".json") and frame_available(dirs["geometry"], int(os.path.splitext(f)[0]))
    )

    records = collect_per_frame_records(dirs["geometry"], image_width, image_height, frames)
    events = group_dominance_events(records, threshold)

    out_dir = out_root
    os.makedirs(out_dir, exist_ok=True)

    events_path = os.path.join(out_dir, "events.csv")
    fieldnames = ["start_frame", "end_frame", "duration_frames", "logical_id", "category",
                  "min_distance_m", "max_bbox_fraction", "mean_bbox_fraction", "mean_visible_fraction"]
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({k: event[k] for k in fieldnames})

    total_frames = len(records)
    dominant_frame_count = sum(1 for r in records if r["bbox_fraction"] >= threshold)
    long_events = [e for e in events if e["duration_frames"] >= long_event_frames]

    dominant_logical_id_counts = {}
    for r in records:
        if r["bbox_fraction"] >= threshold:
            dominant_logical_id_counts[r["logical_id"]] = dominant_logical_id_counts.get(r["logical_id"], 0) + 1
    top_dominant_logical_ids = sorted(dominant_logical_id_counts.items(), key=lambda kv: -kv[1])[:10]

    summary = {
        "run": {"town": town, "route": route, "condition": condition,
                "dominance_threshold": threshold, "long_event_min_frames": long_event_frames},
        "total_frames": total_frames,
        "frames_bbox_fraction_gt_0.30": dominant_frame_count,
        "ratio": (dominant_frame_count / total_frames) if total_frames else None,
        "longest_dominance_event_frames": max((e["duration_frames"] for e in events), default=0),
        "num_long_events": len(long_events),
        "top_dominant_logical_ids": [{"logical_id": lid, "dominant_frame_count": c} for lid, c in top_dominant_logical_ids],
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"[Output] {events_path}")
    print(f"[Output] {summary_path}")

    return summary, events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--town", type=str, required=True)
    parser.add_argument("--route", type=str, required=True)
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--out-root", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "foreground_dominance"))
    parser.add_argument("--dominance-threshold", type=float, default=DEFAULT_DOMINANCE_THRESHOLD,
                         help="Diagnostic-only bbox_fraction threshold (default 0.30). Never used in production.")
    parser.add_argument("--long-event-frames", type=int, default=DEFAULT_LONG_EVENT_FRAMES,
                         help="Diagnostic-only consecutive-frame threshold for a 'long' dominance event (default 50).")
    args = parser.parse_args()

    run(args.dataset_root, args.town, args.route, args.condition, args.out_root,
        args.dominance_threshold, args.long_event_frames)


if __name__ == "__main__":
    main()

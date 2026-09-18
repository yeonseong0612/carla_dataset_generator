"""
scripts/tools/validate_camera_valid_annotations.py

CLAUDE.md task: "Finalize Camera-Valid Annotation Filtering" -- live
validation. Runs collect_dataset.py (subprocess, same TRAFFIC_MANAGER.
PORT runtime-patch bootstrap the earlier validation scripts used -- this
machine's port 8000 is bound by an unrelated local app) and analyzes
what it produces: every frame's labels/object_3d/*.json now has an
"objects" list (camera_valid == True only -- the final annotation list)
and a "rejected_objects" list (camera_valid == False, kept for
diagnostics), each object carrying a "camera_projection" dict
(image_fraction, visible_fraction, bbox_width_px, bbox_height_px,
visible_area_px, camera_valid, invalid_reason) written live by
src/data/annotation.py AnnotationWriter._compute_camera_validity.

This script does not re-derive filtering decisions -- it only reads
what was already decided and re-projects vertices_ego_m for
visualization (same math, via src.data.projection.CameraProjector,
exactly like scripts/tools/visualize_annotations.py already does
post-hoc).
"""

import argparse
import glob
import json
import os
import shutil
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
from src.data.projection import CameraProjector  # noqa: E402
from scripts.tools.validate_frame_object_gamma import sequence_root  # noqa: E402
from scripts.tools.validate_frame_object_gamma_longrun import run_collection_with_log  # noqa: E402

REASON_SHORT = {
    "outside_fov": "FOV",
    "projection_invalid": "FOV",
    "too_truncated": "TRUNC",
    "too_occluded": "OCC",
    "too_small": "SMALL",
}

REASON_COLOR_BGR = {
    "FOV": (255, 0, 255),
    "TRUNC": (0, 165, 255),
    "OCC": (0, 0, 255),
    "SMALL": (0, 255, 255),
}

VALID_COLOR_BGR = (0, 220, 0)


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------

def load_calibration(sequence_dir):
    with open(os.path.join(sequence_dir, "calibration.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def discover_frames(sequence_dir):
    paths = sorted(glob.glob(os.path.join(sequence_dir, "labels", "object_3d", "*.json")))
    frames = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


# ------------------------------------------------------------------
# Re-projection for visualization only (same math as _compute_camera_
# validity / visualize_annotations.py -- read-only, decisions already
# made live and stored in camera_projection/invalid_reason).
# ------------------------------------------------------------------

def clipped_bbox_px(projector, vertices_ego_m):
    v_ego = np.asarray(vertices_ego_m, dtype=np.float64)
    v_cv = projector.ego_to_cv(v_ego)
    uv, valid = projector.project(v_cv)

    if not np.any(valid):
        return None

    u_valid, v_valid = uv[valid, 0], uv[valid, 1]
    u_min, u_max = float(u_valid.min()), float(u_valid.max())
    v_min, v_max = float(v_valid.min()), float(v_valid.max())

    px0 = int(max(min(u_min, u_max), 0))
    px1 = int(min(max(u_min, u_max), projector.width))
    py0 = int(max(min(v_min, v_max), 0))
    py1 = int(min(max(v_min, v_max), projector.height))

    if px1 <= px0 or py1 <= py0:
        return None

    return px0, py0, px1, py1


def draw_object(image, projector, obj, is_valid):
    rect = clipped_bbox_px(projector, obj["bbox_3d"]["vertices_ego_m"])

    if rect is None:
        return

    x0, y0, x1, y1 = rect

    if is_valid:
        color = VALID_COLOR_BGR
        label = "OK"
    else:
        reason = obj.get("camera_projection", {}).get("invalid_reason") or "FOV"
        short = REASON_SHORT.get(reason, reason[:5].upper())
        color = REASON_COLOR_BGR.get(short, (128, 128, 128))
        label = short

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 3)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x0 + 2, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_diagnostic_frame(sequence_dir, projector, frame, out_path):
    frame_id = frame["frame_id"]
    image_path = os.path.join(sequence_dir, "rgb_left", f"{frame_id:06d}.png")
    image = cv2.imread(image_path)

    if image is None:
        return False

    for obj in frame.get("objects", []):
        draw_object(image, projector, obj, is_valid=True)

    for obj in frame.get("rejected_objects", []):
        draw_object(image, projector, obj, is_valid=False)

    n_valid = len(frame.get("objects", []))
    n_rejected = len(frame.get("rejected_objects", []))
    cv2.putText(image, f"frame={frame_id}  valid={n_valid}  rejected={n_rejected}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, f"frame={frame_id}  valid={n_valid}  rejected={n_rejected}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, image)
    return True


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def per_frame_stats(frame):
    objects = frame.get("objects", [])
    rejected = frame.get("rejected_objects", [])
    all_candidates = objects + rejected

    fov_valid = sum(1 for o in all_candidates if (o["camera_projection"].get("image_fraction") or 0) >= cfg.ANNOTATION.MIN_IMAGE_FRACTION)
    occlusion_valid = sum(1 for o in all_candidates if (o["camera_projection"].get("visible_fraction") or 0) >= cfg.ANNOTATION.MIN_VISIBLE_FRACTION)
    size_valid = sum(
        1 for o in all_candidates
        if (o["camera_projection"].get("bbox_width_px") or 0) >= cfg.ANNOTATION.MIN_BBOX_WIDTH_PX
        and (o["camera_projection"].get("bbox_height_px") or 0) >= cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX
        and (o["camera_projection"].get("visible_area_px") or 0) >= cfg.ANNOTATION.MIN_VISIBLE_AREA_PX
    )

    return {
        "frame_id": frame["frame_id"],
        "raw_candidate_count": len(all_candidates),
        "fov_valid_count": fov_valid,
        "occlusion_valid_count": occlusion_valid,
        "size_valid_count": size_valid,
        "final_camera_valid_count": len(objects),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", type=str, default="Town10")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--collection-output-root", type=str, default=os.path.join(cfg.PROJECT.ROOT, "dataset_camera_valid_annotation"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(cfg.PROJECT.ROOT, "outputs", "camera_valid_annotation_validation"))
    parser.add_argument("--keep-raw-dataset", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collection_stdout.log")

    if not args.skip_collection:
        run_collection_with_log(
            args.town, args.route_id, args.condition, args.max_frames,
            args.collection_output_root, log_path, overwrite=True,
        )

    seq_dir = sequence_root(args.collection_output_root, args.town, args.route_id, args.condition)

    calibration = load_calibration(seq_dir)
    projector = CameraProjector.from_calibration(calibration, "rgb_left")

    frames = discover_frames(seq_dir)
    n_frames = len(frames)

    per_frame = [per_frame_stats(f) for f in frames]

    with open(os.path.join(output_dir, "camera_valid_per_frame.json"), "w", encoding="utf-8") as f:
        json.dump(per_frame, f, indent=2)

    total_candidates = sum(r["raw_candidate_count"] for r in per_frame)
    total_final_valid = sum(r["final_camera_valid_count"] for r in per_frame)

    reason_counts = {"outside_fov": 0, "projection_invalid": 0, "too_truncated": 0, "too_occluded": 0, "too_small": 0}
    for f in frames:
        for obj in f.get("rejected_objects", []):
            reason = obj.get("camera_projection", {}).get("invalid_reason")
            if reason in reason_counts:
                reason_counts[reason] += 1

    total_rejected = sum(reason_counts.values())

    # ---- before/after object count (this frame's "raw" = pre-filter
    # distance-only candidates, "after" = final camera-valid) ----
    before_after = {
        "total_raw_candidates": total_candidates,
        "total_final_valid": total_final_valid,
        "total_rejected": total_rejected,
        "retained_ratio": total_final_valid / total_candidates if total_candidates else None,
    }

    # ---- edge cases (section 13) ----
    edge_case_examples = {}
    for f in frames:
        for obj in f.get("rejected_objects", []):
            reason = obj.get("camera_projection", {}).get("invalid_reason")
            key = {"outside_fov": "outside_screen", "too_truncated": "edge_partial", "too_occluded": "heavily_occluded", "too_small": "small_far"}.get(reason)
            if key and key not in edge_case_examples:
                edge_case_examples[key] = {"frame_id": f["frame_id"], "actor_id": obj["actor_id"], "reason": reason}
        for obj in f.get("objects", []):
            if "clearly_visible" not in edge_case_examples:
                cp = obj["camera_projection"]
                if (cp.get("image_fraction") or 0) > 0.9 and (cp.get("visible_fraction") or 0) > 0.9:
                    edge_case_examples["clearly_visible"] = {"frame_id": f["frame_id"], "actor_id": obj["actor_id"]}

    # ---- representative visualizations (>= 6 frames) ----
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    candidate_frame_ids = set()
    for example in edge_case_examples.values():
        candidate_frame_ids.add(example["frame_id"])

    if len(candidate_frame_ids) < 6 and n_frames > 0:
        step = max(n_frames // 6, 1)
        for i in range(6):
            idx = min(i * step, n_frames - 1)
            candidate_frame_ids.add(frames[idx]["frame_id"])

    frames_by_id = {f["frame_id"]: f for f in frames}
    viz_saved = []

    for frame_id in sorted(candidate_frame_ids):
        frame = frames_by_id.get(frame_id)
        if frame is None:
            continue
        out_path = os.path.join(viz_dir, f"{frame_id:06d}.png")
        if render_diagnostic_frame(seq_dir, projector, frame, out_path):
            viz_saved.append(out_path)

    # ---- summary ----
    summary = {
        "run": {"town": args.town, "route_id": args.route_id, "condition": args.condition, "n_frames_analyzed": n_frames},
        "before_after": before_after,
        "rejection_reason_counts": reason_counts,
        "rejection_reason_ratios": {k: (v / total_candidates if total_candidates else None) for k, v in reason_counts.items()},
        "mean_raw_candidates_per_frame": float(np.mean([r["raw_candidate_count"] for r in per_frame])) if per_frame else None,
        "mean_fov_valid_per_frame": float(np.mean([r["fov_valid_count"] for r in per_frame])) if per_frame else None,
        "mean_occlusion_valid_per_frame": float(np.mean([r["occlusion_valid_count"] for r in per_frame])) if per_frame else None,
        "mean_size_valid_per_frame": float(np.mean([r["size_valid_count"] for r in per_frame])) if per_frame else None,
        "mean_final_valid_per_frame": float(np.mean([r["final_camera_valid_count"] for r in per_frame])) if per_frame else None,
        "edge_case_examples": edge_case_examples,
        "visualizations_saved": viz_saved,
        "thresholds": {
            "MIN_IMAGE_FRACTION": cfg.ANNOTATION.MIN_IMAGE_FRACTION,
            "MIN_VISIBLE_FRACTION": cfg.ANNOTATION.MIN_VISIBLE_FRACTION,
            "MIN_BBOX_WIDTH_PX": cfg.ANNOTATION.MIN_BBOX_WIDTH_PX,
            "MIN_BBOX_HEIGHT_PX": cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX,
            "MIN_VISIBLE_AREA_PX": cfg.ANNOTATION.MIN_VISIBLE_AREA_PX,
        },
    }

    with open(os.path.join(output_dir, "camera_valid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 78)
    print("Camera-valid annotation filtering -- live validation")
    print("=" * 78)
    print(json.dumps(summary, indent=2, default=str))
    print()
    print(f"[Output] {output_dir}")

    if not args.keep_raw_dataset:
        print(f"[Cleanup] removing raw sensor data at {seq_dir}")
        shutil.rmtree(seq_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

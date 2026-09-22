"""
scripts/tools/debug_annotation_visibility.py

Diagnostic-only tool for the annotation camera_valid / occlusion audit
(CLAUDE.md: "CARLA dataset generator annotation validity / occlusion
final verification"). Reads an ALREADY-COLLECTED dataset sequence and
independently re-derives, at pixel level, every intermediate quantity
behind image_fraction / visible_fraction / camera_valid, then
cross-checks them against what
src.data.annotation.AnnotationWriter._compute_camera_validity already
decided and stored in labels/object_3d/*.json.

This script:
  - never re-runs CARLA
  - never writes into the dataset directory it reads from
  - never imports/modifies CFG thresholds, annotation.py, projection.py,
    calibration.py, sensor transforms, replay/production architecture

Reused (not reimplemented):
  - src.data.projection.CameraProjector           (ego->cv, project)
  - src.data.collector.depth_to_numpy              (decode formula, for
                                                      the synthetic decoder
                                                      self-test only --
                                                      stored geometry/depth
                                                      is already decoded
                                                      meters, see below)
  - src.data.annotation.SEMANTIC_MAP               (semantic palette)
  - scripts.tools.bbox_geometry.derive_edges_from_vertices (3D cuboid edges)
  - CFG.config.cfg.ANNOTATION.*                    (thresholds, read-only)

`recompute_visibility()` below re-implements the pixel-level occlusion
test from AnnotationWriter._compute_camera_validity (src/data/
annotation.py) verbatim -- same formula, same constants -- because that
production function does not expose intermediate values (per-pixel
masks, depth patch stats) and CLAUDE.md section 5 requires exactly those
to be inspected. This is instrumentation of the existing algorithm, not
a second/divergent implementation: --verify diffs its output against the
stored camera_projection metrics for every object/rejected_object in the
frame, and any nonzero diff is itself reported as diagnostic evidence of
drift between this script and annotation.py (not silently trusted).

Dataset layout expected (src/data/layout.py, unchanged):
    <dataset-root>/<town>/route_<route>/geometry/
        calibration.json
        labels/object_3d/{frame:06d}.json
        depth/{frame:06d}.npy        (float32 meters, decoded)
        semantic/{frame:06d}.npy     (uint8 CARLA semantic tag)
    <dataset-root>/<town>/route_<route>/conditions/<condition>/
        rgb_left/{frame:06d}.png
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARLA_PYTHONAPI = r"C:\CARLA\PythonAPI\carla"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CARLA_PYTHONAPI)

from CFG.config import cfg  # noqa: E402
from src.data.layout import resolve_geometry_root, resolve_condition_root  # noqa: E402
from src.data.projection import CameraProjector  # noqa: E402
from src.data.annotation import SEMANTIC_MAP  # noqa: E402
from src.data.collector import depth_to_numpy  # noqa: E402
from scripts.tools.bbox_geometry import derive_edges_from_vertices, BoxGeometryError  # noqa: E402


DEPTH_TOLERANCE_M = 0.5  # AnnotationWriter._compute_camera_validity, kept identical on purpose


# ============================================================
# IO helpers
# ============================================================

def sequence_dirs(dataset_root, town, route, condition):
    route_dir = os.path.join(dataset_root, town, f"route_{route}")
    condition_dir = os.path.join(route_dir, "conditions", condition)
    geometry_dir = resolve_geometry_root(condition_dir)
    return {
        "route": route_dir,
        "condition": condition_dir,
        "geometry": geometry_dir,
    }


def load_calibration(geometry_dir):
    with open(os.path.join(geometry_dir, "calibration.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation(geometry_dir, frame_id):
    path = os.path.join(geometry_dir, "labels", "object_3d", f"{frame_id:06d}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_depth(geometry_dir, frame_id):
    return np.load(os.path.join(geometry_dir, "depth", f"{frame_id:06d}.npy")).astype(np.float64)


def load_semantic(geometry_dir, frame_id):
    return np.load(os.path.join(geometry_dir, "semantic", f"{frame_id:06d}.npy"))


def load_rgb(condition_dir, frame_id):
    path = os.path.join(condition_dir, "rgb_left", f"{frame_id:06d}.png")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB not found: {path}")
    return image


def frame_available(geometry_dir, frame_id):
    return os.path.isfile(os.path.join(geometry_dir, "labels", "object_3d", f"{frame_id:06d}.json"))


# ============================================================
# Pixel-level visibility recomputation (mirrors
# AnnotationWriter._compute_camera_validity, instrumented)
# ============================================================

def recompute_visibility(projector, vertices_ego_m, depth_m):
    """
    Returns a dict with every intermediate value CLAUDE.md section 5/10
    asks for, plus a full-image boolean visible_mask (True only inside
    the clipped bbox, where the pixel passed the depth test).
    """

    out = {
        "projection_valid": False,
        "invalid_reason": "projection_invalid",
        "u_min": None, "u_max": None, "v_min": None, "v_max": None,
        "unclipped_area_px": None,
        "clip_u0": None, "clip_u1": None, "clip_v0": None, "clip_v1": None,
        "intersection_area_px": None,
        "image_fraction": 0.0,
        "px0": None, "py0": None, "px1": None, "py1": None,
        "bbox_width_px": 0, "bbox_height_px": 0,
        "near_depth_m": None, "far_depth_m": None,
        "depth_patch_min_m": None, "depth_patch_median_m": None, "depth_patch_max_m": None,
        "depth_patch_finite_count": 0,
        "projected_pixel_count": 0,
        "visible_pixel_count": 0,
        "occluded_pixel_count": 0,
        "visible_fraction": 0.0,
        "camera_valid": False,
        "visible_mask_full": np.zeros(depth_m.shape, dtype=bool),
    }

    v_ego = np.asarray(vertices_ego_m, dtype=np.float64)
    if v_ego.shape != (8, 3) or not np.all(np.isfinite(v_ego)):
        return out

    v_cv = projector.ego_to_cv(v_ego)
    uv, valid = projector.project(v_cv)

    if not np.any(valid):
        out["invalid_reason"] = "outside_fov"
        return out

    out["projection_valid"] = True

    u_valid, v_valid = uv[valid, 0], uv[valid, 1]
    u_min, u_max = float(u_valid.min()), float(u_valid.max())
    v_min, v_max = float(v_valid.min()), float(v_valid.max())
    out.update(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)

    unclipped_area = max(u_max - u_min, 0.0) * max(v_max - v_min, 0.0)
    out["unclipped_area_px"] = unclipped_area

    width, height = projector.width, projector.height
    clip_u0, clip_u1 = min(max(u_min, 0.0), width), min(max(u_max, 0.0), width)
    clip_v0, clip_v1 = min(max(v_min, 0.0), height), min(max(v_max, 0.0), height)
    out.update(clip_u0=clip_u0, clip_u1=clip_u1, clip_v0=clip_v0, clip_v1=clip_v1)

    intersect_w, intersect_h = max(clip_u1 - clip_u0, 0.0), max(clip_v1 - clip_v0, 0.0)
    intersection_area = intersect_w * intersect_h
    out["intersection_area_px"] = intersection_area

    if unclipped_area <= 0.0 or intersection_area <= 0.0:
        out["invalid_reason"] = "outside_fov"
        return out

    image_fraction = intersection_area / unclipped_area
    out["image_fraction"] = float(image_fraction)

    px0, py0 = int(math.floor(clip_u0)), int(math.floor(clip_v0))
    px1, py1 = int(math.ceil(clip_u1)), int(math.ceil(clip_v1))
    px0, py0 = max(px0, 0), max(py0, 0)
    px1, py1 = min(px1, width), min(py1, height)
    out.update(px0=px0, py0=py0, px1=px1, py1=py1)

    bbox_width_px, bbox_height_px = max(px1 - px0, 0), max(py1 - py0, 0)
    out.update(bbox_width_px=int(bbox_width_px), bbox_height_px=int(bbox_height_px))

    if bbox_width_px <= 0 or bbox_height_px <= 0:
        out["invalid_reason"] = "too_truncated"
        return out

    near_depth = float(v_cv[valid, 2].min())
    far_depth = float(v_cv[valid, 2].max())
    out.update(near_depth_m=near_depth, far_depth_m=far_depth)

    depth_patch = depth_m[py0:py1, px0:px1]
    finite = np.isfinite(depth_patch)
    out["depth_patch_finite_count"] = int(finite.sum())
    if finite.any():
        out["depth_patch_min_m"] = float(np.min(depth_patch[finite]))
        out["depth_patch_median_m"] = float(np.median(depth_patch[finite]))
        out["depth_patch_max_m"] = float(np.max(depth_patch[finite]))

    visible_mask = (
        (depth_patch >= (near_depth - DEPTH_TOLERANCE_M))
        & (depth_patch <= (far_depth + DEPTH_TOLERANCE_M))
    )

    projected_pixel_count = depth_patch.size
    visible_pixel_count = int(visible_mask.sum())
    visible_fraction = visible_pixel_count / projected_pixel_count if projected_pixel_count > 0 else 0.0

    out["projected_pixel_count"] = int(projected_pixel_count)
    out["visible_pixel_count"] = visible_pixel_count
    out["occluded_pixel_count"] = int(projected_pixel_count - visible_pixel_count)
    out["visible_fraction"] = float(visible_fraction)

    full_mask = np.zeros(depth_m.shape, dtype=bool)
    full_mask[py0:py1, px0:px1] = visible_mask
    out["visible_mask_full"] = full_mask

    # object-consistent depth pixel count (CLAUDE.md section 5): pixels
    # whose rendered depth also falls *within* [near_depth, far_depth] +-
    # tolerance, i.e. plausibly the object's own surface rather than
    # arbitrary background beyond it. NOT used by camera_valid (that's
    # the production algorithm, section 16: do not change it) -- this is
    # an additional diagnostic-only statistic asked for by section 10/11.
    within_object_depth_mask = (
        (depth_patch >= (near_depth - DEPTH_TOLERANCE_M))
        & (depth_patch <= (far_depth + DEPTH_TOLERANCE_M))
    )
    out["object_consistent_pixel_count"] = int(within_object_depth_mask.sum())
    out["object_consistent_fraction"] = (
        float(within_object_depth_mask.sum() / projected_pixel_count) if projected_pixel_count > 0 else 0.0
    )
    full_consistent_mask = np.zeros(depth_m.shape, dtype=bool)
    full_consistent_mask[py0:py1, px0:px1] = within_object_depth_mask
    out["object_consistent_mask_full"] = full_consistent_mask

    camera_valid = (
        image_fraction >= cfg.ANNOTATION.MIN_IMAGE_FRACTION
        and visible_fraction >= cfg.ANNOTATION.MIN_VISIBLE_FRACTION
        and bbox_width_px >= cfg.ANNOTATION.MIN_BBOX_WIDTH_PX
        and bbox_height_px >= cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX
        and visible_pixel_count >= cfg.ANNOTATION.MIN_VISIBLE_AREA_PX
    )
    out["camera_valid"] = camera_valid

    if camera_valid:
        out["invalid_reason"] = None
    elif image_fraction < cfg.ANNOTATION.MIN_IMAGE_FRACTION:
        out["invalid_reason"] = "too_truncated"
    elif visible_fraction < cfg.ANNOTATION.MIN_VISIBLE_FRACTION:
        out["invalid_reason"] = "too_occluded"
    else:
        out["invalid_reason"] = "too_small"

    return out


def verify_against_stored(obj, recomputed, atol=1e-6):
    cp = obj.get("camera_projection", {})
    fields = ["image_fraction", "visible_fraction", "bbox_width_px", "bbox_height_px", "camera_valid"]
    field_map = {"bbox_width_px": "bbox_width_px", "bbox_height_px": "bbox_height_px"}
    diffs = {}
    for field in fields:
        stored = cp.get(field)
        recomp = recomputed.get(field)
        if stored is None or recomp is None:
            continue
        if isinstance(stored, bool) or isinstance(recomp, bool):
            if bool(stored) != bool(recomp):
                diffs[field] = {"stored": stored, "recomputed": recomp}
        elif abs(float(stored) - float(recomp)) > atol:
            diffs[field] = {"stored": stored, "recomputed": recomp}
    return diffs


# ============================================================
# Drawing
# ============================================================

def clipped_rect(rv):
    if rv["px1"] is None or rv["bbox_width_px"] <= 0 or rv["bbox_height_px"] <= 0:
        return None
    return rv["px0"], rv["py0"], rv["px1"], rv["py1"]


def draw_2d_bbox(image, rect, color, label_lines):
    x0, y0, x1, y1 = rect
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 3)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

    ty = y0 - 6
    for line in reversed(label_lines):
        ty_draw = max(ty, 12)
        cv2.putText(image, line, (x0 + 2, ty_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x0 + 2, ty_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        ty -= 14


def draw_3d_cuboid(image, projector, vertices_ego_m, color):
    v_ego = np.asarray(vertices_ego_m, dtype=np.float64)
    v_cv = projector.ego_to_cv(v_ego)
    uv, valid = projector.project(v_cv)

    try:
        edges, _extents = derive_edges_from_vertices(v_ego)
    except BoxGeometryError:
        return

    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        cv2.line(image, p1, p2, color, 1, cv2.LINE_AA)


def label_lines_for(obj, rv):
    cp = obj.get("camera_projection", {})
    return [
        f"L{obj.get('logical_id', '?')} a{obj.get('actor_id')} {obj.get('category')}/{obj.get('subcategory')}",
        f"d={obj.get('distance_m', 0):.1f}m vis={cp.get('visible_fraction')!r} img={cp.get('image_fraction')!r}",
        f"valid={cp.get('camera_valid')} reason={cp.get('invalid_reason')}",
    ]


def render_overlay(rgb_image, projector, objects, bbox_mode, highlight_ids, out_path, valid_color, title):
    image = rgb_image.copy()

    for obj in objects:
        vertices = obj["bbox_3d"]["vertices_ego_m"]
        rv = recompute_visibility(projector, vertices, np.zeros((projector.height, projector.width)))
        rect = clipped_rect(rv)
        if rect is None:
            continue

        logical_id = obj.get("logical_id")
        color = (0, 255, 255) if logical_id in highlight_ids else valid_color

        if bbox_mode in ("3d", "both"):
            draw_3d_cuboid(image, projector, vertices, color)
        if bbox_mode in ("2d", "both"):
            draw_2d_bbox(image, rect, color, label_lines_for(obj, rv))

    cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, image)


def colorize_depth(depth_m, max_depth=100.0):
    clipped = np.clip(depth_m, 0, max_depth)
    normalized = (clipped / max_depth * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    return colored


def colorize_semantic(semantic_arr):
    h, w = semantic_arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for tag, (_name, rgb) in SEMANTIC_MAP.items():
        mask = semantic_arr == tag
        if mask.any():
            out[mask] = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR
    return out


def semantic_histogram(semantic_arr, rect):
    x0, y0, x1, y1 = rect
    patch = semantic_arr[y0:y1, x0:x1]
    total = patch.size
    hist = {}
    for tag, (name, _rgb) in SEMANTIC_MAP.items():
        count = int(np.sum(patch == tag))
        if count > 0:
            hist[name] = {"pixels": count, "fraction": count / total if total else 0.0}
    return hist


def save_mask_png(mask, out_path, rgb_image=None):
    h, w = mask.shape
    if rgb_image is not None:
        canvas = (rgb_image.astype(np.float32) * 0.35).astype(np.uint8)
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask] = (0, 255, 0)
    cv2.imwrite(out_path, canvas)


# ============================================================
# Pairwise occlusion analysis (CLAUDE.md section 8)
# ============================================================

def bbox_iou(r1, r2):
    x0 = max(r1[0], r2[0])
    y0 = max(r1[1], r2[1])
    x1 = min(r1[2], r2[2])
    y1 = min(r1[3], r2[3])
    inter_w = max(x1 - x0, 0)
    inter_h = max(y1 - y0, 0)
    inter = inter_w * inter_h
    a1 = max(r1[2] - r1[0], 0) * max(r1[3] - r1[1], 0)
    a2 = max(r2[2] - r2[0], 0) * max(r2[3] - r2[1], 0)
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    return iou, inter, a1, a2


def pairwise_analysis(objects_with_rv, near_obj_logical_id=None):
    """
    objects_with_rv: list of (obj, rv, rect) for every candidate
    (objects + rejected_objects) in a frame, rect already clipped-2d.
    Returns list of pair records for every (near, far) pair where
    near.distance_m < far.distance_m and both have a valid rect.
    """
    pairs = []
    for i, (near_obj, near_rv, near_rect) in enumerate(objects_with_rv):
        if near_rect is None:
            continue
        if near_obj_logical_id is not None and near_obj.get("logical_id") != near_obj_logical_id:
            continue
        for j, (far_obj, far_rv, far_rect) in enumerate(objects_with_rv):
            if i == j or far_rect is None:
                continue
            if near_obj["distance_m"] >= far_obj["distance_m"]:
                continue
            iou, inter, near_area, far_area = bbox_iou(near_rect, far_rect)
            overlap_ratio_far = inter / far_area if far_area > 0 else 0.0
            pairs.append({
                "near_logical_id": near_obj.get("logical_id"),
                "far_logical_id": far_obj.get("logical_id"),
                "near_distance_m": near_obj["distance_m"],
                "far_distance_m": far_obj["distance_m"],
                "bbox_iou": iou,
                "intersection_area_px": inter,
                "near_bbox_area_px": near_area,
                "far_bbox_area_px": far_area,
                "far_overlap_ratio": overlap_ratio_far,
                "near_near_depth_m": near_rv["near_depth_m"],
                "far_near_depth_m": far_rv["near_depth_m"],
                "far_visible_fraction": far_rv["visible_fraction"],
                "far_camera_valid": far_rv["camera_valid"],
            })
    return pairs


# ============================================================
# Single-frame diagnostic bundle
# ============================================================

def build_objects_with_rv(annotation, projector, depth_m):
    all_candidates = annotation.get("objects", []) + annotation.get("rejected_objects", [])
    bundle = []
    for obj in all_candidates:
        rv = recompute_visibility(projector, obj["bbox_3d"]["vertices_ego_m"], depth_m)
        rect = clipped_rect(rv)
        bundle.append((obj, rv, rect))
    return bundle


def run_single_frame(dataset_root, town, route, frame, condition, out_root, bbox_mode, highlight_ids, verify):
    dirs = sequence_dirs(dataset_root, town, route, condition)
    calibration = load_calibration(dirs["geometry"])
    projector = CameraProjector.from_calibration(calibration, "rgb_left")

    annotation = load_annotation(dirs["geometry"], frame)
    depth_m = load_depth(dirs["geometry"], frame)
    semantic_arr = load_semantic(dirs["geometry"], frame)
    rgb = load_rgb(dirs["condition"], frame)

    out_dir = os.path.join(out_root, town, f"route_{route}", f"frame_{frame:06d}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. rgb_original.png
    cv2.imwrite(os.path.join(out_dir, "rgb_original.png"), rgb)

    objects = annotation.get("objects", [])
    rejected = annotation.get("rejected_objects", [])

    # 2/3/4. overlays
    render_overlay(rgb, projector, objects, bbox_mode, highlight_ids,
                    os.path.join(out_dir, "accepted_overlay.png"), (0, 220, 0),
                    f"frame={frame} ACCEPTED (camera_valid=true) n={len(objects)}")
    render_overlay(rgb, projector, rejected, bbox_mode, highlight_ids,
                    os.path.join(out_dir, "rejected_overlay.png"), (0, 0, 255),
                    f"frame={frame} REJECTED n={len(rejected)}")

    image_all = rgb.copy()
    for obj in objects:
        vertices = obj["bbox_3d"]["vertices_ego_m"]
        rv = recompute_visibility(projector, vertices, depth_m)
        rect = clipped_rect(rv)
        if rect is None:
            continue
        color = (0, 255, 255) if obj.get("logical_id") in highlight_ids else (0, 220, 0)
        if bbox_mode in ("3d", "both"):
            draw_3d_cuboid(image_all, projector, vertices, color)
        if bbox_mode in ("2d", "both"):
            draw_2d_bbox(image_all, rect, color, label_lines_for(obj, rv))
    for obj in rejected:
        vertices = obj["bbox_3d"]["vertices_ego_m"]
        rv = recompute_visibility(projector, vertices, depth_m)
        rect = clipped_rect(rv)
        if rect is None:
            continue
        color = (0, 255, 255) if obj.get("logical_id") in highlight_ids else (0, 0, 255)
        if bbox_mode in ("3d", "both"):
            draw_3d_cuboid(image_all, projector, vertices, color)
        if bbox_mode in ("2d", "both"):
            draw_2d_bbox(image_all, rect, color, label_lines_for(obj, rv))
    cv2.putText(image_all, f"frame={frame} ALL valid={len(objects)} rejected={len(rejected)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_all, f"frame={frame} ALL valid={len(objects)} rejected={len(rejected)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "all_overlay.png"), image_all)

    # 5. depth_visualization.png
    cv2.imwrite(os.path.join(out_dir, "depth_visualization.png"), colorize_depth(depth_m))

    # 6. semantic_visualization.png
    cv2.imwrite(os.path.join(out_dir, "semantic_visualization.png"), colorize_semantic(semantic_arr))

    # per-object recompute + masks + verify
    bundle = build_objects_with_rv(annotation, projector, depth_m)

    per_object = []
    for obj, rv, rect in bundle:
        entry = {
            "logical_id": obj.get("logical_id"),
            "actor_id": obj.get("actor_id"),
            "category": obj.get("category"),
            "subcategory": obj.get("subcategory"),
            "distance_m": obj.get("distance_m"),
            "stored_camera_projection": obj.get("camera_projection"),
            "recomputed": {k: v for k, v in rv.items() if not isinstance(v, np.ndarray)},
        }
        if rect is not None:
            entry["semantic_histogram"] = semantic_histogram(semantic_arr, rect)
        if verify:
            entry["verify_diff"] = verify_against_stored(obj, rv)
        per_object.append(entry)

        if rect is not None and obj.get("logical_id") is not None:
            mask_path = os.path.join(out_dir, f"logical_{int(obj['logical_id']):03d}_visibility_mask.png")
            save_mask_png(rv["visible_mask_full"], mask_path, rgb_image=rgb)

    # pairwise occlusion analysis: every near/far pair among ALL candidates
    pairs = pairwise_analysis(bundle)

    # RGB/depth transform diff (section 7)
    transform_diff = rgb_depth_transform_diff(calibration)

    debug_payload = {
        "frame_id": frame,
        "carla_frame": annotation.get("carla_frame"),
        "timestamp": annotation.get("timestamp"),
        "town": town,
        "route": route,
        "condition": condition,
        "thresholds": {
            "MIN_IMAGE_FRACTION": cfg.ANNOTATION.MIN_IMAGE_FRACTION,
            "MIN_VISIBLE_FRACTION": cfg.ANNOTATION.MIN_VISIBLE_FRACTION,
            "MIN_BBOX_WIDTH_PX": cfg.ANNOTATION.MIN_BBOX_WIDTH_PX,
            "MIN_BBOX_HEIGHT_PX": cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX,
            "MIN_VISIBLE_AREA_PX": cfg.ANNOTATION.MIN_VISIBLE_AREA_PX,
            "DEPTH_TOLERANCE_M": DEPTH_TOLERANCE_M,
        },
        "objects": per_object,
        "pairwise_overlap": pairs,
        "rgb_depth_transform_diff": transform_diff,
    }

    # 7. visibility_debug.json
    with open(os.path.join(out_dir, "visibility_debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2, default=str)

    # 8. visibility_report.txt
    report_lines = build_text_report(debug_payload, annotation)
    with open(os.path.join(out_dir, "visibility_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] frame {frame} diagnostics written to {out_dir}")
    return debug_payload


def rgb_depth_transform_diff(calibration):
    cam = calibration["cameras"]["rgb_left"]
    depth_sensor = calibration["sensors"]["depth"]
    T_rgb = np.array(cam["T_ego_from_camera"])
    T_depth = np.array(depth_sensor["T_ego_from_sensor"])
    pos_diff = np.linalg.norm(T_rgb[:3, 3] - T_depth[:3, 3])
    rot_diff = np.linalg.norm(T_rgb[:3, :3] - T_depth[:3, :3])
    return {
        "position_diff_m": float(pos_diff),
        "rotation_matrix_frobenius_diff": float(rot_diff),
        "rgb_width": cam["width"], "rgb_height": cam["height"], "rgb_fov_deg": cam["fov_deg"],
    }


def build_text_report(payload, annotation):
    lines = []
    lines.append(f"frame_id={payload['frame_id']} carla_frame={payload['carla_frame']} "
                  f"town={payload['town']} route={payload['route']} condition={payload['condition']}")
    lines.append(f"objects(camera_valid=true)={len(annotation.get('objects', []))} "
                 f"rejected_objects={len(annotation.get('rejected_objects', []))}")
    lines.append("")
    lines.append("thresholds: " + json.dumps(payload["thresholds"]))
    lines.append("")
    lines.append("RGB/depth extrinsic diff: position_diff_m=%.6f rotation_frobenius_diff=%.6f" % (
        payload["rgb_depth_transform_diff"]["position_diff_m"],
        payload["rgb_depth_transform_diff"]["rotation_matrix_frobenius_diff"]))
    lines.append("")
    lines.append("=== per-object ===")
    for entry in sorted(payload["objects"], key=lambda e: e["distance_m"]):
        r = entry["recomputed"]
        lines.append(
            f"L{entry['logical_id']} a{entry['actor_id']} {entry['category']}/{entry['subcategory']} "
            f"dist={entry['distance_m']:.2f}m bbox={r['bbox_width_px']}x{r['bbox_height_px']}px "
            f"image_fraction={r['image_fraction']:.4f} visible_fraction={r['visible_fraction']:.4f} "
            f"object_consistent_fraction={r.get('object_consistent_fraction')} "
            f"near_depth={r['near_depth_m']} far_depth={r['far_depth_m']} "
            f"depth_patch[min/med/max]=[{r['depth_patch_min_m']},{r['depth_patch_median_m']},{r['depth_patch_max_m']}] "
            f"camera_valid={r['camera_valid']} reason={r['invalid_reason']}"
        )
        if entry.get("verify_diff"):
            lines.append(f"    !! MISMATCH vs stored: {entry['verify_diff']}")
    lines.append("")
    lines.append("=== pairwise overlap (near.distance < far.distance) ===")
    suspicious = [p for p in payload["pairwise_overlap"]
                  if p["far_overlap_ratio"] > 0.5 and p["far_visible_fraction"] > 0.3 and p["far_camera_valid"]]
    for p in sorted(payload["pairwise_overlap"], key=lambda p: -p["far_overlap_ratio"])[:30]:
        flag = "  <-- SUSPICIOUS" if p in suspicious else ""
        lines.append(
            f"near=L{p['near_logical_id']}({p['near_distance_m']:.1f}m) far=L{p['far_logical_id']}({p['far_distance_m']:.1f}m) "
            f"iou={p['bbox_iou']:.3f} far_overlap_ratio={p['far_overlap_ratio']:.3f} "
            f"far_visible_fraction={p['far_visible_fraction']:.3f} far_camera_valid={p['far_camera_valid']}{flag}"
        )
    return lines


# ============================================================
# Temporal analysis (CLAUDE.md section 12)
# ============================================================

def run_temporal_analysis(dataset_root, town, route, condition, frame_start, frame_end, out_root):
    dirs = sequence_dirs(dataset_root, town, route, condition)
    calibration = load_calibration(dirs["geometry"])
    projector = CameraProjector.from_calibration(calibration, "rgb_left")

    rows = []
    for frame in range(frame_start, frame_end + 1):
        if not frame_available(dirs["geometry"], frame):
            continue
        annotation = load_annotation(dirs["geometry"], frame)
        depth_m = load_depth(dirs["geometry"], frame)
        for obj in annotation.get("objects", []) + annotation.get("rejected_objects", []):
            rv = recompute_visibility(projector, obj["bbox_3d"]["vertices_ego_m"], depth_m)
            rows.append({
                "frame": frame,
                "logical_id": obj.get("logical_id"),
                "distance_m": obj["distance_m"],
                "bbox_w_px": rv["bbox_width_px"],
                "bbox_h_px": rv["bbox_height_px"],
                "visible_fraction": rv["visible_fraction"],
                "camera_valid": rv["camera_valid"],
            })

    out_dir = os.path.join(out_root, town, f"route_{route}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"temporal_{frame_start:06d}_{frame_end:06d}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "logical_id", "distance_m", "bbox_w_px", "bbox_h_px",
                                                "visible_fraction", "camera_valid"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # jump detection: same logical_id, consecutive frames, |delta visible_fraction| > 0.3
    by_lid = {}
    for row in rows:
        by_lid.setdefault(row["logical_id"], []).append(row)
    jumps = []
    for lid, series in by_lid.items():
        series.sort(key=lambda r: r["frame"])
        for a, b in zip(series, series[1:]):
            if b["frame"] - a["frame"] == 1 and abs(b["visible_fraction"] - a["visible_fraction"]) > 0.3:
                jumps.append({"logical_id": lid, "frame_a": a["frame"], "frame_b": b["frame"],
                              "vf_a": a["visible_fraction"], "vf_b": b["visible_fraction"],
                              "dist_a": a["distance_m"], "dist_b": b["distance_m"]})

    jumps_path = os.path.join(out_dir, f"temporal_{frame_start:06d}_{frame_end:06d}_jumps.json")
    with open(jumps_path, "w", encoding="utf-8") as f:
        json.dump(jumps, f, indent=2)

    print(f"[OK] temporal analysis: {csv_path}")
    print(f"[OK] jump candidates ({len(jumps)}): {jumps_path}")
    return rows, jumps


# ============================================================
# Suspicious-case + control-case search (CLAUDE.md sections 13/14)
# ============================================================

def run_suspicious_search(dataset_root, town, route, condition, out_root, top_n=20,
                           overlap_thresh=0.5, vf_thresh=0.3):
    dirs = sequence_dirs(dataset_root, town, route, condition)
    calibration = load_calibration(dirs["geometry"])
    projector = CameraProjector.from_calibration(calibration, "rgb_left")

    frames = sorted(int(os.path.splitext(fn)[0])
                     for fn in os.listdir(os.path.join(dirs["geometry"], "labels", "object_3d"))
                     if fn.endswith(".json"))

    suspicious = []
    controls = {"fully_occluded_far_invalid": [], "partial_far_valid": [],
                "no_overlap_far_valid": [], "boundary_truncated": []}

    for frame in frames:
        annotation = load_annotation(dirs["geometry"], frame)
        depth_m = load_depth(dirs["geometry"], frame)
        bundle = build_objects_with_rv(annotation, projector, depth_m)
        pairs = pairwise_analysis(bundle)

        for p in pairs:
            if (p["far_overlap_ratio"] > overlap_thresh and p["far_visible_fraction"] > vf_thresh
                    and p["far_camera_valid"]):
                suspicious.append({"frame": frame, **p})
            elif p["far_overlap_ratio"] > 0.85 and not p["far_camera_valid"]:
                if len(controls["fully_occluded_far_invalid"]) < 3:
                    controls["fully_occluded_far_invalid"].append({"frame": frame, **p})
            elif 0.05 < p["far_overlap_ratio"] < 0.3 and p["far_camera_valid"]:
                if len(controls["partial_far_valid"]) < 3:
                    controls["partial_far_valid"].append({"frame": frame, **p})
            elif p["far_overlap_ratio"] < 0.02 and p["far_camera_valid"]:
                if len(controls["no_overlap_far_valid"]) < 3:
                    controls["no_overlap_far_valid"].append({"frame": frame, **p})

        for obj, rv, rect in bundle:
            if rect is None:
                continue
            if 0.0 < rv["image_fraction"] < 0.9 and rv["unclipped_area_px"] > rv["intersection_area_px"]:
                if len(controls["boundary_truncated"]) < 3:
                    controls["boundary_truncated"].append({
                        "frame": frame, "logical_id": obj.get("logical_id"),
                        "image_fraction": rv["image_fraction"], "camera_valid": rv["camera_valid"],
                    })

    suspicious.sort(key=lambda r: -(r["far_overlap_ratio"] * r["far_visible_fraction"]))
    top = suspicious[:top_n]

    out_dir = out_root
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "suspicious_cases.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "near_logical_id", "far_logical_id", "near_distance_m", "far_distance_m",
                      "bbox_iou", "far_overlap_ratio", "far_visible_fraction", "far_camera_valid"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in top:
            writer.writerow({k: row[k] for k in fieldnames})

    controls_path = os.path.join(out_dir, "control_cases.json")
    with open(controls_path, "w", encoding="utf-8") as f:
        json.dump(controls, f, indent=2, default=str)

    print(f"[OK] suspicious cases (top {len(top)}/{len(suspicious)} candidates): {csv_path}")
    print(f"[OK] control cases: {controls_path}")
    return top, controls


# ============================================================
# Depth decoder self-test (CLAUDE.md section 6)
# ============================================================

class _FakeRawImage:
    def __init__(self, raw_data, height, width):
        self.raw_data = raw_data
        self.height = height
        self.width = width


def depth_decoder_self_test():
    """
    Independent re-derivation of CARLA's documented depth encoding,
    compared against src.data.collector.depth_to_numpy on synthetic
    BGRA buffers (raw depth bytes are not persisted by the collector --
    only the already-decoded meters array is saved to
    geometry/depth/*.npy -- so this is the decoder verification that can
    be done post-hoc without re-running CARLA).
    """
    rng = np.random.default_rng(0)
    h, w = 4, 4
    b = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    g = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    r = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)

    buf = np.zeros((h, w, 4), dtype=np.uint8)
    buf[:, :, 0], buf[:, :, 1], buf[:, :, 2], buf[:, :, 3] = b, g, r, a

    fake = _FakeRawImage(buf.tobytes(), h, w)
    decoded = depth_to_numpy(fake)

    independent = (r.astype(np.float64) + g.astype(np.float64) * 256.0 + b.astype(np.float64) * 65536.0) \
        / 16777215.0 * 1000.0

    max_abs_err = float(np.max(np.abs(decoded.astype(np.float64) - independent)))
    return {
        "max_abs_error_m": max_abs_err,
        "match": max_abs_err < 1e-3,  # decoded array is float32; ~1e-5 m rounding is expected, not a bug
        "sample_decoded": decoded[0].tolist(),
        "sample_independent": independent[0].tolist(),
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--town", type=str, required=True)
    parser.add_argument("--route", type=str, required=True)
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--frame", type=int, default=None, help="Single frame to fully diagnose.")
    parser.add_argument("--out-root", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "annotation_debug"))
    parser.add_argument("--bbox-mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--highlight-logical-ids", type=str, default="")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--temporal-range", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--suspicious-search", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--depth-decoder-self-test", action="store_true")
    args = parser.parse_args()

    highlight_ids = {int(x) for x in args.highlight_logical_ids.split(",") if x.strip() != ""}

    if args.depth_decoder_self_test:
        result = depth_decoder_self_test()
        print("[Depth decoder self-test]", json.dumps(result, indent=2))

    if args.frame is not None:
        run_single_frame(args.dataset_root, args.town, args.route, args.frame, args.condition,
                          args.out_root, args.bbox_mode, highlight_ids, verify=not args.no_verify)

    if args.temporal_range is not None:
        run_temporal_analysis(args.dataset_root, args.town, args.route, args.condition,
                               args.temporal_range[0], args.temporal_range[1], args.out_root)

    if args.suspicious_search:
        run_suspicious_search(args.dataset_root, args.town, args.route, args.condition,
                               args.out_root, top_n=args.top_n)


if __name__ == "__main__":
    main()

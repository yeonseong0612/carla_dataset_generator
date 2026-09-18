"""
scripts/tools/visualize_annotations.py

Re-projects stored 3D object annotations onto the saved rgb_left image,
and optionally renders an ego-frame BEV (bird's-eye view) sanity plot.

This tool ONLY reads files already on disk:
    - rgb_left/*.png
    - labels/object_3d/*.json
    - calibration.json

It never re-queries the CARLA simulator to regenerate bounding boxes, and
it never modifies any file under the dataset directory. The purpose is to
independently verify the *stored* ground truth, exactly as it was written
by the collector.

Projection pipeline (per calibration.json's documented convention,
p_A = T_A_from_B @ p_B):

    vertices_ego_m (ego frame)
        -> T_camera_from_ego (calibration.json: cameras.<name>.T_camera_from_ego)
        -> CARLA camera frame (x fwd, y right, z up)
        -> T_cv_from_carla (calibration.json: coordinate_systems.T_cv_from_carla)
        -> CV camera frame (x right, y down, z fwd)
        -> K (calibration.json: cameras.<name>.K)
        -> (u, v)

Camera-plane handling: a vertex with z_cv <= 0 is behind the camera and is
NOT projected (division by a non-positive z would be meaningless and can
divide-by-zero). An edge is only drawn when BOTH of its endpoints are in
front of the camera; edges with one endpoint behind the camera are skipped
rather than drawn through an undefined/garbage projection. This under-draws
partially-visible boxes instead of drawing something misleading -- boxes
straddling the camera plane are counted separately in the printed stats.

Object_3d annotations are a scene-level GT (everything within
ANNOTATION.MAX_DISTANCE of the ego vehicle), not a camera-visible-only GT.
Objects that don't project into the rgb_left frame are therefore expected
and are reported as statistics, not treated as errors.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.tools.bbox_geometry import BoxGeometryError, derive_edges_from_vertices
from src.data.projection import CameraProjector

# Visibility-first bbox palette (BGR for OpenCV) -- deliberately NOT
# SEMANTIC_MAP (Cityscapes-style segmentation colors): car/truck/bus/
# motorcycle there are all near-pure dark blue (0,0,142)/(0,0,70)/
# (0,60,100)/(0,0,230), which reads as "everything is blue" once drawn as
# thin bbox lines on a photo. This is a separate, high-saturation,
# maximally-distinct palette for GT box/label rendering only -- the actual
# semantic segmentation panel (visualize_sensor_grid.py) still uses the
# real SEMANTIC_MAP colors, unchanged, since that IS the semantic-class
# ground truth.
CATEGORY_COLOR_BGR = {
    "pedestrian": (40, 40, 255),     # red
    "cyclist": (0, 220, 255),        # yellow
    "motorcyclist": (255, 0, 230),   # magenta/purple
}

VEHICLE_SUBTYPE_COLOR_BGR = {
    "car": (255, 90, 20),    # blue
    "van": (255, 255, 0),    # cyan
    "truck": (0, 140, 255),  # orange
    "bus": (40, 200, 0),     # green
}

FALLBACK_COLOR_BGR = (255, 255, 255)


def category_color_bgr(category, subcategory):
    if category == "vehicle":
        return VEHICLE_SUBTYPE_COLOR_BGR.get(subcategory, VEHICLE_SUBTYPE_COLOR_BGR["car"])

    return CATEGORY_COLOR_BGR.get(category, FALLBACK_COLOR_BGR)


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------

def load_calibration(sequence_root):
    path = os.path.join(sequence_root, "calibration.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation(sequence_root, frame_id):
    path = os.path.join(sequence_root, "labels", "object_3d", f"{frame_id:06d}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb(sequence_root, frame_id, camera_name):
    path = os.path.join(sequence_root, camera_name, f"{frame_id:06d}.png")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return image


# ------------------------------------------------------------------
# Projection (CameraProjector itself now lives in src/data/projection.py
# -- shared with the live annotation pipeline, src/data/annotation.py --
# use CameraProjector.from_calibration(calibration, camera_name) here.)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# RGB projection rendering
# ------------------------------------------------------------------

def draw_label(image, text, org, color, font_scale=0.52, thickness=2):
    """
    Semi-transparent black background box + a small solid color tag
    stripe (so the category color is still visible at a glance) + bold
    white text -- reads clearly over both bright and dark backgrounds,
    unlike a bare colored/outlined string directly on the image. Used
    only for panel-level corner stats (e.g. "points=1234") now -- GT
    bbox text labels were removed per the visibility-refinement task.
    """

    h, w = image.shape[:2]
    x = int(np.clip(org[0], 0, w - 1))
    y = int(np.clip(org[1], 0, h - 1))

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    tag_w = 5
    pad = 3

    bg_x0 = max(0, x - pad - tag_w)
    bg_y0 = max(0, y - text_h - pad - 2)
    bg_x1 = min(w, x + text_w + pad)
    bg_y1 = min(h, y + baseline + pad)

    if bg_x1 <= bg_x0 or bg_y1 <= bg_y0:
        return

    overlay = image[bg_y0:bg_y1, bg_x0:bg_x1].copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.6, image[bg_y0:bg_y1, bg_x0:bg_x1], 0.4, 0, dst=image[bg_y0:bg_y1, bg_x0:bg_x1])

    cv2.rectangle(image, (bg_x0, bg_y0), (bg_x0 + tag_w, bg_y1), color, -1)

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_rgb_projection(image, annotation, projector):
    image = image.copy()
    h, w = image.shape[:2]

    stats = {
        "total": 0,
        "behind_camera": 0,
        "outside_image": 0,
        "projectable": 0,
        "partially_visible": 0,
    }

    for obj in annotation.get("objects", []):
        stats["total"] += 1

        bbox = obj.get("bbox_3d", {})
        vertices = bbox.get("vertices_ego_m")

        if not vertices or len(vertices) != 8:
            continue

        v_ego = np.asarray(vertices, dtype=np.float64)

        if not np.all(np.isfinite(v_ego)):
            continue

        v_cv = projector.ego_to_cv(v_ego)
        uv, valid = projector.project(v_cv)

        if not np.any(valid):
            stats["behind_camera"] += 1
            continue

        if not np.all(valid):
            stats["partially_visible"] += 1

        # Image-rect overlap test using only the in-front vertices.
        u_valid = uv[valid, 0]
        v_valid = uv[valid, 1]

        intersects_image = (
            u_valid.max() >= 0 and u_valid.min() < w
            and v_valid.max() >= 0 and v_valid.min() < h
        )

        if not intersects_image:
            stats["outside_image"] += 1
            continue

        stats["projectable"] += 1

        color = category_color_bgr(obj.get("category"), obj.get("subcategory"))

        try:
            edges, _ = derive_edges_from_vertices(v_ego)
        except BoxGeometryError:
            edges = []

        for i, j in edges:
            if not (valid[i] and valid[j]):
                continue

            p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))

            # Black outline first (wider), then the category color on top
            # (narrower) -- keeps the box readable over both bright and
            # dark/cluttered backgrounds instead of a thin colored line
            # that can disappear against similarly-colored pixels.
            cv2.line(image, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)

        # No text label -- bbox only (see module docstring / CLAUDE task:
        # "라벨은 전부 제거"). u_valid/v_valid above are still needed for
        # the intersects_image check.

    return image, stats


# ------------------------------------------------------------------
# BEV rendering
# ------------------------------------------------------------------

def render_bev(annotation, range_m=60.0, size_px=800):
    """
    Ego-frame BEV, independent of camera calibration. Draws, per object:
      - a solid footprint rectangle reconstructed purely from
        center_ego_m / yaw_ego_deg / dimensions_m (this is what those
        3 fields *claim*; it does not depend on vertices_ego_m at all).
      - a small magenta marker at the centroid of the stored
        vertices_ego_m, connected to the center by a dashed line when the
        two disagree by more than a small tolerance. This makes a
        vertices/center inconsistency directly visible without needing
        the RGB projection.

    x = forward (image up), y = right (image right).
    """

    canvas = np.full((size_px, size_px, 3), 30, dtype=np.uint8)
    scale = size_px / (2.0 * range_m)

    def to_px(x, y):
        px = int(size_px / 2 + y * scale)
        py = int(size_px / 2 - x * scale)
        return px, py

    # Range rings every 20 m.
    for r in np.arange(20.0, range_m + 1e-6, 20.0):
        cv2.circle(canvas, (size_px // 2, size_px // 2), int(r * scale), (60, 60, 60), 1, cv2.LINE_AA)

    # Ego origin + heading (forward = +x = up).
    origin = to_px(0.0, 0.0)
    cv2.drawMarker(canvas, origin, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.arrowedLine(canvas, origin, to_px(4.0, 0.0), (255, 255, 255), 2, tipLength=0.3)

    mismatch_count = 0

    for obj in annotation.get("objects", []):
        bbox = obj.get("bbox_3d", {})
        center = bbox.get("center_ego_m")
        dims = bbox.get("dimensions_m")
        yaw_deg = bbox.get("yaw_ego_deg")
        vertices = bbox.get("vertices_ego_m")

        if center is None or dims is None or yaw_deg is None:
            continue
        if not all(np.isfinite([center[0], center[1], yaw_deg])):
            continue

        cx, cy = float(center[0]), float(center[1])
        length = dims.get("length")
        width = dims.get("width")
        if not (isinstance(length, (int, float)) and isinstance(width, (int, float))):
            continue

        yaw = np.radians(yaw_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        half_l, half_w = length / 2.0, width / 2.0
        local_corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]

        color = category_color_bgr(obj.get("category"), obj.get("subcategory"))

        pts = []
        for lx, ly in local_corners:
            wx = cx + lx * cos_y - ly * sin_y
            wy = cy + lx * sin_y + ly * cos_y
            pts.append(to_px(wx, wy))

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Heading tick from center to front edge midpoint.
        front_mid_x = cx + half_l * cos_y
        front_mid_y = cy + half_l * sin_y
        cv2.line(canvas, to_px(cx, cy), to_px(front_mid_x, front_mid_y), color, 2, cv2.LINE_AA)

        label = obj.get("subcategory") or obj.get("category", "?")
        label_pos = to_px(cx, cy)
        draw_label(canvas, f"{label} #{obj.get('actor_id')}", (label_pos[0] + 6, label_pos[1] - 6), color)

        # Independent cross-check: where do the *stored* vertices actually
        # sit, versus the reconstructed footprint above?
        if vertices is not None and len(vertices) == 8:
            v = np.asarray(vertices, dtype=np.float64)
            if np.all(np.isfinite(v)):
                vmean = v.mean(axis=0)
                offset = float(np.hypot(vmean[0] - cx, vmean[1] - cy))
                if offset > 0.5:
                    mismatch_count += 1
                    vpx = to_px(vmean[0], vmean[1])
                    if 0 <= vpx[0] < size_px and 0 <= vpx[1] < size_px:
                        cv2.drawMarker(canvas, vpx, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)

    draw_label(canvas, f"range={range_m:.0f}m/ring  vertex/center mismatches off-plot: see stdout", (10, size_px - 10), (200, 200, 200))

    return canvas, mismatch_count


# ------------------------------------------------------------------
# Frame selection
# ------------------------------------------------------------------

def discover_available_frames(sequence_root):
    object_dir = os.path.join(sequence_root, "labels", "object_3d")
    files = sorted(glob.glob(os.path.join(object_dir, "*.json")))
    ids = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if name.isdigit():
            ids.append(int(name))
    return sorted(ids)


def pick_samples(available, frames, num_samples):
    if frames is not None:
        missing = [f for f in frames if f not in available]
        if missing:
            print(f"Warning: requested frames not found and will be skipped: {missing}")
        return [f for f in frames if f in available]

    if num_samples is not None and num_samples > 0:
        if num_samples >= len(available):
            return available
        idx = np.linspace(0, len(available) - 1, num_samples)
        idx = sorted(set(int(round(i)) for i in idx))
        return [available[i] for i in idx]

    return available


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproject stored 3D annotations onto rgb_left and render a BEV sanity view.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to a sequence root, e.g. dataset/Town01/route_0/day_clear")
    parser.add_argument("--camera", type=str, default="rgb_left", help="Camera name as used in calibration.json / directory name")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Explicit list of frame ids to render")
    parser.add_argument("--num-samples", type=int, default=None, help="Evenly-spaced number of frames to sample across the sequence")
    parser.add_argument("--output", type=str, default="outputs/annotation_validation", help="Output directory for rendered images")
    parser.add_argument("--bev-range", type=float, default=110.0, help="BEV plot half-range in meters (default covers ANNOTATION.MAX_DISTANCE=100m)")
    parser.add_argument("--no-bev", action="store_true", help="Skip BEV rendering")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    calibration = load_calibration(sequence_root)

    if args.camera not in calibration.get("cameras", {}):
        raise KeyError(f"Camera '{args.camera}' not found in calibration.json (available: {list(calibration.get('cameras', {}).keys())})")

    projector = CameraProjector.from_calibration(calibration, args.camera)

    available = discover_available_frames(sequence_root)
    if not available:
        print(f"No annotation files found under {sequence_root}/labels/object_3d")
        sys.exit(1)

    sample_frames = pick_samples(available, args.frames, args.num_samples)

    if not sample_frames:
        print("No frames selected to render.")
        sys.exit(1)

    print(f"Rendering {len(sample_frames)} frame(s) from {sequence_root} -> {output_dir}")

    totals = {"total": 0, "behind_camera": 0, "outside_image": 0, "projectable": 0, "partially_visible": 0}
    bev_mismatch_total = 0

    for frame_id in sample_frames:
        annotation = load_annotation(sequence_root, frame_id)
        image = load_rgb(sequence_root, frame_id, args.camera)

        rendered, stats = render_rgb_projection(image, annotation, projector)

        for k in totals:
            totals[k] += stats[k]

        out_path = os.path.join(output_dir, f"{frame_id:06d}.png")
        cv2.imwrite(out_path, rendered)

        print(
            f"  frame {frame_id:06d}: total={stats['total']} projectable={stats['projectable']} "
            f"behind_camera={stats['behind_camera']} outside_image={stats['outside_image']} "
            f"partially_visible={stats['partially_visible']} -> {out_path}"
        )

        if not args.no_bev:
            bev, mismatches = render_bev(annotation, range_m=args.bev_range)
            bev_mismatch_total += mismatches
            bev_path = os.path.join(output_dir, f"{frame_id:06d}_bev.png")
            cv2.imwrite(bev_path, bev)

    print()
    print("=" * 70)
    print("Visualization summary")
    print("=" * 70)
    print(f"Frames rendered: {len(sample_frames)}")
    print(f"Total annotated objects: {totals['total']}")
    print(f"Projectable objects (visible in image): {totals['projectable']}")
    print(f"Objects behind camera: {totals['behind_camera']}")
    print(f"Objects outside image: {totals['outside_image']}")
    print(f"Partially visible boxes (straddling camera plane): {totals['partially_visible']}")
    if not args.no_bev:
        print(f"BEV: objects where stored vertices centroid disagrees with center_ego_m by >0.5m: {bev_mismatch_total}")
    print("=" * 70)


if __name__ == "__main__":
    main()

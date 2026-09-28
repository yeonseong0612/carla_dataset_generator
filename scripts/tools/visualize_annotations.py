"""
scripts/tools/visualize_annotations.py

Re-project stored 3D object annotations onto saved RGB images.

Supports both:

Legacy layout:
    <sequence_root>/rgb_left/*.png

Current paired layout:
    <route_root>/
        geometry/
            calibration.json
            labels/object_3d/*.json
        conditions/
            day_clear/rgb_left/*.png
            night_rain/rgb_left/*.png
            ...

The tool ONLY reads stored files. It never queries CARLA and never modifies
dataset contents.

Debug additions:
    --condition
    --logical-ids
    --crop-padding
    --zoom

When --logical-ids is supplied:
    - only selected logical IDs are rendered
    - selected actors get a thick highlighted 3D bbox
    - projected bbox center is marked with a crosshair
    - enlarged crops are written separately

This is particularly useful for replay-validation debugging, because the
canonical annotation is projected onto the replay RGB. If a replay actor has
moved relative to canonical geometry, the rendered actor will visibly drift
away from the canonical projected bbox.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
)

from src.data.layout import resolve_geometry_root  # noqa: E402
from scripts.tools.bbox_geometry import (         # noqa: E402
    BoxGeometryError,
    derive_edges_from_vertices,
)
from src.data.projection import CameraProjector    # noqa: E402


# ============================================================
# Colours
# ============================================================

CATEGORY_COLOR_BGR = {
    "pedestrian": (40, 40, 255),
    "cyclist": (0, 220, 255),
    "motorcyclist": (255, 0, 230),
}

VEHICLE_SUBTYPE_COLOR_BGR = {
    "car": (255, 90, 20),
    "van": (255, 255, 0),
    "truck": (0, 140, 255),
    "bus": (40, 200, 0),
}

FALLBACK_COLOR_BGR = (255, 255, 255)

# Selected logical-id debug rendering.
HIGHLIGHT_COLOR_BGR = (0, 255, 255)
HIGHLIGHT_OUTLINE_BGR = (0, 0, 0)


def category_color_bgr(category, subcategory):
    if category == "vehicle":
        return VEHICLE_SUBTYPE_COLOR_BGR.get(
            subcategory,
            VEHICLE_SUBTYPE_COLOR_BGR["car"],
        )

    return CATEGORY_COLOR_BGR.get(category, FALLBACK_COLOR_BGR)


# ============================================================
# IO
# ============================================================

def load_calibration(sequence_root):
    geometry_root = resolve_geometry_root(sequence_root)
    path = os.path.join(geometry_root, "calibration.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation(sequence_root, frame_id):
    geometry_root = resolve_geometry_root(sequence_root)

    path = os.path.join(
        geometry_root,
        "labels",
        "object_3d",
        f"{frame_id:06d}.json",
    )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_rgb_directory(sequence_root, camera_name, condition=None):
    """
    Resolve RGB directory for both old and current layouts.

    Supported:

    1) Legacy:
       route_0/rgb_left/

    2) Current paired:
       route_0/conditions/night_rain/rgb_left/

    3) condition directory passed directly:
       route_0/conditions/night_rain/
    """

    sequence_root = os.path.abspath(sequence_root)

    # User passed condition directory directly.
    direct = os.path.join(sequence_root, camera_name)

    if os.path.isdir(direct):
        return direct

    # Current paired dataset layout.
    if condition is not None:
        candidate = os.path.join(
            sequence_root,
            "conditions",
            condition,
            camera_name,
        )

        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve RGB directory.\n"
        f"sequence_root = {sequence_root}\n"
        f"condition     = {condition}\n"
        f"camera        = {camera_name}\n\n"
        "Expected either:\n"
        f"  {sequence_root}\\{camera_name}\n"
        "or\n"
        f"  {sequence_root}\\conditions\\<condition>\\{camera_name}"
    )


def load_rgb(sequence_root, frame_id, camera_name, condition=None):
    rgb_dir = resolve_rgb_directory(
        sequence_root,
        camera_name,
        condition=condition,
    )

    path = os.path.join(
        rgb_dir,
        f"{frame_id:06d}.png",
    )

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read RGB image: {path}"
        )

    return image


# ============================================================
# Generic drawing
# ============================================================

def draw_label(
    image,
    text,
    org,
    color,
    font_scale=0.52,
    thickness=2,
):
    h, w = image.shape[:2]

    x = int(np.clip(org[0], 0, w - 1))
    y = int(np.clip(org[1], 0, h - 1))

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    tag_w = 5
    pad = 3

    bg_x0 = max(0, x - pad - tag_w)
    bg_y0 = max(0, y - text_h - pad - 2)
    bg_x1 = min(w, x + text_w + pad)
    bg_y1 = min(h, y + baseline + pad)

    if bg_x1 <= bg_x0 or bg_y1 <= bg_y0:
        return

    roi = image[bg_y0:bg_y1, bg_x0:bg_x1]

    overlay = np.zeros_like(roi)

    cv2.addWeighted(
        overlay,
        0.65,
        roi,
        0.35,
        0,
        dst=roi,
    )

    cv2.rectangle(
        image,
        (bg_x0, bg_y0),
        (bg_x0 + tag_w, bg_y1),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def object_logical_id(obj):
    """
    Prefer logical_id.

    actor_id fallback is only for compatibility with older annotations.
    """

    logical_id = obj.get("logical_id")

    if logical_id is not None:
        try:
            return int(logical_id)
        except (TypeError, ValueError):
            return logical_id

    actor_id = obj.get("actor_id")

    if actor_id is not None:
        try:
            return int(actor_id)
        except (TypeError, ValueError):
            return actor_id

    return None


# ============================================================
# Projection helpers
# ============================================================

def project_object(obj, projector, image_shape):
    """
    Returns projection information for one annotation object.

    None means unusable / fully behind camera / outside image.
    """

    h, w = image_shape[:2]

    bbox = obj.get("bbox_3d", {})
    vertices = bbox.get("vertices_ego_m")

    if not vertices or len(vertices) != 8:
        return None

    v_ego = np.asarray(
        vertices,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(v_ego)):
        return None

    v_cv = projector.ego_to_cv(v_ego)
    uv, valid = projector.project(v_cv)

    if not np.any(valid):
        return {
            "status": "behind_camera",
            "vertices_ego": v_ego,
            "uv": uv,
            "valid": valid,
        }

    u_valid = uv[valid, 0]
    v_valid = uv[valid, 1]

    intersects = (
        u_valid.max() >= 0
        and u_valid.min() < w
        and v_valid.max() >= 0
        and v_valid.min() < h
    )

    if not intersects:
        return {
            "status": "outside_image",
            "vertices_ego": v_ego,
            "uv": uv,
            "valid": valid,
        }

    x0 = float(u_valid.min())
    y0 = float(v_valid.min())
    x1 = float(u_valid.max())
    y1 = float(v_valid.max())

    center = (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
    )

    return {
        "status": "projectable",
        "vertices_ego": v_ego,
        "uv": uv,
        "valid": valid,
        "bbox_2d": (x0, y0, x1, y1),
        "center": center,
        "partially_visible": not np.all(valid),
    }


# ============================================================
# RGB projection rendering
# ============================================================

def draw_projected_box(
    image,
    projection,
    color,
    highlighted=False,
):
    uv = projection["uv"]
    valid = projection["valid"]
    v_ego = projection["vertices_ego"]

    try:
        edges, _ = derive_edges_from_vertices(v_ego)
    except BoxGeometryError:
        edges = []

    if highlighted:
        outer_width = 7
        inner_width = 4
    else:
        outer_width = 4
        inner_width = 2

    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue

        p1 = (
            int(round(uv[i, 0])),
            int(round(uv[i, 1])),
        )
        p2 = (
            int(round(uv[j, 0])),
            int(round(uv[j, 1])),
        )

        cv2.line(
            image,
            p1,
            p2,
            HIGHLIGHT_OUTLINE_BGR,
            outer_width,
            cv2.LINE_AA,
        )

        cv2.line(
            image,
            p1,
            p2,
            color,
            inner_width,
            cv2.LINE_AA,
        )


def draw_crosshair(image, center, color):
    x = int(round(center[0]))
    y = int(round(center[1]))

    cv2.drawMarker(
        image,
        (x, y),
        color,
        cv2.MARKER_CROSS,
        26,
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (x, y),
        7,
        color,
        2,
        cv2.LINE_AA,
    )


def render_rgb_projection(
    image,
    annotation,
    projector,
    logical_ids=None,
):
    image = image.copy()

    selected_ids = (
        set(logical_ids)
        if logical_ids
        else None
    )

    stats = {
        "total": 0,
        "behind_camera": 0,
        "outside_image": 0,
        "projectable": 0,
        "partially_visible": 0,
        "selected_found": 0,
        "selected_projectable": 0,
    }

    selected_projections = []

    for obj in annotation.get("objects", []):
        stats["total"] += 1

        logical_id = object_logical_id(obj)

        if selected_ids is not None:
            if logical_id not in selected_ids:
                continue

            stats["selected_found"] += 1

        projection = project_object(
            obj,
            projector,
            image.shape,
        )

        if projection is None:
            continue

        status = projection["status"]

        if status == "behind_camera":
            stats["behind_camera"] += 1
            continue

        if status == "outside_image":
            stats["outside_image"] += 1
            continue

        stats["projectable"] += 1

        if projection.get("partially_visible"):
            stats["partially_visible"] += 1

        highlighted = selected_ids is not None

        if highlighted:
            stats["selected_projectable"] += 1
            color = HIGHLIGHT_COLOR_BGR
        else:
            color = category_color_bgr(
                obj.get("category"),
                obj.get("subcategory"),
            )

        draw_projected_box(
            image,
            projection,
            color,
            highlighted=highlighted,
        )

        if highlighted:
            draw_crosshair(
                image,
                projection["center"],
                HIGHLIGHT_COLOR_BGR,
            )

            x0, y0, _, _ = projection["bbox_2d"]

            label_y = max(
                25,
                int(round(y0)) - 10,
            )

            draw_label(
                image,
                (
                    f"logical_id={logical_id}  "
                    f"{obj.get('category', '?')}"
                ),
                (
                    max(5, int(round(x0))),
                    label_y,
                ),
                HIGHLIGHT_COLOR_BGR,
                font_scale=0.65,
                thickness=2,
            )

            selected_projections.append(
                {
                    "logical_id": logical_id,
                    "object": obj,
                    "projection": projection,
                }
            )

    return image, stats, selected_projections


# ============================================================
# Selected-object crop
# ============================================================

def make_debug_crop(
    original_image,
    projection,
    logical_id,
    frame_id,
    condition,
    padding,
    zoom,
):
    """
    Crop around canonical projected bbox and enlarge it.

    The crop contains the ORIGINAL RGB plus canonical bbox overlay, which
    makes replay displacement easy to see.
    """

    image = original_image.copy()

    h, w = image.shape[:2]

    x0, y0, x1, y1 = projection["bbox_2d"]

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)

    half_w = bw / 2.0 + padding
    half_h = bh / 2.0 + padding

    crop_x0 = max(
        0,
        int(np.floor(cx - half_w)),
    )
    crop_y0 = max(
        0,
        int(np.floor(cy - half_h)),
    )
    crop_x1 = min(
        w,
        int(np.ceil(cx + half_w)),
    )
    crop_y1 = min(
        h,
        int(np.ceil(cy + half_h)),
    )

    if (
        crop_x1 <= crop_x0
        or crop_y1 <= crop_y0
    ):
        return None

    crop = image[
        crop_y0:crop_y1,
        crop_x0:crop_x1
    ].copy()

    # Canonical projected bbox in crop coordinates.
    bx0 = int(round(x0 - crop_x0))
    by0 = int(round(y0 - crop_y0))
    bx1 = int(round(x1 - crop_x0))
    by1 = int(round(y1 - crop_y0))

    cv2.rectangle(
        crop,
        (bx0, by0),
        (bx1, by1),
        (0, 0, 0),
        7,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        crop,
        (bx0, by0),
        (bx1, by1),
        HIGHLIGHT_COLOR_BGR,
        4,
        cv2.LINE_AA,
    )

    center_crop = (
        int(round(cx - crop_x0)),
        int(round(cy - crop_y0)),
    )

    cv2.drawMarker(
        crop,
        center_crop,
        HIGHLIGHT_COLOR_BGR,
        cv2.MARKER_CROSS,
        30,
        4,
        cv2.LINE_AA,
    )

    header = (
        f"frame={frame_id:06d}  "
        f"id={logical_id}  "
        f"condition={condition or 'legacy'}"
    )

    draw_label(
        crop,
        header,
        (15, 35),
        HIGHLIGHT_COLOR_BGR,
        font_scale=0.65,
        thickness=2,
    )

    if zoom is None or zoom <= 1:
        return crop

    interpolation = (
        cv2.INTER_NEAREST
        if zoom >= 3
        else cv2.INTER_LINEAR
    )

    crop = cv2.resize(
        crop,
        None,
        fx=float(zoom),
        fy=float(zoom),
        interpolation=interpolation,
    )

    return crop


# ============================================================
# BEV rendering
# ============================================================

def render_bev(
    annotation,
    range_m=60.0,
    size_px=800,
    logical_ids=None,
):
    canvas = np.full(
        (size_px, size_px, 3),
        30,
        dtype=np.uint8,
    )

    scale = size_px / (2.0 * range_m)

    selected_ids = (
        set(logical_ids)
        if logical_ids
        else None
    )

    def to_px(x, y):
        px = int(
            size_px / 2
            + y * scale
        )
        py = int(
            size_px / 2
            - x * scale
        )
        return px, py

    for r in np.arange(
        20.0,
        range_m + 1e-6,
        20.0,
    ):
        cv2.circle(
            canvas,
            (size_px // 2, size_px // 2),
            int(r * scale),
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    origin = to_px(0.0, 0.0)

    cv2.drawMarker(
        canvas,
        origin,
        (255, 255, 255),
        cv2.MARKER_CROSS,
        14,
        2,
    )

    cv2.arrowedLine(
        canvas,
        origin,
        to_px(4.0, 0.0),
        (255, 255, 255),
        2,
        tipLength=0.3,
    )

    mismatch_count = 0

    for obj in annotation.get("objects", []):
        logical_id = object_logical_id(obj)

        if (
            selected_ids is not None
            and logical_id not in selected_ids
        ):
            continue

        bbox = obj.get("bbox_3d", {})

        center = bbox.get("center_ego_m")
        dims = bbox.get("dimensions_m")
        yaw_deg = bbox.get("yaw_ego_deg")
        vertices = bbox.get("vertices_ego_m")

        if (
            center is None
            or dims is None
            or yaw_deg is None
        ):
            continue

        if not all(
            np.isfinite(
                [
                    center[0],
                    center[1],
                    yaw_deg,
                ]
            )
        ):
            continue

        cx = float(center[0])
        cy = float(center[1])

        length = dims.get("length")
        width = dims.get("width")

        if not (
            isinstance(length, (int, float))
            and isinstance(width, (int, float))
        ):
            continue

        yaw = np.radians(yaw_deg)

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        half_l = length / 2.0
        half_w = width / 2.0

        local_corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]

        if selected_ids is not None:
            color = HIGHLIGHT_COLOR_BGR
            thickness = 4
        else:
            color = category_color_bgr(
                obj.get("category"),
                obj.get("subcategory"),
            )
            thickness = 2

        pts = []

        for lx, ly in local_corners:
            wx = (
                cx
                + lx * cos_y
                - ly * sin_y
            )

            wy = (
                cy
                + lx * sin_y
                + ly * cos_y
            )

            pts.append(
                to_px(wx, wy)
            )

        pts = np.array(
            pts,
            dtype=np.int32,
        )

        cv2.polylines(
            canvas,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        front_mid_x = (
            cx
            + half_l * cos_y
        )

        front_mid_y = (
            cy
            + half_l * sin_y
        )

        cv2.line(
            canvas,
            to_px(cx, cy),
            to_px(
                front_mid_x,
                front_mid_y,
            ),
            color,
            thickness,
            cv2.LINE_AA,
        )

        label_pos = to_px(cx, cy)

        label = (
            f"id={logical_id} "
            f"{obj.get('subcategory') or obj.get('category', '?')}"
        )

        draw_label(
            canvas,
            label,
            (
                label_pos[0] + 6,
                label_pos[1] - 6,
            ),
            color,
        )

        if (
            vertices is not None
            and len(vertices) == 8
        ):
            v = np.asarray(
                vertices,
                dtype=np.float64,
            )

            if np.all(np.isfinite(v)):
                vmean = v.mean(axis=0)

                offset = float(
                    np.hypot(
                        vmean[0] - cx,
                        vmean[1] - cy,
                    )
                )

                if offset > 0.5:
                    mismatch_count += 1

                    vpx = to_px(
                        vmean[0],
                        vmean[1],
                    )

                    if (
                        0 <= vpx[0] < size_px
                        and 0 <= vpx[1] < size_px
                    ):
                        cv2.drawMarker(
                            canvas,
                            vpx,
                            (0, 0, 255),
                            cv2.MARKER_TILTED_CROSS,
                            10,
                            2,
                        )

    draw_label(
        canvas,
        f"range={range_m:.0f}m",
        (10, size_px - 10),
        (200, 200, 200),
    )

    return canvas, mismatch_count


# ============================================================
# Frame selection
# ============================================================

def discover_available_frames(sequence_root):
    object_dir = os.path.join(
        resolve_geometry_root(sequence_root),
        "labels",
        "object_3d",
    )

    files = sorted(
        glob.glob(
            os.path.join(
                object_dir,
                "*.json",
            )
        )
    )

    ids = []

    for p in files:
        name = os.path.splitext(
            os.path.basename(p)
        )[0]

        if name.isdigit():
            ids.append(int(name))

    return sorted(ids)


def pick_samples(
    available,
    frames,
    num_samples,
):
    if frames is not None:
        missing = [
            f
            for f in frames
            if f not in available
        ]

        if missing:
            print(
                "Warning: requested frames not found "
                f"and will be skipped: {missing}"
            )

        return [
            f
            for f in frames
            if f in available
        ]

    if (
        num_samples is not None
        and num_samples > 0
    ):
        if num_samples >= len(available):
            return available

        idx = np.linspace(
            0,
            len(available) - 1,
            num_samples,
        )

        idx = sorted(
            set(
                int(round(i))
                for i in idx
            )
        )

        return [
            available[i]
            for i in idx
        ]

    return available


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproject stored canonical 3D annotations "
            "onto saved condition RGB and optionally "
            "create logical-id debug crops."
        )
    )

    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help=(
            "Route root, e.g. "
            "D:/carla_dataset_final_v2/Town01/route_2"
        ),
    )

    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help=(
            "Condition under sequence/conditions/, "
            "e.g. day_clear or night_rain."
        ),
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="rgb_left",
        help="Camera name.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Explicit frame ids.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Evenly-spaced number of frames "
            "to sample."
        ),
    )

    parser.add_argument(
        "--logical-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Only visualize these logical IDs. "
            "Example: --logical-ids 88 89"
        ),
    )

    parser.add_argument(
        "--crop-padding",
        type=int,
        default=120,
        help=(
            "Extra pixels around selected actor "
            "bbox crop."
        ),
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=4.0,
        help=(
            "Scale factor for selected-object crop."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/annotation_validation",
        help="Output directory.",
    )

    parser.add_argument(
        "--bev-range",
        type=float,
        default=110.0,
        help="BEV half-range in meters.",
    )

    parser.add_argument(
        "--no-bev",
        action="store_true",
        help="Skip BEV output.",
    )

    args = parser.parse_args()

    sequence_root = os.path.abspath(
        args.sequence
    )

    output_dir = os.path.abspath(
        args.output
    )

    condition_tag = (
        args.condition
        if args.condition
        else "legacy"
    )

    condition_output_dir = os.path.join(
        output_dir,
        condition_tag,
    )

    crop_output_dir = os.path.join(
        condition_output_dir,
        "crops",
    )

    os.makedirs(
        condition_output_dir,
        exist_ok=True,
    )

    if args.logical_ids:
        os.makedirs(
            crop_output_dir,
            exist_ok=True,
        )

    calibration = load_calibration(
        sequence_root
    )

    if args.camera not in calibration.get(
        "cameras",
        {},
    ):
        raise KeyError(
            f"Camera '{args.camera}' not found in "
            "calibration.json "
            f"(available: "
            f"{list(calibration.get('cameras', {}).keys())})"
        )

    projector = CameraProjector.from_calibration(
        calibration,
        args.camera,
    )

    rgb_dir = resolve_rgb_directory(
        sequence_root,
        args.camera,
        condition=args.condition,
    )

    print(
        f"RGB directory : {rgb_dir}"
    )

    available = discover_available_frames(
        sequence_root
    )

    if not available:
        print(
            "No annotation files found under "
            f"{resolve_geometry_root(sequence_root)}"
            "/labels/object_3d"
        )
        sys.exit(1)

    sample_frames = pick_samples(
        available,
        args.frames,
        args.num_samples,
    )

    if not sample_frames:
        print(
            "No frames selected to render."
        )
        sys.exit(1)

    print(
        f"Rendering {len(sample_frames)} frame(s)"
    )
    print(
        f"Sequence  : {sequence_root}"
    )
    print(
        f"Condition : {condition_tag}"
    )
    print(
        f"Logical IDs: {args.logical_ids}"
    )
    print(
        f"Output    : {condition_output_dir}"
    )

    totals = {
        "total": 0,
        "behind_camera": 0,
        "outside_image": 0,
        "projectable": 0,
        "partially_visible": 0,
        "selected_found": 0,
        "selected_projectable": 0,
    }

    bev_mismatch_total = 0

    for frame_id in sample_frames:
        annotation = load_annotation(
            sequence_root,
            frame_id,
        )

        image = load_rgb(
            sequence_root,
            frame_id,
            args.camera,
            condition=args.condition,
        )

        rendered, stats, selected = (
            render_rgb_projection(
                image,
                annotation,
                projector,
                logical_ids=args.logical_ids,
            )
        )

        for key in totals:
            totals[key] += stats[key]

        frame_name = (
            f"{frame_id:06d}.png"
        )

        out_path = os.path.join(
            condition_output_dir,
            frame_name,
        )

        cv2.imwrite(
            out_path,
            rendered,
        )

        print(
            f"frame {frame_id:06d}: "
            f"total={stats['total']} "
            f"projectable={stats['projectable']} "
            f"selected_found={stats['selected_found']} "
            f"selected_projectable="
            f"{stats['selected_projectable']} "
            f"-> {out_path}"
        )

        # --------------------------------------------
        # Selected logical-id crops
        # --------------------------------------------

        for item in selected:
            logical_id = item["logical_id"]
            projection = item["projection"]

            crop = make_debug_crop(
                image,
                projection,
                logical_id,
                frame_id,
                args.condition,
                padding=args.crop_padding,
                zoom=args.zoom,
            )

            if crop is None:
                continue

            crop_name = (
                f"{frame_id:06d}"
                f"_id{logical_id}"
                f"_{condition_tag}"
                f"_crop.png"
            )

            crop_path = os.path.join(
                crop_output_dir,
                crop_name,
            )

            cv2.imwrite(
                crop_path,
                crop,
            )

            print(
                f"    crop logical_id={logical_id}"
                f" -> {crop_path}"
            )

        # --------------------------------------------
        # BEV
        # --------------------------------------------

        if not args.no_bev:
            bev, mismatches = render_bev(
                annotation,
                range_m=args.bev_range,
                logical_ids=args.logical_ids,
            )

            bev_mismatch_total += mismatches

            bev_path = os.path.join(
                condition_output_dir,
                f"{frame_id:06d}_bev.png",
            )

            cv2.imwrite(
                bev_path,
                bev,
            )

    print()
    print("=" * 70)
    print("Visualization summary")
    print("=" * 70)

    print(
        f"Frames rendered       : "
        f"{len(sample_frames)}"
    )

    print(
        f"Condition             : "
        f"{condition_tag}"
    )

    print(
        f"Selected logical IDs  : "
        f"{args.logical_ids}"
    )

    print(
        f"Total annotation objs : "
        f"{totals['total']}"
    )

    print(
        f"Projectable objs      : "
        f"{totals['projectable']}"
    )

    if args.logical_ids:
        print(
            f"Selected found        : "
            f"{totals['selected_found']}"
        )

        print(
            f"Selected projectable  : "
            f"{totals['selected_projectable']}"
        )

    print(
        f"Behind camera         : "
        f"{totals['behind_camera']}"
    )

    print(
        f"Outside image         : "
        f"{totals['outside_image']}"
    )

    print(
        f"Partially visible     : "
        f"{totals['partially_visible']}"
    )

    if not args.no_bev:
        print(
            "BEV vertex/center mismatch >0.5m : "
            f"{bev_mismatch_total}"
        )

    print(
        f"Output                : "
        f"{condition_output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
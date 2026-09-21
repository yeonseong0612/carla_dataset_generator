"""
scripts/tools/visualize_multisensor.py

Synchronized multi-sensor validation view:

    RGB + 3D Bounding Boxes | LiDAR BEV | Radar BEV

This tool ONLY reads files already on disk for one synchronized sample
(rgb_left/*.png, lidar/*.npy, radar*/*.npy, labels/object_3d/*.json,
calibration.json). It never re-queries the CARLA simulator and never
modifies any dataset file.

The RGB + 3D bbox panel reuses the verified projection code from
visualize_annotations.py (CameraProjector / render_rgb_projection) instead
of reimplementing the same projection logic.

LiDAR / Radar BEV
------------------
Points are stored on disk in their own sensor-local frame:

    lidar: [x_lidar, y_lidar, z_lidar, intensity]
    radar/radar_front_left/radar_front_right: [x_radar, y_radar, z_radar, radial_velocity]

They are converted to the ego frame using calibration.json's
sensors.<name>.T_ego_from_sensor (p_ego = T_ego_from_sensor @ p_sensor),
exactly the extrinsic the collector itself uses for LiDAR ROI filtering.
The 3 radars are always transformed to ego individually first and only
concatenated afterward (never concatenated in their own local frames).

Both BEV panels use a fixed ego-frame plot window:

    x_ego (forward): [SENSOR.LIDAR.ROI_FRONT_MIN, SENSOR.LIDAR.ROI_FRONT_MAX]
    y_ego (right):    [-SENSOR.LIDAR.ROI_SIDE, +SENSOR.LIDAR.ROI_SIDE]

with forward mapped to "up" in the image, so the LiDAR panel doubles as a
visual check that no rear points leak in and that the +-40m / 0-120m crop
is respected.

Multi-radar
-----------
The dataset now has 3 radar sensors: "radar" (front, unchanged), plus new
"radar_front_left" / "radar_front_right" corner radars. The Radar BEV
panel defaults to all 3 merged into the ego frame; --radar-mode selects a
single sensor for debugging. --compare additionally renders a front-only
panel and a 4-panel (RGB | LiDAR | Front Radar | Multi-Radar) comparison,
plus a quantitative front-only-vs-merged GT-object radar coverage report.
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

from src.data.layout import resolve_geometry_root  # noqa: E402
from CFG.config import cfg

from scripts.tools.visualize_annotations import (
    CameraProjector,
    category_color_bgr,
    discover_available_frames,
    draw_label,
    load_annotation,
    load_calibration,
    load_rgb,
    pick_samples,
    render_rgb_projection,
)

ROI = {
    "x_min": float(cfg.SENSOR.LIDAR.ROI_FRONT_MIN),
    "x_max": float(cfg.SENSOR.LIDAR.ROI_FRONT_MAX),
    "y_half": float(cfg.SENSOR.LIDAR.ROI_SIDE),
}

ROI_TOLERANCE_M = 1e-3

# "radar" is the existing front radar (name/meaning unchanged for
# compatibility). radar_front_left/radar_front_right are the new corner
# radars, matching src/data/collector.py's Collector.RADAR_SENSORS.
RADAR_SENSOR_NAMES = ("radar", "radar_front_left", "radar_front_right")

RADAR_DISPLAY_NAMES = {
    "radar": "Front",
    "radar_front_left": "Front-Left",
    "radar_front_right": "Front-Right",
}

MERGED_RADAR_KEY = "merged"


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------

def load_lidar(sequence_root, frame_id):
    path = os.path.join(resolve_geometry_root(sequence_root), "lidar", f"{frame_id:06d}.npy")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    return np.load(path)


def load_radar(sequence_root, frame_id, sensor_name="radar"):
    path = os.path.join(resolve_geometry_root(sequence_root), sensor_name, f"{frame_id:06d}.npy")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    return np.load(path)


def check_frame_files(sequence_root, frame_id, camera_name):
    """
    A synchronized sample is the RGB frame, LiDAR, all 3 radars, and the
    GT label sharing local_frame_id. Fail loudly and specifically if any
    of them is missing.
    """

    frame_name = f"{frame_id:06d}"

    required = {
        camera_name: os.path.join(sequence_root, camera_name, f"{frame_name}.png"),
        "lidar": os.path.join(resolve_geometry_root(sequence_root), "lidar", f"{frame_name}.npy"),
        "labels/object_3d": os.path.join(resolve_geometry_root(sequence_root), "labels", "object_3d", f"{frame_name}.json"),
    }

    for sensor_name in RADAR_SENSOR_NAMES:
        required[sensor_name] = os.path.join(resolve_geometry_root(sequence_root), sensor_name, f"{frame_name}.npy")

    missing = [name for name, path in required.items() if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            f"Frame {frame_name}: synchronized sample is missing required file(s) "
            f"{missing} under {sequence_root}"
        )


def get_extrinsic(calibration, sensor_name):
    """
    T_ego_from_sensor for `sensor_name`, as saved by calibration.py.
    """

    sensors = calibration.get("sensors", {})

    if sensor_name not in sensors:
        raise KeyError(
            f"Sensor '{sensor_name}' not found in calibration.json "
            f"(available: {list(sensors.keys())})"
        )

    return np.asarray(sensors[sensor_name]["T_ego_from_sensor"], dtype=np.float64)


def sensor_xyz_to_ego(xyz_sensor, T_ego_from_sensor):
    n = xyz_sensor.shape[0]
    homo = np.hstack([xyz_sensor, np.ones((n, 1))])

    return (T_ego_from_sensor @ homo.T).T[:, :3]


def lidar_points_to_ego(lidar_arr, T_ego_from_lidar):
    xyz_ego = sensor_xyz_to_ego(lidar_arr[:, :3].astype(np.float64), T_ego_from_lidar)
    intensity = lidar_arr[:, 3].astype(np.float64)

    return xyz_ego, intensity


def radar_points_to_ego(radar_arr, T_ego_from_radar):
    xyz_ego = sensor_xyz_to_ego(radar_arr[:, :3].astype(np.float64), T_ego_from_radar)
    velocity = radar_arr[:, 3].astype(np.float64)

    return xyz_ego, velocity


def load_radar_sensor_ego(sequence_root, frame_id, calibration, sensor_name):
    """
    Load one radar's raw local-frame points and transform to ego using
    that radar's own extrinsic. Returns (xyz_ego (N,3), velocity (N,)).
    """

    radar_arr = load_radar(sequence_root, frame_id, sensor_name)
    T_ego_from_radar = get_extrinsic(calibration, sensor_name)

    return radar_points_to_ego(radar_arr, T_ego_from_radar)


def load_all_radars_ego(sequence_root, frame_id, calibration, sensor_names=RADAR_SENSOR_NAMES):
    """
    Load + ego-transform every radar sensor individually (never
    concatenating raw local-frame points across sensors), returning a
    dict {sensor_name: (xyz_ego, velocity)} plus a MERGED_RADAR_KEY entry
    that is the ego-frame concatenation of all of them.
    """

    per_sensor = {
        name: load_radar_sensor_ego(sequence_root, frame_id, calibration, name)
        for name in sensor_names
    }

    xyz_list = [xyz for xyz, _ in per_sensor.values()]
    vel_list = [vel for _, vel in per_sensor.values()]

    merged_xyz = np.concatenate(xyz_list, axis=0) if xyz_list else np.zeros((0, 3))
    merged_vel = np.concatenate(vel_list, axis=0) if vel_list else np.zeros((0,))

    per_sensor[MERGED_RADAR_KEY] = (merged_xyz, merged_vel)

    return per_sensor


# ------------------------------------------------------------------
# BEV coordinate mapping
#
# x_ego (forward) -> image "up" ; y_ego (right) -> image "right".
# The plotted window is exactly ROI, so the canvas border IS the ROI
# boundary.
# ------------------------------------------------------------------

def bev_frame_size(roi, height_px):
    x_span = roi["x_max"] - roi["x_min"]
    y_span = 2.0 * roi["y_half"]

    width_px = max(1, int(round(height_px * y_span / x_span)))

    return width_px, height_px


def bev_to_px(x, y, roi, width_px, height_px):
    y_min = -roi["y_half"]
    y_max = roi["y_half"]

    px = (y - y_min) / (y_max - y_min) * width_px
    py = height_px - (x - roi["x_min"]) / (roi["x_max"] - roi["x_min"]) * height_px

    return px, py


BEV_BACKGROUND = (48, 48, 48)  # dark gray, not near-black -- less eye strain, more room for point/bbox contrast
BEV_GRID_LINE = (95, 95, 95)
BEV_GRID_TEXT = (195, 195, 195)


def draw_bev_grid(canvas, roi, width_px, height_px, step=20.0):
    x = roi["x_min"]

    while x <= roi["x_max"] + 1e-6:
        _, py = bev_to_px(x, 0.0, roi, width_px, height_px)
        py = int(round(np.clip(py, 0, height_px - 1)))

        cv2.line(canvas, (0, py), (width_px, py), BEV_GRID_LINE, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, f"{x:.0f}m", (4, max(12, py - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, BEV_GRID_TEXT, 1, cv2.LINE_AA,
        )

        x += step

    y = -roi["y_half"]

    while y <= roi["y_half"] + 1e-6:
        px, _ = bev_to_px(0.0, y, roi, width_px, height_px)
        px = int(round(np.clip(px, 0, width_px - 1)))

        cv2.line(canvas, (px, 0), (px, height_px), BEV_GRID_LINE, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, f"{y:+.0f}", (min(px + 2, width_px - 26), height_px - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, BEV_GRID_TEXT, 1, cv2.LINE_AA,
        )

        y += step

    # ROI boundary. The BEV window == ROI, so this is the canvas border.
    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), (0, 200, 255), 2)

    # Ego marker at (x=0, y=0), sitting on the near boundary -- black
    # shadow marker underneath so it still pops against a bright cluster
    # of points near the origin, not just against the plain background.
    ex, ey = bev_to_px(0.0, 0.0, roi, width_px, height_px)
    ex = int(round(ex))
    ey = int(round(np.clip(ey, 0, height_px - 1)))

    cv2.drawMarker(canvas, (ex, ey), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 20, 4)
    cv2.drawMarker(canvas, (ex, ey), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.putText(canvas, "EGO", (ex + 8, ey - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "EGO", (ex + 8, ey - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def scatter_points(canvas, points_ego, roi, color_fn, radius=1):
    width_px, height_px = canvas.shape[1], canvas.shape[0]

    x_min, x_max, y_half = roi["x_min"], roi["x_max"], roi["y_half"]

    for i in range(points_ego.shape[0]):
        x, y = points_ego[i, 0], points_ego[i, 1]

        if not (x_min <= x <= x_max and -y_half <= y <= y_half):
            continue

        px, py = bev_to_px(x, y, roi, width_px, height_px)
        ix, iy = int(round(px)), int(round(py))

        if radius <= 0:
            # Smallest possible marker: a single raw pixel, no
            # anti-aliasing bleed. cv2.circle(radius=1, LINE_AA) actually
            # spreads over several pixels for smoothing, so this is
            # strictly smaller -- needed for dense radar BEVs.
            if 0 <= ix < width_px and 0 <= iy < height_px:
                canvas[iy, ix] = color_fn(i)
            continue

        cv2.circle(
            canvas, (ix, iy), radius,
            color_fn(i), -1, cv2.LINE_AA,
        )


def draw_bev_bbox(canvas, obj, roi):
    """
    Overlay the GT footprint reconstructed from center_ego_m /
    dimensions_m / yaw_ego_deg (independent of vertices_ego_m). Bbox
    only -- no text label (see module docstring / CLAUDE task: "라벨은
    전부 제거").
    """

    bbox = obj.get("bbox_3d", {})
    center = bbox.get("center_ego_m")
    dims = bbox.get("dimensions_m")
    yaw_deg = bbox.get("yaw_ego_deg")

    if center is None or dims is None or yaw_deg is None:
        return

    if not all(np.isfinite([center[0], center[1], yaw_deg])):
        return

    cx, cy = float(center[0]), float(center[1])
    length = dims.get("length")
    width = dims.get("width")

    if not (isinstance(length, (int, float)) and isinstance(width, (int, float))):
        return

    # Cheap reject for boxes nowhere near the plotted ROI.
    margin = max(length, width)

    if (
        cx + margin < roi["x_min"]
        or cx - margin > roi["x_max"]
        or abs(cy) - margin > roi["y_half"]
    ):
        return

    yaw = np.radians(yaw_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    half_l, half_w = length / 2.0, width / 2.0
    local_corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]

    color = category_color_bgr(obj.get("category"), obj.get("subcategory"))

    width_px, height_px = canvas.shape[1], canvas.shape[0]

    pts = []

    for lx, ly in local_corners:
        wx = cx + lx * cos_y - ly * sin_y
        wy = cy + lx * sin_y + ly * cos_y

        px, py = bev_to_px(wx, wy, roi, width_px, height_px)
        pts.append((int(np.clip(px, -32000, 32000)), int(np.clip(py, -32000, 32000))))

    pts_arr = np.array(pts, dtype=np.int32)
    # Black outline first (wider) then the category color on top
    # (narrower) -- same treatment as the RGB panel, so a box doesn't
    # blend into a dense point cluster of a similar hue.
    cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pts_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    front_mid_x = cx + half_l * cos_y
    front_mid_y = cy + half_l * sin_y

    p1 = bev_to_px(cx, cy, roi, width_px, height_px)
    p2 = bev_to_px(front_mid_x, front_mid_y, roi, width_px, height_px)

    p1i = (int(np.clip(p1[0], -32000, 32000)), int(np.clip(p1[1], -32000, 32000)))
    p2i = (int(np.clip(p2[0], -32000, 32000)), int(np.clip(p2[1], -32000, 32000)))

    cv2.line(canvas, p1i, p2i, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(canvas, p1i, p2i, color, 2, cv2.LINE_AA)


# ------------------------------------------------------------------
# Per-sensor coloring
# ------------------------------------------------------------------

def lidar_point_color(intensity_val):
    # BGR. High floor (never below ~160) so even low-intensity points stay
    # clearly visible against BEV_BACKGROUND -- previously this floored
    # near (60,50,30), which was almost invisible against a near-black
    # canvas. Bright cyan (low intensity) -> bright yellow-white (high).
    v = float(np.clip(intensity_val, 0.0, 1.0))

    return (int(255 - 80 * v), int(215 + 40 * v), int(60 + 180 * v))


def radar_point_color(velocity_val, vmax=15.0):
    # CARLA radar convention: negative velocity == approaching the sensor.
    # Floor raised from 0.25 to 0.55 so slow-relative-velocity points
    # don't fade to near-black; base colors are fully saturated so the
    # approaching/receding split stays obvious even at the floor.
    t = float(np.clip(abs(velocity_val) / vmax, 0.55, 1.0))
    base = (30, 30, 255) if velocity_val < 0 else (255, 170, 30)  # BGR: red=approaching, cyan-blue=receding

    return tuple(int(c * t) for c in base)


# ------------------------------------------------------------------
# BEV panel rendering
# ------------------------------------------------------------------

def render_lidar_bev(points_ego, intensity, roi, width_px, height_px, annotation):
    canvas = np.full((height_px, width_px, 3), BEV_BACKGROUND, dtype=np.uint8)

    draw_bev_grid(canvas, roi, width_px, height_px)

    # Points first, GT boxes drawn on top -- a dense point cluster can no
    # longer swallow a box's outline (previously boxes were drawn under
    # the points).
    scatter_points(canvas, points_ego, roi, lambda i: lidar_point_color(intensity[i]), radius=1)

    for obj in annotation.get("objects", []):
        draw_bev_bbox(canvas, obj, roi)

    draw_label(canvas, f"points={points_ego.shape[0]}", (8, height_px - 8), (220, 220, 220))

    return canvas


def render_radar_bev(points_ego, velocity, roi, width_px, height_px, annotation, source_label=None):
    canvas = np.full((height_px, width_px, 3), BEV_BACKGROUND, dtype=np.uint8)

    draw_bev_grid(canvas, roi, width_px, height_px)

    # Points first, GT boxes on top -- see render_lidar_bev.
    scatter_points(canvas, points_ego, roi, lambda i: radar_point_color(velocity[i]), radius=1)

    for obj in annotation.get("objects", []):
        draw_bev_bbox(canvas, obj, roi)

    prefix = f"[{source_label}] " if source_label else ""

    draw_label(
        canvas, f"{prefix}detections={points_ego.shape[0]}  red=approaching / blue=receding",
        (8, height_px - 8), (220, 220, 220),
    )

    return canvas


# ------------------------------------------------------------------
# Panel composition
# ------------------------------------------------------------------

def add_title_strip(image, title, strip_height=30, bg=(20, 20, 20), fg=(255, 255, 255)):
    width = image.shape[1]

    strip = np.full((strip_height, width, 3), bg, dtype=np.uint8)
    cv2.putText(strip, title, (10, strip_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)

    return np.vstack([strip, image])


def compose_panels(panels, outer_title):
    """
    panels: list of (title, BGR image) with equal height content.
    """

    max_h = max(image.shape[0] for _, image in panels)

    strips = []

    for title, image in panels:
        if image.shape[0] != max_h:
            new_w = int(round(image.shape[1] * max_h / image.shape[0]))
            image = cv2.resize(image, (new_w, max_h))

        strips.append(add_title_strip(image, title))

    divider = np.full((strips[0].shape[0], 4, 3), (255, 255, 255), dtype=np.uint8)

    row = strips[0]

    for strip in strips[1:]:
        row = np.hstack([row, divider, strip])

    title_bar = np.full((44, row.shape[1], 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, outer_title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([title_bar, row])


def parse_sequence_title(sequence_root, frame_id):
    parts = [p for p in os.path.normpath(sequence_root).split(os.sep) if p]

    town = parts[-3] if len(parts) >= 3 else ""
    route_raw = parts[-2] if len(parts) >= 2 else ""
    condition = parts[-1] if len(parts) >= 1 else ""

    route_label = route_raw.replace("route_", "Route ").replace("_", " ") if route_raw else route_raw

    return f"{town} / {route_label} / {condition} / Frame {frame_id:06d}"


# ------------------------------------------------------------------
# LiDAR ROI validation (numeric, independent of the rendered images)
# ------------------------------------------------------------------

def validate_lidar_roi(sequence_root, frame_ids, T_ego_from_lidar, roi):
    global_min_x = np.inf
    global_max_x = -np.inf
    global_min_y = np.inf
    global_max_y = -np.inf

    total_points = 0
    total_in_roi = 0

    for frame_id in frame_ids:
        lidar_arr = load_lidar(sequence_root, frame_id)

        if lidar_arr.shape[0] == 0:
            continue

        xyz_ego, _ = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)
        x_ego = xyz_ego[:, 0]
        y_ego = xyz_ego[:, 1]

        global_min_x = min(global_min_x, float(x_ego.min()))
        global_max_x = max(global_max_x, float(x_ego.max()))
        global_min_y = min(global_min_y, float(y_ego.min()))
        global_max_y = max(global_max_y, float(y_ego.max()))

        mask = (
            (x_ego >= roi["x_min"])
            & (x_ego <= roi["x_max"])
            & (np.abs(y_ego) <= roi["y_half"])
        )

        total_points += len(lidar_arr)
        total_in_roi += int(mask.sum())

    retention_ratio = (total_in_roi / total_points) if total_points > 0 else 0.0

    return {
        "num_frames": len(frame_ids),
        "min_x_ego": global_min_x,
        "max_x_ego": global_max_x,
        "min_y_ego": global_min_y,
        "max_y_ego": global_max_y,
        "raw_point_count": total_points,
        "roi_point_count": total_in_roi,
        "retention_ratio": retention_ratio,
    }


def print_roi_validation(stats, roi):
    print()
    print("=" * 70)
    print("LiDAR ROI validation (saved dataset, ego frame)")
    print("=" * 70)
    print(f"Frames checked: {stats['num_frames']}")
    print(f"Expected ROI: x_ego in [{roi['x_min']:.1f}, {roi['x_max']:.1f}], "
          f"y_ego in [{-roi['y_half']:.1f}, {roi['y_half']:.1f}]")
    print(f"min(x_ego) = {stats['min_x_ego']:.3f}   max(x_ego) = {stats['max_x_ego']:.3f}")
    print(f"min(y_ego) = {stats['min_y_ego']:.3f}   max(y_ego) = {stats['max_y_ego']:.3f}")

    ok_x_min = stats["min_x_ego"] >= roi["x_min"] - ROI_TOLERANCE_M
    ok_x_max = stats["max_x_ego"] <= roi["x_max"] + ROI_TOLERANCE_M
    ok_y_min = stats["min_y_ego"] >= -roi["y_half"] - ROI_TOLERANCE_M
    ok_y_max = stats["max_y_ego"] <= roi["y_half"] + ROI_TOLERANCE_M

    print(f"PASS" if (ok_x_min and ok_x_max and ok_y_min and ok_y_max) else "FAIL",
          "- saved LiDAR points satisfy the ego-frame ROI (tolerance "
          f"{ROI_TOLERANCE_M:g} m)")

    print(f"Saved point count (all frames): {stats['raw_point_count']}")
    print(f"Points inside ROI:              {stats['roi_point_count']}")
    print(f"Retention ratio:                {stats['retention_ratio']:.4f}")
    print("=" * 70)


# ------------------------------------------------------------------
# Radar detection density statistics (PART B)
# ------------------------------------------------------------------

def compute_radar_density_stats(sequence_root, frame_ids, sensor_names=RADAR_SENSOR_NAMES):
    """
    Per-sensor and merged (summed) radar detection counts across
    frame_ids. Counts are read straight from the saved per-radar .npy
    files (sensor-local frame point count == detection count; no ego
    transform needed just to count points).
    """

    per_sensor_counts = {name: [] for name in sensor_names}
    merged_counts = []

    for frame_id in frame_ids:
        total = 0

        for name in sensor_names:
            n = load_radar(sequence_root, frame_id, name).shape[0]
            per_sensor_counts[name].append(n)
            total += n

        merged_counts.append(total)

    return per_sensor_counts, merged_counts


def summarize_counts(counts):
    values = np.asarray(counts, dtype=np.float64)

    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def print_radar_density_report(
    frame_ids, per_sensor_counts, merged_counts, sensor_names=RADAR_SENSOR_NAMES,
    target_low=5000, target_high=15000, preferred_low=8000, preferred_high=12000,
):
    print()
    print("=" * 70)
    print("Radar detection density (3-radar ego-frame merge)")
    print("=" * 70)
    print(f"Frames measured: {len(frame_ids)}")
    print()

    for name in sensor_names:
        s = summarize_counts(per_sensor_counts[name])
        print(
            f"{RADAR_DISPLAY_NAMES.get(name, name):12s}: "
            f"mean={s['mean']:.1f} min={s['min']:.0f} max={s['max']:.0f} "
            f"median={s['median']:.1f} std={s['std']:.1f}"
        )

    merged_stats = summarize_counts(merged_counts)

    print(
        f"{'Merged':12s}: "
        f"mean={merged_stats['mean']:.1f} min={merged_stats['min']:.0f} max={merged_stats['max']:.0f} "
        f"median={merged_stats['median']:.1f} std={merged_stats['std']:.1f}"
    )
    print()

    in_target = target_low <= merged_stats["mean"] <= target_high
    in_preferred = preferred_low <= merged_stats["mean"] <= preferred_high

    print(
        ("PASS" if in_target else "FAIL"),
        f"- merged mean detections/frame within target [{target_low}, {target_high}]",
    )
    print(
        ("within" if in_preferred else "outside"),
        f"preferred band [{preferred_low}, {preferred_high}]",
    )
    print("=" * 70)

    return merged_stats


# ------------------------------------------------------------------
# Representative frame selection (PART L)
# ------------------------------------------------------------------

def count_near_objects(annotation, x_max, y_abs_max):
    """
    Count GT objects, by category, inside an ego-frame near window
    (0 < x_ego < x_max, |y_ego| < y_abs_max) -- used to rank frames by
    how much nearby traffic they contain.
    """

    counts = {"vehicle": 0, "cyclist": 0, "motorcyclist": 0, "pedestrian": 0}

    for obj in annotation.get("objects", []):
        center = obj.get("bbox_3d", {}).get("center_ego_m")

        if center is None:
            continue

        cx, cy = float(center[0]), float(center[1])

        if 0.0 <= cx <= x_max and abs(cy) <= y_abs_max:
            category = obj.get("category")

            if category in counts:
                counts[category] += 1

    return counts


def select_representative_frames(sequence_root, available_frames, top_k=5, x_max=80.0, y_abs_max=25.0):
    """
    Rank frames by nearby vehicle count (ties broken by total nearby
    actor count, then frame id) and return the top_k frame ids plus their
    counts, so a "busy traffic" sample can be chosen instead of picked
    arbitrarily.
    """

    scored = []

    for frame_id in available_frames:
        annotation = load_annotation(sequence_root, frame_id)
        counts = count_near_objects(annotation, x_max, y_abs_max)
        total = sum(counts.values())

        scored.append((counts["vehicle"], total, frame_id, counts))

    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))

    top = scored[:top_k]

    return [frame_id for _, _, frame_id, _ in top], {frame_id: counts for _, _, frame_id, counts in top}


# ------------------------------------------------------------------
# Multi-radar object coverage (PART M)
# ------------------------------------------------------------------

def object_footprint(obj, margin=0.0):
    bbox = obj.get("bbox_3d", {})
    center = bbox.get("center_ego_m")
    dims = bbox.get("dimensions_m")
    yaw_deg = bbox.get("yaw_ego_deg")

    if center is None or dims is None or yaw_deg is None:
        return None

    length = dims.get("length")
    width = dims.get("width")

    if not (isinstance(length, (int, float)) and isinstance(width, (int, float))):
        return None

    if not all(np.isfinite([center[0], center[1], yaw_deg])):
        return None

    return {
        "cx": float(center[0]),
        "cy": float(center[1]),
        "yaw_deg": float(yaw_deg),
        "half_l": length / 2.0 + margin,
        "half_w": width / 2.0 + margin,
    }


def count_points_in_footprint(points_ego_xy, footprint):
    if points_ego_xy.shape[0] == 0:
        return 0

    yaw = np.radians(footprint["yaw_deg"])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    dx = points_ego_xy[:, 0] - footprint["cx"]
    dy = points_ego_xy[:, 1] - footprint["cy"]

    # Rotate into the box-local frame (inverse of the box's yaw rotation).
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y

    mask = (
        (np.abs(local_x) <= footprint["half_l"])
        & (np.abs(local_y) <= footprint["half_w"])
    )

    return int(mask.sum())


def compute_frame_radar_coverage(
    annotation, radar_xy_by_source, x_max, y_abs_max, footprint_margin=0.3,
):
    """
    For every vehicle/cyclist/motorcyclist GT object within the near
    window, count how many radar detections from each source (per-sensor
    + merged) land inside its ego-frame footprint (expanded by
    footprint_margin to tolerate point spread).
    """

    results = []

    for obj in annotation.get("objects", []):

        if obj.get("category") not in ("vehicle", "cyclist", "motorcyclist"):
            continue

        center = obj.get("bbox_3d", {}).get("center_ego_m")

        if center is None:
            continue

        cx, cy = float(center[0]), float(center[1])

        if not (0.0 <= cx <= x_max and abs(cy) <= y_abs_max):
            continue

        footprint = object_footprint(obj, margin=footprint_margin)

        if footprint is None:
            continue

        counts = {
            source: count_points_in_footprint(xy, footprint)
            for source, xy in radar_xy_by_source.items()
        }

        results.append(
            {
                "actor_id": obj.get("actor_id"),
                "label": obj.get("subcategory") or obj.get("category"),
                "distance_m": obj.get("distance_m"),
                "counts": counts,
            }
        )

    return results


def compute_sequence_radar_coverage(sequence_root, frame_ids, calibration, x_max, y_abs_max, footprint_margin=0.3):
    """
    Same per-object radar coverage as compute_frame_radar_coverage, but
    aggregated over an entire set of frames (e.g. every frame in the
    sequence) rather than only the handful that get rendered to images.
    """

    all_results = []

    for frame_id in frame_ids:
        annotation = load_annotation(sequence_root, frame_id)
        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)
        radar_xy_by_source = {name: xy[:, :2] for name, (xy, _) in radar_by_source.items()}

        all_results.extend(
            compute_frame_radar_coverage(
                annotation, radar_xy_by_source, x_max, y_abs_max, footprint_margin,
            )
        )

    return all_results


def print_radar_coverage_report(all_object_results, threshold, sensor_names=RADAR_SENSOR_NAMES, max_rows=25):
    front_key = sensor_names[0]

    print()
    print("=" * 70)
    print("Front-only vs Multi-Radar GT-object coverage")
    print("=" * 70)
    print(f"Observed-object threshold: >= {threshold} radar point(s) inside the GT footprint")
    print(f"Objects checked: {len(all_object_results)}")
    print()

    # Rows where the front radar alone misses the object but the merged
    # set catches it are the most interesting evidence of improvement, so
    # surface those first when the full list is too long to print.
    def row_priority(row):
        front_hit = row["counts"].get(front_key, 0) >= threshold
        merged_hit = row["counts"].get(MERGED_RADAR_KEY, 0) >= threshold
        return 0 if (merged_hit and not front_hit) else 1

    ordered = sorted(all_object_results, key=row_priority)
    shown = ordered[:max_rows]

    for row in shown:
        counts = row["counts"]
        dist = row["distance_m"]
        dist_str = f"{dist:.1f}m" if isinstance(dist, (int, float)) else "?m"

        parts = ", ".join(
            f"{RADAR_DISPLAY_NAMES.get(name, name)}={counts.get(name, 0)}"
            for name in sensor_names
        )

        print(
            f"  {row['label']} #{row['actor_id']} {dist_str}: "
            f"{parts}, merged={counts.get(MERGED_RADAR_KEY, 0)}"
        )

    if len(ordered) > max_rows:
        print(f"  ... ({len(ordered) - max_rows} more object-observations not shown)")

    total = len(all_object_results)

    front_observed = sum(
        1 for row in all_object_results if row["counts"].get(front_key, 0) >= threshold
    )

    merged_observed = sum(
        1 for row in all_object_results if row["counts"].get(MERGED_RADAR_KEY, 0) >= threshold
    )

    print()
    print(f"Front radar only : {front_observed} / {total} objects observed")
    print(f"Multi-radar merged: {merged_observed} / {total} objects observed")
    print("=" * 70)

    return total, front_observed, merged_observed


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RGB + 3D bbox | LiDAR BEV | Radar BEV synchronized validation view."
    )
    parser.add_argument("--sequence", type=str, required=True, help="Path to a sequence root, e.g. outputs/town10_multiradar_validation/Town10/route_0/conditions/day_clear")
    parser.add_argument("--camera", type=str, default="rgb_left", help="Camera name as used in calibration.json / directory name")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Explicit list of frame ids to render")
    parser.add_argument("--num-samples", type=int, default=None, help="Evenly-spaced number of frames to sample across the sequence")
    parser.add_argument("--auto-select", type=int, default=None, metavar="N", help="Instead of --frames/--num-samples, auto-pick the N frames with the most nearby vehicles (see --near-x-max/--near-y-abs-max)")
    parser.add_argument("--output", type=str, default="outputs/multisensor_validation", help="Output directory for rendered composite images")
    parser.add_argument("--panel-height", type=int, default=640, help="Content height in pixels shared by all panels")
    parser.add_argument("--skip-roi-validation", action="store_true", help="Skip the numeric LiDAR ROI validation pass over the full sequence")
    parser.add_argument("--skip-density-stats", action="store_true", help="Skip the per-sensor/merged radar detection density statistics pass over the full sequence")
    parser.add_argument("--radar-mode", type=str, default=MERGED_RADAR_KEY, choices=[MERGED_RADAR_KEY, *RADAR_SENSOR_NAMES], help="Which radar source the main Radar BEV panel shows (default: merged Front+FrontLeft+FrontRight)")
    parser.add_argument("--compare", action="store_true", help="Also render a front-only 3-panel, a merged 3-panel, and a 4-panel (RGB|LiDAR|Front|Multi-Radar) comparison per frame, plus a front-only-vs-merged GT-object radar coverage report")
    parser.add_argument("--coverage-threshold", type=int, default=1, help="Minimum radar points inside a GT object's footprint to call it radar-observed (used by --compare)")
    parser.add_argument("--near-x-max", type=float, default=80.0, help="Near-window forward bound (m) for --auto-select and --compare coverage")
    parser.add_argument("--near-y-abs-max", type=float, default=25.0, help="Near-window lateral bound (m) for --auto-select and --compare coverage")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    calibration = load_calibration(sequence_root)

    if args.camera not in calibration.get("cameras", {}):
        raise KeyError(
            f"Camera '{args.camera}' not found in calibration.json "
            f"(available: {list(calibration.get('cameras', {}).keys())})"
        )

    projector = CameraProjector(calibration, args.camera)

    T_ego_from_lidar = get_extrinsic(calibration, "lidar")

    available = discover_available_frames(sequence_root)

    if not available:
        print(f"No annotation files found under {sequence_root}/labels/object_3d")
        sys.exit(1)

    if args.auto_select is not None:
        sample_frames, near_counts = select_representative_frames(
            sequence_root, available, top_k=args.auto_select,
            x_max=args.near_x_max, y_abs_max=args.near_y_abs_max,
        )

        print(
            f"Auto-selected {len(sample_frames)} frame(s) by nearby vehicle count "
            f"(0<x_ego<{args.near_x_max:.0f}m, |y_ego|<{args.near_y_abs_max:.0f}m):"
        )

        for frame_id in sample_frames:
            counts = near_counts[frame_id]
            print(f"  frame {frame_id:06d}: vehicles={counts['vehicle']} cyclists={counts['cyclist']} motorcyclists={counts['motorcyclist']} pedestrians={counts['pedestrian']}")

    else:
        sample_frames = pick_samples(available, args.frames, args.num_samples)

    if not sample_frames:
        print("No frames selected to render.")
        sys.exit(1)

    print(f"Rendering {len(sample_frames)} frame(s) from {sequence_root} -> {output_dir}")

    width_px, height_px = bev_frame_size(ROI, args.panel_height)

    for frame_id in sample_frames:
        check_frame_files(sequence_root, frame_id, args.camera)

        annotation = load_annotation(sequence_root, frame_id)
        image = load_rgb(sequence_root, frame_id, args.camera)

        rendered_rgb, rgb_stats = render_rgb_projection(image, annotation, projector)

        lidar_arr = load_lidar(sequence_root, frame_id)
        lidar_xyz_ego, lidar_intensity = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)
        lidar_canvas = render_lidar_bev(lidar_xyz_ego, lidar_intensity, ROI, width_px, height_px, annotation)

        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)

        title = parse_sequence_title(sequence_root, frame_id)

        radar_xyz_ego, radar_velocity = radar_by_source[args.radar_mode]
        main_radar_canvas = render_radar_bev(
            radar_xyz_ego, radar_velocity, ROI, width_px, height_px, annotation,
            source_label=RADAR_DISPLAY_NAMES.get(args.radar_mode, "Front+FL+FR merged"),
        )

        composite = compose_panels(
            [
                ("RGB + 3D Bounding Boxes", rendered_rgb),
                ("LiDAR BEV", lidar_canvas),
                ("Radar BEV", main_radar_canvas),
            ],
            title,
        )

        out_path = os.path.join(output_dir, f"{frame_id:06d}.png")
        cv2.imwrite(out_path, composite)

        radar_point_summary = " ".join(
            f"{RADAR_DISPLAY_NAMES.get(name, name)}={radar_by_source[name][0].shape[0]}"
            for name in RADAR_SENSOR_NAMES
        )
        merged_point_count = radar_by_source[MERGED_RADAR_KEY][0].shape[0]

        print(
            f"  frame {frame_id:06d}: objects={rgb_stats['total']} "
            f"rgb_projectable={rgb_stats['projectable']} "
            f"lidar_pts={lidar_arr.shape[0]} radar[{radar_point_summary} merged={merged_point_count}] -> {out_path}"
        )

        if args.compare:
            front_xyz, front_vel = radar_by_source["radar"]
            merged_xyz, merged_vel = radar_by_source[MERGED_RADAR_KEY]

            front_canvas = render_radar_bev(front_xyz, front_vel, ROI, width_px, height_px, annotation, source_label="Front only")
            merged_canvas = render_radar_bev(merged_xyz, merged_vel, ROI, width_px, height_px, annotation, source_label="Front+FL+FR merged")

            front_only_path = os.path.join(output_dir, f"{frame_id:06d}_front_only.png")
            cv2.imwrite(
                front_only_path,
                compose_panels(
                    [
                        ("RGB + 3D Bounding Boxes", rendered_rgb),
                        ("LiDAR BEV", lidar_canvas),
                        ("Front Radar Only", front_canvas),
                    ],
                    title,
                ),
            )

            multiradar_path = os.path.join(output_dir, f"{frame_id:06d}_multiradar.png")
            cv2.imwrite(
                multiradar_path,
                compose_panels(
                    [
                        ("RGB + 3D Bounding Boxes", rendered_rgb),
                        ("LiDAR BEV", lidar_canvas),
                        ("Merged 3-Radar", merged_canvas),
                    ],
                    title,
                ),
            )

            compare4_path = os.path.join(output_dir, f"{frame_id:06d}_compare4.png")
            cv2.imwrite(
                compare4_path,
                compose_panels(
                    [
                        ("RGB + 3D Bounding Boxes", rendered_rgb),
                        ("LiDAR BEV", lidar_canvas),
                        ("Front Radar Only", front_canvas),
                        ("Merged 3-Radar", merged_canvas),
                    ],
                    title,
                ),
            )

            print(f"    compare -> {front_only_path}, {multiradar_path}, {compare4_path}")

    if args.compare:
        # Aggregate over every frame in the sequence (not just the
        # handful rendered above) for a representative coverage number.
        coverage_results = compute_sequence_radar_coverage(
            sequence_root, available, calibration,
            x_max=args.near_x_max, y_abs_max=args.near_y_abs_max,
        )

        print_radar_coverage_report(coverage_results, args.coverage_threshold)

    if not args.skip_density_stats:
        per_sensor_counts, merged_counts = compute_radar_density_stats(sequence_root, available)
        print_radar_density_report(available, per_sensor_counts, merged_counts)

    if not args.skip_roi_validation:
        roi_stats = validate_lidar_roi(sequence_root, available, T_ego_from_lidar, ROI)
        print_roi_validation(roi_stats, ROI)


if __name__ == "__main__":
    main()

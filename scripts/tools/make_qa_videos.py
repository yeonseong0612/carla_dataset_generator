"""
scripts/tools/make_qa_videos.py

Offline QA video generator for ONE already-collected route directory.

    python scripts/tools/make_qa_videos.py --route-root D:\\carla_dataset_final_v2\\Town07\\route_0

writes, by default, into outputs/qa_videos/<Town>/<route>/:

    multimodal_<condition>.mp4   3x3 grid (default condition: day_clear)
        RGB Left            | RGB Right         | RGB Left + bbox/class + LiDAR
        Depth               | Semantic          | Optical Flow
        LiDAR BEV           | Radar BEV (x3)    | Status / Metadata

    weather_comparison.mp4       2x3 grid of rgb_left at the same frame id
        day_clear   | day_rain   | day_fog
        night_clear | night_rain | night_fog

This tool ONLY reads files already on disk. It never starts / connects to a
CARLA server and never writes anything under the dataset directory.

Dataset layout (src/data/layout.py, src/data/collector.py,
src/data/metadata.py, src/data/annotation.py, scripts/collect_dataset.py):

    route_x/
        geometry/                       canonical, shared by every condition
            calibration.json            cameras.<name>.K / T_camera_from_ego,
                                        sensors.<name>.T_ego_from_sensor
            sequence.json               town / map / recording_hz / ...
            timestamps.csv              frame_id, carla_frame, timestamp
            ego_state.csv               ... speed_mps ... route_progress (%)
            pose/poses.csv
            depth/NNNNNN.npy            float32 HxW, meters
            semantic/NNNNNN.npy         uint8 HxW, CARLA semantic tag
            optical_flow/NNNNNN.npy     float32 HxWx2, CARLA normalized units
            lidar/NNNNNN.npy            float32 Nx4 [x,y,z,intensity], LiDAR frame
            radar/NNNNNN.npy            float32 Nx4 [x,y,z,radial_vel], front radar
            radar_front_left/NNNNNN.npy     (same, own sensor frame)
            radar_front_right/NNNNNN.npy    (same, own sensor frame)
            labels/object_3d/NNNNNN.json    "objects" = camera-valid final labels
            world_state/NNNNNN.json
        conditions/<condition>/
            rgb_left/NNNNNN.png
            rgb_right/NNNNNN.png
            condition.json              weather_parameters, replay_validation
        paired_validation.json

Reused (imported, not re-implemented; production code is not modified):
    src.data.projection.CameraProjector                 LiDAR / bbox -> RGB
    src.data.calibration.carla_transform_to_matrix      cfg extrinsic fallback
    src.data.annotation.SEMANTIC_MAP                    semantic palette
    scripts.tools.visualize_annotations                 calibration / label
                                                        loading, 3D bbox drawing
    scripts.tools.visualize_multisensor                 LiDAR / radar loading,
                                                        sensor->ego, BEV grid
    scripts.tools.visualize_flow                        flow decoding / colors
"""

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, PROJECT_ROOT)

from CFG.config import cfg  # noqa: E402
from src.data.calibration import carla_transform_to_matrix  # noqa: E402
from src.data.layout import (  # noqa: E402
    CONDITIONS_DIRNAME,
    GEOMETRY_DIRNAME,
    condition_dir,
    geometry_dir,
)
from src.data.projection import CameraProjector  # noqa: E402

from scripts.tools.visualize_annotations import (  # noqa: E402
    category_color_bgr,
    render_rgb_projection,
)
from scripts.tools.visualize_flow import (  # noqa: E402
    DEFAULT_BRIGHT_MAX_FLOW,
    draw_flow_legend,
    flow_to_color_bright,
    load_flow,
)
from scripts.tools.visualize_multisensor import (  # noqa: E402
    RADAR_DISPLAY_NAMES,
    RADAR_SENSOR_NAMES,
    ROI as LIDAR_ROI,
    bev_frame_size,
    bev_to_px,
    draw_bev_bbox,
    draw_bev_grid,
    lidar_points_to_ego,
    radar_point_color,
    radar_points_to_ego,
)

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional -- fall back to a plain counter.
    tqdm = None


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

WEATHER_GRID = (
    ("day_clear", "day_rain", "day_fog"),
    ("night_clear", "night_rain", "night_fog"),
)

PANEL_W = 512
PANEL_H = 384

FALLBACK_FPS = 10.0

DEFAULT_MAX_DEPTH_M = 100.0
LIDAR_OVERLAY_MAX_DIST_M = 80.0
LIDAR_BEV_Z_RANGE = (-2.5, 3.0)  # ego-frame height color scale [m]

# Corner radars are yawed +-45 deg with 90 deg FOV / 60 m range
# (CFG/config.py), so their detections reach straight out to the sides; the
# radar BEV window is therefore wider than the (front-only) LiDAR ROI.
RADAR_ROI = {"x_min": 0.0, "x_max": 120.0, "y_half": 60.0}

# cfg attribute per radar directory, same pairing as src/sensors/sensor_rig.py.
RADAR_CFG_ATTR = {
    "radar": "RADAR",
    "radar_front_left": "RADAR_FRONT_LEFT",
    "radar_front_right": "RADAR_FRONT_RIGHT",
}

RADAR_SENSOR_COLOR_BGR = {
    "radar": (60, 60, 255),                 # red
    "radar_front_left": (80, 230, 80),      # green
    "radar_front_right": (255, 170, 40),    # blue
}

# Marker footprint per radar (pixel offsets), so sensors stay distinguishable
# even when points are colored by radial velocity.
RADAR_MARKER_OFFSETS = {
    "radar": [(0, 0), (1, 0), (0, 1), (1, 1)],                      # square
    "radar_front_left": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],  # plus
    "radar_front_right": [(0, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)],  # x
}

REQUIRED_MULTIMODAL = ("rgb_left", "rgb_right", "depth", "semantic", "optical_flow", "lidar")
OPTIONAL_MULTIMODAL = ("labels",) + tuple(RADAR_SENSOR_NAMES)

TEXT_WHITE = (255, 255, 255)
TEXT_DIM = (170, 170, 170)
PANEL_BG = (18, 18, 18)


def warn(message):
    print(f"[QA][WARN] {message}")


def info(message):
    print(f"[QA] {message}")


# ------------------------------------------------------------------
# Semantic palette (src.data.annotation.SEMANTIC_MAP)
# ------------------------------------------------------------------

def load_semantic_map():
    """
    src.data.annotation imports the carla module at import time. Use it when
    available; otherwise read the very same SEMANTIC_MAP literal from that
    file without executing it, so the palette always has a single source.
    """

    try:
        from src.data.annotation import SEMANTIC_MAP

        return SEMANTIC_MAP
    except ImportError:
        pass

    path = os.path.join(PROJECT_ROOT, "src", "data", "annotation.py")

    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SEMANTIC_MAP" for t in node.targets
        ):
            return ast.literal_eval(node.value)

    raise RuntimeError(f"SEMANTIC_MAP not found in {path}")


def build_semantic_lut(semantic_map):
    lut = np.zeros((256, 3), dtype=np.uint8)

    for tag, (_name, rgb) in semantic_map.items():
        lut[int(tag)] = rgb[::-1]  # RGB -> BGR

    return lut


# ------------------------------------------------------------------
# Small file helpers
# ------------------------------------------------------------------

def read_json(path):
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        warn(f"could not read {path}: {exc}")
        return None


def read_frame_csv(path):
    """CSV keyed by int(frame_id) (timestamps.csv / ego_state.csv)."""

    rows = {}

    if not os.path.isfile(path):
        return rows

    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = row.get("frame_id", "")

            if frame.isdigit():
                rows[int(frame)] = row

    return rows


def scan_frame_ids(directory, extension):
    if not os.path.isdir(directory):
        return None

    ids = set()

    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)

        if ext.lower() == extension and stem.isdigit():
            ids.add(int(stem))

    return ids


def frame_file(directory, frame_id, extension):
    return os.path.join(directory, f"{frame_id:06d}{extension}")


def format_ids(ids, limit=8):
    ids = sorted(ids)
    head = ", ".join(str(i) for i in ids[:limit])

    return f"[{head}{', ...' if len(ids) > limit else ''}] ({len(ids)} total)"


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    return out if np.isfinite(out) else None


# ------------------------------------------------------------------
# Route discovery
# ------------------------------------------------------------------

class RouteData:
    """Paths + route-level metadata for one route directory."""

    def __init__(self, route_root):
        self.route_root = os.path.abspath(route_root)

        if os.path.basename(self.route_root) == GEOMETRY_DIRNAME:
            self.route_root = os.path.dirname(self.route_root)

        if os.path.basename(os.path.dirname(self.route_root)) == CONDITIONS_DIRNAME:
            self.route_root = os.path.dirname(os.path.dirname(self.route_root))

        self.geometry = geometry_dir(self.route_root)

        if not os.path.isdir(self.geometry):
            raise FileNotFoundError(
                f"No '{GEOMETRY_DIRNAME}/' directory under {self.route_root} "
                "-- expected a paired route directory (route_x/geometry, route_x/conditions)."
            )

        self.sequence = read_json(os.path.join(self.geometry, "sequence.json")) or {}
        self.calibration = read_json(os.path.join(self.geometry, "calibration.json"))
        self.paired_validation = read_json(os.path.join(self.route_root, "paired_validation.json"))

        if self.calibration is None:
            warn("geometry/calibration.json missing -- LiDAR/bbox projection disabled, "
                 "BEV extrinsics fall back to CFG/config.py")

        self.town = self.sequence.get("town") or os.path.basename(os.path.dirname(self.route_root))
        self.route_name = os.path.basename(self.route_root)

        self.timestamps = read_frame_csv(os.path.join(self.geometry, "timestamps.csv"))
        self.ego_state = read_frame_csv(os.path.join(self.geometry, "ego_state.csv"))

    def condition_path(self, condition):
        return condition_dir(self.route_root, condition)

    def condition_meta(self, condition):
        return read_json(os.path.join(self.condition_path(condition), "condition.json"))

    def recording_fps(self):
        value = as_float(self.sequence.get("recording_hz"))

        if value and value > 0:
            return value, "sequence.json:recording_hz"

        return FALLBACK_FPS, "fallback"

    def modality_dirs(self, condition):
        cond = self.condition_path(condition)

        dirs = {
            "rgb_left": (os.path.join(cond, "rgb_left"), ".png"),
            "rgb_right": (os.path.join(cond, "rgb_right"), ".png"),
            "depth": (os.path.join(self.geometry, "depth"), ".npy"),
            "semantic": (os.path.join(self.geometry, "semantic"), ".npy"),
            "optical_flow": (os.path.join(self.geometry, "optical_flow"), ".npy"),
            "lidar": (os.path.join(self.geometry, "lidar"), ".npy"),
            "labels": (os.path.join(self.geometry, "labels", "object_3d"), ".json"),
        }

        for name in RADAR_SENSOR_NAMES:
            dirs[name] = (os.path.join(self.geometry, name), ".npy")

        return dirs


def apply_frame_range(ids, start, end, stride):
    ids = sorted(i for i in ids if (start is None or i >= start) and (end is None or i <= end))

    return ids[::max(1, int(stride))]


def discover_multimodal_frames(route, condition):
    """
    Frame ids = intersection of every REQUIRED modality actually present on
    disk. Returns (frame_ids, available_modalities). Nothing is assumed to
    be 0..N-1; gaps and mismatches are reported, never silently shifted.
    """

    dirs = route.modality_dirs(condition)
    found = {name: scan_frame_ids(path, ext) for name, (path, ext) in dirs.items()}

    if not found["rgb_left"]:
        raise FileNotFoundError(
            f"No rgb_left frames for condition '{condition}' under "
            f"{dirs['rgb_left'][0]}"
        )

    available = {name for name, ids in found.items() if ids}

    for name in REQUIRED_MULTIMODAL + OPTIONAL_MULTIMODAL:
        if name not in available:
            kind = "required" if name in REQUIRED_MULTIMODAL else "optional"
            warn(f"{kind} modality '{name}' not found ({dirs[name][0]}) -- its panel will show 'unavailable'")

    required = [name for name in REQUIRED_MULTIMODAL if name in available]
    union = set().union(*(found[name] for name in required))
    frames = set.intersection(*(found[name] for name in required))

    dropped = union - frames

    if dropped:
        warn(f"{len(dropped)} frame(s) are missing at least one required modality and are skipped: {format_ids(dropped)}")

        for name in required:
            missing = union - found[name]

            if missing:
                warn(f"    '{name}' missing {format_ids(missing)}")

    for name in OPTIONAL_MULTIMODAL:
        if name in available:
            missing = frames - found[name]

            if missing:
                warn(f"optional '{name}' missing for {format_ids(missing)} -- shown as 'missing' on those frames")

    return sorted(frames), available


def discover_weather_frames(route, conditions, strict):
    """
    Returns (frame_ids, present_conditions): intersection of rgb_left over
    every condition that exists. --strict-weather turns any gap into an error.
    """

    found = {}

    for condition in conditions:
        ids = scan_frame_ids(os.path.join(route.condition_path(condition), "rgb_left"), ".png")

        if not ids:
            message = f"condition '{condition}' has no rgb_left frames ({route.condition_path(condition)})"

            if strict:
                raise FileNotFoundError(message)

            warn(message + " -- panel will show 'unavailable'")
            continue

        found[condition] = ids

    if not found:
        raise FileNotFoundError("No condition has rgb_left frames -- nothing to compare.")

    union = set().union(*found.values())
    frames = set.intersection(*found.values())
    dropped = union - frames

    if dropped:
        details = {c: union - ids for c, ids in found.items() if union - ids}
        message = f"{len(dropped)} frame(s) are not present in every condition: {format_ids(dropped)}"

        if strict:
            raise RuntimeError(
                message + "; " + "; ".join(f"{c} missing {format_ids(m)}" for c, m in details.items())
            )

        warn(message + " -- using the common intersection only")

        for condition, missing in details.items():
            warn(f"    '{condition}' missing {format_ids(missing)}")

    return sorted(frames), [c for c in conditions if c in found]


# ------------------------------------------------------------------
# Extrinsics (calibration.json, CFG fallback)
# ------------------------------------------------------------------

def cfg_extrinsic(sensor_cfg):
    transform = SimpleNamespace(
        location=SimpleNamespace(x=sensor_cfg.X, y=sensor_cfg.Y, z=sensor_cfg.Z),
        rotation=SimpleNamespace(pitch=sensor_cfg.PITCH, yaw=sensor_cfg.YAW, roll=sensor_cfg.ROLL),
    )

    return carla_transform_to_matrix(transform)


def sensor_extrinsic(calibration, sensor_name):
    """(T_ego_from_sensor, source) -- calibration.json first, CFG otherwise."""

    sensors = (calibration or {}).get("sensors", {})

    if sensor_name in sensors and "T_ego_from_sensor" in sensors[sensor_name]:
        return np.asarray(sensors[sensor_name]["T_ego_from_sensor"], dtype=np.float64), "calibration"

    cfg_attr = "LIDAR" if sensor_name == "lidar" else RADAR_CFG_ATTR.get(sensor_name)

    if cfg_attr is None:
        return None, None

    return cfg_extrinsic(getattr(cfg.SENSOR, cfg_attr)), "cfg"


def scaled_projector(calibration, camera_name, scale):
    """
    CameraProjector for an image resized by `scale`: only K changes
    (fx, fy, cx, cy scale linearly), so the projector itself is reused
    unchanged and draws straight onto the downscaled panel.
    """

    base = CameraProjector.from_calibration(calibration, camera_name)
    K = base.K.copy()
    K[:2, :] *= scale

    return CameraProjector(
        K=K,
        T_camera_from_ego=base.T_camera_from_ego,
        T_cv_from_carla=base.T_cv_from_carla,
        width=int(round(base.width * scale)),
        height=int(round(base.height * scale)),
    )


# ------------------------------------------------------------------
# Panel utilities
# ------------------------------------------------------------------

def blank_panel(message=None, sub=None):
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    if message:
        center_text(panel, message, PANEL_H // 2, 0.7, (80, 160, 255))

    if sub:
        center_text(panel, sub, PANEL_H // 2 + 28, 0.45, TEXT_DIM)

    return panel


def center_text(image, text, y, scale, color, thickness=1):
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(4, (image.shape[1] - w) // 2)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_size(width, height):
    scale = min(PANEL_W / width, PANEL_H / height)

    return scale, max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def letterbox(image):
    """Place an image (already <= panel size) centered on a panel canvas."""

    h, w = image.shape[:2]

    if w > PANEL_W or h > PANEL_H:
        _, w, h = fit_size(w, h)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    y0 = (PANEL_H - h) // 2
    x0 = (PANEL_W - w) // 2
    panel[y0:y0 + h, x0:x0 + w] = image

    return panel


def resize_to_panel(image, interpolation=cv2.INTER_AREA):
    _, w, h = fit_size(image.shape[1], image.shape[0])

    return cv2.resize(image, (w, h), interpolation=interpolation)


def put_title(panel, title, right_text=None):
    band = 24
    roi = panel[:band]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(panel, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_WHITE, 1, cv2.LINE_AA)

    if right_text:
        (w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(panel, right_text, (PANEL_W - w - 8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    return panel


def text_box(image, text, org, color=TEXT_WHITE, scale=0.4, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    x = int(np.clip(x, 0, max(0, image.shape[1] - w - 4)))
    y = int(np.clip(y, h + 3, image.shape[0] - base - 1))
    cv2.rectangle(image, (x, y - h - 3), (x + w + 4, y + base + 1), bg, -1)
    cv2.putText(image, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def colormap_colors(values, vmin, vmax, colormap=cv2.COLORMAP_TURBO):
    """Fixed-range colormap lookup for an (N,) array -> (N,3) BGR uint8."""

    t = np.clip((np.asarray(values, dtype=np.float64) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = (t * 255.0).astype(np.uint8).reshape(-1, 1)

    return cv2.applyColorMap(u8, colormap).reshape(-1, 3)


def paint_pixels(canvas, px, py, colors, offsets=((0, 0),)):
    """Vectorized point painting (one pixel per offset per point)."""

    h, w = canvas.shape[:2]
    px = np.round(px).astype(np.int64)
    py = np.round(py).astype(np.int64)

    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        canvas[y[keep], x[keep]] = colors[keep]


# ------------------------------------------------------------------
# Multimodal panels
# ------------------------------------------------------------------

def render_rgb_panel(path, title):
    image = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.isfile(path) else None

    if image is None:
        return put_title(blank_panel("RGB missing", os.path.basename(path)), title), None

    return put_title(letterbox(resize_to_panel(image)), title, f"{image.shape[1]}x{image.shape[0]}"), image


def project_lidar_on_image(image, projector, lidar_xyz_ego):
    cv_points = projector.ego_to_cv(lidar_xyz_ego)
    uv, valid = projector.project(cv_points)

    if not np.any(valid):
        return 0

    u = uv[valid, 0]
    v = uv[valid, 1]
    z = cv_points[valid, 2]

    inside = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    u, v, z = u[inside], v[inside], z[inside]

    order = np.argsort(-z)  # far first, near painted on top
    u, v, z = u[order], v[order], z[order]

    colors = colormap_colors(z, 0.0, LIDAR_OVERLAY_MAX_DIST_M)
    paint_pixels(image, u, v, colors)

    return int(len(z))


def draw_class_labels(image, annotation, projector):
    """Class name at the top-left of each projected box (no actor IDs)."""

    h, w = image.shape[:2]

    for obj in annotation.get("objects", []):
        vertices = obj.get("bbox_3d", {}).get("vertices_ego_m")

        if not vertices or len(vertices) != 8:
            continue

        v_ego = np.asarray(vertices, dtype=np.float64)

        if not np.all(np.isfinite(v_ego)):
            continue

        uv, valid = projector.project(projector.ego_to_cv(v_ego))

        if not np.any(valid):
            continue

        u, v = uv[valid, 0], uv[valid, 1]

        if u.max() < 0 or u.min() >= w or v.max() < 0 or v.min() >= h:
            continue

        name = obj.get("subcategory") or obj.get("category") or "object"
        color = category_color_bgr(obj.get("category"), obj.get("subcategory"))
        x0 = float(np.clip(u.min(), 0, w - 1))
        y0 = float(np.clip(v.min(), 0, h - 1))
        text_box(image, name, (x0, y0 - 2), color=color, scale=0.36)


def render_overlay_panel(rgb_full, annotation, lidar_xyz_ego, calibration, camera_name, lidar_overlay):
    title = "RGB Left + bbox/class" + (" + LiDAR" if lidar_overlay else "")

    if rgb_full is None:
        return put_title(blank_panel("RGB missing"), title), None

    image = resize_to_panel(rgb_full)

    if calibration is None or camera_name not in calibration.get("cameras", {}):
        panel = letterbox(image)
        center_text(panel, "Calibration unavailable", PANEL_H // 2, 0.8, (60, 60, 255), 2)
        return put_title(panel, title), None

    projector = scaled_projector(calibration, camera_name, image.shape[1] / rgb_full.shape[1])

    lidar_in_image = None

    if lidar_overlay and lidar_xyz_ego is not None:
        lidar_in_image = project_lidar_on_image(image, projector, lidar_xyz_ego)

    stats = None

    if annotation is not None:
        # 3rd value (selected_projections) is only populated when logical_ids is given.
        image, stats, _selected = render_rgb_projection(image, annotation, projector)
        draw_class_labels(image, annotation, projector)

    panel = letterbox(image)

    notes = []

    if annotation is None:
        notes.append("labels missing")

    if lidar_overlay and lidar_xyz_ego is None:
        notes.append("lidar missing")
    elif lidar_in_image is not None:
        notes.append(f"lidar px={lidar_in_image}")

    if notes:
        text_box(panel, "  ".join(notes), (6, PANEL_H - 8), color=TEXT_DIM, scale=0.38)

    return put_title(panel, title, f"LiDAR 0-{LIDAR_OVERLAY_MAX_DIST_M:.0f}m" if lidar_overlay else None), stats


def render_depth(path, max_depth):
    title = "Depth"

    if not os.path.isfile(path):
        return put_title(blank_panel("depth missing"), title)

    depth = np.load(path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth = resize_to_panel(depth)
    u8 = (np.clip(depth, 0.0, max_depth) / max_depth * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    panel = letterbox(color)

    # Fixed-scale colorbar (same meters on every frame).
    bar_h, bar_w = PANEL_H - 70, 12
    x0, y0 = PANEL_W - bar_w - 10, 34
    gradient = np.linspace(0, 255, bar_h, dtype=np.uint8).reshape(-1, 1).repeat(bar_w, axis=1)
    panel[y0:y0 + bar_h, x0:x0 + bar_w] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    text_box(panel, "0m", (x0 - 30, y0 + 10), scale=0.36)
    text_box(panel, f"{max_depth:.0f}m", (x0 - 42, y0 + bar_h), scale=0.36)

    return put_title(panel, title, f"fixed 0-{max_depth:.0f} m")


def render_semantic(path, lut):
    title = "Semantic Segmentation"

    if not os.path.isfile(path):
        return put_title(blank_panel("semantic missing"), title)

    ids = np.load(path)

    if ids.ndim == 3:
        ids = ids[:, :, 0]

    ids = resize_to_panel(ids.astype(np.uint8), interpolation=cv2.INTER_NEAREST)

    return put_title(letterbox(lut[ids]), title, "CARLA palette")


def render_flow(path, max_flow):
    title = "Optical Flow"

    if not os.path.isfile(path):
        return put_title(blank_panel("optical flow missing"), title)

    # CARLA flow is normalized to image size, so resizing the field does not
    # change its values.
    flow = resize_to_panel(load_flow(path))
    panel = letterbox(flow_to_color_bright(flow, max_flow=max_flow))
    draw_flow_legend(panel, size=48, margin=30)

    return put_title(panel, title, f"hue=dir, sat=|f|/{max_flow:g}")


def bev_canvas(roi):
    width, height = bev_frame_size(roi, PANEL_H - 26)

    if width > PANEL_W:
        height = int(round(height * PANEL_W / width))
        width = PANEL_W

    canvas = np.full((height, width, 3), (48, 48, 48), dtype=np.uint8)
    draw_bev_grid(canvas, roi, width, height)

    return canvas


def bev_scatter(canvas, roi, xyz_ego, colors, offsets=((0, 0),)):
    x, y = xyz_ego[:, 0], xyz_ego[:, 1]
    keep = (x >= roi["x_min"]) & (x <= roi["x_max"]) & (np.abs(y) <= roi["y_half"])
    px, py = bev_to_px(x[keep], y[keep], roi, canvas.shape[1], canvas.shape[0])
    paint_pixels(canvas, px, py, colors[keep], offsets)

    return int(keep.sum())


def bev_panel(canvas):
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    h, w = canvas.shape[:2]
    y0 = 26 + (PANEL_H - 26 - h) // 2
    x0 = (PANEL_W - w) // 2
    panel[y0:y0 + h, x0:x0 + w] = canvas

    return panel, x0, y0


def render_lidar_bev_panel(lidar_xyz_ego, lidar_source, annotation):
    title = "LiDAR BEV"

    if lidar_xyz_ego is None:
        return put_title(blank_panel("lidar missing"), title)

    roi = LIDAR_ROI
    canvas = bev_canvas(roi)
    colors = colormap_colors(lidar_xyz_ego[:, 2], *LIDAR_BEV_Z_RANGE)
    bev_scatter(canvas, roi, lidar_xyz_ego, colors)

    for obj in (annotation or {}).get("objects", []):
        draw_bev_bbox(canvas, obj, roi)

    panel, x0, _ = bev_panel(canvas)
    lines = [
        f"pts={len(lidar_xyz_ego)}",
        f"color=z {LIDAR_BEV_Z_RANGE[0]:+.1f}..{LIDAR_BEV_Z_RANGE[1]:+.1f}m",
        f"extr={lidar_source}",
    ]

    for i, line in enumerate(lines):
        text_box(panel, line, (x0 + canvas.shape[1] + 6, 50 + 18 * i), color=TEXT_DIM, scale=0.36)

    x_span = f"x {roi['x_min']:.0f}..{roi['x_max']:.0f}m, y +-{roi['y_half']:.0f}m"

    return put_title(panel, title, x_span)


def render_radar_bev_panel(radars, annotation, velocity_mode):
    """radars: {sensor_name: (xyz_ego or None, velocity or None, extr_source)}"""

    title = "Radar BEV (Front / FL / FR)"
    roi = RADAR_ROI
    canvas = bev_canvas(roi)

    # Sensor mount + boresight straight from each sensor's extrinsic, so a
    # wrong extrinsic is visible even on an empty frame.
    for name, (_, _, extrinsic, _) in radars.items():
        if extrinsic is None:
            continue

        radar_cfg = getattr(cfg.SENSOR, RADAR_CFG_ATTR[name])
        origin = extrinsic[:3, 3]

        for angle_deg in (-radar_cfg.HORIZONTAL_FOV / 2.0, 0.0, radar_cfg.HORIZONTAL_FOV / 2.0):
            a = np.radians(angle_deg)
            direction = extrinsic[:3, :3] @ np.array([np.cos(a), np.sin(a), 0.0])
            tip = origin + direction * radar_cfg.RANGE
            p0 = bev_to_px(origin[0], origin[1], roi, canvas.shape[1], canvas.shape[0])
            p1 = bev_to_px(tip[0], tip[1], roi, canvas.shape[1], canvas.shape[0])
            color = tuple(int(c * (0.9 if angle_deg == 0.0 else 0.45)) for c in RADAR_SENSOR_COLOR_BGR[name])
            cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, 1, cv2.LINE_AA)

    for obj in (annotation or {}).get("objects", []):
        draw_bev_bbox(canvas, obj, roi)

    counts = {}

    for name, (xyz, velocity, _, _) in radars.items():
        if xyz is None:
            counts[name] = None
            continue

        counts[name] = len(xyz)

        if velocity_mode:
            colors = np.array([radar_point_color(v) for v in velocity], dtype=np.uint8).reshape(-1, 3)
        else:
            colors = np.tile(np.array(RADAR_SENSOR_COLOR_BGR[name], dtype=np.uint8), (len(xyz), 1))

        bev_scatter(canvas, roi, xyz, colors, RADAR_MARKER_OFFSETS[name])

    panel, x0, y0 = bev_panel(canvas)

    # Legend (sensor color + marker shape, drawn 2x size), top-left of the BEV.
    legend_x, legend_y = x0 + 4, y0 + 4
    cv2.rectangle(panel, (legend_x, legend_y), (legend_x + 128, legend_y + 20 * len(RADAR_SENSOR_NAMES) + 4), (0, 0, 0), -1)

    for i, name in enumerate(RADAR_SENSOR_NAMES):
        y = legend_y + 14 + 20 * i
        color = RADAR_SENSOR_COLOR_BGR[name]
        swatch = np.array([color if not velocity_mode else TEXT_WHITE], dtype=np.uint8)
        big_marker = [(2 * dx + ex, 2 * dy + ey) for dx, dy in RADAR_MARKER_OFFSETS[name] for ex in (0, 1) for ey in (0, 1)]
        paint_pixels(panel, np.array([legend_x + 8]), np.array([y - 4]), swatch, big_marker)
        count = counts.get(name)
        label = f"{RADAR_DISPLAY_NAMES[name]}: {count if count is not None else 'N/A'}"
        cv2.putText(panel, label, (legend_x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    footer = "color=radial vel (red=approach, blue=recede)" if velocity_mode else "color=sensor"
    sources = sorted({src for _, _, _, src in radars.values() if src})
    text_box(panel, f"{footer}  extr={'/'.join(sources) or 'N/A'}", (x0 + 4, PANEL_H - 6), color=TEXT_DIM, scale=0.34)

    x_span = f"x {roi['x_min']:.0f}..{roi['x_max']:.0f}m, y +-{roi['y_half']:.0f}m"

    return put_title(panel, title, x_span), counts


def render_status_panel(lines):
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    y = 44

    for key, value in lines:
        if key is None:
            y += 6
            continue

        cv2.putText(panel, f"{key}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{value}", (190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_WHITE, 1, cv2.LINE_AA)
        y += 19

    return put_title(panel, "Status / Metadata")


def weather_summary(meta):
    if not meta:
        return "N/A"

    params = meta.get("weather_parameters") or {}
    keys = (("precipitation", "rain"), ("fog_density", "fog"), ("sun_altitude_angle", "sun"))
    parts = [f"{label}={params[k]:.0f}" for k, label in keys if isinstance(params.get(k), (int, float))]

    return ", ".join(parts) if parts else "N/A"


def fmt(value, spec, suffix=""):
    return "N/A" if value is None else f"{value:{spec}}{suffix}"


# ------------------------------------------------------------------
# Multimodal frame
# ------------------------------------------------------------------

class MultimodalRenderer:
    def __init__(self, route, condition, available, args):
        self.route = route
        self.condition = condition
        self.available = available
        self.args = args
        self.dirs = route.modality_dirs(condition)
        self.lut = build_semantic_lut(load_semantic_map())
        self.condition_meta = route.condition_meta(condition)

        if self.condition_meta is None:
            warn(f"conditions/{condition}/condition.json missing -- weather shown as N/A")

        self.lidar_extrinsic, self.lidar_source = sensor_extrinsic(route.calibration, "lidar")
        self.radar_extrinsics = {name: sensor_extrinsic(route.calibration, name) for name in RADAR_SENSOR_NAMES}

    def path(self, name, frame_id):
        directory, ext = self.dirs[name]

        return frame_file(directory, frame_id, ext)

    def load_lidar(self, frame_id):
        path = self.path("lidar", frame_id)

        if not os.path.isfile(path):
            return None, None

        raw = np.load(path).reshape(-1, 4)
        xyz, _ = lidar_points_to_ego(raw, self.lidar_extrinsic)

        return xyz, len(raw)

    def load_radars(self, frame_id):
        radars = {}

        for name in RADAR_SENSOR_NAMES:
            extrinsic, source = self.radar_extrinsics[name]
            path = self.path(name, frame_id)

            if not os.path.isfile(path) or extrinsic is None:
                radars[name] = (None, None, extrinsic, source)
                continue

            raw = np.load(path).reshape(-1, 4)
            # Each radar goes to ego with its OWN extrinsic before merging.
            xyz, velocity = radar_points_to_ego(raw, extrinsic)
            radars[name] = (xyz, velocity, extrinsic, source)

        return radars

    def render(self, frame_id):
        annotation = read_json(self.path("labels", frame_id)) if "labels" in self.available else None
        lidar_xyz, lidar_count = self.load_lidar(frame_id)
        radars = self.load_radars(frame_id)

        left_panel, left_full = render_rgb_panel(self.path("rgb_left", frame_id), f"RGB Left ({self.condition})")
        right_panel, _ = render_rgb_panel(self.path("rgb_right", frame_id), f"RGB Right ({self.condition})")

        overlay_panel, overlay_stats = render_overlay_panel(
            left_full, annotation, lidar_xyz, self.route.calibration, "rgb_left", not self.args.no_lidar_overlay,
        )

        depth_panel = render_depth(self.path("depth", frame_id), self.args.max_depth)
        semantic_panel = render_semantic(self.path("semantic", frame_id), self.lut)
        flow_panel = render_flow(self.path("optical_flow", frame_id), self.args.max_flow)
        lidar_panel = render_lidar_bev_panel(lidar_xyz, self.lidar_source, annotation)
        radar_panel, radar_counts = render_radar_bev_panel(radars, annotation, self.args.radar_velocity)

        status_panel = render_status_panel(self.status_lines(frame_id, annotation, overlay_stats, lidar_count, radar_counts))

        rows = [
            [left_panel, right_panel, overlay_panel],
            [depth_panel, semantic_panel, flow_panel],
            [lidar_panel, radar_panel, status_panel],
        ]

        return np.vstack([np.hstack(row) for row in rows])

    def status_lines(self, frame_id, annotation, overlay_stats, lidar_count, radar_counts):
        ts_row = self.route.timestamps.get(frame_id, {})
        ego_row = self.route.ego_state.get(frame_id, {})

        speed = as_float(ego_row.get("speed_mps"))
        progress = as_float(ego_row.get("route_progress"))
        goal = as_float(ego_row.get("goal_distance"))
        timestamp = as_float(ts_row.get("timestamp"))

        if annotation is not None:
            label_count = len(annotation.get("objects", []))
            rejected = len(annotation.get("rejected_objects", []))
        else:
            label_count = rejected = None

        visible = overlay_stats["projectable"] if overlay_stats else None
        radar_values = [radar_counts.get(n) for n in RADAR_SENSOR_NAMES]
        radar_total = sum(v for v in radar_values if v is not None) if any(v is not None for v in radar_values) else None

        return [
            ("Town", self.route.town),
            ("Route", self.route.route_name),
            ("Condition", self.condition),
            ("Frame index", f"{frame_id:06d}"),
            ("CARLA frame", ts_row.get("carla_frame") or "N/A"),
            ("Timestamp", fmt(timestamp, ".2f", " s")),
            ("Ego speed", "N/A" if speed is None else f"{speed * 3.6:.1f} km/h"),
            ("Route progress", fmt(progress, ".1f", " %") + ("" if goal is None else f"  (goal {goal:.0f} m)")),
            ("Weather", weather_summary(self.condition_meta)),
            (None, None),
            ("Labels (camera-valid)", fmt(label_count, "d") + ("" if rejected is None else f"  (rejected {rejected})")),
            ("Visible in RGB", fmt(visible, "d")),
            ("LiDAR points", fmt(lidar_count, "d")),
            ("Radar Front", fmt(radar_counts.get("radar"), "d")),
            ("Radar Front Left", fmt(radar_counts.get("radar_front_left"), "d")),
            ("Radar Front Right", fmt(radar_counts.get("radar_front_right"), "d")),
            ("Radar total", fmt(radar_total, "d")),
        ]


# ------------------------------------------------------------------
# Weather comparison frame
# ------------------------------------------------------------------

class WeatherRenderer:
    def __init__(self, route, present_conditions):
        self.route = route
        self.present = set(present_conditions)
        self.metas = {c: route.condition_meta(c) for row in WEATHER_GRID for c in row}

        report = route.paired_validation

        if report is None:
            self.validation_text = "paired_validation: N/A"
        else:
            verdict = "PASS" if report.get("passed") else "FAIL"
            self.validation_text = (
                f"paired_validation: {verdict} ({report.get('num_frames', 'N/A')} frames, "
                f"{len(report.get('errors') or [])} errors)"
            )

    def condition_tag(self, condition):
        meta = self.metas.get(condition)

        if not meta:
            return None

        if meta.get("rendered_from") == "canonical_run":
            return "canonical"

        passed = (meta.get("replay_validation") or {}).get("passed")

        return "replay OK" if passed is True else ("replay FAIL" if passed is False else "replay N/A")

    def render(self, frame_id):
        rows = []

        for row in WEATHER_GRID:
            panels = []

            for condition in row:
                path = frame_file(os.path.join(self.route.condition_path(condition), "rgb_left"), frame_id, ".png")
                image = cv2.imread(path, cv2.IMREAD_COLOR) if condition in self.present else None

                if image is None:
                    panel = blank_panel("unavailable", condition)
                else:
                    panel = letterbox(resize_to_panel(image))

                panels.append(put_title(panel, condition, self.condition_tag(condition)))

            rows.append(np.hstack(panels))

        grid = np.vstack(rows)
        header = f"{self.route.town} / {self.route.route_name}  |  frame {frame_id:06d}  |  {self.validation_text}"
        text_box(grid, header, (8, 46), scale=0.46)

        return grid


# ------------------------------------------------------------------
# Video writing
# ------------------------------------------------------------------

class VideoWriter:
    """OpenCV mp4v by default; H.264 via an ffmpeg pipe when requested and available."""

    def __init__(self, path, fps, size, codec):
        self.path = path
        self.size = size
        self.proc = None
        self.writer = None

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if codec == "h264":
            ffmpeg = shutil.which("ffmpeg")

            if ffmpeg:
                self.proc = subprocess.Popen(
                    [
                        ffmpeg, "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}",
                        "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        path,
                    ],
                    stdin=subprocess.PIPE,
                )
                return

            warn("ffmpeg not found on PATH -- falling back to OpenCV mp4v")

        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)

        if not self.writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {path}")

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()

            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        elif self.writer is not None:
            self.writer.release()


def progress(iterable, description):
    if tqdm is not None:
        return tqdm(iterable, desc=description, unit="frame")

    items = list(iterable)
    step = max(1, len(items) // 20)

    def generator():
        for i, item in enumerate(items):
            if i % step == 0:
                print(f"[QA] {description}: {i}/{len(items)}")
            yield item

        print(f"[QA] {description}: {len(items)}/{len(items)}")

    return generator()


def write_video(path, frame_ids, render_fn, fps, codec, description):
    first = render_fn(frame_ids[0])
    size = (first.shape[1], first.shape[0])
    writer = VideoWriter(path, fps, size, codec)

    try:
        writer.write(first)

        for frame_id in progress(frame_ids[1:], description):
            writer.write(render_fn(frame_id))
    finally:
        writer.close()

    info(f"wrote {path}  ({len(frame_ids)} frames, {size[0]}x{size[1]}, {fps:g} fps)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline QA videos (multimodal 3x3 + weather 2x3) for one collected route directory.",
    )
    parser.add_argument("--route-root", required=True, help=r"e.g. D:\carla_dataset_final_v2\Town07\route_0")
    parser.add_argument("--condition", default=cfg.WEATHER.DEFAULT, help="condition for the multimodal video (default: day_clear)")
    parser.add_argument("--fps", type=float, default=None, help="output fps (default: sequence.json recording_hz, else 10)")
    parser.add_argument("--stride", type=int, default=1, help="use every N-th frame")
    parser.add_argument("--start-frame", type=int, default=None, help="first frame id (inclusive)")
    parser.add_argument("--end-frame", type=int, default=None, help="last frame id (inclusive)")
    parser.add_argument("--output-dir", default=None, help="default: outputs/qa_videos/<Town>/<route>/")
    parser.add_argument("--videos", choices=("all", "multimodal", "weather"), default="all")
    parser.add_argument("--strict-weather", action="store_true", help="error if any of the 6 conditions lacks a frame")
    parser.add_argument("--no-lidar-overlay", action="store_true", help="no LiDAR points on the RGB+label panel")
    parser.add_argument("--radar-velocity", action="store_true", help="color radar points by radial velocity (marker shape = sensor)")
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH_M, help="fixed depth colormap range [m]")
    parser.add_argument("--max-flow", type=float, default=DEFAULT_BRIGHT_MAX_FLOW, help="fixed optical-flow magnitude clip")
    parser.add_argument("--codec", choices=("mp4v", "h264"), default="mp4v", help="h264 needs ffmpeg on PATH")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    route = RouteData(args.route_root)

    if args.fps is not None:
        fps, fps_source = args.fps, "--fps"
    else:
        fps, fps_source = route.recording_fps()

    output_dir = os.path.abspath(
        args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "qa_videos", route.town, route.route_name)
    )

    info(f"route      : {route.route_root}")
    info(f"town/route : {route.town} / {route.route_name}")
    info(f"fps        : {fps:g} ({fps_source})")
    info(f"output dir : {output_dir}")

    if args.videos in ("all", "multimodal"):
        frames, available = discover_multimodal_frames(route, args.condition)
        frames = apply_frame_range(frames, args.start_frame, args.end_frame, args.stride)

        if not frames:
            warn("multimodal: no frames left after range/stride -- skipped")
        else:
            renderer = MultimodalRenderer(route, args.condition, available, args)
            write_video(
                os.path.join(output_dir, f"multimodal_{args.condition}.mp4"),
                frames, renderer.render, fps, args.codec, "multimodal",
            )

    if args.videos in ("all", "weather"):
        conditions = [c for row in WEATHER_GRID for c in row]
        frames, present = discover_weather_frames(route, conditions, args.strict_weather)
        frames = apply_frame_range(frames, args.start_frame, args.end_frame, args.stride)

        if not frames:
            warn("weather: no frames left after range/stride -- skipped")
        else:
            renderer = WeatherRenderer(route, present)
            write_video(
                os.path.join(output_dir, "weather_comparison.mp4"),
                frames, renderer.render, fps, args.codec, "weather",
            )


if __name__ == "__main__":
    main()

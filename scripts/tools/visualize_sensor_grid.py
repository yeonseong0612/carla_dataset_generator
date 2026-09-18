"""
scripts/tools/visualize_sensor_grid.py

2x3 synchronized multi-sensor validation grid:

    RGB              | Depth Map  | Optical Flow
    Semantic Segment. | LiDAR BEV  | Multi-Radar BEV

This tool ONLY reads files already on disk for one synchronized sample
(rgb_left/*.png, depth/*.npy, optical_flow/*.npy, semantic/*.npy,
lidar/*.npy, radar/radar_front_left/radar_front_right/*.npy,
labels/object_3d/*.json, calibration.json). It never re-queries the CARLA
simulator and never modifies any dataset file. It is visualization-only:
no sensor configuration is read or changed here.

Dataset schema note: the front radar directory is "radar" (not
"radar_front") -- verified against src/data/collector.py's
Collector.RADAR_SENSORS, not assumed.

Reuse
-----
- RGB projection / LiDAR / Radar BEV building blocks (CameraProjector,
  render_rgb_projection, render_lidar_bev, render_radar_bev,
  load_all_radars_ego, ROI, add_title_strip, ...) come from
  visualize_annotations.py / visualize_multisensor.py -- not
  reimplemented here.
- Optical flow HSV visualization (load_flow / flow_to_color) comes from
  visualize_flow.py.
- Semantic segmentation colors come from src/data/annotation.py's
  SEMANTIC_MAP (the project's CARLA semantic palette), so a class ID
  always maps to the same color as elsewhere in this project.

GT bbox overlays (RGB / LiDAR / Radar) are OFF by default -- this tool is
for inspecting raw sensor output -- and can be enabled with --bbox.
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

from scripts.tools.visualize_annotations import (
    CameraProjector,
    discover_available_frames,
    load_annotation,
    load_calibration,
    load_rgb,
    pick_samples,
    render_rgb_projection,
)

from scripts.tools.visualize_multisensor import (
    MERGED_RADAR_KEY,
    RADAR_SENSOR_NAMES,
    ROI,
    add_title_strip,
    bev_frame_size,
    get_extrinsic,
    lidar_points_to_ego,
    load_all_radars_ego,
    load_lidar,
    parse_sequence_title,
    render_lidar_bev,
    render_radar_bev,
)

from scripts.tools.visualize_flow import (
    DEFAULT_BRIGHT_MAX_FLOW,
    draw_flow_legend,
    flow_magnitude_stats,
    flow_to_color,
    flow_to_color_middlebury,
    load_flow,
)

from src.data.annotation import SEMANTIC_MAP

EMPTY_ANNOTATION = {"objects": []}


# ------------------------------------------------------------------
# IO / sync check
# ------------------------------------------------------------------

def load_depth(sequence_root, frame_id):
    path = os.path.join(sequence_root, "depth", f"{frame_id:06d}.npy")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    return np.load(path)


def load_semantic(sequence_root, frame_id):
    path = os.path.join(sequence_root, "semantic", f"{frame_id:06d}.npy")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    return np.load(path)


def flow_path(sequence_root, frame_id):
    return os.path.join(sequence_root, "optical_flow", f"{frame_id:06d}.npy")


def check_frame_files(sequence_root, frame_id, camera_name, require_labels):
    frame_name = f"{frame_id:06d}"

    required = {
        camera_name: os.path.join(sequence_root, camera_name, f"{frame_name}.png"),
        "depth": os.path.join(sequence_root, "depth", f"{frame_name}.npy"),
        "optical_flow": os.path.join(sequence_root, "optical_flow", f"{frame_name}.npy"),
        "semantic": os.path.join(sequence_root, "semantic", f"{frame_name}.npy"),
        "lidar": os.path.join(sequence_root, "lidar", f"{frame_name}.npy"),
    }

    for sensor_name in RADAR_SENSOR_NAMES:
        required[sensor_name] = os.path.join(sequence_root, sensor_name, f"{frame_name}.npy")

    if require_labels:
        required["labels/object_3d"] = os.path.join(sequence_root, "labels", "object_3d", f"{frame_name}.json")

    missing = [name for name, path in required.items() if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            f"Frame {frame_name}: synchronized 6-sensor sample is missing "
            f"required file(s) {missing} under {sequence_root}"
        )


# ------------------------------------------------------------------
# Panel 2 -- Depth Map
# ------------------------------------------------------------------

def render_depth_panel(depth_m, max_depth=100.0):
    # Guard against any NaN/Inf so a bad pixel can't blow out the whole
    # color scale; raw depth_m itself is never modified/re-saved.
    depth_clean = np.nan_to_num(depth_m, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth_clipped = np.clip(depth_clean, 0.0, max_depth)

    depth_u8 = (depth_clipped / max_depth * 255.0).astype(np.uint8)

    colormap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    color = cv2.applyColorMap(depth_u8, colormap)

    height = color.shape[0]
    bar_width = 24

    # 0m (near) at the top, max_depth (far) at the bottom.
    gradient = np.linspace(0, 255, height, dtype=np.uint8).reshape(height, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    bar = cv2.applyColorMap(gradient, colormap)

    divider = np.full((height, 4, 3), (40, 40, 40), dtype=np.uint8)
    color = np.hstack([color, divider, bar])

    cv2.putText(color, "0m", (color.shape[1] - bar_width - 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(color, f"{max_depth:.0f}m", (color.shape[1] - bar_width - 2, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(color, "Depth [m]", (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return color


# ------------------------------------------------------------------
# Panel 4 -- Semantic Segmentation
# ------------------------------------------------------------------

_SEMANTIC_LUT = None


def semantic_lut():
    global _SEMANTIC_LUT

    if _SEMANTIC_LUT is None:
        max_tag = max(SEMANTIC_MAP.keys())
        lut = np.zeros((max_tag + 1, 3), dtype=np.uint8)

        for tag, (_name, rgb) in SEMANTIC_MAP.items():
            lut[tag] = rgb

        _SEMANTIC_LUT = lut

    return _SEMANTIC_LUT


def render_semantic_panel(semantic_ids):
    lut = semantic_lut()

    ids = np.clip(semantic_ids, 0, lut.shape[0] - 1).astype(np.int32)
    rgb = lut[ids]

    # SEMANTIC_MAP colors are RGB; cv2 canvases are BGR.
    bgr = rgb[:, :, ::-1].copy()

    return bgr


# ------------------------------------------------------------------
# Grid composition (2 rows x 3 cols; compose_panels in
# visualize_multisensor.py only builds a single row)
# ------------------------------------------------------------------

def build_row(panels):
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

    return row


def compose_grid(rows, outer_title):
    row_images = [build_row(row) for row in rows]

    max_w = max(image.shape[1] for image in row_images)

    padded = []

    for image in row_images:
        if image.shape[1] != max_w:
            # Center rather than left-align: rows built from
            # differently-shaped panels (wide camera images vs. tall BEV
            # plots) end up with different natural widths, and centering
            # reads as an intentional layout rather than a stray gap.
            extra = max_w - image.shape[1]
            left = extra // 2
            right = extra - left

            image = cv2.copyMakeBorder(
                image, 0, 0, left, right,
                cv2.BORDER_CONSTANT, value=(10, 10, 10),
            )

        padded.append(image)

    divider = np.full((4, max_w, 3), (255, 255, 255), dtype=np.uint8)

    grid = padded[0]

    for image in padded[1:]:
        grid = np.vstack([grid, divider, image])

    title_bar = np.full((44, max_w, 3), (10, 10, 10), dtype=np.uint8)
    cv2.putText(title_bar, outer_title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([title_bar, grid])


# ------------------------------------------------------------------
# Console statistics (PART 15 -- console only, keeps the figure clean)
# ------------------------------------------------------------------

def print_frame_stats(frame_id, depth_m, flow, semantic_ids, lidar_count, merged_radar_count):
    finite_depth = depth_m[np.isfinite(depth_m)]

    if finite_depth.size:
        depth_min = float(finite_depth.min())
        depth_max = float(finite_depth.max())
        depth_median = float(np.median(finite_depth))
    else:
        depth_min = depth_max = depth_median = float("nan")

    flow_stats = flow_magnitude_stats(flow)

    unique_classes = int(np.unique(semantic_ids).size)

    print(
        f"  frame {frame_id:06d}: "
        f"depth[min={depth_min:.1f} max={depth_max:.1f} median={depth_median:.1f}]m "
        f"flow_mag[mean={flow_stats['mean']:.4f} median={flow_stats['p50']:.4f} "
        f"p90={flow_stats['p90']:.4f} p95={flow_stats['p95']:.4f} "
        f"p99={flow_stats['p99']:.4f} max={flow_stats['max']:.4f}] "
        f"semantic_classes={unique_classes} "
        f"lidar_pts={lidar_count} radar_merged_pts={merged_radar_count}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2x3 grid: RGB | Depth | Optical Flow / Semantic | LiDAR BEV | Multi-Radar BEV."
    )
    parser.add_argument("--sequence", type=str, required=True, help="Path to a sequence root, e.g. outputs/town10hd_multiradar_density_validation/Town10/route_0/day_clear")
    parser.add_argument("--camera", type=str, default="rgb_left", help="Camera name as used in calibration.json / directory name")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Explicit list of frame ids to render")
    parser.add_argument("--num-samples", type=int, default=None, help="Evenly-spaced number of frames to sample across the sequence")
    parser.add_argument("--output-dir", type=str, default="outputs/sensor_grid_validation", help="Output directory for rendered grid images")
    parser.add_argument("--panel-height", type=int, default=480, help="Content height in pixels shared by all 6 panels")
    parser.add_argument("--max-depth", type=float, default=100.0, help="Depth colormap clip range in meters")
    parser.add_argument("--flow-max", type=float, default=DEFAULT_BRIGHT_MAX_FLOW, help="Fixed flow-magnitude clip for the Middlebury/RAFT-style Optical Flow panel (measured from representative frames; see visualize_flow.flow_magnitude_stats)")
    parser.add_argument("--flow-percentile", type=float, default=None, help="Use each frame's own magnitude percentile as the clip instead of --flow-max (debug option, less reproducible frame-to-frame)")
    parser.add_argument("--flow-legacy", action="store_true", help="Use the original magnitude-to-brightness flow visualization instead of the Middlebury/RAFT color wheel")
    parser.add_argument("--flow-legend", action="store_true", help="Draw a small direction color-wheel legend on the Optical Flow panel")
    parser.add_argument("--bbox", action="store_true", help="Overlay GT annotation boxes on the RGB/LiDAR/Radar panels (off by default -- this tool is for raw sensor QA)")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)
    output_dir = os.path.abspath(args.output_dir)
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

    sample_frames = pick_samples(available, args.frames, args.num_samples)

    if not sample_frames:
        print("No frames selected to render.")
        sys.exit(1)

    print(f"Rendering {len(sample_frames)} frame(s) from {sequence_root} -> {output_dir}")

    width_px, height_px = bev_frame_size(ROI, args.panel_height)

    for frame_id in sample_frames:
        check_frame_files(sequence_root, frame_id, args.camera, require_labels=args.bbox)

        annotation = load_annotation(sequence_root, frame_id) if args.bbox else EMPTY_ANNOTATION

        # --- Panel 1: RGB -------------------------------------------------
        rgb_image = load_rgb(sequence_root, frame_id, args.camera)

        if args.bbox:
            rgb_panel, _rgb_stats = render_rgb_projection(rgb_image, annotation, projector)
        else:
            rgb_panel = rgb_image

        # --- Panel 2: Depth -------------------------------------------------
        depth_m = load_depth(sequence_root, frame_id)
        depth_panel = render_depth_panel(depth_m, max_depth=args.max_depth)

        # --- Panel 3: Optical Flow ------------------------------------------
        flow = load_flow(flow_path(sequence_root, frame_id))

        if args.flow_legacy:
            flow_panel = flow_to_color(flow)
        else:
            flow_panel = flow_to_color_middlebury(flow, max_flow=args.flow_max, percentile=args.flow_percentile)

        if args.flow_legend:
            flow_panel = draw_flow_legend(flow_panel)

        # --- Panel 4: Semantic -----------------------------------------------
        semantic_ids = load_semantic(sequence_root, frame_id)
        semantic_panel = render_semantic_panel(semantic_ids)

        # --- Panel 5: LiDAR BEV ------------------------------------------------
        lidar_arr = load_lidar(sequence_root, frame_id)
        lidar_xyz_ego, lidar_intensity = lidar_points_to_ego(lidar_arr, T_ego_from_lidar)
        lidar_panel = render_lidar_bev(lidar_xyz_ego, lidar_intensity, ROI, width_px, height_px, annotation)

        # --- Panel 6: Multi-Radar BEV -------------------------------------------
        radar_by_source = load_all_radars_ego(sequence_root, frame_id, calibration)
        merged_xyz, merged_vel = radar_by_source[MERGED_RADAR_KEY]
        radar_panel = render_radar_bev(
            merged_xyz, merged_vel, ROI, width_px, height_px, annotation,
            source_label="Front+FL+FR merged",
        )

        title = parse_sequence_title(sequence_root, frame_id)

        grid = compose_grid(
            [
                [("RGB", rgb_panel), ("Depth Map", depth_panel), ("Optical Flow", flow_panel)],
                [("Semantic Segmentation", semantic_panel), ("LiDAR BEV", lidar_panel), ("Multi-Radar BEV", radar_panel)],
            ],
            title,
        )

        out_path = os.path.join(output_dir, f"{frame_id:06d}.png")
        cv2.imwrite(out_path, grid)

        print_frame_stats(
            frame_id, depth_m, flow, semantic_ids,
            lidar_count=lidar_arr.shape[0],
            merged_radar_count=merged_xyz.shape[0],
        )
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()

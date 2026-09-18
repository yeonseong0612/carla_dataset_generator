"""
scripts/tools/validate_annotations.py

Quantitative sanity validation for generated 3D object annotation JSON
files (labels/object_3d/*.json). This tool ONLY reads files already on
disk -- it never queries the CARLA simulator and never modifies the
dataset.

Usage
-----
    python -m scripts.tools.validate_annotations \
        --sequence dataset/Town01/route_0/day_clear

    python -m scripts.tools.validate_annotations \
        --sequence dataset/Town01/route_0/day_clear \
        --start 0 --end 299

    python -m scripts.tools.validate_annotations \
        --sequence dataset/Town01/route_0/day_clear \
        --frames 0 150 299
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.tools.bbox_geometry import BoxGeometryError, local_box_axes

ALLOWED_CATEGORIES = {"vehicle", "pedestrian", "cyclist", "motorcyclist"}
ALLOWED_VEHICLE_SUBTYPES = {"car", "van", "truck", "bus"}

# ------------------------------------------------------------------
# Tolerances
# ------------------------------------------------------------------

DISTANCE_REL_TOL = 1e-3
DISTANCE_ABS_TOL = 1e-2  # meters

VERTEX_MEAN_TOL_M = 0.10       # vertices centroid vs center_ego_m
Z_RANGE_TOL_M = 0.10           # vertex z-span vs height
EDGE_LENGTH_REL_TOL = 0.05     # 5% relative
EDGE_LENGTH_ABS_TOL = 0.03     # meters

MAX_SANE_DIMENSION_M = 30.0    # anything bigger is not a real vehicle/ped
MAX_SANE_COORD_M = 1000.0      # abs() ceiling for any single coordinate

FRAME_FILE_RE = re.compile(r"^(\d{6})\.json$")


class Issue:
    def __init__(self, severity, frame_id, actor_id, code, message):
        self.severity = severity  # "error" | "warning"
        self.frame_id = frame_id
        self.actor_id = actor_id
        self.code = code
        self.message = message

    def format(self):
        actor_part = f" actor_id={self.actor_id}" if self.actor_id is not None else ""
        frame_part = f"frame={self.frame_id}" if self.frame_id is not None else "frame=?"
        return f"[{self.severity.upper()}] {frame_part}{actor_part} ({self.code}): {self.message}"


def is_finite_vec(value, n):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return False
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return arr.shape == (n,) and np.all(np.isfinite(arr))


def discover_frame_files(sequence_root, start, end, frames):
    object_dir = os.path.join(sequence_root, "labels", "object_3d")

    if not os.path.isdir(object_dir):
        raise FileNotFoundError(f"labels/object_3d directory not found under {sequence_root}")

    all_files = sorted(glob.glob(os.path.join(object_dir, "*.json")))

    if frames is not None:
        wanted = {f"{f:06d}.json" for f in frames}
        selected = [p for p in all_files if os.path.basename(p) in wanted]
        return selected, object_dir

    if start is not None or end is not None:
        lo = start if start is not None else 0
        hi = end if end is not None else 10 ** 6

        selected = []
        for p in all_files:
            m = FRAME_FILE_RE.match(os.path.basename(p))
            if not m:
                continue
            idx = int(m.group(1))
            if lo <= idx <= hi:
                selected.append(p)
        return selected, object_dir

    return all_files, object_dir


def validate_frame(path, issues):
    """
    Validates a single frame JSON file. Returns (frame_id, objects_checked,
    per-object stats) or (None, 0, []) if the frame could not be parsed at
    all.
    """

    basename = os.path.basename(path)
    m = FRAME_FILE_RE.match(basename)
    expected_frame_id = int(m.group(1)) if m else None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(Issue("error", expected_frame_id, None, "json_parse_failed", f"{basename}: {exc}"))
        return None, 0, []

    frame_id = data.get("frame_id")

    if frame_id is None:
        issues.append(Issue("error", expected_frame_id, None, "missing_frame_id", f"{basename}: 'frame_id' missing"))
    elif expected_frame_id is not None and frame_id != expected_frame_id:
        issues.append(Issue(
            "error", expected_frame_id, None, "frame_id_mismatch",
            f"{basename}: frame_id={frame_id} does not match filename",
        ))

    if "carla_frame" not in data:
        issues.append(Issue("error", frame_id, None, "missing_carla_frame", f"{basename}: 'carla_frame' missing"))

    if "timestamp" not in data:
        issues.append(Issue("error", frame_id, None, "missing_timestamp", f"{basename}: 'timestamp' missing"))

    objects = data.get("objects")

    if not isinstance(objects, list):
        issues.append(Issue("error", frame_id, None, "objects_not_list", f"{basename}: 'objects' is not a list"))
        return frame_id, 0, []

    stats = []

    for obj in objects:
        stat = validate_object(frame_id, obj, issues)
        if stat is not None:
            stats.append(stat)

    return frame_id, len(objects), stats


def validate_object(frame_id, obj, issues):
    def err(code, message):
        issues.append(Issue("error", frame_id, actor_id, code, message))

    def warn(code, message):
        issues.append(Issue("warning", frame_id, actor_id, code, message))

    actor_id = obj.get("actor_id")

    if actor_id is None:
        issues.append(Issue("error", frame_id, None, "missing_actor_id", "object has no 'actor_id'"))
        # Cannot reliably continue tagging further issues to an actor id.

    category = obj.get("category")
    subcategory = obj.get("subcategory")

    if category not in ALLOWED_CATEGORIES:
        err("invalid_category", f"category={category!r} not in {sorted(ALLOWED_CATEGORIES)}")

    if category == "vehicle":
        if subcategory not in ALLOWED_VEHICLE_SUBTYPES:
            err("invalid_vehicle_subcategory", f"subcategory={subcategory!r} not in {sorted(ALLOWED_VEHICLE_SUBTYPES)}")

    bbox = obj.get("bbox_3d")

    if not isinstance(bbox, dict):
        err("missing_bbox_3d", "object has no 'bbox_3d' dict")
        return None

    center = bbox.get("center_ego_m")
    dims = bbox.get("dimensions_m")
    yaw = bbox.get("yaw_ego_deg")
    vertices = bbox.get("vertices_ego_m")

    center_ok = is_finite_vec(center, 3)
    if not center_ok:
        err("bad_center", f"center_ego_m invalid or non-finite: {center}")

    rel_vel = obj.get("relative_velocity_ego_mps")
    if not is_finite_vec(rel_vel, 3):
        err("bad_relative_velocity", f"relative_velocity_ego_mps invalid or non-finite: {rel_vel}")

    distance_m = obj.get("distance_m")
    distance_ok = isinstance(distance_m, (int, float)) and math.isfinite(distance_m) and distance_m >= 0
    if not distance_ok:
        err("bad_distance", f"distance_m invalid: {distance_m}")

    if center_ok and distance_ok:
        expected = float(np.linalg.norm(np.asarray(center, dtype=np.float64)))
        tol = DISTANCE_ABS_TOL + DISTANCE_REL_TOL * expected
        if abs(expected - distance_m) > tol:
            err(
                "distance_center_mismatch",
                f"distance_m={distance_m:.4f} but norm(center_ego_m)={expected:.4f} (tol={tol:.4f})",
            )

    yaw_ok = isinstance(yaw, (int, float)) and math.isfinite(yaw)
    if not yaw_ok:
        err("bad_yaw", f"yaw_ego_deg invalid: {yaw}")
    elif not (-180.0 <= yaw < 180.0):
        err("yaw_out_of_range", f"yaw_ego_deg={yaw} not in [-180, 180)")

    dims_ok = False
    length = width = height = None

    if isinstance(dims, dict):
        length = dims.get("length")
        width = dims.get("width")
        height = dims.get("height")

        dims_ok = all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0
            for x in (length, width, height)
        )

        if not dims_ok:
            err("bad_dimensions", f"dimensions_m invalid: {dims}")
        elif max(length, width, height) > MAX_SANE_DIMENSION_M:
            err("dimension_too_large", f"dimensions_m={dims} exceeds sanity ceiling {MAX_SANE_DIMENSION_M}m")
    else:
        err("missing_dimensions", "bbox_3d.dimensions_m missing or not a dict")

    vertices_ok = isinstance(vertices, list) and len(vertices) == 8

    if not vertices_ok:
        err("bad_vertex_count", f"vertices_ego_m has {len(vertices) if isinstance(vertices, list) else 'N/A'} entries, expected 8")
    else:
        for i, v in enumerate(vertices):
            if not is_finite_vec(v, 3):
                err("bad_vertex", f"vertex[{i}]={v} is not a finite 3D point")
                vertices_ok = False

    if vertices_ok:
        varr = np.asarray(vertices, dtype=np.float64)

        if np.any(np.abs(varr) > MAX_SANE_COORD_M):
            err("vertex_coord_absurd", f"a vertex coordinate exceeds {MAX_SANE_COORD_M}m in magnitude")

        # --- vertex mean vs stored center -----------------------------
        if center_ok:
            vmean = varr.mean(axis=0)
            c = np.asarray(center, dtype=np.float64)
            offset = float(np.linalg.norm(vmean - c))
            if offset > VERTEX_MEAN_TOL_M:
                err(
                    "vertex_mean_center_mismatch",
                    f"mean(vertices_ego_m)={vmean.tolist()} vs center_ego_m={center} "
                    f"(offset={offset:.3f}m, tol={VERTEX_MEAN_TOL_M}m)",
                )

        # --- vertex z range vs height -----------------------------------
        if dims_ok:
            z_span = float(varr[:, 2].max() - varr[:, 2].min())
            if abs(z_span - height) > Z_RANGE_TOL_M:
                err(
                    "z_range_height_mismatch",
                    f"vertex z-span={z_span:.3f}m vs height={height:.3f}m (tol={Z_RANGE_TOL_M}m)",
                )

        # --- local box axis extents vs length/width/height (order-free) -
        if dims_ok:
            try:
                _, _, extents, _ = local_box_axes(varr)
                edge_lengths = sorted((2.0 * extents).tolist())
                expected_lengths = sorted([length, width, height])

                for got, exp in zip(edge_lengths, expected_lengths):
                    tol = EDGE_LENGTH_ABS_TOL + EDGE_LENGTH_REL_TOL * exp
                    if abs(got - exp) > tol:
                        err(
                            "edge_length_mismatch",
                            f"local box edge lengths {edge_lengths} do not match "
                            f"dimensions {expected_lengths} within tol",
                        )
                        break
            except BoxGeometryError as exc:
                err("degenerate_box", f"vertices do not form a valid rectangular box: {exc}")

    return {
        "actor_id": actor_id,
        "category": category,
        "subcategory": subcategory,
        "distance_m": distance_m if distance_ok else None,
        "length": length if dims_ok else None,
        "width": width if dims_ok else None,
        "height": height if dims_ok else None,
    }


def check_frame_continuity(frame_ids, issues):
    present = sorted(f for f in frame_ids if f is not None)

    if not present:
        return

    expected = set(range(present[0], present[-1] + 1))
    missing = sorted(expected - set(present))

    seen = set()
    dupes = sorted({f for f in present if (f in seen or seen.add(f))})

    if missing:
        issues.append(Issue(
            "error", None, None, "non_contiguous_frames",
            f"missing frame ids in range [{present[0]}, {present[-1]}]: {missing[:50]}"
            + (" ..." if len(missing) > 50 else ""),
        ))

    if dupes:
        issues.append(Issue("error", None, None, "duplicate_frame_ids", f"duplicate frame ids: {dupes}"))


def summarize(frames_checked, objects_checked, issues, object_stats):
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    print()
    print("=" * 70)
    print("Annotation validation summary")
    print("=" * 70)
    print(f"Frames checked: {frames_checked}")
    print(f"Objects checked: {objects_checked}")
    print()
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print()
        print("Error breakdown by code:")
        by_code = {}
        for i in errors:
            by_code[i.code] = by_code.get(i.code, 0) + 1
        for code, count in sorted(by_code.items(), key=lambda kv: -kv[1]):
            print(f"  {code}: {count}")

    if warnings:
        print()
        print("Warning breakdown by code:")
        by_code = {}
        for i in warnings:
            by_code[i.code] = by_code.get(i.code, 0) + 1
        for code, count in sorted(by_code.items(), key=lambda kv: -kv[1]):
            print(f"  {code}: {count}")

    counts = {"vehicle": 0, "pedestrian": 0, "cyclist": 0, "motorcyclist": 0}
    vehicle_sub_counts = {"car": 0, "van": 0, "truck": 0, "bus": 0}

    distances = []
    lengths = []
    widths = []
    heights = []

    for s in object_stats:
        if s["category"] in counts:
            counts[s["category"]] += 1
        if s["category"] == "vehicle" and s["subcategory"] in vehicle_sub_counts:
            vehicle_sub_counts[s["subcategory"]] += 1
        if s["distance_m"] is not None:
            distances.append(s["distance_m"])
        if s["length"] is not None:
            lengths.append(s["length"])
        if s["width"] is not None:
            widths.append(s["width"])
        if s["height"] is not None:
            heights.append(s["height"])

    print()
    print("Class counts:")
    print(f"  vehicle: {counts['vehicle']}  "
          f"(car: {vehicle_sub_counts['car']}, van: {vehicle_sub_counts['van']}, "
          f"truck: {vehicle_sub_counts['truck']}, bus: {vehicle_sub_counts['bus']})")
    print(f"  pedestrian: {counts['pedestrian']}")
    print(f"  cyclist: {counts['cyclist']}")
    print(f"  motorcyclist: {counts['motorcyclist']}")

    print()
    if distances:
        print(f"Distance range: [{min(distances):.2f}, {max(distances):.2f}] m")
    else:
        print("Distance range: N/A")

    print("Dimension range:")
    if lengths:
        print(f"  length: [{min(lengths):.2f}, {max(lengths):.2f}] m")
    if widths:
        print(f"  width:  [{min(widths):.2f}, {max(widths):.2f}] m")
    if heights:
        print(f"  height: [{min(heights):.2f}, {max(heights):.2f}] m")

    print("=" * 70)

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate CARLA object_3d annotation JSON files.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to a sequence root, e.g. dataset/Town01/route_0/day_clear")
    parser.add_argument("--start", type=int, default=None, help="First frame id (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last frame id (inclusive)")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Explicit list of frame ids to check")
    parser.add_argument("--max-print", type=int, default=80, help="Maximum number of individual issues to print")
    parser.add_argument("--verbose", action="store_true", help="Print every issue (overrides --max-print)")
    args = parser.parse_args()

    sequence_root = os.path.abspath(args.sequence)

    files, object_dir = discover_frame_files(sequence_root, args.start, args.end, args.frames)

    if not files:
        print(f"No annotation JSON files found under {object_dir} for the requested range.")
        sys.exit(1)

    print(f"Validating {len(files)} annotation file(s) in: {object_dir}")

    issues = []
    frame_ids = []
    objects_checked = 0
    object_stats = []

    for path in files:
        frame_id, n_objects, stats = validate_frame(path, issues)
        frame_ids.append(frame_id)
        objects_checked += n_objects
        object_stats.extend(stats)

    if args.frames is None:
        # Continuity only makes sense when checking a contiguous
        # range/whole sequence; an explicit --frames subset is
        # intentionally sparse.
        check_frame_continuity(frame_ids, issues)

    to_print = issues if args.verbose else issues[: args.max_print]

    for issue in to_print:
        print(issue.format())

    if not args.verbose and len(issues) > args.max_print:
        print(f"... ({len(issues) - args.max_print} more issues suppressed, use --verbose to see all)")

    errors, warnings = summarize(len(files), objects_checked, issues, object_stats)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

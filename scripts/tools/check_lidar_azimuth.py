"""
check_lidar_azimuth.py

Checks whether every saved CARLA LiDAR frame covers the intended
front 180-degree field of view, or whether the angular sector moves /
alternates between frames.

Expected dataset:
    <route_root>/
        geometry/
            lidar/
                000000.npy
                000001.npy
                ...

Each LiDAR file:
    float32 Nx4 [x, y, z, intensity]

CARLA sensor-frame convention assumed:
    +x : forward
    +y : right
    +z : up

For the intended front-facing 180 deg LiDAR:
    expected azimuth coverage ~= [-90 deg, +90 deg]
    on EVERY saved frame.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_ROUTE_ROOT = Path(
    r"E:\Dataset\carla\doptimal_dataset_v1\seed_101\Town10\route_0"
)


def circular_coverage_deg(angles_deg):
    """
    Estimate the smallest circular arc containing all points.

    Returns:
        coverage_deg
        arc_start_deg
        arc_end_deg
        largest_empty_gap_deg
    """
    if len(angles_deg) < 2:
        return np.nan, np.nan, np.nan, np.nan

    a = np.mod(np.asarray(angles_deg, dtype=np.float64), 360.0)
    a.sort()

    wrapped = np.concatenate([a, [a[0] + 360.0]])
    gaps = np.diff(wrapped)

    gap_idx = int(np.argmax(gaps))
    largest_gap = float(gaps[gap_idx])

    coverage = 360.0 - largest_gap

    # Occupied arc starts immediately after largest empty gap.
    start = wrapped[gap_idx + 1] % 360.0
    end = wrapped[gap_idx] % 360.0

    def to_pm180(v):
        return ((v + 180.0) % 360.0) - 180.0

    return (
        float(coverage),
        float(to_pm180(start)),
        float(to_pm180(end)),
        largest_gap,
    )


def longest_missing_run(occupied):
    """Longest consecutive run of False bins."""
    if len(occupied) == 0:
        return 0

    # We intentionally do NOT wrap here because front [-90,+90]
    # has meaningful fixed boundaries.
    best = 0
    current = 0

    for value in occupied:
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)

    return best


def analyze_frame(path, front_bins, min_points_per_bin):
    points = np.load(path)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"{path}: expected Nx4-ish LiDAR array, got {points.shape}"
        )

    xyz = np.asarray(points[:, :3], dtype=np.float64)

    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]

    if len(xyz) == 0:
        raise ValueError(f"{path}: no finite LiDAR points")

    x = xyz[:, 0]
    y = xyz[:, 1]

    az = np.degrees(np.arctan2(y, x))

    # Overall angular sector actually present in this callback.
    coverage, arc_start, arc_end, largest_gap = circular_coverage_deg(az)

    # Intended front FOV = [-90, +90].
    front_mask = (az >= -90.0) & (az <= 90.0)
    front_az = az[front_mask]

    hist, edges = np.histogram(front_az, bins=front_bins)

    occupied = hist >= min_points_per_bin

    occupied_fraction = float(np.mean(occupied))
    longest_missing = longest_missing_run(occupied)

    bin_width = float(front_bins[1] - front_bins[0])
    longest_missing_deg = longest_missing * bin_width

    left_count = int(np.sum((az >= -90.0) & (az < 0.0)))
    right_count = int(np.sum((az >= 0.0) & (az <= 90.0)))

    front_count = left_count + right_count
    total_count = len(az)

    if front_count:
        left_fraction = left_count / front_count
        right_fraction = right_count / front_count
    else:
        left_fraction = np.nan
        right_fraction = np.nan

    return {
        "frame": int(path.stem),
        "points": total_count,

        "az_min": float(np.min(az)),
        "az_max": float(np.max(az)),

        "circular_coverage_deg": coverage,
        "arc_start_deg": arc_start,
        "arc_end_deg": arc_end,
        "largest_empty_gap_deg": largest_gap,

        "front_points": front_count,
        "front_fraction": front_count / total_count,

        "left_points": left_count,
        "right_points": right_count,
        "left_fraction": left_fraction,
        "right_fraction": right_fraction,

        "front_bin_occupancy": occupied_fraction,
        "longest_missing_front_deg": longest_missing_deg,

        "hist": hist,
    }


def classify_frame(row, min_front_occupancy, max_missing_deg):
    """
    Simple diagnostic classification.

    GOOD:
        front angular bins broadly populated.

    PARTIAL:
        large angular hole exists inside intended front 180 deg.

    EMPTY:
        almost no front points.
    """
    if row["front_points"] == 0:
        return "EMPTY"

    if (
        row["front_bin_occupancy"] < min_front_occupancy
        or row["longest_missing_front_deg"] > max_missing_deg
    ):
        return "PARTIAL"

    return "GOOD"


def detect_left_right_alternation(rows):
    """
    Detect a suspicious L/R/L/R pattern.

    Uses normalized left-right imbalance:
        +1 -> almost only left
        -1 -> almost only right
    """
    vals = []

    for row in rows:
        total = row["left_points"] + row["right_points"]

        if total == 0:
            vals.append(np.nan)
            continue

        imbalance = (
            row["left_points"] - row["right_points"]
        ) / total

        vals.append(imbalance)

    vals = np.asarray(vals, dtype=np.float64)

    valid = np.isfinite(vals)

    if valid.sum() < 4:
        return None

    v = vals[valid]

    # Adjacent sign flips among strongly one-sided frames.
    strong = np.abs(v) >= 0.35

    if strong.sum() < 4:
        return {
            "strong_frames": int(strong.sum()),
            "alternation_ratio": 0.0,
            "mean_abs_imbalance": float(np.mean(np.abs(v))),
        }

    sv = np.sign(v[strong])

    flips = np.sum(sv[1:] != sv[:-1])
    ratio = flips / max(1, len(sv) - 1)

    return {
        "strong_frames": int(strong.sum()),
        "alternation_ratio": float(ratio),
        "mean_abs_imbalance": float(np.mean(np.abs(v))),
    }


def print_histogram(row, front_bins):
    hist = row["hist"]

    if hist.max() > 0:
        norm = hist / hist.max()
    else:
        norm = hist.astype(float)

    chars = []

    for value in norm:
        if value == 0:
            chars.append(" ")
        elif value < 0.20:
            chars.append("░")
        elif value < 0.50:
            chars.append("▒")
        elif value < 0.80:
            chars.append("▓")
        else:
            chars.append("█")

    print(
        f"        -90° |{''.join(chars)}| +90°"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check per-frame CARLA LiDAR azimuth coverage."
    )

    parser.add_argument(
        "--route-root",
        type=Path,
        default=DEFAULT_ROUTE_ROOT,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Number of frames to inspect. Use 0 for all.",
    )

    parser.add_argument(
        "--bin-deg",
        type=float,
        default=5.0,
        help="Angular histogram resolution.",
    )

    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=1,
        help="Minimum points for an angular bin to count as occupied.",
    )

    parser.add_argument(
        "--min-front-occupancy",
        type=float,
        default=0.70,
        help="Minimum occupied fraction of front [-90,+90] bins.",
    )

    parser.add_argument(
        "--max-missing-deg",
        type=float,
        default=30.0,
        help="Maximum allowed consecutive empty angular span in front FOV.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV.",
    )

    args = parser.parse_args()

    lidar_dir = args.route_root / "geometry" / "lidar"

    if not lidar_dir.is_dir():
        raise SystemExit(
            f"LiDAR directory not found:\n{lidar_dir}"
        )

    files = sorted(
        p for p in lidar_dir.glob("*.npy")
        if p.stem.isdigit()
    )

    if args.start_frame is not None:
        files = [
            p for p in files
            if int(p.stem) >= args.start_frame
        ]

    if args.end_frame is not None:
        files = [
            p for p in files
            if int(p.stem) <= args.end_frame
        ]

    if args.max_frames > 0:
        files = files[:args.max_frames]

    if not files:
        raise SystemExit("No LiDAR frames found.")

    front_bins = np.arange(
        -90.0,
        90.0 + args.bin_deg,
        args.bin_deg,
    )

    rows = []

    print("=" * 78)
    print("LiDAR angular coverage diagnostic")
    print("=" * 78)
    print(f"Route       : {args.route_root}")
    print(f"LiDAR dir   : {lidar_dir}")
    print(f"Frames      : {len(files)}")
    print(f"Front FOV   : -90 .. +90 deg")
    print(f"Bin size    : {args.bin_deg:g} deg")
    print()

    for idx, path in enumerate(files):
        row = analyze_frame(
            path,
            front_bins,
            args.min_points_per_bin,
        )

        row["status"] = classify_frame(
            row,
            args.min_front_occupancy,
            args.max_missing_deg,
        )

        rows.append(row)

        if idx % args.print_every == 0:
            print(
                f"[{row['frame']:06d}] "
                f"{row['status']:7s} "
                f"N={row['points']:5d}  "
                f"coverage={row['circular_coverage_deg']:6.1f}°  "
                f"frontBins={100*row['front_bin_occupancy']:5.1f}%  "
                f"missingMax={row['longest_missing_front_deg']:5.1f}°  "
                f"L/R={row['left_points']:4d}/{row['right_points']:4d} "
                f"({row['left_fraction']:.2f}/{row['right_fraction']:.2f})"
            )

            print_histogram(row, front_bins)

    # ------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------

    statuses = [r["status"] for r in rows]

    good = statuses.count("GOOD")
    partial = statuses.count("PARTIAL")
    empty = statuses.count("EMPTY")

    occupancies = np.array(
        [r["front_bin_occupancy"] for r in rows],
        dtype=float,
    )

    coverages = np.array(
        [r["circular_coverage_deg"] for r in rows],
        dtype=float,
    )

    missing = np.array(
        [r["longest_missing_front_deg"] for r in rows],
        dtype=float,
    )

    alternation = detect_left_right_alternation(rows)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"GOOD frames    : {good}/{len(rows)}")
    print(f"PARTIAL frames : {partial}/{len(rows)}")
    print(f"EMPTY frames   : {empty}/{len(rows)}")

    print(
        "Front occupancy: "
        f"mean={np.nanmean(occupancies)*100:.1f}%  "
        f"min={np.nanmin(occupancies)*100:.1f}%  "
        f"max={np.nanmax(occupancies)*100:.1f}%"
    )

    print(
        "Angular coverage: "
        f"mean={np.nanmean(coverages):.1f}°  "
        f"min={np.nanmin(coverages):.1f}°  "
        f"max={np.nanmax(coverages):.1f}°"
    )

    print(
        "Largest missing FRONT sector: "
        f"mean={np.nanmean(missing):.1f}°  "
        f"max={np.nanmax(missing):.1f}°"
    )

    if alternation:
        print(
            "Left/right diagnostic: "
            f"strong_frames={alternation['strong_frames']}  "
            f"alternation_ratio={alternation['alternation_ratio']:.3f}  "
            f"mean_abs_imbalance={alternation['mean_abs_imbalance']:.3f}"
        )

    print()
    print("Interpretation:")

    if partial == 0 and empty == 0:
        print(
            "  PASS-like result: every inspected frame has broad angular "
            "coverage across the intended front FOV."
        )
    else:
        print(
            "  WARNING: some frames contain large missing angular sectors."
        )
        print(
            "  If consecutive frames alternate left/right coverage, this "
            "suggests the saved callback contains only part of the intended "
            "front 180-degree scan."
        )

    if alternation and alternation["alternation_ratio"] >= 0.60:
        print(
            "  SUSPICIOUS: strong left/right alternation detected."
        )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "frame",
            "status",
            "points",
            "az_min",
            "az_max",
            "circular_coverage_deg",
            "arc_start_deg",
            "arc_end_deg",
            "largest_empty_gap_deg",
            "front_points",
            "front_fraction",
            "left_points",
            "right_points",
            "left_fraction",
            "right_fraction",
            "front_bin_occupancy",
            "longest_missing_front_deg",
        ]

        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {k: row[k] for k in fields}
                )

        print()
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
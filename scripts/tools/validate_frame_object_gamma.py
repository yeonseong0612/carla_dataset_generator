"""
scripts/tools/validate_frame_object_gamma.py

Live-CARLA validation for the Frame-Level Gamma Object Count Policy
(src/simulation/canonical_traffic.py FrameObjectGammaSchedule / CLAUDE.md
task "Frame-Level Gamma Object Count Policy").

Step 1 -- runs an actual collect_dataset.py collection (default: Town10,
route 0, day_clear, cfg.SPAWN.SEED, --max-frames 600, canonical policy)
as a subprocess. A 600-frame Town10 route very likely will NOT reach
route completion, which makes collect_dataset.py's collect_sequence()
raise "Maximum frame count reached before route completion" internally
-- this is EXPECTED (see the prior 600-frame canonical-policy baseline
noted in analyze_canonical_policy.py) and does not affect this script:
collect_dataset.py's own `finally` block still writes/closes every
per-frame artifact this script needs (frame_object_counts.csv,
ego_state.csv, labels/object_3d/*.json, canonical_spawn_summary.json,
rgb_left/*.png) regardless, and its per-sequence exception is caught
inside collect_dataset.py's own main() (exit code stays 0).

Step 2 -- read-only analysis of what got written:
  outputs/frame_object_gamma_validation/
      frame_object_counts.csv     (copied from the sequence's own CSV --
                                    written live, frame by frame, by
                                    collect_dataset.py itself)
      object_count_histogram.png  (actual histogram vs. target integer-
                                    Gamma PMF, bins 0..18)
      target_vs_actual.png        (time series, control-lag visibility)
      density_summary.txt         (CLAUDE.md task sections 16-20's numbers)
      visualizations/NNNNNN.png   (6 representative rgb_left frames with
                                    "Actual objects = N" / "Target = M")
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
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
from src.simulation.spawn_policy import gamma_cdf_numeric  # noqa: E402
from scripts.tools.analyze_canonical_policy import (  # noqa: E402
    load_all_annotations,
    nearest_same_lane_lead_per_frame,
    load_ego_state,
    load_poses,
    rebuild_dense_route,
    final_route_s_from_poses,
)

REPRESENTATIVE_FRAMES = [50, 150, 250, 350, 450, 550]
MAX_BIN = 18

BUCKETS = [
    ("0-3", 0, 3),
    ("4-6", 4, 6),
    ("7-10", 7, 10),
    ("11-14", 11, 14),
    ("15+", 15, None),
]


# ------------------------------------------------------------------
# Step 1: run the actual collection
# ------------------------------------------------------------------

def run_collection(town, route_id, condition, max_frames, output_root, overwrite):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "collect_dataset.py"),
        "--towns", town,
        "--routes", route_id,
        "--conditions", condition,
        "--max-frames", str(max_frames),
        "--output-root", output_root,
        "--background-policy", "canonical",
    ]

    if overwrite:
        cmd.append("--overwrite")

    print("[Run] " + " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"collect_dataset.py exited with code {result.returncode}")


def sequence_root(output_root, town, route_id, condition):
    return os.path.join(output_root, town, f"route_{route_id}", condition)


# ------------------------------------------------------------------
# Step 2: frame_object_counts.csv -> stats
# ------------------------------------------------------------------

def load_frame_object_counts(sequence_dir):
    path = os.path.join(sequence_dir, "frame_object_counts.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- was this sequence collected with "
            f"--background-policy canonical?"
        )

    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame_id": int(row["frame_id"]),
                "target": int(row["target_object_count"]),
                "actual": int(row["actual_object_count"]),
            })

    return rows


def percentile(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def mode_of(values):
    values = np.asarray(values)
    counts = np.bincount(values)

    return int(np.argmax(counts))


def bucket_ratios(values):
    n = len(values)
    ratios = {}

    for name, lo, hi in BUCKETS:
        if hi is None:
            count = sum(1 for v in values if v >= lo)
        else:
            count = sum(1 for v in values if lo <= v <= hi)

        ratios[name] = (count, count / n if n else 0.0)

    return ratios


# ------------------------------------------------------------------
# Target integer-Gamma PMF, bins 0..MAX_BIN (CLAUDE.md task section 14)
# ------------------------------------------------------------------

def target_integer_pmf(shape, scale, min_count, max_count, max_bin=MAX_BIN):
    """
    P(round(Gamma(shape,scale)) == k) for min_count <= k <= max_count,
    with sub-min mass folded into the min_count bin and super-max mass
    folded into the max_count bin -- exactly mirroring how
    FrameObjectGammaSchedule turns a raw Gamma draw into a clipped
    integer target (round then clip). Bins outside [min_count, max_count]
    are 0 (nothing clips there). Sums to exactly 1.0 over 0..max_bin.
    """

    pmf = np.zeros(max_bin + 1, dtype=np.float64)

    def cdf(x):
        return gamma_cdf_numeric(x, shape, scale) if x > 0 else 0.0

    for k in range(min_count, max_count + 1):
        lo_cdf = 0.0 if k == min_count else cdf(k - 0.5)
        hi_cdf = 1.0 if k == max_count else cdf(k + 0.5)
        pmf[k] = hi_cdf - lo_cdf

    return pmf


def jensen_shannon_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ------------------------------------------------------------------
# Plots (cv2, project convention -- no matplotlib; see
# analyze_canonical_policy.py's render_histogram for the same style)
# ------------------------------------------------------------------

def render_count_histogram(actual_counts, target_counts, out_path, width=1100, height=650):
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 70, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    n_bins = len(actual_counts)
    max_value = max(float(actual_counts.max()), float(target_counts.max()), 1.0) * 1.15

    def y_px(value):
        return margin_top + int(plot_h * (1.0 - value / max_value))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), (0, 0, 0), 1)

    group_w = plot_w / n_bins
    bar_w = group_w * 0.35

    for i in range(n_bins):
        group_center = margin_left + group_w * (i + 0.5)

        actual_x = int(group_center - bar_w * 0.6)
        target_x = int(group_center + bar_w * 0.1)

        actual_top = y_px(actual_counts[i])
        target_top = y_px(target_counts[i])
        base_y = margin_top + plot_h

        cv2.rectangle(canvas, (actual_x, actual_top), (int(actual_x + bar_w), base_y), (60, 60, 220), -1)
        cv2.rectangle(canvas, (target_x, target_top), (int(target_x + bar_w), base_y), (60, 180, 60), 2)

        cv2.putText(canvas, str(i), (int(group_center - 6), margin_top + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_value * frac
        y = y_px(value)
        cv2.line(canvas, (margin_left - 4, y), (margin_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (width - 260, 20), (width - 244, 34), (60, 60, 220), -1)
    cv2.putText(canvas, "Actual (observed)", (width - 236, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 260, 40), (width - 244, 54), (60, 180, 60), 2)
    cv2.putText(canvas, "Gamma target (integer PMF x N frames)", (width - 236, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(
        canvas, "objects/frame histogram: actual vs. frame-object Gamma target",
        (margin_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(canvas, "objects in frame", (margin_left + plot_w // 2 - 60, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame count", (10, margin_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def render_target_vs_actual(rows, out_path, width=1400, height=500):
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 50, 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    n = len(rows)
    max_value = max(max(r["target"] for r in rows), max(r["actual"] for r in rows), 1) * 1.15

    def x_px(i):
        return margin_left + int(plot_w * i / max(n - 1, 1))

    def y_px(value):
        return margin_top + int(plot_h * (1.0 - value / max_value))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), (0, 0, 0), 1)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_value * frac
        y = y_px(value)
        cv2.line(canvas, (margin_left - 4, y), (margin_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.0f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    target_pts = np.array([[x_px(i), y_px(r["target"])] for i, r in enumerate(rows)], dtype=np.int32)
    actual_pts = np.array([[x_px(i), y_px(r["actual"])] for i, r in enumerate(rows)], dtype=np.int32)

    cv2.polylines(canvas, [target_pts], False, (60, 180, 60), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [actual_pts], False, (60, 60, 220), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (width - 220, 15), (width - 204, 27), (60, 180, 60), -1)
    cv2.putText(canvas, "Gamma target", (width - 196, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 220, 32), (width - 204, 44), (60, 60, 220), -1)
    cv2.putText(canvas, "Actual", (width - 196, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "target vs. actual objects/frame over time", (margin_left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame_id", (margin_left + plot_w // 2 - 30, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


# ------------------------------------------------------------------
# Representative frame visualizations
# ------------------------------------------------------------------

def render_representative_frames(sequence_dir, rows_by_frame, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgb_dir = os.path.join(sequence_dir, "rgb_left")

    written = []

    for frame_id in REPRESENTATIVE_FRAMES:
        image_path = os.path.join(rgb_dir, f"{frame_id:06d}.png")

        if not os.path.isfile(image_path) or frame_id not in rows_by_frame:
            print(f"[Visualize] skip frame {frame_id}: missing image or count row")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"[Visualize] skip frame {frame_id}: failed to read {image_path}")
            continue

        row = rows_by_frame[frame_id]

        cv2.putText(image, f"Actual objects = {row['actual']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Target = {row['target']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"{frame_id:06d}.png")
        cv2.imwrite(out_path, image)
        written.append(out_path)

    return written


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", type=str, default="Town10")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--collection-output-root", type=str, default=os.path.join(cfg.PROJECT.ROOT, "dataset"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(cfg.PROJECT.ROOT, "outputs", "frame_object_gamma_validation"))
    parser.add_argument("--skip-collection", action="store_true", help="Reuse an already-collected sequence instead of re-running collect_dataset.py.")
    parser.add_argument("--skip-route-projection", action="store_true", help="Skip the live-CARLA final route_s projection.")
    parser.add_argument("--no-overwrite", action="store_true", help="Don't pass --overwrite to collect_dataset.py.")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_collection:
        run_collection(
            args.town, args.route_id, args.condition, args.max_frames,
            args.collection_output_root, overwrite=not args.no_overwrite,
        )

    seq_dir = sequence_root(args.collection_output_root, args.town, args.route_id, args.condition)

    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    # -------------------------------------------------------------
    # frame_object_counts.csv -> outputs/frame_object_gamma_validation/
    # -------------------------------------------------------------

    rows = load_frame_object_counts(seq_dir)
    shutil.copyfile(
        os.path.join(seq_dir, "frame_object_counts.csv"),
        os.path.join(output_dir, "frame_object_counts.csv"),
    )

    rows_by_frame = {r["frame_id"]: r for r in rows}
    n_frames = len(rows)
    actual_values = [r["actual"] for r in rows]
    target_values = [r["target"] for r in rows]

    # -------------------------------------------------------------
    # Section 16 stats
    # -------------------------------------------------------------

    mean_v = float(np.mean(actual_values))
    median_v = float(np.median(actual_values))
    mode_v = mode_of(actual_values)
    std_v = float(np.std(actual_values))
    min_v = int(min(actual_values))
    max_v = int(max(actual_values))
    p10, p25, p75, p90 = (percentile(actual_values, p) for p in (10, 25, 75, 90))
    ratios = bucket_ratios(actual_values)
    top_bucket = max(ratios.items(), key=lambda kv: kv[1][1])

    # -------------------------------------------------------------
    # Section 17: target integer-Gamma PMF vs actual histogram
    # -------------------------------------------------------------

    shape = cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE
    scale = cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE
    min_count = cfg.SPAWN.FRAME_OBJECT_MIN
    max_count = cfg.SPAWN.FRAME_OBJECT_MAX

    target_pmf = target_integer_pmf(shape, scale, min_count, max_count, MAX_BIN)
    target_counts_scaled = target_pmf * n_frames

    actual_counts = np.zeros(MAX_BIN + 1, dtype=int)
    clipped_above_max_bin = 0

    for v in actual_values:
        bin_index = min(v, MAX_BIN)

        if v > MAX_BIN:
            clipped_above_max_bin += 1

        actual_counts[bin_index] += 1

    mae = float(np.mean(np.abs(actual_counts - target_counts_scaled)))
    rmse = float(np.sqrt(np.mean((actual_counts - target_counts_scaled) ** 2)))

    actual_pmf = actual_counts / max(actual_counts.sum(), 1)
    js_divergence = jensen_shannon_divergence(actual_pmf, target_pmf)

    render_count_histogram(actual_counts, target_counts_scaled, os.path.join(output_dir, "object_count_histogram.png"))

    # -------------------------------------------------------------
    # Section 15: target vs actual over time
    # -------------------------------------------------------------

    render_target_vs_actual(rows, os.path.join(output_dir, "target_vs_actual.png"))

    # -------------------------------------------------------------
    # Control lag: mean |actual - target|, and frame-level agreement
    # -------------------------------------------------------------

    diffs = [r["actual"] - r["target"] for r in rows]
    mean_abs_lag = float(np.mean(np.abs(diffs)))
    over_target_frames = sum(1 for d in diffs if d > 0)
    under_target_frames = sum(1 for d in diffs if d < 0)
    within_1 = sum(1 for d in diffs if abs(d) <= 1)
    within_2 = sum(1 for d in diffs if abs(d) <= 2)

    # -------------------------------------------------------------
    # Section 18: spawn/despawn stats
    # -------------------------------------------------------------

    spawn_summary_path = os.path.join(seq_dir, "canonical_spawn_summary.json")
    spawn_summary = {}

    if os.path.isfile(spawn_summary_path):
        with open(spawn_summary_path, "r", encoding="utf-8") as f:
            spawn_summary = json.load(f)

    # -------------------------------------------------------------
    # Section 19: ego mobility
    # -------------------------------------------------------------

    ego_state_rows = load_ego_state(seq_dir)
    speeds_kmh = [float(row["speed_mps"]) * 3.6 for row in ego_state_rows]

    if args.skip_route_projection:
        final_route_s = float("nan")
    else:
        poses = load_poses(seq_dir)
        dense_route = rebuild_dense_route(cfg.CARLA.HOST, cfg.CARLA.PORT, args.town, args.route_id)
        final_route_s = final_route_s_from_poses(poses, dense_route)

    # -------------------------------------------------------------
    # Section 20: same-lane congestion
    # -------------------------------------------------------------

    frames = load_all_annotations(seq_dir)
    leads = nearest_same_lane_lead_per_frame(frames)

    # -------------------------------------------------------------
    # Representative frame visualizations
    # -------------------------------------------------------------

    viz_paths = render_representative_frames(seq_dir, rows_by_frame, os.path.join(output_dir, "visualizations"))

    # -------------------------------------------------------------
    # density_summary.txt
    # -------------------------------------------------------------

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    emit("=" * 78)
    emit("Frame-Level Gamma Object Count Policy -- validation")
    emit(f"sequence: {seq_dir}")
    emit(f"frames analyzed: {n_frames}")
    emit("=" * 78)

    emit()
    emit("1. What Gamma is applied to: N_objects(frame), NOT object distance.")
    emit(f"2. shape={shape} scale={scale} min={min_count} max={max_count} "
         f"(segment length U[{cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MIN},"
         f"{cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MAX}] frames, "
         f"max_new/update={cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE})")
    emit(f"3. theoretical mode = (shape-1)*scale = {(shape - 1) * scale:.2f}   "
         f"theoretical mean = shape*scale = {shape * scale:.2f}")
    emit(f"4. actual objects/frame: mode={mode_v} mean={mean_v:.2f} median={median_v:.2f} "
         f"std={std_v:.2f} min={min_v} max={max_v}")
    emit(f"   p10={p10:.2f} p25={p25:.2f} p75={p75:.2f} p90={p90:.2f}")
    emit()
    emit("   bucket ratios (actual objects/frame):")
    for name, (count, ratio) in ratios.items():
        emit(f"     {name:>6s}: {count:4d} frames ({ratio * 100:5.1f}%)")
    emit(f"   highest-ratio bucket: {top_bucket[0]} ({top_bucket[1][1] * 100:.1f}%) "
         f"-> {'PASS' if top_bucket[0] == '7-10' else 'FAIL'} (expected 7-10)")
    emit(f"5. 7-10 objects/frame ratio: {ratios['7-10'][1] * 100:.1f}% "
         f"({ratios['7-10'][0]}/{n_frames} frames)")

    emit()
    emit(f"6. target-histogram vs actual-histogram (bins 0..{MAX_BIN}):")
    emit(f"   MAE={mae:.3f}  RMSE={rmse:.3f}  Jensen-Shannon divergence={js_divergence:.4f}")
    if clipped_above_max_bin:
        emit(f"   NOTE: {clipped_above_max_bin} frame(s) had actual > {MAX_BIN}, folded into bin {MAX_BIN} for this comparison")

    emit()
    emit(f"7. density control lag: mean|actual-target|={mean_abs_lag:.2f} objects")
    emit(f"   over-target frames  : {over_target_frames} ({over_target_frames / n_frames * 100:.1f}%)")
    emit(f"   under-target frames : {under_target_frames} ({under_target_frames / n_frames * 100:.1f}%)")
    emit(f"   within +/-1 target  : {within_1} ({within_1 / n_frames * 100:.1f}%)")
    emit(f"   within +/-2 target  : {within_2} ({within_2 / n_frames * 100:.1f}%)")

    emit()
    emit("8. spawn/despawn (from canonical_spawn_summary.json):")
    if spawn_summary:
        for key, value in spawn_summary.items():
            emit(f"   {key}: {value}")
    else:
        emit("   canonical_spawn_summary.json not found")

    emit()
    emit(f"9. spawned_inside_visible_roi = {spawn_summary.get('spawned_inside_visible_roi', 'N/A')} "
         f"-> {'PASS (must be 0)' if spawn_summary.get('spawned_inside_visible_roi', -1) == 0 else 'CHECK'}")

    emit()
    emit("10. ego mobility:")
    emit(f"    frames        : {len(speeds_kmh)}")
    emit(f"    avg speed     : {np.mean(speeds_kmh):.2f} km/h")
    emit(f"    median speed  : {np.median(speeds_kmh):.2f} km/h")
    emit(f"    min / max     : {min(speeds_kmh):.2f} / {max(speeds_kmh):.2f} km/h")
    below5 = sum(1 for s in speeds_kmh if s < 5.0) / len(speeds_kmh)
    emit(f"    <5 km/h ratio : {below5 * 100:.1f}%")
    emit(f"    final ego_route_s: {final_route_s:.2f} m")
    emit("    Prior canonical-policy distance-Gamma baseline (600f, Town10 route 0, seed 42):")
    emit("      avg speed=4.79 km/h final_s=37.19m <5km/h=60.0%")

    emit()
    emit("11. same-lane congestion:")
    if leads:
        emit(f"    nearest same-lane lead present in {len(leads)}/{len(frames)} frames, "
             f"mean={np.mean(leads):.2f}m median={np.median(leads):.2f}m min={min(leads):.2f}m")
    else:
        emit("    no same-lane lead observed in any frame")

    emit()
    emit("12. next step: replace get_annotation_candidate_count()'s TEMPORARY_COUNT_BASIS "
         "(src/simulation/canonical_traffic.py) with a true camera-valid annotation count "
         "(FOV + occlusion + min-pixel-size filtering) once that policy lands -- no other "
         "caller/interface change needed.")

    summary_path = os.path.join(output_dir, "density_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print(f"[Output] {os.path.join(output_dir, 'frame_object_counts.csv')}")
    print(f"[Output] {os.path.join(output_dir, 'object_count_histogram.png')}")
    print(f"[Output] {os.path.join(output_dir, 'target_vs_actual.png')}")
    print(f"[Output] {summary_path}")
    for p in viz_paths:
        print(f"[Output] {p}")


if __name__ == "__main__":
    main()

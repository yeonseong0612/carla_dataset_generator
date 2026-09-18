"""
scripts/tools/validate_frame_object_gamma_longrun.py

CLAUDE.md task: "Long-Run Validation for Frame-Level Gamma Object Count
Policy" -- PURE validation, no code changes anywhere else. This script
does not modify CFG/config.py, src/simulation/canonical_traffic.py,
src/simulation/spawn_policy.py, scripts/collect_dataset.py, the
annotation pipeline, or sensor config, and does not import anything from
them in a way that changes their behavior -- it only runs
collect_dataset.py as a subprocess (exactly as
scripts/tools/validate_frame_object_gamma.py already did for the
600-frame baseline) and analyzes what it produces.

Question this answers: was the 600-frame baseline's narrow actual
histogram (mode=10, clustered 10-15, JS divergence=0.32 against the
target Gamma) a short-run artifact (not enough time for spawn/despawn
cycles + Gamma segments to play out), or a structural control-loop
issue? See CLAUDE.md sections 9-17 for the exact analyses required.

Reuses (imports only, no edits) from scripts/tools/validate_frame_object_gamma.py:
  load_frame_object_counts, target_integer_pmf, jensen_shannon_divergence,
  bucket_ratios, mode_of, percentile, render_count_histogram,
  render_target_vs_actual, sequence_root
and from scripts/tools/analyze_canonical_policy.py:
  load_all_annotations, nearest_same_lane_lead_per_frame, load_ego_state

New in this script (not available from the 600-frame tool, and not
obtainable without either re-running collect_dataset.py or modifying it
-- so extracted from its own stdout log instead, which it already prints
unmodified):
  - target segment boundaries/lengths (parsed from
    "[FrameObjectGamma] segment=... frames=[a,b) target=T" lines that
    src/simulation/canonical_traffic.py's FrameObjectGammaSchedule
    already prints)
  - a managed-actor population time series, reconstructed from the
    per-update "[Canonical] update=N ..." / "despawned=... spawned=..."
    lines that CanonicalBackgroundTraffic.update() already prints
    (population[update] = population[update-1] + spawned - despawned,
    seeded from the Phase-1 initial "[Spawn] ..." line count) -- used
    for the population-leak check and a Little's-Law actor-lifetime
    estimate (see estimate_actor_lifetime()'s docstring for the caveat:
    this is an ESTIMATE, not an exact per-actor measurement, since the
    current logging never prints a despawned actor's id).
"""

import argparse
import json
import math
import os
import re
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
from scripts.tools.validate_frame_object_gamma import (  # noqa: E402
    load_frame_object_counts,
    target_integer_pmf,
    jensen_shannon_divergence,
    bucket_ratios,
    mode_of,
    percentile,
    render_count_histogram,
    render_target_vs_actual,
    sequence_root,
    MAX_BIN,
    BUCKETS,
)
from scripts.tools.analyze_canonical_policy import (  # noqa: E402
    load_all_annotations,
    nearest_same_lane_lead_per_frame,
    load_ego_state,
)

FPS = cfg.SIMULATION.FPS
UPDATE_INTERVAL_FRAMES = cfg.SPAWN.UPDATE_INTERVAL_FRAMES

SEGMENT_RE = re.compile(
    r"\[FrameObjectGamma\] segment=(\d+) frames=\[(\d+),(\d+)\) target=(\d+) \(raw_gamma_sample=([\d.]+)\)"
)
UPDATE_RE = re.compile(
    r"\[Canonical\] update=(\d+) ego_s=([\-\d.]+)m frame_target=(\d+) current=(\d+) deficit=(\d+)"
)
DESPAWN_SPAWN_RE = re.compile(
    r"despawned=(\d+)\(behind=(\d+) fwd_cleanup=(\d+) proj=(\d+)\) spawned=(\d+) failed=(\d+) same_lane_rej=(\d+)"
)
INITIAL_SPAWN_RE = re.compile(r"^\[Spawn\] (\w+) id=(\d+) s=")


# ------------------------------------------------------------------
# Step 1: run collect_dataset.py, capturing full stdout to a log file
# (needed for segment/population parsing below -- otherwise identical
# to validate_frame_object_gamma.py's run_collection).
# ------------------------------------------------------------------

# This machine's port 8000 (cfg.TRAFFIC_MANAGER.PORT's on-disk default)
# is bound by an unrelated local app (Notion), not by anything CARLA/
# TrafficManager-related. This task forbids editing CFG/config.py even
# temporarily, so instead of touching the file, the collection
# subprocess below is launched via a `python -c` bootstrap that
# monkey-patches cfg.TRAFFIC_MANAGER.PORT on the in-memory cfg object
# (imported fresh in that subprocess, never written back to disk) before
# calling collect_dataset.main() -- zero bytes of any forbidden file
# change on disk. If this machine's port 8000 is free when this runs,
# TRAFFIC_MANAGER_PORT_OVERRIDE can simply be set to 8000 instead.
TRAFFIC_MANAGER_PORT_OVERRIDE = 8100


def run_collection_with_log(town, route_id, condition, max_frames, output_root, log_path, overwrite=True):
    cli_args = [
        "--towns", town,
        "--routes", route_id,
        "--conditions", condition,
        "--max-frames", str(max_frames),
        "--output-root", output_root,
        "--background-policy", "canonical",
    ]

    if overwrite:
        cli_args.append("--overwrite")

    bootstrap = (
        "import sys; "
        f"sys.path.insert(0, r'{PROJECT_ROOT}'); "
        f"sys.path.insert(0, r'{CARLA_PYTHONAPI}'); "
        "from CFG.config import cfg; "
        f"cfg.TRAFFIC_MANAGER.PORT = {TRAFFIC_MANAGER_PORT_OVERRIDE}; "
        f"sys.argv = ['collect_dataset.py'] + {cli_args!r}; "
        "import scripts.collect_dataset as collect_dataset_module; "
        "collect_dataset_module.main()"
    )
    cmd = [sys.executable, "-c", bootstrap]

    print("[Run] (TRAFFIC_MANAGER.PORT runtime-patched to "
          f"{TRAFFIC_MANAGER_PORT_OVERRIDE}, no file changes) collect_dataset.py " + " ".join(cli_args))

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"collect_dataset.py exited with code {process.returncode}")


# ------------------------------------------------------------------
# Step 2: parse the stdout log for segment + per-update spawn/despawn
# events (see module docstring for why this is the only source).
# ------------------------------------------------------------------

def parse_log(log_path):
    segments = []
    updates = []
    initial_spawn_count = 0
    seen_first_update = False
    pending_update = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = SEGMENT_RE.search(line)

            if m:
                segments.append({
                    "segment_index": int(m.group(1)),
                    "start_frame": int(m.group(2)),
                    "end_frame": int(m.group(3)),
                    "target": int(m.group(4)),
                    "raw_gamma_sample": float(m.group(5)),
                })
                continue

            m = INITIAL_SPAWN_RE.search(line)

            if m and not seen_first_update:
                initial_spawn_count += 1
                continue

            m = UPDATE_RE.search(line)

            if m:
                seen_first_update = True
                pending_update = {
                    "update_index": int(m.group(1)),
                    "frame_target": int(m.group(3)),
                    "current": int(m.group(4)),
                    "deficit": int(m.group(5)),
                }
                continue

            m = DESPAWN_SPAWN_RE.search(line)

            if m and pending_update is not None:
                pending_update.update({
                    "despawned": int(m.group(1)),
                    "despawned_behind": int(m.group(2)),
                    "despawned_fwd_cleanup": int(m.group(3)),
                    "despawned_proj": int(m.group(4)),
                    "spawned": int(m.group(5)),
                    "failed": int(m.group(6)),
                    "same_lane_rej": int(m.group(7)),
                })
                updates.append(pending_update)
                pending_update = None
                continue

    return segments, updates, initial_spawn_count


def reconstruct_population_series(updates, initial_population, update_interval_frames):
    """
    population[update] = population[update-1] + spawned - despawned,
    seeded from the Phase-1 initial spawn count. This is the ONLY way to
    get a managed-actor-count time series without modifying
    canonical_traffic.py to log it directly (it currently isn't
    printed) -- cross-checked in main() against
    canonical_spawn_summary.json's totals as a sanity check.
    """

    series = []
    population = initial_population
    cumulative_spawned = 0
    cumulative_despawned = 0
    cumulative_proj_despawns = 0

    for u in updates:
        population += u["spawned"] - u["despawned"]
        cumulative_spawned += u["spawned"]
        cumulative_despawned += u["despawned"]
        cumulative_proj_despawns += u["despawned_proj"]

        series.append({
            "update_index": u["update_index"],
            "frame_id": u["update_index"] * update_interval_frames,
            "managed_population": population,
            "cumulative_spawned": cumulative_spawned,
            "cumulative_despawned": cumulative_despawned,
            "cumulative_proj_despawns": cumulative_proj_despawns,
        })

    return series


# ------------------------------------------------------------------
# Section 12: actor-lifetime ESTIMATE (Little's Law) -- see caveat in
# module docstring. mean_population / throughput, both measured over a
# steady-state window (last 50% of the run) so warm-up doesn't bias it.
# ------------------------------------------------------------------

def estimate_actor_lifetime(population_series, steady_state_start_frame, fps=FPS):
    steady = [row for row in population_series if row["frame_id"] >= steady_state_start_frame]

    if len(steady) < 2:
        return None

    duration_frames = steady[-1]["frame_id"] - steady[0]["frame_id"]
    duration_seconds = duration_frames / fps
    spawned_in_window = steady[-1]["cumulative_spawned"] - steady[0]["cumulative_spawned"]

    if duration_seconds <= 0 or spawned_in_window <= 0:
        return None

    spawn_rate_per_sec = spawned_in_window / duration_seconds
    mean_population = float(np.mean([row["managed_population"] for row in steady]))
    lifetime_sec = mean_population / spawn_rate_per_sec

    return {
        "mean_population_steady_state": mean_population,
        "spawn_rate_per_sec_steady_state": spawn_rate_per_sec,
        "estimated_mean_lifetime_sec": lifetime_sec,
        "window_start_frame": steady[0]["frame_id"],
        "window_end_frame": steady[-1]["frame_id"],
    }


# ------------------------------------------------------------------
# Section 9: warm-up vs steady-state window stats
# ------------------------------------------------------------------

def window_stats(rows, shape, scale, min_count, max_count):
    actual_values = [r["actual"] for r in rows]
    target_values = [r["target"] for r in rows]

    if not actual_values:
        return None

    diffs = [a - t for a, t in zip(actual_values, target_values)]

    return {
        "n_frames": len(rows),
        "mean": float(np.mean(actual_values)),
        "mode": mode_of(actual_values),
        "bucket_ratios": {name: ratio for name, (_count, ratio) in bucket_ratios(actual_values).items()},
        "mean_abs_lag": float(np.mean(np.abs(diffs))),
    }


def split_windows(rows):
    n = len(rows)
    first_end = int(n * 0.20)
    mid_end = int(n * 0.50)

    return {
        "first_20pct": rows[:first_end],
        "middle_30pct": rows[first_end:mid_end],
        "last_50pct": rows[mid_end:],
    }


# ------------------------------------------------------------------
# Section 10: cumulative histogram convergence
# ------------------------------------------------------------------

def cumulative_convergence(rows, shape, scale, min_count, max_count, checkpoint_step=600):
    target_pmf = target_integer_pmf(shape, scale, min_count, max_count, MAX_BIN)
    checkpoints = list(range(checkpoint_step, len(rows) + 1, checkpoint_step))

    if not checkpoints or checkpoints[-1] != len(rows):
        checkpoints.append(len(rows))

    results = []

    for checkpoint in checkpoints:
        subset = [r["actual"] for r in rows[:checkpoint]]
        actual_counts = np.zeros(MAX_BIN + 1, dtype=int)

        for v in subset:
            actual_counts[min(v, MAX_BIN)] += 1

        actual_pmf = actual_counts / max(actual_counts.sum(), 1)
        js = jensen_shannon_divergence(actual_pmf, target_pmf)
        ratios = bucket_ratios(subset)

        results.append({
            "n_frames": checkpoint,
            "mode": mode_of(subset),
            "mean": float(np.mean(subset)),
            "ratio_7_10": ratios["7-10"][1],
            "js_divergence": js,
        })

    return results


def render_convergence_plot(convergence, out_path, width=1000, height=500):
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 50, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    xs = [c["n_frames"] for c in convergence]
    ys = [c["js_divergence"] for c in convergence]
    max_x = max(xs)
    max_y = max(max(ys), 1e-6) * 1.15

    def x_px(x):
        return margin_left + int(plot_w * x / max_x)

    def y_px(y):
        return margin_top + int(plot_h * (1.0 - y / max_y))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (0, 0, 0), 1)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), (0, 0, 0), 1)

    pts = np.array([[x_px(x), y_px(y)] for x, y in zip(xs, ys)], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (60, 60, 220), 2, cv2.LINE_AA)

    for x, y in zip(xs, ys):
        cv2.circle(canvas, (x_px(x), y_px(y)), 4, (60, 60, 220), -1)
        cv2.putText(canvas, str(x), (x_px(x) - 15, margin_top + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_y * frac
        y = y_px(value)
        cv2.line(canvas, (margin_left - 4, y), (margin_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.3f}", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "cumulative-frame JS divergence vs. target Gamma", (margin_left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "number of frames observed (cumulative)", (margin_left + plot_w // 2 - 110, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


# ------------------------------------------------------------------
# Representative frame visualizations (own copy -- validate_frame_object_
# gamma.py's version hard-codes REPRESENTATIVE_FRAMES=[50..550], too
# early-run-only for a 3000/6000-frame sweep)
# ------------------------------------------------------------------

def render_representative_frames(sequence_dir, rows_by_frame, out_dir, frame_ids):
    os.makedirs(out_dir, exist_ok=True)
    rgb_dir = os.path.join(sequence_dir, "rgb_left")
    written = []

    for frame_id in frame_ids:
        image_path = os.path.join(rgb_dir, f"{frame_id:06d}.png")

        if not os.path.isfile(image_path) or frame_id not in rows_by_frame:
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        row = rows_by_frame[frame_id]
        cv2.putText(image, f"Actual objects = {row['actual']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Target = {row['target']}  (frame {frame_id})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

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
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--collection-output-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--keep-raw-dataset", action="store_true", help="Don't delete the raw per-frame sensor data after extracting analysis outputs.")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "collection_stdout.log")

    run_collection_with_log(
        args.town, args.route_id, args.condition, args.max_frames,
        args.collection_output_root, log_path, overwrite=True,
    )

    seq_dir = sequence_root(args.collection_output_root, args.town, args.route_id, args.condition)

    # ---- frame_object_counts.csv ----
    rows = load_frame_object_counts(seq_dir)
    shutil.copyfile(os.path.join(seq_dir, "frame_object_counts.csv"), os.path.join(output_dir, "frame_object_counts.csv"))
    rows_by_frame = {r["frame_id"]: r for r in rows}
    n_frames = len(rows)
    actual_values = [r["actual"] for r in rows]

    # ---- overall stats ----
    shape, scale = cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE, cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE
    min_count, max_count = cfg.SPAWN.FRAME_OBJECT_MIN, cfg.SPAWN.FRAME_OBJECT_MAX

    target_pmf = target_integer_pmf(shape, scale, min_count, max_count, MAX_BIN)
    actual_counts = np.zeros(MAX_BIN + 1, dtype=int)

    for v in actual_values:
        actual_counts[min(v, MAX_BIN)] += 1

    target_counts_scaled = target_pmf * n_frames
    mae = float(np.mean(np.abs(actual_counts - target_counts_scaled)))
    rmse = float(np.sqrt(np.mean((actual_counts - target_counts_scaled) ** 2)))
    actual_pmf = actual_counts / max(actual_counts.sum(), 1)
    js = jensen_shannon_divergence(actual_pmf, target_pmf)

    diffs = [r["actual"] - r["target"] for r in rows]

    overall = {
        "n_frames": n_frames,
        "mean": float(np.mean(actual_values)),
        "median": float(np.median(actual_values)),
        "mode": mode_of(actual_values),
        "std": float(np.std(actual_values)),
        "min": int(min(actual_values)),
        "max": int(max(actual_values)),
        "p10": percentile(actual_values, 10), "p25": percentile(actual_values, 25),
        "p75": percentile(actual_values, 75), "p90": percentile(actual_values, 90),
        "bucket_ratios": {name: ratio for name, (_c, ratio) in bucket_ratios(actual_values).items()},
        "mae": mae, "rmse": rmse, "js_divergence": js,
        "mean_abs_lag": float(np.mean(np.abs(diffs))),
        "over_target_ratio": sum(1 for d in diffs if d > 0) / n_frames,
        "under_target_ratio": sum(1 for d in diffs if d < 0) / n_frames,
        "within_1_ratio": sum(1 for d in diffs if abs(d) <= 1) / n_frames,
        "within_2_ratio": sum(1 for d in diffs if abs(d) <= 2) / n_frames,
        "unique_actual_counts": sorted(set(actual_values)),
    }

    render_count_histogram(actual_counts, target_counts_scaled, os.path.join(output_dir, "object_count_histogram.png"))
    render_target_vs_actual(rows, os.path.join(output_dir, "target_vs_actual.png"))

    # ---- warm-up vs steady-state ----
    windows = split_windows(rows)
    window_results = {
        name: window_stats(win_rows, shape, scale, min_count, max_count)
        for name, win_rows in windows.items()
    }

    # ---- cumulative convergence ----
    convergence = cumulative_convergence(rows, shape, scale, min_count, max_count)
    render_convergence_plot(convergence, os.path.join(output_dir, "histogram_convergence.png"))

    # ---- log parsing: segments + population reconstruction ----
    segments, updates, initial_spawn_count = parse_log(log_path)
    population_series = reconstruct_population_series(updates, initial_spawn_count, UPDATE_INTERVAL_FRAMES)

    segment_targets = [s["target"] for s in segments]
    segment_durations_frames = [s["end_frame"] - s["start_frame"] for s in segments]

    segment_stats = {
        "n_segments": len(segments),
        "target_min": min(segment_targets) if segment_targets else None,
        "target_max": max(segment_targets) if segment_targets else None,
        "target_mean": float(np.mean(segment_targets)) if segment_targets else None,
        "target_mode": mode_of(segment_targets) if segment_targets else None,
        "duration_frames_mean": float(np.mean(segment_durations_frames)) if segment_durations_frames else None,
        "duration_frames_min": min(segment_durations_frames) if segment_durations_frames else None,
        "duration_frames_max": max(segment_durations_frames) if segment_durations_frames else None,
        "duration_seconds_mean": float(np.mean(segment_durations_frames)) / FPS if segment_durations_frames else None,
    }

    # ---- population leak check ----
    populations = [row["managed_population"] for row in population_series]
    half = len(populations) // 2
    leak_check = {
        "min_population": min(populations) if populations else None,
        "max_population": max(populations) if populations else None,
        "first_half_mean_population": float(np.mean(populations[:half])) if half > 0 else None,
        "second_half_mean_population": float(np.mean(populations[half:])) if half > 0 else None,
        "final_population": populations[-1] if populations else None,
        "final_cumulative_proj_despawns": population_series[-1]["cumulative_proj_despawns"] if population_series else None,
    }

    # ---- actor lifetime estimate (Little's Law, steady-state window) ----
    steady_state_start_frame = int(n_frames * 0.5)
    lifetime_estimate = estimate_actor_lifetime(population_series, steady_state_start_frame)

    # ---- spawn/despawn summary (authoritative totals from collect_dataset.py) ----
    spawn_summary_path = os.path.join(seq_dir, "canonical_spawn_summary.json")
    spawn_summary = {}

    if os.path.isfile(spawn_summary_path):
        with open(spawn_summary_path, "r", encoding="utf-8") as f:
            spawn_summary = json.load(f)

    shutil.copyfile(spawn_summary_path, os.path.join(output_dir, "canonical_spawn_summary.json"))

    # Sanity cross-check: log-parsed cumulative spawned should match
    # canonical_spawn_summary.json's total_spawned exactly.
    parsed_total_spawned = population_series[-1]["cumulative_spawned"] if population_series else 0
    spawn_consistency_ok = parsed_total_spawned == spawn_summary.get("total_spawned")

    spawn_dynamics = {
        "total_spawned": spawn_summary.get("total_spawned"),
        "total_despawned": spawn_summary.get("total_despawned"),
        "spawn_rate_per_100_frames": spawn_summary.get("total_spawned", 0) / n_frames * 100,
        "despawn_rate_per_100_frames": spawn_summary.get("total_despawned", 0) / n_frames * 100,
        "spawned_inside_visible_roi": spawn_summary.get("spawned_inside_visible_roi"),
        "spawned_inside_buffer": spawn_summary.get("spawned_inside_buffer"),
        "log_parse_consistency_check_passed": spawn_consistency_ok,
    }

    # ---- ego mobility ----
    ego_state_rows = load_ego_state(seq_dir)
    speeds_kmh = [float(row["speed_mps"]) * 3.6 for row in ego_state_rows]
    ego_mobility = {
        "avg_speed_kmh": float(np.mean(speeds_kmh)),
        "median_speed_kmh": float(np.median(speeds_kmh)),
        "below_5kmh_ratio": sum(1 for s in speeds_kmh if s < 5.0) / len(speeds_kmh),
        "min_speed_kmh": min(speeds_kmh), "max_speed_kmh": max(speeds_kmh),
    }

    # ---- representative frames ----
    n_viz = 6
    step = max(n_frames // n_viz, 1)
    viz_frame_ids = [min(step * (i + 1) - 1, n_frames - 1) for i in range(n_viz)]
    viz_dir = os.path.join(output_dir, "visualizations")
    viz_paths = render_representative_frames(seq_dir, rows_by_frame, viz_dir, viz_frame_ids)

    # ---- same-lane check (kept lightweight: skip for very large runs to save time, unless needed) ----
    frames = load_all_annotations(seq_dir)
    leads = nearest_same_lane_lead_per_frame(frames)
    same_lane = {
        "frames_with_lead": len(leads), "total_frames": len(frames),
        "mean_lead_m": float(np.mean(leads)) if leads else None,
        "median_lead_m": float(np.median(leads)) if leads else None,
    }

    # ---- write JSON + TXT summaries ----
    full_summary = {
        "run": {"town": args.town, "route_id": args.route_id, "condition": args.condition, "max_frames": args.max_frames, "n_frames_analyzed": n_frames},
        "overall": overall,
        "windows": window_results,
        "cumulative_convergence": convergence,
        "segment_stats": segment_stats,
        "population_leak_check": leak_check,
        "actor_lifetime_estimate": lifetime_estimate,
        "spawn_dynamics": spawn_dynamics,
        "ego_mobility": ego_mobility,
        "same_lane": same_lane,
    }

    with open(os.path.join(output_dir, "longrun_summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"Long-run validation: {n_frames} frames ({args.town} route {args.route_id} {args.condition})")
    emit("=" * 78)
    emit(f"mean={overall['mean']:.2f} median={overall['median']:.2f} mode={overall['mode']} std={overall['std']:.2f} min={overall['min']} max={overall['max']}")
    emit(f"p10={overall['p10']:.2f} p25={overall['p25']:.2f} p75={overall['p75']:.2f} p90={overall['p90']:.2f}")
    emit(f"unique actual counts observed: {overall['unique_actual_counts']}")
    emit("bucket ratios:")
    for name, ratio in overall["bucket_ratios"].items():
        emit(f"  {name:>6s}: {ratio * 100:5.1f}%")
    emit(f"MAE={mae:.3f} RMSE={rmse:.3f} JS={js:.4f}")
    emit(f"mean|actual-target|={overall['mean_abs_lag']:.2f} over-target={overall['over_target_ratio']*100:.1f}% "
         f"within+/-1={overall['within_1_ratio']*100:.1f}% within+/-2={overall['within_2_ratio']*100:.1f}%")
    emit()
    emit("Warm-up vs steady-state:")
    for name, w in window_results.items():
        if w is None:
            continue
        emit(f"  {name:14s} n={w['n_frames']:5d} mean={w['mean']:.2f} mode={w['mode']} mean_abs_lag={w['mean_abs_lag']:.2f}")
    emit()
    emit("Segment stats:")
    emit(f"  n_segments={segment_stats['n_segments']} target[min={segment_stats['target_min']} max={segment_stats['target_max']} "
         f"mean={segment_stats['target_mean']} mode={segment_stats['target_mode']}] "
         f"duration_mean={segment_stats['duration_frames_mean']:.1f}f ({segment_stats['duration_seconds_mean']:.2f}s)")
    emit()
    emit("Population leak check:")
    emit(f"  min={leak_check['min_population']} max={leak_check['max_population']} "
         f"first_half_mean={leak_check['first_half_mean_population']} second_half_mean={leak_check['second_half_mean_population']} "
         f"final={leak_check['final_population']}")
    emit()
    if lifetime_estimate:
        emit(f"Actor lifetime estimate (Little's Law, steady-state): {lifetime_estimate['estimated_mean_lifetime_sec']:.2f} sec "
             f"(mean_pop={lifetime_estimate['mean_population_steady_state']:.2f}, spawn_rate={lifetime_estimate['spawn_rate_per_sec_steady_state']:.3f}/s)")
    emit()
    emit(f"Spawn dynamics: {spawn_dynamics}")
    emit(f"Ego mobility: {ego_mobility}")
    emit(f"Same-lane lead: {same_lane}")

    with open(os.path.join(output_dir, "density_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print(f"[Done] outputs under {output_dir}")

    if not args.keep_raw_dataset:
        print(f"[Cleanup] removing raw sensor data at {seq_dir}")
        shutil.rmtree(seq_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

"""
scripts/tools/validate_frame_object_gamma_controller_fix.py

CLAUDE.md task: "Fix Gamma Object-Count Controller Time-Scale Mismatch
and Revalidate". Runs the post-fix collect_dataset.py (via subprocess,
same pattern as validate_frame_object_gamma_longrun.py) and analyzes the
result against both the 600f baseline and the failed ~787f long-run
(both hard-coded below from the prior session's measured results, not
re-derived -- see CLAUDE.md section 18).

New in this task vs. the two earlier validation scripts (not code
duplication -- these fields didn't exist before this task's controller
fix added them to frame_object_counts.csv):
  managed_population, visible_population, buffer_population,
  spawned_this_frame, pruned_this_frame, naturally_despawned_this_frame,
  cumulative_spawned, cumulative_pruned, cumulative_natural_despawn

Reused unmodified (import only) from the earlier two validation scripts:
  validate_frame_object_gamma.py: sequence_root, target_integer_pmf,
    jensen_shannon_divergence, bucket_ratios, mode_of, percentile,
    MAX_BIN, load_ego_state (via analyze_canonical_policy), etc.
  validate_frame_object_gamma_longrun.py: run_collection_with_log (the
    TRAFFIC_MANAGER.PORT runtime-patch bootstrap -- this machine's port
    8000 is bound by an unrelated local app), parse_log/SEGMENT_RE (for
    target-segment boundaries, still only available from stdout, not the
    CSV), render_representative_frames.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

import carla  # noqa: E402

from src.data.layout import resolve_geometry_root, route_root_of  # noqa: E402
from CFG.config import cfg  # noqa: E402
from scripts.tools.validate_frame_object_gamma import (  # noqa: E402
    target_integer_pmf,
    jensen_shannon_divergence,
    bucket_ratios,
    mode_of,
    percentile,
    sequence_root,
    MAX_BIN,
)
from scripts.tools.validate_frame_object_gamma_longrun import (  # noqa: E402
    run_collection_with_log,
    parse_log,
    render_representative_frames,
)
from scripts.tools.analyze_canonical_policy import (  # noqa: E402
    load_all_annotations,
    nearest_same_lane_lead_per_frame,
    load_ego_state,
)

FPS = cfg.SIMULATION.FPS
UPDATE_INTERVAL_FRAMES = cfg.SPAWN.UPDATE_INTERVAL_FRAMES

# Hard-coded prior-session baselines (CLAUDE.md section 18) -- fixed
# comparison points, never re-derived from a live re-run.
BASELINE_600F = {
    "mean": 11.30, "mode": 10, "range": (10, 15),
    "mean_abs_lag": 4.69, "within_2": 0.307, "over_target": 0.69,
}
BASELINE_LONGRUN_FAILED = {
    "n_frames": 787, "mean_abs_lag": 5.25, "within_2": 0.169,
    "population_start": 11, "population_end": 33,
    "stall_frame": 780, "stall_route_index": 100, "stall_progress": 0.4484,
}


# ------------------------------------------------------------------
# Extended CSV loader (12 columns -- see module docstring)
# ------------------------------------------------------------------

def load_extended_csv(sequence_dir):
    path = os.path.join(resolve_geometry_root(sequence_dir), "frame_object_counts.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame_id": int(row["frame_id"]),
                "target": int(row["target_object_count"]),
                "actual": int(row["actual_object_count"]),
                "managed_population": int(row["managed_population"]),
                "visible_population": int(row["visible_population"]),
                "buffer_population": int(row["buffer_population"]),
                "spawned_this_frame": int(row["spawned_this_frame"]),
                "pruned_this_frame": int(row["pruned_this_frame"]),
                "naturally_despawned_this_frame": int(row["naturally_despawned_this_frame"]),
                "cumulative_spawned": int(row["cumulative_spawned"]),
                "cumulative_pruned": int(row["cumulative_pruned"]),
                "cumulative_natural_despawn": int(row["cumulative_natural_despawn"]),
            })

    return rows


# ------------------------------------------------------------------
# Transition analysis (sections 22-23): rising / falling target
# ------------------------------------------------------------------

def find_transitions(segments):
    transitions = []

    for i in range(1, len(segments)):
        old_target = segments[i - 1]["target"]
        new_target = segments[i]["target"]
        transitions.append({
            "frame": segments[i]["start_frame"],
            "old_target": old_target,
            "new_target": new_target,
            "direction": "falling" if new_target < old_target else ("rising" if new_target > old_target else "flat"),
        })

    return transitions


def analyze_transition(transition, rows_by_frame, n_frames, next_segment_start):
    frame = transition["frame"]

    if frame not in rows_by_frame or frame >= n_frames:
        return None

    actual_at_transition = rows_by_frame[frame]["actual"]
    buffer_population_at_transition = rows_by_frame[frame]["buffer_population"]
    new_target = transition["new_target"]

    window_end = min(next_segment_start, n_frames)
    pruned_in_window = sum(
        rows_by_frame[f]["pruned_this_frame"] for f in range(frame, window_end) if f in rows_by_frame
    )
    spawned_in_window = sum(
        rows_by_frame[f]["spawned_this_frame"] for f in range(frame, window_end) if f in rows_by_frame
    )

    frames_to_within_2 = None

    for f in range(frame, window_end):
        if f not in rows_by_frame:
            continue

        if abs(rows_by_frame[f]["actual"] - new_target) <= 2:
            frames_to_within_2 = f - frame
            break

    best_value_in_window = None

    if transition["direction"] == "falling":
        candidates = [rows_by_frame[f]["actual"] for f in range(frame, window_end) if f in rows_by_frame]
        best_value_in_window = min(candidates) if candidates else None
    elif transition["direction"] == "rising":
        candidates = [rows_by_frame[f]["actual"] for f in range(frame, window_end) if f in rows_by_frame]
        best_value_in_window = max(candidates) if candidates else None

    return {
        **transition,
        "actual_at_transition": actual_at_transition,
        "buffer_population_at_transition": buffer_population_at_transition,
        "spawned_in_window": spawned_in_window,
        "pruned_in_window": pruned_in_window,
        "frames_to_within_2": frames_to_within_2,
        "best_value_in_window": best_value_in_window,
        "overshoot": (best_value_in_window - new_target) if (best_value_in_window is not None and transition["direction"] == "rising") else None,
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def render_population_plot(rows, out_path, width=1400, height=500):
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 50, 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    n = len(rows)
    max_value = max(max(r["managed_population"] for r in rows), 1) * 1.15

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

    managed_pts = np.array([[x_px(i), y_px(r["managed_population"])] for i, r in enumerate(rows)], dtype=np.int32)
    visible_pts = np.array([[x_px(i), y_px(r["visible_population"])] for i, r in enumerate(rows)], dtype=np.int32)
    buffer_pts = np.array([[x_px(i), y_px(r["buffer_population"])] for i, r in enumerate(rows)], dtype=np.int32)

    cv2.polylines(canvas, [managed_pts], False, (60, 60, 220), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [visible_pts], False, (60, 180, 60), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [buffer_pts], False, (220, 140, 60), 1, cv2.LINE_AA)

    legend = [("managed (total)", (60, 60, 220)), ("visible", (60, 180, 60)), ("buffer", (220, 140, 60))]

    for i, (label, color) in enumerate(legend):
        y = 18 + i * 16
        cv2.rectangle(canvas, (width - 220, y), (width - 204, y + 12), color, -1)
        cv2.putText(canvas, label, (width - 196, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "managed / visible / buffer population vs. frame", (margin_left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame_id", (margin_left + plot_w // 2 - 30, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def render_spawn_prune_despawn_plot(rows, out_path, width=1400, height=500):
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 50, 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    n = len(rows)
    max_value = max(
        max(r["cumulative_spawned"] for r in rows),
        max(r["cumulative_pruned"] for r in rows),
        max(r["cumulative_natural_despawn"] for r in rows),
        1,
    ) * 1.15

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

    spawned_pts = np.array([[x_px(i), y_px(r["cumulative_spawned"])] for i, r in enumerate(rows)], dtype=np.int32)
    pruned_pts = np.array([[x_px(i), y_px(r["cumulative_pruned"])] for i, r in enumerate(rows)], dtype=np.int32)
    despawn_pts = np.array([[x_px(i), y_px(r["cumulative_natural_despawn"])] for i, r in enumerate(rows)], dtype=np.int32)

    cv2.polylines(canvas, [spawned_pts], False, (60, 60, 220), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [pruned_pts], False, (150, 60, 200), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [despawn_pts], False, (60, 180, 60), 1, cv2.LINE_AA)

    legend = [("cumulative spawned", (60, 60, 220)), ("cumulative pruned", (150, 60, 200)), ("cumulative natural despawn", (60, 180, 60))]

    for i, (label, color) in enumerate(legend):
        y = 18 + i * 16
        cv2.rectangle(canvas, (width - 260, y), (width - 244, y + 12), color, -1)
        cv2.putText(canvas, label, (width - 236, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "cumulative spawned / pruned / naturally-despawned vs. frame", (margin_left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame_id", (margin_left + plot_w // 2 - 30, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


def render_count_histogram(actual_pmf, target_pmf, out_path, width=1100, height=650):
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 70, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    n_bins = len(actual_pmf)
    max_value = max(float(actual_pmf.max()), float(target_pmf.max()), 1e-6) * 1.15

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
        actual_top = y_px(actual_pmf[i])
        target_top = y_px(target_pmf[i])
        base_y = margin_top + plot_h

        cv2.rectangle(canvas, (actual_x, actual_top), (int(actual_x + bar_w), base_y), (60, 60, 220), -1)
        cv2.rectangle(canvas, (target_x, target_top), (int(target_x + bar_w), base_y), (60, 180, 60), 2)
        cv2.putText(canvas, str(i), (int(group_center - 6), margin_top + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = max_value * frac
        y = y_px(value)
        cv2.line(canvas, (margin_left - 4, y), (margin_left, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.3f}", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (width - 260, 20), (width - 244, 34), (60, 60, 220), -1)
    cv2.putText(canvas, "Actual (normalized)", (width - 236, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (width - 260, 40), (width - 244, 54), (60, 180, 60), 2)
    cv2.putText(canvas, "Gamma target (normalized)", (width - 236, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "objects/frame PMF: actual vs. frame-object Gamma target (normalized)", (margin_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "objects in frame", (margin_left + plot_w // 2 - 60, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "probability", (10, margin_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

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
# Main
# ------------------------------------------------------------------

def analyze_run(seq_dir, log_path, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)

    rows = load_extended_csv(seq_dir)
    shutil.copyfile(os.path.join(resolve_geometry_root(seq_dir), "frame_object_counts.csv"), os.path.join(output_dir, "frame_object_counts.csv"))
    rows_by_frame = {r["frame_id"]: r for r in rows}
    n_frames = len(rows)
    actual_values = [r["actual"] for r in rows]

    shape, scale = cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE, cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE
    min_count, max_count = cfg.SPAWN.FRAME_OBJECT_MIN, cfg.SPAWN.FRAME_OBJECT_MAX
    target_pmf = target_integer_pmf(shape, scale, min_count, max_count, MAX_BIN)

    actual_counts = np.zeros(MAX_BIN + 1, dtype=int)
    for v in actual_values:
        actual_counts[min(v, MAX_BIN)] += 1
    actual_pmf = actual_counts / max(actual_counts.sum(), 1)

    js = jensen_shannon_divergence(actual_pmf, target_pmf)
    render_count_histogram(actual_pmf, target_pmf, os.path.join(output_dir, "object_count_histogram.png"))
    render_target_vs_actual(rows, os.path.join(output_dir, "target_vs_actual.png"))
    render_population_plot(rows, os.path.join(output_dir, "population_vs_frame.png"))
    render_spawn_prune_despawn_plot(rows, os.path.join(output_dir, "spawn_prune_despawn_vs_frame.png"))

    diffs = [r["actual"] - r["target"] for r in rows]
    abs_diffs = [abs(d) for d in diffs]

    overall = {
        "n_frames": n_frames,
        "mean": float(np.mean(actual_values)), "median": float(np.median(actual_values)),
        "mode": mode_of(actual_values), "std": float(np.std(actual_values)),
        "min": int(min(actual_values)), "max": int(max(actual_values)),
        "p10": percentile(actual_values, 10), "p25": percentile(actual_values, 25),
        "p75": percentile(actual_values, 75), "p90": percentile(actual_values, 90),
        "bucket_ratios": {name: ratio for name, (_c, ratio) in bucket_ratios(actual_values).items()},
        "js_divergence": js,
        "mean_abs_lag": float(np.mean(abs_diffs)), "median_abs_lag": float(np.median(abs_diffs)),
        "rmse_per_frame": float(np.sqrt(np.mean(np.square(diffs)))),
        "within_1": sum(1 for d in abs_diffs if d <= 1) / n_frames,
        "within_2": sum(1 for d in abs_diffs if d <= 2) / n_frames,
        "within_3": sum(1 for d in abs_diffs if d <= 3) / n_frames,
        "over_target_ratio": sum(1 for d in diffs if d > 0) / n_frames,
        "under_target_ratio": sum(1 for d in diffs if d < 0) / n_frames,
    }

    # ---- population stability ----
    populations = [r["managed_population"] for r in rows]
    third = n_frames // 3
    population_stability = {
        "initial": populations[0], "max": max(populations), "final": populations[-1],
        "mean_first_third": float(np.mean(populations[:third])) if third > 0 else None,
        "mean_middle_third": float(np.mean(populations[third:2 * third])) if third > 0 else None,
        "mean_final_third": float(np.mean(populations[2 * third:])) if third > 0 else None,
    }

    # ---- spawn/prune/despawn ----
    total_spawned = rows[-1]["cumulative_spawned"]
    total_pruned = rows[-1]["cumulative_pruned"]
    total_natural_despawn = rows[-1]["cumulative_natural_despawn"]

    spawn_summary_path = os.path.join(resolve_geometry_root(seq_dir), "canonical_spawn_summary.json")
    spawn_summary = {}
    if os.path.isfile(spawn_summary_path):
        with open(spawn_summary_path, "r", encoding="utf-8") as f:
            spawn_summary = json.load(f)
    shutil.copyfile(spawn_summary_path, os.path.join(output_dir, "canonical_spawn_summary.json"))

    spawn_prune_despawn = {
        "total_spawned": total_spawned, "total_pruned": total_pruned, "total_natural_despawn": total_natural_despawn,
        "spawn_rate_per_100f": total_spawned / n_frames * 100,
        "prune_rate_per_100f": total_pruned / n_frames * 100,
        "natural_despawn_rate_per_100f": total_natural_despawn / n_frames * 100,
        "spawned_inside_visible_roi": spawn_summary.get("spawned_inside_visible_roi"),
        "pruned_inside_visible_roi": spawn_summary.get("pruned_inside_visible_roi"),
    }

    # ---- actor lifetime (Little's Law, last-50% steady-state window, using per-frame CSV data) ----
    steady_start = n_frames // 2
    steady_rows = rows[steady_start:]
    lifetime_estimate = None
    if len(steady_rows) >= 2:
        duration_seconds = (steady_rows[-1]["frame_id"] - steady_rows[0]["frame_id"]) / FPS
        spawned_in_window = steady_rows[-1]["cumulative_spawned"] - steady_rows[0]["cumulative_spawned"]
        if duration_seconds > 0 and spawned_in_window > 0:
            spawn_rate_per_sec = spawned_in_window / duration_seconds
            mean_population = float(np.mean([r["managed_population"] for r in steady_rows]))
            lifetime_estimate = {
                "mean_population_steady_state": mean_population,
                "spawn_rate_per_sec_steady_state": spawn_rate_per_sec,
                "estimated_mean_lifetime_sec": mean_population / spawn_rate_per_sec,
            }

    # ---- segments + transitions ----
    segments, _updates, initial_spawn_count = parse_log(log_path)
    transitions = find_transitions(segments)
    analyzed_transitions = []
    for i, t in enumerate(transitions):
        next_start = segments[i + 2]["start_frame"] if i + 2 < len(segments) else n_frames
        analyzed = analyze_transition(t, rows_by_frame, n_frames, next_start)
        if analyzed is not None:
            analyzed_transitions.append(analyzed)

    falling_transitions = [t for t in analyzed_transitions if t["direction"] == "falling"]
    rising_transitions = [t for t in analyzed_transitions if t["direction"] == "rising"]

    segment_targets = [s["target"] for s in segments]
    segment_durations = [s["end_frame"] - s["start_frame"] for s in segments]
    segment_stats = {
        "n_segments": len(segments),
        "target_min": min(segment_targets) if segment_targets else None,
        "target_max": max(segment_targets) if segment_targets else None,
        "target_mean": float(np.mean(segment_targets)) if segment_targets else None,
        "target_mode": mode_of(segment_targets) if segment_targets else None,
        "duration_frames_mean": float(np.mean(segment_durations)) if segment_durations else None,
        "duration_seconds_mean": float(np.mean(segment_durations)) / FPS if segment_durations else None,
    }

    # ---- ego mobility ----
    ego_state_rows = load_ego_state(seq_dir)
    speeds_kmh = [float(row["speed_mps"]) * 3.6 for row in ego_state_rows]
    route_progress_values = [float(row["route_progress"]) for row in ego_state_rows if row["route_progress"]]
    ego_mobility = {
        "avg_speed_kmh": float(np.mean(speeds_kmh)), "median_speed_kmh": float(np.median(speeds_kmh)),
        "below_5kmh_ratio": sum(1 for s in speeds_kmh if s < 5.0) / len(speeds_kmh),
        "min_speed_kmh": min(speeds_kmh), "max_speed_kmh": max(speeds_kmh),
        "final_route_progress_pct": route_progress_values[-1] if route_progress_values else None,
    }

    # ---- same-lane congestion near route index ~100 ----
    frames = load_all_annotations(seq_dir)
    leads = nearest_same_lane_lead_per_frame(frames)
    ego_rows_by_frame = {int(row["frame_id"]): row for row in ego_state_rows}
    near_stall_region_frames = [
        f for f, row in ego_rows_by_frame.items()
        if row.get("route_index") not in (None, "") and abs(int(float(row["route_index"])) - 100) <= 5
    ]
    near_stall_leads = []
    for frame in frames:
        if frame["frame_id"] in near_stall_region_frames:
            candidates = [
                float(obj["distance_m"]) for obj in frame.get("objects", [])
                if obj["bbox_3d"]["center_ego_m"][0] > 0 and abs(obj["bbox_3d"]["center_ego_m"][1]) <= 1.75
            ]
            if candidates:
                near_stall_leads.append(min(candidates))

    same_lane = {
        "frames_with_lead": len(leads), "total_frames": len(frames),
        "mean_lead_m": float(np.mean(leads)) if leads else None,
        "median_lead_m": float(np.median(leads)) if leads else None,
        "near_route_index_100_frames_analyzed": len(near_stall_region_frames),
        "near_route_index_100_mean_lead_m": float(np.mean(near_stall_leads)) if near_stall_leads else None,
        "near_route_index_100_frames_with_lead": len(near_stall_leads),
    }

    # ---- representative frames ----
    n_viz = 6
    step = max(n_frames // n_viz, 1)
    viz_frame_ids = [min(step * (i + 1) - 1, n_frames - 1) for i in range(n_viz)]
    viz_paths = render_representative_frames(seq_dir, rows_by_frame, os.path.join(output_dir, "visualizations"), viz_frame_ids)

    full_summary = {
        "label": label, "n_frames": n_frames, "initial_spawn_count": initial_spawn_count,
        "initial_target": segments[0]["target"] if segments else None,
        "initial_actual": rows[0]["actual"] if rows else None,
        "initial_managed_population": rows[0]["managed_population"] if rows else None,
        "overall": overall,
        "population_stability": population_stability,
        "spawn_prune_despawn": spawn_prune_despawn,
        "actor_lifetime_estimate": lifetime_estimate,
        "segment_stats": segment_stats,
        "falling_transitions": falling_transitions,
        "rising_transitions": rising_transitions,
        "ego_mobility": ego_mobility,
        "same_lane": same_lane,
    }

    with open(os.path.join(output_dir, "controller_fix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"Controller-fix validation: {label} -- {n_frames} frames")
    emit("=" * 78)
    emit(f"initial target={full_summary['initial_target']} initial managed_population={full_summary['initial_managed_population']} initial actual={full_summary['initial_actual']}")
    emit(f"mean={overall['mean']:.2f} median={overall['median']:.2f} mode={overall['mode']} std={overall['std']:.2f} min={overall['min']} max={overall['max']}")
    emit("bucket ratios: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in overall["bucket_ratios"].items()))
    emit(f"JS divergence={js:.4f}  mean_abs_lag={overall['mean_abs_lag']:.2f}  median_abs_lag={overall['median_abs_lag']:.2f}  rmse_per_frame={overall['rmse_per_frame']:.2f}")
    emit(f"within+/-1={overall['within_1']*100:.1f}% within+/-2={overall['within_2']*100:.1f}% within+/-3={overall['within_3']*100:.1f}% "
         f"over={overall['over_target_ratio']*100:.1f}% under={overall['under_target_ratio']*100:.1f}%")
    emit()
    emit(f"population: initial={population_stability['initial']} max={population_stability['max']} final={population_stability['final']} "
         f"thirds=[{population_stability['mean_first_third']:.1f}, {population_stability['mean_middle_third']:.1f}, {population_stability['mean_final_third']:.1f}]")
    emit(f"spawn/prune/despawn: {spawn_prune_despawn}")
    if lifetime_estimate:
        emit(f"actor lifetime estimate: {lifetime_estimate['estimated_mean_lifetime_sec']:.2f}s")
    emit(f"segments: {segment_stats}")
    emit()
    emit(f"falling transitions ({len(falling_transitions)}):")
    for t in falling_transitions:
        emit(f"  frame={t['frame']} {t['old_target']}->{t['new_target']} actual_at={t['actual_at_transition']} "
             f"buffer_pop={t['buffer_population_at_transition']} pruned_in_window={t['pruned_in_window']} "
             f"frames_to_within_2={t['frames_to_within_2']} best_value={t['best_value_in_window']}")
    emit()
    emit(f"rising transitions ({len(rising_transitions)}):")
    for t in rising_transitions:
        emit(f"  frame={t['frame']} {t['old_target']}->{t['new_target']} actual_at={t['actual_at_transition']} "
             f"spawned_in_window={t['spawned_in_window']} frames_to_within_2={t['frames_to_within_2']} "
             f"overshoot={t['overshoot']}")
    emit()
    emit(f"ego mobility: {ego_mobility}")
    emit(f"same-lane: {same_lane}")

    with open(os.path.join(output_dir, "density_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return full_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", type=str, default="Town10")
    parser.add_argument("--route-id", type=str, default="0")
    parser.add_argument("--condition", type=str, default="day_clear")
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--collection-output-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--keep-raw-dataset", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collection_stdout.log")

    run_collection_with_log(
        args.town, args.route_id, args.condition, args.max_frames,
        args.collection_output_root, log_path, overwrite=True,
    )

    seq_dir = sequence_root(args.collection_output_root, args.town, args.route_id, args.condition)
    analyze_run(seq_dir, log_path, output_dir, args.label)

    if not args.keep_raw_dataset:
        print(f"[Cleanup] removing raw sensor data at {seq_dir}")
        shutil.rmtree(route_root_of(seq_dir), ignore_errors=True)


if __name__ == "__main__":
    main()

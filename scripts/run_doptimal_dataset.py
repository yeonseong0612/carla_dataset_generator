"""
run_doptimal_dataset.py

Orchestrates D-optimal DOE dataset generation on a local CARLA server.

This script is an orchestration layer ONLY. It never touches CARLA sensors,
traffic or annotation code; every job is one call of the existing
scripts/collect_dataset.py (subprocess, sys.executable, no shell).

    D-optimal CSV (52 Sun/Rain/Fog rows, validated, row order = C001..C052)
      x seeds  (--seeds, default 101 202 303)
      x routes (--town / --routes, routes/<town>.xml)
      -> jobs J000001.. (condition -> seed -> route order)
      -> collect_dataset.py per job
      -> verification, run_complete.json, manifest, resume / retry

How one job maps onto the collector
-----------------------------------
collect_dataset.py already implements paired collection: ONE canonical run
per (town, route) records the full world state (traffic, pedestrians,
labels, depth, ...) and every weather is an RGB-only deterministic replay of
that recording. The runner uses that as-is:

    --output-root <output_root>/seed_<seed>
    --seed <seed>                         (all cfg seeds, see --seed help)
    --towns <town> --routes <route>
    --conditions doe_C017_sun+30_rain-Light_fog-Heavy

The DOE condition name is decoded by src/simulation/weather.py through the
shared level mapping in src/simulation/weather_profiles.py. The first job
of a (seed, route) runs the canonical geometry; every other condition of
that (seed, route) is replayed from it, so conditions sharing a seed differ
ONLY in weather -- traffic realization is identical by construction, not
just "similar". Job identity never depends on execution order.

Output (collector layout, one namespace per seed):

    <output_root>/seed_<seed>/<town>/route_<id>/
        geometry/                      shared GT of this (seed, route)
        conditions/day_clear/          canonical-run RGB (collector default)
        conditions/doe_C001_.../       one DOE job: rgb_left/ rgb_right/
                                       condition.json COMPLETE
                                       run_complete.json   <- this runner

    <experiment_dir>/experiment_manifest.csv, experiment.json, logs/

The CARLA server (CarlaUE4.exe) must already be running; this script never
starts or stops it.

Examples (PowerShell, conda env "carla"):

    python scripts/run_doptimal_dataset.py --design-csv configs/experiments/D_optimal_final_52_conditions.csv --seeds 101 202 303 --town Town01 --routes 0 1 2 --dry-run
    python scripts/run_doptimal_dataset.py --design-csv configs/experiments/D_optimal_final_52_conditions.csv --seeds 101 202 303 --town Town01 --routes 0 --only-condition C001 --only-seed 101 --max-jobs 1
    python scripts/run_doptimal_dataset.py --design-csv configs/experiments/D_optimal_final_52_conditions.csv --seeds 101 202 303 --town Town01 --routes 0 1 2 --resume
    python scripts/run_doptimal_dataset.py --design-csv configs/experiments/D_optimal_final_52_conditions.csv --seeds 101 202 303 --town Town01 --routes 0 1 2 --resume --rerun-failed

Fixed-length production (--target-frames N): each (seed, route) canonical
geometry is exactly N frames (collector --max-frames N --truncate-ok) and
all 52 weathers replay those N frames. sequence.json then says
truncated=True, which is expected here; the verifier instead requires every
per-frame count to be exactly N. The frame policy is part of the experiment
identity (manifest column target_frames, experiment.json/run_complete.json
"frame_policy").

    python scripts/run_doptimal_dataset.py --design-csv configs/experiments/D_optimal_final_52_conditions.csv --seeds 101 202 303 --town Town10HD_Opt --routes 0 --target-frames 1100 --verbose
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CFG.config import cfg  # noqa: E402
from src.data.layout import (  # noqa: E402
    COMPLETE_MARKER,
    condition_dir,
    geometry_dir,
    is_complete,
    route_root,
)
from src.simulation.weather_profiles import (  # noqa: E402
    DOE_SUN_LEVELS,
    DOE_WEATHER_FIELDS,
    FOG_PROFILES,
    RAIN_PROFILES,
    doe_condition_name,
    doe_weather_fields,
    weather_profile_definition,
    weather_profile_hash,
)


# ============================================================
# Defaults
# ============================================================

COLLECTOR_SCRIPT = PROJECT_ROOT / "scripts" / "collect_dataset.py"

DEFAULT_DESIGN_CSV = PROJECT_ROOT / "configs" / "experiments" / "D_optimal_final_52_conditions.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "doptimal_dataset"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "doptimal_experiment"
DEFAULT_EXPERIMENT_ID = "doptimal_final_52"
DEFAULT_SEEDS = [101, 202, 303]

DESIGN_COLUMNS = ("Sun", "Rain", "Fog")
EXPECTED_DESIGN_ROWS = 52

MANIFEST_NAME = "experiment_manifest.csv"
EXPERIMENT_INFO_NAME = "experiment.json"
RUN_COMPLETE_NAME = "run_complete.json"

MANIFEST_FIELDS = [
    "job_id", "condition_id", "design_row", "Sun", "Rain", "Fog",
    "seed", "town", "route_id", "target_frames", "status", "attempts", "output_path",
    "started_at", "finished_at", "exit_code", "error_summary",
]

# Columns that define WHAT a job is; a manifest whose jobs differ in any of
# them belongs to a different experiment and is never merged.
JOB_IDENTITY_FIELDS = [
    "job_id", "condition_id", "design_row", "Sun", "Rain", "Fog",
    "seed", "town", "route_id", "target_frames", "output_path",
]

# Frame policy of the canonical geometry (and therefore of every replay).
#   route_completion : record until the route is completed (default)
#   fixed_length     : --target-frames N, exactly N frames by design. The
#                      collector marks such geometry truncated=True; here
#                      that is expected behaviour, not a failure.
FRAME_POLICY_ROUTE_COMPLETION = "route_completion"
FRAME_POLICY_FIXED_LENGTH = "fixed_length"

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"
STATUSES = (PENDING, RUNNING, COMPLETED, FAILED, SKIPPED)

# Replayed weather vs expected DOE value (carla stores floats as float32).
WEATHER_TOLERANCE = 1e-3

BAR = "=" * 56


class DesignValidationError(ValueError):
    pass


class ManifestMismatchError(RuntimeError):
    pass


class CarlaUnavailableError(RuntimeError):
    pass


def frame_policy(target_frames):
    if target_frames is None:
        return {"mode": FRAME_POLICY_ROUTE_COMPLETION, "target_frames": None}

    return {"mode": FRAME_POLICY_FIXED_LENGTH, "target_frames": int(target_frames)}


def job_target_frames(job):
    value = job.get("target_frames")
    return None if value in (None, "") else int(value)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================
# D-optimal design CSV
# ============================================================

def read_design_csv(path):
    """Returns (header, rows) exactly as stored (UTF-8, optional BOM)."""

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        header = [name.strip() for name in (reader.fieldnames or [])]
        rows = [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]

    return header, rows


def validate_design(header, rows, expected_rows=EXPECTED_DESIGN_ROWS):
    """
    Validates the master design and returns (conditions, report).

    conditions preserve CSV row order: row 1 -> C001, row 2 -> C002, ...
    Nothing is sorted, corrected or regenerated; ANY problem raises
    DesignValidationError listing every issue found.
    """

    errors = []

    missing = [column for column in DESIGN_COLUMNS if column not in header]

    if missing:
        raise DesignValidationError(
            f"design CSV is missing required column(s) {missing}; found {header}"
        )

    extra_columns = [column for column in header if column not in DESIGN_COLUMNS]

    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    sun_levels = set(DOE_SUN_LEVELS)
    rain_levels = set(RAIN_PROFILES)
    fog_levels = set(FOG_PROFILES)

    conditions = []

    for index, row in enumerate(rows, start=1):
        try:
            sun = int(row["Sun"])
        except ValueError:
            errors.append(f"row {index}: Sun={row['Sun']!r} is not an integer")
            continue

        rain = row["Rain"]
        fog = row["Fog"]

        if sun not in sun_levels:
            errors.append(f"row {index}: Sun={sun} not in {sorted(sun_levels)}")
        if rain not in rain_levels:
            errors.append(f"row {index}: Rain={rain!r} not in {list(RAIN_PROFILES)}")
        if fog not in fog_levels:
            errors.append(f"row {index}: Fog={fog!r} not in {list(FOG_PROFILES)}")

        conditions.append({
            "condition_id": f"C{index:03d}",
            "design_row": index,
            "Sun": sun,
            "Rain": rain,
            "Fog": fog,
        })

    triples = Counter((c["Sun"], c["Rain"], c["Fog"]) for c in conditions)

    for triple, count in triples.items():
        if count > 1:
            errors.append(f"duplicate condition {triple} appears {count} times")

    counts = {
        "Sun": Counter(c["Sun"] for c in conditions),
        "Rain": Counter(c["Rain"] for c in conditions),
        "Fog": Counter(c["Fog"] for c in conditions),
    }

    expected_levels = {
        "Sun": list(DOE_SUN_LEVELS),
        "Rain": list(RAIN_PROFILES),
        "Fog": list(FOG_PROFILES),
    }

    for factor, levels in expected_levels.items():
        observed = set(counts[factor])

        if observed != set(levels):
            errors.append(
                f"{factor} level set {sorted(observed, key=str)} != expected {sorted(levels, key=str)}"
            )

        if expected_rows % len(levels) == 0:
            balanced = expected_rows // len(levels)

            for level in levels:
                if counts[factor].get(level, 0) != balanced:
                    errors.append(
                        f"{factor}={level} appears {counts[factor].get(level, 0)} times, expected {balanced}"
                    )

    if errors:
        raise DesignValidationError(
            "design CSV validation failed:\n  - " + "\n  - ".join(errors)
        )

    report = {
        "num_conditions": len(conditions),
        "extra_columns": extra_columns,
        "levels": expected_levels,
        "counts": {
            factor: [(level, counts[factor][level]) for level in levels]
            for factor, levels in expected_levels.items()
        },
    }

    return conditions, report


def load_design(path):
    header, rows = read_design_csv(path)
    return validate_design(header, rows)


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Routes
# ============================================================

def route_ids_in_xml(town):
    """
    Route ids declared in routes/<town>.xml -- the same files and id
    attribute scripts/collect_dataset.py uses (src.navigation.route.
    list_route_ids). Read directly so the runner does not import the
    CARLA agents/planner stack just to validate ids.
    """

    xml_path = PROJECT_ROOT / "routes" / f"{town}.xml"

    if not xml_path.is_file():
        raise FileNotFoundError(f"Route XML not found: {xml_path}")

    return [route.attrib["id"] for route in ET.parse(xml_path).getroot().findall("route")]


# ============================================================
# Jobs
# ============================================================

def seed_output_root(output_root, seed):
    """Collector --output-root of one seed (geometry is per seed)."""

    return Path(output_root) / f"seed_{int(seed)}"


def job_condition_name(job):
    return doe_condition_name(job["condition_id"], job["Sun"], job["Rain"], job["Fog"])


def job_route_path(job, output_root):
    return Path(route_root(str(seed_output_root(output_root, job["seed"])), job["town"], job["route_id"]))


def job_output_dir(job, output_root):
    return Path(condition_dir(str(job_route_path(job, output_root)), job_condition_name(job)))


def expand_jobs(conditions, seeds, town, routes, output_root, target_frames=None):
    """
    Deterministic job list: condition (CSV order) -> seed -> route.
    Every condition gets exactly the same seeds and routes (and frame policy).
    """

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")

    if len(set(routes)) != len(routes):
        raise ValueError(f"--routes contains duplicates: {routes}")

    jobs = []

    for condition in conditions:
        for seed in seeds:
            for route_id in routes:
                job = {
                    "job_id": f"J{len(jobs) + 1:06d}",
                    "condition_id": condition["condition_id"],
                    "design_row": condition["design_row"],
                    "Sun": condition["Sun"],
                    "Rain": condition["Rain"],
                    "Fog": condition["Fog"],
                    "seed": int(seed),
                    "town": town,
                    "route_id": str(route_id),
                    "target_frames": None if target_frames is None else int(target_frames),
                    "status": PENDING,
                    "attempts": 0,
                    "output_path": "",
                    "started_at": "",
                    "finished_at": "",
                    "exit_code": "",
                    "error_summary": "",
                }
                job["output_path"] = str(job_output_dir(job, output_root))
                jobs.append(job)

    expected = len(conditions) * len(seeds) * len(routes)

    if len(jobs) != expected:
        raise AssertionError(f"job expansion produced {len(jobs)} jobs, expected {expected}")

    paths = [job["output_path"] for job in jobs]

    if len(set(paths)) != len(paths):
        raise AssertionError("job output paths collide")

    return jobs


# ============================================================
# Manifest (CSV, atomic writes)
# ============================================================

def _manifest_value(job, field):
    value = job.get(field, "")
    return "" if value is None else str(value)


def write_manifest(path, jobs):
    """temp file -> os.replace, so a crash never leaves a half-written manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    with open(tmp_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for job in jobs:
            writer.writerow({field: _manifest_value(job, field) for field in MANIFEST_FIELDS})

    # os.replace can fail on Windows while another program (e.g. Excel)
    # holds the manifest open.
    for attempt in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def read_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["design_row"] = int(row["design_row"])
        row["Sun"] = int(row["Sun"])
        row["seed"] = int(row["seed"])
        row["attempts"] = int(row["attempts"] or 0)
        # Manifests written before --target-frames existed have no column:
        # they are route-completion experiments.
        row["target_frames"] = job_target_frames(row)

        if row["status"] not in STATUSES:
            raise ManifestMismatchError(f"{path}: unknown status {row['status']!r} for {row['job_id']}")

    return rows


def merge_manifest(expanded, existing, manifest_path):
    """
    Carry status/attempt history from an existing manifest onto the freshly
    expanded job list. The two must describe exactly the same jobs.
    """

    def identity(job):
        return tuple(_manifest_value(job, field) for field in JOB_IDENTITY_FIELDS)

    policies_new = {json.dumps(frame_policy(job_target_frames(j))) for j in expanded}
    policies_old = {json.dumps(frame_policy(job_target_frames(j))) for j in existing}

    if existing and policies_new != policies_old:
        raise ManifestMismatchError(
            f"Existing manifest {manifest_path} was created with frame policy "
            f"{sorted(policies_old)}, this run requests {sorted(policies_new)}. "
            f"A different --target-frames is a different experiment: use a "
            f"different --experiment-dir AND --output-root."
        )

    if [identity(j) for j in expanded] != [identity(j) for j in existing]:
        raise ManifestMismatchError(
            f"Existing manifest {manifest_path} describes a different job set "
            f"(design CSV, seeds, town, routes or output root changed). Use a "
            f"different --experiment-dir for a different experiment, or move "
            f"the old manifest away if you really mean to restart."
        )

    merged = []

    for job, old in zip(expanded, existing):
        job = dict(job)

        for field in ("status", "attempts", "started_at", "finished_at", "exit_code", "error_summary"):
            job[field] = old[field]

        merged.append(job)

    return merged


# ============================================================
# Completion markers / verification
# ============================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path, data):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def count_files(directory, extension):
    if not os.path.isdir(directory):
        return 0

    return sum(1 for entry in os.scandir(directory) if entry.name.endswith(extension))


def has_valid_completion_marker(job, output_root):
    """
    Resume check (cheap): the runner's run_complete.json AND the collector's
    own COMPLETE markers must all be present and agree with this job. The
    manifest status alone is never trusted.
    """

    out_dir = job_output_dir(job, output_root)
    route_path = job_route_path(job, output_root)
    marker_path = out_dir / RUN_COMPLETE_NAME

    if not marker_path.is_file():
        return False, "run_complete.json missing"

    if not (
        is_complete(str(out_dir))
        and is_complete(geometry_dir(str(route_path)))
        and is_complete(condition_dir(str(route_path), cfg.WEATHER.DEFAULT))
    ):
        return False, "collector COMPLETE marker missing"

    try:
        marker = read_json(marker_path)
        collector_marker = read_json(out_dir / COMPLETE_MARKER)
    except (OSError, ValueError) as exc:
        return False, f"unreadable marker: {exc}"

    expected = {
        "status": COMPLETED,
        "job_id": job["job_id"],
        "condition_id": job["condition_id"],
        "seed": int(job["seed"]),
        "town": job["town"],
        "route_id": str(job["route_id"]),
        "weather_profile_hash": weather_profile_hash(),
    }

    for key, value in expected.items():
        if marker.get(key) != value:
            return False, f"run_complete.json {key}={marker.get(key)!r} != {value!r}"

    if collector_marker.get("num_frames") != marker.get("num_frames"):
        return False, "frame count differs between COMPLETE and run_complete.json"

    # Markers written before frame_policy existed are route-completion runs.
    policy = frame_policy(job_target_frames(job))
    marker_policy = marker.get("frame_policy") or frame_policy(None)

    if marker_policy != policy:
        return False, f"run_complete.json frame_policy={marker_policy} != {policy}"

    if policy["target_frames"] is not None and marker.get("num_frames") != policy["target_frames"]:
        return False, f"num_frames={marker.get('num_frames')} != target_frames={policy['target_frames']}"

    return True, ""


def verify_job_outputs(job, output_root, allow_truncated=False, since_unix=None):
    """
    Full post-run check of one job's collector output. Returns
    (ok, info, errors); info feeds run_complete.json.

    The frame policy comes from the job itself (job["target_frames"]):
      route_completion : truncated geometry fails (unless allow_truncated,
                         the smoke-test --truncate-ok)
      fixed_length     : truncated geometry is expected; instead every
                         per-frame count must equal target_frames exactly
    """

    errors = []
    info = {}

    out_dir = job_output_dir(job, output_root)
    route_path = job_route_path(job, output_root)
    geometry_root = Path(geometry_dir(str(route_path)))

    if not is_complete(str(geometry_root)):
        return False, info, [f"canonical geometry not COMPLETE: {geometry_root}"]

    if not is_complete(str(out_dir)):
        return False, info, [f"condition not COMPLETE: {out_dir}"]

    # ---- condition.json: applied weather + replay validation ----------
    try:
        condition_json = read_json(out_dir / "condition.json")
    except (OSError, ValueError) as exc:
        return False, info, [f"condition.json unreadable: {exc}"]

    if condition_json.get("condition") != job_condition_name(job):
        errors.append(f"condition.json condition={condition_json.get('condition')!r}")

    applied = condition_json.get("weather_parameters") or {}
    expected_weather = doe_weather_fields(job["Sun"], job["Rain"], job["Fog"])

    for field in DOE_WEATHER_FIELDS:
        if field not in applied or abs(float(applied[field]) - expected_weather[field]) > WEATHER_TOLERANCE:
            errors.append(f"weather {field}={applied.get(field)} != expected {expected_weather[field]}")

    replay_validation = condition_json.get("replay_validation") or {}
    calibration_check = condition_json.get("calibration_check") or {}

    if replay_validation.get("passed") is not True:
        errors.append("replay validation did not pass")

    if calibration_check.get("equal") is not True:
        errors.append("calibration differs from canonical geometry")

    target_frames = job_target_frames(job)
    num_frames = int(condition_json.get("num_frames") or 0)

    if num_frames <= 0:
        errors.append("num_frames == 0")

    if target_frames is not None and num_frames != target_frames:
        errors.append(f"condition.json num_frames={num_frames} != target_frames={target_frames}")

    # In fixed-length mode every count is checked against the target, so a
    # short/long run can never pass by being self-consistent.
    expected_frames = num_frames if target_frames is None else target_frames

    for camera in ("rgb_left", "rgb_right"):
        count = count_files(out_dir / camera, ".png")

        if count != expected_frames:
            errors.append(f"{camera}: {count} images, expected {expected_frames}")

    # ---- shared geometry: annotations / calibration / seeds -----------
    for name in ("calibration.json", "sequence.json", os.path.join("pose", "poses.csv")):
        if not (geometry_root / name).is_file():
            errors.append(f"geometry file missing: {name}")

    for name in (os.path.join("labels", "object_3d"), "world_state"):
        count = count_files(geometry_root / name, ".json")

        if count != expected_frames:
            errors.append(f"{Path(name).as_posix()}: {count} files, expected {expected_frames}")

    sequence = {}

    try:
        sequence = read_json(geometry_root / "sequence.json")
    except (OSError, ValueError) as exc:
        errors.append(f"sequence.json unreadable: {exc}")

    seeds_recorded = {
        "random_seed": sequence.get("random_seed"),
        "traffic_seed": sequence.get("traffic_seed"),
        "pedestrian_seed": sequence.get("pedestrian_seed"),
        "spawn_seed": (sequence.get("spawn_policy") or {}).get("spawn_seed"),
    }

    for key, value in seeds_recorded.items():
        if value != int(job["seed"]):
            errors.append(f"geometry {key}={value!r} != job seed {job['seed']}")

    truncated = bool(sequence.get("truncated"))

    if target_frames is not None:
        if sequence and sequence.get("num_frames") != target_frames:
            errors.append(
                f"sequence.json num_frames={sequence.get('num_frames')!r} != "
                f"target_frames={target_frames}"
            )

    elif truncated and not allow_truncated:
        errors.append(
            "canonical geometry is TRUNCATED (route not completed); delete "
            f"{route_path} or use another --output-root"
        )

    if sequence.get("debug_mid_route_start"):
        errors.append("geometry was generated with a DEBUG mid-route start")

    # ---- paired validation written by this collector invocation -------
    paired = None

    if since_unix is not None:
        paired_path = route_path / "paired_validation.json"

        if not paired_path.is_file() or paired_path.stat().st_mtime < since_unix - 1:
            errors.append("paired_validation.json was not written by this run")
        else:
            report = read_json(paired_path)
            paired = {"passed": bool(report.get("passed")), "errors": report.get("errors", [])}

            if not paired["passed"]:
                errors.append(f"paired validation failed: {paired['errors'][:3]}")

    info.update({
        "num_frames": num_frames,
        "weather": applied,
        "weather_expected": expected_weather,
        "seeds_recorded": seeds_recorded,
        "geometry_truncated": truncated,
        "frame_policy": frame_policy(target_frames),
        "carla_map": sequence.get("map"),
        "fixed_delta_seconds": sequence.get("fixed_delta_seconds"),
        "simulation_hz": sequence.get("simulation_hz"),
        "recording_hz": sequence.get("recording_hz"),
        "replay_validation_passed": replay_validation.get("passed"),
        "calibration_equal": calibration_check.get("equal"),
        "paired_validation": paired,
    })

    return not errors, info, errors


# ============================================================
# Selection (resume / rerun-failed / debug filters)
# ============================================================

def reconcile_with_outputs(jobs, output_root):
    """
    Make manifest status agree with what is actually on disk:
      completed but markers missing/stale -> pending
      running (runner was killed)          -> pending
      not completed but markers valid      -> completed
    Returns a list of human-readable changes.
    """

    changes = []

    for job in jobs:
        ok, reason = has_valid_completion_marker(job, output_root)

        if job["status"] == COMPLETED and not ok:
            job["status"] = PENDING
            job["error_summary"] = f"was completed, re-queued on resume: {reason}"
            changes.append(f"{job['job_id']}: completed -> pending ({reason})")

        elif job["status"] == RUNNING:
            job["status"] = PENDING
            job["error_summary"] = "interrupted while running"
            changes.append(f"{job['job_id']}: running -> pending (interrupted)")

        elif job["status"] != COMPLETED and ok:
            changes.append(f"{job['job_id']}: {job['status']} -> completed (valid markers on disk)")
            job["status"] = COMPLETED
            job["error_summary"] = ""

    return changes


def apply_filters(jobs, only_conditions=None, only_seeds=None, only_routes=None):
    selected = jobs

    if only_conditions:
        known = {job["condition_id"] for job in jobs}
        unknown = set(only_conditions) - known

        if unknown:
            raise ValueError(f"--only-condition {sorted(unknown)} not in the design (C001..C{len(known):03d})")

        selected = [job for job in selected if job["condition_id"] in set(only_conditions)]

    if only_seeds:
        known = {job["seed"] for job in jobs}
        unknown = set(only_seeds) - known

        if unknown:
            raise ValueError(f"--only-seed {sorted(unknown)} not among --seeds {sorted(known)}")

        selected = [job for job in selected if job["seed"] in set(only_seeds)]

    if only_routes:
        known = {job["route_id"] for job in jobs}
        unknown = set(only_routes) - known

        if unknown:
            raise ValueError(f"--only-route {sorted(unknown)} not among --routes {sorted(known)}")

        selected = [job for job in selected if job["route_id"] in set(only_routes)]

    return selected


def select_jobs(jobs, rerun_failed=False, max_jobs=None):
    """
    Default / --resume : pending + skipped jobs (failed jobs are left alone)
    --rerun-failed     : failed jobs only
    --max-jobs N       : first N of those, in job order
    Completed jobs are never selected.
    """

    wanted = {FAILED} if rerun_failed else {PENDING, SKIPPED}
    selected = [job for job in jobs if job["status"] in wanted]

    if max_jobs is not None:
        selected = selected[:max_jobs]

    return selected


# ============================================================
# Collector command
# ============================================================

def build_collector_command(job, output_root, host, port, rerender=False,
                            smoke_max_frames=None, smoke_truncate_ok=False, target_frames=None):
    """
    argv list for one job (no shell; current interpreter = current conda env).

    target_frames      : production fixed-length geometry. The collector has
                         one mechanism for a frame cap, so this is passed as
                         --max-frames N --truncate-ok.
    smoke_max_frames / : smoke-test passthrough (--max-frames/--truncate-ok
    smoke_truncate_ok    on the runner); never combined with target_frames.
    """

    if target_frames is not None and (smoke_max_frames is not None or smoke_truncate_ok):
        raise ValueError("target_frames cannot be combined with smoke-test max_frames/truncate_ok")

    condition = job_condition_name(job)

    command = [
        sys.executable,
        str(COLLECTOR_SCRIPT),
        "--towns", job["town"],
        "--routes", str(job["route_id"]),
        "--conditions", condition,
        "--seed", str(job["seed"]),
        "--output-root", str(seed_output_root(output_root, job["seed"])),
        "--host", str(host),
        "--port", str(port),
    ]

    # A COMPLETE condition that the runner did not accept (crash before
    # run_complete.json, failed verification) is re-rendered from the
    # existing geometry via the collector's own --rerender-conditions.
    if rerender:
        command += ["--rerender-conditions", condition]

    if target_frames is not None:
        command += ["--max-frames", str(int(target_frames)), "--truncate-ok"]

    if smoke_max_frames is not None:
        command += ["--max-frames", str(smoke_max_frames)]

    if smoke_truncate_ok:
        command.append("--truncate-ok")

    return command


# ============================================================
# CARLA preflight
# ============================================================

def carla_port_open(host, port, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def preflight_carla(host, port, timeout=10.0):
    """Returns {"server_version", "client_version"} or raises CarlaUnavailableError."""

    if not carla_port_open(host, port):
        raise CarlaUnavailableError(
            f"No CARLA server is listening on {host}:{port}.\n"
            f"Start CarlaUE4.exe first (e.g. CarlaUE4.exe -carla-rpc-port={port}), "
            f"wait until the map is loaded, then rerun this command."
        )

    import carla

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        server_version = client.get_server_version()
        client_version = client.get_client_version()
    except RuntimeError as exc:
        raise CarlaUnavailableError(
            f"CARLA server at {host}:{port} did not answer within {timeout:.0f}s: {exc}"
        ) from exc

    return {"server_version": server_version, "client_version": client_version}


# ============================================================
# Provenance
# ============================================================

def git_info():
    def run(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")

    return {"commit": commit, "dirty": None if status is None else bool(status)}


def code_hashes():
    files = [
        "CFG/config.py",
        "scripts/collect_dataset.py",
        "scripts/run_doptimal_dataset.py",
        "src/simulation/weather.py",
        "src/simulation/weather_profiles.py",
    ]

    return {name: file_sha256(PROJECT_ROOT / name) for name in files if (PROJECT_ROOT / name).is_file()}


def build_experiment_info(args, design_path, design_report, jobs):
    return {
        "experiment_id": args.experiment_id,
        "design_csv": str(design_path),
        "design_csv_sha256": file_sha256(design_path),
        "num_conditions": design_report["num_conditions"],
        "seeds": list(args.seeds),
        "town": args.town,
        "routes": list(args.routes),
        "num_jobs": len(jobs),
        "output_root": str(args.output_root),
        "job_order": "condition (CSV order) -> seed -> route",
        "frame_policy": frame_policy(args.target_frames),
        "weather_profile_hash": weather_profile_hash(),
        "weather_profile": weather_profile_definition(),
        "camera": {
            "width": cfg.SENSOR.CAMERA.WIDTH,
            "height": cfg.SENSOR.CAMERA.HEIGHT,
            "fov": cfg.SENSOR.CAMERA.FOV,
        },
        "fixed_delta_seconds": cfg.SIMULATION.FIXED_DELTA_SECONDS,
        "recording_hz": cfg.RECORDING.FPS,
        "canonical_source_condition": cfg.WEATHER.DEFAULT,
    }


# ============================================================
# Job execution
# ============================================================

def run_subprocess(command, log_path, timeout_s=None, echo=False):
    """
    Runs the collector, streaming its output into log_path (and the console
    with --verbose). Returns the exit code (None on timeout).
    """

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# " + subprocess.list2cmdline(command) + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        def pump():
            for line in process.stdout:
                log.write(line)
                if echo:
                    sys.stdout.write(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()

        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join(timeout=10)
            log.write(f"\n# runner: killed after timeout {timeout_s}s\n")
            return None
        except KeyboardInterrupt:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

        reader.join(timeout=10)

    return process.returncode


def extract_error_summary(log_path, limit=300):
    """Best single-line failure reason from a collector log."""

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""

    for line in reversed(lines):
        if line.strip().startswith("Reason    :"):
            return line.split(":", 1)[1].strip()[:limit]

    for line in reversed(lines):
        stripped = line.strip()
        if stripped and ("Error" in stripped or "Exception" in stripped) and not stripped.startswith("File "):
            return stripped[:limit]

    return ""


def build_run_complete(job, info, args, design_path, carla_versions, git, hashes, command, attempt,
                       started_at, finished_at, elapsed_s):
    return {
        "status": COMPLETED,
        "experiment_id": args.experiment_id,
        "job_id": job["job_id"],
        "condition_id": job["condition_id"],
        "design_row": job["design_row"],
        "design_csv": str(design_path),
        "design_csv_sha256": file_sha256(design_path),
        "factors": {"Sun": job["Sun"], "Rain": job["Rain"], "Fog": job["Fog"]},
        "sun_altitude_angle": float(job["Sun"]),
        "collector_condition": job_condition_name(job),
        "weather": info["weather"],
        "weather_expected": info["weather_expected"],
        "weather_profile_hash": weather_profile_hash(),
        "weather_profile": weather_profile_definition(),
        "seed": int(job["seed"]),
        "seeds_recorded": info["seeds_recorded"],
        "town": job["town"],
        "carla_map": info["carla_map"],
        "route_id": str(job["route_id"]),
        "num_frames": info["num_frames"],
        "frame_policy": info["frame_policy"],
        "geometry_truncated": info["geometry_truncated"],
        # fixed_length geometry is truncated by design (not a failure).
        "geometry_truncation_expected": info["frame_policy"]["mode"] == FRAME_POLICY_FIXED_LENGTH,
        "fixed_delta_seconds": info["fixed_delta_seconds"],
        "simulation_hz": info["simulation_hz"],
        "recording_hz": info["recording_hz"],
        "camera": {
            "width": cfg.SENSOR.CAMERA.WIDTH,
            "height": cfg.SENSOR.CAMERA.HEIGHT,
            "fov": cfg.SENSOR.CAMERA.FOV,
            "calibration": "../../geometry/calibration.json",
        },
        "carla": carla_versions,
        "git": git,
        "code_hashes": hashes,
        "replay_validation_passed": info["replay_validation_passed"],
        "calibration_equal": info["calibration_equal"],
        "paired_validation": info["paired_validation"],
        "paths": {
            "condition_dir": job["output_path"],
            "geometry_dir": "../../geometry",
            "annotations": "../../geometry/labels",
            "canonical_source_condition": f"../{cfg.WEATHER.DEFAULT}",
        },
        "collector_command": command,
        "attempt": attempt,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_s, 1),
    }


# ============================================================
# Console output
# ============================================================

def print_preflight(args, design_path, design_report, jobs, selected, manifest_path, existing_manifest):
    num_routes = len(args.routes)

    print(BAR)
    print("D-Optimal Experiment Preflight")
    print(BAR)
    print(f"Design CSV       : {design_path}")
    print(f"Conditions       : {design_report['num_conditions']}")
    print()
    print(f"Sun levels       : {sorted(DOE_SUN_LEVELS)}")
    print(f"Rain levels      : [{', '.join(RAIN_PROFILES)}]")
    print(f"Fog levels       : [{', '.join(FOG_PROFILES)}]")
    print()
    print(f"Seeds            : {list(args.seeds)}")
    print(f"Town             : {args.town}")
    print(f"Routes           : [{', '.join(args.routes)}]")
    print()
    print(f"Frame policy     : {frame_policy(args.target_frames)['mode']}")
    print(f"Target frames    : {'none' if args.target_frames is None else args.target_frames}")
    print()
    print(f"Jobs             : {len(jobs)}  "
          f"({design_report['num_conditions']} x {len(args.seeds)} x {num_routes})")
    print(f"Selected to run  : {len(selected)}")
    print()
    print(f"Output root      : {args.output_root}")
    print(f"Manifest         : {manifest_path}{'  (existing)' if existing_manifest else ''}")
    print(f"Resume           : {args.resume}")
    print(f"Rerun failed     : {args.rerun_failed}")
    print(f"CARLA            : {args.host}:{args.port}")
    print(f"Weather profiles : sha256 {weather_profile_hash()[:16]}...")

    if design_report["extra_columns"]:
        print(f"Note             : extra CSV columns ignored: {design_report['extra_columns']}")

    print(BAR)
    print()

    for factor in ("Sun", "Rain", "Fog"):
        print(f"{factor} level counts:")
        width = max(len(str(level)) for level, _ in design_report["counts"][factor])

        for level, count in sorted(design_report["counts"][factor], key=lambda item: (
            item[0] if factor == "Sun" else list(design_report["levels"][factor]).index(item[0])
        )):
            print(f"  {str(level):<{width}} : {count}")

        print()


def print_run_summary(jobs, conditions, seeds, routes, town):
    print(BAR)
    print("D-Optimal CARLA Dataset Generation")
    print(BAR)
    print(f"Conditions : {len(conditions)}")
    print(f"Seeds      : {len(seeds)}")
    print(f"Routes     : {len(routes)}")
    print(f"Total jobs : {len(jobs)}")
    print(f"Town       : {town}")
    print(BAR)


def print_job_header(position, total, job):
    print()
    print(f"[{position:03d}/{total:03d}] RUNNING")
    print()
    print(f"Job       : {job['job_id']}")
    print(f"Condition : {job['condition_id']}")
    print(f"Sun       : {job['Sun']}")
    print(f"Rain      : {job['Rain']}")
    print(f"Fog       : {job['Fog']}")
    print(f"Seed      : {job['seed']}")
    print(f"Town      : {job['town']}")
    print(f"Route     : {job['route_id']}")
    print()
    print("Output:")
    print(f"  {job['output_path']}")


def format_elapsed(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the D-optimal DOE dataset (design CSV x seeds x routes) through "
            "scripts/collect_dataset.py with a resumable job manifest."
        )
    )

    parser.add_argument("--design-csv", type=Path, default=DEFAULT_DESIGN_CSV)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds applied to EVERY condition (default: 101 202 303).")
    parser.add_argument("--town", type=str, default=cfg.MAP.NAME,
                        help="Town key = routes/<town>.xml (default: cfg.MAP.NAME).")
    parser.add_argument("--routes", type=str, nargs="+", required=True,
                        help="Route ids from routes/<town>.xml, e.g. --routes 0 1 2.")

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR,
                        help="Holds experiment_manifest.csv, experiment.json and logs/.")
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)

    parser.add_argument("--host", type=str, default=cfg.CARLA.HOST)
    parser.add_argument("--port", type=int, default=cfg.CARLA.PORT)

    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing manifest; verified-completed jobs are skipped.")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Run ONLY jobs whose status is failed.")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Run at most the first N selected jobs (testing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate, expand, map weather and build commands; no CARLA, no writes.")

    parser.add_argument("--only-condition", type=str, nargs="+", default=None, help="e.g. C001")
    parser.add_argument("--only-seed", type=int, nargs="+", default=None)
    parser.add_argument("--only-route", type=str, nargs="+", default=None)

    parser.add_argument("--job-timeout-min", type=float, default=0.0,
                        help="Kill a collector job after this many minutes (0 = no limit).")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Stop the run after this many failed jobs in a row.")
    parser.add_argument("--verbose", action="store_true",
                        help="Echo collector output to the console (always written to logs/).")

    parser.add_argument("--target-frames", type=int, default=None,
                        help="Production fixed-length acquisition: every (seed, route) canonical "
                             "geometry is exactly N frames and every DOE weather replays those N "
                             "frames. Default: record until route completion.")

    # Smoke-test passthrough to the collector. Never used for the real
    # dataset: a truncated canonical geometry would be reused by every
    # later job of that (seed, route).
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--truncate-ok", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    args.design_csv = args.design_csv.resolve()
    args.output_root = args.output_root.resolve()
    args.experiment_dir = args.experiment_dir.resolve()

    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be >= 1")

    if args.target_frames is not None:
        if args.target_frames < 1:
            parser.error("--target-frames must be >= 1")

        if args.max_frames is not None or args.truncate_ok:
            parser.error(
                "--target-frames (production fixed-length) cannot be combined with the "
                "smoke-test options --max-frames / --truncate-ok"
            )

    if (args.max_frames is not None or args.truncate_ok) and (
        args.output_root == DEFAULT_OUTPUT_ROOT.resolve()
        or args.experiment_dir == DEFAULT_EXPERIMENT_DIR.resolve()
    ):
        parser.error(
            "--max-frames / --truncate-ok are smoke-test options: pass a non-default "
            "--output-root AND --experiment-dir so production geometry is never truncated."
        )

    return args


# ============================================================
# Main
# ============================================================

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    args = parse_args(argv)

    # ---- design ------------------------------------------------------
    design_path = args.design_csv

    if not design_path.is_file():
        print(f"[ERROR] Design CSV not found: {design_path}")
        print(f"        Place D_optimal_final_52_conditions.csv at {DEFAULT_DESIGN_CSV}")
        print("        or pass --design-csv <path>. The design is never regenerated.")
        return 2

    try:
        conditions, design_report = load_design(design_path)
    except DesignValidationError as exc:
        print(f"[ERROR] {design_path}\n{exc}")
        return 2

    # ---- routes / jobs ------------------------------------------------
    try:
        available_routes = route_ids_in_xml(args.town)
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"[ERROR] {exc}")
        return 2

    unknown_routes = [route for route in args.routes if route not in available_routes]

    if unknown_routes:
        print(f"[ERROR] routes {unknown_routes} not in routes/{args.town}.xml (available: {available_routes})")
        return 2

    try:
        jobs = expand_jobs(conditions, args.seeds, args.town, args.routes, args.output_root,
                           target_frames=args.target_frames)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    manifest_path = args.experiment_dir / MANIFEST_NAME
    existing_manifest = manifest_path.is_file()

    if existing_manifest:
        try:
            jobs = merge_manifest(jobs, read_manifest(manifest_path), manifest_path)
        except ManifestMismatchError as exc:
            print(f"[ERROR] {exc}")
            return 2

        progressed = [job for job in jobs if job["status"] != PENDING]

        if progressed and not (args.resume or args.rerun_failed):
            print(
                f"[ERROR] {manifest_path} already has {len(progressed)} job(s) that are not "
                f"pending. Pass --resume to continue (completed jobs are skipped) or "
                f"--rerun-failed to retry failed jobs."
            )
            return 2

    reconcile_changes = reconcile_with_outputs(jobs, args.output_root)

    try:
        candidates = apply_filters(jobs, args.only_condition, args.only_seed, args.only_route)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    selected = select_jobs(candidates, rerun_failed=args.rerun_failed, max_jobs=args.max_jobs)

    print_preflight(args, design_path, design_report, jobs, selected, manifest_path, existing_manifest)
    print_run_summary(jobs, conditions, args.seeds, args.routes, args.town)

    status_counts = Counter(job["status"] for job in jobs)
    print("Manifest status  : " + ", ".join(f"{s}={status_counts.get(s, 0)}" for s in STATUSES))

    for change in reconcile_changes[:20]:
        print(f"[Reconcile] {change}")

    if len(reconcile_changes) > 20:
        print(f"[Reconcile] ... {len(reconcile_changes) - 20} more")

    # ---- dry run -----------------------------------------------------
    if args.dry_run:
        print()
        print("Rain profiles (src/simulation/weather_profiles.py):")
        for name, values in RAIN_PROFILES.items():
            print(f"  {name:<9}: {values}")
        print("Fog profiles:")
        for name, values in FOG_PROFILES.items():
            print(f"  {name:<9}: {values}")

        preview = selected[:5]
        print()
        print(f"[Dry run] {len(selected)} job(s) would run; first {len(preview)}:")

        for job in preview:
            route_path = job_route_path(job, args.output_root)
            geometry_state = "exists" if is_complete(geometry_dir(str(route_path))) else "to be generated"
            print()
            print(f"  {job['job_id']}  {job['condition_id']}  Sun={job['Sun']} Rain={job['Rain']} "
                  f"Fog={job['Fog']}  seed={job['seed']}  route={job['route_id']}")
            print(f"    condition : {job_condition_name(job)}")
            print(f"    weather   : {doe_weather_fields(job['Sun'], job['Rain'], job['Fog'])}")
            print(f"    output    : {job['output_path']}")
            print(f"    geometry  : {geometry_state}")
            command = build_collector_command(
                job, args.output_root, args.host, args.port, target_frames=args.target_frames,
                smoke_max_frames=args.max_frames, smoke_truncate_ok=args.truncate_ok,
            )
            print(f"    command   : {subprocess.list2cmdline(command)}")

        print()
        print("[Dry run] nothing written, CARLA not contacted.")
        return 0

    if not selected:
        print()
        print("Nothing to run (all selected jobs are completed, or failed without --rerun-failed).")
        write_manifest(manifest_path, jobs)
        return 0

    # ---- CARLA preflight ---------------------------------------------
    try:
        carla_versions = preflight_carla(args.host, args.port)
    except CarlaUnavailableError as exc:
        print()
        print(f"[ERROR] {exc}")
        return 3

    print(f"[Preflight] CARLA server {carla_versions['server_version']} "
          f"(client {carla_versions['client_version']}) at {args.host}:{args.port}")

    if carla_versions["server_version"] != carla_versions["client_version"]:
        print("[WARN] CARLA client/server versions differ.")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.experiment_dir / EXPERIMENT_INFO_NAME,
        {
            **build_experiment_info(args, design_path, design_report, jobs),
            "carla": carla_versions,
            "updated_at": now_iso(),
        },
    )
    write_manifest(manifest_path, jobs)

    git = git_info()
    hashes = code_hashes()
    timeout_s = args.job_timeout_min * 60 if args.job_timeout_min > 0 else None

    totals = Counter()
    consecutive_failures = 0
    run_start = time.time()
    failed_groups = set()

    try:
        for position, job in enumerate(selected, start=1):
            total = len(selected)
            group = (job["seed"], job["route_id"])

            if group in failed_groups:
                job["status"] = SKIPPED
                job["error_summary"] = (
                    f"canonical geometry for seed={job['seed']} route={job['route_id']} "
                    f"failed earlier in this run"
                )
                totals[SKIPPED] += 1
                write_manifest(manifest_path, jobs)
                print(f"[{position:03d}/{total:03d}] SKIPPED | {job['job_id']} | {job['error_summary']}")
                continue

            if not carla_port_open(args.host, args.port):
                print()
                print(f"[ERROR] CARLA server at {args.host}:{args.port} is no longer reachable. "
                      f"Restart CarlaUE4.exe and rerun with --resume (add --rerun-failed for failed jobs).")
                break

            out_dir = job_output_dir(job, args.output_root)
            route_path = job_route_path(job, args.output_root)

            rerender = is_complete(str(out_dir)) and is_complete(geometry_dir(str(route_path)))
            command = build_collector_command(
                job, args.output_root, args.host, args.port,
                rerender=rerender, target_frames=args.target_frames,
                smoke_max_frames=args.max_frames, smoke_truncate_ok=args.truncate_ok,
            )

            job["attempts"] = int(job["attempts"]) + 1
            job["status"] = RUNNING
            job["started_at"] = now_iso()
            job["finished_at"] = ""
            job["exit_code"] = ""
            job["error_summary"] = ""
            write_manifest(manifest_path, jobs)

            print_job_header(position, total, job)

            log_path = args.experiment_dir / "logs" / f"{job['job_id']}_attempt{job['attempts']}.log"
            start = time.time()
            exit_code = run_subprocess(command, log_path, timeout_s=timeout_s, echo=args.verbose)
            elapsed = time.time() - start

            ok, info, errors = verify_job_outputs(
                job, args.output_root, allow_truncated=args.truncate_ok, since_unix=start,
            )

            job["finished_at"] = now_iso()
            job["exit_code"] = "timeout" if exit_code is None else exit_code

            if exit_code == 0 and ok:
                write_json_atomic(
                    out_dir / RUN_COMPLETE_NAME,
                    build_run_complete(
                        job, info, args, design_path, carla_versions, git, hashes, command,
                        job["attempts"], job["started_at"], job["finished_at"], elapsed,
                    ),
                )
                job["status"] = COMPLETED
                totals[COMPLETED] += 1
                consecutive_failures = 0
                print(f"[{position:03d}/{total:03d}] COMPLETED | frames={info['num_frames']} "
                      f"| elapsed={format_elapsed(elapsed)}")
            else:
                reason = extract_error_summary(log_path)

                if exit_code is None:
                    reason = f"timeout after {args.job_timeout_min:g} min"
                elif exit_code != 0 and not reason:
                    reason = f"collector exit code {exit_code}"

                job["error_summary"] = "; ".join(filter(None, [reason] + errors[:3]))[:500]
                job["status"] = FAILED
                totals[FAILED] += 1
                consecutive_failures += 1

                if not is_complete(geometry_dir(str(route_path))):
                    failed_groups.add(group)

                print(f"[{position:03d}/{total:03d}] FAILED | reason={job['error_summary']} "
                      f"| log={log_path}")

            write_manifest(manifest_path, jobs)

            if consecutive_failures >= args.max_consecutive_failures:
                print()
                print(f"[ERROR] {consecutive_failures} consecutive failures -- stopping. "
                      f"Check the logs, then rerun with --resume --rerun-failed.")
                break

    except KeyboardInterrupt:
        for job in selected:
            if job["status"] == RUNNING:
                job["status"] = PENDING
                job["error_summary"] = "interrupted by user"
        write_manifest(manifest_path, jobs)
        print()
        print("[Interrupted] manifest saved; rerun with --resume to continue.")
        return 130

    total_elapsed = time.time() - run_start
    status_counts = Counter(job["status"] for job in jobs)

    print()
    print(BAR)
    print("D-Optimal Run Summary")
    print(BAR)
    print(f"Completed (this run) : {totals[COMPLETED]}")
    print(f"Failed    (this run) : {totals[FAILED]}")
    print(f"Skipped   (this run) : {totals[SKIPPED]}")
    print(f"Not reached          : {len(selected) - sum(totals.values())}")
    print(f"Total elapsed time   : {format_elapsed(total_elapsed)}")
    print(f"Manifest             : " + ", ".join(f"{s}={status_counts.get(s, 0)}" for s in STATUSES))
    print(f"                       {manifest_path}")
    print(BAR)

    return 1 if totals[FAILED] or totals[SKIPPED] else 0


if __name__ == "__main__":
    sys.exit(main())

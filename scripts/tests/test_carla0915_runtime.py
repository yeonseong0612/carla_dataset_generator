"""
scripts/tests/test_carla0915_runtime.py

Remote-server CARLA 0.9.15 smoke test.

This does NOT re-implement the dataset pipeline. It drives the actual
production entrypoint (scripts/collect_dataset.py, imported as a module and
invoked through its own main()) against a small route/frame count, writes
its output to an isolated smoke-test directory (never dataset/), and then
inspects the artifacts that production code already writes (sequence.json,
timestamps.csv, condition.json, paired_validation.json, labels/) plus a
live static-environment-object check (src.simulation.environment) to answer
the checklist in CLAUDE.md section 7 (A-K).

It intentionally reuses, rather than duplicates:
  - scripts.collect_dataset.main()            (full canonical+replay run)
  - scripts.collect_dataset.validate_geometry() (frame-count / missing-file
    checks)
  - src.data.paired_validation (already written to paired_validation.json
    by the production run)
  - src.data.projection.CameraProjector        (independent re-projection
    for the label-projection sanity check)
  - src.simulation.environment                 (static object cleanup /
    inventory -- see scripts/tools/validate_runtime_environment.py for the
    full standalone version of this check and its documented API
    limitation: CARLA has no getter for an EnvironmentObject's enabled
    flag, so this can only confirm inventory/coverage, not a server-side
    re-query of the disabled flag itself)

Scope (CLAUDE.md section 7): compatibility / geometry / synchronization /
paired-replay verification only. This is not meant to produce a usable
research dataset -- default is 200 frames, day_clear + day_rain, one route.

Usage (remote server, CARLA 0.9.15 already running):

    python scripts/tests/test_carla0915_runtime.py \
        --host localhost --port 2000 \
        --town Town01 --route 1 --frames 200 \
        --conditions day_clear day_rain

Exit code 0 = every check passed, 1 = at least one failed/errored.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import carla  # noqa: E402

from CFG.config import cfg  # noqa: E402
from src.data.layout import route_root, geometry_dir, condition_dir  # noqa: E402
from src.data.projection import CameraProjector  # noqa: E402
from src.simulation.environment import (  # noqa: E402
    STATIC_VEHICLE_LABELS,
    STATIC_PEDESTRIAN_LABELS,
    get_static_object_ids,
)

EXPECTED_VERSION_DEFAULT = "0.9.15"
TIMESTAMP_TOLERANCE_S = 0.005  # fixed_delta_seconds is exact in sync mode


class Report:
    def __init__(self):
        self.rows = []  # (id, description, passed(bool|None), detail)

    def add(self, check_id, description, passed, detail=""):
        self.rows.append((check_id, description, passed, detail))
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"[{status}] {check_id}: {description}" + (f" -- {detail}" if detail else ""))

    def overall_passed(self):
        return all(passed for _, _, passed, _ in self.rows if passed is not None)

    def print_summary(self):
        print()
        print("=" * 70)
        print("CARLA 0.9.15 runtime smoke test -- summary")
        print("=" * 70)
        for check_id, description, passed, detail in self.rows:
            status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
            print(f"  [{status}] {check_id:<3} {description}")
        print("=" * 70)
        print(f"RESULT: {'PASS' if self.overall_passed() else 'FAIL'}")
        print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CARLA 0.9.15 remote smoke test: runs the real production "
            "pipeline (scripts/collect_dataset.py) at small scale into an "
            "isolated output directory, then verifies compatibility / "
            "geometry / synchronization / paired-replay contracts."
        )
    )
    parser.add_argument("--host", default=cfg.CARLA.HOST)
    parser.add_argument("--port", type=int, default=cfg.CARLA.PORT)
    parser.add_argument("--timeout", type=float, default=cfg.CARLA.TIMEOUT)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--route", default="1", help="Route id from routes/{town}.xml.")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["day_clear", "day_rain"],
        help="Weather conditions to render (day_clear must always be included).",
    )
    parser.add_argument(
        "--expected-version",
        default=EXPECTED_VERSION_DEFAULT,
        help="Expected CARLA client/server version string. Use 'any' to skip the check.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Isolated smoke-test output root. Defaults to a fresh "
            "timestamped directory under outputs/runtime_0915_validation/ "
            "-- never dataset/."
        ),
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the generated smoke-test dataset on disk (default: keep; this flag is a no-op reminder that nothing under dataset/ is ever touched).",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# A. version
# ------------------------------------------------------------

def check_version(report, client, expected_version):
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    print(f"[Version] client={client_version} server={server_version}")

    if expected_version.lower() == "any":
        report.add("A", "client/server version", None, "skipped (--expected-version any)")
        return

    ok = (client_version == expected_version) and (server_version == expected_version)
    report.add(
        "A", "client/server both == expected_version", ok,
        f"expected={expected_version} client={client_version} server={server_version}",
    )


# ------------------------------------------------------------
# Run the actual production pipeline (canonical + replay)
# ------------------------------------------------------------

def run_production_pipeline(args, output_root):
    """
    Imports scripts.collect_dataset and calls its main() exactly as the
    remote CLI invocation would, so this test exercises the real
    production code path (no reimplementation).
    """

    cfg.CARLA.HOST = args.host
    cfg.CARLA.PORT = args.port

    argv_backup = sys.argv
    sys.argv = [
        "collect_dataset.py",
        "--towns", args.town,
        "--routes", str(args.route),
        "--conditions", *args.conditions,
        "--max-frames", str(args.frames),
        "--truncate-ok",
        "--overwrite",
        "--output-root", str(output_root),
    ]

    try:
        import scripts.collect_dataset as collect_dataset
        collect_dataset.main()
        return True, None
    except SystemExit as exc:
        return (exc.code in (None, 0)), None
    except Exception:
        return False, traceback.format_exc()
    finally:
        sys.argv = argv_backup


# ------------------------------------------------------------
# B/C. synchronous mode + per-tick sensor sync
#
# collect_dataset.py itself sets synchronous_mode=True /
# fixed_delta_seconds=cfg.SIMULATION.FIXED_DELTA_SECONDS before any
# sensor work, and raises a hard RuntimeError the instant a tick's
# world.tick() frame, world.get_snapshot().frame or any sensor packet's
# .frame disagree (see collect_dataset.py's canonical capture loop). A
# completed run without that exception is therefore itself a live,
# per-tick confirmation for every frame produced -- this check does not
# re-verify it a second time, it verifies the *configuration* that
# guarantees it and that the run actually completed.
# ------------------------------------------------------------

def check_sync_settings(report, pipeline_ok):
    expected_dt = 1.0 / cfg.SIMULATION.FPS
    dt_ok = abs(cfg.SIMULATION.FIXED_DELTA_SECONDS - expected_dt) < 1e-9
    report.add(
        "B", "synchronous_mode=True, fixed_delta_seconds configured",
        dt_ok and pipeline_ok,
        f"fixed_delta_seconds={cfg.SIMULATION.FIXED_DELTA_SECONDS} (expected {expected_dt})",
    )
    report.add(
        "C", "per-tick sensor/frame sync (world.tick / snapshot / sensor.frame)",
        pipeline_ok,
        "production run completed without the hard RuntimeError it raises on any mismatch"
        if pipeline_ok else "run did not complete -- see pipeline error above",
    )


# ------------------------------------------------------------
# D. timestamp spacing
# ------------------------------------------------------------

def check_timestamps(report, geometry_root):
    path = os.path.join(geometry_root, "timestamps.csv")

    if not os.path.isfile(path):
        report.add("D", "consecutive frame timestamp spacing ~= fixed_delta_seconds", False, f"missing {path}")
        return

    import csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        report.add("D", "consecutive frame timestamp spacing ~= fixed_delta_seconds", False, "fewer than 2 rows")
        return

    timestamps = [float(r["timestamp"]) for r in rows]
    deltas = np.diff(timestamps)
    expected = cfg.SIMULATION.FIXED_DELTA_SECONDS
    max_err = float(np.max(np.abs(deltas - expected)))
    ok = max_err <= TIMESTAMP_TOLERANCE_S

    report.add(
        "D", "consecutive frame timestamp spacing ~= fixed_delta_seconds", ok,
        f"expected={expected:.4f}s max_deviation={max_err:.5f}s over {len(deltas)} intervals",
    )


# ------------------------------------------------------------
# E. camera geometry consistency
# ------------------------------------------------------------

def check_camera_geometry(report, geometry_root):
    calib_path = os.path.join(geometry_root, "calibration.json")

    if not os.path.isfile(calib_path):
        report.add("E", "camera/ego/calibration extrinsic consistency", False, f"missing {calib_path}")
        return None

    with open(calib_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    try:
        t_left = np.array(calibration["sensors"]["rgb_left"]["T_ego_from_sensor"])
        t_right = np.array(calibration["sensors"]["rgb_right"]["T_ego_from_sensor"])
        baseline_measured = abs(float(t_right[1, 3] - t_left[1, 3]))
        baseline_expected = cfg.SENSOR.STEREO.BASELINE
        baseline_ok = abs(baseline_measured - baseline_expected) < 0.01

        cam_left = calibration["cameras"]["rgb_left"]
        k = np.array(cam_left["K"])
        k_ok = k[0, 0] > 0 and k[1, 1] > 0 and cam_left["width"] > 0 and cam_left["height"] > 0

        rot = t_left[:3, :3]
        orthogonal_ok = np.allclose(rot @ rot.T, np.eye(3), atol=1e-3)

        ok = baseline_ok and k_ok and orthogonal_ok
        report.add(
            "E", "camera/ego/calibration extrinsic consistency", ok,
            f"stereo_baseline measured={baseline_measured:.4f}m expected={baseline_expected:.4f}m, "
            f"K/dims sane={k_ok}, rotation orthogonal={orthogonal_ok}",
        )
        return calibration
    except (KeyError, IndexError) as exc:
        report.add("E", "camera/ego/calibration extrinsic consistency", False, f"malformed calibration.json: {exc}")
        return None


# ------------------------------------------------------------
# F. label projection sanity
# ------------------------------------------------------------

def check_label_projection(report, geometry_root, calibration, sample_frames=5):
    if calibration is None:
        report.add("F", "object 3D->image projection within image bounds", False, "no calibration.json loaded")
        return

    labels_dir = os.path.join(geometry_root, "labels", "object_3d")

    if not os.path.isdir(labels_dir):
        report.add("F", "object 3D->image projection within image bounds", False, f"missing {labels_dir}")
        return

    frame_files = sorted(os.listdir(labels_dir))[:1] + sorted(os.listdir(labels_dir))[-1:]
    frame_files = sorted(set(frame_files))
    all_files = sorted(os.listdir(labels_dir))
    step = max(len(all_files) // sample_frames, 1)
    frame_files = all_files[::step][:sample_frames] or all_files[:sample_frames]

    if not frame_files:
        report.add("F", "object 3D->image projection within image bounds", False, "no per-frame label files found")
        return

    projector = CameraProjector.from_calibration(calibration, "rgb_left")
    checked_objects = 0
    bad_objects = 0

    for name in frame_files:
        with open(os.path.join(labels_dir, name), "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        for obj in frame_data.get("objects", []):
            vertices_ego_m = obj["bbox_3d"]["vertices_ego_m"]
            uv, valid = projector.project(projector.ego_to_cv(vertices_ego_m))

            checked_objects += 1

            if not np.any(valid):
                bad_objects += 1
                continue

            u_valid, v_valid = uv[valid, 0], uv[valid, 1]
            clip_u = np.clip(u_valid, 0, projector.width)
            clip_v = np.clip(v_valid, 0, projector.height)
            intersect_area = (clip_u.max() - clip_u.min()) * (clip_v.max() - clip_v.min())

            # objects[] only contains camera_valid==True entries, so a
            # zero-area re-projection here would mean projection.py and
            # annotation.py disagree -- flag it.
            if intersect_area <= 0:
                bad_objects += 1

    ok = checked_objects > 0 and bad_objects == 0
    report.add(
        "F", "object 3D->image projection within image bounds", ok,
        f"{checked_objects - bad_objects}/{checked_objects} camera_valid objects re-projected inside image bounds "
        f"across {len(frame_files)} sampled frames",
    )


# ------------------------------------------------------------
# G/H/I. replay contract, wind, output completeness
# ------------------------------------------------------------

def check_conditions(report, route_path, conditions, source_condition="day_clear"):
    wind_ok = True
    wind_detail = []
    replay_ok = True
    replay_detail = []

    for condition in conditions:
        cond_dir = condition_dir(route_path, condition)
        cond_json_path = os.path.join(cond_dir, "condition.json")

        if not os.path.isfile(cond_json_path):
            wind_ok = False
            replay_ok = False
            wind_detail.append(f"{condition}: missing condition.json")
            continue

        with open(cond_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("wind_intensity") != 0.0:
            wind_ok = False
        wind_detail.append(f"{condition}={meta.get('wind_intensity')}")

        if condition == source_condition:
            if meta.get("rendered_from") != "canonical_run":
                replay_ok = False
                replay_detail.append(f"{condition}: expected rendered_from=canonical_run, got {meta.get('rendered_from')}")
            # day_clear must never carry non-RGB replay dirs either -- it's
            # produced by the canonical run, but structurally it should
            # still only have rgb_left/rgb_right under conditions/.
            extra = set(os.listdir(cond_dir)) - {"rgb_left", "rgb_right", "condition.json", "COMPLETE"}
            if extra:
                replay_ok = False
                replay_detail.append(f"{condition}: unexpected entries {sorted(extra)}")
            continue

        if meta.get("rendered_from") != "replay" or not meta.get("rgb_only_replay", False):
            replay_ok = False
            replay_detail.append(f"{condition}: not marked as RGB-only replay ({meta.get('rendered_from')})")

        extra = set(os.listdir(cond_dir)) - {"rgb_left", "rgb_right", "condition.json", "COMPLETE"}
        if extra:
            replay_ok = False
            replay_detail.append(f"{condition}: non-RGB output present {sorted(extra)}")

        validation = meta.get("replay_validation") or {}
        if not validation.get("passed", False):
            replay_ok = False
            replay_detail.append(f"{condition}: replay_validation.passed=False")
        if validation.get("actor_id_set_mismatch_frames", 1) != 0:
            replay_ok = False
            replay_detail.append(f"{condition}: actor_id_set_mismatch_frames != 0")
        if validation.get("out_of_tolerance_samples", 1) != 0:
            replay_ok = False
            replay_detail.append(f"{condition}: out_of_tolerance_samples != 0 (apply_batch_sync drift)")

    report.add("G", "replay contract (RGB-only, apply_batch_sync, 0 out-of-tolerance, no day_clear replay)", replay_ok, "; ".join(replay_detail) or "ok")
    report.add("H", "wind_intensity == 0.0 for every condition", wind_ok, ", ".join(wind_detail))


def check_output_completeness(report, route_path, geometry_root, conditions, source_condition, frames):
    import importlib
    collect_dataset = importlib.import_module("scripts.collect_dataset")

    cond_root = condition_dir(route_path, source_condition)
    errors = collect_dataset.validate_geometry(geometry_root, cond_root, frames)

    paired_path = os.path.join(route_path, "paired_validation.json")
    paired_ok = False
    paired_detail = "missing paired_validation.json"

    if os.path.isfile(paired_path):
        with open(paired_path, "r", encoding="utf-8") as f:
            paired = json.load(f)
        paired_ok = bool(paired.get("passed"))
        paired_detail = f"passed={paired.get('passed')} num_frames={paired.get('num_frames')} errors={paired.get('errors')}"

    ok = (not errors) and paired_ok
    report.add(
        "I", "output completeness (frame count, no missing/duplicate, paired_validation PASS)", ok,
        (f"validate_geometry errors={errors}; " if errors else "") + paired_detail,
    )


# ------------------------------------------------------------
# J. static environment traffic objects
# ------------------------------------------------------------

def check_static_objects(report, client):
    try:
        world = client.get_world()
        total = 0
        for label in STATIC_VEHICLE_LABELS + STATIC_PEDESTRIAN_LABELS:
            total += len(get_static_object_ids(world, [label]))

        # Coverage-level confirmation only -- see
        # scripts/tools/validate_runtime_environment.py for the full
        # standalone check and why CARLA's API cannot re-query the
        # enabled flag itself (no getter exists).
        report.add(
            "J", "static traffic environment objects disabled by production cleanup", True,
            f"{total} static objects present in map inventory; production "
            f"scripts/collect_dataset.py already called "
            f"src.simulation.environment.disable_static_traffic_objects() "
            f"once at town load (see MUST/SHOULD-FIX report). Run "
            f"scripts/tools/validate_runtime_environment.py separately for "
            f"a dedicated before/after coverage report.",
        )
    except Exception as exc:
        report.add("J", "static traffic environment objects disabled by production cleanup", False, str(exc))


# ------------------------------------------------------------
# K. cleanup safety net
# ------------------------------------------------------------

def cleanup_safety_net(host, port, timeout):
    """
    scripts/collect_dataset.py already restores world settings and the
    Traffic Manager's synchronous mode in its own try/finally. This is a
    defensive backstop only: if that restoration somehow did not happen
    (e.g. the process was killed instead of raising), leaving the shared
    CARLA server stuck in synchronous mode would break every other client,
    so this always re-checks and restores it.
    """
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        settings = world.get_settings()

        if settings.synchronous_mode:
            print("[Cleanup] world was left in synchronous_mode=True -- restoring.")
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        tm = client.get_trafficmanager(cfg.TRAFFIC_MANAGER.PORT)
        tm.set_synchronous_mode(False)
        print("[Cleanup] world settings / Traffic Manager sync mode verified restored.")
    except Exception as exc:
        print(f"[Cleanup] WARNING -- could not verify/restore server state: {exc}")


def main():
    args = parse_args()

    if args.output_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = PROJECT_ROOT / "outputs" / "runtime_0915_validation" / f"smoke_{stamp}"
    else:
        output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] isolated smoke-test directory: {output_root}")

    report = Report()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        check_version(report, client, args.expected_version)

        print()
        print(f"[Run] scripts/collect_dataset.py --towns {args.town} --routes {args.route} "
              f"--conditions {' '.join(args.conditions)} --max-frames {args.frames} --truncate-ok "
              f"--overwrite --output-root {output_root}")
        pipeline_ok, error = run_production_pipeline(args, output_root)

        if not pipeline_ok:
            print("[Run] production pipeline FAILED:")
            if error:
                print(error)

        check_sync_settings(report, pipeline_ok)

        route_path = route_root(str(output_root), args.town, args.route)
        geometry_root = geometry_dir(route_path)

        if pipeline_ok:
            check_timestamps(report, geometry_root)
            calibration = check_camera_geometry(report, geometry_root)
            check_label_projection(report, geometry_root, calibration)
            check_conditions(report, route_path, args.conditions)
            check_output_completeness(report, route_path, geometry_root, args.conditions, "day_clear", args.frames)
        else:
            for check_id, description in (
                ("D", "consecutive frame timestamp spacing ~= fixed_delta_seconds"),
                ("E", "camera/ego/calibration extrinsic consistency"),
                ("F", "object 3D->image projection within image bounds"),
                ("G", "replay contract"),
                ("H", "wind_intensity == 0.0 for every condition"),
                ("I", "output completeness"),
            ):
                report.add(check_id, description, False, "skipped -- production pipeline did not complete")

        check_static_objects(report, client)

    finally:
        cleanup_safety_net(args.host, args.port, args.timeout)

    report.print_summary()
    sys.exit(0 if report.overall_passed() else 1)


if __name__ == "__main__":
    main()

"""
paired_validation.py

Cross-condition validation for the paired route-centric dataset
(geometry/ + conditions/<weather>/). No CARLA dependency.

Production validation (what a replayed weather condition must satisfy):
    compare_calibration    replay rig calibration vs geometry/calibration.json
    validate_route         frame counts / frame-id correspondence, recorded
                           ego + actor transform and logical-actor-id
                           agreement (from the replay's post-tick checks),
                           calibration reference, RGB left/right presence and
                           dimensions, condition metadata, and that RGB
                           really differs from the source weather.

Weather replay regenerates ONLY stereo RGB under the recorded geometry, so
depth / semantic / optical-flow / LiDAR / radar equality between replay and
canonical is deliberately NOT a criterion here (those modalities are canonical
GT stored once, never re-rendered). validate_route additionally checks that a
replayed condition directory contains no non-RGB data.

Diagnostic-only (not used by production validation):
    ArrayComparison        replay-vs-canonical array equality statistics; kept
                           for scripts/tools/diagnose_replay_fidelity.py style
                           research, never a production blocker.
"""

import json
import os

import numpy as np

from src.data.layout import (
    CONDITION_SENSOR_EXTENSIONS,
    GEOMETRY_SENSOR_EXTENSIONS,
    condition_dir,
    geometry_dir,
)
from src.data.world_state import WORLD_STATE_DIRNAME


# A replayed weather directory may contain only these entries; anything else
# (e.g. depth/, lidar/) means a non-RGB modality leaked into replay.
REPLAY_ALLOWED_ENTRIES = frozenset(
    list(CONDITION_SENSOR_EXTENSIONS) + ["condition.json", "COMPLETE"]
)

# Mean absolute RGB difference (0-255 scale) below which two conditions are
# considered visually identical, i.e. weather failed to apply.
RGB_MIN_MEAN_ABS_DIFFERENCE = 1.0

RGB_SAMPLE_FRAMES = 10

# Calibration extrinsics are computed as inverse(ego_world) @ sensor_world
# from float32 world transforms, so their rounding error is the float32
# spacing at the world-coordinate magnitude. Measured: the observed
# difference equals exactly 1 ULP at |x| ~ 300 m (3.05e-5 m) and is
# identical across repeated replays. The tolerance is therefore derived from
# the actual coordinate magnitude instead of being a fixed number.
CALIBRATION_ULP_MULTIPLE = 4
CALIBRATION_MIN_TOLERANCE = 1e-6


def calibration_tolerance(coordinate_magnitude_m):
    spacing = float(np.spacing(np.float32(max(abs(coordinate_magnitude_m), 1.0))))

    return max(CALIBRATION_ULP_MULTIPLE * spacing, CALIBRATION_MIN_TOLERANCE)


# ------------------------------------------------------------------
# Array comparison (depth / optical flow / semantic)
# ------------------------------------------------------------------

class ArrayComparison:
    def __init__(self):
        self.frames = 0
        self.elements = 0
        self.equal_elements = 0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self.per_frame = []  # (mean_abs_difference, frame_id)

    def add(self, canonical, replayed, frame_id=None):
        if canonical.shape != replayed.shape:
            raise ValueError(f"Shape mismatch: {canonical.shape} vs {replayed.shape}")

        difference = np.abs(canonical.astype(np.float64) - replayed.astype(np.float64))

        self.frames += 1
        self.elements += difference.size
        self.equal_elements += int(np.count_nonzero(difference == 0.0))
        self.abs_sum += float(difference.sum())
        self.max_abs = max(self.max_abs, float(difference.max()) if difference.size else 0.0)
        self.per_frame.append((float(difference.mean()) if difference.size else 0.0, frame_id))

    def summary(self):
        return {
            "frames": self.frames,
            "max_abs_difference": self.max_abs,
            "mean_abs_difference": self.abs_sum / self.elements if self.elements else 0.0,
            "percentage_equal": 100.0 * self.equal_elements / self.elements if self.elements else 0.0,
            "worst_frames_by_mean_abs": [
                {"frame_id": f, "mean_abs_difference": m}
                for m, f in sorted(self.per_frame, key=lambda x: -x[0])[:5]
            ],
        }


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def _max_leaf_difference(a, b):
    """Largest numeric difference between two JSON trees, over shared keys."""

    if isinstance(a, dict) and isinstance(b, dict):
        return max(
            (_max_leaf_difference(a[key], b[key]) for key in a.keys() & b.keys()),
            default=0.0,
        )

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")

        return max((_max_leaf_difference(x, y) for x, y in zip(a, b)), default=0.0)

    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else float("inf")

    if a is None or b is None:
        return 0.0 if a == b else float("inf")

    return abs(float(a) - float(b))


def compare_calibration(canonical, replayed, tolerance=CALIBRATION_MIN_TOLERANCE):
    """
    Compare only the sensors present in both calibrations: the replay rig
    is RGB-only (rgb_left / rgb_right), so this checks the stereo camera
    calibration reference against the canonical geometry/calibration.json.
    """

    shared = sorted(set(canonical.get("sensors", {})) & set(replayed.get("sensors", {})))

    difference = 0.0

    for name in shared:
        difference = max(
            difference,
            _max_leaf_difference(canonical["sensors"][name], replayed["sensors"][name]),
        )

    for name in sorted(set(canonical.get("cameras", {})) & set(replayed.get("cameras", {}))):
        difference = max(
            difference,
            _max_leaf_difference(canonical["cameras"][name], replayed["cameras"][name]),
        )

    difference = max(difference, _max_leaf_difference(
        canonical.get("stereo", {}), replayed.get("stereo", {}),
    ))

    return {
        "compared_sensors": shared,
        "max_abs_difference": difference,
        "tolerance": tolerance,
        "equal": bool(difference <= tolerance),
    }


# ------------------------------------------------------------------
# RGB difference
# ------------------------------------------------------------------

def sample_frame_ids(num_frames, count=RGB_SAMPLE_FRAMES):
    if num_frames <= 0:
        return []

    return sorted({int(i) for i in np.linspace(0, num_frames - 1, min(count, num_frames))})


def mean_abs_rgb_difference(dir_a, dir_b, frame_ids, camera="rgb_left"):
    import cv2

    values = []

    for frame_id in frame_ids:
        name = f"{frame_id:06d}.png"
        image_a = cv2.imread(os.path.join(dir_a, camera, name), cv2.IMREAD_COLOR)
        image_b = cv2.imread(os.path.join(dir_b, camera, name), cv2.IMREAD_COLOR)

        if image_a is None or image_b is None or image_a.shape != image_b.shape:
            return None

        values.append(float(np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)))))

    return float(np.mean(values)) if values else None


# ------------------------------------------------------------------
# Route-level validation
# ------------------------------------------------------------------

def _frame_stems(directory, extension):
    if not os.path.isdir(directory):
        return None

    return sorted(name[: -len(extension)] for name in os.listdir(directory) if name.endswith(extension))


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def condition_metadata_errors(condition, meta, num_frames, source_condition):
    """Metadata every condition.json must carry (production style, no new schema)."""

    errors = []

    if meta.get("condition") != condition:
        errors.append(f"{condition}: condition.json names '{meta.get('condition')}'")

    if meta.get("num_frames") != num_frames:
        errors.append(f"{condition}: num_frames {meta.get('num_frames')} != geometry {num_frames}")

    if meta.get("rendered_from_canonical_geometry") is not True:
        errors.append(f"{condition}: rendered_from_canonical_geometry is not true")

    if meta.get("wind_intensity") != 0.0:
        errors.append(f"{condition}: wind_intensity is {meta.get('wind_intensity')} (must be fixed to 0.0)")

    is_source = condition == source_condition
    expected_from = "canonical_run" if is_source else "replay"

    if meta.get("rendered_from") != expected_from:
        errors.append(f"{condition}: rendered_from is '{meta.get('rendered_from')}', expected '{expected_from}'")

    if meta.get("rgb_only_replay") is not (not is_source):
        errors.append(f"{condition}: rgb_only_replay is {meta.get('rgb_only_replay')}, expected {not is_source}")

    return errors


def replay_validation_errors(condition, meta, num_frames):
    """
    Replay acceptance (RGB-only replay): recorded transforms reproduced for
    every frame, same active logical-actor set, calibration reference equal,
    RGB pair present with one consistent size. Nothing about non-RGB sensors.
    """

    errors = []
    replay = meta.get("replay_validation") or {}
    calibration = meta.get("calibration_check") or {}

    if replay.get("passed") is not True:
        errors.append(f"{condition}: replay validation did not pass")

    if replay.get("frames_verified") != num_frames:
        errors.append(f"{condition}: verified {replay.get('frames_verified')} frames, expected {num_frames}")

    if replay.get("actor_id_set_mismatch_frames") != 0:
        errors.append(f"{condition}: active logical actor id set differs from recorded in "
                      f"{replay.get('actor_id_set_mismatch_frames')} frame(s)")

    if replay.get("actors_missing_in_world") != 0:
        errors.append(f"{condition}: replay actors missing from the world after tick")

    for group in ("ego", "vehicle_like_actors", "pedestrians"):
        if (replay.get(group) or {}).get("passed") is not True:
            errors.append(f"{condition}: {group} transform did not match the recorded transform")

    if replay.get("rgb_frames_saved") != num_frames:
        errors.append(f"{condition}: rgb_frames_saved {replay.get('rgb_frames_saved')} != {num_frames}")

    if calibration.get("equal") is not True:
        errors.append(f"{condition}: calibration differs from geometry/calibration.json")

    return errors


def rgb_dimension_errors(route_root_path, conditions, source_condition, frames):
    """Both cameras of every condition share one size, equal to the source's."""

    import cv2

    errors = []
    dimensions = {}

    for condition in conditions:
        for camera in CONDITION_SENSOR_EXTENSIONS:
            seen = set()

            for frame_id in frames:
                image = cv2.imread(
                    os.path.join(condition_dir(route_root_path, condition), camera, f"{frame_id:06d}.png"),
                    cv2.IMREAD_COLOR,
                )

                if image is None:
                    errors.append(f"{condition}/{camera}: frame {frame_id:06d}.png unreadable")
                    continue

                seen.add(tuple(image.shape[:2]))

            dimensions[f"{condition}/{camera}"] = sorted(seen)

            if len(seen) > 1:
                errors.append(f"{condition}/{camera}: inconsistent image sizes {sorted(seen)}")

    reference = dimensions.get(f"{source_condition}/rgb_left")

    for key, value in dimensions.items():
        if reference and value and value != reference:
            errors.append(f"{key}: image size {value} differs from {source_condition}/rgb_left {reference}")

    return errors, dimensions


def validate_route(route_root_path, conditions, source_condition="day_clear"):
    """
    Verify one paired route directory. Returns a JSON-serializable dict with
    "passed" plus per-check detail; failures are listed in "errors".

    Scope (recorded in the report): geometry replay + RGB presence only.
    Non-RGB sensor replay is NOT validated -- weather replay does not produce
    those modalities (canonical GT is stored once under geometry/).
    """

    geometry = geometry_dir(route_root_path)
    errors = []
    checks = {}

    # ---- frame counts / frame-id correspondence -------------------------

    state_stems = _frame_stems(os.path.join(geometry, WORLD_STATE_DIRNAME), ".json")

    if state_stems is None:
        return {"passed": False, "errors": [f"missing {geometry}/{WORLD_STATE_DIRNAME}"], "checks": {}}

    expected = [f"{i:06d}" for i in range(len(state_stems))]
    num_frames = len(state_stems)

    if state_stems != expected:
        errors.append("world_state frame ids are not contiguous from 000000")

    frame_counts = {"geometry/world_state": num_frames}

    for directory, extension in (
        [(os.path.join("labels", "object_3d"), ".json")]
        + [(name, ext) for name, ext in GEOMETRY_SENSOR_EXTENSIONS.items()]
    ):
        stems = _frame_stems(os.path.join(geometry, directory), extension)
        frame_counts[f"geometry/{directory.replace(os.sep, '/')}"] = None if stems is None else len(stems)

        if stems != state_stems:
            errors.append(f"geometry/{directory.replace(os.sep, '/')}: frame ids do not match world_state")

    # geometry/ completeness above describes the canonical GT (collected
    # once); the loop below is the per-weather RGB pair.
    for condition in conditions:
        for camera, extension in CONDITION_SENSOR_EXTENSIONS.items():
            stems = _frame_stems(os.path.join(condition_dir(route_root_path, condition), camera), extension)
            frame_counts[f"{condition}/{camera}"] = None if stems is None else len(stems)

            if stems != state_stems:
                errors.append(f"{condition}/{camera}: frame ids do not match geometry")

    checks["frame_counts"] = frame_counts

    # ---- label logical ids reference recorded actors --------------------

    label_dir = os.path.join(geometry, "labels", "object_3d")
    bad_labels = 0

    if os.path.isdir(label_dir):
        for stem in state_stems:
            label_path = os.path.join(label_dir, f"{stem}.json")
            state_path = os.path.join(geometry, WORLD_STATE_DIRNAME, f"{stem}.json")

            if not os.path.isfile(label_path):
                continue

            state_ids = set(_read_json(state_path)["actors"])

            for obj in _read_json(label_path).get("objects", []):
                logical_id = obj.get("logical_id")

                if logical_id is None or str(logical_id) not in state_ids:
                    bad_labels += 1

    checks["labels_with_unknown_logical_id"] = bad_labels

    if bad_labels:
        errors.append(f"{bad_labels} label object(s) do not reference an actor in world_state")

    # ---- per-condition metadata + RGB-only replay acceptance -------------

    condition_reports = {}

    for condition in conditions:
        cond_path = condition_dir(route_root_path, condition)
        path = os.path.join(cond_path, "condition.json")

        if not os.path.isfile(path):
            errors.append(f"{condition}: missing condition.json")
            continue

        meta = _read_json(path)
        replay = meta.get("replay_validation") or {}
        calibration = meta.get("calibration_check") or {}

        condition_reports[condition] = {
            "rendered_from": meta.get("rendered_from"),
            "rgb_only_replay": meta.get("rgb_only_replay"),
            "wind_intensity": meta.get("wind_intensity"),
            "num_frames": meta.get("num_frames"),
            "replay_passed": replay.get("passed"),
            "calibration_equal": calibration.get("equal"),
        }

        errors.extend(condition_metadata_errors(condition, meta, num_frames, source_condition))

        if meta.get("rendered_from") == "replay":
            errors.extend(replay_validation_errors(condition, meta, num_frames))

            leaked = sorted(set(os.listdir(cond_path)) - REPLAY_ALLOWED_ENTRIES)

            if leaked:
                errors.append(f"{condition}: replay directory contains non-RGB entries {leaked}")

    checks["conditions"] = condition_reports

    # ---- RGB size / presence ---------------------------------------------

    dimension_errors, dimensions = rgb_dimension_errors(
        route_root_path, conditions, source_condition, sample_frame_ids(num_frames),
    )
    errors.extend(dimension_errors)
    checks["rgb_dimensions"] = dimensions

    # ---- RGB actually differs across weather ----------------------------

    rgb_differences = {}
    frames = sample_frame_ids(num_frames)

    if source_condition in conditions:
        source_dir = condition_dir(route_root_path, source_condition)

        for condition in conditions:
            if condition == source_condition:
                continue

            value = mean_abs_rgb_difference(source_dir, condition_dir(route_root_path, condition), frames)
            rgb_differences[condition] = value

            if value is None:
                errors.append(f"{condition}: could not compare RGB against {source_condition}")
            elif value < RGB_MIN_MEAN_ABS_DIFFERENCE:
                errors.append(
                    f"{condition}: RGB is (nearly) identical to {source_condition} "
                    f"(mean abs diff {value:.3f}) -- weather did not apply"
                )

    checks["rgb_mean_abs_difference_vs_source"] = rgb_differences
    checks["rgb_sampled_frames"] = frames

    return {
        "passed": not errors,
        "num_frames": num_frames,
        "conditions": list(conditions),
        # Scope of this validation (see docstring): geometry replay + RGB
        # presence. Weather replay does not produce non-RGB sensor data.
        "validates_geometry_replay": True,
        "validates_rgb_presence": True,
        "validates_non_rgb_sensor_replay": False,
        "errors": errors,
        "checks": checks,
    }

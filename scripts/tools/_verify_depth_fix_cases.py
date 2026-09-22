"""
scripts/tools/_verify_depth_fix_cases.py

One-off Part-A4 verification helper (not a production tool): re-checks
the confirmed decision-flip cases and control cases from
outputs/annotation_debug/FINDINGS_REPORT.md against the fixed
_compute_camera_validity (via debug_annotation_visibility.recompute_
visibility, already updated to mirror the fix), reading the existing
Town01/route_1 diagnostic dataset on disk -- no CARLA, no regeneration.
"""

import sys
import os

sys.path.insert(0, r"C:\CARLA_Projcet")
sys.path.insert(0, r"C:\CARLA\PythonAPI\carla")

from src.data.projection import CameraProjector
from scripts.tools.debug_annotation_visibility import (
    sequence_dirs, load_calibration, load_annotation, load_depth, build_objects_with_rv,
)

dirs = sequence_dirs("outputs/annotation_debug_dataset", "Town01", "1", "day_clear")
calib = load_calibration(dirs["geometry"])
proj = CameraProjector.from_calibration(calib, "rgb_left")
geo_dir = dirs["geometry"]

flip_cases = [
    (0, 7), (1, 7), (2, 7), (28, 8), (40, 9), (70, 9),
    (143, 6), (144, 6), (145, 6), (146, 6), (147, 6), (153, 6), (154, 6),
]

print("=== decision-flip cases: after-fix camera_valid should now be False ===")
for fr, lid in flip_cases:
    ann = load_annotation(geo_dir, fr)
    depth_m = load_depth(geo_dir, fr)
    bundle = build_objects_with_rv(ann, proj, depth_m)
    for obj, rv, rect in bundle:
        if obj.get("logical_id") == lid:
            cp = obj["camera_projection"]
            print(f"frame={fr} lid={lid} BEFORE(stored) valid={cp['camera_valid']} vf={cp['visible_fraction']:.3f}"
                  f"  ->  AFTER(fixed) valid={rv['camera_valid']} vf={rv['visible_fraction']:.3f} reason={rv['invalid_reason']}")

control_cases = [
    (0, 2, 0, "fully_occluded_expect_invalid"),
    (0, 2, 1, "fully_occluded_expect_invalid"),
    (0, 2, 4, "fully_occluded_expect_invalid"),
    (9, 2, 5, "partial_expect_valid"),
    (10, 2, 5, "partial_expect_valid"),
    (11, 2, 5, "partial_expect_valid"),
    (0, 2, 3, "no_overlap_expect_valid"),
    (0, 2, 8, "no_overlap_expect_valid"),
    (0, 3, 5, "no_overlap_expect_valid"),
]

print()
print("=== control cases: after-fix outcome should still match expectation ===")
for fr, near_lid, far_lid, note in control_cases:
    ann = load_annotation(geo_dir, fr)
    depth_m = load_depth(geo_dir, fr)
    bundle = build_objects_with_rv(ann, proj, depth_m)
    for obj, rv, rect in bundle:
        if obj.get("logical_id") == far_lid:
            cp = obj["camera_projection"]
            print(f"frame={fr} far_lid={far_lid} note={note} BEFORE(stored) valid={cp['camera_valid']}"
                  f"  ->  AFTER(fixed) valid={rv['camera_valid']} vf={rv['visible_fraction']:.3f}")

print()
print("=== boundary truncation control (image_fraction driven, unaffected by this fix) ===")
for fr, lid in [(26, 8), (27, 8), (28, 8)]:
    ann = load_annotation(geo_dir, fr)
    depth_m = load_depth(geo_dir, fr)
    bundle = build_objects_with_rv(ann, proj, depth_m)
    for obj, rv, rect in bundle:
        if obj.get("logical_id") == lid:
            print(f"frame={fr} lid={lid} image_fraction={rv['image_fraction']:.3f} after_valid={rv['camera_valid']}")

# route-wide re-scan for any remaining/newly-introduced decision-flip cases
print()
print("=== route-wide rescan (all 560 frames): remaining decision-flip candidates after fix ===")
frames = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(geo_dir, "labels", "object_3d")) if f.endswith(".json"))
remaining = 0
for fr in frames:
    ann = load_annotation(geo_dir, fr)
    depth_m = load_depth(geo_dir, fr)
    bundle = build_objects_with_rv(ann, proj, depth_m)
    for obj, rv, rect in bundle:
        cp = obj.get("camera_projection", {})
        if cp.get("camera_valid") is False and rv["camera_valid"] is True:
            remaining += 1
            print(f"NEW camera_valid=True after fix (was False before) at frame={fr} lid={obj.get('logical_id')} -- unexpected, investigate")
print(f"remaining/new decision flips found: {remaining}")

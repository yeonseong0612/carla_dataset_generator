"""
scripts/tests/test_annotation_depth_visibility_offline.py

Offline regression tests (no CARLA server needed) for the depth-visibility
upper-bound fix in AnnotationWriter._compute_camera_validity
(src/data/annotation.py, CLAUDE.md PART A).

Confirmed bug case this guards against: Town01/route_1 diagnostic frame
143 (see outputs/annotation_debug/FINDINGS_REPORT.md) -- a Lincoln MKZ
~80-83m away, fully hidden behind a ~13m foreground vehicle in the RGB
image, was recorded camera_valid=true with visible_fraction~0.22-0.27
because the pre-fix depth test only checked a lower bound
(depth_patch >= near_depth - tolerance): far background pixels (a wall/
sidewalk well beyond the object's own far vertex) inside the object's
loose axis-aligned 2D bbox were miscounted as "the object visible".

These tests build a synthetic CameraProjector (src.data.projection,
reused unmodified -- see calibration.camera_intrinsic_matrix /
T_CV_FROM_CARLA, also reused unmodified) with the camera co-located with
ego (identity extrinsic), so a vertex at ego-frame x=N metres forward
projects at depth N with no coordinate-conversion arithmetic needed in
the fixtures themselves, and calls the real, unmodified
AnnotationWriter._compute_camera_validity() against synthetic depth
arrays -- not a second/parallel implementation of the fix.

Run:  python -m unittest scripts.tests.test_annotation_depth_visibility_offline -v
"""

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

from CFG.config import cfg  # noqa: E402
from src.data.annotation import AnnotationWriter  # noqa: E402
from src.data.projection import CameraProjector  # noqa: E402
from src.data.calibration import camera_intrinsic_matrix, T_CV_FROM_CARLA  # noqa: E402


def _make_projector(width=400, height=400, fov_deg=90.0):
    K = camera_intrinsic_matrix(width, height, fov_deg)

    return CameraProjector(
        K=K,
        T_camera_from_ego=np.eye(4, dtype=np.float64),  # camera co-located with ego, identity rotation
        T_cv_from_carla=T_CV_FROM_CARLA,
        width=width,
        height=height,
    )


def _box_vertices(center_x, half_length, half_width, half_height):
    """8 axis-aligned bbox vertices in ego frame (x fwd, y right, z up)."""
    vertices = []

    for dx in (-half_length, half_length):
        for dy in (-half_width, half_width):
            for dz in (-half_height, half_height):
                vertices.append([center_x + dx, dy, dz])

    return vertices


def _clipped_pixel_rect(projector, vertices):
    """
    Mirrors AnnotationWriter._compute_camera_validity's own projection +
    floor/ceil/image-clip (unchanged by this fix, not itself under test)
    -- used only to know where to paint the synthetic depth fixture.
    """

    v_cv = projector.ego_to_cv(np.asarray(vertices, dtype=np.float64))
    uv, valid = projector.project(v_cv)

    u, v = uv[valid, 0], uv[valid, 1]
    u0, u1 = max(float(u.min()), 0.0), min(float(u.max()), projector.width)
    v0, v1 = max(float(v.min()), 0.0), min(float(v.max()), projector.height)

    px0, py0 = int(math.floor(u0)), int(math.floor(v0))
    px1, py1 = int(math.ceil(u1)), int(math.ceil(v1))

    return px0, py0, px1, py1


class DepthVisibilityUpperBoundTest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.projector = _make_projector()
        self.writer = AnnotationWriter(self.tmp_dir, cfg)
        self.writer.projector = self.projector

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _depth_image(self, fill):
        return np.full((self.projector.height, self.projector.width), fill, dtype=np.float32)

    def test_fully_occluded_far_object_background_leakage_excluded(self):
        """
        Confirmed bug case (frame 143 Lincoln MKZ): object at 75-85m, bbox
        interior contains ONLY a near foreground surface (13m, T2-like)
        and far background (150m, wall/street beyond the object) -- no
        pixel anywhere in the bbox is at the object's own depth, i.e. it
        is fully occluded.

        Pre-fix, the one-sided formula (depth_patch >= near_depth - tol)
        counted every far-background(150m) pixel as "visible" -- enough
        to clear MIN_VISIBLE_FRACTION for an 80m object. Post-fix, those
        same pixels are excluded (150 > far_depth + tol) too, so
        visible_fraction collapses to 0 and camera_valid is correctly
        False / too_occluded.
        """
        vertices = _box_vertices(center_x=80.0, half_length=5.0, half_width=30.0, half_height=15.0)
        px0, py0, px1, py1 = _clipped_pixel_rect(self.projector, vertices)
        self.assertGreater(px1 - px0, 4)
        self.assertGreater(py1 - py0, 4)

        depth_m = self._depth_image(150.0)  # far background everywhere
        mid = (px0 + px1) // 2
        depth_m[py0:py1, px0:mid] = 13.0     # near foreground occluder
        # depth_m[py0:py1, mid:px1] stays 150.0 (far background leakage)

        near_depth = float(self.projector.ego_to_cv(np.asarray(vertices))[:, 2].min())
        old_visible_fraction = float(
            (depth_m[py0:py1, px0:px1] >= (near_depth - 0.5)).mean()
        )
        self.assertGreaterEqual(
            old_visible_fraction, cfg.ANNOTATION.MIN_VISIBLE_FRACTION,
            "sanity check: the pre-fix one-sided formula should have wrongly "
            "passed this fully-occluded case -- if this fails, the fixture "
            "no longer reproduces the confirmed bug",
        )

        metrics = self.writer._compute_camera_validity(vertices, depth_m)

        self.assertEqual(metrics["visible_fraction"], 0.0)
        self.assertFalse(metrics["camera_valid"])
        self.assertEqual(metrics["invalid_reason"], "too_occluded")

    def test_fully_visible_object_unchanged(self):
        """
        A fully-visible object (depth fills its own bbox entirely) is
        unaffected by the upper bound -- its own depth is always inside
        its own [near_depth, far_depth] range.
        """
        vertices = _box_vertices(center_x=40.0, half_length=4.0, half_width=25.0, half_height=15.0)
        px0, py0, px1, py1 = _clipped_pixel_rect(self.projector, vertices)
        self.assertGreaterEqual(px1 - px0, cfg.ANNOTATION.MIN_BBOX_WIDTH_PX)
        self.assertGreaterEqual(py1 - py0, cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX)

        depth_m = self._depth_image(40.0)  # object fills the whole frame at its own depth

        metrics = self.writer._compute_camera_validity(vertices, depth_m)

        self.assertEqual(metrics["visible_fraction"], 1.0)
        self.assertTrue(metrics["camera_valid"])
        self.assertIsNone(metrics["invalid_reason"])

    def test_partially_visible_object_only_own_depth_range_counts(self):
        """
        Half the bbox at the object's own depth, half at a nearer
        occluder -- only the object-depth half should count as visible,
        and camera_valid should follow MIN_VISIBLE_FRACTION accordingly.
        """
        vertices = _box_vertices(center_x=50.0, half_length=4.0, half_width=30.0, half_height=15.0)
        px0, py0, px1, py1 = _clipped_pixel_rect(self.projector, vertices)
        mid = (px0 + px1) // 2
        self.assertGreater(mid, px0)
        self.assertGreater(px1, mid)

        depth_m = self._depth_image(500.0)
        depth_m[py0:py1, px0:mid] = 10.0     # near occluder
        depth_m[py0:py1, mid:px1] = 50.0     # object's own depth

        metrics = self.writer._compute_camera_validity(vertices, depth_m)

        expected_fraction = (px1 - mid) / (px1 - px0)
        self.assertAlmostEqual(metrics["visible_fraction"], expected_fraction, places=2)
        self.assertEqual(
            metrics["camera_valid"],
            expected_fraction >= cfg.ANNOTATION.MIN_VISIBLE_FRACTION,
        )

    def test_background_leakage_three_band_case(self):
        """
        CLAUDE.md PART A worked example exactly: object range 75-85m,
        bbox depth = [13m foreground | 80m object | 150m background].
        Expect: 13m excluded, 80m included, 150m excluded.
        """
        vertices = _box_vertices(center_x=80.0, half_length=5.0, half_width=40.0, half_height=15.0)
        px0, py0, px1, py1 = _clipped_pixel_rect(self.projector, vertices)
        third = (px1 - px0) // 3
        self.assertGreater(third, 0)

        depth_m = self._depth_image(150.0)
        depth_m[py0:py1, px0:px0 + third] = 13.0
        depth_m[py0:py1, px0 + third:px0 + 2 * third] = 80.0
        # remaining band (px0 + 2*third : px1) stays 150.0 (background)

        metrics = self.writer._compute_camera_validity(vertices, depth_m)

        total = (px1 - px0) * (py1 - py0)
        middle_band = third * (py1 - py0)
        expected_fraction = middle_band / total
        self.assertAlmostEqual(metrics["visible_fraction"], expected_fraction, places=2)


if __name__ == "__main__":
    unittest.main()

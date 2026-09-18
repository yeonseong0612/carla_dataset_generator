"""
src/data/projection.py

Shared ego-frame -> image-pixel camera projection. Extracted from
scripts/tools/visualize_annotations.py's CameraProjector (the existing
projection helper -- CLAUDE.md task "Finalize Camera-Valid Annotation
Filtering" requires reusing it, not inventing a new one) so both that
post-hoc debug tool AND the live annotation pipeline
(src/data/annotation.py) share exactly one implementation.

Same convention both already used independently before this module
existed:

    vertices_ego_m (ego frame)
        -> T_camera_from_ego -> CARLA camera frame (x fwd, y right, z up)
        -> T_cv_from_carla   -> CV camera frame (x right, y down, z fwd)
        -> K                 -> (u, v)

p_A = T_A_from_B @ p_B (same convention as src/data/calibration.py).
"""

import numpy as np

# A vertex with z_cv <= this is behind (or level with) the camera and
# cannot be projected (division by a non-positive z is meaningless).
Z_CV_EPS = 1e-3


class CameraProjector:
    def __init__(self, K, T_camera_from_ego, T_cv_from_carla, width, height):
        self.K = np.asarray(K, dtype=np.float64)
        self.T_camera_from_ego = np.asarray(T_camera_from_ego, dtype=np.float64)
        self.T_cv_from_carla = np.asarray(T_cv_from_carla, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)

    @classmethod
    def from_calibration(cls, calibration, camera_name):
        """Construct from an already-loaded calibration.json dict (post-hoc
        tools -- e.g. scripts/tools/visualize_annotations.py)."""

        cam = calibration["cameras"][camera_name]

        return cls(
            K=cam["K"],
            T_camera_from_ego=cam["T_camera_from_ego"],
            T_cv_from_carla=calibration["coordinate_systems"]["T_cv_from_carla"],
            width=cam["width"],
            height=cam["height"],
        )

    @classmethod
    def from_live_actors(cls, ego_vehicle, camera_actor):
        """
        Construct directly from live CARLA actors (the annotation
        pipeline, src/data/annotation.py) -- reuses
        src/data/calibration.py's own extraction functions verbatim
        (same values save_calibration() already writes to
        calibration.json for this same camera), so this is not a second
        source of truth for K/extrinsics, just a second consumer of the
        same one. Call once (e.g. AnnotationWriter.__init__) and reuse --
        a rigidly-mounted camera's transform relative to ego does not
        change across a sequence, exactly like calibration.json itself
        being written once at sequence start.
        """

        from src.data.calibration import (
            get_camera_calibration,
            get_sensor_extrinsic,
            invert_transform,
            T_CV_FROM_CARLA,
        )

        camera_info = get_camera_calibration(camera_actor)
        T_ego_from_camera = get_sensor_extrinsic(ego_vehicle, camera_actor)
        T_camera_from_ego = invert_transform(T_ego_from_camera)

        return cls(
            K=camera_info["K"],
            T_camera_from_ego=T_camera_from_ego,
            T_cv_from_carla=T_CV_FROM_CARLA,
            width=camera_info["width"],
            height=camera_info["height"],
        )

    def ego_to_cv(self, points_ego):
        """
        points_ego: (N, 3) array in the ego frame.
        Returns (N, 3) array in the CV camera frame (x right, y down, z fwd).
        """

        points_ego = np.asarray(points_ego, dtype=np.float64)
        n = points_ego.shape[0]
        homo = np.hstack([points_ego, np.ones((n, 1))])

        p_camera_carla = (self.T_camera_from_ego @ homo.T).T[:, :3]

        homo_cam = np.hstack([p_camera_carla, np.ones((n, 1))])
        p_cv = (self.T_cv_from_carla @ homo_cam.T).T[:, :3]

        return p_cv

    def project(self, points_cv):
        """
        points_cv: (N, 3) in CV camera frame.
        Returns (uv (N,2), valid (N,) bool) where valid marks z_cv > eps
        (see Z_CV_EPS -- behind-camera points are NOT projected, robust
        against garbage/division-by-non-positive-z rather than
        projecting a meaningless coordinate).
        """

        points_cv = np.asarray(points_cv, dtype=np.float64)
        z = points_cv[:, 2]
        valid = z > Z_CV_EPS

        uv = np.full((points_cv.shape[0], 2), np.nan, dtype=np.float64)

        if np.any(valid):
            xf = points_cv[valid, 0] / z[valid]
            yf = points_cv[valid, 1] / z[valid]

            uv[valid, 0] = self.K[0, 0] * xf + self.K[0, 2]
            uv[valid, 1] = self.K[1, 1] * yf + self.K[1, 2]

        return uv, valid

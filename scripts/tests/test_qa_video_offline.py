"""
scripts/tests/test_qa_video_offline.py

Offline (no CARLA server) tests for scripts/tools/make_qa_videos.py on a tiny
synthetic route directory laid out like src/data/layout.py + collector.py:

    route_0/geometry/{calibration.json, sequence.json, timestamps.csv,
                      ego_state.csv, depth/, semantic/, optical_flow/, lidar/,
                      radar/, radar_front_left/, radar_front_right/,
                      labels/object_3d/}
    route_0/conditions/<weather>/{rgb_left/, rgb_right/, condition.json}

Run:
    python -m unittest scripts.tests.test_qa_video_offline -v
"""

import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from CFG.config import cfg  # noqa: E402
from src.data.calibration import (  # noqa: E402
    T_CV_FROM_CARLA,
    camera_intrinsic_matrix,
    invert_transform,
)

import scripts.tools.make_qa_videos as qa  # noqa: E402

W, H = 1024, 768
FRAMES = (0, 1, 2, 3)
CONDITIONS = [c for row in qa.WEATHER_GRID for c in row]


def camera_entry(y_offset):
    T_ego_from_cam = np.eye(4)
    T_ego_from_cam[:3, 3] = [cfg.SENSOR.CAMERA.X, y_offset, cfg.SENSOR.CAMERA.Z]
    K = camera_intrinsic_matrix(W, H, cfg.SENSOR.CAMERA.FOV)

    return {
        "width": W, "height": H, "fov_deg": float(cfg.SENSOR.CAMERA.FOV), "K": K.tolist(),
        "T_ego_from_camera": T_ego_from_cam.tolist(),
        "T_camera_from_ego": invert_transform(T_ego_from_cam).tolist(),
    }


def write_route(root, drop=None):
    """drop: {modality_or_condition: [frame ids to omit]}"""

    drop = drop or {}
    geometry = os.path.join(root, "geometry")

    sensors = {"lidar": qa.cfg_extrinsic(cfg.SENSOR.LIDAR)}

    for name, attr in qa.RADAR_CFG_ATTR.items():
        sensors[name] = qa.cfg_extrinsic(getattr(cfg.SENSOR, attr))

    calibration = {
        "coordinate_systems": {"T_cv_from_carla": T_CV_FROM_CARLA.tolist()},
        "cameras": {
            "rgb_left": camera_entry(cfg.SENSOR.STEREO.LEFT_Y),
            "rgb_right": camera_entry(cfg.SENSOR.STEREO.RIGHT_Y),
        },
        "sensors": {n: {"T_ego_from_sensor": T.tolist()} for n, T in sensors.items()},
    }

    os.makedirs(geometry, exist_ok=True)

    with open(os.path.join(geometry, "calibration.json"), "w") as f:
        json.dump(calibration, f)

    with open(os.path.join(geometry, "sequence.json"), "w") as f:
        json.dump({"town": "Town07", "recording_hz": 10.0}, f)

    with open(os.path.join(geometry, "timestamps.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "carla_frame", "timestamp"])

        for i in FRAMES:
            writer.writerow([f"{i:06d}", 100 + 2 * i, 5.0 + 0.1 * i])

    with open(os.path.join(geometry, "ego_state.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "speed_mps", "route_progress", "goal_distance"])

        for i in FRAMES:
            writer.writerow([f"{i:06d}", 10.0, 1.5 * i, 300.0])

    rng = np.random.default_rng(0)

    def save(name, frame, array):
        if frame in drop.get(name, ()):
            return

        path = os.path.join(geometry, name, f"{frame:06d}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)

    box = [[x, y, z] for x in (18.0, 22.5) for y in (-1.0, 1.0) for z in (0.0, 1.5)]
    obj = {
        "category": "vehicle", "subcategory": "car", "actor_id": 42, "logical_id": 7,
        "bbox_3d": {"center_ego_m": [20.25, 0.0, 0.75], "yaw_ego_deg": 0.0,
                    "dimensions_m": {"length": 4.5, "width": 2.0, "height": 1.5},
                    "vertices_ego_m": box},
    }

    for i in FRAMES:
        save("depth", i, np.full((H, W), 30.0, dtype=np.float32))
        save("semantic", i, rng.integers(0, 29, (H, W), dtype=np.uint8))
        save("optical_flow", i, (rng.standard_normal((H, W, 2)) * 0.02).astype(np.float32))
        save("lidar", i, np.column_stack([rng.uniform(1, 100, 500), rng.uniform(-30, 30, 500),
                                          rng.uniform(-2, 1, 500), rng.uniform(0, 1, 500)]).astype(np.float32))

        for name in qa.RADAR_SENSOR_NAMES:
            save(name, i, np.column_stack([rng.uniform(1, 50, 50), rng.uniform(-10, 10, 50),
                                           np.zeros(50), rng.uniform(-5, 5, 50)]).astype(np.float32))

        if i not in drop.get("labels", ()):
            path = os.path.join(geometry, "labels", "object_3d", f"{i:06d}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w") as f:
                json.dump({"frame_id": i, "objects": [obj], "rejected_objects": []}, f)

    for condition in CONDITIONS:
        cond = os.path.join(root, "conditions", condition)

        for cam in ("rgb_left", "rgb_right"):
            os.makedirs(os.path.join(cond, cam), exist_ok=True)

            for i in FRAMES:
                if i in drop.get(condition, ()):
                    continue

                cv2.imwrite(os.path.join(cond, cam, f"{i:06d}.png"), np.full((H, W, 3), 90, np.uint8))

        with open(os.path.join(cond, "condition.json"), "w") as f:
            json.dump({"condition": condition, "rendered_from": "replay",
                       "weather_parameters": {"precipitation": 0.0, "fog_density": 0.0, "sun_altitude_angle": 45.0},
                       "replay_validation": {"passed": True}}, f)

    with open(os.path.join(root, "paired_validation.json"), "w") as f:
        json.dump({"passed": True, "num_frames": len(FRAMES), "errors": []}, f)


class QAVideoOfflineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.route_root = os.path.join(self.tmp, "Town07", "route_0")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def quiet(self, fn, *args, **kwargs):
        with redirect_stdout(io.StringIO()) as out:
            result = fn(*args, **kwargs)

        return result, out.getvalue()

    def test_multimodal_intersection_warns_on_gaps(self):
        write_route(self.route_root, drop={"depth": [2], "radar_front_left": [1]})
        route = qa.RouteData(self.route_root)

        (frames, available), log = self.quiet(qa.discover_multimodal_frames, route, "day_clear")

        self.assertEqual(frames, [0, 1, 3])
        self.assertIn("radar_front_left", available)
        self.assertIn("'depth' missing", log)
        self.assertIn("optional 'radar_front_left' missing", log)

    def test_weather_strict_and_intersection(self):
        write_route(self.route_root, drop={"night_fog": [3]})
        route = qa.RouteData(self.route_root)

        (frames, present), log = self.quiet(qa.discover_weather_frames, route, CONDITIONS, False)
        self.assertEqual(frames, [0, 1, 2])
        self.assertEqual(present, CONDITIONS)
        self.assertIn("night_fog", log)

        with self.assertRaises(RuntimeError):
            self.quiet(qa.discover_weather_frames, route, CONDITIONS, True)

    def test_missing_condition_is_warning_unless_strict(self):
        write_route(self.route_root)
        shutil.rmtree(os.path.join(self.route_root, "conditions", "day_fog"))
        route = qa.RouteData(self.route_root)

        (frames, present), _ = self.quiet(qa.discover_weather_frames, route, CONDITIONS, False)
        self.assertEqual(frames, list(FRAMES))
        self.assertNotIn("day_fog", present)

        with self.assertRaises(FileNotFoundError):
            self.quiet(qa.discover_weather_frames, route, CONDITIONS, True)

        grid = qa.WeatherRenderer(route, present).render(0)
        self.assertEqual(grid.shape, (768, 1536, 3))

    def test_frame_range_and_stride(self):
        self.assertEqual(qa.apply_frame_range([0, 1, 2, 3, 5, 8], 1, 5, 2), [1, 3])
        self.assertEqual(qa.apply_frame_range([4, 0, 2], None, None, 1), [0, 2, 4])

    def test_render_grids_have_expected_size(self):
        write_route(self.route_root)
        route = qa.RouteData(self.route_root)
        args = qa.build_parser().parse_args(["--route-root", self.route_root])

        (frames, available), _ = self.quiet(qa.discover_multimodal_frames, route, "day_clear")
        multimodal = qa.MultimodalRenderer(route, "day_clear", available, args).render(frames[0])
        self.assertEqual(multimodal.shape, (1152, 1536, 3))

        weather = qa.WeatherRenderer(route, CONDITIONS).render(frames[0])
        self.assertEqual(weather.shape, (768, 1536, 3))

    def test_scaled_projector_matches_full_resolution(self):
        write_route(self.route_root)
        calibration = qa.RouteData(self.route_root).calibration

        full = qa.CameraProjector.from_calibration(calibration, "rgb_left")
        half = qa.scaled_projector(calibration, "rgb_left", 0.5)

        points = np.array([[20.0, 2.0, 1.0], [40.0, -5.0, 0.0]])
        uv_full, _ = full.project(full.ego_to_cv(points))
        uv_half, _ = half.project(half.ego_to_cv(points))

        np.testing.assert_allclose(uv_half, uv_full * 0.5, atol=1e-9)

        # A point straight ahead of the left camera lands on its principal point.
        cam = np.asarray(calibration["cameras"]["rgb_left"]["T_ego_from_camera"])[:3, 3]
        uv, valid = full.project(full.ego_to_cv(np.array([[cam[0] + 30.0, cam[1], cam[2]]])))
        self.assertTrue(valid[0])
        np.testing.assert_allclose(uv[0], [W / 2, H / 2], atol=1e-6)

    def test_calibration_missing_does_not_crash(self):
        write_route(self.route_root)
        os.remove(os.path.join(self.route_root, "geometry", "calibration.json"))
        (route, _) = self.quiet(qa.RouteData, self.route_root)
        args = qa.build_parser().parse_args(["--route-root", self.route_root])

        renderer = qa.MultimodalRenderer(route, "day_clear", self.quiet(qa.discover_multimodal_frames, route, "day_clear")[0][1], args)
        self.assertEqual(renderer.lidar_source, "cfg")
        self.assertEqual(renderer.render(0).shape, (1152, 1536, 3))

    def test_semantic_palette_fallback_matches_source(self):
        palette = qa.load_semantic_map()
        self.assertEqual(palette[14][0], "car")

        lut = qa.build_semantic_lut(palette)
        self.assertEqual(tuple(lut[14]), tuple(palette[14][1][::-1]))

    def test_end_to_end_writes_both_videos(self):
        write_route(self.route_root)
        out_dir = os.path.join(self.tmp, "qa_out")

        self.quiet(qa.main, ["--route-root", self.route_root, "--output-dir", out_dir])

        for name, size in (("multimodal_day_clear.mp4", (1536, 1152)), ("weather_comparison.mp4", (1536, 768))):
            path = os.path.join(out_dir, name)
            self.assertTrue(os.path.isfile(path), path)

            capture = cv2.VideoCapture(path)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(FRAMES))
            self.assertEqual((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))), size)
            capture.release()

    def test_dataset_directory_is_not_modified(self):
        write_route(self.route_root)
        before = sorted(os.path.relpath(os.path.join(d, f), self.route_root)
                        for d, _, files in os.walk(self.route_root) for f in files)

        self.quiet(qa.main, ["--route-root", self.route_root, "--output-dir", os.path.join(self.tmp, "qa_out")])

        after = sorted(os.path.relpath(os.path.join(d, f), self.route_root)
                       for d, _, files in os.walk(self.route_root) for f in files)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

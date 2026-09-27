"""
test_doptimal_runner_offline.py

Offline (no CARLA server) tests of the D-optimal orchestration layer:
design CSV validation, condition ids, job expansion, manifest, shared
weather-profile mapping, collector command construction, resume / failed
selection and job-output verification against a fake collector tree.

Run:
    python -m unittest scripts.tests.test_doptimal_runner_offline -v
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_doptimal_dataset as runner  # noqa: E402
from CFG.config import cfg  # noqa: E402
from src.data.layout import geometry_dir, condition_dir, mark_complete  # noqa: E402
from src.simulation import weather_profiles as wp  # noqa: E402

DESIGN_CSV = PROJECT_ROOT / "configs" / "experiments" / "D_optimal_final_52_conditions.csv"


def real_rows():
    return runner.read_design_csv(DESIGN_CSV)


class DesignValidationTests(unittest.TestCase):

    def test_real_design_validates(self):
        conditions, report = runner.load_design(DESIGN_CSV)

        self.assertEqual(len(conditions), 52)
        self.assertEqual(report["num_conditions"], 52)

        for factor in ("Sun", "Rain", "Fog"):
            self.assertEqual([count for _, count in report["counts"][factor]], [13, 13, 13, 13])

    def test_condition_ids_follow_csv_row_order(self):
        conditions, _ = runner.load_design(DESIGN_CSV)
        _, rows = real_rows()

        self.assertEqual([c["condition_id"] for c in conditions], [f"C{i:03d}" for i in range(1, 53)])

        for condition, row in zip(conditions, rows):
            self.assertEqual((condition["Sun"], condition["Rain"], condition["Fog"]),
                             (int(row["Sun"]), row["Rain"], row["Fog"]))

        self.assertEqual((conditions[0]["Sun"], conditions[0]["Rain"], conditions[0]["Fog"]), (60, "Dry", "Clear"))
        self.assertEqual((conditions[-1]["Sun"], conditions[-1]["Rain"], conditions[-1]["Fog"]), (-30, "Heavy", "Heavy"))

    def assertInvalid(self, header, rows, fragment):
        with self.assertRaises(runner.DesignValidationError) as ctx:
            runner.validate_design(header, rows)

        self.assertIn(fragment, str(ctx.exception))

    def test_missing_column(self):
        header, rows = real_rows()
        self.assertInvalid(["Time", "Rain", "Fog"], rows, "missing required column")

    def test_wrong_row_count(self):
        header, rows = real_rows()
        self.assertInvalid(header, rows[:51], "expected 52 rows")

    def test_duplicate_condition(self):
        header, rows = real_rows()
        rows = [dict(r) for r in rows]
        rows[1] = dict(rows[0])
        self.assertInvalid(header, rows, "duplicate condition")

    def test_unknown_level(self):
        header, rows = real_rows()
        rows = [dict(r) for r in rows]
        rows[0]["Fog"] = "Dense"
        self.assertInvalid(header, rows, "Fog='Dense'")

        rows = [dict(r) for r in real_rows()[1]]
        rows[0]["Sun"] = "45"
        self.assertInvalid(header, rows, "Sun=45")

    def test_unbalanced_levels(self):
        header, rows = real_rows()
        rows = [dict(r) for r in rows]
        # C001 (60,Dry,Clear) -> (60,Light,Clear): still unique, Rain unbalanced.
        rows[0]["Rain"] = "Light"
        self.assertInvalid(header, rows, "Rain=Dry appears 12 times")

    def test_non_integer_sun(self):
        header, rows = real_rows()
        rows = [dict(r) for r in rows]
        rows[3]["Sun"] = "noon"
        self.assertInvalid(header, rows, "not an integer")


class JobExpansionTests(unittest.TestCase):

    def setUp(self):
        self.conditions, _ = runner.load_design(DESIGN_CSV)
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_job_counts(self):
        for routes, expected in ((["0"], 156), (["0", "1", "2"], 468)):
            jobs = runner.expand_jobs(self.conditions, [101, 202, 303], "Town01", routes, self.root)
            self.assertEqual(len(jobs), expected)

    def test_order_ids_and_identical_seeds_per_condition(self):
        jobs = runner.expand_jobs(self.conditions, [101, 202, 303], "Town01", ["0", "1", "2"], self.root)

        self.assertEqual(jobs[0]["job_id"], "J000001")
        self.assertEqual(jobs[-1]["job_id"], "J000468")
        self.assertEqual(
            [(j["condition_id"], j["seed"], j["route_id"]) for j in jobs[:4]],
            [("C001", 101, "0"), ("C001", 101, "1"), ("C001", 101, "2"), ("C001", 202, "0")],
        )

        by_condition = {}
        for job in jobs:
            by_condition.setdefault(job["condition_id"], set()).add((job["seed"], job["route_id"]))

        self.assertEqual(len(by_condition), 52)
        for combos in by_condition.values():
            self.assertEqual(combos, {(s, r) for s in (101, 202, 303) for r in ("0", "1", "2")})

    def test_expansion_is_deterministic_and_collision_free(self):
        a = runner.expand_jobs(self.conditions, [101, 202, 303], "Town01", ["0", "1"], self.root)
        b = runner.expand_jobs(self.conditions, [101, 202, 303], "Town01", ["0", "1"], self.root)
        self.assertEqual(a, b)
        self.assertEqual(len({j["output_path"] for j in a}), len(a))

    def test_output_layout_uses_collector_layout_per_seed(self):
        job = runner.expand_jobs(self.conditions, [202], "Town01", ["1"], self.root)[16]
        expected = (self.root / "seed_202" / "Town01" / "route_1" / "conditions"
                    / "doe_C017_sun+30_rain-Light_fog-Clear")  # CSV row 17: 30,Light,Clear
        self.assertEqual(Path(job["output_path"]), expected)

    def test_duplicate_seeds_or_routes_rejected(self):
        with self.assertRaises(ValueError):
            runner.expand_jobs(self.conditions, [101, 101], "Town01", ["0"], self.root)
        with self.assertRaises(ValueError):
            runner.expand_jobs(self.conditions, [101], "Town01", ["0", "0"], self.root)

    def test_routes_come_from_route_xml(self):
        self.assertEqual(sorted(runner.route_ids_in_xml("Town01")), ["0", "1", "2"])


class ManifestTests(unittest.TestCase):

    def setUp(self):
        conditions, _ = runner.load_design(DESIGN_CSV)
        self.root = Path(tempfile.mkdtemp())
        self.jobs = runner.expand_jobs(conditions, [101, 202, 303], "Town01", ["0", "1", "2"], self.root / "data")
        self.path = self.root / "exp" / runner.MANIFEST_NAME

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_roundtrip_and_atomic_write(self):
        runner.write_manifest(self.path, self.jobs)

        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

        with open(self.path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, runner.MANIFEST_FIELDS)

        rows = runner.read_manifest(self.path)
        self.assertEqual(len(rows), 468)
        self.assertEqual(runner.merge_manifest(self.jobs, rows, self.path)[0]["status"], "pending")

    def test_merge_keeps_status_history(self):
        self.jobs[3]["status"] = "failed"
        self.jobs[3]["attempts"] = 2
        self.jobs[3]["error_summary"] = "boom"
        runner.write_manifest(self.path, self.jobs)

        fresh = [dict(j, status="pending", attempts=0, error_summary="") for j in self.jobs]
        merged = runner.merge_manifest(fresh, runner.read_manifest(self.path), self.path)

        self.assertEqual((merged[3]["status"], merged[3]["attempts"], merged[3]["error_summary"]),
                         ("failed", 2, "boom"))

    def test_merge_rejects_different_experiment(self):
        runner.write_manifest(self.path, self.jobs)
        conditions, _ = runner.load_design(DESIGN_CSV)
        other = runner.expand_jobs(conditions, [101, 202, 404], "Town01", ["0", "1", "2"], self.root / "data")

        with self.assertRaises(runner.ManifestMismatchError):
            runner.merge_manifest(other, runner.read_manifest(self.path), self.path)


class WeatherProfileTests(unittest.TestCase):

    def test_all_64_level_triples_map(self):
        for sun in wp.DOE_SUN_LEVELS:
            for rain, rain_values in wp.RAIN_PROFILES.items():
                for fog, fog_values in wp.FOG_PROFILES.items():
                    fields = wp.doe_weather_fields(sun, rain, fog)

                    self.assertEqual(set(fields), set(wp.DOE_WEATHER_FIELDS))
                    self.assertEqual(fields["sun_altitude_angle"], float(sun))
                    self.assertEqual(fields["wind_intensity"], 0.0)
                    for key, value in {**rain_values, **fog_values}.items():
                        self.assertEqual(fields[key], value)

    def test_unknown_levels_rejected(self):
        for args in ((45, "Dry", "Clear"), (60, "Drizzle", "Clear"), (60, "Dry", "Dense")):
            with self.assertRaises(ValueError):
                wp.doe_weather_fields(*args)

    def test_condition_name_roundtrip(self):
        for sun in wp.DOE_SUN_LEVELS:
            name = wp.doe_condition_name("C007", sun, "Moderate", "Heavy")
            self.assertEqual(wp.parse_doe_condition_name(name),
                             {"condition_id": "C007", "sun": sun, "rain": "Moderate", "fog": "Heavy"})

        self.assertEqual(wp.doe_condition_name("C001", -30, "Dry", "Clear"), "doe_C001_sun-30_rain-Dry_fog-Clear")
        self.assertEqual(wp.doe_condition_name("C001", 0, "Dry", "Clear"), "doe_C001_sun0_rain-Dry_fog-Clear")
        self.assertIsNone(wp.parse_doe_condition_name("day_rain"))

        with self.assertRaises(ValueError):
            wp.parse_doe_condition_name("doe_C001_sun+45_rain-Dry_fog-Clear")

    def test_collector_weather_uses_shared_mapping(self):
        from src.simulation.weather import make_weather

        name = wp.doe_condition_name("C017", 30, "Light", "Heavy")
        weather = make_weather(name)
        expected = wp.doe_weather_fields(30, "Light", "Heavy")

        for field, value in expected.items():
            self.assertAlmostEqual(getattr(weather, field), value, places=4)

    def test_legacy_conditions_unchanged(self):
        import carla
        from src.simulation.weather import make_weather

        weather = make_weather("day_rain")
        preset = carla.WeatherParameters.MidRainyNoon
        self.assertEqual(weather.precipitation, preset.precipitation)
        self.assertEqual(weather.wind_intensity, 0.0)

        with self.assertRaises(ValueError):
            make_weather("not_a_condition")

    def test_visualization_tool_uses_same_profiles(self):
        import scripts.visualize_doe_factors as viz

        self.assertEqual(viz.FOG_PROFILES, [(k.lower(), v) for k, v in wp.FOG_PROFILES.items()])
        self.assertEqual(viz.RAIN_PROFILES, [(k.lower(), v) for k, v in wp.RAIN_PROFILES.items()])
        self.assertEqual(viz.CLOUDINESS_FIXED, wp.DOE_FIXED_FIELDS["cloudiness"])
        self.assertEqual(viz.WIND_INTENSITY_FIXED, wp.DOE_FIXED_FIELDS["wind_intensity"])
        self.assertEqual(viz.SUN_AZIMUTH_FIXED, wp.DOE_FIXED_FIELDS["sun_azimuth_angle"])

        # A visualization capture of a DOE triple == the collected weather.
        _, fields = viz.make_weather({
            "sun_altitude_angle": 0.0, **wp.RAIN_PROFILES["Heavy"], **wp.FOG_PROFILES["Moderate"],
        })
        self.assertEqual(viz.weather_fields_to_dict(fields), wp.doe_weather_fields(0, "Heavy", "Moderate"))

    def test_profile_hash_is_stable(self):
        self.assertEqual(wp.weather_profile_hash(), wp.weather_profile_hash())
        self.assertEqual(len(wp.weather_profile_hash()), 64)


class CommandTests(unittest.TestCase):

    def setUp(self):
        conditions, _ = runner.load_design(DESIGN_CSV)
        self.root = Path(tempfile.mkdtemp())
        self.job = runner.expand_jobs(conditions, [303], "Town01", ["2"], self.root)[1]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_command(self):
        command = runner.build_collector_command(self.job, self.root, "localhost", 2000)

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]), PROJECT_ROOT / "scripts" / "collect_dataset.py")

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value("--towns"), "Town01")
        self.assertEqual(value("--routes"), "2")
        self.assertEqual(value("--conditions"), "doe_C002_sun+60_rain-Dry_fog-Light")
        self.assertEqual(value("--seed"), "303")
        self.assertEqual(Path(value("--output-root")), self.root / "seed_303")
        self.assertEqual(value("--port"), "2000")
        self.assertNotIn("--rerender-conditions", command)
        self.assertNotIn("--truncate-ok", command)

    def test_rerender_flag(self):
        command = runner.build_collector_command(self.job, self.root, "localhost", 2000, rerender=True)
        self.assertEqual(command[command.index("--rerender-conditions") + 1],
                         "doe_C002_sun+60_rain-Dry_fog-Light")

    def test_collector_accepts_generated_flags(self):
        import scripts.collect_dataset as cd

        command = runner.build_collector_command(self.job, self.root, "localhost", 2000, rerender=True)
        old = sys.argv
        sys.argv = ["collect_dataset.py"] + command[2:]
        try:
            args = cd.parse_args()
        finally:
            sys.argv = old

        self.assertEqual(args.seed, 303)
        self.assertEqual(args.conditions, ["doe_C002_sun+60_rain-Dry_fog-Light"])
        self.assertEqual(args.rerender_conditions, ["doe_C002_sun+60_rain-Dry_fog-Light"])

    def test_collector_seed_override_sets_every_seed(self):
        import scripts.collect_dataset as cd

        saved = (cfg.RANDOM.SEED, cfg.TRAFFIC.SEED, cfg.PEDESTRIAN.SEED, cfg.SPAWN.SEED)
        try:
            cd.apply_seed_override(202)
            self.assertEqual((cfg.RANDOM.SEED, cfg.TRAFFIC.SEED, cfg.PEDESTRIAN.SEED, cfg.SPAWN.SEED),
                             (202, 202, 202, 202))
        finally:
            cfg.RANDOM.SEED, cfg.TRAFFIC.SEED, cfg.PEDESTRIAN.SEED, cfg.SPAWN.SEED = saved

    def test_collector_defaults_unchanged_without_seed(self):
        import scripts.collect_dataset as cd

        old = sys.argv
        sys.argv = ["collect_dataset.py"]
        try:
            args = cd.parse_args()
        finally:
            sys.argv = old

        self.assertIsNone(args.seed)
        self.assertEqual((args.host, args.port), (cfg.CARLA.HOST, cfg.CARLA.PORT))


def build_fake_job_output(job, output_root, num_frames=3, seed=None, truncated=False, weather_override=None,
                          rgb_counts=None):
    """Mimic what collect_dataset.py leaves on disk for one DOE job."""

    route_path = runner.job_route_path(job, output_root)
    geometry = Path(geometry_dir(str(route_path)))
    source = Path(condition_dir(str(route_path), cfg.WEATHER.DEFAULT))
    out_dir = runner.job_output_dir(job, output_root)
    seed = job["seed"] if seed is None else seed

    rgb_counts = rgb_counts or {}

    for directory in (geometry / "labels" / "object_3d", geometry / "world_state", geometry / "pose",
                      out_dir / "rgb_left", out_dir / "rgb_right"):
        directory.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        (geometry / "labels" / "object_3d" / f"{i:06d}.json").write_text("{}", encoding="utf-8")
        (geometry / "world_state" / f"{i:06d}.json").write_text("{}", encoding="utf-8")

    for camera in ("rgb_left", "rgb_right"):
        for i in range(rgb_counts.get(camera, num_frames)):
            (out_dir / camera / f"{i:06d}.png").write_bytes(b"")

    (geometry / "calibration.json").write_text("{}", encoding="utf-8")
    (geometry / "pose" / "poses.csv").write_text("frame\n", encoding="utf-8")
    (geometry / "sequence.json").write_text(json.dumps({
        "map": "Town01", "fixed_delta_seconds": 0.05, "simulation_hz": 20.0, "recording_hz": 10.0,
        "random_seed": seed, "traffic_seed": seed, "pedestrian_seed": seed,
        "spawn_policy": {"spawn_seed": seed}, "truncated": truncated, "num_frames": num_frames,
    }), encoding="utf-8")

    weather = wp.doe_weather_fields(job["Sun"], job["Rain"], job["Fog"])
    weather.update(weather_override or {})

    (out_dir / "condition.json").write_text(json.dumps({
        "condition": runner.job_condition_name(job),
        "weather_parameters": weather,
        "num_frames": num_frames,
        "replay_validation": {"passed": True},
        "calibration_check": {"equal": True},
    }), encoding="utf-8")

    (route_path / "paired_validation.json").write_text(json.dumps({"passed": True, "errors": []}), encoding="utf-8")

    mark_complete(str(geometry), {"num_frames": num_frames})
    mark_complete(str(source), {"num_frames": num_frames})
    mark_complete(str(out_dir), {"num_frames": num_frames, "rendered_from": "replay"})


class VerificationAndResumeTests(unittest.TestCase):

    def setUp(self):
        conditions, _ = runner.load_design(DESIGN_CSV)
        self.root = Path(tempfile.mkdtemp())
        self.jobs = runner.expand_jobs(conditions, [101, 202, 303], "Town01", ["0"], self.root)
        self.job = self.jobs[0]

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_output_passes(self):
        build_fake_job_output(self.job, self.root)
        ok, info, errors = runner.verify_job_outputs(self.job, self.root, since_unix=0)

        self.assertTrue(ok, errors)
        self.assertEqual(info["num_frames"], 3)
        self.assertEqual(info["seeds_recorded"]["spawn_seed"], 101)
        self.assertEqual(info["weather"]["sun_altitude_angle"], 60.0)

    def test_missing_images_fail(self):
        build_fake_job_output(self.job, self.root)
        os.remove(runner.job_output_dir(self.job, self.root) / "rgb_right" / "000001.png")
        ok, _, errors = runner.verify_job_outputs(self.job, self.root)
        self.assertFalse(ok)
        self.assertTrue(any("rgb_right" in e for e in errors))

    def test_wrong_weather_fails(self):
        build_fake_job_output(self.job, self.root, weather_override={"fog_density": 50.0})
        ok, _, errors = runner.verify_job_outputs(self.job, self.root)
        self.assertFalse(ok)
        self.assertTrue(any("fog_density" in e for e in errors))

    def test_seed_mismatch_fails(self):
        build_fake_job_output(self.job, self.root, seed=42)
        ok, _, errors = runner.verify_job_outputs(self.job, self.root)
        self.assertFalse(ok)
        self.assertTrue(any("seed" in e for e in errors))

    def test_truncated_geometry_fails_unless_allowed(self):
        build_fake_job_output(self.job, self.root, truncated=True)
        self.assertFalse(runner.verify_job_outputs(self.job, self.root)[0])
        self.assertTrue(runner.verify_job_outputs(self.job, self.root, allow_truncated=True)[0])

    def test_exit_code_zero_without_outputs_is_not_complete(self):
        ok, _, errors = runner.verify_job_outputs(self.job, self.root)
        self.assertFalse(ok)
        self.assertIn("geometry not COMPLETE", errors[0])

    def mark_run_complete(self, job):
        build_fake_job_output(job, self.root)
        _, info, _ = runner.verify_job_outputs(job, self.root)
        marker = {"status": "completed", "job_id": job["job_id"], "condition_id": job["condition_id"],
                  "seed": job["seed"], "town": job["town"], "route_id": job["route_id"],
                  "weather_profile_hash": wp.weather_profile_hash(), "num_frames": info["num_frames"]}
        runner.write_json_atomic(runner.job_output_dir(job, self.root) / runner.RUN_COMPLETE_NAME, marker)

    def test_resume_trusts_markers_not_manifest(self):
        done, stale, interrupted, lost = self.jobs[0], self.jobs[1], self.jobs[2], self.jobs[3]

        self.mark_run_complete(done)
        done["status"] = "completed"

        stale["status"] = "completed"       # manifest says done, disk says no
        interrupted["status"] = "running"   # runner was killed mid-job

        self.mark_run_complete(lost)        # done on disk, manifest lost it

        runner.reconcile_with_outputs(self.jobs, self.root)

        self.assertEqual(done["status"], "completed")
        self.assertEqual(stale["status"], "pending")
        self.assertEqual(interrupted["status"], "pending")
        self.assertEqual(lost["status"], "completed")

        selected_ids = {j["job_id"] for j in runner.select_jobs(self.jobs)}
        self.assertNotIn(done["job_id"], selected_ids)
        self.assertNotIn(lost["job_id"], selected_ids)
        self.assertIn(stale["job_id"], selected_ids)

    def test_marker_invalid_after_profile_change_or_missing_collector_marker(self):
        self.mark_run_complete(self.job)
        self.assertTrue(runner.has_valid_completion_marker(self.job, self.root)[0])

        os.remove(runner.job_output_dir(self.job, self.root) / "COMPLETE")
        self.assertFalse(runner.has_valid_completion_marker(self.job, self.root)[0])

    def test_failed_selection(self):
        self.jobs[5]["status"] = "failed"
        self.jobs[6]["status"] = "completed"
        self.jobs[7]["status"] = "skipped"

        failed_only = runner.select_jobs(self.jobs, rerun_failed=True)
        self.assertEqual([j["job_id"] for j in failed_only], [self.jobs[5]["job_id"]])

        default = runner.select_jobs(self.jobs)
        ids = [j["job_id"] for j in default]
        self.assertNotIn(self.jobs[5]["job_id"], ids)
        self.assertNotIn(self.jobs[6]["job_id"], ids)
        self.assertIn(self.jobs[7]["job_id"], ids)
        self.assertEqual(len(default), 156 - 2)

    def test_filters_and_max_jobs(self):
        subset = runner.apply_filters(self.jobs, ["C001"], [101], ["0"])
        self.assertEqual([(j["condition_id"], j["seed"], j["route_id"]) for j in subset], [("C001", 101, "0")])

        self.assertEqual(len(runner.select_jobs(self.jobs, max_jobs=4)), 4)
        self.assertEqual(runner.select_jobs(self.jobs, max_jobs=4)[0]["job_id"], "J000001")

        for kwargs in ({"only_conditions": ["C099"]}, {"only_seeds": [7]}, {"only_routes": ["9"]}):
            with self.assertRaises(ValueError):
                runner.apply_filters(self.jobs, **kwargs)


class MainLoopTests(unittest.TestCase):
    """runner.main() end to end with the CARLA preflight and the collector subprocess faked."""

    def setUp(self):
        from unittest import mock

        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "data"
        self.exp = self.tmp / "exp"
        conditions, _ = runner.load_design(DESIGN_CSV)
        self.jobs = {
            (j["condition_id"], j["seed"], j["route_id"]): j
            for j in runner.expand_jobs(conditions, [101, 202, 303], "Town01", ["0", "1"], self.data)
        }
        self.collector_calls = []
        self.collector_succeeds = True
        self.fake_frames = 3
        self.fake_truncated = False

        def fake_run(command, log_path, timeout_s=None, echo=False):
            self.collector_calls.append(command)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Reason    : fake canonical failure\n", encoding="utf-8")

            if self.collector_succeeds:
                condition = runner.job_condition_name
                name = command[command.index("--conditions") + 1]
                seed = int(command[command.index("--seed") + 1])
                route = command[command.index("--routes") + 1]
                job = next(j for j in self.jobs.values()
                           if condition(j) == name and j["seed"] == seed and j["route_id"] == route)
                build_fake_job_output(job, self.data, num_frames=self.fake_frames, truncated=self.fake_truncated)

            return 0  # the collector exits 0 even when a route fails

        self.patches = [
            mock.patch.object(runner, "preflight_carla",
                              return_value={"server_version": "0.9.16", "client_version": "0.9.16"}),
            mock.patch.object(runner, "carla_port_open", return_value=True),
            mock.patch.object(runner, "run_subprocess", side_effect=fake_run),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, *extra):
        return runner.main([
            "--design-csv", str(DESIGN_CSV), "--seeds", "101", "202", "303", "--town", "Town01",
            "--routes", "0", "1", "--output-root", str(self.data), "--experiment-dir", str(self.exp), *extra,
        ])

    def manifest(self):
        return {row["job_id"]: row for row in runner.read_manifest(self.exp / runner.MANIFEST_NAME)}

    def test_one_job_then_resume(self):
        self.assertEqual(self.run_main("--only-condition", "C001", "--only-seed", "101", "--max-jobs", "1"), 0)

        rows = self.manifest()
        self.assertEqual(len(rows), 52 * 3 * 2)
        self.assertEqual(rows["J000001"]["status"], "completed")
        self.assertEqual(rows["J000001"]["attempts"], 1)
        self.assertEqual(sum(r["status"] == "pending" for r in rows.values()), 311)

        marker = json.loads((Path(rows["J000001"]["output_path"]) / runner.RUN_COMPLETE_NAME).read_text(encoding="utf-8"))
        self.assertEqual((marker["condition_id"], marker["seed"], marker["route_id"]), ("C001", 101, "0"))
        self.assertEqual(marker["factors"], {"Sun": 60, "Rain": "Dry", "Fog": "Clear"})
        self.assertEqual(marker["weather"]["sun_altitude_angle"], 60.0)
        self.assertEqual(marker["carla"]["server_version"], "0.9.16")
        self.assertIn("commit", marker["git"])
        self.assertTrue((self.exp / runner.EXPERIMENT_INFO_NAME).is_file())

        # Without --resume an experiment in progress is not touched.
        self.assertEqual(self.run_main("--only-condition", "C001"), 2)

        # --resume skips the completed job (no collector call for it).
        self.collector_calls.clear()
        self.assertEqual(self.run_main("--resume", "--only-condition", "C001", "--only-seed", "101"), 0)
        self.assertEqual([c[c.index("--routes") + 1] for c in self.collector_calls], ["1"])
        self.assertEqual(self.manifest()["J000002"]["status"], "completed")

    def test_failure_skips_rest_of_group_and_rerun_failed_retries(self):
        self.collector_succeeds = False

        self.assertEqual(self.run_main("--only-seed", "101", "--only-route", "0", "--max-jobs", "3"), 1)
        rows = self.manifest()
        c001, c002, c003 = (self.jobs[(c, 101, "0")]["job_id"] for c in ("C001", "C002", "C003"))

        self.assertEqual(rows[c001]["status"], "failed")
        self.assertIn("fake canonical failure", rows[c001]["error_summary"])
        self.assertEqual((rows[c002]["status"], rows[c003]["status"]), ("skipped", "skipped"))
        self.assertEqual(len(self.collector_calls), 1)

        self.collector_succeeds = True
        self.collector_calls.clear()
        self.assertEqual(self.run_main("--resume", "--rerun-failed"), 0)

        rows = self.manifest()
        self.assertEqual(rows[c001]["status"], "completed")
        self.assertEqual(rows[c001]["attempts"], 2)
        self.assertEqual(rows[c002]["status"], "skipped")  # --rerun-failed runs failed jobs only
        self.assertEqual(len(self.collector_calls), 1)


class CliTests(unittest.TestCase):

    def test_smoke_options_need_separate_roots(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--routes", "0", "--truncate-ok"])

    def test_dry_run_writes_nothing(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            code = runner.main([
                "--design-csv", str(DESIGN_CSV), "--routes", "0", "1", "2", "--dry-run",
                "--output-root", str(tmp / "data"), "--experiment-dir", str(tmp / "exp"),
            ])
            self.assertEqual(code, 0)
            self.assertEqual(list(tmp.iterdir()), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_design_csv_stops(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            code = runner.main(["--design-csv", str(tmp / "nope.csv"), "--routes", "0", "--dry-run"])
            self.assertEqual(code, 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class FixedLengthFrameTests(unittest.TestCase):
    """--target-frames: production fixed-length geometry (not a smoke test)."""

    TARGET = 1100

    def setUp(self):
        self.conditions, _ = runner.load_design(DESIGN_CSV)
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def job(self, target_frames):
        return runner.expand_jobs(self.conditions, [101], "Town10HD_Opt", ["0"], self.root,
                                  target_frames=target_frames)[0]

    # ---- CLI -----------------------------------------------------------
    def test_parse_target_frames(self):
        args = runner.parse_args(["--routes", "0", "--target-frames", "1100"])
        self.assertEqual(args.target_frames, 1100)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.truncate_ok)
        self.assertIsNone(runner.parse_args(["--routes", "0"]).target_frames)

    def test_target_frames_below_one_rejected(self):
        for value in ("0", "-5"):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--routes", "0", "--target-frames", value])

    def test_target_frames_conflicts_with_smoke_options(self):
        roots = ["--output-root", str(self.root / "d"), "--experiment-dir", str(self.root / "e")]
        for smoke in (["--max-frames", "50"], ["--truncate-ok"]):
            with self.assertRaises(SystemExit):
                runner.parse_args(["--routes", "0", "--target-frames", "1100", *smoke, *roots])

        with self.assertRaises(ValueError):
            runner.build_collector_command(self.job(1100), self.root, "localhost", 2000,
                                           target_frames=1100, smoke_max_frames=50)

    # ---- collector command ---------------------------------------------
    def test_target_mode_command(self):
        command = runner.build_collector_command(self.job(1100), self.root, "localhost", 2000, target_frames=1100)
        self.assertEqual(command[command.index("--max-frames") + 1], "1100")
        self.assertIn("--truncate-ok", command)
        self.assertEqual(command.count("--max-frames"), 1)

        import scripts.collect_dataset as cd
        old = sys.argv
        sys.argv = ["collect_dataset.py"] + command[2:]
        try:
            args = cd.parse_args()
        finally:
            sys.argv = old
        self.assertEqual((args.max_frames, args.truncate_ok), (1100, True))

    def test_route_completion_command_has_no_cap(self):
        command = runner.build_collector_command(self.job(None), self.root, "localhost", 2000)
        self.assertNotIn("--max-frames", command)
        self.assertNotIn("--truncate-ok", command)

    # ---- verifier ------------------------------------------------------
    def test_target_mode_accepts_truncated_geometry(self):
        job = self.job(self.TARGET)
        build_fake_job_output(job, self.root, num_frames=self.TARGET, truncated=True)
        ok, info, errors = runner.verify_job_outputs(job, self.root, since_unix=0)

        self.assertTrue(ok, errors)
        self.assertEqual(info["num_frames"], self.TARGET)
        self.assertTrue(info["geometry_truncated"])
        self.assertEqual(info["frame_policy"], {"mode": "fixed_length", "target_frames": self.TARGET})

    def test_target_mode_rejects_wrong_frame_count(self):
        job = self.job(self.TARGET)
        for frames in (self.TARGET - 1, self.TARGET + 1):
            shutil.rmtree(self.root, ignore_errors=True)
            build_fake_job_output(job, self.root, num_frames=frames, truncated=True)
            ok, _, errors = runner.verify_job_outputs(job, self.root)
            self.assertFalse(ok, frames)
            self.assertTrue(any("target_frames=1100" in e for e in errors), errors)

    def test_target_mode_rejects_stereo_count_mismatch(self):
        job = self.job(4)
        for camera in ("rgb_left", "rgb_right"):
            shutil.rmtree(self.root, ignore_errors=True)
            build_fake_job_output(job, self.root, num_frames=4, truncated=True, rgb_counts={camera: 3})
            ok, _, errors = runner.verify_job_outputs(job, self.root)
            self.assertFalse(ok)
            self.assertTrue(any(e.startswith(camera) for e in errors), errors)

    def test_target_mode_rejects_sequence_or_world_state_mismatch(self):
        job = self.job(4)
        build_fake_job_output(job, self.root, num_frames=4, truncated=True)
        geometry = Path(geometry_dir(str(runner.job_route_path(job, self.root))))

        os.remove(geometry / "world_state" / "000003.json")
        ok, _, errors = runner.verify_job_outputs(job, self.root)
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("world_state") for e in errors), errors)

        (geometry / "world_state" / "000003.json").write_text("{}", encoding="utf-8")
        sequence = json.loads((geometry / "sequence.json").read_text(encoding="utf-8"))
        sequence["num_frames"] = 5
        (geometry / "sequence.json").write_text(json.dumps(sequence), encoding="utf-8")
        ok, _, errors = runner.verify_job_outputs(job, self.root)
        self.assertFalse(ok)
        self.assertTrue(any("sequence.json num_frames" in e for e in errors), errors)

    def test_route_completion_still_rejects_truncation(self):
        job = self.job(None)
        build_fake_job_output(job, self.root, num_frames=4, truncated=True)
        self.assertFalse(runner.verify_job_outputs(job, self.root)[0])
        # smoke-test --truncate-ok behaviour unchanged
        self.assertTrue(runner.verify_job_outputs(job, self.root, allow_truncated=True)[0])

    # ---- manifest / resume identity -----------------------------------
    def test_manifest_with_different_target_cannot_merge(self):
        path = self.root / "exp" / runner.MANIFEST_NAME
        jobs = runner.expand_jobs(self.conditions, [101], "Town01", ["0"], self.root / "d", target_frames=1100)
        runner.write_manifest(path, jobs)
        existing = runner.read_manifest(path)

        self.assertEqual(existing[0]["target_frames"], 1100)
        self.assertEqual(len(runner.merge_manifest(jobs, existing, path)), 52)

        for other_target in (1000, None):
            other = runner.expand_jobs(self.conditions, [101], "Town01", ["0"], self.root / "d",
                                       target_frames=other_target)
            with self.assertRaises(runner.ManifestMismatchError) as ctx:
                runner.merge_manifest(other, existing, path)
            self.assertIn("frame policy", str(ctx.exception))

    def test_legacy_manifest_without_column_is_route_completion(self):
        path = self.root / "exp" / runner.MANIFEST_NAME
        jobs = runner.expand_jobs(self.conditions, [101], "Town01", ["0"], self.root / "d")
        runner.write_manifest(path, jobs)

        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        fields = [c for c in runner.MANIFEST_FIELDS if c != "target_frames"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        existing = runner.read_manifest(path)
        self.assertIsNone(existing[0]["target_frames"])
        self.assertEqual(len(runner.merge_manifest(jobs, existing, path)), 52)

        fixed = runner.expand_jobs(self.conditions, [101], "Town01", ["0"], self.root / "d", target_frames=1100)
        with self.assertRaises(runner.ManifestMismatchError):
            runner.merge_manifest(fixed, existing, path)

    def test_completion_marker_checks_frame_policy(self):
        job = self.job(4)
        build_fake_job_output(job, self.root, num_frames=4, truncated=True)
        marker = {"status": "completed", "job_id": job["job_id"], "condition_id": job["condition_id"],
                  "seed": job["seed"], "town": job["town"], "route_id": job["route_id"],
                  "weather_profile_hash": wp.weather_profile_hash(), "num_frames": 4,
                  "frame_policy": runner.frame_policy(4)}
        marker_path = runner.job_output_dir(job, self.root) / runner.RUN_COMPLETE_NAME
        runner.write_json_atomic(marker_path, marker)
        self.assertTrue(runner.has_valid_completion_marker(job, self.root)[0])

        # A legacy marker (no frame_policy) is a route-completion run.
        del marker["frame_policy"]
        runner.write_json_atomic(marker_path, marker)
        self.assertFalse(runner.has_valid_completion_marker(job, self.root)[0])
        self.assertTrue(runner.has_valid_completion_marker(dict(job, target_frames=None), self.root)[0])


class FixedLengthMainTests(MainLoopTests):
    """main() with --target-frames; reuses MainLoopTests' faked CARLA/collector."""

    def setUp(self):
        super().setUp()
        self.fake_frames = 4
        self.fake_truncated = True

    def test_one_job_then_resume(self):  # route-completion case, covered by MainLoopTests
        pass

    def test_failure_skips_rest_of_group_and_rerun_failed_retries(self):  # covered by MainLoopTests
        pass

    def test_metadata_and_resume_identity(self):
        one = ("--only-condition", "C001", "--only-seed", "101", "--only-route", "0", "--max-jobs", "1")
        self.assertEqual(self.run_main("--target-frames", "4", *one), 0)

        command = self.collector_calls[0]
        self.assertEqual(command[command.index("--max-frames") + 1], "4")
        self.assertIn("--truncate-ok", command)

        policy = {"mode": "fixed_length", "target_frames": 4}
        info = json.loads((self.exp / runner.EXPERIMENT_INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["frame_policy"], policy)

        row = self.manifest()["J000001"]
        self.assertEqual((row["status"], row["target_frames"]), ("completed", 4))
        marker = json.loads((Path(row["output_path"]) / runner.RUN_COMPLETE_NAME).read_text(encoding="utf-8"))
        self.assertEqual(marker["frame_policy"], policy)
        self.assertTrue(marker["geometry_truncated"])
        self.assertTrue(marker["geometry_truncation_expected"])
        self.assertEqual(marker["num_frames"], 4)

        # A different frame policy may not resume this experiment.
        self.assertEqual(self.run_main("--resume", "--target-frames", "5"), 2)
        self.assertEqual(self.run_main("--resume"), 2)

        # Same policy resumes and skips the completed job.
        self.collector_calls.clear()
        self.assertEqual(self.run_main("--resume", "--target-frames", "4", "--only-condition", "C001",
                                       "--only-seed", "101"), 0)
        self.assertEqual([c[c.index("--routes") + 1] for c in self.collector_calls], ["1"])

    def test_route_completion_metadata(self):
        self.fake_frames, self.fake_truncated = 3, False
        self.assertEqual(self.run_main("--only-condition", "C001", "--only-seed", "101", "--max-jobs", "1"), 0)
        self.assertNotIn("--max-frames", self.collector_calls[0])

        info = json.loads((self.exp / runner.EXPERIMENT_INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(info["frame_policy"], {"mode": "route_completion", "target_frames": None})
        row = self.manifest()["J000001"]
        marker = json.loads((Path(row["output_path"]) / runner.RUN_COMPLETE_NAME).read_text(encoding="utf-8"))
        self.assertEqual(marker["frame_policy"], {"mode": "route_completion", "target_frames": None})
        self.assertFalse(marker["geometry_truncation_expected"])

    def test_dry_run_shows_frame_policy_and_command(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.run_main("--target-frames", "1100", "--dry-run"), 0)
        text = out.getvalue()

        self.assertIn("Frame policy     : fixed_length", text)
        self.assertIn("Target frames    : 1100", text)
        self.assertIn("--max-frames 1100 --truncate-ok", text)
        self.assertFalse(self.exp.exists())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertIn("Frame policy     : route_completion", out.getvalue())
        self.assertIn("Target frames    : none", out.getvalue())
        self.assertNotIn("--max-frames", out.getvalue())


class SmokeOptionTests(unittest.TestCase):
    """The pre-existing smoke-test passthrough keeps its meaning."""

    def test_smoke_options_with_separate_roots(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            args = runner.parse_args(["--routes", "0", "--max-frames", "50", "--truncate-ok",
                                      "--output-root", str(tmp / "d"), "--experiment-dir", str(tmp / "e")])
            self.assertEqual((args.max_frames, args.truncate_ok, args.target_frames), (50, True, None))

            conditions, _ = runner.load_design(DESIGN_CSV)
            job = runner.expand_jobs(conditions, [101], "Town01", ["0"], tmp)[0]
            command = runner.build_collector_command(job, tmp, "localhost", 2000,
                                                     smoke_max_frames=50, smoke_truncate_ok=True)
            self.assertEqual(command[command.index("--max-frames") + 1], "50")
            self.assertIn("--truncate-ok", command)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

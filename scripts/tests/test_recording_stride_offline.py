"""
scripts/tests/test_recording_stride_offline.py

Offline (no CARLA server) regression tests for the 20 Hz simulation / 10 Hz
dataset-recording task (see CLAUDE.md):

    world / controller / Traffic Manager / Gamma   -> stay at 20 Hz
    RGB / depth / semantic / flow / LiDAR / radar /
    annotation / world_state / pose                -> saved at 10 Hz only

Covers CLAUDE.md task section 18 items 1-3, 5-13, 15-16 with no live CARLA
world needed:

    1.  20 Hz simulation / 10 Hz recording -> stride 2
    2.  simulation tick 0 -> record
    3.  tick 1 -> skip / tick 2 -> record
    5.  saved frame IDs contiguous
    6.  CARLA frame IDs delta = 2
    7.  saved timestamps delta = 0.1
    8.  traffic update still called every simulation tick
    9.  recording writer only every 2 ticks
    10. Gamma timing unchanged (simulation-tick based, not record-based)
    11. annotation only saved on record ticks
    12. world_state only saved on record ticks
    13. pose (metadata) only saved on record ticks
    15. metadata recording_hz=10
    16. resume rejects 20Hz/10Hz mismatch

Item 4 (replay output count == canonical sample count) and the live-CARLA
runtime numbers (14, 19-24 elsewhere in CLAUDE.md) need the real smoke test:

    python scripts/collect_dataset.py --towns Town01 --routes 0 \
        --conditions day_clear day_rain --max-frames 100 --truncate-ok --overwrite

Run:
    python -m unittest scripts.tests.test_recording_stride_offline -v
"""

import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

from CFG.config import cfg  # noqa: E402

import scripts.collect_dataset as collect_dataset  # noqa: E402

from src.data.layout import condition_dir, geometry_dir, mark_complete  # noqa: E402
from src.data.metadata import MetadataWriter  # noqa: E402
from src.data.resume_plan import plan_route_work  # noqa: E402
from src.simulation.timing import is_record_tick, record_stride_ticks  # noqa: E402


# ================================================================
# 1-3: stride / is_record_tick
# ================================================================

class RecordStrideConfigTest(unittest.TestCase):

    def test_stride_is_two_for_20hz_sim_10hz_recording(self):
        self.assertEqual(cfg.SIMULATION.FPS, 20)
        self.assertEqual(cfg.RECORDING.FPS, 10)
        self.assertEqual(record_stride_ticks(cfg), 2)

    def test_simulation_clock_is_untouched(self):
        # CLAUDE.md hard rule: fixed_delta_seconds must never become 0.1.
        self.assertAlmostEqual(cfg.SIMULATION.FIXED_DELTA_SECONDS, 0.05)

    def test_record_tick_pattern(self):
        stride = record_stride_ticks(cfg)
        pattern = [is_record_tick(i, stride) for i in range(6)]
        self.assertEqual(pattern, [True, False, True, False, True, False])

    def test_non_integer_stride_raises(self):
        from types import SimpleNamespace

        bad_cfg = SimpleNamespace(
            SIMULATION=SimpleNamespace(FIXED_DELTA_SECONDS=0.05),
            RECORDING=SimpleNamespace(SAMPLE_INTERVAL_SECONDS=0.07),
        )

        with self.assertRaises(ValueError):
            record_stride_ticks(bad_cfg)


# ================================================================
# 5-7: contiguous dataset ids / CARLA-frame delta / timestamp delta
# ================================================================

class SimulatedRecordLoopTest(unittest.TestCase):
    """
    Exercises the same is_record_tick() selection collect_dataset.py's main
    loop uses, against a fake tick source (arbitrary starting CARLA frame
    number + fixed_delta-spaced timestamps) -- no CARLA world required.
    """

    def test_saved_ids_contiguous_frame_delta_2_timestamp_delta_0_1(self):
        stride = record_stride_ticks(cfg)
        fixed_delta = cfg.SIMULATION.FIXED_DELTA_SECONDS

        saved = []
        carla_frame = 987654  # arbitrary CARLA server frame at loop start
        timestamp = 0.0

        for simulation_tick_idx in range(40):  # 40 ticks = 20 samples @ stride 2
            carla_frame += 1
            timestamp += fixed_delta

            if is_record_tick(simulation_tick_idx, stride):
                saved.append((len(saved), carla_frame, timestamp))

        # Item 5: contiguous dataset frame_id.
        self.assertEqual([s[0] for s in saved], list(range(len(saved))))

        # Item 6: CARLA frame delta == stride (2) between consecutive saves.
        carla_frames = [s[1] for s in saved]
        deltas = [b - a for a, b in zip(carla_frames, carla_frames[1:])]
        self.assertTrue(all(d == stride for d in deltas), deltas)

        # Item 7: saved timestamp delta == cfg.RECORDING.SAMPLE_INTERVAL_SECONDS.
        timestamps = [s[2] for s in saved]
        ts_deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        for d in ts_deltas:
            self.assertAlmostEqual(d, cfg.RECORDING.SAMPLE_INTERVAL_SECONDS, places=9)


# ================================================================
# 8-13: static inspection of generate_canonical_geometry()'s main loop
# ================================================================

class CanonicalLoopStructureTest(unittest.TestCase):
    """
    Static inspection (same style as PreRecordWarmupPlacementTest in
    test_traffic_tuning_offline.py): confirms which calls are gated on
    `if record_tick:` and which run unconditionally every simulation tick.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = inspect.getsource(collect_dataset.generate_canonical_geometry)

    def _index(self, needle):
        index = self.source.index(needle)
        self.assertGreaterEqual(index, 0)
        return index

    def test_loop_variable_is_simulation_tick_idx(self):
        self.assertIn("for simulation_tick_idx in range(\n            max_simulation_ticks\n        ):", self.source)

    def test_record_stride_is_not_a_hard_coded_magic_number(self):
        self.assertIn("record_stride_ticks(cfg)", self.source)
        self.assertIn("is_record_tick(\n                simulation_tick_idx, record_stride,\n            )", self.source)
        self.assertNotIn("% 2 ==", self.source)

    def test_control_and_tick_run_unconditionally_every_simulation_tick(self):
        # route_controller.run_step()/apply_control()/world.tick() must all
        # precede the `if record_tick:` gate -- every simulation tick.
        record_gate_index = self._index("if record_tick:")
        prefix = self.source[:record_gate_index]

        self.assertIn("route_controller\n                .run_step()", prefix)
        self.assertIn("ego.apply_control(", prefix)
        self.assertIn("world.tick()", prefix)

    def test_annotation_world_state_metadata_gated_on_record_tick(self):
        # Items 11 (annotation), 12 (world_state), 13 (pose/metadata):
        # all three calls must be textually inside the `if record_tick:`
        # block, i.e. strictly between it and the next sibling section
        # (the Phase 2 / Gamma comment, at the same 12-space indentation).
        record_gate_index = self._index("if record_tick:")
        phase2_index = self._index("# Phase 2: dynamic density maintenance")
        self.assertLess(record_gate_index, phase2_index)

        record_block = self.source[record_gate_index:phase2_index]

        self.assertIn("metadata.write_frame(", record_block)
        self.assertIn("world_state_recorder.record_frame(", record_block)
        self.assertIn(".write_frame(\n                        record_frame_id,\n                        world,\n                        ego,", record_block)
        self.assertIn("collector.save_frame(\n                    record_frame_id,\n                    packet,\n                )", record_block)

    def test_gamma_and_spawn_manager_update_use_simulation_tick_idx_not_record_frame_id(self):
        # Item 10: Gamma/spawn timing keys off simulation_tick_idx (20 Hz),
        # never record_frame_id / saved_frames (10 Hz).
        self.assertIn("simulation_tick_idx % cfg.SPAWN.UPDATE_INTERVAL_FRAMES == 0", self.source)
        self.assertIn("spawn_manager.update(\n                        simulation_tick_idx,", self.source)
        self.assertIn("target_for_frame(simulation_tick_idx)", self.source)
        self.assertNotIn("spawn_manager.update(\n                        record_frame_id", self.source)

    def test_gamma_update_interval_stride_invariant_is_asserted(self):
        # Item 8/10 safety net: the interdependency between
        # UPDATE_INTERVAL_FRAMES (20 Hz cadence) and the record stride is
        # checked explicitly, not assumed silently.
        self.assertIn("cfg.SPAWN.UPDATE_INTERVAL_FRAMES % record_stride == 0", self.source)

    def test_phase2_block_is_not_nested_inside_record_tick_block(self):
        # The Phase 2 / Gamma section must run at the same indentation as
        # `if record_tick:` (a sibling in the loop body), not nested inside
        # it -- otherwise Gamma maintenance would silently stop running on
        # non-record ticks.
        record_gate_line = next(
            line for line in self.source.splitlines() if line.strip() == "if record_tick:"
        )
        phase2_comment_line = next(
            line for line in self.source.splitlines()
            if line.strip() == "# Phase 2: dynamic density maintenance"
        )

        def indent(line):
            return len(line) - len(line.lstrip(" "))

        self.assertEqual(indent(record_gate_line), indent(phase2_comment_line))


# ================================================================
# Static inspection of replay_condition(): stride-aware replay ticking
# ================================================================

class ReplayLoopStructureTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = inspect.getsource(collect_dataset.replay_condition)

    def test_replay_uses_record_stride_ticks_per_sample(self):
        self.assertIn("record_stride_ticks(cfg)", self.source)
        self.assertIn("ticks_this_frame = 1 if frame_id == 0 else record_stride", self.source)

    def test_replay_sample_count_still_driven_by_canonical_reader_length(self):
        # Item 4 (replay output count == canonical sample count): num_frames
        # comes from len(reader) (canonical's own saved-sample count,
        # unaffected by this task) and the loop iterates exactly that many
        # times -- actual equality is verified live in the CARLA smoke test.
        self.assertIn("for frame_id in range(\n            num_frames\n        ):", self.source)

    def test_intermediate_ticks_do_not_collect_or_save(self):
        loop_index = self.source.index("for frame_id in range(\n            num_frames\n        ):")
        tail = self.source[loop_index:]
        ticks_index = tail.index("for _ in range(ticks_this_frame):")
        collect_index = tail.index("collector.collect_frame(")
        self.assertLess(ticks_index, collect_index)


# ================================================================
# 15: metadata recording_hz
# ================================================================

class MetadataRecordingHzTest(unittest.TestCase):

    def test_sequence_json_declares_recording_hz(self):
        tmp_dir = tempfile.mkdtemp()

        try:
            writer = MetadataWriter(
                sequence_root=tmp_dir,
                map_name="Town01",
                sequence_id="Town01_route_test",
                cfg=cfg,
                route_id="0",
            )
            writer.finalize()

            with open(os.path.join(tmp_dir, "sequence.json"), "r", encoding="utf-8") as file:
                data = json.load(file)

            self.assertEqual(data["simulation_hz"], 20.0)
            self.assertEqual(data["recording_hz"], 10.0)
            self.assertAlmostEqual(data["recording_interval_seconds"], 0.1)
            self.assertEqual(data["recording_stride_ticks"], 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ================================================================
# 16: resume rejects a 20Hz/10Hz mismatch
# ================================================================

class ResumeRecordingHzMismatchTest(unittest.TestCase):

    def _make_geometry(self, route_path, recording_hz_field):
        geometry_root = geometry_dir(route_path)
        os.makedirs(geometry_root, exist_ok=True)

        sequence = {"fps": 20.0, "num_frames": 10}

        if recording_hz_field is not None:
            sequence["recording_hz"] = recording_hz_field

        with open(os.path.join(geometry_root, "sequence.json"), "w", encoding="utf-8") as file:
            json.dump(sequence, file)

        mark_complete(geometry_root)
        mark_complete(condition_dir(route_path, "day_clear"))

    def test_mismatched_recording_hz_raises(self):
        tmp_dir = tempfile.mkdtemp()

        try:
            route_path = os.path.join(tmp_dir, "route_0")
            self._make_geometry(route_path, recording_hz_field=20.0)

            with self.assertRaises(RuntimeError):
                plan_route_work(
                    route_path,
                    ["day_clear", "day_rain"],
                    "day_clear",
                    expected_recording_hz=10.0,
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_legacy_geometry_with_no_recording_hz_field_is_treated_as_fps(self):
        # Pre-this-task datasets have no "recording_hz" field at all -- they
        # recorded every tick, i.e. recording_hz == fps (20.0 here), so this
        # must ALSO be rejected under a 10 Hz config, not silently resumed.
        tmp_dir = tempfile.mkdtemp()

        try:
            route_path = os.path.join(tmp_dir, "route_0")
            self._make_geometry(route_path, recording_hz_field=None)

            with self.assertRaises(RuntimeError):
                plan_route_work(
                    route_path,
                    ["day_clear", "day_rain"],
                    "day_clear",
                    expected_recording_hz=10.0,
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_matching_recording_hz_does_not_raise(self):
        tmp_dir = tempfile.mkdtemp()

        try:
            route_path = os.path.join(tmp_dir, "route_0")
            self._make_geometry(route_path, recording_hz_field=10.0)

            plan = plan_route_work(
                route_path,
                ["day_clear", "day_rain"],
                "day_clear",
                expected_recording_hz=10.0,
            )
            self.assertFalse(plan["run_canonical"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

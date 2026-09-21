"""
diagnose_replay_fidelity.py

DIAGNOSTIC / RESEARCH ONLY -- not part of production validation.

Production weather replay regenerates only stereo RGB under the recorded
geometry (scripts/collect_dataset.py); depth / semantic / optical flow /
LiDAR / radar equality between replay and canonical is NOT an acceptance
criterion there. This tool deliberately spawns those extra sensors in its own
rig to study how faithfully a replay reproduces the canonical run (it was used
to find the set_transform one-tick lag, the wind-animation and LiDAR-dropout
noise sources). It never writes into geometry/ or conditions/.

Does a weather replay reproduce the canonical geometry exactly?

Replays an already-collected canonical route (dataset/<town>/route_<id>/
geometry, produced by scripts/collect_dataset.py) N times under the SAME
weather the canonical run used (default day_clear), so weather cannot
explain any difference, and compares every replay against the canonical
arrays on disk. Never modifies geometry/.

Checks (see CLAUDE.md verification order):
  2. frame synchronization  per-frame world / snapshot / sensor frame ids
  4. physics / dynamics     post-tick velocity, drift, no autopilot/TM
  5. depth detail           exact ratio, MAE/median/p90..p99.9/max,
                            >1cm/10cm/1m/10m ratios, edge/far/interior
                            attribution, diff maps of the worst frames
  6. semantic               array_equal, mismatch ratio, per-class counts
  7. optical flow           EPE mean/median/p95/p99/max, NaN/inf
  8. determinism            transform error per repeat + replay-vs-replay
                            comparison of every modality
  9. calibration            per-field differences vs float32 spacing
  +  frame-offset probe     the same metrics against canonical frame t-1/t+1

Usage (CARLA server running, geometry already collected):
    python scripts/tools/diagnose_replay_fidelity.py \
        --route-root dataset_smoke/Town01/route_1 --repeats 3

Output: <output-dir>/report.json, report.md, frame_sync.csv, diffmaps/
"""

import argparse
import csv
import json
import os
import queue
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_ROOT = Path(r"C:\CARLA")
CARLA_PYTHONAPI = CARLA_ROOT / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

DEPTH_PERCENTILES = (50, 90, 95, 99, 99.9)
DEPTH_THRESHOLDS_M = (0.01, 0.10, 1.0, 10.0)
FAR_DEPTH_M = 900.0          # CARLA depth saturates at 1000 m (sky / far plane)
EDGE_DILATION_PX = 3
POOLED_SAMPLES_PER_FRAME = 4000
IMAGE_SENSORS = ("rgb_left", "rgb_right", "depth", "semantic", "optical_flow")
POINT_SENSORS = ("lidar", "radar", "radar_front_left", "radar_front_right")
SENSOR_NAMES = IMAGE_SENSORS + POINT_SENSORS
NEAREST_NEIGHBOUR_CHUNK = 512
NEAREST_NEIGHBOUR_SAMPLES = 1500   # query points per cloud (full cloud is the target set)


# ============================================================
# Pure metric functions (no CARLA) -- unit tested offline
# ============================================================

def depth_error_stats(canonical, replayed):
    """Per-frame absolute depth error statistics (meters)."""

    diff = np.abs(canonical.astype(np.float64) - replayed.astype(np.float64))
    flat = diff.ravel()
    percentiles = np.percentile(flat, DEPTH_PERCENTILES)

    stats = {
        "exact_equal_ratio": float(np.mean(flat == 0.0)),
        "mae": float(flat.mean()),
        "median": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "max": float(flat.max()),
    }

    for threshold in DEPTH_THRESHOLDS_M:
        stats[f"gt_{threshold:g}m_ratio"] = float(np.mean(flat > threshold))

    return stats, diff


def edge_mask(canonical_depth, canonical_semantic, dilation_px=EDGE_DILATION_PX):
    """
    Pixels near an object boundary of the canonical frame: a semantic-class
    change, or a relative depth jump > 10 % between 4-neighbours; dilated.
    """

    import cv2

    mask = np.zeros(canonical_depth.shape, dtype=bool)

    for axis in (0, 1):
        sem_change = np.diff(canonical_semantic.astype(np.int16), axis=axis) != 0
        depth_a = np.take(canonical_depth, range(0, canonical_depth.shape[axis] - 1), axis=axis)
        depth_b = np.take(canonical_depth, range(1, canonical_depth.shape[axis]), axis=axis)
        depth_jump = np.abs(depth_a - depth_b) > 0.1 * np.maximum(np.minimum(depth_a, depth_b), 1e-3)
        change = sem_change | depth_jump

        if axis == 0:
            mask[:-1, :] |= change
            mask[1:, :] |= change
        else:
            mask[:, :-1] |= change
            mask[:, 1:] |= change

    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)

    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def depth_error_location(canonical_depth, replayed_depth, canonical_semantic, diff, thresholds=(0.01, 1.0)):
    """
    Where do the large errors sit? Counts pixels with error above each
    threshold split into far-plane / object-boundary / interior.
    """

    far = (canonical_depth >= FAR_DEPTH_M) | (replayed_depth >= FAR_DEPTH_M)
    edge = edge_mask(canonical_depth, canonical_semantic) & ~far
    interior = ~far & ~edge

    result = {}

    for threshold in thresholds:
        over = diff > threshold
        result[f"gt_{threshold:g}m"] = {
            "total": int(over.sum()),
            "far_plane": int((over & far).sum()),
            "object_boundary": int((over & edge).sum()),
            "interior": int((over & interior).sum()),
        }

    result["region_pixels"] = {
        "far_plane": int(far.sum()),
        "object_boundary": int(edge.sum()),
        "interior": int(interior.sum()),
    }

    return result


def semantic_stats(canonical, replayed, num_classes=29):
    mismatch = canonical != replayed
    per_class = {}

    for cls in np.unique(canonical[mismatch]):
        per_class[int(cls)] = int(np.sum(mismatch & (canonical == cls)))

    return {
        "array_equal": bool(np.array_equal(canonical, replayed)),
        "mismatch_pixels": int(mismatch.sum()),
        "mismatch_ratio": float(mismatch.mean()),
        "mismatch_by_canonical_class": per_class,
    }


def flow_stats(canonical, replayed):
    finite = np.isfinite(canonical).all(axis=-1) & np.isfinite(replayed).all(axis=-1)
    epe = np.sqrt(((canonical.astype(np.float64) - replayed.astype(np.float64)) ** 2).sum(axis=-1))
    valid = epe[finite]

    if valid.size == 0:
        return {"nonfinite_pixels": int((~finite).sum()), "valid_pixels": 0}

    percentiles = np.percentile(valid, (50, 95, 99))
    magnitude = np.sqrt((canonical.astype(np.float64) ** 2).sum(axis=-1))[finite]

    return {
        "shape": list(canonical.shape),
        "dtype": str(canonical.dtype),
        "nonfinite_pixels": int((~finite).sum()),
        "valid_pixels": int(valid.size),
        "epe_mean": float(valid.mean()),
        "epe_median": float(percentiles[0]),
        "epe_p95": float(percentiles[1]),
        "epe_p99": float(percentiles[2]),
        "epe_max": float(valid.max()),
        "canonical_flow_magnitude_mean": float(magnitude.mean()),
    }


def point_cloud_stats(canonical, replayed):
    """
    LiDAR / radar comparison. Point order is not meaningful, so equal-size
    clouds are compared after a lexicographic sort; differing sizes fall back
    to nearest-neighbour distances (brute force, chunked).
    """

    out = {
        "n_canonical": int(len(canonical)),
        "n_replay": int(len(replayed)),
        "count_equal": bool(len(canonical) == len(replayed)),
    }

    if len(canonical) == 0 or len(replayed) == 0:
        out["exact_equal"] = bool(len(canonical) == len(replayed))
        return out

    a = canonical.astype(np.float64)
    b = replayed.astype(np.float64)

    if len(a) == len(b):
        sa = a[np.lexsort(a.T[::-1])]
        sb = b[np.lexsort(b.T[::-1])]
        out["exact_equal"] = bool(np.array_equal(sa, sb))
        out["sorted_max_abs_difference"] = float(np.abs(sa - sb).max())

    def nearest(x, y):
        distances = []

        if len(x) > NEAREST_NEIGHBOUR_SAMPLES:
            x = x[np.random.default_rng(0).choice(len(x), NEAREST_NEIGHBOUR_SAMPLES, replace=False)]

        for start in range(0, len(x), NEAREST_NEIGHBOUR_CHUNK):
            chunk = x[start:start + NEAREST_NEIGHBOUR_CHUNK, :3]
            d2 = ((chunk[:, None, :] - y[None, :, :3]) ** 2).sum(axis=-1)
            distances.append(np.sqrt(d2.min(axis=1)))

        return np.concatenate(distances)

    forward, backward = nearest(a, b), nearest(b, a)
    out["nn_mean_m"] = float((forward.mean() + backward.mean()) / 2)
    out["nn_max_m"] = float(max(forward.max(), backward.max()))
    out["within_1cm_ratio"] = float((np.mean(forward < 0.01) + np.mean(backward < 0.01)) / 2)

    return out


def label_world_consistency(label_objects, ego_matrix, actors_world, max_distance_m=3.0):
    """
    Labels are stored in the ego frame; map each object centre back to world
    with the recorded ego transform and compare with the recorded position
    of the actor carrying the same logical_id (the bbox centre is offset from
    the actor origin by less than max_distance_m).

    ego_matrix: 4x4 world<-ego; actors_world: {logical_id(str): (x, y, z)}.
    Returns (checked, mismatched, worst_distance_m).
    """

    checked = mismatched = 0
    worst = 0.0

    for obj in label_objects:
        logical_id = obj.get("logical_id")
        checked += 1

        if logical_id is None or str(logical_id) not in actors_world:
            mismatched += 1
            continue

        center_ego = np.array(list(obj["bbox_3d"]["center_ego_m"]) + [1.0])
        center_world = (ego_matrix @ center_ego)[:3]
        distance = float(np.linalg.norm(center_world - np.array(actors_world[str(logical_id)])))
        worst = max(worst, distance)

        if distance > max_distance_m:
            mismatched += 1

    return checked, mismatched, worst


def _numeric_leaves(tree):
    """Flatten nested lists of numbers; None if any non-numeric leaf."""

    if isinstance(tree, list):
        out = []

        for item in tree:
            leaves = _numeric_leaves(item)

            if leaves is None:
                return None

            out.extend(leaves)

        return out

    if isinstance(tree, (int, float)) and not isinstance(tree, bool):
        return [float(tree)]

    return None


def calibration_field_differences(canonical, replayed, prefix=""):
    """
    [(json_path, max_abs_difference)]: one entry per numeric leaf, or per
    numeric matrix/vector (max over its elements), present in both trees.
    """

    if isinstance(canonical, dict) and isinstance(replayed, dict):
        out = []

        for key in sorted(canonical.keys() & replayed.keys()):
            out.extend(calibration_field_differences(canonical[key], replayed[key], f"{prefix}/{key}"))

        return out

    a, b = _numeric_leaves(canonical), _numeric_leaves(replayed)

    if a is not None and b is not None and len(a) == len(b):
        return [(prefix, float(max((abs(x - y) for x, y in zip(a, b)), default=0.0)))]

    return []


def frame_sync_row(local_frame_id, recorded_state, tick_frame, snapshot_frame, sensor_frames, first_tick_frame):
    """
    One row of the synchronization table plus the mismatch flags.
    replay_offset: replay world frame relative to the first replay frame,
    compared with the canonical world frame relative to the canonical first.
    """

    row = {
        "local_frame_id": local_frame_id,
        "recorded_state_frame_id": recorded_state["frame_id"],
        "canonical_world_frame": recorded_state["carla_frame"],
        "replay_tick_frame": tick_frame,
        "replay_snapshot_frame": snapshot_frame,
    }

    for name, frame in sensor_frames.items():
        row[f"{name}_frame"] = frame

    row["tick_vs_snapshot"] = snapshot_frame - tick_frame
    row["max_sensor_offset_vs_tick"] = max(f - tick_frame for f in sensor_frames.values())
    row["min_sensor_offset_vs_tick"] = min(f - tick_frame for f in sensor_frames.values())
    row["sensor_spread"] = max(sensor_frames.values()) - min(sensor_frames.values())
    row["state_id_vs_local"] = recorded_state["frame_id"] - local_frame_id
    row["replay_frames_since_first"] = tick_frame - first_tick_frame

    return row


# ============================================================
# CARLA-dependent part
# ============================================================

def lenient_collect(rig_queues, names, target_frame, timeout=15.0):
    """
    Like Collector.collect_frame but records what each sensor delivered
    instead of raising on a frame mismatch.
    """

    packet = {}
    deadline = time.monotonic() + timeout

    for name in names:
        while True:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise RuntimeError(f"timeout waiting for {name} at frame {target_frame}")

            try:
                data = rig_queues[name].get(timeout=remaining)
            except queue.Empty as exc:
                raise RuntimeError(f"timeout waiting for {name} at frame {target_frame}") from exc

            if data.frame < target_frame:
                continue

            packet[name] = data
            break

    return packet


def load_canonical(geometry_root, name, frame_id):
    return np.load(os.path.join(geometry_root, name, f"{frame_id:06d}.npy"))


def aggregate(rows, keys):
    """mean and max of per-frame values for the given keys."""

    out = {}

    for key in keys:
        values = np.array([row[key] for row in rows if key in row], dtype=np.float64)

        if values.size:
            out[key] = {"mean": float(values.mean()), "max": float(values.max()), "min": float(values.min())}

    return out


def save_diffmap(path, canonical_depth, replayed_depth, diff):
    import cv2

    def to_gray(depth):
        return (np.clip(np.log1p(depth) / np.log1p(1000.0), 0, 1) * 255).astype(np.uint8)

    canonical_img = cv2.cvtColor(to_gray(canonical_depth), cv2.COLOR_GRAY2BGR)
    replay_img = cv2.cvtColor(to_gray(replayed_depth), cv2.COLOR_GRAY2BGR)

    # log-scaled error: 1 mm -> dark, 10 m -> bright
    scaled = np.clip((np.log10(np.maximum(diff, 1e-4)) + 4.0) / 5.0, 0, 1)
    error_img = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    error_img[diff <= 0.0] = 0

    for img, label in ((canonical_img, "canonical depth"), (replay_img, "replay depth"), (error_img, "|diff| log 1e-4..10m")):
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(path, np.concatenate([canonical_img, replay_img, error_img], axis=1))


def run_replay(world, client, reader, condition, geometry_root, canonical_calibration, repeat_index,
               reference_dir, args, diffmap_dir):
    from scripts.collect_dataset import EGO_BLUEPRINT, WARMUP_FRAMES, get_sensor_actors
    from src.data.calibration import build_calibration
    from src.simulation.replay import WorldStateReplayer
    from src.simulation.weather import apply_weather
    from src.sensors.sensor_rig import SensorRig
    from CFG.config import cfg

    num_frames = len(reader) if args.max_frames is None else min(len(reader), args.max_frames)

    weather = apply_weather(world, condition)

    if args.wind_intensity is not None:
        weather.wind_intensity = args.wind_intensity
        world.set_weather(weather)

    first_frame = reader.load_frame(0)

    replayer = WorldStateReplayer(
        world, client, reader, EGO_BLUEPRINT,
        position_tolerance_m=args.position_tolerance,
        rotation_tolerance_deg=args.rotation_tolerance,
    )
    rig = None

    try:
        replayer.start(first_frame)
        sensor_names = IMAGE_SENSORS if args.skip_point_sensors else SENSOR_NAMES
        rig = SensorRig(world, replayer.ego, cfg).spawn(sensor_names=sensor_names)

        # Same order as scripts/collect_dataset.py replay_condition().
        replayer.apply_frame(first_frame)
        world.tick()

        calibration = build_calibration(ego_vehicle=replayer.ego, sensor_actors=get_sensor_actors(rig))

        for _ in range(WARMUP_FRAMES):
            replayer.apply_frame(first_frame)
            world.tick()

        rig.clear_queues()

        sync_rows = []
        depth_rows, semantic_rows, flow_rows = [], [], []
        depth_location_totals = {}
        pooled_depth = []
        worst = []  # (score, frame_id, canonical, replay, diff) kept in memory
        offset_probe = {k: {"depth_mae": [], "semantic_mismatch": [], "flow_epe": []} for k in (-1, 0, 1)}
        vs_reference = {"depth": [], "semantic": [], "optical_flow": []}
        point_rows = {name: [] for name in POINT_SENSORS}
        from src.data.collector import filter_lidar_roi, lidar_to_numpy, radar_to_numpy
        from src.sensors.lidar import create_lidar_transform
        t_ego_from_lidar = np.array(create_lidar_transform(cfg).get_matrix(), dtype=np.float64)
        physics_rows = []
        rng = np.random.default_rng(0)
        first_tick_frame = None

        store_dir = os.path.join(reference_dir, "repeat0")

        for frame_id in range(num_frames):
            state = first_frame if frame_id == 0 else reader.load_frame(frame_id)

            # --- execution order under test (mirrors replay_condition) ---
            replayer.apply_frame(state)
            tick_frame = world.tick()
            packet = lenient_collect(rig.queues, sensor_names, tick_frame)
            snapshot = world.get_snapshot()
            replayer.verify_frame(state)
            # -------------------------------------------------------------

            if first_tick_frame is None:
                first_tick_frame = tick_frame

            sync_rows.append(frame_sync_row(
                frame_id, state, tick_frame, snapshot.frame,
                {name: packet[name].frame for name in sensor_names}, first_tick_frame,
            ))

            # physics / dynamics: everything replayed with physics off must be
            # motionless after the tick (velocities are never applied).
            vehicle_speeds = []
            for logical_id, actor in replayer.actors.items():
                if not reader.actors[logical_id]["blueprint_id"].startswith("walker."):
                    actor_snapshot = snapshot.find(actor.id)
                    if actor_snapshot is not None:
                        velocity = actor_snapshot.get_velocity()
                        vehicle_speeds.append((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)
            ego_velocity = snapshot.find(replayer.ego.id).get_velocity()
            physics_rows.append({
                "ego_speed_after_tick": (ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2) ** 0.5,
                "max_vehicle_speed_after_tick": max(vehicle_speeds) if vehicle_speeds else 0.0,
            })

            from src.data.collector import depth_to_numpy, semantic_to_numpy, optical_flow_to_numpy

            replay_depth = depth_to_numpy(packet["depth"])
            replay_semantic = semantic_to_numpy(packet["semantic"])
            replay_flow = optical_flow_to_numpy(packet["optical_flow"])

            canonical_depth = load_canonical(geometry_root, "depth", frame_id)
            canonical_semantic = load_canonical(geometry_root, "semantic", frame_id)
            canonical_flow = load_canonical(geometry_root, "optical_flow", frame_id)

            stats, diff = depth_error_stats(canonical_depth, replay_depth)
            stats["frame_id"] = frame_id
            depth_rows.append(stats)
            semantic_rows.append({"frame_id": frame_id, **semantic_stats(canonical_semantic, replay_semantic)})
            flow_rows.append({"frame_id": frame_id, **flow_stats(canonical_flow, replay_flow)})

            location = depth_error_location(canonical_depth, replay_depth, canonical_semantic, diff)

            for key, value in location.items():
                if isinstance(value, dict):
                    bucket = depth_location_totals.setdefault(key, {})
                    for sub, count in value.items():
                        bucket[sub] = bucket.get(sub, 0) + count

            flat = diff.ravel()
            pooled_depth.append(flat[rng.integers(0, flat.size, POOLED_SAMPLES_PER_FRAME)])

            worst.append((stats["mae"], frame_id, canonical_depth, replay_depth, diff))
            worst.sort(key=lambda item: -item[0])
            del worst[args.diffmap_count:]

            if not args.skip_point_sensors:
                point_rows["lidar"].append(point_cloud_stats(
                    load_canonical(geometry_root, "lidar", frame_id),
                    filter_lidar_roi(lidar_to_numpy(packet["lidar"]), t_ego_from_lidar, cfg)))

                for radar_name in POINT_SENSORS[1:]:
                    point_rows[radar_name].append(point_cloud_stats(
                        load_canonical(geometry_root, radar_name, frame_id), radar_to_numpy(packet[radar_name])))

            # frame-offset probe (sampled): does the replay match canonical t-1 / t+1 better?
            if frame_id % args.offset_probe_stride == 0 and 0 < frame_id < len(reader) - 1:
                for k in (-1, 0, 1):
                    other = frame_id + k
                    offset_probe[k]["depth_mae"].append(float(np.mean(np.abs(
                        load_canonical(geometry_root, "depth", other).astype(np.float64) - replay_depth))))
                    offset_probe[k]["semantic_mismatch"].append(float(np.mean(
                        load_canonical(geometry_root, "semantic", other) != replay_semantic)))
                    other_flow = load_canonical(geometry_root, "optical_flow", other)
                    offset_probe[k]["flow_epe"].append(float(np.mean(np.sqrt(
                        ((other_flow.astype(np.float64) - replay_flow) ** 2).sum(axis=-1)))))

            # replay-vs-replay determinism
            if repeat_index == 0:
                os.makedirs(store_dir, exist_ok=True)
                np.savez(os.path.join(store_dir, f"{frame_id:06d}.npz"),
                         depth=replay_depth, semantic=replay_semantic, flow=replay_flow)
            else:
                reference = np.load(os.path.join(store_dir, f"{frame_id:06d}.npz"))
                d_stats, _ = depth_error_stats(reference["depth"], replay_depth)
                vs_reference["depth"].append(d_stats)
                vs_reference["semantic"].append(semantic_stats(reference["semantic"], replay_semantic))
                vs_reference["optical_flow"].append(flow_stats(reference["flow"], replay_flow))

            if frame_id == 0 or (frame_id + 1) % 50 == 0:
                print(f"  [repeat {repeat_index}] frame {frame_id + 1}/{num_frames} "
                      f"depth_mae={stats['mae']:.4f} sem_mismatch={semantic_rows[-1]['mismatch_ratio']:.5f}")

        os.makedirs(diffmap_dir, exist_ok=True)

        for _score, frame_id, canonical_depth, replay_depth, diff in worst:
            save_diffmap(os.path.join(diffmap_dir, f"repeat{repeat_index}_frame{frame_id:06d}.png"),
                         canonical_depth, replay_depth, diff)

        pooled = np.concatenate(pooled_depth)

        calibration_diffs = calibration_field_differences(canonical_calibration, calibration)

        return {
            "replay_summary": replayer.summary(),
            "sync_rows": sync_rows,
            "depth_rows": depth_rows,
            "semantic_rows": semantic_rows,
            "flow_rows": flow_rows,
            "depth_location_totals": depth_location_totals,
            "depth_pooled_percentiles": {
                f"p{q:g}": float(np.percentile(pooled, q)) for q in (50, 90, 95, 99, 99.9)
            },
            "offset_probe": {str(k): {m: float(np.mean(v)) if v else None for m, v in d.items()}
                             for k, d in offset_probe.items()},
            "vs_reference": vs_reference,
            "physics_rows": physics_rows,
            "point_rows": point_rows,
            "calibration": calibration,
            "calibration_diffs": calibration_diffs,
            "ego_frame0": first_frame["ego"]["transform"],
        }

    finally:
        if rig is not None:
            try:
                rig.destroy()
            except Exception as exc:
                print(f"[cleanup] rig: {exc}")

        try:
            replayer.destroy_all()
        except Exception as exc:
            print(f"[cleanup] replayer: {exc}")

        try:
            world.tick()
        except RuntimeError:
            pass


def summarize_run(result):
    depth = aggregate(result["depth_rows"], ["exact_equal_ratio", "mae", "median", "p90", "p95", "p99", "p99_9",
                                             "max", "gt_0.01m_ratio", "gt_0.1m_ratio", "gt_1m_ratio", "gt_10m_ratio"])
    semantic_rows = result["semantic_rows"]
    class_totals = {}

    for row in semantic_rows:
        for cls, count in row["mismatch_by_canonical_class"].items():
            class_totals[cls] = class_totals.get(cls, 0) + count

    flow = aggregate([r for r in result["flow_rows"] if "epe_mean" in r],
                     ["epe_mean", "epe_median", "epe_p95", "epe_p99", "epe_max"])

    return {
        "depth": depth,
        "depth_pooled_percentiles_m": result["depth_pooled_percentiles"],
        "depth_error_location_pixels": result["depth_location_totals"],
        "semantic": {
            "frames_array_equal": int(sum(r["array_equal"] for r in semantic_rows)),
            "frames": len(semantic_rows),
            "mismatch_ratio_mean": float(np.mean([r["mismatch_ratio"] for r in semantic_rows])),
            "mismatch_ratio_max": float(np.max([r["mismatch_ratio"] for r in semantic_rows])),
            "mismatch_pixels_by_canonical_class": class_totals,
        },
        "optical_flow": {
            **flow,
            "frames_with_nonfinite": int(sum(r.get("nonfinite_pixels", 0) > 0 for r in result["flow_rows"])),
            "shape": result["flow_rows"][0].get("shape"),
            "dtype": result["flow_rows"][0].get("dtype"),
            "epe_mean_by_frame_first5": [r.get("epe_mean") for r in result["flow_rows"][:5]],
        },
        "frame_offset_probe": result["offset_probe"],
    }


def summarize_points(point_rows):
    out = {}

    for name, rows in point_rows.items():
        if not rows:
            continue

        nn = [r["nn_mean_m"] for r in rows if "nn_mean_m" in r]
        within = [r["within_1cm_ratio"] for r in rows if "within_1cm_ratio" in r]

        out[name] = {
            "frames": len(rows),
            "frames_count_equal": int(sum(r["count_equal"] for r in rows)),
            "frames_exact_equal": int(sum(bool(r.get("exact_equal")) for r in rows)),
            "n_canonical_mean": float(np.mean([r["n_canonical"] for r in rows])),
            "n_replay_mean": float(np.mean([r["n_replay"] for r in rows])),
            "sorted_max_abs_difference_max": float(max((r.get("sorted_max_abs_difference", 0.0) for r in rows), default=0.0)),
            "nn_mean_m_mean": float(np.mean(nn)) if nn else None,
            "nn_max_m_max": float(max((r["nn_max_m"] for r in rows if "nn_max_m" in r), default=0.0)),
            "within_1cm_ratio_mean": float(np.mean(within)) if within else None,
        }

    return out


def labels_and_pose_check(geometry_root, reader):
    """Disk-only checks: labels vs recorded world state, poses.csv vs ego state."""

    import carla

    label_dir = os.path.join(geometry_root, "labels", "object_3d")
    checked = mismatched = 0
    worst = 0.0

    for frame_id in range(len(reader)):
        state = reader.load_frame(frame_id)
        ego = state["ego"]["transform"]
        ego_matrix = np.array(carla.Transform(
            carla.Location(x=ego["x"], y=ego["y"], z=ego["z"]),
            carla.Rotation(roll=ego["roll"], pitch=ego["pitch"], yaw=ego["yaw"])).get_matrix(), dtype=np.float64)
        actors_world = {k: (v["transform"]["x"], v["transform"]["y"], v["transform"]["z"])
                        for k, v in state["actors"].items()}

        with open(os.path.join(label_dir, f"{frame_id:06d}.json"), "r", encoding="utf-8") as f:
            objects = json.load(f)["objects"]

        c, m, w = label_world_consistency(objects, ego_matrix, actors_world)
        checked += c
        mismatched += m
        worst = max(worst, w)

    with open(os.path.join(geometry_root, "pose", "poses.csv"), newline="", encoding="utf-8") as f:
        pose_rows = list(csv.DictReader(f))

    pose_worst = 0.0

    for frame_id, row in enumerate(pose_rows):
        ego = reader.load_frame(frame_id)["ego"]["transform"]

        for key_csv, key_state in (("x", "x"), ("y", "y"), ("z", "z"),
                                   ("roll_deg", "roll"), ("pitch_deg", "pitch"), ("yaw_deg", "yaw")):
            pose_worst = max(pose_worst, abs(float(row[key_csv]) - ego[key_state]))

    return {
        "label_objects_checked": checked,
        "label_objects_not_matching_recorded_actor": mismatched,
        "label_center_to_actor_worst_distance_m": worst,
        "poses_csv_rows": len(pose_rows),
        "poses_csv_vs_world_state_ego_max_abs_difference": pose_worst,
    }


def summarize_sync(rows):
    keys = ["tick_vs_snapshot", "max_sensor_offset_vs_tick", "min_sensor_offset_vs_tick",
            "sensor_spread", "state_id_vs_local"]
    out = {k: sorted({row[k] for row in rows}) for k in keys}
    deltas = np.diff([row["replay_tick_frame"] for row in rows])
    canonical_deltas = np.diff([row["canonical_world_frame"] for row in rows])
    out["replay_tick_frame_deltas"] = sorted({int(d) for d in deltas})
    out["canonical_world_frame_deltas"] = sorted({int(d) for d in canonical_deltas})
    out["all_synchronized"] = bool(
        out["tick_vs_snapshot"] == [0] and out["sensor_spread"] == [0]
        and out["max_sensor_offset_vs_tick"] == [0] and out["state_id_vs_local"] == [0]
        and out["replay_tick_frame_deltas"] == [1] and out["canonical_world_frame_deltas"] == [1]
    )
    return out


def summarize_physics(rows):
    return {
        "ego_speed_after_tick_max_mps": float(max(r["ego_speed_after_tick"] for r in rows)),
        "vehicle_speed_after_tick_max_mps": float(max(r["max_vehicle_speed_after_tick"] for r in rows)),
    }


def source_uses_autopilot_or_tm(path):
    """Static check: replay.py must never touch autopilot / Traffic Manager."""

    text = Path(path).read_text(encoding="utf-8")
    needles = ("set_autopilot", "traffic_manager", "TrafficManager", "RouteController", "BasicAgent",
               "DynamicSpawnManager", "CanonicalBackgroundTraffic", "spawn_actors_gamma_policy")

    return [needle for needle in needles if needle in text]


def main():
    parser = argparse.ArgumentParser(description="Canonical-vs-replay fidelity diagnostics.")
    parser.add_argument("--route-root", type=str, default="dataset_smoke/Town01/route_1")
    parser.add_argument("--condition", type=str, default="day_clear",
                        help="Weather to replay under; use the canonical one to isolate replay effects.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/replay_fidelity")
    parser.add_argument("--wind-intensity", type=float, default=None,
                        help="Override the replay weather's wind (isolates foliage/wind animation as a noise source).")
    parser.add_argument("--skip-point-sensors", action="store_true", help="Do not capture/compare lidar and radar.")
    parser.add_argument("--cross-conditions", nargs="*", default=[],
                        help="After the repeats, replay once under each of these weathers and compare with canonical.")
    parser.add_argument("--diffmap-count", type=int, default=4)
    parser.add_argument("--offset-probe-stride", type=int, default=10)
    parser.add_argument("--position-tolerance", type=float, default=0.05)
    parser.add_argument("--rotation-tolerance", type=float, default=0.5)
    args = parser.parse_args()

    import carla
    from CFG.config import cfg
    from scripts.collect_dataset import resolve_carla_map_name, route_xml_path
    from src.data.layout import geometry_dir
    from src.data.world_state import WorldStateReader
    from src.simulation.environment import disable_static_traffic_objects

    route_root = os.path.abspath(args.route_root)
    geometry_root = geometry_dir(route_root)
    output_dir = os.path.abspath(args.output_dir)
    diffmap_dir = os.path.join(output_dir, "diffmaps")
    reference_dir = os.path.join(output_dir, "_reference")

    os.makedirs(diffmap_dir, exist_ok=True)

    with open(os.path.join(geometry_root, "sequence.json"), "r", encoding="utf-8") as f:
        sequence = json.load(f)

    with open(os.path.join(geometry_root, "calibration.json"), "r", encoding="utf-8") as f:
        canonical_calibration = json.load(f)

    reader = WorldStateReader(geometry_root)

    with open(os.path.join(geometry_root, "timestamps.csv"), newline="", encoding="utf-8") as f:
        timestamps = list(csv.DictReader(f))

    timestamp_check = {
        "csv_frames_match_world_state": all(
            int(row["carla_frame"]) == reader.load_frame(i)["carla_frame"] for i, row in enumerate(timestamps[:len(reader)])
        ),
        "csv_rows": len(timestamps),
        "world_states": len(reader),
        "canonical_timestamp_delta_s": sorted({round(float(b["timestamp"]) - float(a["timestamp"]), 6)
                                               for a, b in zip(timestamps, timestamps[1:])}),
        "cfg_fixed_delta_s": cfg.SIMULATION.FIXED_DELTA_SECONDS,
    }

    client = carla.Client(cfg.CARLA.HOST, cfg.CARLA.PORT)
    client.set_timeout(cfg.CARLA.TIMEOUT)

    town = sequence["town"]
    world = client.load_world(resolve_carla_map_name(route_xml_path(town)))
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.SIMULATION.FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    disable_static_traffic_objects(world)

    runs = []

    try:
        for repeat in range(args.repeats):
            print(f"[diagnose] replay {repeat + 1}/{args.repeats} ({args.condition}, {len(reader)} frames)")
            runs.append(run_replay(world, client, reader, args.condition, geometry_root, canonical_calibration,
                                   repeat, reference_dir, args, diffmap_dir))
    finally:
        world.apply_settings(original_settings)

    cross_weather = {}

    for other in args.cross_conditions:
        print(f"[diagnose] cross-weather replay: {other}")
        cross_run = run_replay(world, client, reader, other, geometry_root, canonical_calibration,
                               0, os.path.join(output_dir, "_cross"), args, os.path.join(diffmap_dir, other))
        cross_weather[other] = {
            **summarize_run(cross_run),
            "points": summarize_points(cross_run["point_rows"]),
            "transform": {k: cross_run["replay_summary"][k] for k in ("passed", "ego", "vehicle_like_actors")},
        }

    # ---------------- assemble the report ----------------

    report = {
        "route_root": route_root,
        "condition": args.condition,
        "geometry_source_condition": sequence.get("geometry_source_condition"),
        "repeats": args.repeats,
        "frames": len(reader) if args.max_frames is None else min(len(reader), args.max_frames),
        "timestamp_check": timestamp_check,
        "replay_source_static_check_forbidden_symbols": source_uses_autopilot_or_tm(PROJECT_ROOT / "src" / "simulation" / "replay.py"),
        "frame_sync": summarize_sync(runs[0]["sync_rows"]),
        "physics_after_tick": [summarize_physics(r["physics_rows"]) for r in runs],
        "transform_repeatability": [
            {k: r["replay_summary"][k] for k in ("passed", "ego", "vehicle_like_actors", "pedestrians",
                                                 "out_of_tolerance_samples", "worst_out_of_tolerance",
                                                 "actor_id_set_mismatch_frames", "total_actor_spawns")}
            for r in runs
        ],
        "canonical_vs_replay": [summarize_run(r) for r in runs],
        "replay_vs_replay": [],
        "wind_intensity_override": args.wind_intensity,
        "point_sensors": [summarize_points(r["point_rows"]) for r in runs],
        "cross_weather": cross_weather,
        "labels_and_pose": labels_and_pose_check(geometry_root, reader),
    }

    for r in runs[1:]:
        ref = r["vs_reference"]
        report["replay_vs_replay"].append({
            "depth_exact_equal_ratio_mean": float(np.mean([s["exact_equal_ratio"] for s in ref["depth"]])),
            "depth_mae_mean": float(np.mean([s["mae"] for s in ref["depth"]])),
            "depth_max": float(np.max([s["max"] for s in ref["depth"]])),
            "semantic_frames_array_equal": int(sum(s["array_equal"] for s in ref["semantic"])),
            "semantic_mismatch_ratio_mean": float(np.mean([s["mismatch_ratio"] for s in ref["semantic"]])),
            "flow_epe_mean": float(np.mean([s["epe_mean"] for s in ref["optical_flow"] if "epe_mean" in s])),
            "flow_epe_max": float(np.max([s["epe_max"] for s in ref["optical_flow"] if "epe_max" in s])),
        })

    # ---- calibration ----
    diffs = runs[0]["calibration_diffs"]
    diffs_sorted = sorted(diffs, key=lambda item: -item[1])
    ego_xyz = runs[0]["ego_frame0"]
    coordinate_magnitude = max(abs(ego_xyz["x"]), abs(ego_xyz["y"]), abs(ego_xyz["z"]))
    calibration_between_repeats = [
        max((d for _p, d in calibration_field_differences(runs[0]["calibration"], r["calibration"])), default=0.0)
        for r in runs[1:]
    ]
    report["calibration"] = {
        "max_abs_difference_vs_canonical": diffs_sorted[0][1] if diffs_sorted else None,
        "top_fields": [{"path": p, "abs_difference": d} for p, d in diffs_sorted[:12]],
        "fields_over_1e-6": sum(1 for _p, d in diffs if d > 1e-6),
        "fields_compared": len(diffs),
        "ego_frame0_max_abs_coordinate_m": coordinate_magnitude,
        "float32_spacing_at_that_coordinate": float(np.spacing(np.float32(coordinate_magnitude))),
        "max_abs_difference_between_replay_repeats": calibration_between_repeats,
    }

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(output_dir, "frame_sync.csv"), "w", newline="", encoding="utf-8") as f:
        rows = runs[0]["sync_rows"]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    shutil.rmtree(reference_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, "_cross"), ignore_errors=True)

    print(json.dumps({k: report[k] for k in ("frame_sync", "transform_repeatability", "calibration")}, indent=2))
    print(f"[diagnose] wrote {output_dir}/report.json")


if __name__ == "__main__":
    main()

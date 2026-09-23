"""
scripts/tools/analyze_route_stall.py

Offline (no CARLA server) stall analysis of an existing -- possibly
partial / still-running -- canonical geometry sequence. Identifies the
persistent forward blocker in front of a stalled ego from world_state/,
labels/, ego_state.csv and actors.json, and reports whether the SAME
actor stays stopped in front of the ego and what the ego controller was
commanding (throttle / brake) meanwhile.

Road/lane ids and route arc length are optional: they are added when the
`carla` Python module and the map's OpenDRIVE file are available
(carla.Map is built from the .xodr file locally, no server needed).

Usage (PowerShell):

    python scripts/tools/analyze_route_stall.py `
        --dataset-root D:\\carla_dataset --town Town10 --route 1 `
        --start-frame 3900 --end-frame 4200

Writes <geometry>/stall_analysis_<start>_<end>/{frames.csv,summary.json}
unless --out-dir is given.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.stall_diagnostics import (  # noqa: E402
    FORWARD_CORRIDOR_HALF_WIDTH_M,
    FORWARD_SEARCH_MAX_M,
    find_nearest_forward_actor,
)

VEHICLE_LIKE = ("vehicle", "motorcyclist", "cyclist")

# 1 km/h: same threshold as the live stall condition.
STATIONARY_SPEED_MPS = 1.0 / 3.6
# An actor "moved" in a frame if it travelled more than this since the
# previous recorded frame (0.1 s apart at 10 Hz).
MOVEMENT_EPS_M = 0.05
TRAFFIC_LIGHT_NEAR_M = 30.0


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def resolve_geometry(dataset_root, town, route):
    root = Path(dataset_root)

    for candidate in (root / town / f"route_{route}" / "geometry", root / "geometry", root):
        if (candidate / "world_state").is_dir():
            return candidate

    raise FileNotFoundError(f"No geometry/world_state under {root} for {town} route {route}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_by_frame(path):
    rows = {}

    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["frame_id"])] = row
            except (KeyError, ValueError):
                continue

    return rows


def load_labels(label_dir, frame_id):
    path = label_dir / f"{frame_id:06d}.json"

    if not path.exists():
        return {}

    try:
        data = load_json(path)
    except (json.JSONDecodeError, OSError):
        return {}

    return {
        obj["logical_id"]: obj
        for obj in data.get("objects", [])
        if obj.get("logical_id") is not None
    }


def speed_of(velocity):
    return math.sqrt(velocity["x"] ** 2 + velocity["y"] ** 2 + velocity["z"] ** 2)


def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# Optional map (road / lane / route arc length)
# ------------------------------------------------------------------

class OfflineMap:
    def __init__(self, xodr_path, route_xml=None, route_id=None, carla_root=None):
        carla_root = Path(carla_root or os.environ.get("CARLA_ROOT", r"C:\CARLA"))
        sys.path.insert(0, str(carla_root / "PythonAPI" / "carla"))

        import carla  # noqa: WPS433 (optional dependency)

        self.carla = carla

        with open(xodr_path, "r", encoding="utf-8") as f:
            self.map = carla.Map(Path(xodr_path).stem, f.read())

        self.arc_table = None

        if route_xml is not None and route_id is not None:
            from src.navigation.route import build_dense_route, load_route_from_xml, project_control_points
            from src.simulation.spawn_policy import build_route_arc_length_table

            _town, control_points = load_route_from_xml(route_xml, route_id)
            self.dense_route = build_dense_route(self.map, project_control_points(self.map, control_points), 2.0)
            self.arc_table = build_route_arc_length_table(self.dense_route)

    def road_lane(self, x, y, z=0.0):
        waypoint = self.map.get_waypoint(self.carla.Location(x=x, y=y, z=z))

        if waypoint is None:
            return None, None, None

        return waypoint.road_id, waypoint.lane_id, waypoint.is_junction

    def route_s(self, route_index):
        if self.arc_table is None or route_index is None:
            return None

        return self.arc_table[max(0, min(int(route_index), len(self.arc_table) - 1))]


def find_xodr(map_name, carla_root):
    carla_root = Path(carla_root or os.environ.get("CARLA_ROOT", r"C:\CARLA"))
    stem = Path(map_name).name
    known = carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{stem}.xodr"

    if known.exists():
        return known

    for candidate in (carla_root / "CarlaUE4" / "Content").rglob(f"{stem}.xodr"):
        return candidate

    return None


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def nearest_traffic_light(traffic_lights, x, y):
    best = None

    for key, state in traffic_lights.items():
        try:
            lx, ly, _lz = (float(v) for v in key.split("_"))
        except ValueError:
            continue

        distance = math.hypot(lx - x, ly - y)

        if distance <= TRAFFIC_LIGHT_NEAR_M and (best is None or distance < best[1]):
            best = (key, distance, state)

    return best


def analyze(geometry, start_frame, end_frame, offline_map=None, corridor=FORWARD_CORRIDOR_HALF_WIDTH_M):
    state_dir = geometry / "world_state"
    label_dir = geometry / "labels" / "object_3d"

    actors_meta = {}

    if (geometry / "actors.json").exists():
        actors_meta = {int(k): v for k, v in load_json(geometry / "actors.json").items()}

    ego_rows = load_csv_by_frame(geometry / "ego_state.csv")

    frames = []
    last_actor_positions = {}
    last_movement_frame = {}

    available = sorted(int(p.stem) for p in state_dir.glob("*.json"))
    selected = [f for f in available if start_frame <= f <= end_frame]

    if not selected:
        raise RuntimeError(
            f"No world_state frames in [{start_frame}, {end_frame}] "
            f"(available {available[0] if available else '-'}..{available[-1] if available else '-'})"
        )

    for frame_id in selected:
        try:
            state = load_json(state_dir / f"{frame_id:06d}.json")
        except (json.JSONDecodeError, OSError):
            # The last file of a still-running sequence may be half-written.
            continue

        ego_tf = state["ego"]["transform"]
        ego_xy = (ego_tf["x"], ego_tf["y"])
        ego_yaw = ego_tf["yaw"]
        ego_speed = speed_of(state["ego"]["velocity"])

        candidates = []
        actor_states = {}

        for key, record in state["actors"].items():
            logical_id = int(key)
            meta = actors_meta.get(logical_id, {})
            tf = record["transform"]
            actor_states[logical_id] = record

            previous = last_actor_positions.get(logical_id)
            if previous is None or math.hypot(tf["x"] - previous[0], tf["y"] - previous[1]) > MOVEMENT_EPS_M:
                last_movement_frame[logical_id] = frame_id
            last_actor_positions[logical_id] = (tf["x"], tf["y"])

            if meta.get("category", "vehicle") in VEHICLE_LIKE:
                candidates.append((logical_id, tf["x"], tf["y"]))

        hit = find_nearest_forward_actor(ego_xy, ego_yaw, candidates, max_lateral_m=corridor)
        hit_any = find_nearest_forward_actor(ego_xy, ego_yaw, candidates, max_lateral_m=None)

        labels = load_labels(label_dir, frame_id)
        ego_row = ego_rows.get(frame_id, {})

        row = {
            "frame_id": frame_id,
            "timestamp": state.get("timestamp"),
            "ego_x": ego_tf["x"],
            "ego_y": ego_tf["y"],
            "ego_yaw": ego_yaw,
            "ego_speed_kmh": ego_speed * 3.6,
            "ego_throttle": fnum(ego_row.get("throttle")),
            "ego_brake": fnum(ego_row.get("brake")),
            "ego_steer": fnum(ego_row.get("steer")),
            "ego_hand_brake": ego_row.get("hand_brake"),
            "ego_route_index": ego_row.get("route_index"),
            "ego_route_progress": fnum(ego_row.get("route_progress")),
            "ego_road_id": None,
            "ego_lane_id": None,
            "ego_route_s": None,
            "blocker_logical_id": None,
            "blocker_forward_any_logical_id": hit_any["key"] if hit_any else None,
            "blocker_forward_any_distance_m": hit_any["longitudinal_m"] if hit_any else None,
            "n_actors": len(state["actors"]),
            "n_stale_omissions": len(state.get("stale_actor_omissions", [])),
        }

        ego_light = nearest_traffic_light(state.get("traffic_lights", {}), *ego_xy)
        row["ego_nearest_tl"] = f"{ego_light[0]}@{ego_light[1]:.1f}m={ego_light[2]}" if ego_light else None

        if offline_map is not None:
            row["ego_road_id"], row["ego_lane_id"], _junction = offline_map.road_lane(*ego_xy, ego_tf["z"])
            row["ego_route_s"] = offline_map.route_s(fnum(row["ego_route_index"]))

        if hit is not None:
            logical_id = hit["key"]
            record = actor_states[logical_id]
            tf = record["transform"]
            meta = actors_meta.get(logical_id, {})
            label = labels.get(logical_id, {})
            light_state = label.get("light_state") or {}

            row.update({
                "blocker_logical_id": logical_id,
                "blocker_actor_id": meta.get("canonical_actor_id", label.get("actor_id")),
                "blocker_type": meta.get("blueprint_id", label.get("type_id")),
                "blocker_color": (meta.get("attributes") or {}).get("color"),
                "blocker_longitudinal_m": hit["longitudinal_m"],
                "blocker_lateral_m": hit["lateral_m"],
                "blocker_x": tf["x"],
                "blocker_y": tf["y"],
                "blocker_yaw": tf["yaw"],
                "blocker_speed_kmh": speed_of(record["velocity"]) * 3.6,
                "blocker_light_state_raw": record.get("light_state"),
                "blocker_brake_light": light_state.get("brake") if isinstance(light_state, dict) else None,
                "blocker_in_labels": bool(label),
                "blocker_first_frame": meta.get("first_frame"),
                "blocker_last_frame": meta.get("last_frame"),
                "blocker_last_movement_frame": last_movement_frame.get(logical_id),
            })

            blocker_light = nearest_traffic_light(state.get("traffic_lights", {}), tf["x"], tf["y"])
            row["blocker_nearest_tl"] = (
                f"{blocker_light[0]}@{blocker_light[1]:.1f}m={blocker_light[2]}" if blocker_light else None
            )

            # What is in front of the blocker (queue / deadlock evidence).
            next_hit = find_nearest_forward_actor(
                (tf["x"], tf["y"]), tf["yaw"],
                [c for c in candidates if c[0] != logical_id],
                max_lateral_m=corridor,
            )
            if next_hit is not None:
                next_record = actor_states[next_hit["key"]]
                row["blocker_front_logical_id"] = next_hit["key"]
                row["blocker_front_distance_m"] = next_hit["longitudinal_m"]
                row["blocker_front_speed_kmh"] = speed_of(next_record["velocity"]) * 3.6
                row["blocker_front_type"] = actors_meta.get(next_hit["key"], {}).get("blueprint_id")

            if offline_map is not None:
                road, lane, junction = offline_map.road_lane(tf["x"], tf["y"], tf["z"])
                row["blocker_road_id"], row["blocker_lane_id"], row["blocker_is_junction"] = road, lane, junction

        frames.append(row)

    return frames, actors_meta


def summarize(frames, actors_meta, start_frame, end_frame):
    ego_speeds = [f["ego_speed_kmh"] for f in frames]
    first, last = frames[0], frames[-1]
    ego_displacement = math.hypot(last["ego_x"] - first["ego_x"], last["ego_y"] - first["ego_y"])

    stationary = [f for f in frames if f["ego_speed_kmh"] < 1.0]

    # Longest continuous stationary run.
    longest, current, longest_range, current_start = 0, 0, None, None
    for f in frames:
        if f["ego_speed_kmh"] < 1.0:
            if current == 0:
                current_start = f["frame_id"]
            current += 1
            if current > longest:
                longest, longest_range = current, (current_start, f["frame_id"])
        else:
            current = 0

    blocker_counts = Counter(f["blocker_logical_id"] for f in frames if f["blocker_logical_id"] is not None)
    persistent = None

    if blocker_counts:
        logical_id, visible_frames = blocker_counts.most_common(1)[0]
        rows = [f for f in frames if f["blocker_logical_id"] == logical_id]
        meta = actors_meta.get(logical_id, {})
        displacement = math.hypot(rows[-1]["blocker_x"] - rows[0]["blocker_x"], rows[-1]["blocker_y"] - rows[0]["blocker_y"])
        persistent = {
            "logical_id": logical_id,
            "actor_id": meta.get("canonical_actor_id", rows[0].get("blocker_actor_id")),
            "type": meta.get("blueprint_id", rows[0].get("blocker_type")),
            "color": (meta.get("attributes") or {}).get("color"),
            "mean_longitudinal_m": sum(r["blocker_longitudinal_m"] for r in rows) / len(rows),
            "min_longitudinal_m": min(r["blocker_longitudinal_m"] for r in rows),
            "max_longitudinal_m": max(r["blocker_longitudinal_m"] for r in rows),
            "displacement_m": displacement,
            "mean_speed_kmh": sum(r["blocker_speed_kmh"] for r in rows) / len(rows),
            "max_speed_kmh": max(r["blocker_speed_kmh"] for r in rows),
            "road_lane": sorted({(r.get("blocker_road_id"), r.get("blocker_lane_id")) for r in rows}, key=str),
            "is_junction": sorted({r.get("blocker_is_junction") for r in rows}, key=str),
            "visible_frames": visible_frames,
            "window_frames": len(frames),
            "stationary_frames": sum(1 for r in rows if r["blocker_speed_kmh"] < 1.0),
            "brake_light_on_frames": sum(1 for r in rows if r.get("blocker_brake_light")),
            "first_appearance_frame": meta.get("first_frame"),
            "last_recorded_frame": meta.get("last_frame"),
            "last_movement_frame": rows[-1].get("blocker_last_movement_frame"),
            "nearest_tl_states": dict(Counter(r.get("blocker_nearest_tl") for r in rows).most_common(5)),
            "front_of_blocker": dict(
                Counter(
                    f"{r.get('blocker_front_logical_id')}:{r.get('blocker_front_type')}"
                    for r in rows
                ).most_common(3)
            ),
            "front_of_blocker_mean_speed_kmh": _mean(r.get("blocker_front_speed_kmh") for r in rows),
            "front_of_blocker_mean_gap_m": _mean(r.get("blocker_front_distance_m") for r in rows),
        }

    return {
        "frame_range": [start_frame, end_frame],
        "frames_analyzed": len(frames),
        "ego": {
            "start_xy": [first["ego_x"], first["ego_y"]],
            "end_xy": [last["ego_x"], last["ego_y"]],
            "displacement_m": ego_displacement,
            "mean_speed_kmh": sum(ego_speeds) / len(ego_speeds),
            "max_speed_kmh": max(ego_speeds),
            "stationary_frames": len(stationary),
            "longest_stationary_run_frames": longest,
            "longest_stationary_range": longest_range,
            "stationary_duration_s": longest * 0.1,
            "route_index_range": [first.get("ego_route_index"), last.get("ego_route_index")],
            "route_s_range": [first.get("ego_route_s"), last.get("ego_route_s")],
            "road_lane": sorted({(f.get("ego_road_id"), f.get("ego_lane_id")) for f in frames}, key=str),
            "stationary_mean_throttle": _mean(f["ego_throttle"] for f in stationary),
            "stationary_mean_brake": _mean(f["ego_brake"] for f in stationary),
            "stationary_frames_brake_gt_0_1": sum(1 for f in stationary if (f["ego_brake"] or 0) > 0.1),
            "stationary_frames_throttle_gt_0_2": sum(1 for f in stationary if (f["ego_throttle"] or 0) > 0.2),
            "nearest_tl_states": dict(Counter(f.get("ego_nearest_tl") for f in stationary).most_common(5)),
        },
        "persistent_forward_actor": persistent,
        "forward_blocker_ids_seen": dict(blocker_counts.most_common(10)),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value, spec=".2f"):
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def print_report(summary, frames, step):
    ego = summary["ego"]
    blocker = summary["persistent_forward_actor"]

    print("[STALL OFFLINE ANALYSIS]")
    print()
    print(f"Frame range: {summary['frame_range'][0]}-{summary['frame_range'][1]} ({summary['frames_analyzed']} frames)")
    print()
    print("Ego:")
    print(f"  displacement        = {_fmt(ego['displacement_m'])} m")
    print(f"  mean speed          = {_fmt(ego['mean_speed_kmh'])} km/h (max {_fmt(ego['max_speed_kmh'])})")
    print(
        f"  stationary duration = {_fmt(ego['stationary_duration_s'], '.1f')} s "
        f"(longest run {ego['longest_stationary_run_frames']} frames {ego['longest_stationary_range']}; "
        f"{ego['stationary_frames']} stationary frames total)"
    )
    print(f"  route index         = {ego['route_index_range']}  route_s = {ego['route_s_range']}")
    print(f"  road/lane           = {ego['road_lane']}")
    print(
        f"  control while stopped: mean throttle={_fmt(ego['stationary_mean_throttle'], '.3f')} "
        f"mean brake={_fmt(ego['stationary_mean_brake'], '.3f')} "
        f"frames brake>0.1={ego['stationary_frames_brake_gt_0_1']} "
        f"throttle>0.2={ego['stationary_frames_throttle_gt_0_2']}"
    )
    print(f"  nearest TL while stopped = {ego['nearest_tl_states']}")
    print()

    if blocker is None:
        print(f"Nearest persistent forward actor: NONE in corridor (+-{FORWARD_CORRIDOR_HALF_WIDTH_M} m, {FORWARD_SEARCH_MAX_M} m)")
    else:
        print("Nearest persistent forward actor:")
        print(f"  logical_id    = {blocker['logical_id']}")
        print(f"  actor_id      = {blocker['actor_id']}")
        print(f"  type          = {blocker['type']} (color {blocker['color']})")
        print(
            f"  mean distance = {_fmt(blocker['mean_longitudinal_m'])} m "
            f"(min {_fmt(blocker['min_longitudinal_m'])}, max {_fmt(blocker['max_longitudinal_m'])})"
        )
        print(f"  displacement  = {_fmt(blocker['displacement_m'])} m")
        print(f"  mean speed    = {_fmt(blocker['mean_speed_kmh'])} km/h (max {_fmt(blocker['max_speed_kmh'])})")
        print(f"  road/lane     = {blocker['road_lane']} junction={blocker['is_junction']}")
        print(f"  brake light on frames = {blocker['brake_light_on_frames']}")
        print(f"  nearest TL    = {blocker['nearest_tl_states']}")
        print(
            f"  in front of blocker   = {blocker['front_of_blocker']} "
            f"mean gap={_fmt(blocker['front_of_blocker_mean_gap_m'])} m "
            f"mean speed={_fmt(blocker['front_of_blocker_mean_speed_kmh'])} km/h"
        )
        print()
        print("Persistence:")
        print(f"  visible frames    = {blocker['visible_frames']} / {blocker['window_frames']}")
        print(f"  stationary frames = {blocker['stationary_frames']}")
        print(f"  first appearance  = {blocker['first_appearance_frame']}")
        print(f"  last movement     = {blocker['last_movement_frame']}")
        print(f"  last recorded     = {blocker['last_recorded_frame']}")

    print()
    print(f"Forward blocker ids seen (frames): {summary['forward_blocker_ids_seen']}")
    print()
    print(
        "frame  ego_v  thr   brk   route  ego_s   road/lane   blk_id  type                          "
        "dist   blk_v  brkL  blk_road/lane  front(id,gap,v)"
    )

    for f in frames:
        if f["frame_id"] % step != 0:
            continue

        front = (
            f"{f.get('blocker_front_logical_id')},{_fmt(f.get('blocker_front_distance_m'), '.1f')},"
            f"{_fmt(f.get('blocker_front_speed_kmh'), '.1f')}"
            if f.get("blocker_front_logical_id") is not None else "-"
        )
        print(
            f"{f['frame_id']:05d} {f['ego_speed_kmh']:6.2f} {_fmt(f['ego_throttle'], '.2f'):>5} "
            f"{_fmt(f['ego_brake'], '.2f'):>5} {str(f['ego_route_index']):>5} {_fmt(f['ego_route_s'], '.1f'):>6} "
            f"{str(f['ego_road_id']) + '/' + str(f['ego_lane_id']):>11} "
            f"{str(f['blocker_logical_id']):>6}  {str(f.get('blocker_type', '-'))[:28]:28} "
            f"{_fmt(f.get('blocker_longitudinal_m'), '.1f'):>5} {_fmt(f.get('blocker_speed_kmh'), '.2f'):>6} "
            f"{str(f.get('blocker_brake_light', '-')):>5} "
            f"{str(f.get('blocker_road_id', '-')) + '/' + str(f.get('blocker_lane_id', '-')):>13}  {front}"
        )


def write_outputs(out_dir, frames, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []

    for f in frames:
        for key in f:
            if key not in keys:
                keys.append(key)

    with open(out_dir / "frames.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(frames)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline stall analysis of a canonical geometry sequence.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--town", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--summary-step", type=int, default=10, help="Per-frame table stride (10 = 1 s at 10 Hz).")
    parser.add_argument("--corridor-half-width", type=float, default=FORWARD_CORRIDOR_HALF_WIDTH_M)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--carla-root", default=None, help="Default: $CARLA_ROOT or C:\\CARLA")
    parser.add_argument("--xodr", default=None, help="OpenDRIVE file; auto-detected under CARLA_ROOT if omitted.")
    parser.add_argument("--no-map", action="store_true", help="Skip road/lane/route_s (no carla import).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    geometry = resolve_geometry(args.dataset_root, args.town, args.route)

    offline_map = None

    if not args.no_map:
        try:
            map_name = load_json(geometry / "sequence.json").get("map", "") if (geometry / "sequence.json").exists() else ""
            xodr = Path(args.xodr) if args.xodr else find_xodr(map_name or f"{args.town}HD", args.carla_root)
            route_xml = PROJECT_ROOT / "routes" / f"{args.town}.xml"
            if xodr is None:
                raise FileNotFoundError(f"OpenDRIVE for map '{map_name}' not found")
            offline_map = OfflineMap(xodr, route_xml if route_xml.exists() else None, args.route, args.carla_root)
            print(f"[Map] road/lane + route_s from {xodr}")
        except Exception as exc:  # optional enrichment only
            print(f"[Map] skipped road/lane/route_s: {exc}")

    frames, actors_meta = analyze(
        geometry, args.start_frame, args.end_frame, offline_map, corridor=args.corridor_half_width,
    )
    summary = summarize(frames, actors_meta, args.start_frame, args.end_frame)
    summary["geometry"] = str(geometry)

    print_report(summary, frames, args.summary_step)

    out_dir = Path(args.out_dir) if args.out_dir else geometry / f"stall_analysis_{args.start_frame}_{args.end_frame}"
    write_outputs(out_dir, frames, summary)
    print()
    print(f"[Saved] {out_dir / 'frames.csv'}")
    print(f"[Saved] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

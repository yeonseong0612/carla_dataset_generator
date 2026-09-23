"""
collect_dataset.py

Paired multi-condition CARLA dataset collection.

Core invariant (see CLAUDE.md): for a fixed (town, route, frame_id) the
scene geometry is identical across every weather condition; only the
appearance (RGB) changes.

Pipeline
--------
Town
  └─ Route
      ├─ Stage 1: canonical geometry generation (once per route, day_clear)
      │     ├─ Spawn ego / traffic / pedestrians, spawn sensors
      │     ├─ Run RouteController + canonical background traffic
      │     ├─ Record exact world state every frame   -> geometry/world_state
      │     ├─ Save shared geometry / GT              -> geometry/
      │     └─ Save the day_clear RGB from the same run -> conditions/day_clear
      │
      └─ Stage 2: deterministic weather replay (every OTHER condition)
            ├─ Read geometry/world_state (source of truth)
            ├─ Recreate the recorded actors, set every transform per frame
            │   (no BasicAgent / RouteController / Traffic Manager / Gamma)
            ├─ Apply weather, tick, save stereo RGB ONLY -> conditions/<weather>
            └─ Verify post-tick transforms == recorded transforms

What is stored where
--------------------
Canonical geometry and ALL GT sensors are collected ONCE (canonical run,
day_clear): RGB left/right, depth, semantic, optical flow, LiDAR, 3 radars,
labels, pose, calibration, world_state. Weather replay regenerates ONLY the
stereo RGB pair under the recorded geometry -- it does not replay, recreate
or save depth / semantic / optical flow / LiDAR / radar / labels / pose /
calibration (the replay sensor rig contains rgb_left + rgb_right only, see
src/sensors/sensor_rig.py REPLAY_SENSOR_PROFILE). day_clear is never
replayed: its RGB comes from the canonical run itself.

Wind is intentionally fixed to zero in every condition
(src/simulation/weather.py): paired conditions vary weather appearance while
preserving geometry as much as possible.

Output
------
dataset/{town}/route_{id}/
    geometry/                shared GT, stored once (+ COMPLETE)
    conditions/{weather}/    rgb_left/ rgb_right/ condition.json COMPLETE
    paired_validation.json   cross-condition correspondence report

Resume: geometry/ (canonical GT + day_clear RGB) and each
conditions/{weather}/ (that weather's RGB) carry their own COMPLETE marker; a
re-run replays only the missing weathers (src/data/resume_plan.py).

The script assumes:
- src.navigation.route / controller
- src.simulation.weather / replay / canonical_traffic
- src.sensors.sensor_rig
- src.data.collector / calibration / metadata / annotation / world_state
are already implemented.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import carla


# ============================================================
# Python path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Overridable via the CARLA_ROOT environment variable so the same source
# tree runs unmodified on a remote server whose CARLA install path differs
# from this machine's (kept as the fallback default for local continuity).
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", r"C:\CARLA"))
CARLA_PYTHONAPI = CARLA_ROOT / "PythonAPI" / "carla"

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

sys.path.insert(
    0,
    str(CARLA_PYTHONAPI),
)


# ============================================================
# Project imports
# ============================================================

from CFG.config import cfg

from src.navigation.route import (
    list_route_ids,
    load_route_from_xml,
    project_control_points,
    build_dense_route,
)

from src.navigation.controller import RouteController

from src.simulation.vehicle import destroy_vehicle

from src.simulation.environment import disable_static_traffic_objects

from src.simulation.traffic import (
    configure_traffic_manager,
    destroy_traffic_vehicles,
)

from src.simulation.pedestrian import (
    destroy_pedestrians,
)

from src.simulation.spawn_policy import (
    DynamicSpawnManager,
    VEHICLE_LIKE_CATEGORIES,
    spawn_actors_gamma_policy,
)
from src.simulation.canonical_traffic import (
    CanonicalBackgroundTraffic,
    get_annotation_candidate_count,
    scale_initial_category_counts,
)

from src.sensors.sensor_rig import (
    CANONICAL_SENSOR_PROFILE,
    REPLAY_SENSOR_PROFILE,
    SensorRig,
)
from src.data.collector import Collector
from src.data.calibration import build_calibration, save_calibration
from src.data.metadata import MetadataWriter
from src.data.annotation import AnnotationWriter
from src.data.layout import (
    GEOMETRY_SENSOR_EXTENSIONS,
    clear_complete,
    condition_dir,
    geometry_dir,
    is_complete,
    mark_complete,
    route_root,
)
from src.data.paired_validation import (
    calibration_tolerance,
    compare_calibration,
    validate_route,
)
from src.data.resume_plan import plan_route_work
from src.data.world_state import (
    TrafficLightRecorder,
    WorldStateReader,
    WorldStateRecorder,
)
from src.simulation.replay import (
    DEFAULT_POSITION_TOLERANCE_M,
    DEFAULT_ROTATION_TOLERANCE_DEG,
    WorldStateReplayer,
    weather_to_dict,
)
from src.simulation.timing import is_record_tick, record_stride_ticks
from src.simulation.weather import WEATHER_WIND_INTENSITY, apply_weather


# ============================================================
# Default collection configuration
# ============================================================

ROUTE_SAMPLING_RESOLUTION = 2.0

CRUISE_SPEED_KMH = 30.0
MIN_CURVE_SPEED_KMH = 12.0

TRAFFIC_LIGHT_POLICY = "obey"

WARMUP_FRAMES = 20

# Safety guard.
# Normal collection should terminate by route completion.
MAX_FRAMES_PER_SEQUENCE = 20000

COLLECTOR_TIMEOUT = 10.0

FLUSH_INTERVAL = 100

EGO_BLUEPRINT = "vehicle.tesla.model3"


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Collect paired multi-condition CARLA dataset: one canonical "
            "geometry sequence per (town, route), replayed under every "
            "weather condition."
        )
    )

    parser.add_argument(
        "--towns",
        nargs="+",
        default=None,
        help=(
            "Town names. "
            "Example: --towns Town01 Town02"
        ),
    )

    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help=(
            "Route IDs. "
            "If omitted, all routes in the XML are used."
        ),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "Weather conditions to render. "
            "If omitted, cfg.WEATHER.CONDITIONS is used. The canonical "
            "condition (cfg.WEATHER.DEFAULT, day_clear) is always rendered "
            "as part of geometry generation."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=os.path.join(
            cfg.PROJECT.ROOT,
            "dataset",
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES_PER_SEQUENCE,
        help=(
            "Maximum number of SAVED dataset samples (10 Hz), not "
            "simulation ticks. The world still ticks/controls/drives "
            "traffic at 20 Hz (cfg.SIMULATION.FPS); "
            "cfg.RECORDING.STRIDE_TICKS simulation ticks elapse per saved "
            "sample. E.g. --max-frames 100 bounds the run to ~10 "
            "simulation seconds (100 samples x 0.1 s), which used to be "
            "expressed as --max-frames 200 at the old 20 Hz recording "
            "rate -- divide any old --max-frames value by "
            "cfg.RECORDING.STRIDE_TICKS to get the equivalent under this "
            "flag."
        ),
    )

    parser.add_argument(
        "--truncate-ok",
        action="store_true",
        help=(
            "Treat --max-frames as an intended cap (smoke tests): reaching "
            "it finalizes a truncated canonical sequence instead of failing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete the selected routes' existing output (geometry AND every "
            "condition) and regenerate the whole route from scratch. To "
            "re-render only some weather conditions without touching "
            "geometry use --rerender-conditions."
        ),
    )

    parser.add_argument(
        "--rerender-conditions",
        nargs="+",
        default=None,
        help=(
            "Re-render these weather conditions' RGB by replaying the "
            "existing canonical geometry (geometry is never deleted or "
            "regenerated). The canonical source condition (day_clear) cannot "
            "be re-rendered this way; use --overwrite."
        ),
    )

    parser.add_argument(
        "--replay-position-tolerance",
        type=float,
        default=DEFAULT_POSITION_TOLERANCE_M,
        help="Max post-tick position error (m) for replay verification.",
    )

    parser.add_argument(
        "--replay-rotation-tolerance",
        type=float,
        default=DEFAULT_ROTATION_TOLERANCE_DEG,
        help="Max post-tick rotation error (deg) for replay verification.",
    )

    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Disable background traffic (canonical run).",
    )

    parser.add_argument(
        "--no-pedestrians",
        action="store_true",
        help="Disable pedestrians (canonical run).",
    )

    parser.add_argument(
        "--background-policy",
        type=str,
        default="canonical",
        choices=["canonical", "dynamic"],
        help=(
            "Background traffic maintenance policy of the canonical run. "
            "'canonical' (default): actors are only ever spawned in a buffer "
            "zone ahead of the visible sensor ROI, never inside it (see "
            "src/simulation/canonical_traffic.py). 'dynamic': the earlier "
            "Phase 2/2.5 per-bin Gamma-deficit replenishment "
            "(src/simulation/spawn_policy.py DynamicSpawnManager), kept "
            "for reference/comparison, not the production default. Replay "
            "never runs either policy."
        ),
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def route_xml_path(town):
    return os.path.join(
        cfg.PROJECT.ROOT,
        "routes",
        f"{town}.xml",
    )


def resolve_carla_map_name(xml_path):
    """
    The route XML's declared town (the actual client.load_world()
    identifier, e.g. "Town10HD") can differ from the --towns key /
    routes/<town>.xml filename stem (e.g. "Town10"). Read it from the
    XML itself rather than guessing a naming suffix.
    """

    route_ids = list_route_ids(xml_path)

    if not route_ids:
        raise ValueError(f"No routes found in {xml_path}")

    town_names = {
        load_route_from_xml(xml_path, route_id)[0]
        for route_id in route_ids
    }

    if len(town_names) != 1:
        raise ValueError(
            f"Route XML {xml_path} declares multiple town values: "
            f"{sorted(town_names)}"
        )

    return next(iter(town_names))



def get_sensor_actors(rig):

    if hasattr(
        rig,
        "sensors",
    ):
        return rig.sensors

    if hasattr(
        rig,
        "actors",
    ):
        return rig.actors

    raise AttributeError(
        "SensorRig must expose sensor actors "
        "through rig.sensors or rig.actors."
    )


# ============================================================
# Ego
# ============================================================

def spawn_ego_at_route_start(
    world,
    dense_route,
):
    if not dense_route:
        raise RuntimeError(
            "Dense route is empty."
        )

    blueprint = (
        world
        .get_blueprint_library()
        .find(EGO_BLUEPRINT)
    )

    if blueprint.has_attribute(
        "role_name"
    ):
        blueprint.set_attribute(
            "role_name",
            "hero",
        )

    waypoint = dense_route[0][0]

    transform = carla.Transform(
        waypoint.transform.location,
        waypoint.transform.rotation,
    )

    transform.location.z += 0.3

    ego = world.try_spawn_actor(
        blueprint,
        transform,
    )

    if ego is None:
        raise RuntimeError(
            "Failed to spawn ego vehicle "
            "at route start."
        )

    return ego


# ============================================================
# Validation
# ============================================================

def count_files(directory, extension):

    if not os.path.isdir(
        directory
    ):
        return 0

    return len(
        [
            name
            for name in os.listdir(directory)
            if name.endswith(extension)
        ]
    )


def count_csv_rows(path):

    if not os.path.isfile(
        path
    ):
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.reader(file)
        )

    return max(
        len(rows) - 1,
        0,
    )


def validate_geometry(
    geometry_root,
    condition_root,
    expected_frames,
):
    """
    Structural validation of a freshly generated canonical run: shared
    geometry files under geometry_root plus the source condition's RGB.
    """
    errors = []

    # --------------------------------------------------------
    # Sequence-level files
    # --------------------------------------------------------

    required_files = [
        "calibration.json",
        "sequence.json",
        "actors.json",
        "timestamps.csv",
        "ego_state.csv",
        os.path.join("pose", "poses.csv"),
    ]

    for name in required_files:

        path = os.path.join(
            geometry_root,
            name,
        )

        if not os.path.isfile(
            path
        ):
            errors.append(
                f"Missing: {path}"
            )

    # --------------------------------------------------------
    # Per-frame file counts
    # --------------------------------------------------------

    file_checks = {
        os.path.join(geometry_root, "world_state"): ".json",
        os.path.join(geometry_root, "labels", "object_3d"): ".json",
        os.path.join(condition_root, "rgb_left"): ".png",
        os.path.join(condition_root, "rgb_right"): ".png",
    }

    for name, extension in GEOMETRY_SENSOR_EXTENSIONS.items():
        file_checks[os.path.join(geometry_root, name)] = extension

    for directory, extension in (
        file_checks.items()
    ):

        count = count_files(
            directory,
            extension,
        )

        if count != expected_frames:

            errors.append(
                f"{directory}: "
                f"expected {expected_frames}, "
                f"found {count}"
            )

    # --------------------------------------------------------
    # CSV counts
    # --------------------------------------------------------

    for name, path in {
        "timestamps": os.path.join(geometry_root, "timestamps.csv"),
        "ego_state": os.path.join(geometry_root, "ego_state.csv"),
        "poses": os.path.join(geometry_root, "pose", "poses.csv"),
    }.items():

        count = count_csv_rows(
            path
        )

        if (
            count is not None
            and count != expected_frames
        ):
            errors.append(
                f"{name}: "
                f"expected {expected_frames}, "
                f"found {count}"
            )

    return errors


# ============================================================
# Route preparation
# ============================================================

def prepare_route(
    world,
    xml_path,
    route_id,
):

    town, control_points = (
        load_route_from_xml(
            xml_path,
            route_id,
        )
    )

    carla_map = world.get_map()

    control_waypoints = (
        project_control_points(
            carla_map,
            control_points,
        )
    )

    dense_route = (
        build_dense_route(
            carla_map,
            control_waypoints,
            sampling_resolution=(
                ROUTE_SAMPLING_RESOLUTION
            ),
        )
    )

    if len(
        dense_route
    ) < 2:
        raise RuntimeError(
            f"Route {route_id} "
            "produced fewer than 2 "
            "dense waypoints."
        )

    return (
        town,
        control_points,
        dense_route,
    )




# ============================================================
# Runtime logging (observability only; no behaviour)
# ============================================================

def log_world_sensors(world, label):
    """Print every sensor actually alive in the CARLA world (id / type / parent)."""

    sensors = sorted(
        world.get_actors().filter("sensor.*"),
        key=lambda actor: actor.id,
    )

    print(
        f"[Sensors:{label}] {len(sensors)} sensor(s) alive in world"
    )

    for sensor in sensors:
        print(
            f"[Sensors:{label}]   id={sensor.id} type={sensor.type_id} "
            f"parent={sensor.parent.id if sensor.parent is not None else None}"
        )


def log_runtime_weather(world, label):
    """Print the weather the running world reports (world.get_weather())."""

    weather = world.get_weather()

    print(
        f"[Weather:{label}] runtime wind_intensity={weather.wind_intensity} "
        f"cloudiness={weather.cloudiness} precipitation={weather.precipitation} "
        f"fog_density={weather.fog_density} sun_altitude={weather.sun_altitude_angle}"
    )


def configure_traffic_lights(world, cfg):
    """
    Apply cfg.TRAFFIC_LIGHT.{GREEN,YELLOW,RED}_TIME_S to every traffic
    light in the just-loaded town, once. Only each state's duration is
    changed via CARLA's own set_green_time/set_yellow_time/set_red_time --
    group membership, state, and CARLA's own transition scheduling are
    untouched. A light that raises while being configured is counted and
    reported, not silently skipped.
    """

    lights = world.get_actors().filter("traffic.traffic_light")

    configured = 0
    failures = []

    for light in lights:
        try:
            light.set_green_time(cfg.TRAFFIC_LIGHT.GREEN_TIME_S)
            light.set_yellow_time(cfg.TRAFFIC_LIGHT.YELLOW_TIME_S)
            light.set_red_time(cfg.TRAFFIC_LIGHT.RED_TIME_S)
            configured += 1
        except RuntimeError as exc:
            failures.append((light.id, str(exc)))

    print(
        f"[TrafficLight] configured {configured} lights: "
        f"green={cfg.TRAFFIC_LIGHT.GREEN_TIME_S}s "
        f"yellow={cfg.TRAFFIC_LIGHT.YELLOW_TIME_S}s "
        f"red={cfg.TRAFFIC_LIGHT.RED_TIME_S}s"
    )

    if failures:
        print(f"[TrafficLight] failed to configure {len(failures)} lights:")

        for light_id, reason in failures:
            print(f"[TrafficLight]   id={light_id} reason={reason}")

    return {"configured": configured, "failed": len(failures)}


STATIONARY_SPEED_THRESHOLD_MPS = 0.5
STATIONARY_DIAGNOSTIC_FRAMES = (0, 1, 5, 10)


def log_stationary_vehicle_diagnostic(local_frame_id, managed_actors):
    """
    Diagnostic only (CLAUDE.md PART F): at a handful of early recorded
    frames, report how many managed vehicle-like actors are still below
    STATIONARY_SPEED_THRESHOLD_MPS, split into "stopped at a red light"
    (expected/normal) vs. "other" (candidate not-yet-initialized traffic).
    Never destroys or otherwise touches any actor.
    """

    if local_frame_id not in STATIONARY_DIAGNOSTIC_FRAMES:
        return

    total = 0
    moving = 0
    stationary_red_light = 0
    stationary_other = 0

    for managed in managed_actors:
        if managed["category"] not in VEHICLE_LIKE_CATEGORIES:
            continue

        actor = managed["actor"]

        if actor is None or not actor.is_alive:
            continue

        total += 1

        velocity = actor.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

        if speed >= STATIONARY_SPEED_THRESHOLD_MPS:
            moving += 1
            continue

        is_red_light = False

        try:
            if actor.is_at_traffic_light():
                is_red_light = actor.get_traffic_light_state() == carla.TrafficLightState.Red
        except RuntimeError:
            is_red_light = False

        if is_red_light:
            stationary_red_light += 1
        else:
            stationary_other += 1

    stationary = stationary_red_light + stationary_other
    stationary_ratio = (stationary / total) if total else 0.0

    print(
        f"[StationaryDiagnostic] frame={local_frame_id} total={total} "
        f"moving={moving} stationary={stationary} "
        f"(red_light={stationary_red_light} other={stationary_other}) "
        f"stationary_ratio={stationary_ratio:.3f}"
    )


# ============================================================
# Condition / route metadata helpers
# ============================================================

GEOMETRY_REPLAY_VERSION = 1


def write_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def write_condition_json(
    cond_dir,
    condition,
    source_condition,
    weather,
    num_frames,
    rendered_from,
    replay_validation=None,
    calibration_check=None,
):
    """
    Per-condition metadata: weather parameters (incl. the fixed wind), the
    geometry it was rendered from, frame count and replay validation result.

    rendered_from == "canonical_run": the source condition's RGB, produced by
        the canonical run itself (not a replay).
    rendered_from == "replay": RGB-only replay of the recorded geometry; no
        non-RGB sensor was spawned, replayed or saved.
    """

    is_replay = rendered_from == "replay"

    write_json(
        os.path.join(cond_dir, "condition.json"),
        {
            "condition": condition,
            "weather_parameters": weather_to_dict(weather),
            # Wind is intentionally fixed to zero for every condition so
            # paired conditions vary appearance while preserving geometry.
            "wind_intensity": float(weather.wind_intensity),
            "source_geometry": os.path.join("..", "..", "geometry").replace("\\", "/"),
            "geometry_source_condition": source_condition,
            "geometry_replay_version": GEOMETRY_REPLAY_VERSION,
            "rendered_from": rendered_from,
            "rendered_from_canonical_geometry": True,
            "rgb_only_replay": is_replay,
            "sensors_saved": list(REPLAY_SENSOR_PROFILE),
            "num_frames": num_frames,
            "recording_hz": float(cfg.RECORDING.FPS),
            "replay_validation": replay_validation,
            "calibration_check": calibration_check,
        },
    )


def update_sequence_conditions(geometry_root, conditions):
    """Keep geometry/sequence.json's "conditions" equal to what is complete."""

    path = os.path.join(geometry_root, "sequence.json")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    data["conditions"] = list(conditions)

    write_json(path, data)


# ============================================================
# Stage 1: canonical geometry generation
# ============================================================

def generate_canonical_geometry(
    world,
    client,
    traffic_manager,
    town,
    route_id,
    dense_route,
    source_condition,
    planned_conditions,
    route_path,
    max_frames,
    truncate_ok=False,
    no_traffic=False,
    no_pedestrians=False,
    background_policy="canonical",
):
    """
    Run the production driving + traffic simulation ONCE for this
    (town, route) under source_condition, recording:

        geometry/   calibration, labels, depth/flow/semantic/lidar/radar,
                    ego pose/state, actors.json + world_state/ (replay source)
        conditions/<source_condition>/   RGB from this same run
    """

    geometry_root = geometry_dir(route_path)
    source_dir = condition_dir(route_path, source_condition)

    os.makedirs(
        geometry_root,
        exist_ok=True,
    )

    os.makedirs(
        source_dir,
        exist_ok=True,
    )

    ego = None
    rig = None

    collector = None
    metadata = None
    world_state_recorder = None

    traffic_actors = {}
    walkers = []
    walker_controllers = []

    spawn_manager = None

    frame_object_csv_file = None
    frame_object_csv_writer = None

    saved_frames = 0
    truncated = False

    sequence_start = time.time()

    try:

        print()
        print(
            "========================================"
        )
        print(
            "Stage 1   : canonical geometry"
        )
        print(
            f"Town      : {town}"
        )
        print(
            f"Route     : {route_id}"
        )
        print(
            f"Condition : {source_condition}"
        )
        print(
            f"Output    : {route_path}"
        )
        print(
            "========================================"
        )

        # ====================================================
        # Weather
        # ====================================================

        weather = apply_weather(
            world,
            source_condition,
        )

        # ====================================================
        # Ego
        # ====================================================

        ego = spawn_ego_at_route_start(
            world,
            dense_route,
        )

        # Allow spawn state to propagate.
        world.tick()

        # ====================================================
        # Controller
        # ====================================================

        route_controller = (
            RouteController(
                vehicle=ego,
                dense_route=dense_route,
                cruise_speed=(
                    CRUISE_SPEED_KMH
                ),
                min_curve_speed=(
                    MIN_CURVE_SPEED_KMH
                ),
                traffic_light_policy=(
                    TRAFFIC_LIGHT_POLICY
                ),
            )
        )

        # ====================================================
        # Gamma-policy initial actor spawn
        # (route-relative, initial spawn only -- see
        # src/simulation/spawn_policy.py)
        # ====================================================

        if background_policy == "canonical":
            # CLAUDE.md task section 12: the initial scene should be
            # sized around the first frame-object Gamma segment's
            # target, not the old fixed cfg.SPAWN.N_*-sum population.
            # Composition ratio preserved; safe-route spawn + spacing
            # checks below can still legitimately place fewer.
            initial_counts = scale_initial_category_counts(
                cfg, no_traffic=no_traffic, no_pedestrians=no_pedestrians,
            )
            print(
                f"[FrameObjectGamma] initial scene target="
                f"{initial_counts['initial_frame_object_target']} -> "
                f"vehicles={initial_counts['n_vehicles']} "
                f"motorcycles={initial_counts['n_motorcycles']} "
                f"bicycles={initial_counts['n_bicycles']} "
                f"pedestrians={initial_counts['n_pedestrians']}"
            )
            spawn_result = spawn_actors_gamma_policy(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                n_vehicles=initial_counts["n_vehicles"],
                n_motorcycles=initial_counts["n_motorcycles"],
                n_bicycles=initial_counts["n_bicycles"],
                n_pedestrians=initial_counts["n_pedestrians"],
            )
        else:
            spawn_result = spawn_actors_gamma_policy(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                n_vehicles=0 if no_traffic else None,
                n_motorcycles=0 if no_traffic else None,
                n_bicycles=0 if no_traffic else None,
                n_pedestrians=0 if no_pedestrians else None,
            )

        traffic_actors = spawn_result["traffic_actors"]
        walkers = spawn_result["walkers"]
        walker_controllers = spawn_result["walker_controllers"]
        walker_speeds = spawn_result["walker_speeds"]

        # ====================================================
        # Background traffic maintenance policy
        # (continues the exact same RNG stream the initial spawn
        # used -- see src/simulation/spawn_policy.py /
        # src/simulation/canonical_traffic.py)
        # ====================================================

        category_totals = {
            "vehicle": 0 if no_traffic else cfg.SPAWN.N_VEHICLES,
            "motorcyclist": 0 if no_traffic else cfg.SPAWN.N_MOTORCYCLES,
            "cyclist": 0 if no_traffic else cfg.SPAWN.N_BICYCLES,
            "pedestrian": 0 if no_pedestrians else cfg.SPAWN.N_PEDESTRIANS,
        }

        if background_policy == "dynamic":
            spawn_manager = DynamicSpawnManager(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                spawn_result["rng"],
                category_totals=category_totals,
            )
        else:
            spawn_manager = CanonicalBackgroundTraffic(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                spawn_result["rng"],
                category_totals=category_totals,
            )

        spawn_manager.register_initial_actors(spawn_result)

        if not no_pedestrians and walkers:

            # Walks initial pedestrians along their own sidewalk (not
            # start_pedestrians()'s map-wide random destination) so they
            # stay trackable in the route corridor -- see
            # DynamicSpawnManager.start_initial_pedestrians().
            spawn_manager.start_initial_pedestrians()

        # ====================================================
        # Sensor rig
        # ====================================================

        rig = SensorRig(
            world,
            ego,
            cfg,
        ).spawn(
            profile="canonical"
        )

        print(
            f"[Sensors] "
            f"{len(rig)}"
        )

        # IMPORTANT:
        # Resolve attached sensor transforms
        # before reading calibration.
        world.tick()

        # ====================================================
        # Calibration
        # ====================================================

        sensor_actors = (
            get_sensor_actors(
                rig
            )
        )

        calibration = (
            save_calibration(
                ego_vehicle=ego,
                sensor_actors=(
                    sensor_actors
                ),
                output_path=os.path.join(
                    geometry_root,
                    "calibration.json",
                ),
            )
        )

        stereo = (
            calibration
            .get(
                "stereo",
                {},
            )
        )

        print(
            "[Calibration] "
            f"baseline="
            f"{stereo.get('baseline_m', 0.0):.4f} m"
        )

        # ====================================================
        # Collector: shared geometry -> geometry/, RGB -> source
        # condition directory
        # ====================================================

        collector = Collector(
            rig=rig,
            cfg=cfg,
            geometry_root=geometry_root,
            condition_root=source_dir,
            timeout=(
                COLLECTOR_TIMEOUT
            ),
        )

        # ====================================================
        # Metadata
        # ====================================================

        sequence_extra = {
            "town": town,
            "geometry_source_condition": source_condition,
            "geometry_replay_version": GEOMETRY_REPLAY_VERSION,
            "conditions": list(planned_conditions),
            "truncated": False,
            # Canonical GT is collected once and shared by every weather;
            # replay only re-renders the stereo RGB pair.
            "shared_geometry": True,
            "canonical_source_weather": source_condition,
            "canonical_gt_modalities": list(CANONICAL_SENSOR_PROFILE) + [
                "labels", "pose", "calibration", "world_state",
            ],
            "replay_modalities": list(REPLAY_SENSOR_PROFILE),
            "wind_intensity": WEATHER_WIND_INTENSITY,
            "actors": "actors.json",
            "world_state": "world_state/",
            "canonical_spawn_summary": (
                "canonical_spawn_summary.json"
                if background_policy == "canonical" else None
            ),
        }

        metadata = MetadataWriter(
            sequence_root=geometry_root,
            map_name=(
                world.get_map().name
            ),
            sequence_id=(
                f"{town}_"
                f"route_{route_id}"
            ),
            cfg=cfg,
            route_id=route_id,
            sequence_extra=sequence_extra,
            spawn_info={
                "background_policy": background_policy,
                "initial_frame_object_target": (
                    initial_counts["initial_frame_object_target"]
                    if background_policy == "canonical" else None
                ),
                "initial_spawn_requested": dict(spawn_result["requested"]),
                "initial_spawn_actual": {
                    **{
                        category: len(actors)
                        for category, actors in traffic_actors.items()
                    },
                    "pedestrian": len(walkers),
                },
                # Dynamic population is not a single count; per-run
                # spawn/despawn statistics live in this sibling file.
                "dynamic_population_summary": (
                    "canonical_spawn_summary.json"
                    if background_policy == "canonical" else None
                ),
            },
        )

        # ====================================================
        # World-state recording (the replay source of truth) +
        # annotation (labels reference the persistent logical ids)
        # ====================================================

        world_state_recorder = WorldStateRecorder(
            geometry_root,
            traffic_light_recorder=TrafficLightRecorder(
                world
            ),
        )

        annotation_writer = (
            AnnotationWriter(
                geometry_root,
                cfg,
                ego=ego,
                left_camera_actor=rig.get_sensor("rgb_left"),
                logical_id_resolver=(
                    world_state_recorder.logical_id_for
                ),
            )
        )

        # ====================================================
        # Frame-Level Gamma Object Count Policy: per-frame
        # target/actual log (canonical policy only -- see
        # src/simulation/canonical_traffic.py FrameObjectGammaSchedule)
        # ====================================================

        if background_policy == "canonical":
            frame_object_csv_file = open(
                os.path.join(geometry_root, "frame_object_counts.csv"),
                "w",
                newline="",
                encoding="utf-8",
            )
            frame_object_csv_writer = csv.writer(frame_object_csv_file)
            frame_object_csv_writer.writerow([
                "frame_id", "target_object_count", "actual_object_count",
                "managed_population", "visible_population", "buffer_population",
                "spawned_this_frame", "pruned_this_frame", "naturally_despawned_this_frame",
                "cumulative_spawned", "cumulative_pruned", "cumulative_natural_despawn",
            ])

        # ====================================================
        # Warmup
        # ====================================================
        # cfg.SPAWN.PRE_RECORD_WARMUP_TICKS (production, tunable
        # independent of replay's own WARMUP_FRAMES below): lets
        # just-spawned background traffic leave its at-rest state before
        # frame_id=0 is recorded. Ego is not driven here (no
        # route_controller.run_step()/apply_control() call), so it stays
        # at its spawn transform. No collector/metadata/world_state/
        # annotation call happens in this loop, so no warm-up frame is
        # ever written to disk; rig.clear_queues() below flushes any
        # sensor data these ticks produced.

        for _ in range(
            cfg.SPAWN.PRE_RECORD_WARMUP_TICKS
        ):
            world.tick()

        if hasattr(
            rig,
            "clear_queues",
        ):
            rig.clear_queues()

        log_world_sensors(world, f"canonical:{source_condition}")
        log_runtime_weather(world, f"canonical:{source_condition}")

        # ====================================================
        # Main loop
        # ====================================================

        # visible_population/buffer_population only change at
        # spawn_manager.update()'s own cadence (recomputing them every
        # frame would mean re-running route projection for every managed
        # actor every frame) -- these hold their last-known value between
        # updates, same convention as the target/actual columns already
        # did before this task.
        last_visible_population = 0
        last_buffer_population = 0

        # 20 Hz simulation / 10 Hz recording (CLAUDE.md): every world tick
        # below still runs control + Traffic Manager + Gamma maintenance;
        # dataset writes (collector/metadata/world_state/annotation) only
        # happen on every record_stride-th tick. simulation_tick_idx is the
        # 20 Hz tick counter (0-indexed from this loop's first tick, NOT
        # the CARLA server frame and NOT the dataset frame_id);
        # record_frame_id is the contiguous 10 Hz dataset sample id
        # (== saved_frames at the moment it is assigned). --max-frames now
        # bounds saved dataset SAMPLES, not simulation ticks.

        record_stride = record_stride_ticks(cfg)

        assert cfg.SPAWN.UPDATE_INTERVAL_FRAMES % record_stride == 0, (
            "cfg.SPAWN.UPDATE_INTERVAL_FRAMES must stay a multiple of the "
            "record stride: background-traffic maintenance runs on the "
            "20 Hz simulation clock (never slowed down by recording), and "
            "this invariant is what guarantees a fresh camera-valid "
            "annotation count is always available on the ticks it fires."
        )

        max_simulation_ticks = max_frames * record_stride

        for simulation_tick_idx in range(
            max_simulation_ticks
        ):

            # ------------------------------------------------
            # Vehicle control (every simulation tick -- 20 Hz, never
            # gated on recording)
            # ------------------------------------------------

            control = (
                route_controller
                .run_step()
            )

            ego.apply_control(
                control
            )

            # ------------------------------------------------
            # Simulation
            # ------------------------------------------------

            carla_frame = (
                world.tick()
            )

            snapshot = (
                world.get_snapshot()
            )

            if (
                snapshot.frame
                != carla_frame
            ):
                raise RuntimeError(
                    "Snapshot mismatch: "
                    f"tick={carla_frame}, "
                    f"snapshot="
                    f"{snapshot.frame}"
                )

            timestamp = (
                snapshot
                .timestamp
                .elapsed_seconds
            )

            # ------------------------------------------------
            # Controller / route status (every simulation tick: ego can
            # complete or get stuck between two recorded samples)
            # ------------------------------------------------

            route_status = (
                route_controller
                .get_status()
            )

            record_tick = is_record_tick(
                simulation_tick_idx, record_stride,
            )

            record_frame_id = None
            annotation_counts = None

            if record_tick:

                record_frame_id = saved_frames

                # --------------------------------------------
                # Sensor packet (only the sensors that actually emit on
                # this tick -- cfg.RECORDING.SAMPLE_INTERVAL_SECONDS
                # sensor_tick, see src/sensors/*)
                # --------------------------------------------

                packet = (
                    collector.collect_frame(
                        carla_frame
                    )
                )

                collector.save_frame(
                    record_frame_id,
                    packet,
                )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                metadata.write_frame(
                    frame_id=(
                        record_frame_id
                    ),
                    carla_frame=(
                        carla_frame
                    ),
                    timestamp=(
                        timestamp
                    ),
                    ego=ego,
                    route_status=(
                        route_status
                    ),
                )

                # --------------------------------------------
                # World state (exact scene of THIS recorded sample,
                # recorded before the spawn manager below mutates the
                # actor set and before annotation so labels can
                # reference logical ids)
                # --------------------------------------------

                world_state_recorder.record_frame(
                    record_frame_id,
                    carla_frame,
                    timestamp,
                    snapshot,
                    ego,
                    spawn_manager.managed_actors,
                )

                # --------------------------------------------
                # Annotation
                # --------------------------------------------

                annotation_counts = (
                    annotation_writer
                    .write_frame(
                        record_frame_id,
                        world,
                        ego,
                        depth_raw=packet["depth"],
                    )
                )

                saved_frames += 1

                # --------------------------------------------
                # Diagnostic only (CLAUDE.md PART F): initial stationary
                # traffic ratio at a few early recorded samples.
                # Read-only, never spawns/despawns/destroys anything.
                # --------------------------------------------

                log_stationary_vehicle_diagnostic(
                    record_frame_id, spawn_manager.managed_actors,
                )

            # ------------------------------------------------
            # Phase 2: dynamic density maintenance
            #
            # Runs on simulation_tick_idx's own 20 Hz cadence
            # (cfg.SPAWN.UPDATE_INTERVAL_FRAMES simulation ticks) --
            # NEVER slowed down by the recording stride (CLAUDE.md
            # section 12). The assert above guarantees every tick this
            # fires on is also a record tick, so a fresh camera-valid
            # annotation count is always available here.
            #
            # Runs after this tick's sensors/annotation are already
            # captured and saved (when it is a record tick), so any actor
            # spawned/despawned here only takes effect starting from the
            # *next* world.tick() -- it never disturbs the sample just
            # collected.
            # ------------------------------------------------

            spawned_this_frame = 0
            pruned_this_frame = 0
            naturally_despawned_this_frame = 0

            if (
                spawn_manager is not None
                and simulation_tick_idx % cfg.SPAWN.UPDATE_INTERVAL_FRAMES == 0
            ):
                current_object_count = (
                    get_annotation_candidate_count(annotation_counts)
                    if annotation_counts is not None
                    else None
                )

                if background_policy == "canonical":
                    update_snapshot = spawn_manager.update(
                        simulation_tick_idx,
                        current_object_count=current_object_count,
                    )
                    spawned_this_frame = update_snapshot["spawned"]
                    pruned_this_frame = update_snapshot.get("pruned_this_update", 0)
                    naturally_despawned_this_frame = update_snapshot["despawned"]
                    last_visible_population = update_snapshot["visible_population"]
                    last_buffer_population = update_snapshot["buffer_population"]
                else:
                    spawn_manager.update(
                        simulation_tick_idx
                    )

            # ------------------------------------------------
            # Frame-Level Gamma Object Count Policy: per-record-sample
            # target/actual/population/spawn-prune-despawn log. Only
            # written on record ticks (current_object_count/
            # annotation_counts are only fresh then); this diagnostic
            # file is not a dataset modality, so sampling it at 10 Hz
            # instead of 20 Hz changes nothing CLAUDE.md tracks.
            # ------------------------------------------------

            if record_tick and frame_object_csv_writer is not None:
                frame_object_csv_writer.writerow([
                    record_frame_id,
                    spawn_manager.frame_object_schedule.target_for_frame(simulation_tick_idx),
                    get_annotation_candidate_count(annotation_counts),
                    len(spawn_manager.managed_actors),
                    last_visible_population,
                    last_buffer_population,
                    spawned_this_frame,
                    pruned_this_frame,
                    naturally_despawned_this_frame,
                    spawn_manager.total_spawned,
                    spawn_manager.total_density_pruned,
                    spawn_manager.total_despawned,
                ])

            # ------------------------------------------------
            # Periodic flush
            # ------------------------------------------------

            if (
                record_tick
                and saved_frames
                % FLUSH_INTERVAL
                == 0
            ):
                collector.flush()
                metadata.flush()
                world_state_recorder.flush()

                if frame_object_csv_file is not None:
                    frame_object_csv_file.flush()

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if record_tick and (
                record_frame_id == 0
                or saved_frames % 100 == 0
            ):

                print(
                    f"[{saved_frames:06d}] "
                    f"CARLA={carla_frame} "
                    f"route="
                    f"{route_status.get('route_index')}"
                    f"/"
                    f"{route_status.get('route_length', 1)-1} "
                    f"progress="
                    f"{route_status.get('progress', 0.0):.1f}% "
                    f"speed="
                    f"{route_status.get('speed_kmh', 0.0):.1f} "
                    f"target="
                    f"{route_status.get('target_speed_kmh', 0.0):.1f} "
                    f"curve="
                    f"{route_status.get('curve_angle_deg', 0.0):.1f} "
                    f"red="
                    f"{route_status.get('waiting_red_light', False)} "
                    f"V="
                    f"{annotation_counts.get('vehicle', 0)} "
                    f"P="
                    f"{annotation_counts.get('pedestrian', 0)}"
                )

            # ------------------------------------------------
            # Completed
            # ------------------------------------------------

            if route_status.get(
                "completed",
                False,
            ):

                print(
                    f"[Complete] "
                    f"Route {route_id} "
                    f"at frame "
                    f"{saved_frames}"
                )

                break

            # ------------------------------------------------
            # Stuck
            # ------------------------------------------------

            if route_status.get(
                "stuck",
                False,
            ):

                raise RuntimeError(
                    "Vehicle stuck: "
                    f"route={route_id}, "
                    f"condition={source_condition}, "
                    f"progress="
                    f"{route_status.get('progress', 0.0):.2f}%, "
                    f"index="
                    f"{route_status.get('route_index')}"
                )

        else:

            if not truncate_ok:
                raise RuntimeError(
                    f"Maximum sample count "
                    f"{max_frames} reached "
                    "before route completion."
                )

            truncated = True
            sequence_extra["truncated"] = True

            print(
                f"[Truncated] --truncate-ok: stopping at "
                f"{saved_frames} frames (route not completed)."
            )

        # ====================================================
        # Finalize
        # ====================================================

        collector.flush()

        world_state_recorder.finalize()

        metadata.finalize()
        metadata = None

        collector.close()
        collector = None

        # ====================================================
        # Validation
        # ====================================================

        errors = validate_geometry(
            geometry_root,
            source_dir,
            saved_frames,
        )

        elapsed = (
            time.time()
            - sequence_start
        )

        if errors:

            print(
                "[Validation] FAIL"
            )

            for error in errors:
                print(
                    f"    {error}"
                )

            raise RuntimeError(
                "Canonical geometry validation failed."
            )

        print(
            "[Validation] PASS"
        )

        print(
            f"[Geometry] "
            f"{saved_frames} frames, "
            f"{elapsed:.1f} sec"
        )

        # The source condition's RGB came from this very run, so its
        # frames correspond to geometry by construction.
        write_condition_json(
            source_dir,
            source_condition,
            source_condition,
            weather,
            saved_frames,
            rendered_from="canonical_run",
        )

        mark_complete(
            geometry_root,
            {"num_frames": saved_frames, "truncated": truncated},
        )

        mark_complete(
            source_dir,
            {"num_frames": saved_frames, "rendered_from": "canonical_run"},
        )

        return {
            "status": "completed",
            "frames": saved_frames,
            "elapsed": elapsed,
            "root": route_path,
        }

    # ========================================================
    # Cleanup
    # ========================================================

    finally:

        if metadata is not None:

            try:
                metadata.close()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"metadata: {exc}"
                )

        if world_state_recorder is not None:

            try:
                world_state_recorder.finalize()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"world_state: {exc}"
                )

        if collector is not None:

            try:
                collector.close()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"collector: {exc}"
                )

        if rig is not None:

            try:
                rig.destroy()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"rig: {exc}"
                )

        if frame_object_csv_file is not None:

            try:
                frame_object_csv_file.close()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"frame_object_csv: {exc}"
                )

        # Frame-Level Gamma Object Count Policy: spawn/despawn summary
        # (CLAUDE.md task section 18) -- written here (not the
        # try-block's normal "Finalize" section) because a capped
        # validation run can legitimately raise "Maximum frame count
        # reached before route completion" without ever reaching that
        # section; finally always runs regardless.
        if spawn_manager is not None and hasattr(spawn_manager, "frame_object_schedule"):

            try:
                summary = {
                    "total_spawned": spawn_manager.total_spawned,
                    "total_despawned": spawn_manager.total_despawned,
                    "total_despawned_behind": spawn_manager.total_despawned_behind,
                    "total_despawned_forward_cleanup": spawn_manager.total_despawned_forward_cleanup,
                    "total_projection_failure_despawns": spawn_manager.total_projection_failure_despawns,
                    "total_same_lane_spawn_rejections": spawn_manager.total_same_lane_spawn_rejections,
                    "spawned_inside_visible_roi": spawn_manager.total_spawned_inside_visible_roi,
                    "spawned_inside_buffer": spawn_manager.total_spawned_inside_buffer,
                    "max_new_actors_in_single_update": max(
                        (h["spawned"] for h in spawn_manager.history), default=0,
                    ),
                    "frame_object_max_new_per_update_cfg": cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE,
                    # Controller-fix task: symmetric downward-control stats.
                    "total_density_pruned": spawn_manager.total_density_pruned,
                    "pruned_inside_visible_roi": spawn_manager.total_pruned_inside_visible_roi,
                    "max_pruned_in_single_update": max(
                        (h.get("pruned_this_update", 0) for h in spawn_manager.history), default=0,
                    ),
                    "frame_object_max_prune_per_update_cfg": cfg.SPAWN.FRAME_OBJECT_MAX_PRUNE_PER_UPDATE,
                    "frame_object_target_interval_min_cfg": cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MIN,
                    "frame_object_target_interval_max_cfg": cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MAX,
                    "final_managed_population": len(spawn_manager.managed_actors),
                    "updates": spawn_manager.update_index,
                }

                write_json(
                    os.path.join(geometry_root, "canonical_spawn_summary.json"),
                    summary,
                )
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"canonical_spawn_summary: {exc}"
                )

        # spawn_manager.managed_actors is the authoritative superset of
        # every actor still alive at this point -- the initial Gamma
        # spawn plus everything Phase 2 replenishment added since (some
        # of the original walkers/traffic_actors may already be gone,
        # despawned by spawn_manager.update() during the run). Destroy
        # through it instead of the original lists to avoid stale
        # references and redundant destroy calls.
        if spawn_manager is not None:

            try:
                spawn_manager.destroy_all()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"spawn_manager: {exc}"
                )

        else:

            try:
                destroy_pedestrians(
                    walkers,
                    walker_controllers,
                )
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"pedestrians: {exc}"
                )

            try:
                destroy_traffic_vehicles(
                    traffic_actors
                )
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"traffic: {exc}"
                )

        if ego is not None:

            try:
                destroy_vehicle(
                    ego
                )
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"ego: {exc}"
                )

        # Important:
        # process destroy requests before starting
        # the next sequence.
        try:
            world.tick()
        except RuntimeError:
            pass


# ============================================================
# Stage 2: deterministic weather replay
# ============================================================

def replay_condition(
    world,
    client,
    reader,
    canonical_calibration,
    condition,
    source_condition,
    route_path,
    position_tolerance_m=DEFAULT_POSITION_TOLERANCE_M,
    rotation_tolerance_deg=DEFAULT_ROTATION_TOLERANCE_DEG,
):
    """
    Render one weather condition's stereo RGB by replaying the recorded
    canonical world state. No driving/traffic simulation runs here: every
    frame the ego and every actor are placed at their recorded transforms
    (apply_batch_sync, physics off), then only rgb_left / rgb_right are
    captured and saved.

    Nothing else is replayed or written: depth / semantic / optical flow /
    LiDAR / radar / labels / pose / calibration are canonical-only GT and are
    not spawned, captured or saved here. Replay validation is limited to
    geometry/transform correspondence, RGB presence and calibration
    reference equality.
    """

    if condition == source_condition:
        raise ValueError(
            f"'{condition}' is the canonical source condition; its RGB comes "
            f"from the canonical run and must not be replayed."
        )

    cond_dir = condition_dir(route_path, condition)

    if os.path.exists(cond_dir):
        shutil.rmtree(cond_dir)

    os.makedirs(cond_dir, exist_ok=True)

    num_frames = len(reader)

    replayer = None
    rig = None
    collector = None

    start_time = time.time()

    try:

        print()
        print(
            "========================================"
        )
        print(
            "Stage 2   : weather replay"
        )
        print(
            f"Route     : {route_path}"
        )
        print(
            f"Condition : {condition}  ({num_frames} frames)"
        )
        print(
            "========================================"
        )

        weather = apply_weather(
            world,
            condition,
        )

        first_frame = reader.load_frame(0)

        replayer = WorldStateReplayer(
            world,
            client,
            reader,
            EGO_BLUEPRINT,
            position_tolerance_m=position_tolerance_m,
            rotation_tolerance_deg=rotation_tolerance_deg,
        )

        replayer.start(first_frame)

        # RGB-only rig: no depth / semantic / flow / lidar / radar sensor
        # exists during replay (no callbacks, no GPU load, nothing to save).
        sensor_names = REPLAY_SENSOR_PROFILE

        rig = SensorRig(
            world,
            replayer.ego,
            cfg,
        ).spawn(
            profile="replay"
        )

        # Resolve attached sensor transforms before reading calibration.
        replayer.apply_frame(first_frame)
        world.tick()

        log_world_sensors(world, f"replay:{condition}")

        calibration_check = compare_calibration(
            canonical_calibration,
            build_calibration(
                ego_vehicle=replayer.ego,
                sensor_actors=get_sensor_actors(
                    rig
                ),
            ),
            tolerance=calibration_tolerance(
                max(
                    abs(first_frame["ego"]["transform"][axis])
                    for axis in ("x", "y", "z")
                )
            ),
        )

        print(
            "[Calibration] "
            f"max diff vs geometry = {calibration_check['max_abs_difference']:.3e} "
            f"({'equal' if calibration_check['equal'] else 'DIFFERENT'})"
        )

        collector = Collector(
            rig=rig,
            cfg=cfg,
            condition_root=cond_dir,
            required_sensors=sensor_names,
            timeout=COLLECTOR_TIMEOUT,
        )

        # Same warmup as the canonical run (rendering / exposure settle),
        # holding the frame-0 scene.
        for _ in range(
            WARMUP_FRAMES
        ):
            replayer.apply_frame(first_frame)
            world.tick()

        rig.clear_queues()

        log_runtime_weather(world, f"replay:{condition}")

        rgb_dimensions = {}

        saved_frames = 0

        # Canonical world_state samples are already 10 Hz (one per
        # cfg.RECORDING.SAMPLE_INTERVAL_SECONDS); this world still ticks at
        # 20 Hz (fixed_delta_seconds unchanged, CLAUDE.md section 13), and
        # the replay RGB sensors emit at the same 10 Hz cadence (sensor_tick
        # = cfg.RECORDING.SAMPLE_INTERVAL_SECONDS, see src/sensors/camera.py)
        # -- so each replay sample needs record_stride world ticks before
        # the sensors actually produce a new callback.
        record_stride = record_stride_ticks(cfg)

        for frame_id in range(
            num_frames
        ):

            frame_state = (
                first_frame
                if frame_id == 0
                else reader.load_frame(frame_id)
            )

            replayer.apply_frame(
                frame_state
            )

            # frame_id == 0's pose has already been held since before the
            # warmup loop above (same first_frame transform reapplied every
            # warmup tick), so the immediate next tick is already a capture
            # tick for the 10 Hz RGB sensors -- identical warmup-tick-count
            # parity to the canonical run's own frame 0 (both are "1 resolve
            # tick + WARMUP ticks" before their main loop, see
            # generate_canonical_geometry()). Every later frame's transform
            # changes right here, so it needs the full record_stride ticks
            # (one settling tick + one capturing tick, at stride 2) before
            # the RGB sensors emit under the new pose. The settling tick(s)
            # send no new transform (physics stays off, so the actor does
            # not move) and nothing is captured/saved for them (CLAUDE.md
            # section 13) -- only the LAST tick's carla_frame is collected.
            ticks_this_frame = 1 if frame_id == 0 else record_stride

            carla_frame = None

            for _ in range(ticks_this_frame):
                carla_frame = world.tick()

            packet = (
                collector.collect_frame(
                    carla_frame
                )
            )

            collector.save_frame(
                frame_id,
                packet,
            )

            for name in sensor_names:
                rgb_dimensions.setdefault(name, set()).add(
                    (int(packet[name].width), int(packet[name].height))
                )

            replayer.verify_frame(
                frame_state
            )

            saved_frames += 1

            if (
                frame_id == 0
                or saved_frames % 100 == 0
            ):
                print(
                    f"[{condition} {saved_frames:06d}/{num_frames:06d}] "
                    f"ego_pos_err_max="
                    f"{replayer.ego_errors.position_max:.4f} m "
                    f"actors={len(replayer.actors)}"
                )

        replay_validation = replayer.summary()

        replay_validation["rgb_frames_saved"] = min(
            count_files(os.path.join(cond_dir, "rgb_left"), ".png"),
            count_files(os.path.join(cond_dir, "rgb_right"), ".png"),
        )

        replay_validation["rgb_dimensions"] = {
            name: sorted(dimensions)
            for name, dimensions in rgb_dimensions.items()
        }

        if (
            replay_validation["rgb_frames_saved"] != num_frames
            or any(len(dimensions) != 1 for dimensions in rgb_dimensions.values())
        ):
            replay_validation["passed"] = False

        write_condition_json(
            cond_dir,
            condition,
            source_condition,
            weather,
            saved_frames,
            rendered_from="replay",
            replay_validation=replay_validation,
            calibration_check=calibration_check,
        )

        elapsed = (
            time.time()
            - start_time
        )

        if not replay_validation["passed"] or not calibration_check["equal"]:

            print(
                "[Replay validation] FAIL"
            )
            print(
                json.dumps(
                    replay_validation,
                    indent=2,
                )
            )

            raise RuntimeError(
                f"Replay validation failed for {condition}."
            )

        print(
            "[Replay validation] PASS "
            f"ego_max={replay_validation['ego']['position_max_m']:.4f} m "
            f"vehicles_max={replay_validation['vehicle_like_actors']['position_max_m']:.4f} m "
            f"pedestrians_max={replay_validation['pedestrians']['position_max_m']:.4f} m"
        )

        print(
            f"[Replay] {condition}: "
            f"{saved_frames} frames, "
            f"{elapsed:.1f} sec"
        )

        mark_complete(
            cond_dir,
            {"num_frames": saved_frames, "rendered_from": "replay"},
        )

        return {
            "status": "completed",
            "frames": saved_frames,
            "elapsed": elapsed,
        }

    finally:

        if collector is not None:

            try:
                collector.close()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"collector: {exc}"
                )

        if rig is not None:

            try:
                rig.destroy()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"rig: {exc}"
                )

        if replayer is not None:

            try:
                replayer.destroy_all()
            except Exception as exc:
                print(
                    "[Cleanup] "
                    f"replayer: {exc}"
                )

        try:
            world.tick()
        except RuntimeError:
            pass


# ============================================================
# One route: canonical geometry + all weather renderings
# ============================================================

def order_conditions(requested, source_condition):
    """Source condition first (it is rendered by the canonical run)."""

    ordered = [source_condition]

    for condition in requested:
        if condition not in ordered:
            ordered.append(condition)

    return ordered


def process_route(
    world,
    client,
    traffic_manager,
    town,
    route_id,
    dense_route,
    conditions,
    source_condition,
    output_root,
    args,
):
    """
    Canonical geometry (+ the source condition's RGB) once, then RGB-only
    replay of the remaining weathers. What runs is decided by
    src/data/resume_plan.py plan_route_work():

      * geometry COMPLETE (canonical GT + day_clear RGB) -> never re-collected
      * COMPLETE weather conditions are skipped
      * only missing weathers (or --rerender-conditions) are replayed
      * the source condition (day_clear) is never replayed
      * --overwrite (or an incomplete geometry) regenerates the whole route
    """

    route_path = route_root(
        output_root,
        town,
        route_id,
    )

    geometry_root = geometry_dir(route_path)

    counts = {
        "geometry_completed": 0,
        "conditions_completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    plan = plan_route_work(
        route_path,
        conditions,
        source_condition,
        rerender_conditions=args.rerender_conditions,
        overwrite=args.overwrite,
        expected_recording_hz=cfg.RECORDING.FPS,
    )

    # --------------------------------------------------------
    # Stage 1: canonical geometry + source-condition RGB
    # --------------------------------------------------------

    if plan["run_canonical"]:

        # A partial / stale / overwritten geometry invalidates every
        # condition that was rendered from it.
        if plan["delete_route"]:
            shutil.rmtree(route_path)

        try:

            generate_canonical_geometry(
                world=world,
                client=client,
                traffic_manager=traffic_manager,
                town=town,
                route_id=route_id,
                dense_route=dense_route,
                source_condition=source_condition,
                planned_conditions=conditions,
                route_path=route_path,
                max_frames=args.max_frames,
                truncate_ok=args.truncate_ok,
                no_traffic=args.no_traffic,
                no_pedestrians=args.no_pedestrians,
                background_policy=args.background_policy,
            )

            counts["geometry_completed"] += 1
            counts["conditions_completed"] += 1

        except Exception as exc:

            counts["failed"] += 1

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                "[CANONICAL GEOMETRY FAILED]"
            )
            print(
                f"Town      : {town}"
            )
            print(
                f"Route     : {route_id}"
            )
            print(
                f"Reason    : {exc}"
            )
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            # No canonical geometry -> nothing to replay for this route.
            return counts

    else:

        print(
            f"[Skip] Geometry + {source_condition} RGB complete: {geometry_root}"
        )

    # --------------------------------------------------------
    # Stage 2: RGB-only replay of the missing / re-requested weathers
    # --------------------------------------------------------

    for condition in plan["skip"]:

        counts["skipped"] += 1

        print(
            f"[Skip] Condition complete: {condition_dir(route_path, condition)}"
        )

    reader = None
    canonical_calibration = None

    for condition in plan["replay"]:

        cond_dir = condition_dir(
            route_path,
            condition,
        )

        try:

            if reader is None:

                reader = WorldStateReader(
                    geometry_root
                )

                with open(
                    os.path.join(geometry_root, "calibration.json"),
                    "r",
                    encoding="utf-8",
                ) as file:
                    canonical_calibration = json.load(file)

            replay_condition(
                world=world,
                client=client,
                reader=reader,
                canonical_calibration=canonical_calibration,
                condition=condition,
                source_condition=source_condition,
                route_path=route_path,
                position_tolerance_m=args.replay_position_tolerance,
                rotation_tolerance_deg=args.replay_rotation_tolerance,
            )

            counts["conditions_completed"] += 1

        except Exception as exc:

            counts["failed"] += 1

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                "[REPLAY FAILED]"
            )
            print(
                f"Town      : {town}"
            )
            print(
                f"Route     : {route_id}"
            )
            print(
                f"Condition : {condition}"
            )
            print(
                f"Reason    : {exc}"
            )
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            clear_complete(
                cond_dir
            )

            continue

    # --------------------------------------------------------
    # Cross-condition validation over everything now complete
    # --------------------------------------------------------

    complete_conditions = [
        condition
        for condition in conditions
        if is_complete(condition_dir(route_path, condition))
    ]

    update_sequence_conditions(
        geometry_root,
        complete_conditions,
    )

    report = validate_route(
        route_path,
        complete_conditions,
        source_condition=source_condition,
    )

    write_json(
        os.path.join(route_path, "paired_validation.json"),
        report,
    )

    print(
        f"[Paired validation] {'PASS' if report['passed'] else 'FAIL'} "
        f"({len(complete_conditions)}/{len(conditions)} conditions, "
        f"{report.get('num_frames')} frames)"
    )

    for error in report["errors"]:
        print(
            f"    {error}"
        )

    if not report["passed"]:
        counts["failed"] += 1

    return counts


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output_root = os.path.abspath(
        args.output_root
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    towns = (
        args.towns
        if args.towns is not None
        else [cfg.MAP.NAME]
    )

    requested_conditions = (
        args.conditions
        if args.conditions is not None
        else list(
            cfg.WEATHER.CONDITIONS
        )
    )

    source_condition = cfg.WEATHER.DEFAULT

    conditions = order_conditions(
        requested_conditions,
        source_condition,
    )

    if args.rerender_conditions:

        # Fail fast, before any CARLA work: unknown conditions, or the
        # canonical source condition (its RGB is not produced by replay).
        plan_route_work(
            "<validate-only>",
            conditions,
            source_condition,
            rerender_conditions=args.rerender_conditions,
            expected_recording_hz=cfg.RECORDING.FPS,
        )

    # --------------------------------------------------------
    # CARLA client
    # --------------------------------------------------------

    client = carla.Client(
        cfg.CARLA.HOST,
        cfg.CARLA.PORT,
    )

    client.set_timeout(
        cfg.CARLA.TIMEOUT
    )

    world = None
    original_settings = None
    traffic_manager = None

    totals = {
        "geometry_completed": 0,
        "conditions_completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    try:

        for town in towns:

            print()
            print(
                "########################################"
            )
            print(
                f"# TOWN: {town}"
            )
            print(
                "########################################"
            )

            # =================================================
            # Load town
            # =================================================

            xml_path = (
                route_xml_path(
                    town
                )
            )

            if not os.path.isfile(
                xml_path
            ):
                raise FileNotFoundError(
                    f"Route XML not found: "
                    f"{xml_path}"
                )

            carla_map_name = (
                resolve_carla_map_name(
                    xml_path
                )
            )

            world = client.load_world(
                carla_map_name
            )

            original_settings = (
                world.get_settings()
            )

            settings = (
                world.get_settings()
            )

            settings.synchronous_mode = (
                True
            )

            settings.fixed_delta_seconds = (
                cfg.SIMULATION
                .FIXED_DELTA_SECONDS
            )

            world.apply_settings(
                settings
            )

            # =================================================
            # Static environment cleanup
            # =================================================
            # Map-embedded static vehicles/pedestrians must be disabled
            # before any ego/NPC/sensor is spawned.

            static_removed = (
                disable_static_traffic_objects(
                    world
                )
            )

            print(
                f"[Environment] {town} "
                f"static vehicles disabled: "
                f"{static_removed['vehicles']}"
            )

            print(
                f"[Environment] {town} "
                f"static pedestrians disabled: "
                f"{static_removed['pedestrians']}"
            )

            # =================================================
            # Traffic light cycle (applied once per town load; state
            # duration only -- see configure_traffic_lights())
            # =================================================

            configure_traffic_lights(world, cfg)

            # =================================================
            # Traffic manager (canonical geometry generation only)
            # =================================================

            traffic_manager = (
                configure_traffic_manager(
                    client,
                    cfg,
                )
            )

            # =================================================
            # Route IDs
            # =================================================
            # xml_path was already resolved above (needed before
            # client.load_world()).

            if args.routes is None:

                route_ids = (
                    list_route_ids(
                        xml_path
                    )
                )

            else:

                route_ids = (
                    args.routes
                )

            print(
                f"[Town] Routes: "
                f"{route_ids}"
            )

            # =================================================
            # Routes
            # =================================================

            for route_id in route_ids:

                (
                    route_town,
                    control_points,
                    dense_route,
                ) = prepare_route(
                    world,
                    xml_path,
                    route_id,
                )

                print()
                print(
                    f"[Route {route_id}] "
                    f"control="
                    f"{len(control_points)}, "
                    f"dense="
                    f"{len(dense_route)}"
                )

                route_counts = process_route(
                    world=world,
                    client=client,
                    traffic_manager=traffic_manager,
                    town=town,
                    route_id=route_id,
                    dense_route=dense_route,
                    conditions=conditions,
                    source_condition=source_condition,
                    output_root=output_root,
                    args=args,
                )

                for key, value in route_counts.items():
                    totals[key] += value

            # =================================================
            # Restore town before next map
            # =================================================

            if (
                traffic_manager
                is not None
            ):
                try:
                    traffic_manager.set_synchronous_mode(
                        False
                    )
                except RuntimeError:
                    pass

            if (
                original_settings
                is not None
            ):
                world.apply_settings(
                    original_settings
                )

            world = None
            original_settings = None
            traffic_manager = None

    # ========================================================
    # Global cleanup
    # ========================================================

    finally:

        if (
            traffic_manager
            is not None
        ):

            try:
                traffic_manager.set_synchronous_mode(
                    False
                )
            except RuntimeError:
                pass

        if (
            world is not None
            and original_settings
            is not None
        ):

            try:
                world.apply_settings(
                    original_settings
                )
            except Exception:
                pass

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        " DATASET COLLECTION SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"Geometry sequences generated : {totals['geometry_completed']}"
    )

    print(
        f"Conditions rendered          : {totals['conditions_completed']}"
    )

    print(
        f"Conditions skipped           : {totals['skipped']}"
    )

    print(
        f"Failed                       : {totals['failed']}"
    )

    print(
        f"Root                         : {output_root}"
    )

    print(
        "========================================"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()

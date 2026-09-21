"""
collect_dataset.py

Final CARLA dataset collection pipeline.

Pipeline
--------
Town
  └─ Route
      └─ Weather condition
          ├─ Load route
          ├─ Spawn ego
          ├─ Spawn traffic / pedestrians
          ├─ Spawn sensors
          ├─ Save calibration
          ├─ Run RouteController
          ├─ Collect synchronized sensor data
          ├─ Save metadata
          ├─ Save annotations
          └─ Validate / finalize / cleanup

The script assumes:
- src.route
- src.controller
- src.weather
- src.sensor_rig
- src.collector
- src.calibration
- src.utils.metadata
- src.annotation
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

CARLA_ROOT = Path(r"C:\CARLA")
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
    spawn_actors_gamma_policy,
)
from src.simulation.canonical_traffic import (
    CanonicalBackgroundTraffic,
    get_annotation_candidate_count,
    scale_initial_category_counts,
)

from src.sensors.sensor_rig import SensorRig
from src.data.collector import Collector
from src.data.calibration import save_calibration
from src.data.metadata import MetadataWriter
from src.data.annotation import AnnotationWriter
from src.simulation.weather import apply_weather


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
            "Collect synchronized CARLA dataset sequences."
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
            "Weather conditions. "
            "If omitted, cfg.WEATHER.CONDITIONS is used."
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
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sequences.",
    )

    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Disable background traffic.",
    )

    parser.add_argument(
        "--no-pedestrians",
        action="store_true",
        help="Disable pedestrians.",
    )

    parser.add_argument(
        "--background-policy",
        type=str,
        default="canonical",
        choices=["canonical", "dynamic"],
        help=(
            "Background traffic maintenance policy. 'canonical' (default): "
            "actors are only ever spawned in a buffer zone ahead of the "
            "visible sensor ROI, never inside it (see "
            "src/simulation/canonical_traffic.py). 'dynamic': the earlier "
            "Phase 2/2.5 per-bin Gamma-deficit replenishment "
            "(src/simulation/spawn_policy.py DynamicSpawnManager), kept "
            "for reference/comparison, not the production default."
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


def sequence_root(
    output_root,
    town,
    route_id,
    condition,
):
    return os.path.join(
        output_root,
        town,
        f"route_{route_id}",
        condition,
    )


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


def validate_sequence(
    root,
    expected_frames,
):
    errors = []

    # --------------------------------------------------------
    # Sequence-level files
    # --------------------------------------------------------

    required_files = [
        "calibration.json",
        "sequence.json",
        "timestamps.csv",
        "ego_state.csv",
    ]

    for name in required_files:

        path = os.path.join(
            root,
            name,
        )

        if not os.path.isfile(
            path
        ):
            errors.append(
                f"Missing: {path}"
            )

    pose_path = os.path.join(
        root,
        "pose",
        "poses.csv",
    )

    for path in [
        pose_path,
    ]:
        if not os.path.isfile(path):
            errors.append(
                f"Missing: {path}"
            )

    # --------------------------------------------------------
    # Sensor file counts
    # --------------------------------------------------------

    file_checks = {
        "rgb_left": ".png",
        "rgb_right": ".png",
        "depth": ".npy",
        "optical_flow": ".npy",
        "semantic": ".npy",
        "lidar": ".npy",
        "radar": ".npy",
        "radar_front_left": ".npy",
        "radar_front_right": ".npy",
    }

    for directory, extension in (
        file_checks.items()
    ):

        count = count_files(
            os.path.join(
                root,
                directory,
            ),
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

    csv_paths = {
        "timestamps":
            os.path.join(
                root,
                "timestamps.csv",
            ),

        "ego_state":
            os.path.join(
                root,
                "ego_state.csv",
            ),

        "poses":
            pose_path,
    }

    for name, path in (
        csv_paths.items()
    ):

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
# One sequence
# ============================================================

def collect_sequence(
    world,
    client,
    traffic_manager,
    town,
    route_id,
    dense_route,
    condition,
    output_root,
    max_frames,
    overwrite=False,
    no_traffic=False,
    no_pedestrians=False,
    background_policy="canonical",
):

    root = sequence_root(
        output_root,
        town,
        route_id,
        condition,
    )

    # --------------------------------------------------------
    # Existing sequence handling
    # --------------------------------------------------------

    if os.path.exists(
        root
    ):

        if overwrite:
            shutil.rmtree(
                root
            )

        else:
            print(
                f"[Skip] Exists: "
                f"{root}"
            )
            return {
                "status": "skipped",
                "frames": 0,
            }

    os.makedirs(
        root,
        exist_ok=True,
    )

    ego = None
    rig = None

    collector = None
    metadata = None

    traffic_actors = {}
    walkers = []
    walker_controllers = []

    spawn_manager = None

    frame_object_csv_file = None
    frame_object_csv_writer = None

    saved_frames = 0

    sequence_start = time.time()

    try:

        print()
        print(
            "========================================"
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
            f"Output    : {root}"
        )
        print(
            "========================================"
        )

        # ====================================================
        # Weather
        # ====================================================

        apply_weather(
            world,
            condition,
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

        if background_policy == "dynamic":
            spawn_manager = DynamicSpawnManager(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                spawn_result["rng"],
                category_totals={
                    "vehicle": 0 if no_traffic else cfg.SPAWN.N_VEHICLES,
                    "motorcyclist": 0 if no_traffic else cfg.SPAWN.N_MOTORCYCLES,
                    "cyclist": 0 if no_traffic else cfg.SPAWN.N_BICYCLES,
                    "pedestrian": 0 if no_pedestrians else cfg.SPAWN.N_PEDESTRIANS,
                },
            )
        else:
            spawn_manager = CanonicalBackgroundTraffic(
                world,
                ego,
                dense_route,
                traffic_manager,
                cfg,
                spawn_result["rng"],
                category_totals={
                    "vehicle": 0 if no_traffic else cfg.SPAWN.N_VEHICLES,
                    "motorcyclist": 0 if no_traffic else cfg.SPAWN.N_MOTORCYCLES,
                    "cyclist": 0 if no_traffic else cfg.SPAWN.N_BICYCLES,
                    "pedestrian": 0 if no_pedestrians else cfg.SPAWN.N_PEDESTRIANS,
                },
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
        ).spawn()

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
                    root,
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
        # Collector
        # ====================================================

        collector = Collector(
            rig=rig,
            sequence_root=root,
            cfg=cfg,
            timeout=(
                COLLECTOR_TIMEOUT
            ),
        )

        # ====================================================
        # Metadata
        # ====================================================

        metadata = MetadataWriter(
            sequence_root=root,
            map_name=(
                world.get_map().name
            ),
            sequence_id=(
                f"{town}_"
                f"route_{route_id}_"
                f"{condition}"
            ),
            cfg=cfg,
            route_id=route_id,
            condition=condition,
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
        # Annotation
        # ====================================================

        annotation_writer = (
            AnnotationWriter(
                root,
                cfg,
                ego=ego,
                left_camera_actor=rig.get_sensor("rgb_left"),
            )
        )

        # ====================================================
        # Frame-Level Gamma Object Count Policy: per-frame
        # target/actual log (canonical policy only -- see
        # src/simulation/canonical_traffic.py FrameObjectGammaSchedule)
        # ====================================================

        if background_policy == "canonical":
            frame_object_csv_file = open(
                os.path.join(root, "frame_object_counts.csv"),
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

        for _ in range(
            WARMUP_FRAMES
        ):
            world.tick()

        if hasattr(
            rig,
            "clear_queues",
        ):
            rig.clear_queues()

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

        for local_frame_id in range(
            max_frames
        ):

            # ------------------------------------------------
            # Vehicle control
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
            # Sensor packet
            # ------------------------------------------------

            packet = (
                collector.collect_frame(
                    carla_frame
                )
            )

            collector.save_frame(
                local_frame_id,
                packet,
            )

            # ------------------------------------------------
            # Controller / route status
            # ------------------------------------------------

            route_status = (
                route_controller
                .get_status()
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata.write_frame(
                frame_id=(
                    local_frame_id
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

            # ------------------------------------------------
            # Annotation
            # ------------------------------------------------

            annotation_counts = (
                annotation_writer
                .write_frame(
                    local_frame_id,
                    world,
                    ego,
                    depth_raw=packet["depth"],
                )
            )

            saved_frames += 1

            # ------------------------------------------------
            # Phase 2: dynamic density maintenance
            #
            # Runs after this frame's sensors/annotation are already
            # captured and saved, so any actor spawned/despawned here
            # only takes effect starting from the *next* world.tick() --
            # it never disturbs the frame just collected.
            #
            # Computed before the CSV write below (not after, as in the
            # prior frame-object-Gamma task) so spawned/pruned/despawned-
            # this-frame and the cumulative_* columns can reflect THIS
            # update if one ran on this frame.
            # ------------------------------------------------

            current_object_count = get_annotation_candidate_count(annotation_counts)

            spawned_this_frame = 0
            pruned_this_frame = 0
            naturally_despawned_this_frame = 0

            if (
                spawn_manager is not None
                and local_frame_id % cfg.SPAWN.UPDATE_INTERVAL_FRAMES == 0
            ):
                if background_policy == "canonical":
                    update_snapshot = spawn_manager.update(
                        local_frame_id,
                        current_object_count=current_object_count,
                    )
                    spawned_this_frame = update_snapshot["spawned"]
                    pruned_this_frame = update_snapshot.get("pruned_this_update", 0)
                    naturally_despawned_this_frame = update_snapshot["despawned"]
                    last_visible_population = update_snapshot["visible_population"]
                    last_buffer_population = update_snapshot["buffer_population"]
                else:
                    spawn_manager.update(
                        local_frame_id
                    )

            # ------------------------------------------------
            # Frame-Level Gamma Object Count Policy: per-frame
            # target/actual/population/spawn-prune-despawn log
            # (every frame; spawned/pruned/despawned-this-frame are 0 on
            # frames where spawn_manager.update() didn't run this frame).
            # ------------------------------------------------

            if frame_object_csv_writer is not None:
                frame_object_csv_writer.writerow([
                    local_frame_id,
                    spawn_manager.frame_object_schedule.target_for_frame(local_frame_id),
                    current_object_count,
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
                saved_frames
                % FLUSH_INTERVAL
                == 0
            ):
                collector.flush()
                metadata.flush()

                if frame_object_csv_file is not None:
                    frame_object_csv_file.flush()

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if (
                local_frame_id == 0
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
                    f"condition={condition}, "
                    f"progress="
                    f"{route_status.get('progress', 0.0):.2f}%, "
                    f"index="
                    f"{route_status.get('route_index')}"
                )

        else:

            raise RuntimeError(
                f"Maximum frame count "
                f"{max_frames} reached "
                "before route completion."
            )

        # ====================================================
        # Finalize
        # ====================================================

        collector.flush()

        metadata.finalize()
        metadata = None

        collector.close()
        collector = None

        # ====================================================
        # Validation
        # ====================================================

        errors = validate_sequence(
            root,
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
                "Sequence validation failed."
            )

        print(
            "[Validation] PASS"
        )

        print(
            f"[Sequence] "
            f"{saved_frames} frames, "
            f"{elapsed:.1f} sec"
        )

        return {
            "status": "completed",
            "frames": saved_frames,
            "elapsed": elapsed,
            "root": root,
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
        # try-block's normal "Finalize" section) because a 600-frame
        # validation run legitimately raises "Maximum frame count
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

                with open(os.path.join(root, "canonical_spawn_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
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

    conditions = (
        args.conditions
        if args.conditions is not None
        else list(
            cfg.WEATHER.CONDITIONS
        )
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

    completed = 0
    skipped = 0
    failed = 0

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
            # Traffic manager
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

                # =============================================
                # Conditions
                # =============================================

                for condition in conditions:

                    try:

                        result = (
                            collect_sequence(
                                world=world,
                                client=client,
                                traffic_manager=(
                                    traffic_manager
                                ),
                                town=town,
                                route_id=(
                                    route_id
                                ),
                                dense_route=(
                                    dense_route
                                ),
                                condition=(
                                    condition
                                ),
                                output_root=(
                                    output_root
                                ),
                                max_frames=(
                                    args.max_frames
                                ),
                                overwrite=(
                                    args.overwrite
                                ),
                                no_traffic=(
                                    args.no_traffic
                                ),
                                no_pedestrians=(
                                    args.no_pedestrians
                                ),
                                background_policy=(
                                    args.background_policy
                                ),
                            )
                        )

                        if (
                            result["status"]
                            == "completed"
                        ):
                            completed += 1

                        elif (
                            result["status"]
                            == "skipped"
                        ):
                            skipped += 1

                    except Exception as exc:

                        failed += 1

                        print()
                        print(
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        )
                        print(
                            "[SEQUENCE FAILED]"
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

                        # Continue with next sequence.
                        continue

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
        f"Completed : {completed}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Failed    : {failed}"
    )

    print(
        f"Root      : {output_root}"
    )

    print(
        "========================================"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
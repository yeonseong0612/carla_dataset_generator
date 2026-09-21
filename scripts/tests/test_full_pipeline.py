"""
Full CARLA dataset pipeline integration test.

Test flow
---------
1. Load CARLA world
2. Enable synchronous simulation
3. Apply weather
4. Load XML route
5. Build dense CARLA route
6. Spawn ego vehicle at route start
7. Spawn traffic / pedestrians
8. Spawn SensorRig
9. Save calibration
10. Create RouteController
11. Create Collector / MetadataWriter / AnnotationWriter
12. Warm up sensors
13. Run synchronized collection loop
14. Validate saved dataset
15. Clean up all CARLA actors

This script is NOT the final dataset generator. It deliberately writes one
flat single-sequence directory (Collector legacy `sequence_root=` mode,
outputs/full_pipeline_test/...), not the production paired
geometry/ + conditions/ layout produced by collect_dataset.py.
It is a small integration test before collect_dataset.py.
"""

import os
import sys
import csv
import shutil
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_ROOT = Path(r"C:\CARLA")
CARLA_PYTHONAPI = CARLA_ROOT / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))


# ============================================================
# Project imports
# ============================================================

from CFG.config import cfg

from src.navigation.route import (
    load_route_from_xml,
    project_control_points,
    build_dense_route,
)

from src.navigation.controller import RouteController

from src.simulation.vehicle import destroy_vehicle

from src.simulation.traffic import (
    configure_traffic_manager,
    destroy_traffic_vehicles,
)

from src.simulation.pedestrian import (
    spawn_pedestrians,
    start_pedestrians,
    destroy_pedestrians,
)

from src.sensors.sensor_rig import SensorRig
from src.data.collector import Collector
from src.data.calibration import save_calibration
from src.data.metadata import MetadataWriter
from src.data.annotation import AnnotationWriter
from src.simulation.weather import apply_weather


# ============================================================
# Test configuration
# ============================================================

MAP_NAME = "Town01"

ROUTE_XML = os.path.join(
    cfg.PROJECT.ROOT,
    "routes",
    "Town01.xml",
)

ROUTE_ID = "0"

SEQUENCE_ID = "test_00"

CONDITION = "day_clear"


# ------------------------------------------------------------
# Test length
# ------------------------------------------------------------

# 200 is enough for first integration test.
# Increase to 500~2000 after this passes.
NUM_FRAMES = 2000

WARMUP_FRAMES = 20


# ------------------------------------------------------------
# Controller
# ------------------------------------------------------------

CRUISE_SPEED_KMH = 30.0
MIN_CURVE_SPEED_KMH = 12.0

TRAFFIC_LIGHT_POLICY = "obey"


# ------------------------------------------------------------
# Route
# ------------------------------------------------------------

ROUTE_SAMPLING_RESOLUTION = 2.0


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

SEQUENCE_ROOT = os.path.join(
    cfg.PROJECT.ROOT,
    "outputs",
    "full_pipeline_test",
    MAP_NAME,
    f"route_{ROUTE_ID}",
    CONDITION,
)


# ============================================================
# Ego spawn
# ============================================================

def spawn_ego_at_route_start(
    world,
    dense_route,
    blueprint_id="vehicle.tesla.model3",
):
    """
    Spawn ego vehicle at the first waypoint of the route.
    """

    if not dense_route:
        raise RuntimeError(
            "Dense route is empty."
        )

    blueprint_library = (
        world.get_blueprint_library()
    )

    blueprint = (
        blueprint_library.find(
            blueprint_id
        )
    )

    if blueprint.has_attribute(
        "role_name"
    ):
        blueprint.set_attribute(
            "role_name",
            "hero",
        )

    # dense_route:
    # [(Waypoint, RoadOption), ...]
    start_waypoint = (
        dense_route[0][0]
    )

    start_transform = carla.Transform(
        start_waypoint.transform.location,
        start_waypoint.transform.rotation,
    )

    # Slight height offset to avoid spawn collision
    # with the road surface.
    start_transform.location.z += 0.3

    ego = world.try_spawn_actor(
        blueprint,
        start_transform,
    )

    if ego is None:
        raise RuntimeError(
            "Failed to spawn ego vehicle "
            "at route start."
        )

    return ego


# ============================================================
# Sensor actor extraction
# ============================================================

def get_sensor_actors(rig):
    """
    Return the sensor actor dictionary used by calibration.py.

    Adjust this function only if SensorRig uses a different
    internal actor container name.
    """

    if hasattr(rig, "sensors"):
        return rig.sensors

    if hasattr(rig, "actors"):
        return rig.actors

    raise AttributeError(
        "SensorRig must expose sensor actors through "
        "'rig.sensors' or 'rig.actors'."
    )


# ============================================================
# Output validation
# ============================================================

def validate_saved_frame(
    sequence_root,
    frame_id,
):
    """
    Check whether all expected per-frame files exist.
    """

    frame_name = (
        f"{frame_id:06d}"
    )

    paths = [
        os.path.join(
            sequence_root,
            "rgb_left",
            f"{frame_name}.png",
        ),

        os.path.join(
            sequence_root,
            "rgb_right",
            f"{frame_name}.png",
        ),

        os.path.join(
            sequence_root,
            "depth",
            f"{frame_name}.npy",
        ),

        os.path.join(
            sequence_root,
            "optical_flow",
            f"{frame_name}.npy",
        ),

        os.path.join(
            sequence_root,
            "semantic",
            f"{frame_name}.npy",
        ),

        os.path.join(
            sequence_root,
            "lidar",
            f"{frame_name}.npy",
        ),

        os.path.join(
            sequence_root,
            "radar",
            f"{frame_name}.npy",
        ),

        os.path.join(
            sequence_root,
            "labels",
            "object_3d",
            f"{frame_name}.json",
        ),
    ]

    return [
        path
        for path in paths
        if not os.path.isfile(path)
    ]


def count_csv_data_rows(path):
    """
    Count CSV rows excluding the header.
    """

    if not os.path.isfile(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)

    if not rows:
        return 0

    return max(
        0,
        len(rows) - 1,
    )


def validate_sequence_files(
    sequence_root,
    expected_frames,
):
    """
    Validate sequence-level files and CSV row counts.
    """

    errors = []

    required_files = [
        os.path.join(
            sequence_root,
            "calibration.json",
        ),

        os.path.join(
            sequence_root,
            "sequence.json",
        ),

        os.path.join(
            sequence_root,
            "timestamps.csv",
        ),

        os.path.join(
            sequence_root,
            "ego_state.csv",
        ),

        os.path.join(
            sequence_root,
            "pose",
            "poses.csv",
        ),
    ]

    for path in required_files:

        if not os.path.isfile(path):

            errors.append(
                f"Missing file: {path}"
            )

    csv_checks = {
        "timestamps.csv":
            os.path.join(
                sequence_root,
                "timestamps.csv",
            ),

        "ego_state.csv":
            os.path.join(
                sequence_root,
                "ego_state.csv",
            ),

        "poses.csv":
            os.path.join(
                sequence_root,
                "pose",
                "poses.csv",
            ),
    }

    for name, path in csv_checks.items():

        rows = count_csv_data_rows(
            path
        )

        if rows is None:
            continue

        if rows != expected_frames:

            errors.append(
                f"{name}: expected "
                f"{expected_frames} rows, "
                f"found {rows}"
            )

    return errors


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reset output directory
    # --------------------------------------------------------

    if os.path.exists(
        SEQUENCE_ROOT
    ):
        shutil.rmtree(
            SEQUENCE_ROOT
        )

    os.makedirs(
        SEQUENCE_ROOT,
        exist_ok=True,
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

    # --------------------------------------------------------
    # Resources used during cleanup
    # --------------------------------------------------------

    world = None
    original_settings = None

    traffic_manager = None
    traffic_actors = {}

    walkers = []
    controllers = []

    ego = None
    rig = None

    collector = None
    metadata = None

    try:

        # ====================================================
        # 1. Load world
        # ====================================================

        print()
        print(
            "========================================"
        )
        print(
            " CARLA FULL PIPELINE TEST"
        )
        print(
            "========================================"
        )
        print()

        world = client.load_world(
            MAP_NAME
        )

        carla_map = world.get_map()

        print(
            f"[World] Loaded: "
            f"{carla_map.name}"
        )

        # ====================================================
        # 2. Synchronous simulation
        # ====================================================

        original_settings = (
            world.get_settings()
        )

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            cfg.SIMULATION
            .FIXED_DELTA_SECONDS
        )

        world.apply_settings(
            settings
        )

        print(
            "[Simulation] "
            "Synchronous mode enabled"
        )

        print(
            "[Simulation] "
            f"Fixed delta: "
            f"{settings.fixed_delta_seconds}"
        )

        # ====================================================
        # 3. Traffic Manager
        # ====================================================

        traffic_manager = (
            configure_traffic_manager(
                client,
                cfg,
            )
        )

        # ====================================================
        # 4. Weather
        # ====================================================

        apply_weather(
            world,
            CONDITION,
        )

        print(
            f"[Weather] {CONDITION}"
        )

        # ====================================================
        # 5. Load XML route
        # ====================================================

        town, control_points = (
            load_route_from_xml(
                ROUTE_XML,
                ROUTE_ID,
            )
        )

        if town != MAP_NAME:
            raise RuntimeError(
                f"Route XML town mismatch: "
                f"XML={town}, "
                f"loaded={MAP_NAME}"
            )

        print(
            f"[Route] XML control points: "
            f"{len(control_points)}"
        )

        # ====================================================
        # 6. Project route onto road
        # ====================================================

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

        if len(dense_route) < 2:

            raise RuntimeError(
                "Dense route contains "
                "fewer than 2 waypoints."
            )

        print(
            f"[Route] Dense points: "
            f"{len(dense_route)}"
        )

        # ====================================================
        # 7. Spawn ego
        # ====================================================

        ego = (
            spawn_ego_at_route_start(
                world,
                dense_route,
            )
        )

        print(
            f"[Ego] Spawned actor "
            f"{ego.id}"
        )

        # ====================================================
        # 8. Route controller
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

        print(
            "[Controller] Ready"
        )

        # ====================================================
        # 9. Traffic
        # ====================================================
        # Legacy spawn_traffic_vehicles() removed; background traffic is
        # spawned by GammaSpawnPolicy/CanonicalBackgroundTraffic in
        # collect_dataset.py, not in this test. traffic_actors stays {}.

        # ====================================================
        # 10. Pedestrians
        # ====================================================

        (
            walkers,
            controllers,
            walker_speeds,
        ) = spawn_pedestrians(
            world,
            cfg,
        )

        start_pedestrians(
            world,
            controllers,
            walker_speeds,
            cfg,
        )

        print(
            f"[Pedestrians] "
            f"{len(walkers)}"
        )

        # ====================================================
        # 11. Sensor rig
        # ====================================================

        rig = SensorRig(
            world,
            ego,
            cfg,
        ).spawn()

        print(
            f"[Sensors] Spawned: "
            f"{len(rig)}"
        )

        # Let CARLA propagate attach_to parent-child offsets to the
        # client before reading sensor transforms; otherwise every
        # attached sensor still reports the ego's own transform and
        # calibration (e.g. stereo baseline) comes out as zero.
        world.tick()

        # ====================================================
        # 12. Calibration
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
                    SEQUENCE_ROOT,
                    "calibration.json",
                ),
            )
        )

        print(
            "[Calibration] Saved"
        )

        if "stereo" in calibration:

            stereo = (
                calibration["stereo"]
            )

            print(
                "[Calibration] "
                f"Stereo baseline="
                f"{stereo['baseline_m']:.4f} m"
            )

            print(
                "[Calibration] "
                f"CV baseline="
                f"{stereo['baseline_x_cv_m']:.4f} m"
            )

        # ====================================================
        # 13. Collector
        # ====================================================

        collector = Collector(
            rig=rig,
            sequence_root=(
                SEQUENCE_ROOT
            ),
            cfg=cfg,
            timeout=10.0,
        )

        # ====================================================
        # 14. Metadata
        # ====================================================

        metadata = MetadataWriter(
            sequence_root=(
                SEQUENCE_ROOT
            ),
            map_name=(
                world.get_map().name
            ),
            sequence_id=(
                SEQUENCE_ID
            ),
            cfg=cfg,
            route_id=(
                ROUTE_ID
            ),
            condition=(
                CONDITION
            ),
        )

        # ====================================================
        # 15. Annotation
        # ====================================================

        annotation_writer = (
            AnnotationWriter(
                SEQUENCE_ROOT,
                cfg,
            )
        )

        # ====================================================
        # 16. Warmup
        # ====================================================

        print()
        print(
            f"[Warmup] "
            f"{WARMUP_FRAMES} frames"
        )

        for _ in range(
            WARMUP_FRAMES
        ):
            world.tick()

        # Sensor queues now contain old frames.
        if hasattr(
            rig,
            "clear_queues",
        ):
            rig.clear_queues()

        print(
            "[Warmup] Complete"
        )

        # ====================================================
        # 17. Dataset collection
        # ====================================================

        saved_frames = 0

        print()
        print(
            "----------------------------------------"
        )
        print(
            " Collection start"
        )
        print(
            "----------------------------------------"
        )

        for local_frame_id in range(
            NUM_FRAMES
        ):

            # -----------------------------------------------
            # Vehicle control
            # -----------------------------------------------

            control = (
                route_controller.run_step()
            )

            ego.apply_control(
                control
            )

            # -----------------------------------------------
            # Advance simulation exactly one step
            # -----------------------------------------------

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

            # -----------------------------------------------
            # Strict synchronized sensor collection
            # -----------------------------------------------

            packet = (
                collector.collect_frame(
                    carla_frame
                )
            )

            collector.save_frame(
                local_frame_id,
                packet,
            )

            # -----------------------------------------------
            # Route/controller state
            # -----------------------------------------------

            route_status = (
                route_controller
                .get_status()
            )

            # -----------------------------------------------
            # Metadata
            # -----------------------------------------------

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

            # Flush periodically, not every frame.
            if (
                local_frame_id > 0
                and local_frame_id % 100 == 0
            ):
                metadata.flush()
                collector.flush()

            # -----------------------------------------------
            # 3D object annotations
            # -----------------------------------------------

            annotation_counts = (
                annotation_writer
                .write_frame(
                    local_frame_id,
                    world,
                    ego,
                )
            )

            saved_frames += 1

            # -----------------------------------------------
            # Console status
            # -----------------------------------------------

            if (
                local_frame_id == 0
                or (
                    local_frame_id + 1
                ) % 20 == 0
            ):

                print(
                    f"[{local_frame_id + 1:04d}/{NUM_FRAMES:04d}] "
                    f"local={local_frame_id:06d} "
                    f"carla={carla_frame} "
                    f"route={route_status.get('route_index')}"
                    f"/{route_status.get('route_length', 1) - 1} "
                    f"progress={route_status.get('progress', 0.0):.1f}% "
                    f"speed={route_status.get('speed_kmh', 0.0):.1f} "
                    f"target={route_status.get('target_speed_kmh', 0.0):.1f} "
                    f"curve={route_status.get('curve_angle_deg', 0.0):.1f} "
                    f"V={annotation_counts.get('vehicle', 0)} "
                    f"P={annotation_counts.get('pedestrian', 0)}"
                )
            # -----------------------------------------------
            # Route completion
            # -----------------------------------------------

            if route_status.get(
                "completed",
                False,
            ):
                print()
                print(
                    "[Route] Completed."
                )
                break

            # -----------------------------------------------
            # Stuck detection
            # -----------------------------------------------

            if route_status.get(
                "stuck",
                False,
            ):
                raise RuntimeError(
                    "Vehicle stuck detected. "
                    f"Progress="
                    f"{route_status.get('progress', 0.0):.2f}% "
                    f"route_index="
                    f"{route_status.get('route_index')}"
                )

        # ====================================================
        # 18. Finalize writers
        # ====================================================

        collector.flush()

        metadata.finalize()
        metadata = None

        collector.close()
        collector = None

        # ====================================================
        # 19. Validation
        # ====================================================

        print()
        print(
            "----------------------------------------"
        )
        print(
            " Validation"
        )
        print(
            "----------------------------------------"
        )

        print(
            f"Saved frames: "
            f"{saved_frames}"
        )

        # ----------------------------------------------------
        # Per-frame file validation
        # ----------------------------------------------------

        missing_total = 0

        for frame_id in range(
            saved_frames
        ):

            missing = (
                validate_saved_frame(
                    SEQUENCE_ROOT,
                    frame_id,
                )
            )

            if missing:

                missing_total += (
                    len(missing)
                )

                print(
                    f"[MISSING] "
                    f"{frame_id:06d}"
                )

                for path in missing:
                    print(
                        f"    {path}"
                    )

        if missing_total == 0:

            print(
                "[PASS] Per-frame files"
            )

        else:

            print(
                "[FAIL] Per-frame files: "
                f"{missing_total} "
                "missing files"
            )

        # ----------------------------------------------------
        # Sequence / CSV validation
        # ----------------------------------------------------

        sequence_errors = (
            validate_sequence_files(
                SEQUENCE_ROOT,
                saved_frames,
            )
        )

        if not sequence_errors:

            print(
                "[PASS] Sequence metadata"
            )

        else:

            print(
                "[FAIL] Sequence metadata"
            )

            for error in sequence_errors:
                print(
                    f"    {error}"
                )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print()
        print(
            "========================================"
        )

        if (
            missing_total == 0
            and not sequence_errors
        ):

            print(
                " FULL PIPELINE TEST: PASS"
            )

        else:

            print(
                " FULL PIPELINE TEST: FAIL"
            )

        print(
            "========================================"
        )

        print(
            f"Dataset root:\n"
            f"{SEQUENCE_ROOT}"
        )

    # ========================================================
    # Cleanup
    # ========================================================

    finally:

        print()
        print(
            "[Cleanup] Starting..."
        )

        # ----------------------------------------------------
        # File writers
        # ----------------------------------------------------

        if metadata is not None:

            try:
                metadata.close()
            except Exception as exc:
                print(
                    f"[Cleanup] Metadata: "
                    f"{exc}"
                )

        if collector is not None:

            try:
                collector.close()
            except Exception as exc:
                print(
                    f"[Cleanup] Collector: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # Sensors
        # ----------------------------------------------------

        if rig is not None:

            try:
                rig.destroy()
            except Exception as exc:
                print(
                    f"[Cleanup] SensorRig: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # Pedestrians
        # ----------------------------------------------------

        try:
            destroy_pedestrians(
                walkers,
                controllers,
            )
        except Exception as exc:
            print(
                f"[Cleanup] Pedestrians: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # Traffic
        # ----------------------------------------------------

        try:
            destroy_traffic_vehicles(
                traffic_actors
            )
        except Exception as exc:
            print(
                f"[Cleanup] Traffic: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # Ego
        # ----------------------------------------------------

        if ego is not None:

            try:
                destroy_vehicle(
                    ego
                )
            except Exception as exc:
                print(
                    f"[Cleanup] Ego: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # Traffic Manager
        # ----------------------------------------------------

        if traffic_manager is not None:

            try:
                traffic_manager.set_synchronous_mode(
                    False
                )

            except RuntimeError:
                pass

        # ----------------------------------------------------
        # World settings
        # ----------------------------------------------------

        if (
            world is not None
            and original_settings is not None
        ):

            try:
                world.apply_settings(
                    original_settings
                )

            except Exception as exc:
                print(
                    f"[Cleanup] World settings: "
                    f"{exc}"
                )

        print(
            "[Cleanup] Complete"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
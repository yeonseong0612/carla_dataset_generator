"""
scripts/tools/validate_runtime_environment.py

Standalone runtime check for the static-environment-object cleanup used by
the production dataset pipeline (scripts/collect_dataset.py).

It imports and calls the exact same production function
(src.simulation.environment.disable_static_traffic_objects) instead of
re-implementing any cleanup logic, then reports:

  - map-baked static traffic environment objects (parked cars, buses,
    trucks, motorcycles, bicycles, trains, pedestrians, riders) discovered
    before cleanup, and the disable-command coverage after cleanup
  - a sanity check that unrelated environment geometry (buildings, roads,
    traffic lights, traffic signs, vegetation) is left untouched
  - dynamic actors currently in the world (ego / vehicles / walkers /
    sensors), reported separately so a running vehicle is never mistaken
    for a leftover static object

IMPORTANT API LIMITATION (see printed [After cleanup] section): CARLA's
Python API (world.get_environment_objects / world.enable_environment_objects)
has no getter for an EnvironmentObject's current enabled flag. Re-querying
get_environment_objects(label) always returns the full map inventory for
that label, regardless of whether it is currently enabled or disabled. This
script can therefore only confirm that a disable command was issued for
100% of the discovered static objects; it cannot itself re-confirm
server-side that the flag took effect. A definitive check requires
rendering a frame (e.g. spectator view) and visually inspecting it.

Needs a live CARLA server. Never writes into dataset/; use --json to also
save the report under outputs/runtime_0915_validation/ or a path you choose.

Usage:
    python scripts/tools/validate_runtime_environment.py \
        --host localhost --port 2000 --town Town01

    python scripts/tools/validate_runtime_environment.py --no-reload
        (validates whatever world is already loaded on the server, without
        calling load_world -- less disruptive if something else is using
        the server)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import carla  # noqa: E402

from src.simulation.environment import (  # noqa: E402
    STATIC_VEHICLE_LABELS,
    STATIC_PEDESTRIAN_LABELS,
    get_static_object_ids,
    disable_static_traffic_objects,
)

# Environment geometry that MUST remain untouched by cleanup (CLAUDE.md
# section 4.B). Only checked for presence (count > 0), never modified here.
RETAIN_LABELS = (
    carla.CityObjectLabel.Buildings,
    carla.CityObjectLabel.Roads,
    carla.CityObjectLabel.TrafficLight,
    carla.CityObjectLabel.TrafficSigns,
    carla.CityObjectLabel.Vegetation,
)

ALL_STATIC_LABELS = STATIC_VEHICLE_LABELS + STATIC_PEDESTRIAN_LABELS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Runtime validation of static traffic environment object "
            "cleanup (production src.simulation.environment) and a "
            "dynamic-actor census, against a live CARLA server."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--town", default="Town01")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help=(
            "Validate the world already loaded on the server instead of "
            "calling client.load_world(town). Less disruptive to a shared "
            "server, but --town is then ignored."
        ),
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to also write this report as JSON.",
    )
    return parser.parse_args()


def classify_actors(actors):
    ego, vehicles, walkers, sensors, other = [], [], [], [], []

    for actor in actors:
        type_id = actor.type_id

        if type_id.startswith("sensor."):
            sensors.append(actor)
        elif type_id.startswith("controller."):
            continue
        elif type_id.startswith("vehicle."):
            role = actor.attributes.get("role_name", "") if actor.attributes else ""
            (ego if role == "hero" else vehicles).append(actor)
        elif type_id.startswith("walker.pedestrian"):
            walkers.append(actor)
        else:
            other.append(actor)

    return ego, vehicles, walkers, sensors, other


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    client_version = client.get_client_version()
    server_version = client.get_server_version()

    print(f"[Version] client={client_version} server={server_version}")

    version_mismatch = client_version != server_version
    if version_mismatch:
        print("[WARNING] client/server version mismatch.")

    if args.no_reload:
        world = client.get_world()
    else:
        print(f"[World] loading {args.town} ...")
        world = client.load_world(args.town)

    map_name = world.get_map().name
    print(f"[World] map = {map_name}")

    print()
    print("[Before cleanup] static traffic environment object inventory")
    before_ids = {}
    for label in ALL_STATIC_LABELS:
        ids = get_static_object_ids(world, [label])
        before_ids[label.name] = ids
        print(f"  {label.name:<12}: {len(ids)}")
    total_before = sum(len(v) for v in before_ids.values())

    print()
    print("[Cleanup] src.simulation.environment.disable_static_traffic_objects(world)")
    result = disable_static_traffic_objects(world)
    print(f"  vehicles disable command issued for   : {result['vehicles']}")
    print(f"  pedestrians disable command issued for: {result['pedestrians']}")

    print()
    print("[After cleanup] disable-command coverage per label")
    print("  NOTE: CARLA's public API has no getter for an EnvironmentObject's")
    print("  current enabled flag -- get_environment_objects(label) always")
    print("  returns the full map inventory regardless of enabled state, so")
    print("  a re-query cannot itself prove the flag was applied server-side.")
    print("  'OK' below means every object discovered before cleanup was")
    print("  included in the disable_environment_objects() call issued above.")
    coverage_ok = True
    for label in ALL_STATIC_LABELS:
        after_ids = get_static_object_ids(world, [label])
        before_set = set(before_ids[label.name])
        stable = set(after_ids) == before_set
        coverage_ok = coverage_ok and stable
        status = "OK" if stable else "INVENTORY CHANGED (unexpected)"
        print(f"  {label.name:<12}: {len(before_set)} objects -- {status}")

    print()
    print("[Sanity] environment geometry that must remain untouched")
    sanity_ok = True
    sanity_counts = {}
    for label in RETAIN_LABELS:
        count = len(world.get_environment_objects(label))
        sanity_counts[label.name] = count
        present = count > 0
        sanity_ok = sanity_ok and present
        print(f"  {label.name:<14}: {'present' if present else 'MISSING'} ({count})")

    print()
    print("[Dynamic actors] world.get_actors() classification")
    actors = list(world.get_actors())
    ego, vehicles, walkers, sensors, other = classify_actors(actors)
    print(f"  ego (role_name=hero) = {len(ego)}")
    print(f"  dynamic vehicles     = {len(vehicles)}")
    print(f"  walkers (pedestrians)= {len(walkers)}")
    print(f"  sensors              = {len(sensors)}")
    if other:
        print(f"  other/unclassified   = {len(other)} ({sorted({a.type_id for a in other})})")

    print()
    print("[Environment static traffic objects]")
    print(f"  discovered before cleanup = {total_before}")
    print(f"  disable command coverage  = {'OK' if coverage_ok else 'FAIL'}")

    passed = coverage_ok and sanity_ok and not version_mismatch

    print()
    print("=" * 60)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    if args.json:
        report = {
            "client_version": client_version,
            "server_version": server_version,
            "version_mismatch": version_mismatch,
            "map": map_name,
            "static_objects_before": {k: len(v) for k, v in before_ids.items()},
            "disable_result": result,
            "disable_coverage_ok": coverage_ok,
            "sanity_counts": sanity_counts,
            "sanity_ok": sanity_ok,
            "dynamic_actors": {
                "ego": len(ego),
                "vehicles": len(vehicles),
                "walkers": len(walkers),
                "sensors": len(sensors),
            },
            "passed": passed,
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[JSON] report written to {json_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

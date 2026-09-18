"""
environment.py

Removes map-embedded static traffic participants (parked cars, static
pedestrian-like props, ...) from a freshly loaded CARLA world, before any
ego/NPC/sensor is spawned. This keeps them from showing up as spurious
clutter in RGB/LiDAR/Radar returns and in ground-truth annotation, while
leaving environment geometry (buildings, roads, traffic lights/signs,
poles, vegetation, walls, guard rails, ...) untouched.

CARLA exposes these as carla.EnvironmentObject instances via
world.get_environment_objects(label), distinct from spawned
vehicle.*/walker.* actors. world.enable_environment_objects(ids, False)
hides them and removes their collision. This never touches any actor
(ego, traffic-manager NPCs, or pedestrians we spawn ourselves) -- only
objects baked into the map itself.

Label verification
-------------------
carla.CityObjectLabel does NOT expose a single "Vehicles" label in the
CARLA build this project targets -- only per-type vehicle labels. Static
pedestrian-shaped props are exposed via CityObjectLabel.Pedestrians;
CityObjectLabel.Rider (a static person posed on a bike/motorcycle) is
person-like rather than vehicle-like, so it is grouped with pedestrians
instead. Re-verify with:

    [n for n in dir(carla.CityObjectLabel) if not n.startswith("_")]
"""

import carla

STATIC_VEHICLE_LABELS = (
    carla.CityObjectLabel.Car,
    carla.CityObjectLabel.Bus,
    carla.CityObjectLabel.Truck,
    carla.CityObjectLabel.Motorcycle,
    carla.CityObjectLabel.Bicycle,
    carla.CityObjectLabel.Train,
)

STATIC_PEDESTRIAN_LABELS = (
    carla.CityObjectLabel.Pedestrians,
    carla.CityObjectLabel.Rider,
)


def get_static_object_ids(world, labels):
    ids = []

    for label in labels:
        ids.extend(obj.id for obj in world.get_environment_objects(label))

    return ids


def disable_static_traffic_objects(world):
    """
    Disable every map-embedded static vehicle and pedestrian-like
    environment object in `world`.

    Call this right after the map is loaded and before spawning ego,
    NPCs, or sensors.

    Returns
    -------
    dict with "vehicles" and "pedestrians" keys: the number of static
    objects disabled in each category.
    """

    vehicle_ids = get_static_object_ids(world, STATIC_VEHICLE_LABELS)
    pedestrian_ids = get_static_object_ids(world, STATIC_PEDESTRIAN_LABELS)

    if vehicle_ids:
        world.enable_environment_objects(vehicle_ids, False)

    if pedestrian_ids:
        world.enable_environment_objects(pedestrian_ids, False)

    return {
        "vehicles": len(vehicle_ids),
        "pedestrians": len(pedestrian_ids),
    }


def enumerate_static_traffic_objects(world):
    """
    Debug/validation helper: per-label static object counts without
    disabling anything (used to check what a map actually contains).
    """

    counts = {}

    for label in STATIC_VEHICLE_LABELS + STATIC_PEDESTRIAN_LABELS:
        counts[label.name] = len(world.get_environment_objects(label))

    return counts

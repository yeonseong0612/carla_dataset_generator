"""
Disable map-embedded static traffic objects before spawning ego/NPCs.

CARLA maps may contain baked-in parked vehicles and pedestrian-like
environment objects. These are EnvironmentObject instances, not spawned
vehicle/walker actors, so they are not controlled by the dataset traffic
pipeline.

They are disabled to avoid uncontrolled clutter in RGB/LiDAR/Radar and
ground-truth annotations. Buildings, roads, traffic lights, vegetation,
and other environment geometry are left untouched.
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
    
    counts = {}

    for label in STATIC_VEHICLE_LABELS + STATIC_PEDESTRIAN_LABELS:
        counts[label.name] = len(world.get_environment_objects(label))

    return counts

import os
import json
import math
import numpy as np


PASSENGER_CAR_FALLBACK = {
    "vehicle.bmw.grandtourer",
    "vehicle.mini.cooper_s",
}


def get_category(actor):
    if actor.type_id.startswith("walker.pedestrian."):
        return "pedestrian", None

    if not actor.type_id.startswith("vehicle."):
        return None, None

    base_type = actor.attributes.get("base_type", "").lower()

    if base_type == "bicycle":
        return "cyclist", None

    if base_type == "motorcycle":
        return "motorcyclist", None

    if base_type == "van":
        return "vehicle", "van"

    if base_type == "truck":
        return "vehicle", "truck"

    if base_type == "bus":
        return "vehicle", "bus"

    if base_type == "car" or actor.type_id in PASSENGER_CAR_FALLBACK:
        return "vehicle", "car"

    return None, None


def transform_point(matrix, point):
    p = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)
    result = matrix @ p
    return result[:3]


def transform_vector(matrix, vector):
    v = np.array([vector.x, vector.y, vector.z, 0.0], dtype=np.float64)
    result = matrix @ v
    return result[:3]


def normalize_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


class AnnotationWriter:
    def __init__(self, sequence_root, cfg):
        self.sequence_root = sequence_root
        self.cfg = cfg
        self.object_3d_dir = os.path.join(sequence_root, "labels", "object_3d")
        os.makedirs(self.object_3d_dir, exist_ok=True)

    def _get_target_actors(self, world, ego):
        targets = []

        for actor in world.get_actors():
            if actor.id == ego.id:
                continue

            category, subcategory = get_category(actor)

            if category is None:
                continue

            targets.append((actor, category, subcategory))

        return targets

    def _make_annotation(self, actor, category, subcategory, ego):
        ego_transform = ego.get_transform()
        actor_transform = actor.get_transform()
        bbox = actor.bounding_box

        world_to_ego = np.array(ego_transform.get_inverse_matrix(), dtype=np.float64)

        bbox_center_world = actor_transform.transform(bbox.location)
        center_ego = transform_point(world_to_ego, bbox_center_world)

        distance = math.sqrt(center_ego[0] ** 2 + center_ego[1] ** 2 + center_ego[2] ** 2)

        if distance > self.cfg.ANNOTATION.MAX_DISTANCE:
            return None

        velocity_world = actor.get_velocity()
        velocity_ego = transform_vector(world_to_ego, velocity_world)

        dimensions = [
            bbox.extent.x * 2.0,
            bbox.extent.y * 2.0,
            bbox.extent.z * 2.0
        ]

        object_yaw = actor_transform.rotation.yaw + bbox.rotation.yaw
        yaw_ego = normalize_angle(object_yaw - ego_transform.rotation.yaw)

        data = {
            "actor_id": actor.id,
            "type_id": actor.type_id,
            "category": category,
            "center_ego": center_ego.tolist(),
            "dimensions": dimensions,
            "yaw_ego": yaw_ego,
            "velocity_ego": velocity_ego.tolist(),
            "distance": distance
        }

        if subcategory is not None:
            data["subcategory"] = subcategory

        if category == "pedestrian":
            data["age"] = actor.attributes.get("age", "")
            data["gender"] = actor.attributes.get("gender", "")

        return data

    def write_frame(self, local_frame_id, world, ego):
        objects = []

        for actor, category, subcategory in self._get_target_actors(world, ego):
            annotation = self._make_annotation(actor, category, subcategory, ego)

            if annotation is not None:
                objects.append(annotation)

        frame_name = f"{local_frame_id:06d}"
        path = os.path.join(self.object_3d_dir, f"{frame_name}.json")

        data = {
            "frame_id": local_frame_id,
            "objects": objects
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        counts = {
            "pedestrian": 0,
            "vehicle": 0,
            "cyclist": 0,
            "motorcyclist": 0
        }

        for obj in objects:
            counts[obj["category"]] += 1

        return counts
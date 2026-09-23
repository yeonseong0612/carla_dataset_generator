import os
import json
import math

import carla
import numpy as np

from src.data.projection import CameraProjector
from src.data.collector import depth_to_numpy
from src.simulation.actor_lifecycle import is_stale_actor_error, require_ego_alive

SEMANTIC_MAP = {
    0: ("unlabelled", (0, 0, 0)),
    1: ("road", (128, 64, 0)),
    2: ("sidewalk", (244, 35, 232)),
    3: ("building", (70, 70, 70)),
    4: ("wall", (102, 102, 156)),
    5: ("fence", (190, 153, 153)),
    6: ("pole", (153, 153, 153)),
    7: ("traffic light", (250, 170, 30)),
    8: ("traffic sign", (220, 220, 0)),
    9: ("vegetation", (107, 142, 35)),
    10: ("terrain", (152, 251, 152)),
    11: ("sky", (70, 130, 180)),
    12: ("pedestrian", (220, 20, 60)),
    13: ("rider", (255, 0, 0)),
    14: ("car", (0, 0, 142)),
    15: ("truck", (0, 0, 70)),
    16: ("bus", (0, 60, 100)),
    17: ("train", (0, 80, 100)),
    18: ("motorcycle", (0, 0, 230)),
    19: ("bicycle", (119, 11, 32)),
    20: ("static", (110, 190, 160)),
    21: ("dynamic", (170, 120, 50)),
    22: ("other", (55, 90, 80)),
    23: ("water", (45, 60, 150)),
    24: ("road line", (157, 234, 50)),
    25: ("ground", (81, 0, 81)),
    26: ("bridge", (150, 100, 100)),
    27: ("rail track", (230, 150, 140)),
    28: ("guard rail", (180, 165, 180)),
}

# CARLA semantic tag -> our dataset taxonomy
SEMANTIC_TO_CATEGORY = {
    12: ("pedestrian", None),
    14: ("vehicle", "car"),
    15: ("vehicle", "truck"),
    16: ("vehicle", "bus"),
    18: ("motorcyclist", None),
    19: ("cyclist", None),
}

# Some CARLA vehicle blueprints may not expose a useful base_type.
PASSENGER_CAR_FALLBACK = {
    "vehicle.bmw.grandtourer",
    "vehicle.mini.cooper_s",
}

def transform_point(matrix, point):
    p = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)

    result = matrix @ p

    return result[:3]

def transform_vector(matrix, vector):
    v = np.array([vector.x, vector.y, vector.z, 0.0], dtype=np.float64)

    result = matrix @ v

    return result[:3]

def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def get_semantic_tags(actor):
    return [int(tag) for tag in getattr(actor, "semantic_tags", [])]

def get_semantic_name(actor):
    for tag in get_semantic_tags(actor):
        if tag in SEMANTIC_MAP:
            return SEMANTIC_MAP[tag][0]

    return "unknown"

def get_category(actor):
    # --------------------------------------------------------
    # Pedestrian
    # --------------------------------------------------------

    if actor.type_id.startswith("walker.pedestrian."):
        return "pedestrian", None

    # --------------------------------------------------------
    # Ignore non-vehicle/non-pedestrian actors
    # --------------------------------------------------------

    if not actor.type_id.startswith("vehicle."):
        return None, None

    base_type = actor.attributes.get("base_type", "").lower()

    # --------------------------------------------------------
    # Primary classification: CARLA base_type
    # --------------------------------------------------------

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

    if base_type == "car":
        return "vehicle", "car"

    # --------------------------------------------------------
    # Known blueprint fallback
    # --------------------------------------------------------

    if actor.type_id in PASSENGER_CAR_FALLBACK:
        return "vehicle", "car"

    # --------------------------------------------------------
    # Semantic-tag fallback
    # --------------------------------------------------------

    for semantic_tag in get_semantic_tags(actor):
        if semantic_tag in SEMANTIC_TO_CATEGORY:
            return SEMANTIC_TO_CATEGORY[semantic_tag]

    return None, None

def vehicle_light_state_to_dict(actor):
    state = actor.get_light_state()

    return {
        "position": bool(
            state & carla.VehicleLightState.Position
        ),
        "low_beam": bool(
            state & carla.VehicleLightState.LowBeam
        ),
        "high_beam": bool(
            state & carla.VehicleLightState.HighBeam
        ),
        "brake": bool(
            state & carla.VehicleLightState.Brake
        ),
        "reverse": bool(
            state & carla.VehicleLightState.Reverse
        ),
        "left_blinker": bool(
            state & carla.VehicleLightState.LeftBlinker
        ),
        "right_blinker": bool(
            state & carla.VehicleLightState.RightBlinker
        ),
        "fog": bool(
            state & carla.VehicleLightState.Fog
        ),
        "interior": bool(
            state & carla.VehicleLightState.Interior
        ),
        "special1": bool(
            state & carla.VehicleLightState.Special1
        ),
        "special2": bool(
            state & carla.VehicleLightState.Special2
        ),
    }


class AnnotationWriter:
    def __init__(self, sequence_root, cfg, ego=None, left_camera_actor=None, logical_id_resolver=None):
        """
        ego, left_camera_actor: if both are given, camera-valid filtering
        (FOV/truncation, depth-based occlusion, minimum pixel size -- see
        _compute_camera_validity) is applied and write_frame() requires
        a depth_raw argument. left_camera_actor is always "rgb_left" --
        stereo/right-camera validity is not evaluated (task: annotation-
        valid is judged from the left camera only). If either is None,
        AnnotationWriter falls back to its pre-filtering behavior
        (distance-only, no camera_valid/invalid_reason fields) so
        existing callers that don't pass a camera keep working.
        """

        self.sequence_root = sequence_root
        self.cfg = cfg

        # Optional callable: canonical CARLA actor id -> persistent logical
        # id (src/data/world_state.py WorldStateRecorder.logical_id_for).
        # When given, every annotation carries "logical_id" so labels can be
        # joined to geometry/world_state across weather replays.
        self.logical_id_resolver = logical_id_resolver

        self.object_3d_dir = os.path.join(sequence_root, "labels", "object_3d")

        os.makedirs(self.object_3d_dir, exist_ok=True)

        self.projector = None

        if ego is not None and left_camera_actor is not None:
            self.projector = CameraProjector.from_live_actors(ego, left_camera_actor)

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

    def _compute_camera_validity(self, vertices_ego_m, depth_m):
        """
        FOV/truncation + depth-based occlusion + minimum pixel-size
        filtering against self.projector (left RGB camera only). Never
        raises on a degenerate projection -- camera-behind points,
        zero-area boxes, etc. all just resolve to camera_valid=False
        with an invalid_reason instead ("카메라 뒤쪽 point는 ... robust하게
        처리한다").

        image_fraction: (2D-bbox area, clipped to the image) / (2D-bbox
        area, unclipped) -- computed from the projected vertices' own
        axis-aligned extent (task section 2), not a convex hull.

        visible_fraction (occlusion, section 3): reuses the depth camera
        already in the sensor rig (co-located + pixel-aligned with
        rgb_left -- see src/sensors/camera.py, same transform/width/
        height/fov for both). A pixel inside the clipped 2D bbox is
        counted "visible" if the depth image's value there falls within
        the object's own projected depth range -- not nearer than its
        closest projected vertex, and not farther than its farthest
        projected vertex (both with a small tolerance for depth-buffer/
        mesh-vs-bbox slack) -- i.e. nothing closer than the object
        itself is rendered at that pixel, AND the pixel isn't just open
        background beyond the object leaking into its (rectangular,
        not silhouette-shaped) bbox. The upper bound matters because
        the bbox is axis-aligned, not a silhouette (see below): without
        it, a far object's small bbox sitting mostly on empty background
        past it (e.g. a wall, sidewalk, or open street strictly farther
        than the object's own far vertex) has that background counted
        as "the object visible", which does not agree with the actual
        rendered image. This is a simple, standard depth-threshold
        heuristic (not a per-pixel object mask -- CARLA's semantic
        segmentation camera is class-level only, not per-instance, so
        it can't distinguish "this actor" from another same-class actor
        and isn't a better fit here), applied over the bbox rectangle
        itself since no finer per-pixel silhouette is available; this
        does mean a loose/rectangular bbox around a visually thin
        object (e.g. a pedestrian) can under-count its own
        visible_fraction, a known limitation of this simple approach
        rather than a bug.
        """

        metrics = {
            "image_fraction": 0.0,
            "visible_fraction": 0.0,
            "bbox_width_px": 0,
            "bbox_height_px": 0,
            "visible_area_px": 0,
            "camera_valid": False,
            "invalid_reason": "projection_invalid",
        }

        v_ego = np.asarray(vertices_ego_m, dtype=np.float64)

        if v_ego.shape != (8, 3) or not np.all(np.isfinite(v_ego)):
            return metrics

        v_cv = self.projector.ego_to_cv(v_ego)
        uv, valid = self.projector.project(v_cv)

        if not np.any(valid):
            metrics["invalid_reason"] = "outside_fov"
            return metrics

        u_valid, v_valid = uv[valid, 0], uv[valid, 1]
        u_min, u_max = float(u_valid.min()), float(u_valid.max())
        v_min, v_max = float(v_valid.min()), float(v_valid.max())
        unclipped_area = max(u_max - u_min, 0.0) * max(v_max - v_min, 0.0)

        width, height = self.projector.width, self.projector.height
        clip_u0, clip_u1 = min(max(u_min, 0.0), width), min(max(u_max, 0.0), width)
        clip_v0, clip_v1 = min(max(v_min, 0.0), height), min(max(v_max, 0.0), height)
        intersect_w, intersect_h = max(clip_u1 - clip_u0, 0.0), max(clip_v1 - clip_v0, 0.0)
        intersection_area = intersect_w * intersect_h

        if unclipped_area <= 0.0 or intersection_area <= 0.0:
            metrics["invalid_reason"] = "outside_fov"
            return metrics

        image_fraction = intersection_area / unclipped_area
        metrics["image_fraction"] = float(image_fraction)

        px0, py0 = int(math.floor(clip_u0)), int(math.floor(clip_v0))
        px1, py1 = int(math.ceil(clip_u1)), int(math.ceil(clip_v1))
        px0, py0 = max(px0, 0), max(py0, 0)
        px1, py1 = min(px1, width), min(py1, height)
        bbox_width_px, bbox_height_px = max(px1 - px0, 0), max(py1 - py0, 0)
        metrics["bbox_width_px"] = int(bbox_width_px)
        metrics["bbox_height_px"] = int(bbox_height_px)

        if bbox_width_px <= 0 or bbox_height_px <= 0:
            metrics["invalid_reason"] = "too_truncated"
            return metrics

        near_depth = float(v_cv[valid, 2].min())
        far_depth = float(v_cv[valid, 2].max())
        depth_tolerance_m = 0.5

        depth_patch = depth_m[py0:py1, px0:px1]
        visible_mask = (
            (depth_patch >= (near_depth - depth_tolerance_m))
            & (depth_patch <= (far_depth + depth_tolerance_m))
        )

        projected_pixel_count = depth_patch.size
        visible_pixel_count = int(visible_mask.sum())
        visible_fraction = visible_pixel_count / projected_pixel_count if projected_pixel_count > 0 else 0.0
        metrics["visible_fraction"] = float(visible_fraction)
        metrics["visible_area_px"] = visible_pixel_count

        camera_valid = (
            image_fraction >= self.cfg.ANNOTATION.MIN_IMAGE_FRACTION
            and visible_fraction >= self.cfg.ANNOTATION.MIN_VISIBLE_FRACTION
            and bbox_width_px >= self.cfg.ANNOTATION.MIN_BBOX_WIDTH_PX
            and bbox_height_px >= self.cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX
            and visible_pixel_count >= self.cfg.ANNOTATION.MIN_VISIBLE_AREA_PX
        )
        metrics["camera_valid"] = camera_valid

        if camera_valid:
            metrics["invalid_reason"] = None
        elif image_fraction < self.cfg.ANNOTATION.MIN_IMAGE_FRACTION:
            metrics["invalid_reason"] = "too_truncated"
        elif visible_fraction < self.cfg.ANNOTATION.MIN_VISIBLE_FRACTION:
            metrics["invalid_reason"] = "too_occluded"
        else:
            metrics["invalid_reason"] = "too_small"

        return metrics

    def _make_annotation(self, actor, category, subcategory, ego, depth_m=None):

        ego_transform = ego.get_transform()
        actor_transform = actor.get_transform()

        bbox = actor.bounding_box

        # World -> ego transformation
        world_to_ego = np.array(ego_transform.get_inverse_matrix(), dtype=np.float64,)

        # carla.Transform.transform() mutates its argument in place, so
        # bbox.location itself must never be passed directly: doing so
        # silently rewrites bbox.location from local to world coordinates,
        # which then corrupts the subsequent bbox.get_world_vertices() call
        # (it would apply actor_transform a second time on top of an
        # already-world-frame location).
        bbox_location_local = carla.Location(
            x=bbox.location.x,
            y=bbox.location.y,
            z=bbox.location.z,
        )

        bbox_center_world = actor_transform.transform(bbox_location_local)

        center_ego = transform_point(world_to_ego, bbox_center_world)

        distance_m = float(np.linalg.norm(center_ego))

        if distance_m > self.cfg.ANNOTATION.MAX_DISTANCE:
            return None

        dimensions_m = {
            "length": float(bbox.extent.x * 2.0),
            "width": float(bbox.extent.y * 2.0),
            "height": float(bbox.extent.z * 2.0),
        }

      
        object_yaw_world_deg = actor_transform.rotation.yaw + bbox.rotation.yaw
    

        yaw_ego_deg = normalize_angle_deg(object_yaw_world_deg - ego_transform.rotation.yaw)

        vertices_world = bbox.get_world_vertices(actor_transform)

        vertices_ego_m = [transform_point( world_to_ego, vertex).tolist() for vertex in vertices_world]

        actor_velocity_world = actor.get_velocity()
        ego_velocity_world = ego.get_velocity()

        relative_velocity_world = carla.Vector3D(
            x=(actor_velocity_world.x - ego_velocity_world.x),
            y=(actor_velocity_world.y - ego_velocity_world.y),
            z=(actor_velocity_world.z - ego_velocity_world.z)
        )

        relative_velocity_ego = transform_vector(world_to_ego, relative_velocity_world,)

        semantic_tags = get_semantic_tags(actor)
        semantic_name = get_semantic_name(actor)

        data = {
            "actor_id": int(actor.id),
            "type_id": actor.type_id,

            "semantic_tags": semantic_tags,
            "semantic_name": semantic_name,

            "category": category,

            "bbox_3d": {
                "center_ego_m": center_ego.tolist(),
                "dimensions_m": dimensions_m,
                "yaw_ego_deg": float(yaw_ego_deg),
                "vertices_ego_m": vertices_ego_m,
            },

            "relative_velocity_ego_mps":
                relative_velocity_ego.tolist(),

            "distance_m": distance_m,
        }


        if self.logical_id_resolver is not None:
            data["logical_id"] = self.logical_id_resolver(actor.id)

        if subcategory is not None:
            data["subcategory"] = subcategory

        if actor.type_id.startswith("vehicle."):
            data["light_state"] = (vehicle_light_state_to_dict(actor))
  
        if category == "pedestrian":
            data["age"] = actor.attributes.get("age", "")

            data["gender"] = actor.attributes.get("gender", "")

        # Camera-valid filtering (left RGB camera only -- see __init__ /
        # _compute_camera_validity). Only evaluated when this writer was
        # constructed with a camera (self.projector is not None) and the
        # caller supplied this frame's depth image; older/other callers
        # that don't pass a camera keep the pre-filtering behavior
        # (every distance-filtered actor is annotation-valid).
        if self.projector is not None and depth_m is not None:
            camera_metrics = self._compute_camera_validity(vertices_ego_m, depth_m)
            data["camera_projection"] = camera_metrics
        else:
            data["camera_projection"] = {
                "image_fraction": None, "visible_fraction": None,
                "bbox_width_px": None, "bbox_height_px": None, "visible_area_px": None,
                "camera_valid": True, "invalid_reason": None,
            }

        return data

    def write_frame(self, local_frame_id, world, ego, depth_raw=None):
        """
        depth_raw: this frame's raw CARLA depth-camera image (e.g.
        packet["depth"]), required for occlusion filtering whenever this
        writer was constructed with a camera (self.projector is not
        None). Decoded once per frame via
        src.data.collector.depth_to_numpy (reused, not re-implemented).
        """

        require_ego_alive(ego, f"annotation frame {local_frame_id}")

        depth_m = depth_to_numpy(depth_raw) if (self.projector is not None and depth_raw is not None) else None

        raw_candidates = []  # every distance-filtered actor, before camera-valid filtering
        objects = []         # final annotation list -- camera-valid only (task section 5)
        rejected_objects = []  # camera-valid == False, kept for diagnostics/visualization only

        targets = self._get_target_actors(world, ego)

        stale_actor_omissions = []

        for actor, category, subcategory in targets:
            # A background actor that left the server registry after the
            # snapshot (its get_light_state() RPC fails) is dropped from this
            # frame as a whole -- never a partial annotation. Ego reads in
            # _make_annotation are snapshot-local; if one still fails the ego
            # is gone, which require_ego_alive turns into a fatal error.
            try:
                annotation = self._make_annotation(actor, category, subcategory, ego, depth_m)
            except RuntimeError as exc:
                if not is_stale_actor_error(exc):
                    raise
                require_ego_alive(ego, f"annotation frame {local_frame_id}")
                stale_actor_omissions.append(int(actor.id))
                print(
                    f"[Annotation] frame={local_frame_id} omitted stale "
                    f"background actor id={actor.id} ({category})"
                )
                continue

            if annotation is None:
                continue

            raw_candidates.append(annotation)

            if annotation["camera_projection"]["camera_valid"]:
                objects.append(annotation)
            else:
                rejected_objects.append(annotation)

        snapshot = world.get_snapshot()

        carla_frame = int(snapshot.frame)
        timestamp = float(snapshot.timestamp.elapsed_seconds)

        frame_name = f"{local_frame_id:06d}"

        path = os.path.join(self.object_3d_dir, f"{frame_name}.json")

        data = {
            "frame_id": int(local_frame_id),
            "carla_frame": carla_frame,
            "timestamp": timestamp,
            "coordinate_system": {
                "frame": "ego",
                "x": "forward",
                "y": "right",
                "z": "up",
            },
            # Final annotation list -- camera_valid actors only (task
            # section 5: "camera_valid == False인 actor는 최종 annotation
            # list에서 제외한다"). This is what get_annotation_candidate_
            # count()'s caller sums (see the counts dict below) -- when
            # this writer was built with a camera, "objects" already IS
            # the camera-valid annotation count Gamma now reads.
            "objects": objects,
            # Diagnostics only -- NOT part of the final annotation list.
            # Kept so validation/visualization tooling doesn't need to
            # re-run projection to see why something was excluded.
            "rejected_objects": rejected_objects,
        }

        # Only present when non-empty (see world_state.py record_frame).
        if stale_actor_omissions:
            data["stale_actor_omissions"] = stale_actor_omissions

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


        counts = {"pedestrian": 0, "vehicle": 0, "cyclist": 0, "motorcyclist": 0}

        for obj in objects:
            category = obj["category"]

            if category in counts:
                counts[category] += 1

        return counts
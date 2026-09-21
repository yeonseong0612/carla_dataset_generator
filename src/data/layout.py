"""
layout.py

Route-centric paired dataset layout (pure path/marker helpers, no CARLA).

    dataset/{town}/route_{id}/
        geometry/                 canonical geometry, stored once
        conditions/{condition}/   weather-specific data (rgb_left / rgb_right)

Geometry is generated once per (town, route); every weather condition is
rendered from that exact recorded geometry (see CLAUDE.md).
"""

import json
import os
import time


GEOMETRY_DIRNAME = "geometry"
CONDITIONS_DIRNAME = "conditions"
COMPLETE_MARKER = "COMPLETE"

# Modalities stored once under geometry/ (shared by every condition).
GEOMETRY_SENSOR_EXTENSIONS = {
    "depth": ".npy",
    "optical_flow": ".npy",
    "semantic": ".npy",
    "lidar": ".npy",
    "radar": ".npy",
    "radar_front_left": ".npy",
    "radar_front_right": ".npy",
}

# Modalities stored per weather condition.
CONDITION_SENSOR_EXTENSIONS = {
    "rgb_left": ".png",
    "rgb_right": ".png",
}


def route_root(output_root, town, route_id):
    return os.path.join(output_root, town, f"route_{route_id}")


def geometry_dir(route_root_path):
    return os.path.join(route_root_path, GEOMETRY_DIRNAME)


def condition_dir(route_root_path, condition):
    return os.path.join(route_root_path, CONDITIONS_DIRNAME, condition)


def resolve_geometry_root(sequence_path):
    """
    Map any path a tool may be given to the directory holding the shared
    geometry (calibration.json, labels/, depth/, lidar/, pose/, ...):

        .../route_x/conditions/<cond>  -> .../route_x/geometry
        .../route_x/geometry           -> itself
        .../route_x                    -> .../route_x/geometry
        anything else (flat layout)    -> itself
    """

    path = os.path.abspath(sequence_path)
    parent = os.path.dirname(path)

    if os.path.basename(parent) == CONDITIONS_DIRNAME:
        return os.path.join(os.path.dirname(parent), GEOMETRY_DIRNAME)

    if os.path.basename(path) == GEOMETRY_DIRNAME:
        return path

    candidate = os.path.join(path, GEOMETRY_DIRNAME)

    if os.path.isdir(candidate):
        return candidate

    return path


def route_root_of(sequence_path):
    """
    The route directory that owns a condition/geometry directory (used by
    tools that delete a collected run); a flat legacy directory is its own
    root.
    """

    path = os.path.abspath(sequence_path)
    parent = os.path.dirname(path)

    if os.path.basename(parent) == CONDITIONS_DIRNAME:
        return os.path.dirname(parent)

    if os.path.basename(path) == GEOMETRY_DIRNAME:
        return parent

    return path


def resolve_condition_root(sequence_path, default_condition="day_clear"):
    """
    Map a tool-supplied path to the directory holding RGB images.

        .../route_x/conditions/<cond> -> itself
        .../route_x/geometry          -> .../route_x/conditions/<default>
        .../route_x                   -> .../route_x/conditions/<default>
        anything else (flat layout)   -> itself
    """

    path = os.path.abspath(sequence_path)
    parent = os.path.dirname(path)

    if os.path.basename(parent) == CONDITIONS_DIRNAME:
        return path

    if os.path.basename(path) == GEOMETRY_DIRNAME:
        return os.path.join(os.path.dirname(path), CONDITIONS_DIRNAME, default_condition)

    candidate = os.path.join(path, CONDITIONS_DIRNAME, default_condition)

    if os.path.isdir(candidate):
        return candidate

    return path


# ------------------------------------------------------------------
# Completion markers (condition-level resume)
# ------------------------------------------------------------------

def is_complete(directory):
    return os.path.isfile(os.path.join(directory, COMPLETE_MARKER))


def mark_complete(directory, info=None):
    os.makedirs(directory, exist_ok=True)

    payload = {"completed_at_unix": time.time()}

    if info:
        payload.update(info)

    with open(os.path.join(directory, COMPLETE_MARKER), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clear_complete(directory):
    path = os.path.join(directory, COMPLETE_MARKER)

    if os.path.isfile(path):
        os.remove(path)

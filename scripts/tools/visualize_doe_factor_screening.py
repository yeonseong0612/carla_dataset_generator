"""
scripts/tools/visualize_doe_factor_screening.py

DOE (Design of Experiments) factor/level SCREENING tool for the 3
weather factors used in the D-optimal 52-condition design
(D_optimal_52_with_baseline_(...).csv):

    Rain          : Dry / Light / Moderate / Heavy
    Fog           : Clear / Light / Moderate / Heavy
    Illumination  : sun_altitude_angle = 60 / 30 / 0 / -15 / -30

Purpose (see CLAUDE.md): NOT to run ANOVA, NOT to recompute the
D-optimal design, NOT to auto-delete any factor/level. This tool only
produces One-Factor-at-a-Time (OFAT) visual comparisons and a set of
quantitative image-difference metrics so a human can decide whether any
level should be merged/dropped before the design is finalized.

Standalone tool: does not import or modify scripts/collect_dataset.py or
anything that would change production dataset-generation behavior. It
DOES reuse a few pieces of already-existing, non-production-mutating
code for consistency with the real pipeline:
    - CFG.config.cfg                          (camera mount/size, sim FPS)
    - src.navigation.route                    (route XML -> dense route)
    - src.simulation.weather.WEATHER_WIND_INTENSITY
                                               (production wind=0 policy)
Everything else (weather level definitions, capture loop, metrics,
figures) is new and lives only in this file.

Assumes a CARLA 0.9.16 server is ALREADY RUNNING on the target town
(this script only connects to it, per user instruction -- it never
launches or reloads CARLA.exe/the world unless --force-load-world is
passed explicitly).

--------------------------------------------------------------------
Same-scene guarantee
--------------------------------------------------------------------
For a given frame index, the ego vehicle is teleported once (physics
disabled, autopilot off) to a fixed route waypoint and never moves again
while every level of a factor is swept through -- only carla.WeatherParameters
changes between captures at that frame. The camera is rigidly attached
to the ego with the same relative transform used by the production
sensor rig (CFG.SENSOR.CAMERA). No traffic/pedestrians are spawned by
this script. This guarantees identical geometry across all levels of a
factor at a given frame, and across the 3 factors at the same frame
(they share the same baseline-fixed-other-factors weather).

--------------------------------------------------------------------
Dependency policy (CLAUDE.md section 6)
--------------------------------------------------------------------
Only numpy / OpenCV / matplotlib (all already installed in this
project's `carla` conda env) are used. SSIM is a small local
reimplementation (scikit-image is NOT installed here -- see below).
LPIPS is NOT computed (torch/lpips not installed) -- its column is left
as None/empty everywhere, never silently omitted, so its absence is
visible in every metrics file rather than installing a new heavy
dependency to satisfy an "if possible" requirement.
"""

import argparse
import csv
import json
import queue
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"
sys.path.insert(0, str(CARLA_PYTHONAPI))

import carla  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CFG.config import cfg  # noqa: E402
from src.navigation.route import (  # noqa: E402
    load_route_from_xml,
    project_control_points,
    build_dense_route,
)
from src.simulation.weather import WEATHER_WIND_INTENSITY  # noqa: E402


# ====================================================================
# Fixed config (edit here, not via new CLI plumbing, per repo convention
# -- see scripts/visualize_doe_factors.py's same "config-at-top" style)
# ====================================================================

DEFAULT_ROUTE_XML = PROJECT_ROOT / "routes" / "Town10.xml"
DEFAULT_ROUTE_ID = "0"
DEFAULT_TOWN_SUBSTRING = "Town10HD"          # matches Town10HD / Town10HD_Opt

DEFAULT_DESIGN_CSV = (
    PROJECT_ROOT / "dataset" / "doe_input" / "D_optimal_52_with_baseline_(최종).csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doe_factor_screening"

ROUTE_SAMPLING_RESOLUTION = 2.0              # same value collect_dataset.py uses
DEFAULT_FRAME_PERCENTAGES = [20, 40, 60, 80]  # CLAUDE.md section 8 example

FIXED_DELTA_SECONDS = float(cfg.SIMULATION.FIXED_DELTA_SECONDS)
DEFAULT_WARMUP_TICKS = 8
TELEPORT_SETTLE_TICKS = 3

EGO_BLUEPRINT = "vehicle.tesla.model3"
EGO_Z_LIFT = 0.3   # same convention as scripts/collect_dataset.py spawn_ego_at_route_start

# Camera mount reused verbatim from the production sensor config so the
# screened appearance matches what the real front RGB camera would see.
CAMERA_WIDTH = int(cfg.SENSOR.CAMERA.WIDTH)
CAMERA_HEIGHT = int(cfg.SENSOR.CAMERA.HEIGHT)
CAMERA_FOV = float(cfg.SENSOR.CAMERA.FOV)
CAMERA_LOCATION = carla.Location(x=float(cfg.SENSOR.CAMERA.X), y=0.0, z=float(cfg.SENSOR.CAMERA.Z))
CAMERA_ROTATION = carla.Rotation(
    pitch=float(cfg.SENSOR.CAMERA.PITCH), yaw=float(cfg.SENSOR.CAMERA.YAW), roll=float(cfg.SENSOR.CAMERA.ROLL)
)

# Weather fields that are NOT one of the 3 screened factors, held fixed
# for every single capture in this whole tool. wind_intensity is pulled
# directly from src/simulation/weather.py's production policy constant
# (CLAUDE.md section 3: "wind_intensity는 모든 조건에서 0.0으로 고정") instead
# of being redefined here. cloudiness / sun_azimuth values match the
# constants already used for this exact same screening purpose in
# scripts/visualize_doe_factors.py, kept identical for consistency
# between the two tools.
CLOUDINESS_FIXED = 10.0
SUN_AZIMUTH_FIXED = 0.0

assert WEATHER_WIND_INTENSITY == 0.0, (
    "src.simulation.weather.WEATHER_WIND_INTENSITY changed from the 0.0 "
    "production policy this screening tool assumes -- CLAUDE.md section 3 "
    "requires wind_intensity=0.0 for every screened condition."
)

# --------------------------------------------------------------------
# Level definitions. Rain/Fog values match scripts/visualize_doe_factors.py's
# RAIN_PROFILES/FOG_PROFILES (same names, same 4-level scheme as the
# D-optimal CSV's Rain/Fog columns) so both screening tools describe the
# same physical conditions. Illumination values are exactly the D-optimal
# CSV's Time levels.
# --------------------------------------------------------------------

RAIN_LEVELS = OrderedDict([
    ("Dry",      dict(precipitation=0.0,  wetness=0.0,  precipitation_deposits=0.0)),
    ("Light",    dict(precipitation=25.0, wetness=30.0, precipitation_deposits=15.0)),
    ("Moderate", dict(precipitation=50.0, wetness=60.0, precipitation_deposits=40.0)),
    ("Heavy",    dict(precipitation=80.0, wetness=90.0, precipitation_deposits=70.0)),
])

FOG_LEVELS = OrderedDict([
    ("Clear",    dict(fog_density=0.0,  fog_distance=0.0,   fog_falloff=0.0)),
    ("Light",    dict(fog_density=25.0, fog_distance=100.0, fog_falloff=1.0)),
    ("Moderate", dict(fog_density=50.0, fog_distance=50.0,  fog_falloff=1.0)),
    ("Heavy",    dict(fog_density=75.0, fog_distance=20.0,  fog_falloff=1.0)),
])

ILLUM_LEVELS = OrderedDict([
    (60,  dict(sun_altitude_angle=60.0)),
    (30,  dict(sun_altitude_angle=30.0)),
    (0,   dict(sun_altitude_angle=0.0)),
    (-15, dict(sun_altitude_angle=-15.0)),
    (-30, dict(sun_altitude_angle=-30.0)),
])

FACTOR_LEVELS = OrderedDict([
    ("Rain", RAIN_LEVELS),
    ("Fog", FOG_LEVELS),
    ("Illumination", ILLUM_LEVELS),
])

# Which key(s) of `cfg` in build_weather's `fields` dict best summarize
# each factor's level, for figure captions.
def format_key_params(factor_name, fields):
    if factor_name == "Rain":
        return (f"precip={fields['precipitation']:.0f} "
                f"wet={fields['wetness']:.0f} dep={fields['precipitation_deposits']:.0f}")
    if factor_name == "Fog":
        return (f"density={fields['fog_density']:.0f} "
                f"dist={fields['fog_distance']:.0f} falloff={fields['fog_falloff']:.1f}")
    if factor_name == "Illumination":
        return f"sun_alt={fields['sun_altitude_angle']:.0f} deg"
    return ""


# Heuristic-only classification thresholds (CLAUDE.md section 12: this is
# a labeling aid for the human reviewer, NEVER an automatic
# removal decision). Primary signal = SSIM (perceptual similarity),
# corroborated by MAD in the report text.
SSIM_CLEARLY_DISTINCT_MAX = 0.85       # ssim <  this -> clearly distinct
SSIM_MODERATELY_DISTINCT_MAX = 0.95    # ssim <  this -> moderately distinct
SSIM_WEAKLY_DISTINCT_MAX = 0.985       # ssim <  this -> weakly distinct
                                        # ssim >= this -> visually near-duplicate


def classify_pair(ssim_value):
    """Heuristic-only bucket label -- see thresholds above. Never used to
    auto-drop anything; only written into report/summary text."""
    if ssim_value < SSIM_CLEARLY_DISTINCT_MAX:
        return "clearly distinct"
    if ssim_value < SSIM_MODERATELY_DISTINCT_MAX:
        return "moderately distinct"
    if ssim_value < SSIM_WEAKLY_DISTINCT_MAX:
        return "weakly distinct"
    return "visually near-duplicate (heuristic)"


# ====================================================================
# Weather construction
# ====================================================================

def build_weather(rain_level, fog_level, illum_level):
    fields = {
        "cloudiness": CLOUDINESS_FIXED,
        "wind_intensity": WEATHER_WIND_INTENSITY,
        "sun_azimuth_angle": SUN_AZIMUTH_FIXED,
    }
    fields.update(RAIN_LEVELS[rain_level])
    fields.update(FOG_LEVELS[fog_level])
    fields.update(ILLUM_LEVELS[illum_level])

    weather = carla.WeatherParameters(**{k: float(v) for k, v in fields.items()})
    return weather, fields


def build_overrides(factor_name, level_name, baseline_levels):
    """Returns (rain_level, fog_level, illum_level) for this factor/level,
    with the two non-swept factors held at the baseline CSV row's levels."""
    rain_level = level_name if factor_name == "Rain" else baseline_levels["Rain"]
    fog_level = level_name if factor_name == "Fog" else baseline_levels["Fog"]
    illum_level = level_name if factor_name == "Illumination" else baseline_levels["Illumination"]
    return rain_level, fog_level, illum_level


# ====================================================================
# D-optimal CSV
# ====================================================================

def read_design_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"D-optimal design CSV is empty: {path}")

    return rows


def get_baseline_levels(design_rows):
    baseline_rows = [r for r in design_rows if str(r["is_baseline"]).strip().lower() == "true"]

    if len(baseline_rows) != 1:
        raise ValueError(
            f"Expected exactly 1 is_baseline=True row in the D-optimal CSV, found {len(baseline_rows)}."
        )

    row = baseline_rows[0]

    return {
        "Rain": row["Rain"].strip(),
        "Fog": row["Fog"].strip(),
        "Illumination": int(float(row["Time"].strip())),
        "design_id": row["design_id"].strip(),
    }


def analyze_design_csv(design_rows):
    """CLAUDE.md section 14: level frequency + how many of the 52 designs
    would be affected if a level were removed. Read-only analysis -- never
    recomputes or mutates the D-optimal design itself."""

    factor_columns = {"Rain": "Rain", "Fog": "Fog", "Illumination": "Time"}
    n_total = len(design_rows)
    analysis = {}

    for factor_name, column in factor_columns.items():
        counts = OrderedDict()

        for level_name in FACTOR_LEVELS[factor_name]:
            counts[str(level_name)] = 0

        for row in design_rows:
            value = row[column].strip()

            if factor_name == "Illumination":
                value = str(int(float(value)))

            counts[value] = counts.get(value, 0) + 1

        analysis[factor_name] = {
            "n_total_designs": n_total,
            "level_counts": counts,
            "level_removal_impact": {
                level: {
                    "designs_containing_level": count,
                    "designs_remaining_if_removed": n_total - count,
                }
                for level, count in counts.items()
            },
        }

    return analysis


# ====================================================================
# Image metrics (numpy / OpenCV only -- see module docstring)
# ====================================================================

def to_luminance(bgr_image):
    """ITU-R BT.601 luma. Input is cv2-native BGR (uint8 or float)."""
    b = bgr_image[:, :, 0].astype(np.float64)
    g = bgr_image[:, :, 1].astype(np.float64)
    r = bgr_image[:, :, 2].astype(np.float64)
    return 0.114 * b + 0.587 * g + 0.299 * r


def ssim_gray(img_a_gray, img_b_gray, sigma=1.5, dynamic_range=255.0):
    """
    Structural similarity index between two single-channel float images,
    Gaussian-windowed (matches skimage.metrics.structural_similarity(...,
    gaussian_weights=True, sigma=1.5) numerically). scikit-image is NOT
    installed in this project's environment (see module docstring), so
    this is a small local reimplementation using only cv2.GaussianBlur.
    """

    a = img_a_gray.astype(np.float64)
    b = img_b_gray.astype(np.float64)

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    mu_a = cv2.GaussianBlur(a, (0, 0), sigma)
    mu_b = cv2.GaussianBlur(b, (0, 0), sigma)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (0, 0), sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (0, 0), sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (0, 0), sigma) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    return float((numerator / denominator).mean())


def hist_distance_bhattacharyya(img_a_bgr, img_b_bgr, bins=32):
    """Optional extra metric (CLAUDE.md section 6). Mean per-channel
    Bhattacharyya distance between normalized histograms; 0 = identical
    distributions, 1 = fully disjoint."""

    distances = []

    for channel in range(3):
        hist_a = cv2.calcHist([img_a_bgr], [channel], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img_b_bgr], [channel], None, [bins], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=1.0, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_b, hist_b, alpha=1.0, norm_type=cv2.NORM_L1)
        distances.append(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))

    return float(np.mean(distances))


def edge_density(img_bgr, low_threshold=100, high_threshold=200):
    """Optional extra metric (CLAUDE.md section 6). Fraction of Canny
    edge pixels -- a crude proxy for scene detail / visibility."""

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return float(np.count_nonzero(edges)) / edges.size


def compute_pair_metrics(img_a_bgr, img_b_bgr):
    """All required metrics (CLAUDE.md section 6) between two same-size
    BGR images, plus the optional histogram/edge extras. LPIPS is left as
    None -- torch/lpips are not installed in this environment and are not
    force-installed per the section-6 dependency policy."""

    a = img_a_bgr.astype(np.float64)
    b = img_b_bgr.astype(np.float64)
    diff = a - b

    lum_a = to_luminance(img_a_bgr)
    lum_b = to_luminance(img_b_bgr)

    edge_a = edge_density(img_a_bgr)
    edge_b = edge_density(img_b_bgr)

    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "luminance_a": float(lum_a.mean()),
        "luminance_b": float(lum_b.mean()),
        "luminance_diff": float(lum_b.mean() - lum_a.mean()),
        "contrast_a": float(lum_a.std()),
        "contrast_b": float(lum_b.std()),
        "contrast_diff": float(lum_b.std() - lum_a.std()),
        "ssim": ssim_gray(lum_a, lum_b),
        "hist_distance_bhattacharyya": hist_distance_bhattacharyya(img_a_bgr, img_b_bgr),
        "edge_density_a": edge_a,
        "edge_density_b": edge_b,
        "edge_density_diff": edge_b - edge_a,
        "lpips": None,  # optional dependency not installed -- see module docstring
    }


def region_split_thirds(img_bgr):
    """Row-wise thirds of a forward-facing camera frame: top ~ far/sky,
    middle ~ mid-range, bottom ~ near/road. Used for Fog's distance-band
    analysis (CLAUDE.md section 10)."""

    h = img_bgr.shape[0]

    return {
        "far": img_bgr[0:h // 3, :, :],
        "mid": img_bgr[h // 3: 2 * h // 3, :, :],
        "near": img_bgr[2 * h // 3:, :, :],
    }


def fog_regional_metrics(img_bgr, base_bgr):
    out = {}

    regions_img = region_split_thirds(img_bgr)
    regions_base = region_split_thirds(base_bgr)

    for region_name in ("far", "mid", "near"):
        lum = to_luminance(regions_img[region_name])
        lum_base = to_luminance(regions_base[region_name])
        diff = np.abs(regions_img[region_name].astype(np.float64) - regions_base[region_name].astype(np.float64))

        out[f"{region_name}_contrast"] = float(lum.std())
        out[f"{region_name}_contrast_diff_from_baseline"] = float(lum.std() - lum_base.std())
        out[f"{region_name}_mad_from_baseline"] = float(diff.mean())

    return out


def rain_extra_metrics(img_bgr, base_bgr):
    """Road-region (bottom half) brightness/edge-texture proxies for
    wetness/reflection/puddle visibility (CLAUDE.md section 9)."""

    h = img_bgr.shape[0]
    lower = img_bgr[h // 2:, :, :]
    lower_base = base_bgr[h // 2:, :, :]

    lum = to_luminance(lower)
    lum_base = to_luminance(lower_base)

    return {
        "road_region_brightness": float(lum.mean()),
        "road_region_brightness_diff_from_baseline": float(lum.mean() - lum_base.mean()),
        "road_region_edge_density": edge_density(lower),
        "road_region_edge_density_diff_from_baseline": edge_density(lower) - edge_density(lower_base),
    }


def illumination_extra_metrics(img_bgr):
    """CLAUDE.md section 11: overall luminance / dynamic range / dark
    region ratio / sky brightness / a visibility proxy."""

    lum = to_luminance(img_bgr)
    h = lum.shape[0]
    sky = lum[: max(1, int(h * 0.15)), :]
    lower_half = img_bgr[h // 2:, :, :]

    p1, p99 = np.percentile(lum, [1, 99])

    return {
        "overall_luminance": float(lum.mean()),
        "dynamic_range_p1_p99": float(p99 - p1),
        "dark_region_ratio": float(np.mean(lum < 30.0)),
        "sky_brightness": float(sky.mean()),
        "lower_half_edge_density": edge_density(lower_half),
    }


# ====================================================================
# CARLA connection / scene setup
# ====================================================================

def connect_carla(host, port, timeout):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def get_world(client, town_substring, allow_load, route_town_hint):
    world = client.get_world()
    current_map = world.get_map().name

    if town_substring in current_map:
        print(f"[World] using already-running map '{current_map}' (assumes CARLA was started by the user)")
        return world

    if not allow_load:
        raise RuntimeError(
            f"Currently loaded CARLA map is '{current_map}', which does not contain "
            f"'{town_substring}'. This tool does not reload the world by default "
            f"(it assumes an already-running server, per user instruction). Either "
            f"load '{route_town_hint}' yourself in the running server, pass "
            f"--town-substring matching your current map plus a --route-xml/--route-id "
            f"for that town, or pass --force-load-world to let this tool call "
            f"client.load_world() (slow, disruptive to any other client on this server)."
        )

    print(f"[World] '{current_map}' does not match '{town_substring}', loading (--force-load-world)...")
    return client.load_world(route_town_hint)


def setup_synchronous(world, fixed_delta_seconds):
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    return original_settings


def select_frames(world, route_xml, route_id, sampling_resolution, percentages):
    town, control_points = load_route_from_xml(route_xml, route_id)
    carla_map = world.get_map()
    control_waypoints = project_control_points(carla_map, control_points)
    dense_route = build_dense_route(carla_map, control_waypoints, sampling_resolution)

    frames = []

    for pct in percentages:
        index = int(round((pct / 100.0) * (len(dense_route) - 1)))
        index = max(0, min(index, len(dense_route) - 1))
        waypoint = dense_route[index][0]

        frames.append({
            "percentage": pct,
            "route_index": index,
            "route_total_points": len(dense_route),
            "transform": waypoint.transform,
            "location": {
                "x": waypoint.transform.location.x,
                "y": waypoint.transform.location.y,
                "z": waypoint.transform.location.z,
            },
        })

    return frames, town


def spawn_ego(world, transform):
    blueprint = world.get_blueprint_library().find(EGO_BLUEPRINT)

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "doe_screening")

    spawn_transform = carla.Transform(
        carla.Location(transform.location.x, transform.location.y, transform.location.z + EGO_Z_LIFT),
        transform.rotation,
    )

    ego = world.try_spawn_actor(blueprint, spawn_transform)

    if ego is None:
        raise RuntimeError(f"Failed to spawn ego ({EGO_BLUEPRINT}) at {spawn_transform}")

    # Same-scene guarantee: nothing may move the ego across a whole sweep.
    ego.set_simulate_physics(False)
    ego.apply_control(carla.VehicleControl())

    return ego


def teleport_ego(ego, transform):
    lifted = carla.Transform(
        carla.Location(transform.location.x, transform.location.y, transform.location.z + EGO_Z_LIFT),
        transform.rotation,
    )
    ego.set_transform(lifted)


def spawn_camera(world, ego):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
    blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    blueprint.set_attribute("fov", str(CAMERA_FOV))

    camera_transform = carla.Transform(CAMERA_LOCATION, CAMERA_ROTATION)
    camera = world.spawn_actor(blueprint, camera_transform, attach_to=ego)

    return camera


def wait_for_frame(world, image_queue, ticks, timeout=10.0):
    """Ticks `ticks` times, draining each tick's frame so the queue never
    builds latency and the next capture_image() always starts empty."""

    for _ in range(ticks):
        frame_id = world.tick()
        image = image_queue.get(timeout=timeout)

        if image.frame != frame_id:
            raise RuntimeError(
                f"Warmup frame mismatch: ticked {frame_id}, got queued frame {image.frame}."
            )


def capture_image(world, image_queue, out_path, timeout=10.0):
    frame_id = world.tick()
    image = image_queue.get(timeout=timeout)

    if image.frame != frame_id:
        raise RuntimeError(f"Capture frame mismatch: ticked {frame_id}, got queued frame {image.frame}.")

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = array[:, :, :3]  # CARLA raw_data is BGRA
    cv2.imwrite(str(out_path), bgr)

    return frame_id


def cleanup(world, original_settings, original_weather, camera, ego):
    if camera is not None:
        try:
            camera.stop()
        except RuntimeError:
            pass
        try:
            if camera.is_alive:
                camera.destroy()
        except RuntimeError:
            pass

    if ego is not None:
        try:
            if ego.is_alive:
                ego.destroy()
        except RuntimeError:
            pass

    if world is not None:
        if original_weather is not None:
            try:
                world.set_weather(original_weather)
            except RuntimeError as exc:
                print(f"[Cleanup] failed to restore original weather: {exc}")

        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError as exc:
                print(f"[Cleanup] failed to restore world settings: {exc}")


# ====================================================================
# Capture loop
# ====================================================================

def run_captures(world, ego, camera, image_queue, frames, baseline_levels, output_dir, warmup_ticks):
    """
    Returns factor_records: {factor_name: [per-frame dict, ...]}, where
    each per-frame dict has:
        frame_index, frame_percentage,
        images: {level_name: Path}, fields: {level_name: weather-fields-dict}
    """

    factor_records = {factor_name: [] for factor_name in FACTOR_LEVELS}

    for frame_index, frame in enumerate(frames):
        print(f"[Frame {frame_index}] route {frame['percentage']}% "
              f"(index {frame['route_index']}/{frame['route_total_points']}) -- teleporting ego")

        teleport_ego(ego, frame["transform"])
        wait_for_frame(world, image_queue, TELEPORT_SETTLE_TICKS)

        for factor_name, levels in FACTOR_LEVELS.items():
            frame_dir = output_dir / factor_name.lower() / "frames" / f"frame{frame_index:02d}"
            frame_dir.mkdir(parents=True, exist_ok=True)

            level_images = {}
            level_fields = {}

            for level_name in levels:
                rain_level, fog_level, illum_level = build_overrides(factor_name, level_name, baseline_levels)
                weather, fields = build_weather(rain_level, fog_level, illum_level)
                world.set_weather(weather)

                wait_for_frame(world, image_queue, warmup_ticks)

                safe_name = str(level_name).replace("-", "neg").replace(" ", "_")
                out_path = frame_dir / f"{safe_name}.png"
                capture_image(world, image_queue, out_path)

                level_images[level_name] = out_path
                level_fields[level_name] = fields

                print(f"    [{factor_name}] {level_name}: {format_key_params(factor_name, fields)}  -> {out_path.name}")

            factor_records[factor_name].append({
                "frame_index": frame_index,
                "frame_percentage": frame["percentage"],
                "images": level_images,
                "fields": level_fields,
            })

    return factor_records


# ====================================================================
# Metrics tables
# ====================================================================

def compute_factor_tables(factor_name, frame_records, baseline_level):
    level_names = list(FACTOR_LEVELS[factor_name].keys())

    baseline_diff_rows = []
    adjacent_rows = []
    extra_rows = []

    for record in frame_records:
        images = {name: cv2.imread(str(path)) for name, path in record["images"].items()}

        for name, img in images.items():
            if img is None:
                raise RuntimeError(f"Failed to read captured image: {record['images'][name]}")

        base_img = images[baseline_level]

        for level_name in level_names:
            m = compute_pair_metrics(base_img, images[level_name])

            baseline_diff_rows.append({
                "factor": factor_name,
                "frame_index": record["frame_index"],
                "frame_percentage": record["frame_percentage"],
                "level": level_name,
                "is_baseline_level": level_name == baseline_level,
                **m,
            })

            if factor_name == "Illumination":
                extra_rows.append({
                    "frame_index": record["frame_index"],
                    "level": level_name,
                    **illumination_extra_metrics(images[level_name]),
                })
            elif factor_name == "Rain":
                extra_rows.append({
                    "frame_index": record["frame_index"],
                    "level": level_name,
                    **rain_extra_metrics(images[level_name], base_img),
                })
            elif factor_name == "Fog":
                extra_rows.append({
                    "frame_index": record["frame_index"],
                    "level": level_name,
                    **fog_regional_metrics(images[level_name], base_img),
                })

        for level_a, level_b in zip(level_names, level_names[1:]):
            pm = compute_pair_metrics(images[level_a], images[level_b])

            adjacent_rows.append({
                "factor": factor_name,
                "frame_index": record["frame_index"],
                "frame_percentage": record["frame_percentage"],
                "level_a": level_a,
                "level_b": level_b,
                **pm,
            })

    return baseline_diff_rows, adjacent_rows, extra_rows


def aggregate_rows(rows, group_keys, value_keys):
    """Mean/std of `value_keys` grouped by `group_keys` (across frames).
    None values (LPIPS) are skipped."""

    groups = OrderedDict()

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    aggregated = []

    for key, group_rows in groups.items():
        out = dict(zip(group_keys, key))
        out["n_frames"] = len(group_rows)

        for value_key in value_keys:
            values = [r[value_key] for r in group_rows if r.get(value_key) is not None]

            if values:
                out[f"{value_key}_mean"] = float(np.mean(values))
                out[f"{value_key}_std"] = float(np.std(values))
            else:
                out[f"{value_key}_mean"] = None
                out[f"{value_key}_std"] = None

        aggregated.append(out)

    return aggregated


METRIC_VALUE_KEYS = [
    "mean_abs_diff", "rmse", "luminance_diff", "contrast_diff", "ssim",
    "hist_distance_bhattacharyya", "edge_density_diff",
]


# ====================================================================
# Figures
# ====================================================================

def render_comparison_grid(factor_name, frame_records, baseline_level, out_path):
    level_names = list(FACTOR_LEVELS[factor_name].keys())
    n_rows = len(frame_records)
    n_cols = len(level_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    for row, record in enumerate(frame_records):
        for col, level_name in enumerate(level_names):
            ax = axes[row][col]
            img_bgr = cv2.imread(str(record["images"][level_name]))
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])

            title = str(level_name) + (" (baseline)" if level_name == baseline_level else "")
            caption = format_key_params(factor_name, record["fields"][level_name])
            ax.set_title(f"{title}\n{caption}", fontsize=8)

            if col == 0:
                ax.set_ylabel(f"route {record['frame_percentage']}%", fontsize=9)

    fig.suptitle(
        f"{factor_name} OFAT screening -- baseline level = {baseline_level} "
        f"(other 2 factors held at baseline)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[Figure] {out_path}")


def render_diff_grid(factor_name, frame_records, baseline_level, out_path, vmax=80.0):
    level_names = list(FACTOR_LEVELS[factor_name].keys())
    n_rows = len(frame_records)
    n_cols = len(level_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)
    im = None

    for row, record in enumerate(frame_records):
        base_img = cv2.imread(str(record["images"][baseline_level])).astype(np.float64)

        for col, level_name in enumerate(level_names):
            ax = axes[row][col]
            img = cv2.imread(str(record["images"][level_name])).astype(np.float64)
            diff = np.abs(img - base_img).mean(axis=2)

            im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            title = str(level_name) + (" (baseline, diff=0)" if level_name == baseline_level else "")
            ax.set_title(title, fontsize=8)

            if col == 0:
                ax.set_ylabel(f"route {record['frame_percentage']}%", fontsize=9)

    fig.suptitle(
        f"{factor_name}: |pixel diff| from baseline ({baseline_level}), "
        f"colorscale capped at {vmax:.0f}/255",
        fontsize=12,
    )
    fig.subplots_adjust(right=0.90, top=0.90)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="mean |ΔRGB| per pixel (0-255)")

    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[Figure] {out_path}")


# ====================================================================
# CSV / JSON writers
# ====================================================================

def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"[CSV] skip empty: {path}")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[CSV] {path}")


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {path}")


# ====================================================================
# Report
# ====================================================================

def build_report_markdown(context):
    lines = []
    a = lines.append

    a("# DOE Factor Screening Report")
    a("")
    a("Heuristic screening only -- see CLAUDE.md section 12/15: no factor or "
      "level is auto-removed here. All classifications below are labeled "
      "explicitly as heuristic; the removal decision is left to the human reviewer.")
    a("")

    a("## 1. Baseline condition used")
    a(f"- Source: `is_baseline=True` row in `{context['design_csv']}` "
      f"(design_id={context['baseline_levels']['design_id']})")
    a(f"- Rain = **{context['baseline_levels']['Rain']}**, "
      f"Fog = **{context['baseline_levels']['Fog']}**, "
      f"Illumination (sun_altitude_angle) = **{context['baseline_levels']['Illumination']}**")
    a("")

    a("## 2-4. Actual CARLA WeatherParameters per level")
    a("Full values are in `weather_parameters.json`. Key fields per level:")
    for factor_name in FACTOR_LEVELS:
        a(f"\n**{factor_name}**")
        for level_name, overrides in FACTOR_LEVELS[factor_name].items():
            fields = context["level_fields"][factor_name][level_name]
            a(f"- {level_name}: {format_key_params(factor_name, fields)} "
              f"(cloudiness={fields['cloudiness']:.0f}, wind_intensity={fields['wind_intensity']:.1f}, "
              f"sun_azimuth={fields['sun_azimuth_angle']:.0f})")
    a("")

    a("## 5. Generated visualizations")
    for factor_name in FACTOR_LEVELS:
        d = factor_name.lower()
        a(f"- `{d}/comparison.png`, `{d}/diff_from_baseline.png`, `{d}/metrics.csv`, "
          f"`{d}/adjacent_pairs.csv`, `{d}/extra_metrics.csv`")
    a("- `summary.csv` (all factors' adjacent-level pairs, aggregated across frames)")
    a("- `summary_baseline_diff.csv` (all factors' per-level baseline diffs, aggregated)")
    a("")

    a("## 6. Frame selection method")
    a(f"- Route: `{context['route_xml']}` route id `{context['route_id']}` (town `{context['route_town']}`), "
      f"dense route sampled every {ROUTE_SAMPLING_RESOLUTION} m")
    a(f"- {len(context['frames'])} frames at route progress "
      f"{[f['percentage'] for f in context['frames']]}% "
      f"(dense route length = {context['frames'][0]['route_total_points']} points)")
    a("- Same ego transform (position+rotation) and camera pose reused across every level of every "
      "factor at a given frame -- only carla.WeatherParameters changes between captures.")
    a("")

    a("## 7-8. Factor-level quantitative summary (mean over frames; adjacent-level pairs)")
    for factor_name in FACTOR_LEVELS:
        a(f"\n**{factor_name}**")
        for row in context["adjacent_summary"][factor_name]:
            a(f"- {row['level_a']} vs {row['level_b']}: "
              f"MAD={row['mean_abs_diff_mean']:.2f}, SSIM={row['ssim_mean']:.4f}, "
              f"luminance_diff={row['luminance_diff_mean']:+.2f}, "
              f"edge_density_diff={row['edge_density_diff_mean']:+.4f} "
              f"-> **{classify_pair(row['ssim_mean'])}**")
    a("")

    a("## 9. Levels/pairs where the change looks small (heuristic: 'weakly distinct' or "
      "'visually near-duplicate')")
    weak = context["weak_or_duplicate_pairs"]
    if weak:
        for w in weak:
            a(f"- {w['factor']}: {w['level_a']} vs {w['level_b']} -> {w['assessment']} "
              f"(SSIM={w['ssim_mean']:.4f}, MAD={w['mean_abs_diff_mean']:.2f})")
    else:
        a("- None flagged at the current heuristic thresholds.")
    a("")

    a("## 10. Possible near-duplicate levels")
    dup = [w for w in weak if w["assessment"].startswith("visually near-duplicate")]
    if dup:
        for w in dup:
            a(f"- {w['factor']}: {w['level_a']} ~ {w['level_b']} (SSIM={w['ssim_mean']:.4f})")
    else:
        a("- None flagged at the current heuristic thresholds.")
    a("")

    a("## 11. Candidates that look too extreme / potentially unrealistic")
    a("- Not automatically flagged (no reliable numeric proxy for 'realism'). "
      "Reviewer should check `comparison.png` for each factor, especially the Heavy rain "
      "and Heavy fog panels and the -30 deg illumination panel, by eye.")
    a("")

    a("## 12. D-optimal 52-design level frequency")
    for factor_name, info in context["design_analysis"].items():
        a(f"\n**{factor_name}** (n={info['n_total_designs']} designs)")
        for level, count in info["level_counts"].items():
            a(f"- {level}: {count} designs")
    a("")

    a("## 13. Designs affected if a level were removed")
    for factor_name, info in context["design_analysis"].items():
        a(f"\n**{factor_name}**")
        for level, impact in info["level_removal_impact"].items():
            a(f"- remove {level}: {impact['designs_containing_level']} of "
              f"{info['n_total_designs']} designs contain it "
              f"({impact['designs_remaining_if_removed']} would remain untouched)")
    a("")

    a("## 14. Candidates needing further human review")
    if weak:
        for w in weak:
            a(f"- {w['factor']} {w['level_a']}/{w['level_b']}: {w['assessment']}")
    else:
        a("- No factor/level pair fell below the near-duplicate/weakly-distinct heuristic "
          "thresholds; still recommend a visual pass over all comparison.png files before finalizing.")
    a("")
    a("_No factor or level was deleted or auto-decided by this tool. This report only "
      "provides visualization + metrics for a human decision (CLAUDE.md section 15/17)._")

    return "\n".join(lines)


# ====================================================================
# Main
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DOE Rain/Fog/Illumination factor screening tool.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--route-xml", type=str, default=str(DEFAULT_ROUTE_XML))
    parser.add_argument("--route-id", type=str, default=DEFAULT_ROUTE_ID)
    parser.add_argument("--town-substring", type=str, default=DEFAULT_TOWN_SUBSTRING)
    parser.add_argument("--force-load-world", action="store_true",
                         help="Allow this tool to call client.load_world() if the running "
                              "server isn't already on --town-substring (default: refuse and exit).")

    parser.add_argument("--design-csv", type=str, default=str(DEFAULT_DESIGN_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--frame-percentages", type=str, default=",".join(map(str, DEFAULT_FRAME_PERCENTAGES)),
                         help="Comma-separated route-progress percentages, e.g. 20,40,60,80")
    parser.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS)
    parser.add_argument("--diff-vmax", type=float, default=80.0,
                         help="Fixed colorscale max (0-255) for diff_from_baseline.png heatmaps.")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_percentages = [int(p.strip()) for p in args.frame_percentages.split(",") if p.strip()]

    if not (1 <= args.warmup_ticks <= 60):
        raise ValueError("--warmup-ticks should be a small positive number (5-15 typical).")

    # ----------------------------------------------------------------
    # D-optimal CSV
    # ----------------------------------------------------------------
    design_csv_path = Path(args.design_csv)
    design_rows = read_design_csv(design_csv_path)
    baseline_levels = get_baseline_levels(design_rows)
    design_analysis = analyze_design_csv(design_rows)

    print("=" * 70)
    print(f"Baseline (design_id={baseline_levels['design_id']}): "
          f"Rain={baseline_levels['Rain']}, Fog={baseline_levels['Fog']}, "
          f"Illumination={baseline_levels['Illumination']}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # CARLA connection (does not launch/reload the server by default)
    # ----------------------------------------------------------------
    client = connect_carla(args.host, args.port, args.timeout)
    route_town_hint, _control_points_preview = load_route_from_xml(args.route_xml, args.route_id)
    world = get_world(client, args.town_substring, args.force_load_world, route_town_hint)

    original_settings = setup_synchronous(world, FIXED_DELTA_SECONDS)
    original_weather = world.get_weather()

    ego = None
    camera = None

    try:
        frames, route_town = select_frames(world, args.route_xml, args.route_id, ROUTE_SAMPLING_RESOLUTION, frame_percentages)
        print(f"[Route] {len(frames)} frames selected from route '{args.route_id}' ({route_town}): "
              f"{[f['percentage'] for f in frames]}%")

        ego = spawn_ego(world, frames[0]["transform"])
        camera = spawn_camera(world, ego)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        world.tick()
        image_queue.get(timeout=10.0)

        factor_records = run_captures(world, ego, camera, image_queue, frames, baseline_levels, output_dir, args.warmup_ticks)

        # ------------------------------------------------------------
        # Per-factor metrics + figures
        # ------------------------------------------------------------
        baseline_level_by_factor = {
            "Rain": baseline_levels["Rain"],
            "Fog": baseline_levels["Fog"],
            "Illumination": baseline_levels["Illumination"],
        }

        all_adjacent_rows = []
        all_baseline_rows = []
        adjacent_summary_by_factor = {}
        level_fields_by_factor = {name: {} for name in FACTOR_LEVELS}

        for factor_name, frame_records in factor_records.items():
            baseline_level = baseline_level_by_factor[factor_name]
            factor_dir = output_dir / factor_name.lower()
            factor_dir.mkdir(parents=True, exist_ok=True)

            for level_name, fields in frame_records[0]["fields"].items():
                level_fields_by_factor[factor_name][level_name] = fields

            baseline_rows, adjacent_rows, extra_rows = compute_factor_tables(factor_name, frame_records, baseline_level)

            write_csv(factor_dir / "metrics.csv", baseline_rows)
            write_csv(factor_dir / "adjacent_pairs.csv", adjacent_rows)
            write_csv(factor_dir / "extra_metrics.csv", extra_rows)

            baseline_summary = aggregate_rows(baseline_rows, ["factor", "level"], METRIC_VALUE_KEYS)
            adjacent_summary = aggregate_rows(adjacent_rows, ["factor", "level_a", "level_b"], METRIC_VALUE_KEYS)

            write_csv(factor_dir / "metrics_summary.csv", baseline_summary)
            write_csv(factor_dir / "adjacent_pairs_summary.csv", adjacent_summary)

            adjacent_summary_by_factor[factor_name] = adjacent_summary
            all_adjacent_rows.extend(adjacent_summary)
            all_baseline_rows.extend(baseline_summary)

            render_comparison_grid(factor_name, frame_records, baseline_level, factor_dir / "comparison.png")
            render_diff_grid(factor_name, frame_records, baseline_level, factor_dir / "diff_from_baseline.png", vmax=args.diff_vmax)

        # ------------------------------------------------------------
        # Top-level summary.csv (section 13) + baseline-diff summary
        # ------------------------------------------------------------
        for row in all_adjacent_rows:
            row["assessment"] = classify_pair(row["ssim_mean"])

        write_csv(output_dir / "summary.csv", all_adjacent_rows)
        write_csv(output_dir / "summary_baseline_diff.csv", all_baseline_rows)

        weak_or_duplicate_pairs = [
            row for row in all_adjacent_rows
            if row["assessment"] in ("weakly distinct", "visually near-duplicate (heuristic)")
        ]

        # ------------------------------------------------------------
        # weather_parameters.json (section 3)
        # ------------------------------------------------------------
        write_json(output_dir / "weather_parameters.json", {
            "baseline_levels": baseline_levels,
            "fixed_fields": {
                "cloudiness": CLOUDINESS_FIXED,
                "wind_intensity": WEATHER_WIND_INTENSITY,
                "sun_azimuth_angle": SUN_AZIMUTH_FIXED,
            },
            "levels": level_fields_by_factor,
        })

        write_json(output_dir / "doe_csv_summary.json", design_analysis)

        write_json(output_dir / "frames.json", {
            "route_xml": str(args.route_xml),
            "route_id": args.route_id,
            "route_town": route_town,
            "sampling_resolution_m": ROUTE_SAMPLING_RESOLUTION,
            "frames": [
                {k: v for k, v in f.items() if k != "transform"}
                for f in frames
            ],
        })

        # ------------------------------------------------------------
        # Final report
        # ------------------------------------------------------------
        report_md = build_report_markdown({
            "design_csv": design_csv_path,
            "baseline_levels": baseline_levels,
            "level_fields": level_fields_by_factor,
            "route_xml": args.route_xml,
            "route_id": args.route_id,
            "route_town": route_town,
            "frames": frames,
            "adjacent_summary": adjacent_summary_by_factor,
            "weak_or_duplicate_pairs": weak_or_duplicate_pairs,
            "design_analysis": design_analysis,
        })

        (output_dir / "screening_report.md").write_text(report_md, encoding="utf-8")
        print(f"[Report] {output_dir / 'screening_report.md'}")

        print()
        print(f"[Done] Screening complete. See {output_dir}")

    finally:
        cleanup(world, original_settings, original_weather, camera, ego)


if __name__ == "__main__":
    main()

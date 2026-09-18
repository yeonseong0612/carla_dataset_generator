"""
scripts/visualize_doe_factors.py

Standalone DOE (Design of Experiments) weather-factor visualization tool.

NOT part of the dataset-generation pipeline (scripts/collect_dataset.py) --
this script exists purely so a team can visually compare how individual
CARLA WeatherParameters fields (and a few composite "severity profiles")
affect a front RGB camera image, in order to pick DOE factors/levels for
a later full-factorial experiment. It does not touch, import, or modify
any file under src/ or scripts/collect_dataset.py.

Assumes a CARLA 0.9.16 server is already running separately (this script
only connects to it -- it never launches CARLA.exe).

--------------------------------------------------------------------
Same-scene guarantee
--------------------------------------------------------------------
Everything that is NOT the weather parameter under test is fixed for the
entire run: the ego vehicle is spawned once (at --spawn-index) and never
moves again (autopilot off, physics disabled via
set_simulate_physics(False) so nothing nudges it), the camera is spawned
once and attached with a fixed relative transform, and the world is
ticked in synchronous mode so nothing advances except when this script
explicitly calls world.tick(). No traffic/pedestrians are spawned (no
NPCs => nothing to keep transforms fixed FOR -- the town's own static
content, if any, is untouched by this script and therefore stays fixed
by construction).

--------------------------------------------------------------------
CARLA 0.9.16 WeatherParameters verification (see section 11 of the task)
--------------------------------------------------------------------
Verified live against the installed carla module (help(carla.WeatherParameters)):
  cloudiness, precipitation, precipitation_deposits, wind_intensity,
  sun_azimuth_angle, sun_altitude_angle, fog_density, fog_distance,
  fog_falloff, wetness, scattering_intensity, mie_scattering_scale,
  rayleigh_scattering_scale, dust_storm all exist as float fields on this
  build. Only the first 9 core fields (all except the last 4 scattering/
  dust fields) are used for the sweeps below, per the task's explicit
  scope -- scattering_intensity/mie_scattering_scale are version-fragile
  and deliberately NOT exercised here.

  fog_distance semantics (verified against this build's own presets,
  e.g. ClearNoon has fog_density=2, fog_distance=0.75 -- i.e. even the
  "clear" preset has a nonzero, near-0 fog_distance): fog_distance is
  the distance (in meters) from the camera at which fog starts to
  appear, NOT a toggle. fog_distance=0 does NOT mean "no fog" -- it
  means the fog (whatever fog_density says) starts immediately at the
  camera, i.e. the *most* fog-affected framing, not the least. Whether
  fog is visually present at all is controlled by fog_density; at
  fog_density=0, fog_distance's value has no visible effect (nothing to
  start). This is why the fog_distance sweep (factor C) fixes
  fog_density to a nonzero mid value (50) instead of 0 -- at
  fog_density=0 every fog_distance value would look identical.

  fog_falloff: fog "height" falloff / density-with-altitude exponent.
  0 = fog is lighter than air and blankets the whole scene at any
  height; ~1 = roughly air density, reaches normal building height;
  >5 = fog compresses to a thin layer at ground level. The candidate
  values [0.0, 0.2, 0.5, 1.0, 2.0] all fall inside this meaningful
  range and were kept as given.
"""

import argparse
import glob
import json
import os
import queue
import sys
from pathlib import Path

import cv2
import numpy as np

CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"
sys.path.insert(0, str(CARLA_PYTHONAPI))

import carla  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ====================================================================
# Scene / camera config -- edit here to change defaults without
# touching CLI plumbing (per task section 4/10: "코드에서 상수/config로
# 쉽게 변경 가능하도록 한다").
# ====================================================================

DEFAULT_TOWN = "Town10HD_Opt"
DEFAULT_SPAWN_INDEX = 0

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FOV = 90.0

# Front camera, typical AV mounting position (task section 4: x in
# [1.5, 2.0]m, z in [1.5, 2.0]m). Rotation left at (0,0,0) relative to
# the vehicle -- looks straight down the vehicle's forward (+X) axis.
CAMERA_LOCATION = carla.Location(x=1.6, y=0.0, z=1.6)
CAMERA_ROTATION = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

FIXED_DELTA_SECONDS = 0.05
DEFAULT_WARMUP_TICKS = 5

# Weather fields NOT swept in this task (deliberately excluded --
# task section 1): held fixed at these values for every capture,
# everywhere, so they can never confound a sweep.
CLOUDINESS_FIXED = 10.0
WIND_INTENSITY_FIXED = 10.0
SUN_AZIMUTH_FIXED = 0.0

# "Weather degradation off" baseline reused by every non-rain/fog sweep
# so only the swept field(s) differ from a clean baseline.
DEGRADATION_OFF = dict(
    fog_density=0.0, fog_distance=0.0, fog_falloff=0.0,
    precipitation=0.0, wetness=0.0, precipitation_deposits=0.0,
)

# Baseline sun used whenever illumination itself isn't being swept.
BASELINE_SUN_ALTITUDE = 60.0

WEATHER_FIELDS = (
    "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
    "sun_azimuth_angle", "sun_altitude_angle",
    "fog_density", "fog_distance", "fog_falloff", "wetness",
)


# ====================================================================
# Weather construction helpers
# ====================================================================

def make_weather(overrides):
    """
    Build a carla.WeatherParameters from CLOUDINESS_FIXED / WIND_INTENSITY_FIXED
    / SUN_AZIMUTH_FIXED / BASELINE_SUN_ALTITUDE / DEGRADATION_OFF as a base,
    with `overrides` replacing any of WEATHER_FIELDS on top.
    """

    fields = {
        "cloudiness": CLOUDINESS_FIXED,
        "wind_intensity": WIND_INTENSITY_FIXED,
        "sun_azimuth_angle": SUN_AZIMUTH_FIXED,
        "sun_altitude_angle": BASELINE_SUN_ALTITUDE,
        **DEGRADATION_OFF,
        **overrides,
    }

    return carla.WeatherParameters(**{k: float(v) for k, v in fields.items()}), fields


def weather_fields_to_dict(fields):
    return {key: float(fields[key]) for key in WEATHER_FIELDS}


# ====================================================================
# Sweep / profile definitions (task sections 5-6)
# ====================================================================
#
# Each sweep entry: (subdir_name, [(filename, weather_overrides, label), ...])
# Composite profiles are visualization candidates only, NOT final DOE
# levels (task section 6's explicit caveat).

def format_signed(v):
    v = int(v)
    if v > 0:
        return f"+{v}"
    return str(v)


def illumination_sweep():
    values = [60, 30, 0, -15, -30]
    items = []

    for v in values:
        filename = f"sun_{format_signed(v)}.png"
        overrides = {"sun_altitude_angle": float(v)}  # DEGRADATION_OFF already zeroes fog/rain
        label = f"Sun Altitude = {v} deg"
        items.append((filename, overrides, label))

    return "illumination", items


def fog_density_sweep():
    values = [0, 20, 40, 60, 80]
    items = []

    for v in values:
        filename = f"fog_density_{v:03d}.png"
        overrides = {"fog_density": float(v)}  # fog_distance/falloff stay at DEGRADATION_OFF (0)
        label = f"Fog Density = {v}"
        items.append((filename, overrides, label))

    return "fog_density", items


def fog_distance_sweep():
    # fog_density fixed to a nonzero mid value (50) -- see module
    # docstring: at fog_density=0 every fog_distance value looks
    # identical (nothing to start), so this sweep would be meaningless
    # against the DEGRADATION_OFF baseline.
    values = [0, 10, 25, 50, 100]
    items = []

    for v in values:
        filename = f"fog_distance_{v:03d}.png"
        overrides = {"fog_density": 50.0, "fog_distance": float(v), "fog_falloff": 1.0}
        label = f"Fog Distance = {v} m"
        items.append((filename, overrides, label))

    return "fog_distance", items


def fog_falloff_sweep():
    values = [0.0, 0.2, 0.5, 1.0, 2.0]
    items = []

    for v in values:
        filename = f"fog_falloff_{v:.1f}.png"
        overrides = {"fog_density": 50.0, "fog_distance": 25.0, "fog_falloff": float(v)}
        label = f"Fog Falloff = {v:.1f}"
        items.append((filename, overrides, label))

    return "fog_falloff", items


def precipitation_sweep():
    values = [0, 20, 40, 60, 80]
    items = []

    for v in values:
        filename = f"precipitation_{v:03d}.png"
        overrides = {"precipitation": float(v)}  # wetness/deposits stay 0 -- isolates precipitation's own visual (streaks/particles)
        label = f"Precipitation = {v}"
        items.append((filename, overrides, label))

    return "precipitation", items


def wetness_sweep():
    values = [0, 25, 50, 75, 100]
    items = []

    for v in values:
        # precipitation held at 0 (not just "a fixed value") so the
        # sweep isolates wetness's own road-reflectivity effect, not a
        # rain+wetness combination.
        filename = f"wetness_{v:03d}.png"
        overrides = {"wetness": float(v)}
        label = f"Wetness = {v}"
        items.append((filename, overrides, label))

    return "wetness", items


def precipitation_deposits_sweep():
    values = [0, 25, 50, 75, 100]
    items = []

    for v in values:
        filename = f"deposits_{v:03d}.png"
        overrides = {"precipitation_deposits": float(v)}
        label = f"Precipitation Deposits = {v}"
        items.append((filename, overrides, label))

    return "precipitation_deposits", items


# Composite profiles -- visualization candidates only, NOT final DOE
# levels (task section 6).

FOG_PROFILES = [
    ("clear", dict(fog_density=0.0, fog_distance=0.0, fog_falloff=0.0)),
    ("light", dict(fog_density=25.0, fog_distance=100.0, fog_falloff=1.0)),
    ("moderate", dict(fog_density=50.0, fog_distance=50.0, fog_falloff=1.0)),
    ("heavy", dict(fog_density=75.0, fog_distance=20.0, fog_falloff=1.0)),
]

RAIN_PROFILES = [
    ("dry", dict(precipitation=0.0, wetness=0.0, precipitation_deposits=0.0)),
    ("light", dict(precipitation=25.0, wetness=30.0, precipitation_deposits=15.0)),
    ("moderate", dict(precipitation=50.0, wetness=60.0, precipitation_deposits=40.0)),
    ("heavy", dict(precipitation=80.0, wetness=90.0, precipitation_deposits=70.0)),
]


def fog_profile_sweep():
    items = []

    for name, overrides in FOG_PROFILES:
        filename = f"fog_profile_{name}.png"
        label = f"Fog Profile: {name.capitalize()}"
        items.append((filename, dict(overrides), label))

    return "fog_profiles", items


def rain_profile_sweep():
    items = []

    for name, overrides in RAIN_PROFILES:
        filename = f"rain_profile_{name}.png"
        label = f"Rain Profile: {name.capitalize()}"
        items.append((filename, dict(overrides), label))

    return "rain_profiles", items


SWEEP_BUILDERS = {
    "illumination": illumination_sweep,
    "fog_density": fog_density_sweep,
    "fog_distance": fog_distance_sweep,
    "fog_falloff": fog_falloff_sweep,
    "precipitation": precipitation_sweep,
    "wetness": wetness_sweep,
    "precipitation_deposits": precipitation_deposits_sweep,
    "fog_profiles": fog_profile_sweep,
    "rain_profiles": rain_profile_sweep,
}

FACTOR_CHOICES = ["all"] + list(SWEEP_BUILDERS.keys())


# ====================================================================
# CARLA connection / scene setup
# ====================================================================

def connect_carla(host, port, timeout=120.0):
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    return client


def setup_world(client, town, fixed_delta_seconds):
    """
    Reuses the already-running server's current world if it's already on
    `town` (cheap); otherwise loads it (slow, disruptive to anything else
    using this server -- but this script assumes it owns the server for
    its run, per task assumptions). Returns (world, original_settings)
    so cleanup() can restore synchronous_mode/fixed_delta_seconds exactly.
    """

    world = client.get_world()

    if town not in world.get_map().name:
        print(f"[World] current map '{world.get_map().name}' != '{town}', loading...")
        world = client.load_world(town)
    else:
        print(f"[World] reusing already-loaded map '{world.get_map().name}'")

    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    return world, original_settings


def spawn_ego(world, spawn_index):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find("vehicle.tesla.model3")

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "doe_visualization")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Map has no spawn points.")

    if not (0 <= spawn_index < len(spawn_points)):
        raise ValueError(f"--spawn-index {spawn_index} out of range [0, {len(spawn_points) - 1}]")

    transform = spawn_points[spawn_index]

    ego = world.try_spawn_actor(blueprint, transform)

    if ego is None:
        raise RuntimeError(f"Failed to spawn ego at spawn point {spawn_index}: {transform}")

    # Same-scene guarantee (task section 2): nothing may move the ego
    # across the whole sweep. autopilot stays off (default) and physics
    # is disabled outright so nothing (gravity settling, a stray
    # collision impulse) can nudge it between captures.
    ego.set_simulate_physics(False)
    ego.apply_control(carla.VehicleControl())

    return ego, transform


def spawn_camera(world, ego, width, height, fov):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))

    camera_transform = carla.Transform(CAMERA_LOCATION, CAMERA_ROTATION)
    camera = world.spawn_actor(blueprint, camera_transform, attach_to=ego)

    return camera, camera_transform


def set_weather(world, weather):
    world.set_weather(weather)


# ====================================================================
# Deterministic tick / capture (task section 11.4-11.5: no deadlock, no
# stale-frame reuse)
# ====================================================================

def wait_for_frame(world, image_queue, ticks, timeout=10.0):
    """
    Ticks `ticks` times, draining (and discarding) each tick's camera
    frame so the queue never builds up latency -- this is the actual
    "let the weather change settle into rendering" warmup, not just a
    fixed sleep. The LAST tick's frame is deliberately also drained here
    (not left for capture_image) so capture_image's own tick always
    starts from an empty queue -- otherwise a stale warmup frame could
    be mistaken for the real capture.
    """

    for _ in range(ticks):
        frame_id = world.tick()
        image = image_queue.get(timeout=timeout)

        if image.frame != frame_id:
            raise RuntimeError(
                f"Warmup frame mismatch: ticked {frame_id}, got queued frame {image.frame} "
                f"-- stale frame in queue, camera callback may be lagging."
            )


def capture_image(world, image_queue, out_path, timeout=10.0):
    """
    Ticks exactly once more and saves THAT frame's image -- guarantees
    the saved PNG corresponds to the weather state as of this exact
    tick, never a leftover frame from before set_weather() was called.
    """

    frame_id = world.tick()
    image = image_queue.get(timeout=timeout)

    if image.frame != frame_id:
        raise RuntimeError(
            f"Capture frame mismatch: ticked {frame_id}, got queued frame {image.frame}."
        )

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = array[:, :, :3]  # CARLA raw_data is BGRA; drop alpha for a plain BGR PNG (cv2 native order)

    cv2.imwrite(out_path, bgr)

    return frame_id


# ====================================================================
# Sweep execution
# ====================================================================

def run_parameter_sweep(world, image_queue, sweep_name, items, out_dir, warmup_ticks, metadata_images):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for filename, overrides, label in items:
        weather, fields = make_weather(overrides)
        set_weather(world, weather)

        wait_for_frame(world, image_queue, warmup_ticks)

        out_path = os.path.join(out_dir, filename)
        capture_image(world, image_queue, out_path)

        metadata_images[f"{sweep_name}/{filename}"] = weather_fields_to_dict(fields)
        saved_paths.append((out_path, label))

        print(f"[Capture] {sweep_name}/{filename}  ({label})")

    return saved_paths


def run_profile_sweep(world, image_queue, sweep_name, items, out_dir, warmup_ticks, metadata_images):
    # Profiles are just parameter sweeps with a name instead of a swept
    # numeric value -- same execution path.
    return run_parameter_sweep(world, image_queue, sweep_name, items, out_dir, warmup_ticks, metadata_images)


# ====================================================================
# Contact sheets
# ====================================================================

def create_contact_sheet(entries, out_path, panel_width=480, label_height=44, font_scale=0.65):
    """
    entries: [(image_path, label), ...]. Panels are resized to the same
    panel_width (preserving aspect ratio -- 1280x720 source at 480px
    wide is still ~270px tall, enough to judge fog/rain/lighting
    differences) and placed in a single row with the label rendered in a
    white strip above each panel.
    """

    panels = []

    for image_path, label in entries:
        image = cv2.imread(image_path)

        if image is None:
            print(f"[ContactSheet] skip missing/unreadable image: {image_path}")
            continue

        h, w = image.shape[:2]
        scale = panel_width / w
        resized = cv2.resize(image, (panel_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

        canvas = np.full((resized.shape[0] + label_height, panel_width, 3), 255, dtype=np.uint8)
        canvas[label_height:, :, :] = resized

        cv2.putText(
            canvas, label, (8, int(label_height * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA,
        )

        panels.append(canvas)

    if not panels:
        print(f"[ContactSheet] no panels for {out_path}, skipping")
        return

    sheet = np.hstack(panels)
    cv2.imwrite(out_path, sheet)
    print(f"[ContactSheet] {out_path}")


# ====================================================================
# Metadata
# ====================================================================

def save_metadata(path, scene_info, metadata_images):
    data = {"scene": scene_info, "images": metadata_images}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[Metadata] {path}")


# ====================================================================
# Cleanup
# ====================================================================

def cleanup(world, original_settings, camera, ego):
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

    if world is not None and original_settings is not None:
        try:
            world.apply_settings(original_settings)
        except RuntimeError as exc:
            print(f"[Cleanup] failed to restore world settings: {exc}")


# ====================================================================
# CLI / main
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent DOE weather-factor visualization tool (not part of the dataset pipeline)."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default=DEFAULT_TOWN)
    parser.add_argument("--spawn-index", type=int, default=DEFAULT_SPAWN_INDEX)
    parser.add_argument("--factor", type=str, default="all", choices=FACTOR_CHOICES)
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "doe_visualization"),
    )
    parser.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)

    return parser.parse_args()


def main():
    args = parse_args()

    if not (1 <= args.warmup_ticks <= 60):
        raise ValueError("--warmup-ticks should be a small positive number (task suggests 3-10).")

    scene_id = f"scene_{args.spawn_index:03d}"
    scene_dir = os.path.join(args.output_dir, args.town, scene_id)
    contact_sheet_dir = os.path.join(scene_dir, "contact_sheets")

    client = connect_carla(args.host, args.port)

    world = None
    original_settings = None
    ego = None
    camera = None

    try:
        world, original_settings = setup_world(client, args.town, FIXED_DELTA_SECONDS)

        ego, ego_transform = spawn_ego(world, args.spawn_index)
        camera, camera_transform = spawn_camera(world, ego, args.width, args.height, args.fov)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # Let the spawn itself settle before the first real capture --
        # otherwise the very first wait_for_frame() tick can race the
        # server-side actor/sensor spawn RPCs.
        world.tick()
        image_queue.get(timeout=10.0)

        metadata_images = {}

        factors = list(SWEEP_BUILDERS.keys()) if args.factor == "all" else [args.factor]

        for factor in factors:
            sweep_name, items = SWEEP_BUILDERS[factor]()
            out_dir = os.path.join(scene_dir, sweep_name)
            runner = run_profile_sweep if factor in ("fog_profiles", "rain_profiles") else run_parameter_sweep

            saved = runner(
                world, image_queue, sweep_name, items, out_dir, args.warmup_ticks, metadata_images,
            )

            os.makedirs(contact_sheet_dir, exist_ok=True)
            create_contact_sheet(saved, os.path.join(contact_sheet_dir, f"{sweep_name}.png"))

        scene_info = {
            "town": args.town,
            "spawn_index": args.spawn_index,
            "ego_transform": {
                "x": ego_transform.location.x, "y": ego_transform.location.y, "z": ego_transform.location.z,
                "pitch": ego_transform.rotation.pitch, "yaw": ego_transform.rotation.yaw, "roll": ego_transform.rotation.roll,
            },
            "camera_transform_relative_to_ego": {
                "x": camera_transform.location.x, "y": camera_transform.location.y, "z": camera_transform.location.z,
                "pitch": camera_transform.rotation.pitch, "yaw": camera_transform.rotation.yaw, "roll": camera_transform.rotation.roll,
            },
            "width": args.width, "height": args.height, "fov": args.fov,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "warmup_ticks": args.warmup_ticks,
            "fixed_fields": {
                "cloudiness": CLOUDINESS_FIXED,
                "wind_intensity": WIND_INTENSITY_FIXED,
                "sun_azimuth_angle": SUN_AZIMUTH_FIXED,
            },
        }

        save_metadata(os.path.join(scene_dir, "metadata.json"), scene_info, metadata_images)

        print()
        print(f"[Done] {len(metadata_images)} images captured under {scene_dir}")

    finally:
        cleanup(world, original_settings, camera, ego)


if __name__ == "__main__":
    main()

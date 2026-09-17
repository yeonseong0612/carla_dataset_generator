"""
capture_weather.py

Standalone CARLA weather capture/inspection tool.

Cycles through all canonical weather conditions, prints their
WeatherParameters values, and saves one reference screenshot per
condition. This is a manual verification tool, not part of the
production dataset collection pipeline (see scripts/collect_dataset.py).
"""

import sys
import os
import queue
import time
import argparse
from pathlib import Path

import carla

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.weather import apply_weather


WEATHER_IDS = {
    "day_clear": "W00",
    "night_clear": "W01",
    "day_rain": "W02",
    "night_rain": "W03",
    "day_fog": "W04",
    "night_fog": "W05"
}


def get_weather_conditions():
    return list(WEATHER_IDS.keys())


def print_weather(condition, weather):
    print()
    print("=" * 60)
    print(f"Weather: {condition}")
    print("=" * 60)

    print(f"cloudiness               : {weather.cloudiness}")
    print(f"precipitation             : {weather.precipitation}")
    print(f"precipitation_deposits    : {weather.precipitation_deposits}")
    print(f"wetness                   : {weather.wetness}")
    print(f"wind_intensity            : {weather.wind_intensity}")

    print(f"sun_azimuth_angle         : {weather.sun_azimuth_angle}")
    print(f"sun_altitude_angle        : {weather.sun_altitude_angle}")

    print(f"fog_density               : {weather.fog_density}")
    print(f"fog_distance              : {weather.fog_distance}")
    print(f"fog_falloff               : {weather.fog_falloff}")

def create_capture_camera(world, width=1920, height=1080, fov=90.0):
    spectator = world.get_spectator()

    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))

    if camera_bp.has_attribute("gamma"):
        camera_bp.set_attribute("gamma", "2.2")

    camera = world.spawn_actor(
        camera_bp,
        spectator.get_transform(),
    )

    image_queue = queue.Queue()

    camera.listen(image_queue.put)

    return camera, image_queue


def capture_weather_image(
    world,
    camera,
    image_queue,
    output_path,
    settle_time=2.0,
):
    # 날씨가 렌더링에 충분히 반영될 때까지 기다림
    time.sleep(settle_time)

    # 기다리는 동안 쌓인 오래된 프레임 전부 제거
    while not image_queue.empty():
        try:
            image_queue.get_nowait()
        except queue.Empty:
            break

    # queue를 비운 이후 생성되는 새로운 프레임 1장 획득
    try:
        image = image_queue.get(timeout=5.0)

    except queue.Empty:
        raise RuntimeError(
            "Failed to receive RGB image from CARLA camera."
        )

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image.save_to_disk(output_path)

    print(
        f"[Capture] Saved: {output_path} "
        f"(frame={image.frame}, timestamp={image.timestamp:.3f})"
    )

def test_weather(host = "127.0.0.1", port = 2000, duration = 5.0, capture_dir="weather_captures", capture_width=1920, capture_height=1080, capture_delay=2.0):
    print("[CARLA] Connecting...")

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    world = client.get_world()

    print("[CARLA] Connected")
    print(f"[CARLA] Map: {world.get_map().name}")

    original_weather = world.get_weather()

    camera = None

    try:
        # ====================================================
        # Capture camera
        # ====================================================

        camera, image_queue = create_capture_camera(
            world,
            width=capture_width,
            height=capture_height,
        )

        conditions = get_weather_conditions()

        print()
        print("Weather test")
        print("-" * 60)
        print(f"Conditions   : {len(conditions)}")
        print(
            f"Capture size : "
            f"{capture_width}x{capture_height}"
        )
        print(f"Capture dir  : {capture_dir}")

        if duration > 0:
            print(
                f"Duration     : "
                f"{duration:.1f} sec / condition"
            )
        else:
            print("Mode         : manual")

        print("-" * 60)

        for index, condition in enumerate(conditions):

            # -----------------------------------------------
            # Apply weather
            # -----------------------------------------------

            weather = apply_weather(
                world,
                condition,
            )

            print_weather(
                condition,
                weather,
            )

            print(
                f"[{index + 1}/{len(conditions)}] "
                f"Applied: {condition}"
            )

            # -----------------------------------------------
            # Capture
            # -----------------------------------------------

            filename = (
                f"{index + 1:02d}_"
                f"{condition}.png"
            )

            output_path = os.path.join(
                capture_dir,
                filename,
            )

            capture_weather_image(
                world=world,
                camera=camera,
                image_queue=image_queue,
                output_path=output_path,
                settle_time=capture_delay,
            )

            # -----------------------------------------------
            # Wait
            # -----------------------------------------------

            if duration > 0:

                remaining_time = max(
                    0.0,
                    duration - capture_delay,
                )

                if remaining_time > 0:
                    time.sleep(remaining_time)

            else:

                input(
                    "\nPress Enter for next weather..."
                )

    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.destroy()
            except Exception:
                pass
        print()
        print("[CARLA] Restoring original weather...")

        world.set_weather(original_weather)

        print("[CARLA] Weather restored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Weather Test")

    parser.add_argument("--host", default="127.0.0.1", help="CARLA server port")
    parser.add_argument("--port", default=2000, type=int, help="CARLA server port")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per weather condition. Set 0 for manual mode.")
    parser.add_argument(
        "--capture-dir",
        default="weather_captures",
        help="Directory for weather screenshots",
    )

    parser.add_argument(
        "--capture-width",
        type=int,
        default=1920,
        help="Screenshot width",
    )

    parser.add_argument(
        "--capture-height",
        type=int,
        default=1080,
        help="Screenshot height",
    )

    parser.add_argument(
        "--capture-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before capturing after weather change",
    )

    args = parser.parse_args()

    test_weather(host=args.host, port=args.port, duration=args.duration, capture_dir=args.capture_dir, capture_width=args.capture_width, capture_height=args.capture_height, capture_delay=args.capture_delay)

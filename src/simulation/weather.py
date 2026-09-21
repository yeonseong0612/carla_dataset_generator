'''
weather.py

Canonical weather configuration for CARLA dataset generation.

## Conditions

day_clear
day_rain
day_fog
night_clear
night_rain
night_fog

## Policy

1. Clear / Rain:
   - Use CARLA built-in WeatherParameters presets directly.

2. Fog:
   - CARLA has no named FoggyNoon / FoggyNight preset in the
     currently used version.
   - Therefore, fog conditions are defined explicitly using
     carla.WeatherParameters with fixed fog-related parameters.

3. Day / Night pairs:
   - Environmental severity is kept identical within each weather type.
   - The main difference between day and night conditions is the
     sun altitude angle and resulting illumination.

4. Wind is intentionally fixed to zero for EVERY condition:
   - Paired conditions vary weather appearance while preserving geometry as
     much as possible.
   - Vegetation animation follows world time, so with wind > 0 the same
     recorded geometry replayed at a different world time renders
     foliage/grass at different positions. make_weather() overrides
     wind_intensity AFTER the preset / custom parameters are built, so it
     also holds for the CARLA presets (which carry their own wind values).
'''

import carla


# Fixed for all conditions; see policy 4 above.
WEATHER_WIND_INTENSITY = 0.0


def make_weather(condition):
    weather = _base_weather(condition)

    # Last step on purpose: presets (ClearNoon, MidRainyNoon, ...) define
    # their own wind_intensity, which must not leak into the paired dataset.
    weather.wind_intensity = WEATHER_WIND_INTENSITY

    return weather


def _base_weather(condition):
    if condition == "day_clear":
        return carla.WeatherParameters.ClearNoon

    elif condition == "night_clear":
        return carla.WeatherParameters.ClearNight

    elif condition == "day_rain":
        return carla.WeatherParameters.MidRainyNoon

    elif condition == "night_rain":
        return carla.WeatherParameters.MidRainyNight

    elif condition == "day_fog":
        return carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=WEATHER_WIND_INTENSITY,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=45.0,
            fog_density=50.0,
            fog_distance=10.0,
            fog_falloff=0.1,
            wetness=0.0,
        )

    elif condition == "night_fog":
        return carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=WEATHER_WIND_INTENSITY,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-35.0,
            fog_density=50.0,
            fog_distance=10.0,
            fog_falloff=0.1,
            wetness=0.0,
        )

    raise ValueError(f"Unknown weather condition: {condition}")

def apply_weather(world, condition):
    weather = make_weather(condition)
    world.set_weather(weather)

    return weather

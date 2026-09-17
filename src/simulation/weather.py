'''
weather.py

Canonical weather configuration for CARLA dataset generarion.

Conditions
-----------
day_clear
day_rain
day_fog
night_clear
night_rain
night_fog

Policy
-----------
1. Celar / Rain:
    - Use CARLA built-in WeatherParameters presets directrly.

2. Fog:
    - CARLA has no named FoggyNoon / FoggyNight preset in the
      currently used version, so CloudyNoon / CloudyNight are used
      as base presets and only fog-related parameters are moified.

3. Day / Night pairs:
    - Environmental severity is kept identical.
      The main differnece is CARLA's day/night illumionation.

'''

import carla


def make_weather(condition):
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
            wind_intensity=10.0,
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
            wind_intensity=10.0,
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

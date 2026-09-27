"""
weather_profiles.py

Single source of truth for the D-optimal DOE weather factors
(Sun / Rain / Fog) and their mapping to carla.WeatherParameters fields.

Used by:
    scripts/visualize_doe_factors.py              (factor visualization)
    scripts/tools/visualize_doe_factor_screening.py
    src/simulation/weather.py                     (DOE condition names ->
                                                   dataset collection)
    scripts/run_doptimal_dataset.py               (DOE orchestration)

so the images used to pick the levels and the collected DOE dataset can
never drift apart. The Rain/Fog values below were moved here verbatim from
scripts/visualize_doe_factors.py (RAIN_PROFILES / FOG_PROFILES); keys use
the D-optimal CSV's level spelling.

Pure Python (no carla import at module level) so the DOE runner's
validation / dry-run / offline tests work without a CARLA server.

DOE condition names
-------------------
The dataset collector (scripts/collect_dataset.py) identifies a weather by
a condition-name string (conditions/<name>/ on disk). A DOE condition is
encoded as a self-describing name, e.g.

    doe_C017_sun+30_rain-Light_fog-Heavy

src/simulation/weather.py make_weather() decodes such a name back into
Sun/Rain/Fog levels through parse_doe_condition_name(); the weather is
derived ONLY from the sun/rain/fog tokens, the C-id is a label for humans
and for the runner's job bookkeeping.
"""

import hashlib
import json
import re
from collections import OrderedDict

from src.simulation.weather import WEATHER_WIND_INTENSITY


# ====================================================================
# Factor levels
# ====================================================================

# Sun factor: the CSV value IS weather.sun_altitude_angle (degrees).
DOE_SUN_LEVELS = (60, 30, 0, -30)

RAIN_PROFILES = OrderedDict([
    ("Dry",      dict(precipitation=0.0,  wetness=0.0,  precipitation_deposits=0.0)),
    ("Light",    dict(precipitation=25.0, wetness=30.0, precipitation_deposits=15.0)),
    ("Moderate", dict(precipitation=50.0, wetness=60.0, precipitation_deposits=40.0)),
    ("Heavy",    dict(precipitation=80.0, wetness=90.0, precipitation_deposits=70.0)),
])

FOG_PROFILES = OrderedDict([
    ("Clear",    dict(fog_density=0.0,  fog_distance=0.0,   fog_falloff=0.0)),
    ("Light",    dict(fog_density=25.0, fog_distance=100.0, fog_falloff=1.0)),
    ("Moderate", dict(fog_density=50.0, fog_distance=50.0,  fog_falloff=1.0)),
    ("Heavy",    dict(fog_density=75.0, fog_distance=20.0,  fog_falloff=1.0)),
])

# Weather fields that are NOT DOE factors, held fixed for every DOE
# condition (and every DOE visualization capture). Wind comes from the
# production paired-dataset policy in src/simulation/weather.py (0.0:
# vegetation must not move between replayed conditions).
DOE_FIXED_FIELDS = OrderedDict([
    ("cloudiness", 10.0),
    ("sun_azimuth_angle", 0.0),
    ("wind_intensity", float(WEATHER_WIND_INTENSITY)),
])

# Every field a DOE weather sets explicitly (fixed + factor fields).
DOE_WEATHER_FIELDS = (
    "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
    "sun_azimuth_angle", "sun_altitude_angle",
    "fog_density", "fog_distance", "fog_falloff", "wetness",
)


# ====================================================================
# Level -> WeatherParameters fields
# ====================================================================

def doe_weather_fields(sun, rain, fog):
    """
    Numeric WeatherParameters fields for one DOE (Sun, Rain, Fog) level
    triple. Raises ValueError for any level outside the DOE design.
    """

    sun = int(sun)

    if sun not in DOE_SUN_LEVELS:
        raise ValueError(f"Unknown DOE Sun level {sun!r}; expected one of {list(DOE_SUN_LEVELS)}")

    if rain not in RAIN_PROFILES:
        raise ValueError(f"Unknown DOE Rain level {rain!r}; expected one of {list(RAIN_PROFILES)}")

    if fog not in FOG_PROFILES:
        raise ValueError(f"Unknown DOE Fog level {fog!r}; expected one of {list(FOG_PROFILES)}")

    fields = dict(DOE_FIXED_FIELDS)
    fields["sun_altitude_angle"] = float(sun)
    fields.update(RAIN_PROFILES[rain])
    fields.update(FOG_PROFILES[fog])

    return {name: float(fields[name]) for name in DOE_WEATHER_FIELDS}


def make_doe_weather(sun, rain, fog):
    import carla

    return carla.WeatherParameters(**doe_weather_fields(sun, rain, fog))


# ====================================================================
# DOE condition names
# ====================================================================

_DOE_NAME_RE = re.compile(
    r"^doe_(?P<condition_id>C\d{3,})_sun(?P<sun>[+-]?\d+)_rain-(?P<rain>[A-Za-z]+)_fog-(?P<fog>[A-Za-z]+)$"
)


def _format_sun(sun):
    sun = int(sun)
    return f"+{sun}" if sun > 0 else str(sun)


def doe_condition_name(condition_id, sun, rain, fog):
    """E.g. ("C017", 30, "Light", "Heavy") -> doe_C017_sun+30_rain-Light_fog-Heavy."""

    doe_weather_fields(sun, rain, fog)  # validate levels

    name = f"doe_{condition_id}_sun{_format_sun(sun)}_rain-{rain}_fog-{fog}"

    if not _DOE_NAME_RE.match(name):
        raise ValueError(f"Invalid DOE condition id {condition_id!r} (expected e.g. 'C001').")

    return name


def parse_doe_condition_name(name):
    """
    Returns {"condition_id", "sun", "rain", "fog"} for a DOE condition name,
    None for any other string (e.g. the legacy 'day_rain'). A string that
    looks like a DOE name but carries an unknown level raises ValueError.
    """

    match = _DOE_NAME_RE.match(str(name))

    if match is None:
        return None

    parsed = {
        "condition_id": match.group("condition_id"),
        "sun": int(match.group("sun")),
        "rain": match.group("rain"),
        "fog": match.group("fog"),
    }

    doe_weather_fields(parsed["sun"], parsed["rain"], parsed["fog"])

    return parsed


# ====================================================================
# Provenance
# ====================================================================

def weather_profile_definition():
    """JSON-serializable dump of every value this module maps levels to."""

    return {
        "sun_levels": list(DOE_SUN_LEVELS),
        "rain_profiles": {name: dict(values) for name, values in RAIN_PROFILES.items()},
        "fog_profiles": {name: dict(values) for name, values in FOG_PROFILES.items()},
        "fixed_fields": dict(DOE_FIXED_FIELDS),
    }


def weather_profile_hash():
    """sha256 of weather_profile_definition() -- changes iff a mapping value changes."""

    payload = json.dumps(weather_profile_definition(), sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

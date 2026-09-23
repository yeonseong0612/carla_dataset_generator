"""
resume_plan.py

Pure (no CARLA) decision of what work a route needs, so resume / rerender /
overwrite behaviour can be unit tested offline.

Semantics:

    geometry COMPLETE   canonical geometry AND the canonical source
                        condition's RGB (day_clear) finished -- both are
                        written by the single canonical run, so they are
                        only valid together.
    condition COMPLETE  that weather's stereo RGB finished.

The source condition (day_clear) is NEVER scheduled for replay: its RGB comes
from the canonical run itself. Only the other weathers are replayed.
"""

import json
import os

from src.data.layout import condition_dir, geometry_dir, is_complete


def _check_recording_hz_compatibility(route_path, expected_recording_hz):
    """
    10 Hz-recording task (CLAUDE.md section 17): never silently resume a
    route directory whose canonical geometry was recorded at a different
    sample rate than the current cfg.RECORDING.FPS.

    Pre-this-task datasets have no "recording_hz" field at all -- they
    recorded every simulation tick (recording_hz == fps); a missing field
    is treated as that legacy value, not as "unknown/skip the check".
    """

    sequence_path = os.path.join(geometry_dir(route_path), "sequence.json")

    if not os.path.isfile(sequence_path):
        # geometry_ok already requires a COMPLETE marker; a missing
        # sequence.json alongside it is a different (pre-existing) failure
        # mode, not this check's job to diagnose.
        return

    with open(sequence_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    existing_hz = data.get("recording_hz", data.get("fps"))

    if existing_hz is None or abs(float(existing_hz) - float(expected_recording_hz)) > 1e-6:
        raise RuntimeError(
            f"Resume blocked: '{sequence_path}' was recorded at "
            f"recording_hz={existing_hz}, but the current config is "
            f"cfg.RECORDING.FPS={expected_recording_hz}. Mixing sampling "
            f"rates within one route directory is not allowed -- use a "
            f"different --output-root or --overwrite this route."
        )


def plan_route_work(
    route_path,
    conditions,
    source_condition,
    rerender_conditions=(),
    overwrite=False,
    expected_recording_hz=None,
):
    """
    Returns a dict:

        delete_route   remove the whole route directory before working
        run_canonical  run Stage 1 (geometry + source-condition RGB)
        replay         weathers to replay (RGB only), in order
        skip           weathers already COMPLETE and not re-requested

    --overwrite          -> route deleted, canonical re-run, every weather
                            replayed afterwards.
    geometry incomplete  -> same (a partial/stale geometry invalidates every
                            condition rendered from it).
    geometry complete    -> canonical is never re-run; only missing weathers
                            (or those in rerender_conditions) are replayed.
    rerender_conditions  -> replay exactly these (geometry untouched); the
                            source condition cannot be re-rendered by replay.

    expected_recording_hz: when given and geometry is COMPLETE (and
        --overwrite is not set), the existing geometry/sequence.json's
        recording_hz must match this value or a RuntimeError is raised --
        resuming a 20 Hz-recorded route under a 10 Hz config (or vice
        versa) fails loudly instead of silently mixing sample rates.
    """

    rerender = set(rerender_conditions or [])

    if source_condition in rerender:
        raise ValueError(
            f"--rerender-conditions cannot include the canonical source "
            f"condition '{source_condition}': its RGB is produced by the "
            f"canonical run itself. Use --overwrite to regenerate the route."
        )

    unknown = rerender - set(conditions)

    if unknown:
        raise ValueError(
            f"--rerender-conditions {sorted(unknown)} are not among the "
            f"selected conditions {list(conditions)}."
        )

    replayable = [condition for condition in conditions if condition != source_condition]

    geometry_ok = (
        is_complete(geometry_dir(route_path))
        and is_complete(condition_dir(route_path, source_condition))
    )

    if geometry_ok and not overwrite and expected_recording_hz is not None:
        _check_recording_hz_compatibility(route_path, expected_recording_hz)

    if overwrite or not geometry_ok:
        return {
            "delete_route": bool(os.path.exists(route_path)),
            "run_canonical": True,
            "replay": list(replayable),
            "skip": [],
        }

    replay = []
    skip = []

    for condition in replayable:
        if condition in rerender or not is_complete(condition_dir(route_path, condition)):
            replay.append(condition)
        else:
            skip.append(condition)

    return {
        "delete_route": False,
        "run_canonical": False,
        "replay": replay,
        "skip": skip,
    }

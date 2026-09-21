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

import os

from src.data.layout import condition_dir, geometry_dir, is_complete


def plan_route_work(
    route_path,
    conditions,
    source_condition,
    rerender_conditions=(),
    overwrite=False,
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

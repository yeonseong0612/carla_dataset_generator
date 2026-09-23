"""
timing.py

Shared helper for the 20 Hz simulation / 10 Hz dataset-recording split (see
CLAUDE.md). World tick rate, controller, Traffic Manager and Gamma all stay
on cfg.SIMULATION's 20 Hz clock; only a subset of ticks is written to disk.

record_stride_ticks() is the single source of the "every Nth tick" stride so
no call site hard-codes a magic number (e.g. `if tick % 2 == 0`).
"""


def record_stride_ticks(cfg):
    """
    Number of simulation ticks per saved dataset sample
    (cfg.RECORDING.SAMPLE_INTERVAL_SECONDS / cfg.SIMULATION.FIXED_DELTA_SECONDS).

    Raises if the ratio is not an exact integer: a fractional stride would
    mean dataset samples drift out of phase with a fixed 0.1 s cadence.
    """

    ratio = cfg.RECORDING.SAMPLE_INTERVAL_SECONDS / cfg.SIMULATION.FIXED_DELTA_SECONDS
    stride = round(ratio)

    if abs(ratio - stride) > 1e-6:
        raise ValueError(
            "cfg.RECORDING.SAMPLE_INTERVAL_SECONDS must be an exact "
            "multiple of cfg.SIMULATION.FIXED_DELTA_SECONDS "
            f"(got {cfg.RECORDING.SAMPLE_INTERVAL_SECONDS} / "
            f"{cfg.SIMULATION.FIXED_DELTA_SECONDS} = {ratio})."
        )

    if stride < 1:
        raise ValueError(f"record_stride_ticks() computed to {stride} (< 1).")

    return stride


def is_record_tick(simulation_tick_idx, stride_ticks):
    """
    simulation_tick_idx: 0-indexed count of world.tick() calls since the
    main recording loop started (NOT the CARLA server frame number, and NOT
    the saved dataset frame_id -- see CLAUDE.md section 12).

    True on tick 0, stride_ticks, 2*stride_ticks, ... so the very first
    tick of the main loop is always a record tick.
    """

    return simulation_tick_idx % stride_ticks == 0

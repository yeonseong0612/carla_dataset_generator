"""
scripts/tools/check_bus_removal_static.py

Pure-Python static/sanity checks for the "Remove Bus Completely from
Canonical Dataset Traffic" task. Replaces the earlier
check_bus_policy_static.py (deleted -- it tested the frequency-
management policy this task removes: BUS_SUBTYPE_WEIGHT/
MAX_MANAGED_BUSES/MAX_VISIBLE_BUSES/BUS_SAME_LANE_MIN_DISTANCE and their
helpers no longer exist).

CARLA is NOT launched -- only fake stand-in blueprint objects that
satisfy the same duck-typed interface CARLA's own blueprint objects
expose (has_attribute/get_attribute/as_str), exercised against
is_bus_blueprint (src/simulation/traffic.py) and the exact same
filter-then-choose sequence GammaSpawnPolicy.__init__ (src/simulation/
spawn_policy.py) and CanonicalBackgroundTraffic.__init__ (src/
simulation/canonical_traffic.py) each run once on their own local
vehicle_pools["vehicle"].

Run: python scripts/tools/check_bus_removal_static.py
Exit code 0 = all checks passed, 1 = at least one failed.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARLA_PYTHONAPI = Path(r"C:\CARLA") / "PythonAPI" / "carla"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CARLA_PYTHONAPI))

from CFG.config import cfg  # noqa: E402
from src.simulation.traffic import is_bus_blueprint  # noqa: E402


class FakeAttribute:
    def __init__(self, value):
        self._value = value

    def as_str(self):
        return self._value


class FakeBlueprint:
    def __init__(self, base_type, blueprint_id):
        self.id = blueprint_id
        self._base_type = base_type

    def has_attribute(self, name):
        return name == "base_type"

    def get_attribute(self, name):
        return FakeAttribute(self._base_type)


def make_mixed_pool():
    pool = []
    for i in range(6):
        pool.append(FakeBlueprint("car", f"vehicle.fake_car_{i}"))
    for i in range(2):
        pool.append(FakeBlueprint("van", f"vehicle.fake_van_{i}"))
    for i in range(2):
        pool.append(FakeBlueprint("truck", f"vehicle.fake_truck_{i}"))
    for i in range(2):
        pool.append(FakeBlueprint("bus", f"vehicle.fake_bus_{i}"))
    return pool


def filter_bus(pool):
    """The exact filter both GammaSpawnPolicy.__init__ and
    CanonicalBackgroundTraffic.__init__ apply to their own
    self.vehicle_pools["vehicle"]."""

    return [bp for bp in pool if not is_bus_blueprint(bp)]


failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def main():
    print("=" * 78)
    print("Bus removal static/sanity checks (no CARLA connection)")
    print("=" * 78)

    pool = make_mixed_pool()
    filtered = filter_bus(pool)

    # ---- bus blueprints are filtered out ----
    check("mixed pool has 12 blueprints (6 car, 2 van, 2 truck, 2 bus)", len(pool) == 12)
    check("filtered pool excludes all bus blueprints", all(not is_bus_blueprint(bp) for bp in filtered))
    check("filtered pool size == non-bus count (10)", len(filtered) == 10)

    # ---- car/van/truck remain selectable ----
    car_present = any("car" in bp.id for bp in filtered)
    van_present = any("van" in bp.id for bp in filtered)
    truck_present = any("truck" in bp.id for bp in filtered)
    check("car blueprints remain in filtered pool", car_present)
    check("van blueprints remain in filtered pool", van_present)
    check("truck blueprints remain in filtered pool", truck_present)

    # ---- 1000+ simulated draws produce zero buses ----
    rng = np.random.default_rng(cfg.SPAWN.SEED)
    n_draws = 2000
    draws = [filtered[rng.integers(0, len(filtered))] for _ in range(n_draws)]
    bus_draws = [bp for bp in draws if is_bus_blueprint(bp)]
    check(f"{n_draws} draws from the filtered pool -> zero bus blueprints selected", len(bus_draws) == 0,
          detail=f"{len(bus_draws)} bus draws observed")
    check(f"{n_draws} draws still include car/van/truck (not degenerate)",
          len({bp.id for bp in draws}) > 1)

    # ---- bus probability is EXACTLY 0, not just reduced ----
    # (unlike the removed weighted policy, there is no bus in the
    # candidate list at all -- confirm the pool itself, not just a
    # sample, contains none.)
    check("bus probability is exactly 0 (bus blueprint count in filtered pool == 0)",
          sum(1 for bp in filtered if is_bus_blueprint(bp)) == 0)

    # ---- determinism: same seed -> identical draw sequence ----
    rng_a = np.random.default_rng(cfg.SPAWN.SEED)
    rng_b = np.random.default_rng(cfg.SPAWN.SEED)
    seq_a = [filtered[rng_a.integers(0, len(filtered))].id for _ in range(200)]
    seq_b = [filtered[rng_b.integers(0, len(filtered))].id for _ in range(200)]
    check("same-seed rng -> identical 200-draw blueprint sequence (determinism)", seq_a == seq_b)

    # ---- a bus-only pool degrades gracefully (documents the edge case;
    # does not need to "work", just not crash the filter itself) ----
    bus_only_pool = [FakeBlueprint("bus", f"vehicle.fake_bus_only_{i}") for i in range(3)]
    bus_only_filtered = filter_bus(bus_only_pool)
    check("filtering an all-bus pool yields an empty list (no crash, no bus leaks through)",
          bus_only_filtered == [])

    print()
    print("=" * 78)

    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)

    print("RESULT: all checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()

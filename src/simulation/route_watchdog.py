"""
Route-level no-progress watchdog for canonical geometry collection.

Production safety guard, independent of RouteController's vehicle stuck
detector: it only looks at dense-route arc-length progress (ego_s) on the
simulation clock, so speed/throttle/brake changes, traffic-light changes,
red-light waits, sub-km/h jitter and route-index noise never reset it.
Only a real forward progress of >= min_progress_m past the anchor does.
"""


class RouteNoProgressError(RuntimeError):
    """Canonical route made no meaningful route progress within the timeout."""


class RouteProgressWatchdog:

    def __init__(self, timeout_s, min_progress_m):
        self.timeout_s = float(timeout_s)
        self.min_progress_m = float(min_progress_m)
        self.anchor_ego_s = None
        self.last_progress_sim_time = None

    def reset(self, ego_s, sim_time):
        self.anchor_ego_s = float(ego_s)
        self.last_progress_sim_time = float(sim_time)

    def no_progress_duration(self, sim_time):
        return float(sim_time) - self.last_progress_sim_time

    def update(self, ego_s, sim_time):
        """
        Returns True when no meaningful progress happened for timeout_s
        simulation seconds. The anchor only moves forward: a lower ego_s
        (projection noise / small reversing) never moves it back.
        """

        if self.anchor_ego_s is None:
            self.reset(ego_s, sim_time)
            return False

        if float(ego_s) >= self.anchor_ego_s + self.min_progress_m:
            self.reset(ego_s, sim_time)
            return False

        return self.no_progress_duration(sim_time) >= self.timeout_s

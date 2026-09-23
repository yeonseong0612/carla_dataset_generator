import queue

from src.sensors.camera import create_camera_rig_blueprints, create_camera_rig_transforms
from src.sensors.lidar import create_lidar_blueprint, create_lidar_transform
from src.sensors.radar import create_radar_blueprint, create_radar_transform
from src.sensors.utils import spawn_sensor, destroy_sensors

CAMERA_NAMES = ["rgb_left", "rgb_right", "depth", "optical_flow", "semantic"]

# (sensor name, cfg.SENSOR attribute holding that radar's own config).
# "radar" keeps its existing name/config (the front radar) for
# compatibility with existing code and data.
RADAR_RIG_SPECS = [
    ("radar", "RADAR"),
    ("radar_front_left", "RADAR_FRONT_LEFT"),
    ("radar_front_right", "RADAR_FRONT_RIGHT"),
]


# Sensor profiles (which sensors a run spawns).
#
#   canonical : the full GT rig, used once per (town, route) by the canonical
#               geometry run (RGB + depth / semantic / optical flow / LiDAR /
#               3 radars). "radar" is the front radar (kept name).
#   replay    : weather replay only re-renders the stereo RGB pair under the
#               recorded geometry; nothing else is spawned, so no unused
#               sensor callbacks / GPU load and no non-RGB data can be written.
CANONICAL_SENSOR_PROFILE = tuple(
    CAMERA_NAMES + ["lidar"] + [name for name, _cfg_attr in RADAR_RIG_SPECS]
)
REPLAY_SENSOR_PROFILE = ("rgb_left", "rgb_right")

SENSOR_PROFILES = {
    "canonical": CANONICAL_SENSOR_PROFILE,
    "replay": REPLAY_SENSOR_PROFILE,
}


class SensorRig:
    def __init__(self, world, ego, cfg):
        self.world = world
        self.ego = ego
        self.cfg = cfg
        self.sensors = {}
        self.queues = {}

    def _attach_sensor(self, name, blueprint, transform):
        sensor = spawn_sensor(self.world, self.ego, blueprint, transform)
        sensor_queue = queue.Queue()
        sensor.listen(sensor_queue.put)
        self.sensors[name] = sensor
        self.queues[name] = sensor_queue
        return sensor

    def spawn(self, sensor_names=None, profile=None):
        """
        Spawn the full rig (default), a named profile ("canonical" /
        "replay", see SENSOR_PROFILES), or an explicit subset of sensor
        names (diagnostic tools). profile and sensor_names are exclusive.
        """

        if profile is not None:
            if sensor_names is not None:
                raise ValueError("Pass either profile or sensor_names, not both.")

            if profile not in SENSOR_PROFILES:
                raise ValueError(f"Unknown sensor profile: {profile}")

            sensor_names = SENSOR_PROFILES[profile]

        wanted = None if sensor_names is None else set(sensor_names)

        camera_blueprints = create_camera_rig_blueprints(self.world, self.cfg)
        camera_transforms = create_camera_rig_transforms(self.cfg)

        for name in CAMERA_NAMES:
            if wanted is None or name in wanted:
                self._attach_sensor(name, camera_blueprints[name], camera_transforms[name])

        if wanted is None or "lidar" in wanted:
            self._attach_sensor("lidar", create_lidar_blueprint(self.world, self.cfg), create_lidar_transform(self.cfg))

        for sensor_name, cfg_attr in RADAR_RIG_SPECS:
            if wanted is None or sensor_name in wanted:
                radar_cfg = getattr(self.cfg.SENSOR, cfg_attr)
                radar_blueprint = create_radar_blueprint(
                    self.world, radar_cfg, sensor_tick=self.cfg.RECORDING.SAMPLE_INTERVAL_SECONDS,
                )
                self._attach_sensor(sensor_name, radar_blueprint, create_radar_transform(radar_cfg))

        if wanted is not None:
            unknown = wanted - set(self.sensors)

            if unknown:
                raise ValueError(f"Unknown sensor name(s) requested: {sorted(unknown)}")

        return self

    def get_sensor(self, name):
        return self.sensors[name]

    def get_queue(self, name):
        return self.queues[name]

    def get_sensor_names(self):
        return list(self.sensors.keys())

    def clear_queues(self):
        for sensor_queue in self.queues.values():
            while not sensor_queue.empty():
                try:
                    sensor_queue.get_nowait()
                except queue.Empty:
                    break

    def destroy(self):
        destroy_sensors(list(self.sensors.values()))
        self.sensors.clear()
        self.queues.clear()

    def __len__(self):
        return len(self.sensors)

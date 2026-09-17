import queue

from src.sensors.camera import create_camera_rig_blueprints, create_camera_rig_transforms
from src.sensors.lidar import create_lidar_blueprint, create_lidar_transform
from src.sensors.radar import create_radar_blueprint, create_radar_transform
from src.sensors.gnss import create_gnss_blueprint, create_gnss_transform
from src.sensors.imu import create_imu_blueprint, create_imu_transform
from src.sensors.utils import spawn_sensor, destroy_sensors

CAMERA_NAMES = ["rgb_left", "rgb_right", "depth", "optical_flow", "semantic"]

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

    def spawn(self):
        camera_blueprints = create_camera_rig_blueprints(self.world, self.cfg)
        camera_transforms = create_camera_rig_transforms(self.cfg)

        for name in CAMERA_NAMES:
            self._attach_sensor(name, camera_blueprints[name], camera_transforms[name])

        self._attach_sensor("lidar", create_lidar_blueprint(self.world, self.cfg), create_lidar_transform(self.cfg))
        self._attach_sensor("radar", create_radar_blueprint(self.world, self.cfg), create_radar_transform(self.cfg))
        self._attach_sensor("gnss", create_gnss_blueprint(self.world, self.cfg), create_gnss_transform(self.cfg))
        self._attach_sensor("imu", create_imu_blueprint(self.world, self.cfg), create_imu_transform(self.cfg))

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
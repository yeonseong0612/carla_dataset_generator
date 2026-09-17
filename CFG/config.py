from pathlib import Path

from easydict import EasyDict

cfg = EasyDict()

cfg.CARLA = EasyDict()
cfg.CARLA.HOST = "localhost"
cfg.CARLA.PORT = 2000
cfg.CARLA.TIMEOUT = 60.0

cfg.TRAFFIC_MANAGER = EasyDict()
cfg.TRAFFIC_MANAGER.PORT = 8000

cfg.SIMULATION = EasyDict()
cfg.SIMULATION.FPS = 20
cfg.SIMULATION.FIXED_DELTA_SECONDS = 1.0 / cfg.SIMULATION.FPS

cfg.PROJECT = EasyDict()
# CFG/config.py lives at <project_root>/CFG/config.py, so the project
# root is this file's grandparent. Resolved dynamically instead of
# hard-coded so the project can be checked out to any path.
cfg.PROJECT.ROOT = str(Path(__file__).resolve().parents[1])

cfg.MAP = EasyDict()
cfg.MAP.NAME = "Town01"
# Kept in sync with the route XML files actually present under routes/.
cfg.MAP.LIST = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town07", "Town10", "Town12", "Town15"]

cfg.RANDOM = EasyDict()
cfg.RANDOM.SEED = 42


# Sensor configuration

cfg.SENSOR = EasyDict()

#################################################################
### Camera
#################################################################

cfg.SENSOR.CAMERA = EasyDict()
cfg.SENSOR.CAMERA.WIDTH = 640
cfg.SENSOR.CAMERA.HEIGHT = 375
# cfg.SENSOR.CAMERA.WIDTH = 1280
# cfg.SENSOR.CAMERA.HEIGHT = 512
cfg.SENSOR.CAMERA.FOV = 81.8
cfg.SENSOR.CAMERA.FPS = 20

cfg.SENSOR.CAMERA.X = 1.5
cfg.SENSOR.CAMERA.Z = 1.7

cfg.SENSOR.CAMERA.ROLL = 0.0
cfg.SENSOR.CAMERA.PITCH = 0.0
cfg.SENSOR.CAMERA.YAW = 0.0

cfg.SENSOR.STEREO = EasyDict()
cfg.SENSOR.STEREO.BASELINE = 0.54
cfg.SENSOR.STEREO.LEFT_Y = -cfg.SENSOR.STEREO.BASELINE / 2.0
cfg.SENSOR.STEREO.RIGHT_Y = cfg.SENSOR.STEREO.BASELINE / 2.0

#################################################################
###  LiDAR
#################################################################

cfg.SENSOR.LIDAR = EasyDict()
cfg.SENSOR.LIDAR.CHANNELS = 16
cfg.SENSOR.LIDAR.RANGE = 120.0
cfg.SENSOR.LIDAR.POINTS_PER_SECOND = 20000
cfg.SENSOR.LIDAR.ROTATION_FREQUENCY = 10.0
cfg.SENSOR.LIDAR.UPPER_FOV = 2.0
cfg.SENSOR.LIDAR.LOWER_FOV = -24.3

cfg.SENSOR.LIDAR.X = 0.0
cfg.SENSOR.LIDAR.Y = 0.0
cfg.SENSOR.LIDAR.Z = 2.2

cfg.SENSOR.LIDAR.ROLL = 0.0
cfg.SENSOR.LIDAR.PITCH = 0.0
cfg.SENSOR.LIDAR.YAW = 0.0

#################################################################
### Radar
#################################################################

cfg.SENSOR.RADAR = EasyDict()
cfg.SENSOR.RADAR.HORIZONTAL_FOV = 60.0
cfg.SENSOR.RADAR.VERTICAL_FOV = 20.0
cfg.SENSOR.RADAR.RANGE =120.0
cfg.SENSOR.RADAR.POINTS_PER_SECOND = 150000

cfg.SENSOR.RADAR.X = 2.3
cfg.SENSOR.RADAR.Y = 0.0
cfg.SENSOR.RADAR.Z = 0.7

cfg.SENSOR.RADAR.ROLL = 0.0
cfg.SENSOR.RADAR.PITCH = 0.0
cfg.SENSOR.RADAR.YAW = 0.0

cfg.SENSOR.RADAR.FPS = 20

#################################################################
### GNSS
#################################################################

cfg.SENSOR.GNSS = EasyDict()

cfg.SENSOR.GNSS.X = 0.0
cfg.SENSOR.GNSS.Y = 0.0
cfg.SENSOR.GNSS.Z = 2.0

cfg.SENSOR.GNSS.NOISE_LAT_STDDEV = 0.0
cfg.SENSOR.GNSS.NOISE_LON_STDDEV = 0.0
cfg.SENSOR.GNSS.NOISE_ALT_STDDEV = 0.0

cfg.SENSOR.GNSS.NOISE_LAT_BIAS = 0.0
cfg.SENSOR.GNSS.NOISE_LON_BIAS = 0.0
cfg.SENSOR.GNSS.NOISE_ALT_BIAS = 0.0

cfg.SENSOR.GNSS.NOISE_SEED = 42


#################################################################
### IMU
#################################################################

cfg.SENSOR.IMU = EasyDict()

cfg.SENSOR.IMU.X = 0.0
cfg.SENSOR.IMU.Y = 0.0
cfg.SENSOR.IMU.Z = 1.0

cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_X = 0.0
cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_Y = 0.0
cfg.SENSOR.IMU.NOISE_ACCEL_STDDEV_Z = 0.0

cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_X = 0.0
cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_Y = 0.0
cfg.SENSOR.IMU.NOISE_GYRO_STDDEV_Z = 0.0

cfg.SENSOR.IMU.NOISE_GYRO_BIAS_X = 0.0
cfg.SENSOR.IMU.NOISE_GYRO_BIAS_Y = 0.0
cfg.SENSOR.IMU.NOISE_GYRO_BIAS_Z = 0.0

cfg.SENSOR.IMU.NOISE_SEED = 42

#################################################################
### Traffic
#################################################################

cfg.TRAFFIC = EasyDict()
cfg.TRAFFIC.NUM_VEHICLES = 20
cfg.TRAFFIC.NUM_CYCLISTS = 4
cfg.TRAFFIC.NUM_MOTORCYCLISTS = 4

cfg.TRAFFIC.SEED = 42
cfg.TRAFFIC.MIN_DISTANCE_TO_EGO = 15.0

cfg.TRAFFIC.SPEED_DIFFERENCE = 0.0
cfg.TRAFFIC.AUTO_LANE_CHANGE = True

cfg.TRAFFIC.IGNORE_LIGHTS_PERCENTAGE = 0.0
cfg.TRAFFIC.IGNORE_SIGNS_PERCENTAGE = 0.0
cfg.TRAFFIC.IGNORE_VEHICLES_PERCENTAGE = 0.0
cfg.TRAFFIC.IGNORE_WALKERS_PERCENTAGE = 0.0

#################################################################
### Pedestrian
#################################################################

cfg.PEDESTRIAN = EasyDict()
cfg.PEDESTRIAN.NUM_WALKERS = 80
cfg.PEDESTRIAN.SEED = 42
cfg.PEDESTRIAN.RUN_PERCENTAGE = 0.0
cfg.PEDESTRIAN.CROSS_PERCENTAGE = 0.1

#################################################################
### Annotation
#################################################################

cfg.ANNOTATION = EasyDict()
cfg.ANNOTATION.MAX_DISTANCE = 100.0

cfg.ANNOTATION.CLASSES = ["pedestrian", "vehicle", "cyclist", "motorcyclist"]

cfg.ANNOTATION.VEHICLE_SUBTYPES = ["car", "van", "truck", "bus"]

#################################################################
### weather conditions
#################################################################

cfg.WEATHER = EasyDict()
cfg.WEATHER.DEFAULT = "day_clear"
cfg.WEATHER.CONDITIONS = ["day_clear", "day_rain", "day_fog", "night_clear", "night_rain", "night_fog"]
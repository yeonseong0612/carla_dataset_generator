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

#################################################################
### Dataset recording rate (10 Hz-recording task)
###
### World / physics / controller / Traffic Manager / Gamma all stay on
### cfg.SIMULATION's 20 Hz clock above -- untouched by this section. Only
### the final saved dataset SAMPLE rate is lower: every RECORD_STRIDE_TICKS
### -th world tick is written to disk (RGB/depth/semantic/flow/LiDAR/radar/
### annotation/world_state/pose), the ticks in between still run controller/
### traffic/physics but are never collected or saved. See
### src/simulation/timing.py is_record_tick() / scripts/collect_dataset.py.
#################################################################
cfg.RECORDING = EasyDict()
cfg.RECORDING.FPS = 10
cfg.RECORDING.SAMPLE_INTERVAL_SECONDS = 1.0 / cfg.RECORDING.FPS

# = SAMPLE_INTERVAL_SECONDS / FIXED_DELTA_SECONDS (0.1 / 0.05 = 2). Must be
# an exact integer -- RECORDING.FPS is required to evenly divide
# SIMULATION.FPS; see the assertion in src/simulation/timing.py.
cfg.RECORDING.STRIDE_TICKS = round(
    cfg.RECORDING.SAMPLE_INTERVAL_SECONDS / cfg.SIMULATION.FIXED_DELTA_SECONDS
)

cfg.PROJECT = EasyDict()
cfg.PROJECT.ROOT = str(Path(__file__).resolve().parents[1])

cfg.MAP = EasyDict()
cfg.MAP.NAME = "Town01"
cfg.MAP.LIST = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town07", "Town10", "Town12", "Town15"]

cfg.RANDOM = EasyDict()
cfg.RANDOM.SEED = 42


cfg.SENSOR = EasyDict()

#################################################################
### Camera
#################################################################

cfg.SENSOR.CAMERA = EasyDict()
# cfg.SENSOR.CAMERA.WIDTH = 640
# cfg.SENSOR.CAMERA.HEIGHT = 375
cfg.SENSOR.CAMERA.WIDTH = 1024
cfg.SENSOR.CAMERA.HEIGHT = 768
cfg.SENSOR.CAMERA.FOV = 60.0
cfg.SENSOR.CAMERA.FPS = 20

cfg.SENSOR.CAMERA.X = 1.5
cfg.SENSOR.CAMERA.Z = 1.7

cfg.SENSOR.CAMERA.ROLL = 0.0
cfg.SENSOR.CAMERA.PITCH = 0.0
cfg.SENSOR.CAMERA.YAW = 0.0

cfg.SENSOR.STEREO = EasyDict()
cfg.SENSOR.STEREO.BASELINE = 0.24
cfg.SENSOR.STEREO.LEFT_Y = -cfg.SENSOR.STEREO.BASELINE / 2.0
cfg.SENSOR.STEREO.RIGHT_Y = cfg.SENSOR.STEREO.BASELINE / 2.0

#################################################################
###  LiDAR
#################################################################
cfg.SENSOR.LIDAR = EasyDict()

cfg.SENSOR.LIDAR.CHANNELS = 16

cfg.SENSOR.LIDAR.RANGE = 130.0

# 10 Hz-recording task: LiDAR now captures once per cfg.RECORDING
# sample (sensor_tick = RECORDING.SAMPLE_INTERVAL_SECONDS, see
# src/sensors/lidar.py), so rotation_frequency is set to RECORDING.FPS (was
# SIMULATION.FPS = 20) -- one full 360 deg sweep per saved LiDAR sample,
# same "one callback = one complete scan" semantics as before, just at the
# new 10 Hz sample rate instead of the old 20 Hz one.
#
# Verified directly against a live CARLA 0.9.16 server (not assumed from
# docs, per CLAUDE.md section 7): scaling sensor_tick and rotation_frequency
# down together like this keeps the PER-SCAN point count statistically
# unchanged from the old 20 Hz config (~4.68k pts/scan in a static Town01
# control scan either way) -- each delivered scan is still exactly one full
# rotation over the same scene, so it carries the same point count; only
# the callback rate halves (20 Hz -> 10 Hz), which is the actual efficiency
# win. points_per_second itself is unchanged.
cfg.SENSOR.LIDAR.POINTS_PER_SECOND = 200000
cfg.SENSOR.LIDAR.ROTATION_FREQUENCY = float(cfg.RECORDING.FPS)

cfg.SENSOR.LIDAR.HORIZONTAL_FOV = 180.0

cfg.SENSOR.LIDAR.UPPER_FOV = 15.0
cfg.SENSOR.LIDAR.LOWER_FOV = -15.0

cfg.SENSOR.LIDAR.ROI_FRONT_MIN = 0.0
cfg.SENSOR.LIDAR.ROI_FRONT_MAX = 120.0
cfg.SENSOR.LIDAR.ROI_SIDE = 40.0

cfg.SENSOR.LIDAR.X = 1.5
cfg.SENSOR.LIDAR.Y = 0.0
cfg.SENSOR.LIDAR.Z = 1.6

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

cfg.SENSOR.RADAR.POINTS_PER_SECOND = 100000

cfg.SENSOR.RADAR.X = 2.3
cfg.SENSOR.RADAR.Y = 0.0
cfg.SENSOR.RADAR.Z = 0.7

cfg.SENSOR.RADAR.ROLL = 0.0
cfg.SENSOR.RADAR.PITCH = 0.0
cfg.SENSOR.RADAR.YAW = 0.0

cfg.SENSOR.RADAR.FPS = 20

cfg.SENSOR.RADAR_FRONT_LEFT = EasyDict()
cfg.SENSOR.RADAR_FRONT_LEFT.HORIZONTAL_FOV = 90.0
cfg.SENSOR.RADAR_FRONT_LEFT.VERTICAL_FOV = 20.0
cfg.SENSOR.RADAR_FRONT_LEFT.RANGE = 60.0
cfg.SENSOR.RADAR_FRONT_LEFT.POINTS_PER_SECOND = 60000

cfg.SENSOR.RADAR_FRONT_LEFT.X = 2.0
cfg.SENSOR.RADAR_FRONT_LEFT.Y = -0.9
cfg.SENSOR.RADAR_FRONT_LEFT.Z = 0.5

cfg.SENSOR.RADAR_FRONT_LEFT.ROLL = 0.0
cfg.SENSOR.RADAR_FRONT_LEFT.PITCH = 0.0
cfg.SENSOR.RADAR_FRONT_LEFT.YAW = -45.0

cfg.SENSOR.RADAR_FRONT_LEFT.FPS = 20

cfg.SENSOR.RADAR_FRONT_RIGHT = EasyDict()
cfg.SENSOR.RADAR_FRONT_RIGHT.HORIZONTAL_FOV = 90.0
cfg.SENSOR.RADAR_FRONT_RIGHT.VERTICAL_FOV = 20.0
cfg.SENSOR.RADAR_FRONT_RIGHT.RANGE = 60.0
cfg.SENSOR.RADAR_FRONT_RIGHT.POINTS_PER_SECOND = 60000

cfg.SENSOR.RADAR_FRONT_RIGHT.X = 2.0
cfg.SENSOR.RADAR_FRONT_RIGHT.Y = 0.9
cfg.SENSOR.RADAR_FRONT_RIGHT.Z = 0.5

cfg.SENSOR.RADAR_FRONT_RIGHT.ROLL = 0.0
cfg.SENSOR.RADAR_FRONT_RIGHT.PITCH = 0.0
cfg.SENSOR.RADAR_FRONT_RIGHT.YAW = 45.0

cfg.SENSOR.RADAR_FRONT_RIGHT.FPS = 20



#################################################################
### Traffic
#################################################################

cfg.TRAFFIC = EasyDict()
cfg.TRAFFIC.NUM_VEHICLES = 40
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
### Spawn Policy (Phase 1 -- route-relative Gamma initial placement)
#################################################################

cfg.SPAWN = EasyDict()

cfg.SPAWN.SEED = 42

cfg.SPAWN.GAMMA_SHAPE = 2.0
cfg.SPAWN.GAMMA_SCALE = 15.0

cfg.SPAWN.MIN_DISTANCE = 10.0
cfg.SPAWN.MAX_DISTANCE = 100.0

cfg.SPAWN.N_VEHICLES = 25
cfg.SPAWN.N_MOTORCYCLES = 4
cfg.SPAWN.N_BICYCLES = 2
cfg.SPAWN.N_PEDESTRIANS = 8

cfg.SPAWN.MIN_EGO_SPACING = 10.0

cfg.SPAWN.MIN_VEHICLE_SPACING = 8.0
cfg.SPAWN.MIN_CROSS_LANE_SPACING = 3.0

# Same-lane, in-front-of-ego candidates closer than this (route-relative
# longitudinal distance, not Euclidean) are rejected at spawn time only --
# see src.simulation.spawn_policy.same_lane_front_gap_ok(), shared by
# initial placement (GammaSpawnPolicy) and canonical runtime buffer
# replenishment (CanonicalBackgroundTraffic). Adjacent/opposite lanes and
# actors behind ego are unaffected. Does not despawn/teleport actors that
# later approach ego naturally while driving.
# Raised 25.0 -> 80.0 (traffic-generation final tuning task) to further
# reduce how often a single same-lane lead vehicle dominates the camera
# view; the rule/paths are unchanged, only this threshold moved.
cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M = 80.0

# Ticks to advance the simulation (world.tick(), no controls/recording)
# after ego + initial traffic + traffic-manager configuration are all in
# place, but before frame_id=0 is recorded -- lets background traffic
# leave its just-spawned at-rest state before the dataset starts, without
# moving ego or writing any warm-up frame to disk. See
# scripts/collect_dataset.py generate_canonical_geometry(). At
# cfg.SIMULATION.FIXED_DELTA_SECONDS (1/20 s), 20 ticks = 1 simulation
# second.
cfg.SPAWN.PRE_RECORD_WARMUP_TICKS = 20

cfg.SPAWN.MIN_PEDESTRIAN_SPACING = 2.5

cfg.SPAWN.MAX_ATTEMPTS = 20

#################################################################
### Spawn Policy Phase 2 -- dynamic density maintenance
#################################################################

cfg.SPAWN.BIN_SIZE = 10.0

cfg.SPAWN.UPDATE_INTERVAL_FRAMES = 20
cfg.SPAWN.MAX_NEW_ACTORS_PER_UPDATE = 3

cfg.SPAWN.DESPAWN_BEHIND_DISTANCE = 40.0
cfg.SPAWN.FORWARD_CLEANUP_DISTANCE = 150.0

#################################################################
### Spawn Policy Phase 2.5 -- dynamic spawn stability fixes
#################################################################

cfg.SPAWN.MAX_PROJECTION_FAILURE_UPDATES = 10


cfg.SPAWN.MAX_ROUTE_PROJECTION_DISTANCE = 20.0


cfg.SPAWN.MIN_SAME_LANE_EGO_SPAWN_DISTANCE = 30.0

#################################################################
### Canonical Background Traffic Policy
### (src/simulation/canonical_traffic.py -- replaces Phase 2/2.5's
### per-bin Gamma-deficit replenishment as the production default;
### DynamicSpawnManager above is kept unmodified/unused, not deleted)
#################################################################

# Visible/training ROI reuses cfg.SENSOR.LIDAR.ROI_FRONT_MAX (0-120m) --
# not duplicated here. New actors are never spawned inside it; they only
# ever enter it by natural forward motion after being placed in the
# buffer zone ahead (120m < relative_s <= CANONICAL_BUFFER_MAX).
# Despawn-behind reuses cfg.SPAWN.DESPAWN_BEHIND_DISTANCE (40m).
cfg.SPAWN.CANONICAL_BUFFER_MAX = 160.0

# Each background vehicle gets one fixed +/- this percent TM
# vehicle_percentage_speed_difference, drawn once at spawn time from the
# seeded RNG (deterministic, not TM's own randomization) -- enough that a
# stream of vehicles isn't perfectly lockstep, without introducing
# cut-in/lane-change interactions (auto_lane_change is forced OFF).
cfg.SPAWN.CANONICAL_SPEED_JITTER_PCT = 10.0

# Bus is excluded from spawn eligibility entirely (task: "Remove Bus
# Completely from Canonical Dataset Traffic") -- see is_bus_blueprint in
# src/simulation/traffic.py, applied in both GammaSpawnPolicy.__init__
# (src/simulation/spawn_policy.py) and CanonicalBackgroundTraffic.
# __init__ (src/simulation/canonical_traffic.py). No config parameter
# needed for a hard exclusion; a prior frequency-management version of
# this policy (BUS_SUBTYPE_WEIGHT/MAX_MANAGED_BUSES/MAX_VISIBLE_BUSES/
# BUS_SAME_LANE_MIN_DISTANCE) was tried and removed after live
# validation showed an unweighted Phase-1 bus could still dominate the
# scene -- see outputs/bus_policy_live_validation/ for that record.

#################################################################
### Frame-Level Gamma Object Count Policy
### (src/simulation/canonical_traffic.py FrameObjectGammaSchedule --
### THIS is the production density target: Gamma is applied to
### N_objects(frame), NOT to object distance. cfg.SPAWN.GAMMA_SHAPE/
### GAMMA_SCALE/MIN_DISTANCE/MAX_DISTANCE above stay distance-domain and
### are kept only for Phase 1's lane/waypoint sampling + backward
### compatibility -- they are no longer the density-shaping target in
### canonical production mode.)
#################################################################

# target ~ round(Gamma(shape, scale)), clipped to
# [FRAME_OBJECT_MIN, FRAME_OBJECT_MAX]. mode = (shape-1)*scale = 8,
# mean = shape*scale = 9 -- tune against the actual histogram in
# outputs/frame_object_gamma_validation/, not hard-coded.
#
# Recalibration ("Recalibrate Gamma Target to Camera-Valid Object
# Count" task): the prior shape=5.0/scale=2.0/min=2/max=18 (mode=8,
# mean=10) was calibrated BEFORE the Gamma controller's actual_count
# input switched to the camera-valid annotation count (see
# src/data/annotation.py AnnotationWriter._compute_camera_validity /
# get_annotation_candidate_count). Camera-valid filtering roughly halves
# the visible count vs. the old distance-only basis (~50% retention,
# see outputs/final_integration_validation/), so the OLD target (mean
# 12.03 once segment sampling is included) sat far above what the new,
# stricter count could ever reach -- observed live: mean_actual=5.45,
# within +/-2 only 12.8%, controller stuck in a near-permanent
# under-target/spawn-more state (population drifting upward, 11->30
# over 1500 frames, 0 prunes the entire run -- see CLAUDE.md task
# history / outputs/final_integration_validation/final_integration_
# summary.json for the full diagnosis). shape/scale/min/max are the
# ONLY things this recalibration changes -- FrameObjectGammaSchedule's
# sampling code, the spawn/prune controller, and the camera-valid
# filtering pipeline are all untouched.
cfg.SPAWN.FRAME_OBJECT_GAMMA_SHAPE = 9.0
cfg.SPAWN.FRAME_OBJECT_GAMMA_SCALE = 1.0

cfg.SPAWN.FRAME_OBJECT_MIN = 4
cfg.SPAWN.FRAME_OBJECT_MAX = 15

# A single Gamma-drawn target is held for a whole segment of consecutive
# frames (segment length itself uniform-random in this range), never
# resampled every frame -- resampling every frame makes the target
# whiplash frame-to-frame, which the spawn/despawn control loop can never
# actually track and just produces unnatural traffic churn.
#
# Controller-fix task: long-run validation (outputs/frame_object_gamma_
# longrun_3000/) measured actor lifetime ~15-16s against the old 2-5s
# (40-100 frame) segment duration -- a 4-5x time-scale mismatch that let
# managed population grow unboundedly across segments (11->33) and
# eventually hard-stalled ego around frame ~780. Raised to bring segment
# duration close to the measured actor lifetime instead.
cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MIN = 240
cfg.SPAWN.FRAME_OBJECT_TARGET_INTERVAL_MAX = 400

# Gradual convergence: caps how many new buffer-zone actors
# CanonicalBackgroundTraffic may spawn in a single update() call to close
# a frame-object deficit (never spawns the whole deficit in one update).
cfg.SPAWN.FRAME_OBJECT_MAX_NEW_PER_UPDATE = 2

# Symmetric downward control (controller-fix task): caps how many
# not-yet-visible ("future entrant") buffer-zone actors
# CanonicalBackgroundTraffic may prune in a single update() call to close
# an over-target excess. Visible-ROI actors are never eligible -- see
# CanonicalBackgroundTraffic.update()'s pruning block.
cfg.SPAWN.FRAME_OBJECT_MAX_PRUNE_PER_UPDATE = 2

#################################################################
### Traffic Light Cycle (production town initialization, applied once
### per town load -- see scripts/collect_dataset.py
### configure_traffic_lights(). CARLA-managed group/state transition
### logic itself is untouched; only each state's duration is shortened
### to reduce how long a route sits stopped at red.)
#################################################################

cfg.TRAFFIC_LIGHT = EasyDict()
cfg.TRAFFIC_LIGHT.GREEN_TIME_S = 8.0
cfg.TRAFFIC_LIGHT.YELLOW_TIME_S = 2.0
cfg.TRAFFIC_LIGHT.RED_TIME_S = 8.0

#################################################################
### Annotation
#################################################################

cfg.ANNOTATION = EasyDict()
cfg.ANNOTATION.MAX_DISTANCE = 100.0

cfg.ANNOTATION.CLASSES = ["pedestrian", "vehicle", "cyclist", "motorcyclist"]

cfg.ANNOTATION.VEHICLE_SUBTYPES = ["car", "van", "truck", "bus"]

#################################################################
### Camera-valid annotation filtering (left RGB camera only -- see
### src/data/annotation.py AnnotationWriter / src/data/projection.py)
#################################################################

# image_fraction = (projected bbox area clipped to the image) /
# (unclipped projected bbox area). Filters actors that are mostly
# outside the frame / truncated at its edge.
cfg.ANNOTATION.MIN_IMAGE_FRACTION = 0.20

# visible_fraction = (pixels within the clipped 2D bbox whose depth-
# image value is not nearer than the object's own closest projected
# vertex) / (total pixels within the clipped 2D bbox). Filters actors
# mostly hidden behind something closer to the camera.
cfg.ANNOTATION.MIN_VISIBLE_FRACTION = 0.20

# Minimum pixel footprint of the final CLIPPED 2D bbox. All three must
# hold, or the actor is filtered as too_small.
cfg.ANNOTATION.MIN_BBOX_WIDTH_PX = 5
cfg.ANNOTATION.MIN_BBOX_HEIGHT_PX = 10
cfg.ANNOTATION.MIN_VISIBLE_AREA_PX = 50

#################################################################
### weather conditions
#################################################################

cfg.WEATHER = EasyDict()
cfg.WEATHER.DEFAULT = "day_clear"
cfg.WEATHER.CONDITIONS = ["day_clear", "day_rain", "day_fog", "night_clear", "night_rain", "night_fog"]
현재 CARLA dataset generator의 traffic generation을 최종 조정하려고 합니다.

이번 작업의 목적은 정확히 다음 3가지입니다.

1. Ego와 동일 진행 lane의 전방 80 m에 차량류가 spawn되지 않도록 강화
2. Dataset recording 시작 직후 정지 차량이 많이 보이는 문제를 줄이기 위해 짧은 pre-recording traffic warm-up 추가
3. Traffic light cycle을 짧게 조정하여 신호 대기로 route가 오래 정체되는 현상 완화

중요:
- 이번 작업은 traffic behavior만 최소 수정합니다.
- annotation bug fix는 이미 완료되었으므로 건드리지 마세요.
- canonical/replay architecture, sensor suite, weather, dataset layout은 변경하지 마세요.
- runtime 중 가까워졌다는 이유로 차량을 destroy/teleport하지 마세요.
- 테스트는 최대한 빠르게 진행하세요.
- full 6-weather replay는 이번 검증에서 필요 없습니다.


======================================================================
PART A. SAME-LANE FRONT SPAWN EXCLUSION을 80 m로 강화
======================================================================

현재 다음 logic이 이미 구현되어 있습니다.

- 공용 helper:
  same_lane_front_gap_ok()

- initial spawn:
  GammaSpawnPolicy

- runtime spawn:
  CanonicalBackgroundTraffic

- config:
  cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M = 25.0

이 구조 자체는 이미 regression test를 통과했습니다.

따라서 구조를 다시 만들지 말고 기존 값을:

    25.0 m
        ↓
    80.0 m

로 변경하세요.

즉:

    cfg.SPAWN.MIN_SAME_LANE_FRONT_GAP_M = 80.0

로 설정합니다.


----------------------------------------------------------------------
A-1. 적용 조건
----------------------------------------------------------------------

다음 경우에만 reject:

- actor category가 vehicle 계열
  - car
  - truck / van
  - motorcycle
  - bicycle
- ego와 동일 진행 lane
- ego보다 앞쪽
- longitudinal forward distance:
      0 < relative_s < 80 m

그러면 spawn candidate를 reject.


----------------------------------------------------------------------
A-2. 유지해야 할 경우
----------------------------------------------------------------------

아래는 기존 정책 유지:

- adjacent lane
- opposite-direction lane
- ego 뒤쪽
- pedestrian
- 교차로를 가로지르는 다른 lane actor

즉 단순 Euclidean radius 80 m로 모든 actor를 막으면 안 됩니다.


----------------------------------------------------------------------
A-3. Initial / runtime 모두 동일 적용
----------------------------------------------------------------------

반드시:

[Spawn]

과

[Canonical-Spawn]

두 경로 모두 기존 same_lane_front_gap_ok() helper를 재사용해야 합니다.

중복 조건을 새로 만들지 마세요.


----------------------------------------------------------------------
A-4. runtime despawn 금지
----------------------------------------------------------------------

이미 주행 중 자연스럽게:

80 m
→ 50 m
→ 20 m
→ 10 m

까지 가까워지는 경우는 허용합니다.

신호 대기나 자연 traffic interaction일 수 있기 때문입니다.

다음 logic은 추가하지 마세요.

distance < X → destroy
bbox large → destroy
dominance → teleport


======================================================================
PART B. PRE-RECORDING TRAFFIC WARM-UP
======================================================================

현재 초기 actor spawn 직후 곧바로 dataset recording을 시작하기 때문에
초반 프레임에서 NPC 차량이 아직 정지해 있는 장면이 발생할 수 있습니다.

이를 줄이기 위해 recording 전에 짧은 warm-up을 추가하세요.


----------------------------------------------------------------------
B-1. Config
----------------------------------------------------------------------

production config에 다음 값을 추가하세요.

권장:

    cfg.SPAWN.PRE_RECORD_WARMUP_TICKS = 20

현재 simulation은:

    fixed_delta_seconds = 0.05

이므로:

    20 ticks = 1 simulation second

입니다.

우선 1초만 사용하세요.

불필요하게 2~5초로 늘리지 마세요.


----------------------------------------------------------------------
B-2. Warm-up 위치
----------------------------------------------------------------------

순서:

world load
→ static environment cleanup
→ ego spawn
→ initial traffic spawn
→ Traffic Manager / controllers 설정
→ PRE_RECORD_WARMUP_TICKS 실행
→ sensor / recording synchronization 정리
→ frame_id = 0부터 실제 dataset 기록

중요:

warm-up frame은 dataset에 저장되면 안 됩니다.

아래에 포함하지 마세요.

- RGB
- depth
- semantic
- optical flow
- LiDAR
- radar
- annotation
- world_state
- pose
- frame count


----------------------------------------------------------------------
B-3. Ego behavior
----------------------------------------------------------------------

가능하면 warm-up 동안 ego vehicle은 초기 위치에 그대로 유지하고
background traffic만 안정화시키세요.

즉 warm-up 때문에 route 시작 위치가 달라지면 안 됩니다.

현재 RouteController가 recording 시작 후 작동하는 구조라면
그 흐름을 그대로 유지하세요.

ego를 억지로 이동시킨 뒤 다시 teleport하지 마세요.


----------------------------------------------------------------------
B-4. Sensor queue 문제 방지
----------------------------------------------------------------------

warm-up 중 sensor가 이미 spawn된 상태라면
warm-up frame이 sensor queue에 남아서
recording frame 0과 섞이면 안 됩니다.

가능하면:

traffic warm-up
→ 그 후 production sensors spawn

순서가 가장 깔끔합니다.

현재 architecture상 sensors를 먼저 만들어야 한다면
warm-up 종료 후 queue를 확실히 flush/reset하세요.

기존 frame synchronization invariant를 깨지 마세요.


======================================================================
PART C. TRAFFIC LIGHT CYCLE 단축
======================================================================

현재 route가 traffic light에서 너무 오래 기다리는 상황을 줄이고 싶습니다.

CARLA 0.9.15에서 사용하는 실제 TrafficLight API를 확인한 뒤
production town initialization에 traffic light timing configuration을 추가하세요.

CARLA API가 지원한다면 기존:

set_green_time()
set_yellow_time()
set_red_time()

를 사용하세요.


----------------------------------------------------------------------
C-1. Config
----------------------------------------------------------------------

CFG/config.py의 traffic 관련 section에 다음 production parameter를 추가하세요.

권장 시작값:

    GREEN_TIME_S  = 8.0
    YELLOW_TIME_S = 2.0
    RED_TIME_S    = 8.0

즉 신호를 매우 비현실적으로 빠르게 깜빡이게 하지 않고,
기존보다 대기만 줄이는 수준으로 설정합니다.

예:

cfg.TRAFFIC_LIGHT.GREEN_TIME_S = 8.0
cfg.TRAFFIC_LIGHT.YELLOW_TIME_S = 2.0
cfg.TRAFFIC_LIGHT.RED_TIME_S = 8.0

현재 config style에 맞게 naming은 조정해도 됩니다.


----------------------------------------------------------------------
C-2. 적용 위치
----------------------------------------------------------------------

town이 load된 직후 한 번만 적용하세요.

예:

client.load_world()
→ static environment cleanup
→ configure_traffic_lights()
→ Traffic Manager setup
→ route loop

각 frame마다 traffic-light duration을 다시 설정하지 마세요.


----------------------------------------------------------------------
C-3. Traffic light group/state logic은 건드리지 말 것
----------------------------------------------------------------------

중요:

이번 작업에서는:

- traffic light state 강제 Green
- traffic light freeze
- traffic light trigger volume 수정
- group sequence 변경
- 모든 신호를 동시에 Green

같은 방식은 사용하지 마세요.

오직 각 state duration만 단축하세요.

CARLA가 관리하는 traffic light group state transition은 그대로 유지합니다.


----------------------------------------------------------------------
C-4. Logging
----------------------------------------------------------------------

town load 시 한 번만 아래처럼 출력하세요.

예:

[TrafficLight] configured 36 lights:
green=8.0s yellow=2.0s red=8.0s

API가 일부 traffic light에서 실패하면
조용히 무시하지 말고 count와 reason을 출력하세요.


======================================================================
PART D. OFFLINE TEST — 최대한 빠르게
======================================================================

전체 CARLA runtime 전에 offline test부터 수행하세요.

기존 전체 72개를 다시 모두 돌려도 오래 걸리지 않는다면 실행하세요.

그 외 최소한 신규 관련 test만 반드시 실행하세요.


----------------------------------------------------------------------
D-1. Spawn gap tests
----------------------------------------------------------------------

기존 same-lane tests를 80 m 기준으로 업데이트하세요.

필수:

same lane front 10m  → reject
same lane front 25m  → reject
same lane front 79.9m → reject
same lane front 80.0m → accept
same lane front 100m → accept

adjacent lane 10m → unaffected / accept by this rule

opposite lane → unaffected

same lane behind → unaffected

pedestrian → unaffected


----------------------------------------------------------------------
D-2. Warm-up tests
----------------------------------------------------------------------

가능하면 pure/helper-level test로:

- configured warmup ticks = 20
- recording frame count에는 warm-up frame 미포함
- frame_id starts from 0 after warm-up

를 검증하세요.

큰 mocking framework를 추가하지 마세요.


----------------------------------------------------------------------
D-3. Traffic light tests
----------------------------------------------------------------------

CARLA 없이 가능한 부분은:

- default config 8 / 2 / 8
- traffic-light configure helper가 get_actors().filter(...) 결과를 순회
- set_green_time
- set_yellow_time
- set_red_time

를 정확히 호출하는지 fake/mock object로 간단히 검사하세요.


======================================================================
PART E. FAST RUNTIME TEST
======================================================================

이번 변경은 weather/replay와 무관하므로
200-frame day_clear+day_rain 전체 smoke를 다시 할 필요 없습니다.

최대한 빠르게:

Town01
route 1
day_clear ONLY
100 frames

만 생성하세요.

예:

python scripts/collect_dataset.py \
    --towns Town01 \
    --routes 1 \
    --conditions day_clear \
    --max-frames 100 \
    --truncate-ok \
    --overwrite \
    --output-root outputs/traffic_policy_fast_test

실제 CLI 형식에 맞게 Windows PowerShell command도 제공하세요.


----------------------------------------------------------------------
E-1. 빠른 runtime PASS 기준
----------------------------------------------------------------------

다음만 확인하세요.

1. canonical validation PASS
2. frame sync mismatch 0
3. initial [Spawn]에서 same-lane front <80 m 차량 없음
4. runtime [Canonical-Spawn]에서 same-lane front <80 m 차량 없음
5. adjacent/opposite lane의 가까운 차량은 허용됨
6. PRE_RECORD_WARMUP_TICKS=20 실제 적용
7. first recorded frame 이전에 warm-up 완료
8. Traffic light timing:
   green=8
   yellow=2
   red=8
9. no exception
10. Failed=0


======================================================================
PART F. 초기 정지 차량 quick diagnostic
======================================================================

첫 recorded frame에서 managed vehicle velocity를 확인하세요.

최소:

frame 0
frame 1
frame 5
frame 10

에 대해:

- total managed vehicles
- moving vehicles
- stationary vehicles
- stationary ratio

를 diagnostic log로 출력하세요.

stationary 기준 예:

speed < 0.5 m/s

단, red traffic light에 의해 정상적으로 정지한 차량과
초기화가 안 돼 정지한 차량을 가능하면 구분하세요.

이번 작업에서 stationary actor를 강제로 삭제하지는 마세요.

warm-up으로 상태가 개선되는지만 확인합니다.


======================================================================
PART G. FOREGROUND DOMINANCE QUICK CHECK
======================================================================

100-frame fast test가 끝나면 기존:

scripts/tools/analyze_foreground_dominance.py

를 실행하세요.

확인:

frames_bbox_fraction_gt_0.30
ratio
longest_dominance_event_frames
num_long_events

기존 수정 후 200-frame smoke baseline:

ratio = 0.04
longest = 8 frames
num_long_events = 0

새 80m 정책에서는 적어도
장시간 한 차량이 camera center를 지배하는 현상이 다시 증가하지 않는지 확인하세요.

100-frame sample이라 통계적 비교를 과도하게 해석하지 마세요.


======================================================================
PART H. 변경하지 말아야 할 것
======================================================================

절대 변경하지 마세요.

- annotation logic
- depth visibility thresholds
- Gamma distribution
- target object count
- category proportions
- adjacent lane spawn policy
- opposite lane spawn policy
- speed jitter
- replay architecture
- RGB-only replay
- wind
- weather parameter
- sensor calibration
- camera resolution
- route XML
- dataset layout


======================================================================
PART I. 최종 보고
======================================================================

작업 후 다음만 간결하게 보고하세요.

1. 수정 파일 목록

2. Same-lane exclusion
- 기존 25m
- 변경 후 80m
- initial/runtime 양쪽 적용 확인

3. Warm-up
- ticks
- simulation seconds
- sensors/recording보다 어느 시점에 적용되는지

4. Traffic lights
- 적용 API
- green/yellow/red durations
- configured light count

5. Offline tests
- PASS / FAIL
- test count

6. Fast 100-frame CARLA smoke
- elapsed time
- Validation PASS 여부
- Failed count

7. 실제 spawn 확인
- closest initial same-lane front vehicle distance
- closest runtime-spawn same-lane front vehicle distance
- <80m violation count

8. Initial stationary actor diagnostic
- first recorded frames의 stationary ratio

9. Foreground dominance
- ratio
- longest event
- long event count

10. regression 여부

11. 최종 판정:

READY FOR PRODUCTION

READY WITH CAVEATS

NOT READY


======================================================================
핵심 원칙
======================================================================

이번 정책의 목적은:

"ego 진행 방향의 중앙 시야를 최대한 확보하면서,
옆 차선/반대 차선/보행자 등 주변 perception diversity는 유지"

하는 것입니다.

따라서 ego same-lane 전방 traffic은 80m까지 spawn하지 않되,
전체 traffic density 자체를 크게 줄이면 안 됩니다.

또한 신호등은 제거하거나 무시하는 것이 아니라
cycle duration만 8/2/8초로 단축합니다.

테스트는 최대한 빠르게 수행하고,
이번 변경과 무관한 day_rain replay나 full-route test는 하지 마세요.
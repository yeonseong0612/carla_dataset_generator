# Remote Server CARLA 0.9.15 Validation

This document is the exact sequence to run on the remote server after
copying this repository over, to confirm CARLA 0.9.15 compatibility,
static-environment-object cleanup, synchronization, and paired
canonical/replay correctness, before any real dataset collection begins.

Everything here was prepared without a live CARLA connection (see the
accompanying audit report). **No CARLA 0.9.15 runtime test has actually been
executed as part of preparing this document** -- these are the commands to
run, not a report of results.

---

## A. Python environment

CARLA 0.9.15's client package must match the server exactly. Use a
dedicated environment so it never collides with any other CARLA version
already installed on the machine.

```bash
conda create -n carla0915 python=3.10 -y
conda activate carla0915
```

Install the CARLA 0.9.15 Python API. Prefer the wheel that ships inside the
CARLA 0.9.15 distribution itself over a generic PyPI install, since it is
guaranteed to match the server binary exactly:

```bash
# Preferred: install the wheel bundled with the CARLA 0.9.15 build
pip install <CARLA_0.9.15_ROOT>/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-*.whl

# Alternative, only if the bundled wheel is unavailable / wrong ABI:
pip install carla==0.9.15
```

Then install this project's remaining dependencies:

```bash
pip install -r requirements.txt
```

Verify the import:

```bash
python -c "import carla; c = carla.Client('localhost', 2000); print('carla module OK')"
```

Set `CARLA_ROOT` so `scripts/collect_dataset.py` can find
`agents.navigation.*` (shipped inside the CARLA distribution, not
pip-installable) at its actual install path on this machine -- **do not**
rely on the old hardcoded `C:\CARLA` default:

```powershell
# PowerShell
$env:CARLA_ROOT = "D:\CARLA_0.9.15"
```

```bash
# bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
```

---

## B. CARLA server 실행 (Windows 기준 예시)

```powershell
cd D:\CARLA_0.9.15
.\CarlaUE4.exe -quality-level=Epic -windowed -ResX=1280 -ResY=720
```

DirectX 11이 필요하면:

```powershell
.\CarlaUE4.exe -dx11 -quality-level=Epic -windowed -ResX=1280 -ResY=720
```

서버가 완전히 로드될 때까지 (맵 로딩 로그가 끝날 때까지) 기다린 뒤 다음 단계로
진행하세요.

---

## C. Server connection 확인

새 PowerShell 창에서 (서버는 계속 실행 중인 상태로):

```powershell
conda activate carla0915
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); print('client', c.get_client_version()); print('server', c.get_server_version()); print(c.get_world().get_map().name)"
```

`client`와 `server` 버전이 둘 다 `0.9.15`로 출력되어야 합니다. 하나라도 다르면
아래 단계로 진행하지 말고 먼저 버전을 맞추세요.

---

## D. Environment validation (static object cleanup)

```powershell
python scripts\tools\validate_runtime_environment.py `
    --host localhost `
    --port 2000 `
    --town Town01
```

```bash
# bash
python scripts/tools/validate_runtime_environment.py \
    --host localhost \
    --port 2000 \
    --town Town01
```

이 스크립트는 `src/simulation/environment.py`의 production cleanup 함수를 그대로
호출합니다 (별도 구현 없음). 출력에 있는 `[After cleanup]` 섹션의 안내문을
반드시 읽으세요 -- CARLA Python API에는 EnvironmentObject의 현재 enabled 상태를
다시 조회하는 getter가 없으므로, 이 스크립트는 "발견된 모든 정적 객체에 대해
disable 명령이 실제로 전달되었는지"까지만 확인합니다. 시각적으로 완전히
확인하려면 Town01을 스폰 전/후로 렌더링해서 주차 차량/보행자가 화면에 보이지
않는지 직접 확인하세요.

옵션: `--json <path>` 로 결과를 JSON으로도 저장할 수 있습니다 (`dataset/` 바깥
아무 경로나 사용).

---

## E. CARLA 0.9.15 smoke test

```powershell
python scripts\tests\test_carla0915_runtime.py `
    --host localhost `
    --port 2000 `
    --town Town01 `
    --route 1 `
    --frames 200 `
    --conditions day_clear day_rain
```

```bash
# bash
python scripts/tests/test_carla0915_runtime.py \
    --host localhost \
    --port 2000 \
    --town Town01 \
    --route 1 \
    --frames 200 \
    --conditions day_clear day_rain
```

이 스크립트는 `scripts/collect_dataset.py`의 `main()`을 실제로 호출해서 (동일한
production 코드 경로) canonical geometry 생성 + day_rain replay를 수행한 뒤,
결과물을 검사합니다. 출력 위치는 기본적으로
`outputs/runtime_0915_validation/smoke_<timestamp>/` 이며 `dataset/`은 절대
건드리지 않습니다. `--output-root <path>` 로 위치를 직접 지정할 수도 있습니다.

스크립트 종료 시 `RESULT: PASS` 또는 `RESULT: FAIL`과 함께 A~K 항목별
PASS/FAIL/SKIP 표가 출력됩니다. 실패 시 중간에 종료되어도 `finally` 블록이
world 설정과 Traffic Manager synchronous mode를 복구합니다 (다른 클라이언트에
영향이 남지 않도록).

---

## F. PASS 기준

아래를 모두 만족해야 최종 PASS로 판단합니다.

- client/server 버전이 둘 다 `0.9.15`
- `validate_runtime_environment.py`: RESULT PASS (static object disable
  coverage OK, building/road/traffic light sanity OK)
- `test_carla0915_runtime.py`: RESULT PASS, 특히
  - canonical 200 frames 생성 완료 (`--truncate-ok`로 잘려도 실패 아님)
  - sensor missing = 0, duplicate = 0 (`validate_geometry` 결과)
  - sync mismatch = 0 (production 파이프라인이 예외 없이 완주)
  - replay(day_rain)가 RGB-only (`rgb_left`, `rgb_right`만 존재)
  - replay transform out-of-tolerance = 0 (`replay_validation.out_of_tolerance_samples == 0`)
  - wind_intensity = 0.0 (모든 condition)
  - `paired_validation.json` -> `passed: true`
  - 스크립트 종료 후 cleanup 단계 오류 없음

하나라도 FAIL이면 그 항목의 상세 로그를 이 저장소의 이슈/기록에 남기고, 원인이
"코드 문제"인지 "0.9.15 API 차이"인지 구분해서 보고하세요 (특히
`carla.CityObjectLabel.{Car,Bus,Truck,Motorcycle,Bicycle,Train,Rider}`,
`carla.WeatherParameters.ClearNight/MidRainyNight` 는 사전 감사에서
VERIFY-ON-REMOTE로 표시된 항목입니다 -- 아래 한 줄로 가장 먼저 확인하세요):

```bash
python -c "import carla; print(all(hasattr(carla.CityObjectLabel, n) for n in ['Car','Bus','Truck','Motorcycle','Bicycle','Train','Pedestrians','Rider'])); print(hasattr(carla.WeatherParameters, 'ClearNight'), hasattr(carla.WeatherParameters, 'MidRainyNight'))"
```

---

## 참고: 실행 순서 요약

1. A -- conda env 생성 + CARLA 0.9.15 wheel 설치 + `CARLA_ROOT` 설정
2. B -- CarlaUE4.exe 실행, 맵 로딩 완료까지 대기
3. C -- 버전/연결 확인 (한 줄 명령)
4. D -- `validate_runtime_environment.py` 실행, RESULT 확인
5. E -- `test_carla0915_runtime.py` 실행, RESULT 확인
6. F 기준과 대조해서 최종 PASS/FAIL 판단

# Meta Quest 3/3S 빠른 시작

이 문서는 이미 Isaac Lab scene을 실행할 수 있는 PC에서 Meta Quest 연결과 첫
HDF5 episode 저장까지의 최소 흐름만 설명한다. CloudXR SDK 설치, 인증서 또는
네트워크를 새로 구성해야 한다면 [Runtime 실행 가이드](QUEST_RUNTIME_SERVICE.md)와
[Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md)를 함께 사용한다.

## 실행 경로 구분

| 목표 | 실행기 | 실제 Quest/OpenXR 필요 | 데이터 저장 |
|---|---|---:|---:|
| PC에서 scene과 카메라 확인 | `preview_quest_local.sh` | 아니요 | 아니요 |
| PC 브라우저/IWER bridge 확인 | `preview_quest_browser.sh` | 아니요 | 아니요 |
| 실제 Quest로 teleoperation | `collect_quest_teleop.sh` | 예 | HDF5/LeRobot |

PC 브라우저 미리보기 성공은 CloudXR/OpenXR 실기 연결 성공을 뜻하지 않는다.

## 1. Isaac 환경 확인

```bash
cd /absolute/path/to/HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./setup.sh --check-only
./quest_doctor.sh
```

Quest 없이 로봇과 head/wrist camera부터 확인한다.

```bash
./preview_quest_local.sh --robot-model s200062
```

확인이 끝나면 Isaac Sim 창을 닫는다.

## 2. 세션 환경 불러오기

반복 실행용 환경 파일을 준비했다면 새 터미널마다 명시적으로 불러온다.

```bash
cd /absolute/path/to/HumanoidScene
source .external/quest-session.env
./quest_doctor.sh --require-runtime
```

최소한 다음 값이 현재 PC의 실제 파일과 LAN 주소를 가리켜야 한다.

- `ISAACLAB_PYTHON`
- `XR_RUNTIME_JSON`
- `CLOUDXR_RUNTIME_DIR`
- `CLOUDXR_HOST`
- 인증서를 사용하는 경우 `CLOUDXR_CERTIFICATE`, `CLOUDXR_KEY`

`.env.example`은 예시이며 launcher가 자동으로 읽지 않는다.

## 3. 프로세스 실행 순서

각 프로세스는 별도 터미널에서 유지한다.

1. CloudXR Runtime service
2. Quest용 HTTP 또는 HTTPS 웹 서버
3. Quest Browser에서 페이지 접속 후 `CONNECT`
4. Isaac Lab collector

설치된 Runtime의 정확한 명령과 TLS 구성은
[Quest Runtime 실행](QUEST_RUNTIME_SERVICE.md)을 따른다. `XR_RUNTIME_JSON`을
export하는 것만으로 Runtime service가 실행되는 것은 아니다.

Quest에서 사용하는 주소는 PC의 LAN IP다. Quest에서 `localhost`와
`127.0.0.1`은 PC가 아니라 Quest 자신을 가리킨다.

## 4. 첫 HDF5 episode 수집

Runtime이 실행 중이고 Quest client가 연결된 다음 새 터미널에서 실행한다.

```bash
cd /absolute/path/to/HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
source .external/quest-session.env

./collect_quest_teleop.sh \
  --robot-model s200062 \
  --dataset-format hdf5 \
  --max-episodes 0 \
  --episode-seconds 0 \
  --no-auto-start
```

`--xr-runtime-json`을 환경 파일에 저장하지 않았다면 명령에 직접 추가한다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --dataset-format hdf5 \
  --no-auto-start
```

터미널에서 다음 상태를 확인한다.

```text
[TRACKING] left=True, right=True, head=True
[DATA] New HDF5 file: .../kuavo_quest_<timestamp>_<id>.hdf5
```

기본 입력은 좌우 controller다. 맨손만 사용할 때는 `--input-mode hands`, 실행
중 controller/hand 전환을 시험할 때는 `--hand-switch`를 사용한다.

## 5. 기본 조작

| Quest 입력 | PC 키 | 동작 |
|---|---|---|
| 왼쪽 `X` | `C` | 시점과 자세 기준 보정, 따라오기 중지 |
| 오른쪽 `A` | `T` | 저장 없이 따라오기 시작/중지 |
| 오른쪽 `B` | `P` | 녹화와 따라오기 시작/중지 |
| 왼쪽 `Y` | `H` | head/wrist camera panel 표시/숨김 |
| 왼쪽 stick | — | base 전후/좌우 이동 |
| 오른쪽 stick | — | base 회전과 torso 높이 |
| 왼쪽 grip | — | 누르는 동안 자유 시점 |
| 양쪽 index trigger | — | gripper 닫기/열기 |
| — | `M` | 완료한 episode를 성공으로 종료 |
| — | `R` | 실패로 종료하고 scene reset |

권장 순서는 `X` 보정 → `A`로 움직임 확인 → `B`로 녹화 → 실제 작업 완료 후
`M`이다. 전체 보정 규칙과 tracking loss 처리는
[Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)를 참고한다.

## 6. 관찰자 화면

기본값은 성능을 위해 desktop render와 PC camera preview가 OFF다. 다른 사람이
작업을 볼 때는 viewport 하나만 켠다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --desktop-render \
  --no-camera-preview \
  --render-quality performance \
  --scene-detail compact
```

Isaac Sim의 메인 viewport를 고정 3인칭 시점으로 조절한다. head와 두 wrist
preview 창을 모두 여는 것보다 이 방식이 가볍다. 자세한 부하 차이와 선택 기준은
[Quest 화면·성능 가이드](QUEST3_DISPLAY_AND_PERFORMANCE.md)를 참고한다.

## 7. 다음 단계

- LeRobot Dataset v3: [Quest 상세 가이드의 LeRobot 절](QUEST3_KUAVO_TELEOP_GUIDE.md#9-lerobot-dataset-v3-수집)
- Rack box 배치: [Workcell 편집 가이드](ISAACSIM_WORKCELL_GUIDE.md)
- 재실행 및 인증서: [Quest Runtime 실행](QUEST_RUNTIME_SERVICE.md)
- 연결 실패와 VRAM 문제: [Quest 상세 문제 해결](QUEST3_KUAVO_TELEOP_GUIDE.md#11-문제-해결)

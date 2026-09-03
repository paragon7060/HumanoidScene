# HumanoidScene: Kuavo Isaac Lab workcell

Kuavo humanoid가 경사진 랙의 열린 박스를 컨베이어의 빈 공간으로 옮기고,
모든 박스를 처리한 뒤 안전 펜스의 물리 버튼을 누르는 Isaac Lab 환경이다.

제공 기능:

- standalone scene과 manager-based robustness environment
- Isaac Sim 배치 편집 및 위치·회전·크기 캡처
- Meta Quest 3/3S 양팔 teleoperation
- head/wrist camera와 HDF5·LeRobot Dataset v3 수집
- GR00T N1.5/N1.7 online/offline evaluation와 headless MP4 기록

기본 로봇은 내장 2-finger gripper와 양쪽 D405가 있는 `s200062`이다.
`--robot-model s63`과 `--robot-model s56`도 선택할 수 있다. S56은
`--gripper s56_qiangnao` 또는 S200062 hand/D405를 이식한
`--gripper s56_twofinger`를 고를 수 있다. `--gripper none`은 손 geometry가
제거된 별도 bare-wrist S56 USD를 선택한다.

## 문서 바로가기

| 하려는 작업 | 문서 |
|---|---|
| 처음 설치하고 scene 실행 | [설치 및 첫 실행](docs/INSTALL.md) |
| 목적별 문서 찾기 | [문서 목차](docs/README.md) |
| Isaac Sim에서 배치 편집·캡처 | [Workcell 편집](docs/ISAACSIM_WORKCELL_GUIDE.md) |
| Meta Quest를 처음 연결하고 수집 | [Quest 빠른 시작](docs/QUEST3_QUICKSTART.md) |
| 실제 수집기 SDK·인증서 준비 및 간편 실행 | [수집기 설치·실행](docs/QUEST_COLLECTOR_SETUP.md) |
| Quest 전체 옵션과 조작법 | [Quest 상세 가이드](docs/QUEST3_KUAVO_TELEOP_GUIDE.md) |
| 관찰자 화면과 성능 설정 | [Quest 화면·성능](docs/QUEST3_DISPLAY_AND_PERFORMANCE.md) |
| 준비된 CloudXR 환경 재실행 | [Quest Runtime 실행](docs/QUEST_RUNTIME_SERVICE.md) |
| GR00T N1.7 범용 평가 | [GR00T 평가](docs/GROOT_N1_7_EVAL_GUIDE.md) |
| RwH-Kuavo V2를 S56에서 평가 | [S56 checkpoint-40K 평가](docs/RWH_KUAVO_V2_S56_EVAL.md) |
| 새 로봇/체크포인트를 평가 파이프라인에 추가 | [로봇 모델 평가 파이프라인](docs/ROBOT_MODEL_EVAL_PIPELINE.md) |
| 기존 상세 scene 설명과 검증 기록 | [프로젝트 상세 참조](PROJECT_REFERENCE.md) |

## 지원 환경

- Ubuntu Linux x86_64
- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- conda 환경 `env_isaaclab_232`
- 평가 영상을 저장할 때 FFmpeg (`ffmpeg` 실행 파일)

Isaac Sim/Lab, CloudXR, LeRobot과 policy weight는 각 라이선스와 GPU 환경에 맞게
별도로 설치한다. Kuavo/Rack/Box USD, URDF, mesh와 기본 layout JSON은 저장소에
포함되어 있다.

## 1. 설치와 첫 실행

```bash
git clone git@github.com:paragon7060/HumanoidScene.git
cd HumanoidScene
./install_isaaclab_stable.sh
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./setup.sh
./run_scene.sh --prefill 2
```

이미 환경을 설치했다면 새 터미널에서 다음을 준비한다.

```bash
cd /absolute/path/to/HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./doctor.sh
```

`/absolute/path/...`는 실제 경로로 바꾼다.

## 2. 실행 명령 요약

| 목적 | 명령 |
|---|---|
| 일반 scene | `./run_scene.sh` |
| 컨베이어에 기존 상자 2개 배치 | `./run_scene.sh --prefill 2` |
| manager-based 환경 | `./run_manager_env.sh --num-envs 1 --steps 240` |
| Quest 없이 카메라·로봇 확인 | `./preview_quest_local.sh --robot-model s200062` |
| PC 브라우저 XR bridge 확인 | `./preview_quest_browser.sh` |
| 실제 Quest 데이터 수집 | `./collect_quest_teleop.sh --xr-runtime-json /path/openxr_cloudxr.json` |
| GR00T wiring smoke test | `./eval_groot.sh --mock-policy --headless --episodes 1 --max-steps 5` |
| S56 GR00T N1.5 + 3-view 영상 | `./eval_rwh_kuavo_v2_s56.sh --headless --video-out artifacts/eval/s56.mp4 --episodes 1` |

전체 인자는 실행기에 `--help`를 붙여 확인한다.

```bash
./run_scene.sh --help
./run_manager_env.sh --help
./collect_quest_teleop.sh --help
```

## 3. Rack box 구성

선반 번호는 `1=하단`, `2=중단`, `3=상단`이다. Box 종류는 `small`, `medium`,
`large`, `xlarge`다.

```bash
./run_scene.sh --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'

./run_manager_env.sh --num-envs 1 --steps 100000 --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'
```

반복 배치는 JSON으로 선택할 수 있다.

```json
{
  "shelves": {
    "1": {"small": 2, "medium": 1},
    "2": ["large", "xlarge"],
    "3": ["medium", "large", "xlarge"]
  }
}
```

```bash
./run_scene.sh --rack-box-layout /absolute/path/rack_layout.json
```

사용하지 않는 box는 Rack 내부가 아니라
`/World/envs/env_0/Workcell/StagingBoxes`에 독립적으로 유지된다. 상세 배치 및
dictionary 설정은 [Workcell 편집 가이드](docs/ISAACSIM_WORKCELL_GUIDE.md)를 참고한다.

## 4. Isaac Sim 배치 편집

로봇이 서는 **rack–conveyor 외형 간격**은 현재 **1.10 m**다. 좌표를 직접 계산하지
않고 다음 명령으로 확인·조절할 수 있다. Rack/로봇은 유지하고 conveyor만 이동한다.

```bash
./set_workcell_gap.sh                        # 현재 간격 확인
./set_workcell_gap.sh --gap 1.20 --dry-run   # 1.20 m로 변경할 위치 미리보기
./set_workcell_gap.sh --gap 1.20             # JSON 저장; scene/eval 재시작 시 적용
```

이는 벨트 내부 물리면이 아닌 프레임 포함 외형 기준이다. 백업·별도 layout 사용법과
지원하는 배치 조건은 [간격 조절](docs/ISAACSIM_WORKCELL_GUIDE.md#51a-rackconveyor-통로-간격을-숫자로-조절)에 있다.

권장 흐름:

1. `run_scene.sh`로 scene을 연다.
2. Isaac Sim에서 Rack, Box, ButtonStation, SafetyFence의 위치·회전·크기를 조절한다.
3. layout과 rack-relative box pose를 캡처한다.
4. 저장한 JSON을 standalone, manager-based, Quest 실행에 공통 적용한다.

```bash
./capture_layout.sh --help
./capture_rack_box_poses.sh --help
```

prim 경로와 world/local 좌표 규칙은
[Isaac Sim Workcell 가이드](docs/ISAACSIM_WORKCELL_GUIDE.md)를 따른다.

## 5. Box flap friction과 randomization

고정값:

```bash
./run_scene.sh \
  --flap-static-friction 0.45 \
  --flap-dynamic-friction 0.32
```

standalone 시작 시 randomization:

```bash
./run_scene.sh --randomize-flap-friction \
  --flap-static-friction-range 0.25 0.75 \
  --flap-dynamic-friction-range 0.15 0.50
```

manager-based 환경은 reset마다 randomize한다.

```bash
./run_manager_env.sh --num-envs 8 --steps 100000 \
  --flap-static-friction-range 0.25 0.75 \
  --flap-dynamic-friction-range 0.15 0.50
```

결정론적 manager 실행은 `--no-randomize-flap-friction`을 사용한다.

## 6. Task-system smoke test

`--auto-demo`는 slot 예약, queue push, 완료 판정, 물리 버튼 gating과 컨베이어
시작을 확인하는 scripted oracle이다. 로봇 manipulation 성능 검증은 아니다.

```bash
./run_scene.sh --auto-demo --prefill 2 --ignore-captured-box-poses
```

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode performance \
  --auto-demo --demo-speed 4 --prefill 2 --steps 720 \
  --ignore-captured-box-poses
```

버튼 press/release만 확인:

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode performance \
  --verify-button --steps 300 --ignore-captured-box-poses
```

## 7. Meta Quest teleoperation

실제 OpenXR 수집기를 처음 준비한다면 [수집기 설치·실행](docs/QUEST_COLLECTOR_SETUP.md)을
따른다. `setup_quest_collector.sh`가 SDK·인증서·환경 파일·수집용 웹 snapshot을
별도로 준비하고, 이후에는 아래 명령만 사용한다.

```bash
./quest_collector.sh check     # 서비스/시뮬레이터 없이 파일·SDK 로딩 점검
./quest_collector.sh runtime   # 터미널 1
./quest_collector.sh web       # 터미널 2, HTTPS 8443
# Quest에서 Manual backend / PC IP / 49100으로 CONNECT한 뒤:
./quest_collector.sh collect   # 터미널 3, 수동 녹화 시작
```

각 서비스는 별도 터미널에서 유지한다. 생성된 환경 파일을 자동으로 읽으며,
preview의 HTTP 8080/WebSocket 8765와 구분한다.

Meta Quest는 여러 프로세스와 네트워크 설정이 필요하므로 전체 설치법을 루트 README에
중복하지 않는다. 처음에는 [Quest 빠른 시작](docs/QUEST3_QUICKSTART.md)을 따른다.

CloudXR 구성을 마친 환경의 기본 수집:

```bash
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"

./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --robot-model s200062 \
  --dataset-format hdf5 \
  --no-auto-start
```

다른 작업자가 볼 수 있는 단순 관찰 화면:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --desktop-render \
  --no-camera-preview \
  --render-quality performance \
  --scene-detail compact
```

기존 desktop viewport 하나를 고정 3인칭 시점으로 사용한다. 추가 camera preview와
GPU 비용은 [Quest 화면·성능 가이드](docs/QUEST3_DISPLAY_AND_PERFORMANCE.md)에 정리되어 있다.

## 8. HDF5와 LeRobot Dataset v3

먼저 HDF5로 수집을 확인한 뒤, Isaac 환경을 변경하지 않도록 LeRobot v3 writer는
별도 Python 환경으로 연결한다.

```bash
export LEROBOT_PYTHON="/absolute/path/to/lerobot-v3-environment/bin/python"
"$LEROBOT_PYTHON" -c \
  'from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == "v3.0"'

./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --dataset-format both \
  --lerobot-root datasets/quest_session_002_lerobot \
  --lerobot-repo-id local/kuavo_quest_teleop
```

카메라 해상도, 로봇 모델 또는 action schema를 바꾸면 새 dataset root를 사용한다.
세부 schema는 [Quest 상세 가이드](docs/QUEST3_KUAVO_TELEOP_GUIDE.md#9-lerobot-dataset-v3-수집)를 참고한다.

## 9. Evaluation과 테스트

```bash
./eval_groot.sh --help
./eval_groot.sh --mock-policy --headless --episodes 1 --max-steps 5
./offline_eval_groot.sh --help
./download_rwh_kuavo_v2_checkpoint.sh
./eval_rwh_kuavo_v2_s56.sh --mock-policy --headless --no-camera-preview --episodes 1 --max-steps 5
```

GR00T N1.5 checkpoint-40K의 16-D arm/claw schema를 S56과 이식된 S200062
2-finger/D405 end-effector에 연결하는 방법은
[RwH-Kuavo V2 S56 평가 가이드](docs/RWH_KUAVO_V2_S56_EVAL.md)를
참고한다. Isaac Lab과 LeRobot 0.5.x는 별도 Conda 프로세스로 유지된다.

실제 checkpoint를 `pick up the box` instruction으로 headless 평가하면서 head와
좌우 wrist 영상을 하나의 MP4로 저장한다.

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 \
  --video-out artifacts/eval/rwh_s56_pick_box.mp4 \
  --metrics-out artifacts/eval/rwh_s56_pick_box.json \
  --trace-out artifacts/eval/rwh_s56_pick_box_trace.json
```

MP4의 화면 순서는 `head_cam_h | wrist_cam_l | wrist_cam_r`이다. 여러 episode를
실행하면 `_ep000`, `_ep001`, ... 파일로 나뉜다. 기존 파일을 교체할 때만
`--overwrite-video`를 명시한다. 전용 launcher는 학습 데이터셋과 동일한 10 Hz를
기본값으로 사용하므로 240 step이 24초에 해당한다.

### 모델과 체크포인트 재사용 범위

Isaac 환경, 별도 LeRobot worker, headless 영상, metrics와 per-step trace는 모든
모델이 공유하는 파이프라인이다. 다만 임의의 로봇/checkpoint가 자동으로 호환되는
것은 아니다.

| 변경 범위 | 기존 파이프라인 재사용 |
|---|---|
| USD/URDF geometry만 바뀌고 joint/body/camera schema가 동일 | 모델 registry와 asset 추가 후 대부분 그대로 사용 |
| 기존 manager action schema를 사용하는 Kuavo 모델 | `eval_groot.sh`의 기본 profile 재사용 가능 |
| RwH 16-D `left7/claw/right7/claw` checkpoint | 현재 전용 profile은 검증된 S56 integrated hand만 허용 |
| joint 순서, action 차원, claw 의미 또는 camera key가 다름 | 새 policy profile과 adapter 및 단위 테스트 필요 |

새 모델은 최소한 robot registry, gripper preset, controlled-joint 순서, camera body와
extrinsic, checkpoint feature shape를 명시해야 한다. 상세 체크리스트는
[로봇 모델 평가 파이프라인](docs/ROBOT_MODEL_EVAL_PIPELINE.md)에 있다. 이름이 같은
joint라도 link 길이, joint origin, camera 위치가 다르면 같은 policy 성능을
보장하지 않는다.

### S56 checkpoint-40K 현재 검증 상태

2-finger 중앙 `bar_4`가 고정되어 있던 문제를 수정했다. S56/S200062의 양손에
finger–bar 물리 hinge를 추가하고 수동 링크의 독립 PD 구동을 제거했다.
3회 개폐·reset 검증에서 연결점 최대 오차는 각각 약 0.00193/0.00244 mm였다.
구조의 단순화 범위와 재현 명령은
[4-bar 연결 검증](docs/GRIPPER_CONFIGURATION.md#s200062--s56-physical-four-bar-closure)에 있다.

2026-09-02 **4-bar 수정 후 실제 checkpoint-40K 재평가**를 완료했다.
`pick up the box`, 10 Hz, 24초/240 step, seed 42에서 양쪽 claw의 개폐 명령 추종과
14개 arm의 다음-step 목표 추종 MAE `0.0155 rad`를 확인했다. 다만 영상에서
성공적인 박스 들기는 관찰되지 않았으며 기존 manager task progress는 0이었다.
이 manager의 성공 조건은 pick-only가 아니므로 별도 집기 성공률로 해석하면 안 된다.
영상은 `artifacts/eval/rwh_s56_fourbar_fixed_24s.mp4`에 head/left wrist/right wrist
순서로 저장했고, 지표와 trace도 같은 디렉터리에 있다. 생성 결과물은 Git에 포함하지 않는다.

재평가는 기존과 동일한 conda LeRobot **0.5.1**에서 실행했다. 학습 환경 0.5.2와의
완전한 일치, strict checkpoint loading, target-task data, camera/visual domain과
pick 전용 성공 판정은 아직 검증 과제로 남아 있다. 1회 rollout의 추종 오차 차이만으로
정책 성능이 개선됐다고 판단할 수 없다. 이전 `corrected_twofinger` 기록은 dexterous
mesh 중복만 제거했던 **4-bar 수정 전** 결과이므로 구분해서 비교해야 한다.

체크포인트의 saved preprocessor에 있는 `new_embodiment=31`은 평가에서도 override
없이 그대로 사용한다. texture는 단순히 고해상도로 만드는 것보다 학습 카메라의
노출, 감마, white balance, FOV, extrinsic과 재질/조명 분포를 맞추는 것이 먼저다.
전체 진단과 재현 조건은
[S56 checkpoint-40K 평가 가이드](docs/RWH_KUAVO_V2_S56_EVAL.md#validated-rollout-status)를
참고한다.

```bash
pytest -q
```

단위 테스트는 Isaac Sim을 시작하지 않는다. GPU/VR/실물 Quest 검증은 별도이며,
`quest_doctor.sh` 통과만으로 실제 영상과 tracking 성공을 보장하지 않는다.

## 저장소 구조

```text
HumanoidScene/
├── src/kuavo_isaaclab_scene/  # scene, teleop, recorder, evaluation
│   ├── assets/                # USD, URDF, mesh, texture
│   └── configs/               # packaged fallback configs
├── configs/                   # editable deployment configs
├── scripts/                   # canonical launch/capture utilities
├── integrations/cloudxr/      # Quest browser bridge
├── tests/                     # simulator-independent tests
├── docs/                      # task-focused guides
├── PROJECT_REFERENCE.md       # 기존 상세 project 설명
└── *.sh                       # root launch wrappers
```

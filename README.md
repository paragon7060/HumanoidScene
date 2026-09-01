# HumanoidScene: Kuavo Isaac Lab workcell

Kuavo humanoid가 경사진 랙의 열린 박스를 컨베이어의 빈 공간으로 옮기고,
모든 박스를 처리한 뒤 안전 펜스의 물리 버튼을 누르는 Isaac Lab 환경이다.

제공 기능:

- standalone scene과 manager-based robustness environment
- Isaac Sim 배치 편집 및 위치·회전·크기 캡처
- Meta Quest 3/3S 양팔 teleoperation
- head/wrist camera와 HDF5·LeRobot Dataset v3 수집
- GR00T N1.7 online/offline evaluation

기본 로봇은 내장 2-finger gripper와 양쪽 D405가 있는 `s200062`이다.
`--robot-model s63`과 `--robot-model s56`도 선택할 수 있다.

## 문서 바로가기

| 하려는 작업 | 문서 |
|---|---|
| 처음 설치하고 scene 실행 | [설치 및 첫 실행](docs/INSTALL.md) |
| 목적별 문서 찾기 | [문서 목차](docs/README.md) |
| Isaac Sim에서 배치 편집·캡처 | [Workcell 편집](docs/ISAACSIM_WORKCELL_GUIDE.md) |
| Meta Quest를 처음 연결하고 수집 | [Quest 빠른 시작](docs/QUEST3_QUICKSTART.md) |
| Quest 전체 옵션과 조작법 | [Quest 상세 가이드](docs/QUEST3_KUAVO_TELEOP_GUIDE.md) |
| 관찰자 화면과 성능 설정 | [Quest 화면·성능](docs/QUEST3_DISPLAY_AND_PERFORMANCE.md) |
| 준비된 CloudXR 환경 재실행 | [Quest Runtime 실행](docs/QUEST_RUNTIME_SERVICE.md) |
| GR00T N1.7 평가 | [GR00T 평가](docs/GROOT_N1_7_EVAL_GUIDE.md) |
| 기존 상세 scene 설명과 검증 기록 | [프로젝트 상세 참조](PROJECT_REFERENCE.md) |

## 지원 환경

- Ubuntu Linux x86_64
- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- conda 환경 `env_isaaclab_232`

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
```

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

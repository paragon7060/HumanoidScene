# Meta Quest 3 → Kuavo Isaac Lab teleoperation/data collection

이 문서는 현재 workcell을 Meta Quest 3에서 보고, OpenXR hand tracking으로 Kuavo의 양팔과 머리를 조작하면서 LeRobot Dataset v3 또는 HDF5 demonstration을 수집하는 절차를 정리한다.

## 1. 구현 구조

```text
Quest 3 WebXR hand/head tracking
        ↓ CloudXR Runtime 6.x / OpenXR
Isaac Lab OpenXRDevice
        ↓ Kuavo 안전 상대-pose mapper
left wrist 6D → left 7-DoF differential IK
right wrist 6D → right 7-DoF differential IK
HMD yaw/pitch → zhead_1_joint / zhead_2_joint
        ↓
ManagerBasedRLEnv + Kuavo head/wrist RGB cameras
        ↓ XRSceneView head-locked compositor
head RGB full-screen + left/right wrist overlays
        ↓
HDF5 recorder / isolated LeRobot Dataset v3 writer
```

관련 파일:

- `src/kuavo_isaaclab_scene/teleop_env.py`: manager-based 양팔 IK/head action 환경
- `src/kuavo_isaaclab_scene/quest_openxr.py`: Isaac Lab v2.3.2 raw hand/head tracking 호환 어댑터
- `src/kuavo_isaaclab_scene/quest_runtime.py`: GUI를 열지 않는 OpenXR/CloudXR 사전 점검
- `src/kuavo_isaaclab_scene/teleop_mapping.py`: tracking 유효성, calibration, smoothing, safety clamp
- `src/kuavo_isaaclab_scene/teleop_recorder.py`: RAM에 누적하지 않는 HDF5 writer
- `src/kuavo_isaaclab_scene/teleop_lerobot_recorder.py`: Isaac Lab과 별도 v3 writer process 사이의 recorder client
- `src/kuavo_isaaclab_scene/lerobot_writer_worker.py`: LeRobot v3 `create/resume/add_frame/save_episode/finalize` worker
- `src/kuavo_isaaclab_scene/xr_camera_overlay.py`: Quest head-locked head/wrist camera compositor
- `src/kuavo_isaaclab_scene/collect_quest_teleop.py`: 실행/episode 제어
- `collect_quest_teleop.sh`: 루트 실행 wrapper

환경은 `Isaac-Kuavo-QuestTeleop-v0`으로도 등록되어 있다. 실제 Quest 수집에는 raw hand/head pose와 camera를 함께 기록하는 전용 `collect_quest_teleop.sh`를 사용한다.

## 2. 중요한 전제

Meta Quest 경로도 scene과 동일한 안정 버전 조합인 **Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + Python 3.11**을 사용한다. 별도의 구버전 Isaac 환경은 필요하지 않다.

`nvidia-cloudxr-6.2.0.tgz`는 CloudXR.js 웹 클라이언트용 npm 패키지다. Isaac Lab 프로세스에 OpenXR tracking을 공급하는 **CloudXR Runtime 패키지 및 `openxr_cloudxr.json`**은 별도로 필요하다.

이전에 USD를 확인한 Spatial/Kit 109 프로세스 내부의 CloudXR extension을 Isaac Sim 5.1/Kit 107.3 프로세스에 그대로 섞어 로드하면 안 된다. Isaac Lab은 v2.3.2에 포함된 `isaaclab.python.xr.openxr.kit` experience를 실행하고 외부 CloudXR Runtime을 `XR_RUNTIME_JSON`으로 선택한다.

Isaac Lab v2.3.2의 `OpenXRDevice`는 retargeter requirement에 따라 조회할 tracking feature를 선택한다. Kuavo는 자체 mapper를 사용하므로 `RawQuestOpenXRDevice`가 hand/head requirement를 명시적으로 활성화하고 upstream raw dict를 그대로 반환한다. 이 어댑터가 없으면 retargeter 없는 직접 생성 경로에서 tracking dict가 비어 있을 수 있다.

현재 Kuavo USD에는 `zarm_l1..l7`, `zarm_r1..r7`만 있고 actuated finger/gripper joint가 없다. 따라서 현재 구현은:

- Quest 손목 위치/회전으로 양팔을 제어한다.
- thumb-index pinch 거리와 양손 26개 관절은 데이터에 저장한다.
- 실제 손가락 닫기 명령은 생성하지 않는다.

향후 dexterous hand 또는 gripper USD를 추가하면 기록된 pinch 및 hand-joint 데이터를 gripper action term에 연결할 수 있다.

## 3. 첫 실행

새 conda 환경을 활성화하고 GUI 없이 Quest stack부터 확인한다.

```bash
cd HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./quest_doctor.sh
```

`Quest compatibility: OK`가 나오면 OpenXR experience, `omni.kit.xr.*`,
`isaacsim.xr.openxr` extension이 모두 발견된 것이다. 이 단계에서는 CloudXR
Runtime이 없어도 정상이다.

### 3.0 Quest 연결 전 로컬 화면 확인

Meta Quest나 CloudXR Runtime 없이 정확한 teleop scene과 robot camera 3개를 먼저 확인할 수 있다.

```bash
cd HumanoidScene
./preview_quest_local.sh
```

Isaac Sim main viewport와 함께 `robustness_camera`(Kuavo head), `left_wrist_camera`, `right_wrist_camera`의 작은 창 세 개가 열린다. 기본값 `--steps 0`은 창을 닫을 때까지 실행한다.

Kuavo 머리가 움직일 때 head camera 영상도 함께 변하는지 보려면:

```bash
./preview_quest_local.sh --head-sweep
```

부하가 크면 다음처럼 해상도를 낮춘다.

```bash
./preview_quest_local.sh \
  --head-camera-width 320 --head-camera-height 180 \
  --wrist-camera-width 160 --wrist-camera-height 120
```

이 모드는 OpenXR를 명시적으로 끄므로 Quest가 없어도 실행된다. 다만 CloudXR streaming, 실제 HMD pose, hand tracking, Quest 내부 head-locked overlay는 확인하지 못한다. 이 네 항목은 실제 Quest 또는 별도의 OpenXR client가 필요하다.

### 3.0.1 PC 브라우저에서 Quest 상호작용 검증

`preview_quest_local.sh`보다 한 단계 더 나아가, 데스크톱 Chrome의 IWER(가상 Quest 3) 입력이 실제 manager-based IsaacLab 환경을 움직이는지 확인할 수 있다. 이 경로는 외부 CloudXR Runtime/codec을 사용하지 않는다. WebSocket으로 WebXR tracking을 전달하고 Kuavo camera를 JPEG로 되돌려 보내므로 다음 항목을 한꺼번에 검증한다.

- IWER HMD 회전 → Kuavo head yaw/pitch
- IWER 좌우 controller 이동/회전 → Kuavo 양팔 differential IK
- Kuavo 단안 head camera → 브라우저 전체 XR 화면
- 좌우 wrist camera → 브라우저 화면 아래쪽 inset

터미널 1에서 정확한 teleop 환경과 로컬 브리지를 실행한다.

```bash
cd HumanoidScene
./preview_quest_browser.sh
```

다음 로그가 나오면 준비된 것이다.

```text
[READY] Browser bridge: ws://127.0.0.1:8765
```

터미널 2에서 NVIDIA가 제공한 CloudXR.js npm package 경로를 지정하고, 저장소의
재현 가능한 setup script로 pinned CloudXR sample과 local backend patch를 준비한다.
NVIDIA package 자체와 upstream checkout은 Git에 포함되지 않는다.

```bash
cd HumanoidScene
export CLOUDXR_NPM_TGZ=/absolute/path/to/nvidia-cloudxr-6.2.0.tgz
./setup_quest_browser.sh
npm --prefix .external/cloudxr-js-samples/simple run dev-server
```

별도 checkout을 쓰려면 setup 전에
`CLOUDXR_JS_SAMPLES_DIR=/absolute/path/to/cloudxr-js-samples`를 export한다.

Chrome에서 `http://localhost:8080`을 열고 다음을 선택한다.

```text
Select Server Backend: Local Kuavo IsaacLab (IWER/Quest)
Server IP:             127.0.0.1
Port:                  8765
Mode:                  VR
```

`CONNECT LOCAL ISAACLAB`을 누른다. 성공하면 버튼은 `LOCAL ISAACLAB (streaming)`으로 바뀌고 IsaacLab 터미널에는 다음이 표시된다.

```text
[BROWSER] connected_clients=1
[TRACKING] left=True, right=True, head=True
```

IWER panel에서 HMD와 좌우 controller pose를 움직인다. 첫 유효 frame은 안전 calibration이므로 두 번째 pose 변화부터 로봇이 움직인다. 데스크톱 IWER controller는 손목 pose로 사용하며 thumb/index tip은 packet 호환성을 위해 합성된다. 실제 Quest browser에서 hand tracking이 제공되면 실제 wrist/thumb/index WebXR joint를 사용한다.

이 로컬 경로는 CloudXR codec/네트워크 품질을 검증하는 것이 아니라, 우리가 구성한 IsaacLab scene의 카메라와 Quest 제어 mapping이 상호작용하는지를 검증한다. PC 검증 후 Quest에서 같은 bridge를 시험하려면 IsaacLab 쪽을 `./preview_quest_browser.sh --bridge-host 0.0.0.0`으로 실행하고, Quest 페이지에서 Server IP를 PC의 LAN IP로 바꾼다. 8765는 인증 없는 개발용 포트이므로 신뢰할 수 있는 로컬 네트워크에서만 연다.

### 3.1 CloudXR Runtime 준비

CloudXR Runtime 6.x 설치 디렉터리에서 다음 파일을 찾는다.

```bash
find /path/to/cloudxr-runtime -name openxr_cloudxr.json -print
```

Runtime은 해당 배포 패키지의 management/launch 절차대로 먼저 실행한다. Quest에서 사용 중인 CloudXR.js simple client는 Isaac Lab PC의 CloudXR Runtime(WebRTC 기본 포트 49100)에 연결한다.

실행 전에 manifest JSON 형식과 `runtime.library_path`까지 검사한다.

```bash
./quest_doctor.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --require-runtime
```

### 3.2 Isaac Lab collector 실행

```bash
cd HumanoidScene

./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --dataset-format lerobot \
  --lerobot-root datasets/kuavo_quest_lerobot \
  --lerobot-repo-id paragon7060/kuavo_quest_teleop \
  --max-episodes 20 \
  --episode-seconds 30
```

이미 shell에서 runtime을 지정했다면 argument를 생략해도 된다.

```bash
export XR_RUNTIME_JSON=/absolute/path/to/openxr_cloudxr.json
./collect_quest_teleop.sh \
  --dataset-format lerobot \
  --lerobot-root datasets/session_001_lerobot
```

입력이 들어오면 로그가 다음처럼 바뀐다.

```text
[TRACKING] left=True, right=True, head=True
[DATA] Recording demo_00000
```

양손이 처음 인식된 프레임은 calibration에만 사용되며 action은 0이다. tracking을 잃었다가 다시 찾은 손도 첫 프레임에 재calibration되므로 갑자기 팔이 튀지 않는다.

## 4. 조작 및 episode 제어

기본값은 `--auto-start`다. 양손 tracking이 유효해지면 자동으로 recording을 시작한다.

| 동작 | Quest teleop message | Isaac Sim desktop key |
|---|---|---|
| recording 시작 | `START` | `P` |
| recording 종료/실패 | `STOP` | `P` |
| 환경 reset/episode 폐기 | `RESET` | `R` |
| 성공 demonstration으로 종료 | - | `M` |

성공한 작업은 desktop Isaac Sim 창에서 `M`을 누른다. 현재 CloudXR.js simple sample이 START/STOP/RESET message를 보내지 않아도 `--auto-start`와 desktop key를 사용할 수 있다.

LeRobot 모드에서는 기본적으로 `M`으로 성공 처리한 episode만 저장한다. `STOP`, `RESET`, time limit episode는 학습 데이터에 섞이지 않도록 폐기된다. 실패 episode도 분석용으로 보존하려면 `--lerobot-save-failed`를 추가한다.

자동 시작을 끄려면:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --no-auto-start
```

## 5. Rack box 배치와 함께 실행

기존 rack box CLI를 collector에서도 사용할 수 있다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --rack-boxes '1:small*2;2:medium,large;3:xlarge*2' \
  --dataset datasets/layout_a.hdf5
```

Isaac Sim에서 캡처한 정확한 pose JSON을 쓰려면:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --rack-box-poses configs/rack_box_poses.json \
  --dataset datasets/captured_layout.hdf5
```

## 6. Head camera와 Quest 시야

- Quest의 일반 stereo scene view 위에 불투명한 head-locked XR UI를 놓는다.
- 실제 Kuavo `scene["robustness_camera"]` 단안 RGB가 UI 전체를 채운다.
- `scene["left_wrist_camera"]`, `scene["right_wrist_camera"]` 영상이 왼쪽/오른쪽 작은 창으로 표시된다.
- HMD yaw/pitch는 Kuavo `zhead_1_joint`, `zhead_2_joint`에 연결되므로 실제 `head_camera_base`가 Quest 회전을 따라간다.
- Kuavo에는 머리 translation joint가 없으므로 Quest의 위치 이동은 robot head에 적용되지 않는다.
- desktop Isaac Sim에도 head/left wrist/right wrist mini viewport가 열린다.

Quest camera compositor는 기본으로 활성화된다. 원래 stereo scene view로 돌아가려면:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --no-quest-camera-overlay
```

화면이 머리 뒤쪽에 보이는 경우 OpenXR runtime의 forward-axis 차이이므로 다음 옵션으로 반전한다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --xr-overlay-forward-axis +z
```

화면의 눈앞 거리 조절:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --xr-overlay-distance 0.8
```

### Quest에서 확인할 항목

1. 연결 직후 검은 화면이 잠시 보인 뒤 head camera 영상이 전체 시야를 채우는지 확인한다.
2. 왼쪽 중앙에 `LEFT WRIST`, 오른쪽 중앙에 `RIGHT WRIST` 영상이 표시되는지 확인한다.
3. Quest를 좌우로 돌렸을 때 Kuavo `zhead_1_joint`가 회전하고 head camera 영상도 함께 변하는지 확인한다.
4. Quest를 위아래로 돌렸을 때 `zhead_2_joint`가 제한 범위 안에서 회전하는지 확인한다.
5. 몸을 앞뒤로 움직이는 translation은 Kuavo 머리에 적용되지 않는 것이 정상이다. 현재 model에는 yaw/pitch 두 관절만 있다.

패널이 보이지 않으면 먼저 `--xr-overlay-forward-axis +z`를 시도한다. 패널은 보이지만 가장자리에 원래 stereo scene이 노출되면 `--xr-overlay-distance`를 작게 조절한다. 실제 확인 결과를 기준으로 `QuestCameraOverlayCfg.plane_width_m`, `plane_height_m`을 조절할 수 있다.

카메라 해상도 변경:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --head-camera-width 640 \
  --head-camera-height 360 \
  --wrist-camera-width 240 \
  --wrist-camera-height 180
```

기본 해상도는 head 640×360, wrist 각각 240×180이다. 메모리 사용량을 줄이려면 해상도와 depth 기록을 함께 낮춘다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --no-record-depth \
  --no-camera-preview
```

## 7. Domain randomization

기본 수집은 deterministic reset이다. robustness 데이터를 섞으려면 다음처럼 실행한다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --domain-randomization \
  --dataset datasets/randomized.hdf5
```

이 옵션은 기존 manager-based scene의 material/mass/actuator/gravity/flap friction/lighting/mover/cargo disturbance event를 활성화한다. 먼저 deterministic demonstration을 충분히 수집한 뒤 randomized session을 별도 파일로 만드는 편이 데이터 관리에 쉽다.

## 8. 제어 감도

손 1 m 이동당 robot command 비율은 position gain으로 조정한다.

```bash
# 더 천천히
./collect_quest_teleop.sh --position-gain 1.0 ...

# 더 빠르게
./collect_quest_teleop.sh --position-gain 2.0 ...
```

기본 안전 제한은 control step마다 translation 2.5 cm, rotation 0.12 rad이다. 이 제한과 smoothing은 `TeleopMappingCfg`에 모여 있다.

## 9. LeRobot Dataset v3 수집

Isaac Sim 환경의 NumPy/Torch를 교체하지 않기 위해 LeRobot Dataset v3 writer는
별도 Python 환경에서 실행한다. 이 저장소에서 확인한 기준 조합은 Python 3.12,
LeRobot 0.6.0, Dataset format v3.0이다.

```bash
export LEROBOT_PYTHON=/absolute/path/to/lerobot-v3-environment/bin/python
"${LEROBOT_PYTHON}" -c \
  'from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == "v3.0"'
```

일반적인 v3 수집 명령:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --dataset-format lerobot \
  --lerobot-root datasets/kuavo_quest_lerobot \
  --lerobot-repo-id paragon7060/kuavo_quest_teleop
```

HDF5도 동시에 남기려면:

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --dataset-format both \
  --dataset datasets/kuavo_quest_raw.hdf5 \
  --lerobot-root datasets/kuavo_quest_lerobot \
  --lerobot-repo-id paragon7060/kuavo_quest_teleop
```

다른 Python 3.12 LeRobot v3 환경을 쓰려면 `LEROBOT_PYTHON`을 export하거나
`--lerobot-python /path/to/env/bin/python`으로 지정한다. launcher는 알려진 conda
경로도 탐색하며, worker는 시작할 때 format이 정확히 `v3.0`인지 확인한다.

기본 camera 저장은 MP4다. 개별 PNG가 필요하면 `--no-lerobot-use-videos`를 사용한다. 기존 dataset에 이어서 수집할 때에는 FPS, camera 해상도, wrist camera 포함 여부, box/button 수가 최초 schema와 같아야 한다. 이 값이 바뀌면 새 `--lerobot-root`를 사용한다.

v3 주요 feature:

```text
observation.state                    [T, 16]
observation.velocity                 [T, 16]
observation.ee_pose                  [T, 14]
observation.images.head              head camera MP4/image
observation.images.left_wrist        left wrist MP4/image
observation.images.right_wrist       right wrist MP4/image
observation.openxr.head_pose          [T, 7]
observation.openxr.left_hand          [T, 182]
observation.openxr.right_hand         [T, 182]
observation.pinch_distance            [T, 2]
observation.tracking_valid            [T, 3]
observation.box_root_pose             [T, number_of_boxes * 7]
observation.button_joint_position     [T, number_of_button_joints]
action                                [T, 14]
next.done / next.success              [T, 1]
task                                  natural-language task
```

프로세스 정상 종료 시 worker가 반드시 `dataset.finalize()`를 호출한다. 결과는 v3의 chunked 구조인 `data/chunk-*/file-*.parquet`, `meta/episodes/chunk-*/file-*.parquet`, `meta/tasks.parquet`, `videos/.../file-*.mp4`로 생성된다. 작업별 success/reason과 scene metadata는 추가 sidecar인 `meta/kuavo_episode_metadata.jsonl`에도 기록된다.

간단한 확인:

```bash
${LEROBOT_PYTHON} - <<'PY'
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="paragon7060/kuavo_quest_teleop",
    root="datasets/kuavo_quest_lerobot",
)
print(dataset)
print(dataset.features)
PY
```

## 10. HDF5 데이터 구조

```text
/data/demo_00000
  attrs: success, end_reason, num_samples, joint_names, ...
  /samples
    action                         [T, 14]
    robot_joint_position           [T, 16]
    robot_joint_velocity           [T, 16]
    left_end_effector_pose_w       [T, 7]
    right_end_effector_pose_w      [T, 7]
    openxr_left_hand               [T, 26, 7]
    openxr_right_hand              [T, 26, 7]
    openxr_head_pose               [T, 7]
    pinch_distance_m               [T, 2]
    tracking_valid                 [T, 3]
    box_root_pose_w                [T, number_of_boxes, 7]
    button_joint_position          [T, ...]
    head_rgb                       [T, H, W, 3]
    left_wrist_rgb                 [T, H_w, W_w, 3] (optional)
    right_wrist_rgb                [T, H_w, W_w, 3] (optional)
    head_depth_m                   [T, H, W] (optional)
    sim_time_s / wall_time_s       [T]
```

간단한 확인:

```bash
${ISAACLAB_PYTHON} - <<'PY'
import h5py

path = "datasets/kuavo_quest_teleop.hdf5"
with h5py.File(path, "r") as f:
    for name, demo in f["data"].items():
        print(name, demo.attrs["num_samples"], demo.attrs["success"])
        print("  action:", demo["samples/action"].shape)
        print("  head_rgb:", demo["samples/head_rgb"].shape)
PY
```

## 11. 문제 해결

### Quest 화면은 연결되지만 tracking이 계속 False

- Quest 브라우저에서 hand tracking permission을 허용했는지 확인한다.
- controller가 아니라 bare-hand tracking mode로 양손을 카메라에 보이게 한다.
- Isaac Lab console에서 세 값이 모두 True로 바뀌는지 확인한다.
- CloudXR video 연결과 OpenXR input 전달은 별개이므로, 영상만 보인다고 hand input이 반드시 들어온 것은 아니다.

### `XR_ERROR_RUNTIME_FAILURE` 또는 OpenXR runtime을 찾지 못함

- `XR_RUNTIME_JSON`이 npm tgz가 아니라 `openxr_cloudxr.json`을 가리키는지 확인한다.
- CloudXR Runtime process/service를 Isaac Lab보다 먼저 실행한다.
- Runtime과 Isaac Lab을 같은 사용자 권한으로 실행하고 manifest 내부 library path가 존재하는지 확인한다.

### Isaac Sim/다른 Electron 앱이 함께 종료됨

XR render와 head/wrist RTX camera 3개를 동시에 쓰므로 GPU VRAM 압력이 생길 수 있다. 문제가 있으면 320×180 head, 160×120 wrist로 낮추고 `--no-record-depth --no-camera-preview`를 함께 사용한다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --head-camera-width 320 --head-camera-height 180 \
  --wrist-camera-width 160 --wrist-camera-height 120 \
  --no-record-depth --no-camera-preview
```

### 팔은 움직이지만 상자를 잡을 수 없음

현재 Kuavo asset에는 손가락 또는 gripper actuator가 없기 때문이다. 손목/팔 IK와 hand tracking 자체의 문제가 아니다. 실제 grasp data를 만들려면 Kuavo hand/gripper articulation을 USD에 추가하고 pinch-to-gripper action을 연결해야 한다.

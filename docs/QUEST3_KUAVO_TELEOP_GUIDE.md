# Meta Quest 3 → Kuavo Isaac Lab teleoperation/data collection

이 문서는 현재 workcell을 Meta Quest 3에서 보고, OpenXR hand tracking으로 Kuavo의 양팔과 머리를 조작하면서 LeRobot Dataset v3 또는 HDF5 demonstration을 수집하는 절차를 정리한다.

처음 설치한다면 [README의 다운로드부터 첫 저장까지 안내](../README.md#quest-collection)를
먼저 따른다. Runtime SDK와 npm `.tgz`의 공식 다운로드, JSON 경로 설정,
Linux 서비스 준비, Quest 브라우저의 HTTP/HTTPS 설정, 포트 구분을 단계별로 설명한다.
이 문서는 그 이후의 조작·카메라·데이터 schema 세부 설정을 다룬다.

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
native stereo scene + small upper-left/right wrist overlays
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
- `src/kuavo_isaaclab_scene/xr_camera_overlay.py`: Quest head-locked small wrist-camera panels
- `src/kuavo_isaaclab_scene/collect_quest_teleop.py`: 실행/episode 제어
- `collect_quest_teleop.sh`: 루트 실행 wrapper

환경은 `Isaac-Kuavo-QuestTeleop-v0`으로도 등록되어 있다. 실제 Quest 수집에는 raw hand/head pose와 camera를 함께 기록하는 전용 `collect_quest_teleop.sh`를 사용한다.

## 2. 중요한 전제

Meta Quest 경로도 scene과 동일한 안정 버전 조합인 **Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + Python 3.11**을 사용한다. 별도의 구버전 Isaac 환경은 필요하지 않다.

`nvidia-cloudxr-6.2.0.tgz`는 CloudXR.js 웹 클라이언트용 npm 패키지다. Isaac Lab 프로세스에 OpenXR tracking을 공급하는 **CloudXR Runtime 패키지 및 `openxr_cloudxr.json`**은 별도로 필요하다.

이전에 USD를 확인한 Spatial/Kit 109 프로세스 내부의 CloudXR extension을 Isaac Sim 5.1/Kit 107.3 프로세스에 그대로 섞어 로드하면 안 된다. Isaac Lab은 v2.3.2에 포함된 `isaaclab.python.xr.openxr.kit` experience를 실행하고 외부 CloudXR Runtime을 `XR_RUNTIME_JSON`으로 선택한다.

Isaac Lab v2.3.2의 `OpenXRDevice`는 retargeter requirement에 따라 조회할 tracking feature를 선택한다. Kuavo는 자체 mapper를 사용하므로 `RawQuestOpenXRDevice`가 hand/head requirement를 명시적으로 활성화하고 upstream raw dict를 그대로 반환한다. 이 어댑터가 없으면 retargeter 없는 직접 생성 경로에서 tracking dict가 비어 있을 수 있다.

기본 S200062 USD에는 양쪽 2-finger gripper와 D405 형상이 직접 포함된다. 각 손의
4개 linkage joint를 하나의 binary action으로 동기 제어한다. 비교용
`--robot-model s63`에서는 기존 외장 8-joint Robotiq 기반 claw를 사용한다.
따라서 현재 구현은:

- 기본 모드는 Quest 컨트롤러의 위치/회전으로 양팔을 제어하고 검지 트리거로 gripper를 조작한다.
- `--input-mode hands`에서는 손목 위치/회전, thumb-index pinch 거리와 양손 26개 관절을 사용·저장한다.
- 맨손 모드는 thumb-index 거리가 0.055 m 이하이면 같은 쪽 gripper를 닫고, 그보다 크거나
  tracking이 유효하지 않으면 연다. 컨트롤러 모드는 tracking 손실 시 마지막 gripper 목표를 유지한다.
- `--gripper none`으로 기존 14-D handless teleop action도 재현할 수 있다.

다른 hand USD와 mount pose는 [Gripper configuration](GRIPPER_CONFIGURATION.md)의
JSON preset으로 교체한다.

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
./preview_quest_local.sh --robot-model s200062
```

Isaac Sim main viewport와 함께 `robustness_camera`(Kuavo head), `left_wrist_camera`, `right_wrist_camera`의 작은 창 세 개가 열린다. 기본값 `--steps 0`은 창을 닫을 때까지 실행한다.

Kuavo 머리가 움직일 때 head camera 영상도 함께 변하는지 보려면:

```bash
./preview_quest_local.sh --head-sweep
```

같은 Quest/카메라 경로에서 S63을 비교하려면:

```bash
./preview_quest_local.sh --robot-model s63 --head-sweep
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

이 미리보기는 데이터를 저장하지 않는다. Quest의 HTTP 접속에는 해당 개발
origin에 대한 WebXR 허용 설정도 필요하다. HTTPS를 사용한다면 로컬 bridge도
별도의 TLS 종단이 필요하며, 페이지를 HTTPS로 바꾸는 것만으로 평문 8765 서버가
WSS 서버가 되지는 않는다. 실제 OpenXR 수집은 미리보기를 종료하고
`Manual Input IP:Port` 백엔드와 CloudXR Runtime에 연결한다.

### 3.1 CloudXR Runtime 준비

[Runtime 다운로드와 JSON 설정](../README.md#quest-files),
[Linux 서비스 준비 및 현재 제공 범위](../README.md#quest-runtime-service)를 먼저 확인한다.
SDK와 `openxr_cloudxr.json`만 준비해도 서비스가 자동 시작되는 것은 아니다.
이 저장소의 `run_cloudxr_runtime.sh`로 서비스를 실행할 수 있다.
[실행기·HTTPS 설정 안내](QUEST_RUNTIME_SERVICE.md)를 참고한다. SDK 바이너리는
별도 설치하며, 서비스는 수집 중 계속 실행되어 있어야 한다.

CloudXR Runtime 6.x 설치 디렉터리에서 다음 파일을 찾는다.

```bash
find /path/to/cloudxr-runtime -name openxr_cloudxr.json -print
```

Runtime은 해당 배포 패키지의 management/launch 절차대로 먼저 실행한다. Quest에서 사용 중인 CloudXR.js simple client는 Isaac Lab PC의 CloudXR Runtime(WebRTC 기본 포트 49100)에 연결한다.

49100은 WebSocket 신호용 TCP 포트이며, WebRTC 미디어는 기본 UDP 47998을 사용한다.
HTTPS 페이지에서는 WSS 연결이 필요하다. 이 경우
[Quest 브라우저·프록시 설정](../README.md#quest-headset)에 따라 TLS 프록시를
준비하고 프록시 포트를 선택한다. 실제 SDK/GPU 조합의 스트리밍은 별도 실기
검증 대상이다. 아래 doctor는 JSON과 라이브러리 파일 존재만 검사하며 서비스
상태나 GPU 지원을 확인하지 않는다.

실행 전에 manifest JSON 형식과 `runtime.library_path`까지 검사한다.

```bash
./quest_doctor.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --require-runtime
```

### 3.2 Isaac Lab collector 실행

먼저 Quest 웹 클라이언트를 Runtime에 CONNECT한다. `auto-webrtc`는 클라이언트
연결 전에는 OpenXR 시스템 정보를 제공하지 않으므로 수집기보다 클라이언트를 먼저 연결한다.

```bash
cd HumanoidScene

./collect_quest_teleop.sh \
  --robot-model s200062 \
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

양쪽 입력의 첫 인식·재인식 프레임은 기준 보정에만 사용하여 새로운 팔 이동량을
추가하지 않는다. 따라오기가 켜져 있으면 추적 손실 중에도 기존 유효 목표는 유지한다.

## 4. 조작 및 episode 제어

기본 입력은 **컨트롤러**(`--input-mode controllers`)다. 컨트롤러를 들고 조작한다.
위치·회전 변화가 같은 쪽 팔에 연결되고, 검지 트리거를 절반 이상 당기면 그리퍼를
닫고 놓으면 연다. 입력을 잠깐 잃으면 팔은 마지막 유효 목표를 계속 따라가고 그리퍼 목표도 유지한다. 재인식 첫 프레임은
팔 보정에만 쓴다. `--input-mode hands`는 기존 맨손 wrist·pinch 모드를 선택한다.
모드는 실행 시 고정되며 추적 손실을 이유로 서로 전환하지 않는다.

기본값은 `--no-auto-start --max-episodes 0 --episode-seconds 0`이다. 앱을 켜 둔 채
버튼으로 시도를 반복한다. `--auto-start`를 명시하면 유효한 양쪽 입력으로 자동 녹화한다.

처음에는 `--no-auto-start`를 권장한다. 아래 버튼은 OpenXR controller press에
연결되어 있으며, 누르면 수집기 터미널에 `[BUTTON] X pressed ...` 같은 로그가 찍힌다.

| 동작 | Quest 컨트롤러 | Isaac Sim desktop key |
|---|---|---|
| 시점·자세 보정 | 왼쪽 `X` | `C` |
| 저장 없이 따라오기 시작/멈춤 | 오른쪽 `A` | `T` |
| recording 시작/종료(실패) | 오른쪽 `B` | `P` |
| 손목 패널 표시/숨김 | 왼쪽 `Y` | `H` |
| 환경 reset/현재 시도 실패 종료 | — | `R` |
| 성공 demonstration으로 종료 | — | `M` |

1. 편한 자세로 정면을 보고 `X`/`C`를 누른다. 현재 HMD 시점을 Kuavo head camera
   위치·방향에 맞추고 손·머리 움직임의 기준을 다시 잡는다. 따라오기는 꺼진다.
2. 양쪽 컨트롤러를 보이고 `[TRACKING] left=True, right=True, head=True input=controllers`를 확인한다.
   `A`/`T`를 눌러 저장 없이 따라오기를 켜고 작은 움직임부터 시험한다.
3. `B`/`P`로 녹화를 시작한다. 양쪽 입력 추적이 유효해야 실제 녹화가 시작된다.
   추적 대기 중에는 `REC WAIT`가 표시되고 `B`/`P`를 다시 누르면 예약이 취소된다.
4. 성공이면 PC에서 `M`, 실패/중단이면 `B`/`P`를 누른다. 녹화 중 `A`/`T`를
   눌러도 따라오기를 멈추고 현재 녹화를 실패로 종료한다.

보정은 녹화 중 차단된다. 따라오기만 멈출 때는 마지막 머리·그리퍼 목표를
유지하고 남아 있는 팔 목표를 현재 자세로 취소한다. 에피소드 종료 시 파일만 닫고
장면은 유지한다. `R`만 초기 장면으로 되돌린다. `--max-episodes 1`을 명시하면
한 시도 후 앱까지 종료하므로 연속 조작에는 사용하지 않는다.
HMD 시점 이동은 가상 시점 보정이며 실제 로봇의 머리 위치 관절을 추가하지 않는다.

팔 목표는 현재 손끝이 아닌 이전 목표에 상대 이동량을 누적한다. 팔이 입력보다 느릴 때
남은 목표가 다음 프레임에 사라지는 것을 막는다. 각 IK 보정 단계의 오차 입력은
15 cm/0.5 rad로 제한하되 원래 목표는 유지한다. 컨트롤러를
멈추거나 잠깐 추적을 잃어도 남은 목표를 따라간다. A/B로 명시적으로 정지하거나
X로 보정할 때만 현재 자세에 멈춘다. 주변을 보면서 조작하고 정지는 A로 한다.
`[MOTION]`에는 실제 손끝 이동량과 목표 오차(mm)가 3초 간격으로 표시된다.
시뮬레이터 자체 검증은 Isaac Lab Python으로 `scripts/verify_quest_ik.py`를 실행한다.

**맨손 모드에서는 버튼과 손 추적이 별개다.** `--input-mode hands`로 실행했다면
컨트롤러를 잡을 때 Quest가 맨손 추적을 중단할 수 있다. 이 경우 버튼을 누른 뒤
컨트롤러를 내려놓고 양손을 보이거나 PC 키를 쓴다. 기본 controllers 모드에는 해당하지 않는다.
패널에는 `FOLLOW ON/OFF`, `REC ON/OFF/WAIT`, 입력이 없으면 `CHECK CTRL` 또는 `CHECK HANDS`가 표시된다.
버튼 로그가 없으면 controller 입력 전달을 확인하고 PC 키를 사용한다.

컨트롤러 모드 HDF5에는 `openxr_left_controller`/`openxr_right_controller`를 `[2,7]`로
추가 저장한다. 첫 행은 `x,y,z,qw,qx,qy,qz`, 둘째는 `stick_x,stick_y,trigger,squeeze,button_0,button_1,reserved`다.
LeRobot에는 같은 값이 `observation.openxr.left_controller`/`right_controller`의 14차원 벡터로 저장된다.
에피소드 metadata의 `input_mode`가 모드를 나타내며, `tracking_valid`의 left/right는 선택한 입력의 유효성이다.
컨트롤러 모드에서 수집하지 않은 손가락 pose와 pinch는 NaN이다. 합성 손가락 데이터는 기록하지 않는다.
입력 모드를 바꾸면 LeRobot도 새 dataset root를 사용한다(컨트롤러 필드 유무가 달라짐).

성공한 작업은 desktop Isaac Sim 창에서 `M`을 누른다. 별도 클라이언트의 `START`/`STOP`/`RESET`
teleop message도 계속 지원하며, CloudXR.js sample에서 이 메시지를 보내지 않아도 위 버튼·PC 키를 사용할 수 있다.

LeRobot 모드에서는 기본적으로 `M`으로 성공 처리한 episode만 저장한다. `STOP`, `RESET`, time limit episode는 학습 데이터에 섞이지 않도록 폐기된다. 실패 episode도 분석용으로 보존하려면 `--lerobot-save-failed`를 추가한다.

HDF5는 샘플이 있는 실패·reset·시간 초과 episode도 저장한다. 따라서 `both`
모드에서 HDF5와 LeRobot의 episode 수는 다를 수 있다. `M`은 작업자가 성공을
판정하는 입력이며, 실제 작업을 완료했을 때 사용한다.

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

- 중앙에는 Quest의 일반 stereo scene view를 유지한다. 큰 head RGB 패널로 덮지 않는다.
- `scene["left_wrist_camera"]`, `scene["right_wrist_camera"]` 영상이 시야 왼쪽 위/오른쪽 위 작은 창으로 표시된다.
- 패널은 눈앞 0.35 m, 크기 0.20×0.15 m이며 머리를 따라 고정된다. `Y`/`H`로 숨겨도 카메라 기록은 유지된다.
- 실제 Kuavo `scene["robustness_camera"]` RGB는 dataset에 계속 저장된다. 중앙의 stereo 시점과 이 단안 영상은 동일하지 않다.
- 따라오기 또는 녹화 중 HMD yaw/pitch는 Kuavo `zhead_1_joint`, `zhead_2_joint`에 연결된다. `X`/`C`로 현재 시점을 head camera에 정렬한다.
- Kuavo에는 머리 translation joint가 없으므로 Quest의 위치 이동은 robot head에 적용되지 않는다.
- desktop Isaac Sim에도 head/left wrist/right wrist mini viewport가 열린다.

손목 패널은 기본으로 활성화된다. 시작부터 패널을 만들지 않으려면:

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

1. 중앙에 원래 stereo 장면이 보이고 큰 검은 패널이 없는지 확인한다.
2. 왼쪽 위 `LEFT WRIST`, 오른쪽 위 `RIGHT WRIST`에 영상과 상태 표시가 보이는지 확인한다.
3. `X`/`C`로 정면을 맞춘 뒤 `A`/`T`를 누르고, 머리를 좌우로 돌릴 때 Kuavo `zhead_1_joint`가 회전하는지 확인한다.
4. 따라오기 중 Quest를 위아래로 돌렸을 때 `zhead_2_joint`가 제한 범위 안에서 회전하는지 확인한다.
5. 몸을 앞뒤로 움직이는 translation은 Kuavo 머리에 적용되지 않는 것이 정상이다. 현재 model에는 yaw/pitch 두 관절만 있다.

패널이 검으면 `[CAMERA] Left/Right wrist RGB: min=..., max=..., mean=...`를 확인한다.
`max`가 0보다 큰데 패널만 검다면 카메라 자체가 아니라 XR UI 표시 경로를 점검한다.
위젯은 단일 자식 Frame 안에 이미지 레이아웃을 넣고, 첫 유효 영상 전에는 패널을 숨긴다.
`Y`/`H`로 숨겨 중앙 장면과 구분할 수 있다. 뒤쪽에 패널이 생긴 경우에만
`--xr-overlay-forward-axis +z`를 시도한다. 중앙을 비우는 배치는 의도된 것이며
화면 전체를 덮기 위해 거리를 줄일 필요는 없다. 크기·위치는 `QuestCameraOverlayCfg`에서 조절한다.

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
  --lerobot-root datasets/kuavo_quest_lerobot \
  --lerobot-repo-id paragon7060/kuavo_quest_teleop
```

다른 Python 3.12 LeRobot v3 환경을 쓰려면 `LEROBOT_PYTHON`을 export하거나
`--lerobot-python /path/to/env/bin/python`으로 지정한다. launcher는 알려진 conda
경로도 탐색하며, worker는 시작할 때 format이 정확히 `v3.0`인지 확인한다.

HDF5는 시도마다 새 파일로 분리되며 기존 파일을 재사용하지 않는다. 지정한
`--dataset`은 첫 파일이며 이미 있으면 오류로 중단한다. 다음 시도는 같은 폴더의
새 고유 파일을 사용한다. 저장 경로는 `[DATA] New HDF5 file`에 표시된다.

기본 camera 저장은 MP4다. 개별 PNG가 필요하면 `--no-lerobot-use-videos`를 사용한다. 기존 **LeRobot** dataset에 이어서 수집할 때에는 FPS, camera 해상도, wrist camera 포함 여부, box/button 수가 최초 schema와 같아야 한다. 이 값이 바뀌면 새 `--lerobot-root`를 사용한다.

v3 주요 feature:

```text
observation.state                    [T, 24] (S200062: 16 upper-body + 8 gripper joints)
# S63/Robotiq comparison: [T, 32] (16 upper-body + 16 gripper joints)
observation.velocity                 [T, 32]
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
action                                [T, 16] (14 IK/head + left/right gripper)
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
    action                         [T, 16]
    robot_joint_position           [T, 32]
    robot_joint_velocity           [T, 32]
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

path = "datasets/SESSION_FILE.hdf5"  # [INFO] Dataset에 출력된 실제 파일 경로
with h5py.File(path, "r") as f:
    for name, demo in f["data"].items():
        print(name, demo.attrs["num_samples"], demo.attrs["success"])
        print("  action:", demo["samples/action"].shape)
        print("  head_rgb:", demo["samples/head_rgb"].shape)
PY
```

## 11. 문제 해결

### Quest 화면은 연결되지만 tracking이 계속 False

- 먼저 `[TRACKING] ... input=controllers/hands`로 선택한 입력을 확인한다.
- controllers 모드는 좌우 컨트롤러가 켜져 있고 Quest와 연결되어 있는지, 추적 가능한 위치인지 확인한다.
- hands 모드만 hand tracking permission을 허용하고 컨트롤러를 내려놓은 뒤 맨손을 카메라에 보인다.
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

### Gripper가 forearm과 겹침

기본 preset은 대회용 Robotiq 2F-85 기반 Leju claw를 양쪽 손목에 사용한다.
S63 비교 모드는 `preview_quest_local.sh --robot-model s63 --gripper robotiq_2f85`에서
확인한 뒤 `configs/grippers.json`의 해당 side
`robot_mount_pos`/`robot_mount_rot`를 조정한다. S200062의 gripper와 D405 mount는
로봇 URDF에 직접 정의되어 있으므로 외장 gripper mount JSON을 사용하지 않는다.

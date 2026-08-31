# Meta Quest 3 → Kuavo Isaac Lab teleoperation/data collection

이 문서는 현재 workcell을 Meta Quest 3에서 보고, OpenXR controller tracking으로 Kuavo의 양팔·베이스·허리·머리를 조작하면서 LeRobot Dataset v3 또는 HDF5 demonstration을 수집하는 절차를 정리한다.

처음 설치한다면 [README의 다운로드부터 첫 저장까지 안내](../README.md#quest-collection)를
먼저 따른다. Runtime SDK와 npm `.tgz`의 공식 다운로드, JSON 경로 설정,
Linux 서비스 준비, Quest 브라우저의 HTTP/HTTPS 설정, 포트 구분을 단계별로 설명한다.
이 문서는 그 이후의 조작·카메라·데이터 schema 세부 설정을 다룬다.

## 1. 구현 구조

```text
Quest 3 WebXR controller/head tracking
        ↓ CloudXR Runtime 6.x / OpenXR
Isaac Lab OpenXRDevice
        ↓ Kuavo absolute grip-position mapper
left grip xyz → left 7-DoF position-priority IK
right grip xyz → right 7-DoF position-priority IK
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
- `--gripper none`은 gripper 채널만 제외한다. 현재 기본 absolute+body 구성에서는 22-D action이다.

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

기본 absolute 모드는 추적 손실 중 마지막 목표를 유지하고 재인식하면 현재 위치를 목표로 쓴다.

## 4. 조작 및 episode 제어

기본 입력은 **컨트롤러**(`--input-mode controllers`)다. VR 손잡이 위치가 같은 쪽
로봇 손끝의 절대 목표다. 검지 트리거는 절반 이상 당기면 닫고 놓으면 연다.
생성·R 리셋도 열린 자세다. 추적이 끊기면 마지막 목표를 유지하며 맨손으로 자동 전환하지 않는다.
맨손은 `--input-mode hands`로 별도 실행한다.

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
| 베이스 전후/좌우 이동 | 왼쪽 스틱 | — |
| 허리 좌우 회전/몸통 상하 | 오른쪽 스틱 | — |
| 누르는 동안 자유 시점, 놓으면 head 복귀 | 왼쪽 아래 그립 트리거 | — |
| 놓으면 open, 당기면 close | 양쪽 위 검지 트리거 | — |

1. 편한 자세로 정면을 보고 `X`/`C`를 누른다. 현재 HMD 시점을 Kuavo head camera
   위치·방향에 맞추고 손·머리 움직임의 기준을 다시 잡는다. 따라오기는 꺼진다.
2. 양쪽 컨트롤러를 보이고 `[TRACKING] left=True, right=True, head=True input=controllers`를 확인한다.
   `A`/`T`를 눌러 저장 없이 따라오기를 켜고 작은 움직임부터 시험한다.
3. `B`/`P`로 녹화를 시작한다. 양쪽 입력 추적이 유효해야 실제 녹화가 시작된다.
   추적 대기 중에는 `REC WAIT`가 표시되고 `B`/`P`를 다시 누르면 예약이 취소된다.
4. 성공이면 PC에서 `M`, 실패/중단이면 `B`/`P`를 누른다. 녹화 중 `A`/`T`를
   눌러도 따라오기를 멈추고 현재 녹화를 실패로 종료한다.

녹화 중 X는 파일을 `operator_calibration`으로 닫고 보정한다. A/B로 명시적으로
멈추면 팔은 현재 자세에 멈춘다. 검지 트리거는 팔 정지 중에도 작동한다. 에피소드 종료 시 파일만 닫고
장면은 유지한다. `R`만 초기 장면으로 되돌린다. `--max-episodes 1`을 명시하면
한 시도 후 앱까지 종료하므로 연속 조작에는 사용하지 않는다.
HMD 시점 이동은 가상 시점 보정이며 실제 로봇의 머리 위치 관절을 추가하지 않는다.

기본 `--controller-mapping absolute`는 위치 gain이나 이동량 누적을 사용하지 않는다.
X는 시점을 head에 맞추고 손잡이와 tool의 방향 차이만 보정한다. 위치 오프셋을
추가하지 않으므로 컨트롤러가 보이는 곳과 손끝 목표가 같다. 기본 `--arm-orientation-weight 0.5`는 위치와 손잡이 방향을 함께 추종한다.
0은 방향 추종을 끄는 진단 옵션이다. 회전 목표가 관절 제한을 벗어나면 위치·방향 오차가 남을 수 있다. 보정 입력은 15cm/0.5rad, 관절 목표 변화는 물리
스텝당 0.12rad 이내다. 남은 목표는 A/B 정지 또는 X 보정 때만 현재 자세로 바뀐다.
기존 상대 이동은 `--controller-mapping relative`; `--position-gain`은 이 모드에만 적용된다.

스틱은 A/B로 따라오기를 켠 상태에서 작동한다. 최대 베이스 속도 0.25m/s,
허리 yaw 0.5rad/s·±1.4rad, 몸통 높이는 초기 대비 0~+40cm·최대 0.12m/s다.
높이는 knee/leg/waist pitch를 함께 제어해 몸통을 세운 채 조절한다.
S200062 초기 자세는 높이 하한이므로 초기 상태의 아래 입력은 움직이지 않는다. 위로 올린 뒤 아래로 내린다.
`[BODY]`에서 오른쪽 stick, enabled, 목표와 실제 관절값을 비교한다.
왼쪽 아래 그립을 누른 자유 시점에서는 스틱 제어도 정지한다. 베이스는
**fixed articulation root를 이동시키는 시뮬레이션용 평면 제어**다. 실제 바퀴 주행
정책이나 충돌 회피가 아니므로 작은 입력으로 조작하고 물체 접촉 시 A로 멈춘다.

S200062 gripper의 관절 제한은 ±0.698rad이며 0rad는 거의 닫힌 자세다.
열림은 f_bar_1/3=-0.25, b_bar_1/3=+0.25rad(메시 안쪽 간격 약 9cm),
닫힘은 각각 -0.005/+0.005rad로 작은 여유를 둔다. 실제 USD는 4절 링크의 트리 근사이므로
실물 기구학과 동일하다고 가정하지 않는다. 생성·R 리셋 모두 열림 자세다.

팔꿈치를 조금 굽힌 준비 자세로 생성한다. simulation PD stiffness/damping은
팔 800/50, 몸통 지지 8000/200, 허리 yaw 800/50이며 기존 effort limit은 유지한다. 팔 게인은
`--arm-stiffness`, `--arm-damping`으로 조절한다. 실물 로봇용 게인이 아니다.
높은 곳이 도달 범위를 벗어나면 오른쪽 스틱으로 몸통을 올린다. `[MOTION]`의
목표 오차(mm)를 확인한다. 실제 물리 검증: `scripts/verify_quest_controls.py`.

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
입력/제어 모드나 action schema가 바뀌면 LeRobot도 새 dataset root를 사용한다.
새 기본 action 24차원: 좌/우 tool xyz+qwqxqyqz(base frame), head yaw/pitch,
좌/우 gripper, base forward/left 속도, knee/leg/waist pitch/yaw 목표. 이름은 `action_layout`에 저장한다.

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
- 패널은 눈앞 0.35 m, 크기 0.14×0.105 m이며 머리를 따라 고정된다. `Y`/`H`로 숨겨도 카메라 기록은 유지된다.
- 실제 Kuavo `scene["robustness_camera"]` RGB는 dataset에 계속 저장된다. 중앙의 stereo 시점과 이 단안 영상은 동일하지 않다.
- 따라오기 또는 녹화 중 HMD yaw/pitch는 Kuavo `zhead_1_joint`, `zhead_2_joint`에 연결된다. `X`/`C`로 현재 시점을 head camera에 정렬한다.
- Kuavo에는 머리 translation joint가 없으므로 Quest의 위치 이동은 robot head에 적용되지 않는다.
- 기본 시점의 위치는 head camera에 붙으며 HMD 회전은 그대로 반영해 주변을 볼 수 있다. 왼쪽 아래 그립을 누르면 room-scale 자유 시점, 놓으면 head 복귀다.
- 자유 시점 동안 새 팔·몸통 명령은 멈추고 마지막 팔 목표는 유지한다. HDF5 `free_view`로 구분한다.
- PC 카메라 창은 `--camera-preview`, desktop 장면 렌더는 `--desktop-render`로 켠다. 기본은 OFF다.

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

기본 해상도는 head 640×360, wrist 각각 240×180이며 depth 저장은 OFF다.
S200062의 head와 D405 모두 URDF body +X를 ROS optical +Z로 변환한다.
이 변환이 없으면 head 녹화 영상이 작업대 대신 천장 쪽을 볼 수 있다.
수집용 장면에서 사람·배경 이동 로봇을 제거한다. 기본 `--scene-detail compact`는
창고 배경 랙·팔레트·카트·적재물도 비활성화하고, 로컬 작업 상자 선택 시 사용하지 않는
legacy tote 9개·cargo 12개는 생성하지 않는다. 작업용 랙·상자·컨베이어와 공장 바닥·벽·천장·
조명·재질·텍스처는 유지한다. 원본 USD 파일은 바꾸지 않으며 `--scene-detail full`로 비교할 수 있다.
단일 환경은 CPU 물리/IK, GPU RTX 렌더를 기본으로 쓴다. 훈련용 관측은 계산하지 않고
HDF5는 영상 프레임별 chunk로 저장한다. `--control-hz 60`은 시뮬레이션 시간 기준이다.
실제 속도는 `[PERF]`의 wall-clock Hz를 확인한다. 60Hz에는 전체 프레임이 16.7ms 이내여야 한다.
`--profile-steps 120`으로 병목을 측정하고 `--capture-xr`로 실제 XR 출력 PNG를 저장한다.
`--xr-resolution-scale`은 XR 렌더 버퍼의 가로·세로 배율이다. 기본 1.0으로
런타임 권장 해상도를 그대로 유지한다. 0.5배 실험은 약 19Hz에 그쳤으며 사용감·선명도
손실이 커 기본값에 적용하지 않았다. 수집용 head/wrist RGB 해상도와는 별개다.
`--render-quality quality --xr-resolution-scale 1.0`은 효과·선명도 우선 설정이다.

XR 중 데스크톱 viewport의 업데이트를 끄면 Kit 107.3에서 카메라 RGB도 빈 배열이 될 수 있다.
녹화/패널 사용 중에는 **PC용 160×90 렌더를 유지**하고, 원시 annotator 출력이 유효한지 먼저 검사한다.
빈 프레임은 저장하지 않고 다음 프레임을 기다린다. Quest 양안과 저장 RGB 해상도는 그대로다.
실제 XR 녹화 재시작·파일 검증: `scripts/verify_quest_recording.py`
(합성 테스트 파일은 `artifacts/recording-verification/`에만 저장).

양쪽 손목 카메라를 완전히 끄려면 `--no-wrist-cameras`를 추가한다.
센서를 생성하지 않으므로 손목 렌더·패널·영상 저장이 모두 꺼진다. head 카메라와
VR 양안 해상도, 컨트롤러 조작은 유지한다. `--no-record-wrist-cameras`는 저장만,
`--no-quest-camera-overlay`는 패널만 끄므로 카메라 렌더 비용 제거와는 다르다.
손목 영상을 다시 사용하려면 `--wrist-cameras`로 재실행한다. 영상 feature 구성이
달라지므로 LeRobot 사용 시 기존 데이터셋에 이어 쓰지 말고 새 root를 지정한다.

2026-08-31 추가 검증: 배경 소품 56개를 제외해 factory prim이 3417→1354개로 줄었고,
legacy 물리 물체 21개도 제외했다. 같은 head 카메라 방향·VR 배율 1.0·손목 OFF 조건에서
합성 XR 녹화 루프는 full 약 16.13Hz, compact 약 16.15Hz로 유의미한 차이가 없었다.
확인 시 Quest 외부 소켓이 없었으므로 실제 스트리밍 FPS 비교는 아니다.
두 구성 모두 녹화 재시작 2회, 시도당 유효 RGB 88프레임(360×640×3)을 별도 HDF5로
저장하고 재개 후 영상이 갱신되는 것을 확인했다. 시작 직후 빈 2프레임은 쓰지 않았다.
`artifacts/quest-scene-comparison.json`은 이 로컬 검증 결과다.
실제 착용한 손목 OFF/추종 ON의 이전 측정 중앙값은 25.55Hz였지만, 위 합성 비교와
조건이 다르므로 자산 제거에 따른 향상률로 계산하지 않는다.
30Hz 제어를 선택하려면 `--control-hz 30`을 사용한다. 실제 30Hz 보장은 아니다.
최종 compact 실행에서 Quest 외부 연결을 확인하고 오른쪽 스틱 입력·실제 허리 회전/승강을 확인했다.
제어 주기 30Hz 설정의 실제 루프는 추종 약 24Hz, 녹화 약 20~22Hz였다.
B로 시작하고 X 보정으로 닫은 실제 파일에서 head RGB 281프레임, 관절값·컨트롤러값,
증가하는 타임스탬프를 검증했다. 일부 큰 팔 목표에서는 위치·방향 오차가 상당히 남으므로,
작은 도달 가능 목표의 물리 검증 결과를 전체 VR 작업영역 성능으로 해석하면 안 된다.

2026-08-31 손목 센서 ON/OFF 비교: 같은 공장 장면, VR 배율 1.0, head 640×360,
depth OFF, 녹화·추종 OFF에서 안정화 후 각각 30초 측정한 `[PERF]` 중앙값은
**ON 15.4Hz / OFF 15.5Hz**였다. 다만 측정 후 CloudXR 소켓 확인에서 Quest의 외부
연결이 없고 PC 내부 연결만 남아 있었다. 이는 **대기 상태 참고값**으로,
실제 착용·스트리밍 중 손목 카메라 제거 효과를 입증한 결과가 아니다.
렌더 경로에는 Kit 갱신과 XR 대기도 포함되므로 이 결과만으로 창고 geometry나
GPU 렌더가 단독 원인이라고 판단하지 않는다. Quest 연결을 유지한 비교가 필요하다.

2026-08-31 RTX 3060 검증에서 렌더 배율 1.0의 단일 환경은 약 15.4Hz(녹화 OFF, Quest 연결 미확인 대기 조건)였다.
이는 네트워크 포함 headset 지연 측정값이 아니라 수집기 루프 속도이며, 60Hz 달성을 뜻하지 않는다.
실제 물리 테스트의 15cm 위쪽 목표 오차는 기존 게인 약 5cm → 변경 후 약 2cm,
더 높은 1.2m 목표에서는 약 3.7cm였다. 접촉 없는 별도 위치의 결과이므로 랙 접촉이나
도달 범위 밖의 목표에서는 오차가 더 커질 수 있다.

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
observation.state                    [T, 28] (S200062: 20 arm/head/body + 8 gripper joints)
# S63/Robotiq comparison: [T, 36] (20 arm/head/body + 16 gripper joints)
observation.velocity                 [T, 28] (S200062)
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
action                                [T, 24] (arms 14 + head 2 + grippers 2 + body 6)
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
    action                         [T, 24] (default)
    robot_joint_position           [T, 28] (S200062)
    robot_joint_velocity           [T, 28] (S200062)
    robot_root_pose_w              [T, 7]
    free_view                      [T]
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

입력 축은 [OpenXR 규격](https://registry.khronos.org/OpenXR/specs/1.0-khr/html/xrspec.html#input-suggested-bindings)에 따라 +Y가 스틱 위쪽이다. WebXR Gamepad 원시 축 부호와 혼동하지 않는다.

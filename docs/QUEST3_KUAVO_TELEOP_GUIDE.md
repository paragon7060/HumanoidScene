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
- `--gripper none`은 gripper 채널만 제외한다. 현재 scaled/absolute+body 구성에서는 22-D action이다.

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

기본 scaled 모드는 추적 손실 중 마지막 목표와 보정 기준을 유지한다. 재인식하면 기존 기준 대비
컨트롤러 변위에 배율을 적용하며, 자동으로 보정 기준을 바꾸지는 않는다.

## 4. 조작 및 episode 제어

기본 입력은 **컨트롤러**(`--input-mode controllers`)다. 보정 기준 대비 손잡이 이동을
1.1배 확대하여 같은 쪽 로봇 손끝의 목표로 사용한다. 검지 트리거는 절반 이상 당기면 닫고 놓으면 연다.
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
| 베이스 좌우 회전/몸통 상하 | 오른쪽 스틱 | — |
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

기본 `--controller-mapping scaled --position-gain 1.1`는 손 이동을 조금 확대하고 A 재보정으로 도달 범위를 보완한다.
정면을 보고 X로 시점을 맞춘 다음, 양손을 몸 앞에 편하게 두고 A를 누른다. 각 손의 첫 유효한
컨트롤러 위치·방향과 현재 로봇 손끝 자세를 연결해 기준으로 삼는다. 그 위치에서 손을 앞으로/아래로
20cm 옮기면 손끝 목표는 22cm 이동한다. 방향에는 위치 gain을 곱하지 않는다.

A로 정지한 뒤 손을 편한 위치·방향으로 옮기고 A를 다시 누르면 현재 로봇 손끝 위치·방향부터
새로운 기준으로 이어간다. 이전 명령에 아직 도달하지 못했더라도 실제 손끝 자세를 기준으로 잡는다.
아래쪽 파지에서 손목을 더 돌리기 어려우면 조금 회전 → A 정지 → 손목을 편하게 되돌림 → A 재개를
반복할 수 있다. 회전은 몸통 기준 grip quaternion의 보정 시점 대비 변화량을 1:1로 적용한다.
aim pose 유무에 따라 회전 기준을 바꾸지 않으며 컨트롤러와 gripper의 절대 방향이 일치할 필요도 없다.
**녹화 중 A 정지는 현재 시도를 종료한다. 같은 녹화를 유지하는 일시정지 기능은 아니다.**
추적이 잠깐 끊기는 것만으로 기준을 바꾸지는 않는다. 몸통 좌표계에서 변위를
계산하므로 베이스 이동·회전이나 몸통 높이 변경은 배율만큼 과장되지 않는다.
`--position-gain 1.8`처럼 1.0~3.0 범위에서 조절할 수 있지만, gain이 크면 손 떨림도 커진다.
현재 기본값은 1.1이며 팔 속도·가속도 제한과 필터는 유지한다.

데이터 수집에서는 우선 녹화 전에 A로 파지에 편한 손목 기준을 잡고, B로 시작한 뒤 연속 수행한다.
물체가 안정되기를 기다리는 정지와 작업자가 컨트롤러를 다시 잡기 위한 정지는 구분해야 한다.
후자를 많이 포함하면 작업과 무관한 대기 동작을 학습할 가능성이 있으나, 실제 영향은 학습 정책과
데이터 분포에 따라 검증해야 한다. 현재 A는 파일을 끝내므로 같은 녹화 안에서 재보정 대기를 기록하지 않는다.
향후 녹화 중 재보정을 지원한다면 원본 시간축과 상태를 보존하고 별도 clutch 표시를 남겨 학습 시
구간을 구분하는 편이 안전하다. 정지 중에도 상자·덮개가 움직일 수 있으므로 단순 삭제 후 이어 붙이지 않는다.
이 구간 표시/학습 필터 기능은 아직 구현하지 않았다.

위치·방향 재보정, 회전 1:1, 몸통 회전, 추적 손실·재인식, 양손 독립성 및 quaternion 부호 변경을
포함한 단위·소켓 테스트 105개를 통과했다. 이번 변경은 매핑 검증이며 실제 Quest 파지 성공률은 별도 확인한다.

확대 모드에서는 VR 컨트롤러와 손끝이 항상 같은 위치일 수 없다. 기존 1:1 위치 일치를 원하면
`--controller-mapping absolute`를 사용한다. X는 시점·머리 기준을 재설정하고 따라오기를 정지한다.
absolute 모드에서는 S200062의 tool -Z(접근 방향)는 OpenXR aim -Z(검지 pointing), tool +X(집게 닫힘 축)는
그 방향에 수직으로 투영한 grip -Z(엄지 쪽)에 맞춘다. X를 누를 때마다 임의의 회전 오프셋을
저장하지 않는다(scaled 모드는 A 재개 시 방향 오프셋도 잡는다). 실제 손가락 관절 측정은 아니며, 컨트롤러 좌표계로 검지·엄지 방향을 근사한다.
aim이 없으면 grip -Y를 접근 방향으로 쓴다. 좌표 규약은
[OpenXR specification](https://registry.khronos.org/OpenXR/specs/1.0-khr/html/xrspec.html)을 참고한다.
기본 `--arm-orientation-weight 0.5`는 위치와 방향을 함께 추종하며, 0은 방향 추종을 끄는 진단 옵션이다.

팔 IK는 제어 tick마다 한 번 계산하고 물리 substep 사이에는 같은 관절 목표를 유지한다.
입력 필터 시정수는 45ms, 관절 속도/가속도 제한은 1.5rad/s, 12rad/s²다.
중력 보상과 제한된 목표 누적으로 처짐을 줄이되, 관절 목표가 실제 값보다 0.1rad 이상
앞서 누적되지 않게 한다. 도달 범위·관절 제한·물체 접촉 때문에 오차가 남을 수 있다.
컨트롤러를 멈추거나 추적을 잃어도 남은 목표를 유지하며, A/B 정지 또는 X 보정 때만 현재 자세로 바뀐다.
기존 상대 이동은 `--controller-mapping relative`; 이 모드에도 `--position-gain`이 적용된다.

스틱은 A/B로 따라오기를 켠 상태에서 작동한다. 최대 베이스 속도 0.25m/s,
베이스 yaw 최대 1.2rad/s(약 69°/s, 누적 회전 제한 없음), 몸통 높이는 초기 대비 0~+40cm·최대 0.12m/s다.
높이는 knee/leg/waist pitch를 함께 제어해 몸통을 세운 채 조절한다.
S200062 초기 자세는 높이 하한이므로 초기 상태의 아래 입력은 움직이지 않는다. 위로 올린 뒤 아래로 내린다.
`[BODY]`에서 오른쪽 stick, enabled, 목표와 실제 관절값을 비교한다.
왼쪽 아래 그립을 누른 자유 시점에서는 스틱 제어도 정지한다. 베이스는
**fixed articulation root의 위치·방향을 바꾸고 네 바퀴 회전을 동기화하는 시뮬레이션용 평면 제어**다.
S200062 바퀴는 omni roller 대신 원통 collider를 사용하므로 이 모드에서는 바퀴 collider만 끈다.
바퀴 접촉력으로 주행하는 모델이나 충돌 회피가 아니므로 작은 입력으로 조작하고 물체 접촉 시 A로 멈춘다.

S200062 gripper의 관절 제한은 ±0.698rad이며 0rad는 거의 닫힌 자세다.
열림은 f_bar_1/3=-0.25, b_bar_1/3=+0.25rad(메시 안쪽 간격 약 9cm),
닫힘 목표는 네 관절 모두 0rad다. 기존 ±0.005rad 목표는 얇은 벽보다 넓은 간격을 남길 수 있어 제거했다. 실제 USD는 4절 링크의 트리 근사이므로
실물 기구학과 동일하다고 가정하지 않는다. 생성·R 리셋 모두 열림 자세다.

팔꿈치를 조금 굽힌 준비 자세로 생성한다. simulation PD stiffness/damping은
팔 800/50, 몸통 지지 8000/200, 허리 yaw 800/50이며 기존 effort limit은 유지한다. 팔 게인은
`--arm-stiffness`, `--arm-damping`으로 조절한다. 실물 로봇용 게인이 아니다.
S200062 원본에서 관성이 누락된 손·카메라·좌표 링크가 USD에서 각각 1kg으로 들어가
한쪽 손 부근에 약 16kg이 추가되어 있었다. Quest 환경은 이 34개 링크만
`assets/kuavo_s200062/teleop_inertials.json`의 질량·COM·관성으로 보완한다(한쪽 합계 0.743kg).
**시뮬레이션용 추정값이며 제조사 측정값이 아니다.** 관성은 메시 AABB의 균일 직육면체 근사다.
작은 그리퍼 링크의 수치적 진동을 억제하기 위해 모터 armature 0.001kg·m²(추정),
articulation solver iteration 32/8을 사용하며 gripper effort 상한 5Nm는 유지한다.
URDF에 명시된 팔·몸통 관성과 원본 USD는 수정하지 않는다. 실물 동역학 일치를 위해서는 실제 부품 관성으로 교체해야 한다.

HMD 회전은 보정 시 머리의 로컬 기준으로 계산한다. 왼쪽을 보면 head yaw 양수,
위를 보면 head pitch 음수이며 베이스 회전과 분리한다. 매 프레임 이전 HMD 방향으로
teleport하지 않고 anchor의 위치만 갱신하여 물리적인 머리 회전이 상쇄되지 않게 한다.

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
좌/우 gripper, base forward/left/yaw 속도, knee/leg/waist pitch 목표. 이름은 `action_layout`에 저장한다.
기존 24차원과 길이는 같아도 마지막 6개 의미가 바뀌었으므로 파일의 이름 목록을 기준으로 해석한다.

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

### 2026-09-01 팔·방향·베이스 회귀 검증 (손 충돌체 보완 전)

30Hz 제어/120Hz 물리, S200062, CPU 물리, compact 환경에서 검증했다.
실제 HMD 센서 노이즈나 네트워크 지연 측정과는 구분한다.

| 검증 | 결과 |
|---|---|
| FK로 만든 도달 가능한 전방 파지 자세 | 양팔 위치 약 0.91mm, 방향 약 0.00185rad 오차 |
| 방향 유지하며 위로 10cm | 위치 약 1.21mm 오차 |
| 베이스 좌우 회전 | ±1.2rad/s 입력 1초 후 1.2rad 회전/원위치, 네 바퀴 회전 확인 |
| 몸통 상하 | +24cm 목표에 +23.1cm, 내려오면 초기 높이 복귀 |
| 머리 좌우·상하 | yaw ±0.3rad / pitch ∓0.2rad 목표, 방향·오차 검증 통과 |
| 그리퍼 | 열림 약 90mm, 닫힘 약 2.5mm, reset 열림 |

`verify_quest_stability.py`의 고정 목표 측정에서 높이 든 자세의 프레임 간 평균 이동은
기존 약 2.49mm에서 0.21mm로 감소했다. 이 값에는 느린 수렴도 포함되므로 순수 센서 jitter로
해석하지 않는다. 도달 가능한 전방 파지로 복귀한 뒤에는 위치 약 0.94mm 오차로 수렴했다.
**손목을 아래로 유지한 높은 목표에는 아직 약 8.6cm, 그 자세로 복귀하는 8초 구간에는 약 11.2cm
위치 오차가 남는다.** 특히 S200062 손목 마지막 관절 한계는 ±0.698rad(40°)이므로 모든 컨트롤러
6D pose를 실현할 수 있는 것은 아니다. 목표를 버리거나 조용히 재보정하지는 않는다.

```bash
PYTHONPATH=src KUAVO_ROBOT_MODEL=s200062 python scripts/verify_quest_controls.py
PYTHONPATH=src KUAVO_ROBOT_MODEL=s200062 python scripts/verify_quest_stability.py
# XR runtime 환경을 먼저 불러온 뒤 실행. 합성 녹화는 artifacts/에만 저장한다.
PYTHONPATH=src KUAVO_ROBOT_MODEL=s200062 python scripts/verify_quest_recording.py
```

이전 제어 비교는 이전 `teleop_ik.py`를 별도 경로에 추출하여
`verify_quest_stability.py --legacy-ik <path> --legacy-physics`로 실행한다.
실험은 다른 collector를 종료한 뒤 한 번에 하나만 실행한다.

XR 녹화 중지·재시작 검증도 통과했다. 두 개의 서로 다른 HDF5에 각각 640×360 RGB
88프레임이 저장됐고, head 회전에 따른 프레임 변화도 확인했다. 이 합성 녹화는 실제 demonstration이 아니다.
단위·소켓 회귀 테스트 101개 통과. 실제 Quest 착용 상태의 방향 감각과 tracking noise는 재연결 후 별도 확인한다.

### 팔 길이 보정과 상자·버튼 접촉

기본 확대 매핑은 `scaled`, gain은 1.1이다. `X`로 시점을 맞추고 손을 편하게 든 상태에서 `A`를 누른다.
팔을 더 뻗거나 손목을 더 돌리기 어려우면 `A` 정지 → 손을 편한 위치·방향으로 이동 → `A` 재개로 기준을 옮긴다.
예를 들어 아래로 20cm 움직이면 로봇 목표는 22cm 내려간다. 실제 로봇의 관절 한계는 그대로다.

S200062의 접촉 형상에 다음 문제를 확인하여 Quest 환경에서 보완했다.

- 움직이는 손가락에는 원래 충돌체가 없었다. 각 손가락과 gripper housing에 별도 convex hull을 추가했다.
- 손목의 단순 원통은 관절 아래 140mm까지 내려오지만 실제 손목 메시는 67mm다. 집게 공간을 막는 원통을 손목 메시 충돌체로 대체했다.
- 도구 좌표 링크의 5mm 구형 충돌체 두 개를 비활성화했다. 좌표 프레임은 접촉 물체가 아니다.
- 닫힘에 연동되지 않는 보조 4절 링크에는 충돌을 추가하지 않는다. 원본은 닫힌 기구가 아닌 트리 근사다.
- 손목·손가락·housing은 총 8개 충돌 메시를 사용하며, 집게 사이를 하나의 hull로 채우지 않는다.
- 얇은 상자 벽·덮개와 손에 2mm contact offset / 0 rest offset을 사용하고 speculative CCD를 켰다.
- 닫힘 목표는 0rad로 바꿨다. 기존 ±0.005rad는 얇은 벽을 놓치는 잔여 간격을 남겼다.
- 손 접촉 재질의 static/dynamic friction은 1.0/0.8, restitution은 0이다. 시뮬레이션용 설정이며 재질 실측값은 아니다.

상자 벽을 시각적으로 두껍게 만들거나 재질·텍스처를 바꾸지는 않는다. 물체를 손에 붙이는 고정 조인트나
자동 파지 로직도 사용하지 않는다. 상자의 덮개는 여전히 경첩으로 움직이므로 한쪽 덮개만 들면 상자가
기울 수 있다. 실제 수집에서는 양쪽 덮개를 비슷한 깊이로 잡고 천천히 들어 올린다.

초록색 버튼은 원래 최대 18mm 이동하는 `ButtonJoint`와 복귀 스프링이 있다. 손 충돌체가 없어 접촉을
전달하지 못했던 문제를 수정했다. 6mm 이상 누르면 터미널에 `[CONTACT] Green button PRESSED`,
2mm 미만으로 돌아오면 `RELEASED`가 표시된다. 위치는 기존 `button_joint_position` 데이터에 기록한다.
누름 동작과 작업 성공 판정은 별개이며 모든 상자 배치 등 기존 성공 조건을 자동 우회하지 않는다.

```bash
# 기존 수집기를 종료한 뒤 동일 Isaac Lab 환경에서 실행
PYTHONPATH=src KUAVO_ROBOT_MODEL=s200062 python scripts/verify_quest_contacts.py
```

검증용 장면은 실제 SmallBox 자산을 0.2×0.2×0.18m로 사용한다(벽 두께 2mm, 총 질량 0.52kg).
실제 robot gripper를 닫고 몸통 관절로 들어 올린다. 초기 배치 후에는 상자 pose를 덮어쓰거나 손에
부착하지 않는다. 접촉력, 받침대와의 간격, 놓았을 때 복귀, 별도 충돌 probe로 버튼 누름·복귀를 검사한다.
이것은 통제된 단일 그리퍼 검증이며 Quest 사용자의 모든 양손 파지 자세를 검증한 것은 아니다.

2026-09-01 접촉 보완 후 검증 결과:

| 검사 | 결과 |
| --- | --- |
| 2mm 덮개 파지 | 양쪽 손가락 접촉력 약 33.0N / 23.5N |
| 몸통 관절로 들어 올림 | 상자 몸체 75.7mm 상승, 바닥 전체가 받침대 위 21.2mm 이상 |
| 그리퍼 개방 | 상자가 받침대로 복귀, 바닥 간격 −0.5mm |
| 버튼 접촉·해제 | 최대 12.0mm 눌림, 해제 후 0.000mm로 복귀 |
| 기존 팔·base·몸통·head·gripper 제어 | 회귀 검증 통과, 닫힌 집게 간격 약 0.1mm |
| 단위·소켓 테스트 | 103개 통과 |

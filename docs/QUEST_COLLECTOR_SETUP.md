# 실제 Quest 수집기: 처음 준비부터 재실행까지

이 문서는 `collect_quest_teleop.sh`의 **OpenXR/CloudXR 데이터 수집 경로**를 다룬다.
`preview_quest_browser.sh`는 별도 개발용 bridge이며 여기서는 실행하지 않는다.
아래 명령은 모두 clone한 저장소 루트에서 실행한다.

RL 보상을 검사하려면 기존 연결 뒤 `./quest_collector.sh collect --rl-reward-debug`로
실행한다. 이 모드는 데이터 저장 대신 VR에 항목별 보상을 표시한다.
조작·제약은 [Quest reward 검사](RL_QUEST_REWARD_DEBUG.md)를 참고한다.
V2 grasp SAC 시연 전이를 저장하려면 같은 연결에서
`--rl-reward-debug 2 --rl-demo-dataset <새 파일 경로>`를 사용한다.

## 상황별 바로가기

| 지금 상황 | 할 일 |
|---|---|
| 이 PC에서 처음 준비 | [1. 처음 준비하기](#1-처음-준비하기) → [4. 시작 전 점검](#4-시작-전-점검) |
| **Wi-Fi·공유기를 바꿨거나 PC IP가 달라짐** | [3. 네트워크나 IP가 바뀌었을 때](#3-네트워크나-ip가-바뀌었을-때) |
| 이미 준비된 PC에서 다시 수집 | [5. 매번 실행: 터미널 3개](#5-매번-실행-터미널-3개) |
| 인증서 30일 만료 | [8. 문제 해결](#8-문제-해결)의 인증서 만료 항목 |
| 다른 PC로 옮김 | `session.env`를 복사하지 말고 1절부터 다시 실행 |

처음 준비할 때 한 번만 하는 일과 네트워크가 바뀔 때마다 하는 일은 다음과 같다.

| 작업 | 처음 준비 | 네트워크·IP 변경 | 매번 실행 |
|---|:---:|:---:|:---:|
| Runtime SDK 다운로드, 웹 client build | ✅ | — | — |
| `setup_quest_collector.sh --host <IP>` | ✅ | ✅ (`--update-config`) | — |
| Quest에서 49100·8443 인증서 신뢰 | ✅ | ✅ | — |
| `runtime` / `web` / Quest CONNECT / `collect` | ✅ | ✅ | ✅ |

## 1. 처음 준비하기

이미 [Isaac 설치](INSTALL.md)를 끝낸 Linux PC가 대상이다. 기준은 Python 3.11,
Isaac Sim 5.1.0, Isaac Lab 2.3.2, NVIDIA Runtime 6.2.1 및 CloudXR.js 6.2.0이다.
시스템에 Python 3, `curl`, OpenSSL과 C++17 컴파일러(`c++`)가 필요하다.
LeRobot은 선택 사항이며 별도 환경의 Dataset v3 writer를 사용한다.

PC의 Quest 연결용 LAN IPv4를 먼저 확인한다.

```bash
ip -4 -brief address
```

유선망과 Wi-Fi가 동시에 있는 PC에서는 기본 인터넷 경로가 아니라 **Quest와 통신하는
인터페이스의 IP**를 선택한다. IP는 자동 추정하지 않고 `--host`로 명시한다.
출력의 `172.30.1.14/24`처럼 `/` 앞부분만 사용한다. `docker0`, VPN 등 Quest와
관계없는 인터페이스는 제외한다.

### 웹 클라이언트가 이미 빌드되어 있는 PC

`<PC_WIFI_IP>`와 Python 경로는 실제 값으로 바꾼다. NVIDIA
[CloudXR 라이선스](https://developer.download.nvidia.com/cloudxr/EULA/NVIDIA_CloudXR_GA_License_without_Data_Collection_25Feb2025.pdf)를
확인한 뒤 다운로드 옵션을 사용한다.

```bash
./setup_quest_collector.sh \
  --host <PC_WIFI_IP> \
  --download-runtime \
  --isaaclab-python /absolute/path/to/env_isaaclab_232/bin/python
```

스크립트가 하는 일:

1. NVIDIA 공식 NGC에서 Runtime 6.2.1을 내려받고 고정 SHA-256을 확인한다.
2. 안전한 경로로 SDK를 압축 해제하고 header/library/OpenXR manifest를 확인한다.
3. 해당 IP용 30일 유효 자체 서명 TLS 인증서를 만든다.
4. 기존 웹 build를 수집기 전용 폴더에 복사한다. 원본 preview 파일은 변경하지 않는다.
5. 로컬 `session.env`에 경로·IP·인증서를 기록한다. shell 시작 파일은 변경하지 않는다.

**이 명령은 Runtime, 웹 서버, Isaac Sim을 시작하지 않는다.** GPU 드라이버, Isaac
패키지, 방화벽, 공유기, 시스템 인증서 저장소도 변경하지 않는다.

SDK를 브라우저로 직접 내려받았다면 자동 다운로드 대신 사용한다.

```bash
./setup_quest_collector.sh \
  --host <PC_WIFI_IP> \
  --sdk-archive /absolute/path/to/CloudXR-6.2.1-Linux-sdk.tar.gz
```

다운로드는 [NGC Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/cloudxr-runtime)의
6.2.1 Linux SDK를 사용한다. checksum이 다르면 중단하며 다른 버전을 조용히 설치하지 않는다.
사용자 정의 경로의 Isaac Lab checkout은 `--isaaclab-dir /path/to/IsaacLab`로 지정한다.

### 웹 클라이언트도 없는 새 PC

[CloudXR.js](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/cloudxr-js)에서
`nvidia-cloudxr-6.2.0.tgz`를 별도로 받는다. Runtime SDK와 다른 파일이다.
Node/npm을 준비한 뒤 별도 checkout을 빌드한다.

```bash
export CLOUDXR_NPM_TGZ=/absolute/path/to/nvidia-cloudxr-6.2.0.tgz
export CLOUDXR_JS_SAMPLES_DIR="$PWD/.external/quest-collector-source"
./setup_quest_browser.sh
npm --prefix "$CLOUDXR_JS_SAMPLES_DIR/simple" run build

./setup_quest_collector.sh \
  --host <PC_WIFI_IP> \
  --download-runtime \
  --browser-build "$CLOUDXR_JS_SAMPLES_DIR/simple/build" \
  --isaaclab-python /absolute/path/to/env_isaaclab_232/bin/python
```

`npm install`이나 빌드는 SDK 다운로드와 별개다. setup 도구는 없는 build를 무시하고
진행하지 않으며, `index.html`이 없는 경우 이 절차를 안내하고 멈춘다.

## 2. 생성 파일과 재설정 규칙

```text
.external/quest-collector/
├── downloads/                # NVIDIA 원본 SDK archive
├── runtime/6.2.1/            # SDK, LICENSE.txt, header, library, manifest
├── certs/<PC-IP>/            # server.crt / server.key
├── browser/simple/build/    # 수집용 웹 snapshot
├── bin/                     # check/runtime 시 컴파일되는 전용 실행기
└── session.env              # 이 PC에서만 사용하는 환경변수
```

`.external/`은 Git에서 제외된다. SDK는 Git에 재배포하지 않는다. `session.env`와
개인 키는 권한 0600이며, 다른 PC에서는 그대로 복사하지 말고 setup을 다시 실행한다.
같은 설정으로 재실행하면 SDK·인증서·웹 snapshot·환경 파일을 재사용한다.

기존 환경과 다른 설정은 기본적으로 거부한다. IP/경로를 바꾸려면 이전 설정을
백업한 후 갱신하는 `--update-config`를 명시한다. 이 옵션은 `session.env` 전체를
새 값으로 다시 쓴다. 처음 setup에서 지정한 사용자 정의 인자를 빠뜨리면 기본값으로
돌아가므로 **매번 함께 전달한다.**

| 처음 setup에서 지정했다면 | `--update-config` 때도 전달 | 빠뜨리면 |
|---|---|---|
| `--isaaclab-python` | 같은 경로 | `~/{anaconda3,miniconda3,miniforge3}/envs/env_isaaclab_232` 자동 탐색 |
| `--isaaclab-dir` | 같은 경로 | `.external/IsaacLab-v2.3.2` |
| `--web-port` | 같은 포트 | 8443 |
| `--lerobot-python` | 같은 경로 | `LEROBOT_PYTHON`이 설정에서 **빠짐** |

현재 값은 `./quest_collector.sh info` 또는 `grep -v KEY .external/quest-collector/session.env`로
확인한다. 네트워크 변경 절차는 [3절](#3-네트워크나-ip가-바뀌었을-때)에 모았다.

인증서 만료 시 새 쌍을 만들고 환경 파일을 갱신한다. 이전 인증서는 보존된다.
IP만 바뀐 경우에는 `--renew-certificate`가 필요 없다.

```bash
./setup_quest_collector.sh \
  --host <PC_WIFI_IP> --renew-certificate --update-config
```

preview가 바뀌어도 수집용 snapshot은 자동으로 변경되지 않는다. 원본 build를
완성한 뒤 명시적으로 갱신한다. 이전 snapshot도 backup 폴더로 보존한다.

```bash
./setup_quest_collector.sh \
  --host <PC_WIFI_IP> --refresh-web \
  --browser-build /absolute/path/to/simple/build
```

## 3. 네트워크나 IP가 바뀌었을 때

Wi-Fi·공유기를 바꾸거나 공유기가 새 주소를 배정하면 PC IP가 달라진다. 수집기 설정과
TLS 인증서는 setup 때의 IP에 묶여 있으므로 아래를 **한 번** 진행한다. SDK 다운로드,
웹 client build, Isaac 설치는 다시 하지 않는다.

> **알아채는 방법:** `./quest_collector.sh info`가
> `[WARN] Configured CLOUDXR_HOST ... is not assigned to this PC`를 표시하거나
> `check`·`runtime`·`web`이 같은 내용의 `[ERROR]`로 멈춘다. 메시지에 현재 IPv4 목록과
> 현재 설정값을 채운 갱신 명령이 함께 나온다.

### 3-1. 새 IP 확인

```bash
ip -4 -brief address
```

Quest가 연결될 네트워크의 인터페이스를 고른다. 예를 들어
`wlp3s0  UP  172.30.1.14/24`라면 새 IP는 `172.30.1.14`다.

### 3-2. 설정 갱신과 점검

```bash
./setup_quest_collector.sh \
  --host <NEW_PC_IP> \
  --isaaclab-python /absolute/path/to/env_isaaclab_232/bin/python \
  --update-config
./quest_collector.sh check
```

- 새 IP용 30일 인증서를 `certs/<NEW_PC_IP>/`에 만든다. `--renew-certificate`는 필요 없다.
- 이전 `session.env`는 `session.env.backup-*`로, 이전 인증서는 원래 폴더에 그대로 남는다.
- 예전 네트워크로 돌아가면 같은 명령을 예전 IP로 실행한다. 유효기간이 하루 넘게 남은
  기존 인증서를 재사용하므로 Quest에서 다시 신뢰할 필요가 없을 수 있다. 하루 미만이면
  `--renew-certificate`를 추가하라는 오류가 나온다.
- [2절](#2-생성-파일과-재설정-규칙)의 표처럼 처음 지정한 사용자 정의 인자는 함께 전달한다.

### 3-3. Quest에서 다시 신뢰하고 연결

1. Quest를 PC와 **같은 Wi-Fi**에 연결한다.
2. [5절](#5-매번-실행-터미널-3개)의 터미널 1·2로 `runtime`과 `web`을 실행한다.
3. Quest 브라우저에서 `https://<NEW_PC_IP>:49100`을 열어 새 인증서의 IP·지문을 확인하고
   신뢰한다. 확인 뒤 빈 화면이나 HTTP 오류가 나오는 것은 정상이다.
4. `https://<NEW_PC_IP>:8443`을 열어 같은 방식으로 신뢰한다.
5. **Server IP 입력칸을 새 IP로 바꾼다.** 이전 값이 남아 있을 수 있다. Port는 49100이다.
6. `CONNECT` 후 Runtime 터미널의 `[CLIENT] Connected`를 확인하고 `collect`를 실행한다.

### 3-4. 새 네트워크에서 확인할 것

`check`는 Quest가 PC에 실제로 닿는지 검사하지 않는다. 아래는 사용자가 확인한다.

| 항목 | 확인 내용 |
|---|---|
| 같은 네트워크 | Quest와 PC의 IP 앞자리 대역이 같아야 한다(예: 둘 다 `172.30.1.x`) |
| 기기 간 통신 | 게스트망·AP(클라이언트) 격리가 켜진 망에서는 8443 페이지부터 열리지 않는다. 설정 갱신으로는 해결되지 않으므로 다른 망을 사용한다 |
| 신뢰하는 망 | Runtime 신호 포트는 모든 인터페이스에 열릴 수 있다. 공용 Wi-Fi에서는 사용하지 않는다 |
| PC 연결 방식 | PC 유선 + Quest Wi-Fi를 권장한다. PC도 Wi-Fi이면 끊김·지연이 늘 수 있으니 `[PERF]`와 화질을 확인한다 |
| 반복되는 IP 변경 | 공유기의 DHCP 예약으로 PC IP를 고정하면 매번 3절을 반복하지 않아도 된다 |

## 4. 시작 전 점검

```bash
./quest_collector.sh info
./quest_collector.sh check
```

`check`는 다음을 확인한다. 최초 한 번 C++ 실행기를 컴파일한다.
**서비스나 Isaac Sim은 시작하지 않는다.**

- 설정된 `CLOUDXR_HOST`가 현재 이 PC의 IP인지(아니면 [3절](#3-네트워크나-ip가-바뀌었을-때) 안내 후 중단)
- 인증서 유효기간과 인증서 IP
- SDK 버전·library loading, Isaac/OpenXR metadata, 웹 파일 존재

Quest가 PC에 닿는지, Quest 영상·실제 tracking·encoding 성능은 검증하지 않는다.
`runtime`·`web`도 시작 전에 같은 IP·인증서 검사를 한다.

여러 네트워크에 연결된 PC는 실행 전 방화벽 정책을 확인한다. Runtime SDK는
`--host`와 무관하게 신호 포트를 모든 인터페이스에 열 수 있다. 허용 대상은
신뢰하는 Quest/LAN으로 제한하고, 연구실 유선망이나 인터넷에 서비스를 공개하지 않는다.
TLS는 암호화이지 사용자 인증이 아니다. 도구는 방화벽 전체 해제나 포트 포워딩을 하지 않는다.

## 5. 매번 실행: 터미널 3개

각 터미널에서 저장소 루트로 이동한다. `quest_collector.sh`가 환경 파일을 자동으로
읽으므로 매번 conda activate나 `source`를 반복할 필요는 없다.

### 터미널 1 — Runtime

```bash
./quest_collector.sh runtime
```

`[READY] CloudXR wss://<PC IP>:49100; media UDP 47998`를 확인하고 유지한다.

### 터미널 2 — HTTPS 웹페이지

```bash
./quest_collector.sh web
```

`[READY] Quest browser: https://<PC IP>:8443`를 확인하고 유지한다. preview의
HTTP 8080과 겹치지 않는다. `--web-port`로 setup한 경우에는 해당 포트를 사용한다.

### Quest — 인증서 확인과 CONNECT

1. Quest를 PC와 통신 가능한 같은 신뢰 LAN에 연결한다.
2. `./quest_collector.sh info`에 나온 HTTPS 페이지를 연다.
3. 직접 만든 인증서의 IP·지문을 확인하고 신뢰 여부를 사용자가 결정한다.
4. `https://<PC IP>:49100`도 방문해 Runtime 인증서를 확인한다. 일반 웹페이지가
   아니므로 인증서 확인 뒤 HTTP 오류나 빈 화면이 나올 수 있다.
5. 다시 8443 페이지에서 아래 값을 선택하고 `CONNECT`한다.

| 항목 | 값 |
|---|---|
| Server Backend | Manual Input IP:Port |
| Server IP | setup에서 지정한 PC LAN IP |
| Port | 49100 |
| Immersive Mode | VR Immersive |
| Video Codec | 첫 점검은 H.264 |
| Device Frame Rate | 72 FPS |
| Per-eye Width/Height | 처음 진단할 때 1024/1024; 정상 사용값이 있으면 유지 |
| Proxy URL / Media Address / Media Port | 기본은 비워 둠 |

Runtime 터미널의 `[CLIENT] Connected`를 확인한 다음 수집기를 실행한다.
native WSS를 사용하므로 별도 48322 프록시는 필요 없다. `8765`와
`Local Kuavo IsaacLab`은 preview 전용이며 수집 설정이 아니다.

인증서 지문은 다음으로 확인한다(개인 키를 출력하지 않음).

```bash
source .external/quest-collector/session.env
openssl x509 -in "$CLOUDXR_CERTIFICATE" -noout -dates -fingerprint -sha256
```

### 터미널 3 — 수집기

```bash
./quest_collector.sh collect
```

이 명령부터 실제 Isaac Sim이 실행된다. 기본은 S63 + leju-twofinger, controllers, CPU physics/IK,
30Hz 제어 설정, compact scene, XR 배율 1.0, 양쪽 wrist RGB, head/depth OFF, HDF5다.
기본 초기 상태는 `configs/initial_states.json`의 `s63_leju_vr_collect_01`이다. 이 preset은
양팔 14/14.5°, elbow -37°, wrist 22.5/23°, head yaw/pitch 1/3°와
`waist_yaw_link`의 x/y/z 0/0/1.30 m, pitch/yaw 0°를 저장한다. reset에도 같은 상태가
다시 적용되며 HDF5 episode metadata에 preset 이름과 전체 값이 기록된다.
PC desktop observer는 OFF이고 camera annotator에 필요한 최소 160×90 render만 유지한다.
별도 PC camera preview도 OFF다. head RGB가 데이터셋에 필요할 때만 `--head-camera`를
추가한다. 최초 EULA가 나타나면
사용자가 직접 확인한다. `30Hz`는 설정값이며 실제 속도는 `[PERF]`로 판단한다.

`[XR] OpenXR session and display are active.`와 양팔/head tracking을 확인한다.
preview 시뮬레이터와 동시에 돌리면 GPU/입력 혼동이 생길 수 있으므로 첫 검증은
수집기만 실행한다. 이 실행기는 다른 앱을 자동 종료하지 않는다.

### V2 grasp SAC 시연 수집

Runtime·웹 서버·Quest CONNECT는 위 절차와 같다. 터미널 3의 기본 수집 명령을
아래 명령으로 바꾸면 SAC의 staged-grasp 환경에서 controller 입력을 RL action으로
변환해 전이를 기록한다. 사용할 GPU 번호와 출력 파일명을 정한다.

```bash
CUDA_VISIBLE_DEVICES=0 ./quest_collector.sh collect \
  --robot-model s63 --gripper leju-twofinger \
  --rl-reward-debug 2 --device cuda:0 \
  --no-rl-demo-self-collision \
  --rl-demo-dataset datasets/v2_grasp_quest_001.hdf5
```

예시는 self-collision을 끈 학습에 맞춘다. 켠 학습이면 해당 `--no-` 옵션을 뺀다.
`X/C` 보정, `A/T` 실행·일시정지, `B/R` 시도 중단·새 장면 reset이다.
성공·안전 위반·timeout 시 terminal 전이를 저장하고 일시정지한다.
파일은 기존 경로를 덮어쓰지 않으며, `--max-episodes`로 종료할 시도 수를
정할 수 있다. 기존 수집 형식은 옵션 없는 `collect`로 유지된다.
정확한 HDF5 필드와 SAC loader 미연결 범위는
[Quest RL 시연 수집](RL_QUEST_REWARD_DEBUG.md)을 참고한다.

다른 저장 자세는 이름으로 선택한다. wrapper 기본 옵션보다 사용자가 뒤에 적은 값이 우선한다.

```bash
./quest_collector.sh collect --initial-state s63_leju_ready_01
./quest_collector.sh collect \
  --initial-state custom_pose \
  --initial-states-file /absolute/path/to/initial_states.json
```

각 preset은 `robot_model`과 `gripper`가 현재 선택과 일치해야 한다. RL reward debug는 자체 RL
초기 상태를 사용하므로 이 두 옵션을 받지 않는다.

### 시작할 때 다른 scene 설정 선택

`./quest_collector.sh collect --scene-config /absolute/path/to/scene_config.py`로
`configure(cfg)`를 제공하는 **신뢰하는 로컬 Python 설정 파일**을 선택할 수 있다.
설정은 환경 생성/물리 초기화 전에 적용된다. 생략하면 기존 scene을 그대로 사용한다.
추가 rigid object의 pose도 기록하려면 설정 파일에 `RECORDING_OBJECTS = ("asset_name",)`을 둔다.
작업/객체 수가 바뀌면 기존 LeRobot root를 재사용하지 않는다.

이 PC의 임시 큐브 촬영 설정은 `bash artifacts/temporary/run_tiny_cube_vr.sh`로 실행한다.
세부 사용법은 로컬 `artifacts/temporary/TINY_CUBE_VR.md`에 있다. `artifacts/`는 Git에 포함되지 않는다.
기존 작업 박스 대신 큐브 하나를 처음부터 생성하며, 기본 scene의 박스 배치는 변경하지 않는다.
**수집기 실행 중에는 Pause 상태라도 물리 prim 삭제/추가, layer 교체, USD Open/Import를 하지 않는다.**
physics tensor view가 무효화되면 Play 재개로 복구되지 않으므로 수집기를 재실행한다.

## 6. 조작·저장·종료

`X` 보정 → `A`로 따라오기 확인 → `B`로 녹화 → 작업 완료 후 PC `M`으로 성공 저장.
PC 없이 진행하려면 작업 완료 후 오른쪽 그립을 `--success-hold-seconds`(기본 1.5초) 동안
꾹 눌러도 `M`과 동일하게 성공 저장되고 다음 시도를 받을 준비 상태가 된다(`--hand-switch` 실행 중에는 비활성).
PC 키는 Isaac Sim 창에 포커스를 둔다. `B` 중지는 실패, `R`은 실패 종료 후 reset이다.
녹화 중 `A`/`X`는 현재 시도를 종료하므로 단순 일시정지로 사용하지 않는다.
자세한 mapping은 [Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)를 따른다.

HDF5는 `datasets/kuavo_quest_<timestamp>_<id>.hdf5`에 시도별로 저장한다. 경로는
`[DATA] New HDF5 file`에 표시되며 실패 데이터도 남을 수 있다. 녹화를 끝내고
`success`, `end_reason`, `num_samples`를 확인한다.

종료는 **녹화 종료 → 수집기 Ctrl+C 및 저장 정리 대기 → Quest disconnect →
Runtime Ctrl+C → 웹 서버 Ctrl+C** 순서다. 서버는 foreground로만 실행되며
자동 시작 서비스/백그라운드 daemon을 등록하지 않는다.

## 7. LeRobot과 옵션 변경

별도 v3 writer 환경은 준비 단계에서 지정한다. Isaac 환경에 LeRobot을 설치하지 않는다.

```bash
./setup_quest_collector.sh --host <PC_WIFI_IP> \
  --isaaclab-python /absolute/path/to/env_isaaclab_232/bin/python \
  --lerobot-python /absolute/path/to/lerobot-v3/bin/python --update-config
```

이후 IP 변경 등으로 `--update-config`를 다시 실행할 때도 `--lerobot-python`을 함께 전달한다.

HDF5와 LeRobot 동시 수집:

```bash
./quest_collector.sh collect \
  --dataset-format both \
  --lerobot-root datasets/quest_session_001 \
  --lerobot-repo-id local/kuavo_quest_teleop
```

실패 episode까지 LeRobot에 보존하려면 `--lerobot-save-failed`를 추가한다. 카메라
해상도·FPS·로봇 모델·action schema를 바꾸면 새 dataset root를 사용한다. Hub에
자동 업로드하지 않는다.

wrapper 뒤의 옵션은 기본값보다 우선한다.

```bash
./quest_collector.sh collect --no-desktop-render
./quest_collector.sh collect --hand-switch
./quest_collector.sh collect --input-mode hands
```

기존 저수준 `run_cloudxr_runtime.sh`, `run_quest_browser.sh`,
`collect_quest_teleop.sh`도 변경 없이 사용할 수 있다. 직접 사용할 때만
`.external/quest-collector/session.env`를 source한다.

## 8. 문제 해결

| 증상 | 확인 |
|---|---|
| Missing config | setup을 먼저 실행했는지 확인 |
| `CLOUDXR_HOST ... is not assigned to this PC` | 네트워크·IP 변경. [3절](#3-네트워크나-ip가-바뀌었을-때) 진행 |
| `[WARN] Could not verify ...` | IP 소유 여부를 검사하지 못함(권한 제한 등). `ip -4 -brief address`와 `info`의 IP가 같은지 직접 확인 |
| SHA-256 mismatch | 6.2.1 공식 Linux SDK인지 확인; 파일을 지우거나 검사를 우회하지 않음 |
| 다른 Python/OpenXR 경로 | `--isaaclab-python`, `--isaaclab-dir` 지정 |
| 인증서 만료 | `--renew-certificate --update-config`(처음 쓴 사용자 정의 인자 포함), Quest에서 새 인증서 확인 |
| 새 IP로 갱신 후 Quest에서 인증서 경고 | 정상. 3-3의 순서로 49100과 8443을 다시 신뢰 |
| 8443 페이지 접속 실패 | `https`, PC LAN IP, web 실행, 같은 Wi-Fi, 게스트망·AP 격리 확인([3-4](#3-4-새-네트워크에서-확인할-것)) |
| Address already in use | 해당 포트의 기존 프로세스 확인; 도구는 자동 종료하지 않음 |
| 페이지는 열리지만 CONNECT 실패 | Server IP 입력칸이 새 IP인지, Runtime 49100 인증서, Manual backend, 방화벽 확인 |
| LeRobot 설정이 사라짐 | `--update-config` 때 `--lerobot-python`을 빠뜨림. 다시 포함해 실행 |
| FORM_FACTOR_UNAVAILABLE | Quest CONNECT 이후 collect를 실행했는지 확인 |
| LeRobot 데이터 없음 | v3 writer 설정, 녹화 시작, 성공 처리, 정상 종료 확인 |

공식 구성/라이선스: [Runtime 다운로드](https://docs.nvidia.com/cloudxr-sdk/latest/getting_cloudxr.html),
[Quest 클라이언트 설정](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/client_setup.html).

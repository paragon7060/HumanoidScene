# 실제 Quest 수집기: 처음 준비부터 재실행까지

이 문서는 `collect_quest_teleop.sh`의 **OpenXR/CloudXR 데이터 수집 경로**를 다룬다.
`preview_quest_browser.sh`는 별도 개발용 bridge이며 여기서는 실행하지 않는다.
아래 명령은 모두 clone한 저장소 루트에서 실행한다.

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
백업한 후 갱신하는 `--update-config`를 명시한다. 처음 사용한 Python/LeRobot 등
사용자 정의 인자도 함께 전달한다.

```bash
./setup_quest_collector.sh --host <NEW_PC_IP> --update-config
```

인증서 만료 시 새 쌍을 만들고 환경 파일을 갱신한다. 이전 인증서는 보존된다.

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

## 3. 시작 전 점검

```bash
./quest_collector.sh info
./quest_collector.sh check
```

`check`는 인증서 유효기간/IP, SDK 버전·library loading, Isaac/OpenXR metadata와
웹 파일 존재를 확인한다. 최초 한 번 C++ 실행기를 컴파일한다. **서비스나 Isaac Sim은
시작하지 않는다.** Quest 영상·실제 tracking·encoding 성능을 검증하는 명령은 아니다.

여러 네트워크에 연결된 PC는 실행 전 방화벽 정책을 확인한다. Runtime SDK는
`--host`와 무관하게 신호 포트를 모든 인터페이스에 열 수 있다. 허용 대상은
신뢰하는 Quest/LAN으로 제한하고, 연구실 유선망이나 인터넷에 서비스를 공개하지 않는다.
TLS는 암호화이지 사용자 인증이 아니다. 도구는 방화벽 전체 해제나 포트 포워딩을 하지 않는다.

## 4. 매번 실행: 터미널 3개

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

이 명령부터 실제 Isaac Sim이 실행된다. 기본은 S200062, controllers, CPU physics/IK,
30Hz 제어 설정, compact scene, XR 배율 1.0, head/wrist RGB, depth OFF, HDF5다.
PC desktop viewport는 ON, 별도 PC camera preview는 OFF다. 최초 EULA가 나타나면
사용자가 직접 확인한다. `30Hz`는 설정값이며 실제 속도는 `[PERF]`로 판단한다.

`[XR] OpenXR session and display are active.`와 양팔/head tracking을 확인한다.
preview 시뮬레이터와 동시에 돌리면 GPU/입력 혼동이 생길 수 있으므로 첫 검증은
수집기만 실행한다. 이 실행기는 다른 앱을 자동 종료하지 않는다.

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

## 5. 조작·저장·종료

`X` 보정 → `A`로 따라오기 확인 → `B`로 녹화 → 작업 완료 후 PC `M`으로 성공 저장.
PC 키는 Isaac Sim 창에 포커스를 둔다. `B` 중지는 실패, `R`은 실패 종료 후 reset이다.
녹화 중 `A`/`X`는 현재 시도를 종료하므로 단순 일시정지로 사용하지 않는다.
자세한 mapping은 [Quest 상세 가이드](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)를 따른다.

HDF5는 `datasets/kuavo_quest_<timestamp>_<id>.hdf5`에 시도별로 저장한다. 경로는
`[DATA] New HDF5 file`에 표시되며 실패 데이터도 남을 수 있다. 녹화를 끝내고
`success`, `end_reason`, `num_samples`를 확인한다.

종료는 **녹화 종료 → 수집기 Ctrl+C 및 저장 정리 대기 → Quest disconnect →
Runtime Ctrl+C → 웹 서버 Ctrl+C** 순서다. 서버는 foreground로만 실행되며
자동 시작 서비스/백그라운드 daemon을 등록하지 않는다.

## 6. LeRobot과 옵션 변경

별도 v3 writer 환경은 준비 단계에서 지정한다. Isaac 환경에 LeRobot을 설치하지 않는다.

```bash
./setup_quest_collector.sh --host <PC_WIFI_IP> \
  --lerobot-python /absolute/path/to/lerobot-v3/bin/python --update-config
```

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

## 7. 문제 해결

| 증상 | 확인 |
|---|---|
| Missing config | setup을 먼저 실행했는지 확인 |
| SHA-256 mismatch | 6.2.1 공식 Linux SDK인지 확인; 파일을 지우거나 검사를 우회하지 않음 |
| 다른 Python/OpenXR 경로 | `--isaaclab-python`, `--isaaclab-dir` 지정 |
| 인증서 만료/새 IP | `--renew-certificate --update-config`, Quest에서 새 인증서 확인 |
| 8443 페이지 접속 실패 | `https`, PC LAN IP, web 실행, 기기 간 통신 차단 확인 |
| Address already in use | 해당 포트의 기존 프로세스 확인; 도구는 자동 종료하지 않음 |
| 페이지는 열리지만 CONNECT 실패 | Runtime 49100 인증서, Manual backend, 방화벽 확인 |
| FORM_FACTOR_UNAVAILABLE | Quest CONNECT 이후 collect를 실행했는지 확인 |
| LeRobot 데이터 없음 | v3 writer 설정, 녹화 시작, 성공 처리, 정상 종료 확인 |

공식 구성/라이선스: [Runtime 다운로드](https://docs.nvidia.com/cloudxr-sdk/latest/getting_cloudxr.html),
[Quest 클라이언트 설정](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/client_setup.html).

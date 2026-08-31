# 준비된 PC에서 Quest teleop 시작하기

이 저장소는 별도로 설치한 NVIDIA CloudXR Runtime 6.x를 실행하는
`run_cloudxr_runtime.sh`와 빌드된 웹 클라이언트를 제공하는
`run_quest_browser.sh`를 포함한다. 실제 로봇 하드웨어가 아니라 Isaac Lab 속
Kuavo를 조작하는 구성이다. SDK 다운로드부터 필요한 경우
[README](../README.md#quest-collection)를 먼저 따른다.

## 준비 결과와 검증 범위

2026-08-31 구성 기준: Node 22.23.2, CloudXR.js 6.2.0, Runtime 6.2.1/API 1.0.7.
SDK와 Node는 `.external` 아래에만 설치하고 Isaac conda 환경은 변경하지 않았다.
아래가 확인됐다.

- Node 공식 SHA256 확인, 고정된 NVIDIA 샘플 패치 적용, 웹 production build 성공.
- Runtime 라이브러리 로드, WS/WSS 서비스 기동·정상 종료 성공.
- PC HTTPS 페이지와 Runtime TLS 인증서를 직접 검증, OpenXR 파일 사전 점검 통과.
- 브라우저 UI의 Manual backend, VR 모드, H.264 선택 확인.
- 실제 OpenXR 앱·디스플레이 세션 활성화, Quest에서 기본 장면 표시 확인.
- 맨손·머리 추적 수신과 head/양손목 센서의 0이 아닌 RGB 프레임 확인.
- 수집기 정상 종료 뒤 별도 HDF5 파일이 읽히는 것 확인(점검 세션은 0 episode).
- 실제 X 버튼으로 보정 실행, Y로 패널 표시/숨김, B 버튼 이벤트 수신 확인.

**추가 실기 검증이 필요한 것:** 새 손목 패널의 Quest 내 영상 표시, A로 팔 따라오기,
B로 실제 녹화 시작·종료, 작업을 끝까지 수행한 수집 데이터, 장시간 스트리밍 성능.
현재 NVIDIA 문서는 Ada/Blackwell GPU를 명시하므로 기동 성공을 RTX 3060의
공식 지원으로 해석하지 않는다.
[공식 요구사항](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/runtime_req.html)을 확인한다.

## 터미널 1: Runtime

모든 터미널은 저장소 루트에서 시작한다. 이 PC에 생성한
`.external/quest-session.env`에는 실제 설치 경로·LAN IP·인증서 경로가 들어 있다.
이 파일과 인증서/개인 키는 Git에 포함하지 않는다. 다른 PC에서는
[.env.example](../.env.example)의 변수를 실제 경로로 지정한다.

```bash
source .external/quest-session.env
./run_cloudxr_runtime.sh --check
./quest_doctor.sh --require-runtime
./run_cloudxr_runtime.sh \
  --host "$CLOUDXR_HOST" \
  --certificate "$CLOUDXR_CERTIFICATE" \
  --key "$CLOUDXR_KEY"
```

`[READY] CloudXR wss://<PC IP>:49100; media UDP 47998`가 나오면 켜 둔다.
실행기는 C++17 컴파일러로 최초 실행 시 빌드하며 SDK의 `cxrServiceAPI.h`와
`libcloudxr.so`가 필요하다. SDK 위치가 다르면 `CLOUDXR_RUNTIME_DIR`을 설정한다.
이 서비스는 웹 클라이언트용 `auto-webrtc`를 사용한다.

주의: Runtime 6.2.1은 `endpoint-ip`를 설정해도 신호 포트가 `0.0.0.0:49100`에
열리는 것이 확인됐다. `--host`를 접근 제한으로 사용하지 않는다. 여러 네트워크에
연결된 PC라면 방화벽으로 허용 범위를 제한한 뒤 사용한다. 준비 과정에서는
방화벽이나 라우터 설정을 변경하지 않았다.

진단만 할 때는 `--seconds 10`을 추가하면 기동 후 10초 뒤 정상 종료한다.
종료는 `Ctrl+C`로 한다. 로그는 `artifacts/cloudxr-logs/`에 있다.
API 오류가 나면 실행기는 0이 아닌 종료 코드를 반환한다.

## 터미널 2: HTTPS 웹페이지

```bash
source .external/quest-session.env
./run_quest_browser.sh \
  --host "$CLOUDXR_HOST" \
  --certificate "$CLOUDXR_CERTIFICATE" \
  --key "$CLOUDXR_KEY"
```

이 명령은 `.external/cloudxr-js-samples/simple/build`의 정적 파일만 제공한다.
webpack 개발 서버를 LAN에 열지 않는다. UI 코드를 수정했다면 먼저 다음을 실행한다.

```bash
source .external/quest-session.env
npm --prefix .external/cloudxr-js-samples/simple run build
```

브라우저를 새로 고쳐 변경을 반영한다. 샘플 개발 의존성의 npm audit에는
현재 4 moderate / 1 high가 보고되며 무작정 `audit fix --force`를 적용하지 않았다.
이 실행 경로에서는 개발 서버를 실행하지 않지만, 개발 의존성 자체가 수정된 것은 아니다.

## Quest에서 직접 할 설정

1. Quest와 PC를 같은 신뢰할 수 있는 LAN에 연결하고 좌우 컨트롤러를 준비한다.
   `--input-mode hands`를 사용할 때만 맨손 추적을 켠다.
2. Quest Browser에서 `https://<CLOUDXR_HOST>:8080`을 연다.
3. 이 PC에서 만든 자체 서명 인증서를 확인하고 신뢰할지 **사용자가 직접** 결정한다.
   연결할 IP가 맞는지 확인한다. 브라우저의 인증서 경고는 자동으로 우회하지 않는다.
4. Runtime의 `https://<CLOUDXR_HOST>:49100`도 방문해 같은 인증서를 확인한다.
   Runtime은 일반 웹 서버가 아니므로 빈 화면이나 HTTP 오류가 나올 수 있다.
5. 8080 페이지로 돌아가 아래 설정을 선택하고 `CONNECT`한다.

`Accept cert`를 눌렀을 때 `about:blank#blocked`가 나타나면 Quest 주소창에
`https://<CLOUDXR_HOST>:49100`을 직접 입력한다. 기존 샘플의 새 탭 링크가
차단되는 경우를 줄이도록 이 저장소의 패치는 인증서 링크를 같은 탭에서 연다.
인증서 주소와 내용을 직접 확인한 뒤, 뒤로 가기로 8080 페이지에 돌아온다.
이 변경은 인증서를 자동 신뢰하거나 브라우저의 보안 검증을 해제하지 않는다.
이미 설치한 샘플은 `./setup_quest_browser.sh --patch-only` 후 production build를
다시 하고 Quest 페이지를 새로 고치면 반영된다.

| 항목 | 값 |
|---|---|
| Server Backend | **Manual Input IP:Port** |
| Server IP | `CLOUDXR_HOST`에 지정한 PC LAN IP |
| Port | **49100** |
| Immersive Mode | **VR Immersive** |
| Video Codec | 우선 **H.264**로 점검; 실제 GPU/헤드셋 인코딩 지원은 별도 검증 |
| Per-Eye Width / Height | 첫 점검은 **1024 / 1024** |
| Device Frame Rate | **72 FPS** |
| Proxy URL / Media Address / Media Port | 비워 두고 기본 ICE 사용 |

`Local Kuavo IsaacLab`/`8765`는 수집용이 아니다. 이 구성은 Runtime 자체 TLS를
사용하므로 별도 WSS 프록시 `48322`도 필요 없다. `Load defaults`를 나중에
선택하면 해상도·코덱 등이 바뀔 수 있으므로 최종 값을 다시 확인한다.
WebXR·손 추적 권한 요청은 헤드셋에서 허용한다.

`[CLIENT] Connected`를 Runtime 터미널에서 확인한 다음 수집기를 실행한다.
`auto-webrtc`의 OpenXR 시스템 정보는 클라이언트 연결 후 준비되므로 이 순서가
중요하다. 클라이언트 연결 자체도 영상/손 추적의 종단 간 검증은 아니다.
[공식 API 설명](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_runtime/runtime_mgmt_api.html#device-profile)을 참고한다.

## 터미널 3: 수집기

```bash
source .external/quest-session.env
./collect_quest_teleop.sh \
  --robot-model s200062 \
  --dataset-format hdf5 \
  --max-episodes 0 \
  --episode-seconds 0 \
  --no-auto-start
```

Isaac Sim 최초 실행 시 NVIDIA EULA가 표시되면 사용자가 내용을 확인하고 직접
동의해야 한다. 준비 과정에서는 `OMNI_KIT_ACCEPT_EULA`를 설정하거나 동의 파일을
만들지 않았다. GPU 구동과 장면 로딩이 끝나면 양쪽 컨트롤러를 인식시켜
`[TRACKING] ... True`를 확인한다. 위 명령은 녹화를 자동 시작하지 않는다.

수집기는 장면 생성 후 지정된 `XR_RUNTIME_JSON`을 Kit의 Custom Runtime으로
선택하고 XR 출력 세션을 시작한다. `[XR] OpenXR session and display are active.`와
Runtime 터미널의 `[OPENXR] App connected`가 나와야 실제 앱 연결 단계가 완료된다.
XR 확장 로딩 또는 `[CLIENT] Connected`만으로는 이 단계가 완료된 것이 아니다.
출력이 활성화되지 않으면 수집기는 오류를 내므로, 종료 로그에서 런타임 경로와
OpenXR 오류를 확인한다. 최초 실행의 셰이더·재질 준비 중에는 수 분이 걸릴 수 있다.

정면을 보고 Quest **X**(PC **C**)로 시점을 로봇 head camera에 맞춘다.
**A**(**T**)로 저장 없이 따라오기 시작/멈춤, **B**(**P**)로 녹화 시작/멈춤,
**Y**(**H**)로 양쪽 위 작은 손목 패널 표시/숨김을 제어한다.
중앙은 기본 stereo 장면이며 머리 RGB 패널이 가리지 않는다. 머리 영상은 계속 기록된다.
녹화 중 X는 현재 파일을 `operator_calibration`으로 닫고 보정하며, 녹화 중 따라오기를 멈추면 현재 시도도 실패로 종료된다.

버튼 수신은 `[BUTTON]`, 실제 녹화는 `[DATA] Recording ...`로 구분해서 확인한다.
기본 `--input-mode controllers`는 컨트롤러 위치로 팔을 움직이고, 각 검지
트리거를 절반 이상 당기면 해당 gripper를 닫는다. 컨트롤러를 계속 들고 조작한다.
맨손을 원할 때만 `--input-mode hands`로 다시 실행한다. 현재 모드는 `[INFO] Arm input mode`
및 `[TRACKING] ... input=...`에서 확인한다. 머리만 추적되면 팔은 움직이지 않는다.
[보정·조작 절차](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)를 참고한다.

팔이 움직이지 않거나 너무 작게 움직이면 `A`로 `FOLLOW ON`을 만든 뒤 `[MOTION]`을
확인한다. 컨트롤러 인식, 입력 명령, 실제 손끝 이동은 서로 다른 확인 항목이다.
현재 제어는 목표를 프레임마다 버리지 않고 유지한다. 손이 멈추거나 잠깐 추적을
잃어도 마지막 목표를 따라가며, A/B 정지 또는 X 보정 시 현재 자세에 멈춘다.
실제 관절·충돌·중력 때문에 목표 오차가 남을 수 있으므로 화면을 보며 조정한다.

패널이 검으면 `[CAMERA] Left/Right wrist RGB`의 `max`가 0인지 먼저 확인한다.
센서 영상이 정상인데 패널만 검다면 XR UI 경로 문제다. 현재 위젯은 Frame 안에
이미지 레이아웃을 직접 배치하며, 첫 유효 프레임 전의 검은 자리표시는 숨긴다.
손목 패널은 눈앞 0.35 m에 표시한다. 더 조절하려면 `--xr-overlay-distance`를 쓰되
near plane인 0.08 m보다는 크게 둔다.

HDF5는 **녹화 시도마다** 날짜·시간·고유값을 넣은 별도 파일을 만든다. 정확한 경로는
`[DATA] New HDF5 file: ...`에서 확인한다. `--dataset`은 첫 파일 이름이며 이후에는
같은 폴더의 새 고유 파일을 쓴다. 기존 파일을 덮어쓰거나 이어 쓰지 않는다.
기본 `--max-episodes 0 --episode-seconds 0 --no-auto-start`는 앱을 유지하고 수동으로
녹화한다. A/B로 시도를 끝낸 뒤 다시 B로 새 파일에 수집할 수 있다. R만 장면을 초기화한다.
`--max-episodes 1`을 쓰면 시도 종료 후 앱까지 정상 종료한다. 오류로 꺼진 것과 구분한다.
오류가 난 기존 파일도 자동 삭제하거나 복구 명목으로 수정하지 않는다.
실행 중에는 `Ctrl+C`로 수집기를 종료해 HDF5 정리가 완료된 뒤 창을 닫는다.

PC Isaac Sim 창에서 `M`은 작업자 판정 성공, `P`는 시작/중지, `R`은 reset이다.
HDF5는 샘플이 있는 실패 시도도 보존한다.
[첫 파일 확인](../README.md#quest-first-recording) 후 LeRobot을 연결한다.

## 인증서와 종료

준비 과정에서 생성한 인증서는 30일 유효하며 PC의 현재 LAN IP와 localhost용이다.
다음으로 유효기간과 지문을 확인한다. IP가 바뀌거나 만료되면 인증서를 새로 만들고
환경 파일의 주소를 갱신한 뒤, 사용자가 Quest에서 다시 확인해야 한다.

```bash
source .external/quest-session.env
openssl x509 -in "$CLOUDXR_CERTIFICATE" -noout -dates -fingerprint -sha256
```

각 서버 터미널에서 `Ctrl+C`를 눌러 종료한다. 인증서 파일이 암호화 통신을
제공해도 사용자 인증을 제공하지는 않는다. 서비스는 신뢰하는 LAN에서만 사용하고,
라우터 포트 포워딩이나 방화벽 전체 해제는 하지 않는다.

준비 과정에서 이미 실행 중인 서버에는 중복 실행하지 않는다.
`.external/quest-ready-state.json`에 당시 서버 PID와 접속 URL을 기록했다.
재시작 전 `ss -ltnp 'sport = :8080 or sport = :49100'`으로 현재 프로세스를 확인한다.

### 현재 조작·성능 기본값

왼쪽 스틱은 베이스 전후/좌우 이동, 오른쪽 스틱은 베이스 yaw(최대 1.2rad/s)/몸통 높이 조절이다.
A/B로 따라오기를 켠 뒤 사용한다. 베이스는 simulation fixed-root 이동·회전이며 바퀴 회전을 동기화한다. 접촉력 기반 주행은 아니다.
시점은 robot head에 붙으며 왼쪽 아래 그립을 누르는 동안 room-scale 자유 시점이다.
자유 시점에서는 새 팔·몸통 명령을 멈추고 기존 팔 목표를 유지한다.
양쪽 위 검지 트리거는 놓으면 open, 당기면 close이며 생성·R 리셋도 open이다.
기본은 absolute 위치 매핑, depth OFF, 추가 PC 렌더는 센서 사용 시에만 160×90로 유지, CPU 물리/IK와 GPU RTX 렌더다.
`--arm-orientation-weight 0.5`가 기본이며 검지 pointing/엄지 축을 그리퍼 접근/닫힘 축에 대응시킨다.
머리는 보정된 HMD 로컬 기준으로 좌우/상하를 계산한다. 팔 제어·질량 보완 내용은
[조작·물리 설정](QUEST3_KUAVO_TELEOP_GUIDE.md#4-조작-및-episode-제어)을 참고한다.
`--scene-detail compact`로 불필요한 배경 props와 legacy bodies를 제거하되 재질은 유지한다.
원래 배경과 비교하려면 `--scene-detail full`을 사용한다.
`--control-hz 60`과 실제 wall-clock 60Hz는 다르므로 `[PERF]`로 확인한다.
VR 렌더 배율은 1.0으로 유지하며 RTX 3060 대기 상태 실측은 약 15~16Hz(녹화 OFF)다.
측정 후 Quest 외부 연결이 없는 것이 확인됐으므로 실제 착용·스트리밍 속도로 해석하지 않는다.
60Hz를 보장하는 설정이 아니다. 고정 컨베이어는 collider를 유지하되 속도 reset 대상에서
제외해 `Body must be non-kinematic` 오류를 방지한다.
프로파일: `--profile-steps 120`; XR PNG 캡처: `--capture-xr`.

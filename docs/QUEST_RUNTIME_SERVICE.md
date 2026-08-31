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

**아직 확인되지 않은 것:** 실제 Quest 영상·손 추적, Isaac Lab 수집기와 Runtime의
실기 연결, HDF5 저장, RTX 3060의 CloudXR 인코딩 성능·호환성.
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

1. Quest와 PC를 같은 신뢰할 수 있는 LAN에 연결하고 손 추적을 켠다.
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
  --dataset datasets/quest_first_session.hdf5 \
  --max-episodes 1 \
  --episode-seconds 60
```

Isaac Sim 최초 실행 시 NVIDIA EULA가 표시되면 사용자가 내용을 확인하고 직접
동의해야 한다. 준비 과정에서는 `OMNI_KIT_ACCEPT_EULA`를 설정하거나 동의 파일을
만들지 않았다. GPU 구동과 장면 로딩이 끝나면 양손을 인식시켜
`[TRACKING] ... True`와 `[DATA] Recording ...`를 확인한다.

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

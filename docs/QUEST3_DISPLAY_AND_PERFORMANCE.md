# Meta Quest 화면과 성능 설정

이 문서는 Quest 사용자가 보는 XR 화면, dataset camera, PC 관찰자 화면을 구분하고
성능 우선 설정을 선택하는 방법을 설명한다.

## 화면 종류

| 화면 | 용도 | 기본 상태 |
|---|---|---|
| Quest stereo scene | 착용자가 주변을 보고 조작 | ON |
| Quest camera panels | 좌우 wrist camera 확인 | ON |
| Head camera RGB | 선택적 dataset observation | 기본 OFF, `--head-camera`로 ON |
| PC desktop viewport | 옆 사람이 scene 관찰 | OFF |
| PC camera preview 3개 | head/left/right camera 개별 확인 | OFF |

Quest 중앙 화면은 일반 stereo scene이다. 실제 Kuavo head camera의 단안 RGB와
동일하지 않다. Head camera와 좌우 wrist camera는 dataset에 저장할 수 있지만,
Quest camera overlay에는 중앙을 비우고 좌우 wrist panel 두 개만 표시한다. 기본
수집은 성능을 위해 head camera sensor를 생성하거나 저장하지 않는다. head RGB가
정말 필요한 dataset에서만 `--head-camera`를 추가한다.

## 권장 preset

### 1. 데이터 수집 성능 우선

기본값을 그대로 사용한다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --no-desktop-render \
  --no-camera-preview \
  --render-quality performance \
  --scene-detail compact
```

이 모드는 유용한 크기의 PC 관찰 화면을 렌더링하지 않는다. 다만 Kit 107.3에서
desktop viewport update를 완전히 끄면 RTX camera annotator가 빈 frame을 반환할
수 있어, 수집기 내부에서는 필요한 동안 작은 160×90 render를 유지할 수 있다.

### 2. 한 명의 관찰자 화면

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --desktop-render \
  --no-camera-preview \
  --render-quality performance \
  --scene-detail compact
```

Isaac Sim 메인 viewport 하나를 작업 공간 전체가 보이는 고정 3인칭 시점으로 둔다.
새 observer camera나 viewport를 추가하지 않고 기존 viewport를 재사용한다.

같은 viewport의 시점을 VR camera에서 고정 camera로 바꾸는 것 자체는 별도
viewport를 하나 더 만드는 것보다 부담이 작다. 그러나 기본 160×90 보조 render보다
큰 desktop viewport를 계속 갱신하므로 데이터 수집 성능은 일부 낮아질 수 있다.

### 3. 카메라 설치 상태 점검

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --camera-preview \
  --no-desktop-render
```

`--camera-preview`는 활성화된 camera sensor를 PC의 작은 창으로 표시한다. 기본은
left/right wrist만 표시하며, head도 확인하려면 `--head-camera`를 함께 넣는다. 센서
방향이나 영상 유효성을 확인할 때만 사용하고 장시간 수집에서는 끄는 것을 권장한다.

## 성능 비용의 일반적인 순서

GPU와 scene에 따라 수치는 달라지지만 대체로 다음 순서로 비용이 증가한다.

1. 이미 렌더된 texture를 단순 mirror/blit
2. 기존 desktop viewport 하나를 낮은 해상도로 갱신
3. head와 양 wrist camera preview를 동시에 표시
4. 고해상도 observer camera와 render product를 별도로 추가

XR 자체가 양안을 렌더하고 dataset용 RTX camera도 사용하므로, 별도 viewport와
render product를 늘릴수록 GPU 시간과 VRAM 사용량이 커진다.

## 실행 중 판단 기준

- 수집기 `[PERF]` 로그의 wall-clock Hz를 비교한다.
- Quest 영상의 지연, 끊김, dropped frame을 함께 확인한다.
- 목표 `--control-hz 60`은 설정값이며 실제 60Hz를 보장하지 않는다.
- observer를 켠 뒤 control rate가 의미 있게 낮아지면 viewport 크기를 줄이거나
  `--no-desktop-render`로 되돌린다.
- Slack, 브라우저 같은 다른 GPU 앱이 종료되면 VRAM 압력을 먼저 의심한다.

간단한 비교는 같은 scene과 같은 Quest 연결 상태에서 observer OFF/ON을 각각
30초 이상 실행하고 `[PERF]` 중앙값을 비교한다. 외부 Quest가 연결되지 않은 대기
상태의 측정값을 실제 착용·streaming 성능으로 해석하지 않는다.

## VRAM 압력이 있을 때

카메라 해상도를 낮추고 head/depth와 PC preview를 끈다. Head sensor는 기본 OFF다.

```bash
./collect_quest_teleop.sh \
  --xr-runtime-json /absolute/path/to/openxr_cloudxr.json \
  --head-camera-width 320 \
  --head-camera-height 180 \
  --wrist-camera-width 160 \
  --wrist-camera-height 120 \
  --no-record-depth \
  --no-camera-preview
```

위 예시에서 head RGB도 저장하려면 `--head-camera`를 추가한다. `--record-depth`도
`--head-camera`와 함께 사용할 때만 유효하다.

양 wrist sensor 자체가 필요 없다면 `--no-wrist-cameras`를 사용할 수 있다. 현재
이 옵션은 두 wrist panel overlay도 함께 끈다.
`--no-record-wrist-cameras`는 저장만 끄고, `--no-quest-camera-overlay`는 Quest
panel만 끄므로 sensor render 비용까지 제거하는 옵션은 아니다.

`quest_collector.sh collect`는 기본적으로 전체 PC viewport 대신 내부 160×90
최소 viewport를 사용한다. PC 관찰 화면이 꼭 필요할 때만 `--desktop-render`를
명시한다. RL reward debug에서 collider/marker는 각각 `--rl-collision-view`,
`--rl-grasp-markers`를 넣을 때만 생성한다. Reward HUD는 기본 ON이고 10 Hz로
갱신한다. 완전히 끄려면 `--no-rl-reward-hud`를 사용한다.

RL Debug의 grasp/contact/collision **판정**은 marker나 overlay와 별개다. 이 판정은
실제 reward·성공·실패 조건이므로 `--rl-reward-debug`에서 항상 계산하고, 일반 teleop
수집 환경에는 해당 RL contact sensor와 reward manager를 생성하지 않는다. 일반 수집의
`--self-collision`은 로봇 자체 충돌을 막는 별도 안전 장치다.

RTX 3060, CUDA physics, 60 control step(20 step warm-up), OpenXR/CloudXR 미연결 조건의
병목 비교는 다음과 같았다. 절대 Hz는 headset 실행값이 아니며 상대 비용 판단용이다.

| 조건 | ms/control step | 판단 |
|---|---:|---|
| RTX renderer, camera sensor 없음, rollers ON | 371.1 | 동일 renderer 기준 |
| wrist RGB 240×180 두 개, rollers ON | 386.8 | sensor 두 개가 약 15.7 ms 추가 |
| wrist 160×120 또는 15 Hz | 385.2 / 387.4 | 유의미한 개선 없음 |
| wrist DLSS / FXAA | 386.8 / 387.5 | 차이 없음 |
| wrist, shadows ON / OFF | 386.8 / 386.8 | 차이 없음 |
| RL reward core, rollers OFF | 206.5 | headless physics/reward 기준 |
| RL reward core, rollers ON | 443.9 | roller와 contact filter가 지배적 |

따라서 기본 Quest 해상도나 wrist 화질을 먼저 낮추는 것은 권장하지 않는다. 다음
개선 우선순위는 roller rigid-body/joint 수 축소, RL obstacle contact filter 구조 개선,
실제 Quest 연결 상태에서 GPU-native wrist overlay와 UI on-demand redraw 측정이다.

카메라 feature 구성을 바꾸면 기존 LeRobot dataset에 이어 쓰지 말고 새 dataset
root를 사용한다.

## 관련 문서

- [Quest 빠른 시작](QUEST3_QUICKSTART.md)
- [Quest 전체 조작과 dataset](QUEST3_KUAVO_TELEOP_GUIDE.md)
- [CloudXR Runtime 재실행](QUEST_RUNTIME_SERVICE.md)

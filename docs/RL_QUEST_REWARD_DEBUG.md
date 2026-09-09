# Quest로 움직이며 RL reward 확인

기존 Quest 수집기에 `--rl-reward-debug`만 추가한다. CloudXR Runtime, HTTPS 주소,
Quest CONNECT 절차는 [기존 수집기](QUEST_COLLECTOR_SETUP.md)와 같다.
이 옵션이 없으면 기존 수집 동작은 그대로다. **옵션이 있을 때는 데이터 저장이나
정책 학습 대신 현재 flap-pick RL 환경 1개를 수동 조작한다.**

## 실행

저장소 루트에서 기존과 같은 순서로 실행한다. Runtime/web이 이미 실행 중이면
중복 실행하지 않는다. 일반 Isaac 수집기와 이 검사 모드를 동시에 실행하지 않는다.

```bash
# 터미널 1
./quest_collector.sh runtime
# 터미널 2
./quest_collector.sh web
# Quest에서 기존 HTTPS 페이지 → CONNECT 후, 터미널 3
./quest_collector.sh collect --rl-reward-debug
```

기존 head/wrist 영상 패널 옵션도 유지한다. reward 텍스트와 stereo 장면만으로
가볍게 확인하려면 `--no-quest-camera-overlay --no-camera-preview`를 추가한다.
이 경우 RGB 카메라 센서를 생성하지 않는다.

기존에 `collect_quest_teleop.sh`를 직접 사용했다면 기존 명령 끝에
`--rl-reward-debug`를 추가해도 된다. `XR_RUNTIME_JSON` 등 기존 연결 설정은 필요하다.
체크포인트는 필요 없다. 기존 학습을 중단하거나 재시작하지 않지만, GPU 자원이
부족한 PC에서는 학습과 XR을 동시에 실행하지 않는 것이 좋다.

## 조작

| 입력 | 검사 모드 동작 |
|---|---|
| X / PC C | 시점 보정 후 일시정지 |
| A / PC T | 물리·팔 추종 시작/일시정지, 손 위치·회전 기준 재설정 |
| 오른쪽 검지 트리거 | 오른쪽 gripper 닫기; 놓으면 열기. 기본 오른손 실험은 왼쪽 입력 무시 |
| B / PC R | `quest_ready_02`로 초기화하고 정지 (**녹화 버튼 아님**) |
| Y / PC H | reward 패널 표시/숨김 |

정면을 보고 X → 양 컨트롤러를 편한 자세에 두고 A → 손을 움직여 검사한다.
도달 범위가 부족하면 A로 멈추고 손을 편한 위치로 옮긴 뒤 A로 다시 시작한다.
일시정지 동안 물리·task 시간은 멈추고 VR 화면은 계속 렌더링된다.
추적이 끊겨도 정지하며, 복구 후 A를 눌러야 재개한다.

성공·실패·timeout 때 마지막 reward를 고정 표시한다. RL 내부 자동 reset은 유지하므로
장면은 초기 상태로 돌아가지만 패널은 종료 직전 값이다. B로 새 시도를 준비한다.
초기 안정화 동안은 `SETTLING`으로 표시하고 손 입력을 적용하지 않는다.

## VR 표시값

정면에서 약간 아래, 눈앞 28cm의 머리 고정 텍스트 패널에 모든 활성 reward 항목을
약 10Hz로 표시한다. 카메라 패널은 기존처럼 위쪽에 둔다. 표시기는 reward 텍스트를
불투명 RGBA 이미지로 만들고, 기존 카메라 패널과 같은 XR 레이어·GPU 이미지 경로로
전송한다. 일반 `ui.Label`의 생성 여부만으로 VR 표시 성공을 판단하지 않는다.

- 각 항목: `raw × weight × control_dt`, 즉 **실제 해당 스텝의 보상 기여도**.
- `TOTAL`: 해당 스텝의 전체 보상. `RETURN`: 이번 시도의 누적 보상.
- 상승 높이(cm), 유지 시간(s), 좌/우 파지 판정, 좌/우 flap 목표 거리(cm).
- 정지 중에는 마지막 스텝 값이며 새 보상이 발생하지 않는다.

예를 들어 30Hz에서 lift raw=1, weight=5라면 `lift: +0.16667`이다.
성공 보너스 weight=150은 실제 성공 스텝에 `success: +150.00000`으로 표시한다.
TensorBoard `Episode_Reward/*`의 에피소드 정규화 값과 단위가 다르다.
0.1초보다 짧은 비종료 접촉은 화면 갱신 사이에 지나갈 수 있다. 이 모드는 전체
시계열 저장기가 아니며, 종료 보너스는 별도로 마지막 값을 유지한다.

접근 → 접촉 → 파지 → 들어 올리기 → 유지 순으로 `reaching`, `flap_contact`,
`lift`, `holding`, `stable_grasp`가 변하는지 본다. 밀기·충돌 시에는
`prelift_disturbance`, `collision`도 함께 확인한다. 보상은 양손에 각각 독립적으로
배분되는 것이 아니라 task 전체에 계산되며, 현재 기본 성공 조건은 **오른손 한 손 파지**다.

### 들어 올렸는데 성공하지 않을 때

패널 **맨 위**의 `CHECK`는 아직 만족하지 않은 성공 전제조건이다. `FAIL`에는
실제 종료 원인(장애물 충돌·낙하·영역 이탈·안착 실패 등)을 별도로 표시한다.

| 표시 | 의미 |
|---|---|
| `grasp` | 지정 오른손 flap 파지가 인정되지 않음 |
| `height` | 안착 기준 상승 높이가 6cm를 넘지 못함 |
| `tilt` | 기울기가 40° 기준을 넘음 |
| `speed` | 박스 선속도 0.08m/s 또는 각속도 0.35rad/s 기준을 넘음 |
| `contact` | 파지 손가락의 두 허용 flap 외 잔여 접촉력이 10N 기준을 넘음 |

`pass - keep holding`이면 위 전제조건은 통과했다. 준비 완료 후 0.5초 연속 유지가
필요하며, 별도의 장애물 충돌·안착 실패 등 종료 조건이 성공보다 우선한다.

`R flap_right`와 `R flap_left` 각각의 `raw`는 현재 샘플의 최초 파지 조건, `held`는 제한된
접촉 유예까지 적용한 유지 판정이다. `region=11`은 두 손가락 접촉점이 모두 허용
영역 안에 있다는 뜻이고, `opp=1`은 양면 배치, `F=a/b N`은 해당 flap 접촉력이다.
`gap=...s`는 **접촉 누락 시간**이며 손가락 간격이 아니다. 유지 중 `raw=0, held=1`은
작은 미끄러짐·힘 변화·짧은 누락 때문에 발생할 수 있다. `slip`은 파지 기준 대비
손가락 중점 이동, `open+`는 손가락 간격 증가량(mm)이다. 최초 파지 전에는 둘 다 0이다.

물리가 진행 중이면 상승 여부와 무관하게 터미널에도 `[RL GRASP] blocked=...`
진단을 초당 한 번, 종료 스텝에는 즉시 출력한다. 문제가 계속되면 이 줄을 전달하면 된다.
disturbance는 초기 선반 위치·자세 비용과 운동 비용을 분리했다. 1cm 이상 상승하면
전자만 해제되며, 파지 없이 움직이는 박스에는 운동 비용이 남는다. 상세 식과 유지
판정 허용치는 [flap-pick 정의](RL_FLAP_PICK.md)에 있다.

파지 유지 수정 후에는 검사 프로세스를 재시작한다. 이 수정은 학습과 검사 모드가
공유하는 RL 판정이다. 후보 ID observation과 기본 8-action 계약도 변경되어, 수정 전 checkpoint를
직접 재개하는 대신 새 학습을 시작해야 한다.

## 기존 수집기와 동일한 부분 / RL에 맞추는 부분

- 동일: OpenXR 장치·연결, scaled controller 위치/회전 매핑, position gain,
  URDF IK·arm response·orientation weight, head 위치 기준 시점, render quality,
  XR resolution scale, desktop render 옵션.
- RL 유지: `configs/rl_pick_arms_only.py`, named initial pose, 활성 팔 action 순서(기본 오른팔 7+1)와
  증분 크기, actuator·마찰, 관측·reward·충돌·성공/실패 판정과 제어 주기.
  입력만 정책 대신 Quest/IK에서 만든다. 수집기의 별도 관절 구동값·중력 보상·
  self-collision guard를 추가하지 않는다. 따라서 손의 응답은 일반 수집기와 다를 수 있다.
- base·waist·head는 RL대로 고정하며 stick 이동은 사용하지 않는다. HMD를 돌려
  둘러보는 것은 가능하지만 로봇 head joint를 움직이지는 않는다.
- 기본 `active_arm="right"`에서는 왼팔·왼손도 고정된다. 왼쪽 컨트롤러의 X/Y 버튼은
  시점/패널 제어에 계속 쓰지만 왼팔 IK·왼쪽 trigger action은 생성하지 않는다.
  동작 중 tracking 검사는 HMD와 활성 오른쪽 컨트롤러만 필요하다.
- head/wrist RGB 패널·해상도·desktop camera preview 옵션은 그대로 사용한다.
  이 영상은 표시 전용이고 RL observation에는 들어가지 않는다. Depth·데이터 기록·
  공장 배경은 생성하지 않는다. `--control-hz`, `--episode-seconds`, `--arm-stiffness`,
  `--arm-damping`, recording 관련 옵션은 이 모드에서 적용하지 않는다.
- 최소 구현은 **controllers + scaled**만 지원한다. `--hand-switch`, hands,
  relative/absolute mapping, `--scene-config`, `--domain-randomization`은 거부한다.
  수집기의 `--auto-start`와 무관하게 A로 명시적으로 시작한다.

실제 로봇 안전 제어기가 아니다. USD 물체를 순간 이동시키지 않고 기존 RL action을
통해 물리적으로 움직여 접촉 보상을 검사한다.

## 설정 수정

- `configs/rl_pick_arms_only.py`: 초기 프리셋 이름, 손 선택, 성공 조건과 실험별 설정.
- `src/kuavo_isaaclab_scene/rl/managers/rewards.py`: reward 가중치.
- `src/kuavo_isaaclab_scene/rl/mdp/rewards.py`: 항목별 계산식.

실험별 가중치를 바꾸려면 config의 기존 `configure(env_cfg, agent_cfg)` 안에
`env_cfg.rewards.reaching.weight = 6.0`처럼 추가한다. 학습 중인 config를 바꾸기보다
복사본을 만들어 다음과 같이 선택하는 것을 권장한다.

```bash
./quest_collector.sh collect --rl-reward-debug --rl-config configs/my_reward_probe.py
```

설정 수정은 **검사 프로세스를 재시작**해야 반영된다. 자동 파일 reload는 없다.
로컬 초기 자세·gripper config를 읽으므로 기존 학습의 `manifest.json`/`env.yaml`과
다르면 그 checkpoint가 사용한 환경의 완전한 재현은 아니다.

구현은 `rl/debug/`에 격리되어 있고 학습 runner에서는 import하지 않는다.
RewardProbe는 reward 계산 후, 자동 reset 전에 이미 계산된 값을 읽을 뿐
RewardManager.compute를 추가 호출하지 않는다. 변경 후 시뮬레이터 실행/VR 시각 검증은
수행하지 않았으며 실제 패널 위치·크기와 조작성은 사용자가 확인한다.

## 패널이 안 보이거나 오류 로그가 나오는 경우

- Y/H를 누르면 `[RL REWARD] Reward panel ON/OFF`와 `[XR REWARD TEXTURE]`가 출력된다.
  `ON`, `layer=True`, `panel=True`, `widget_ready=True`인데도 안 보이면 시점 방향이나
  가림 문제일 수 있다. 먼저 카메라 패널을 끄고 확인하고, 필요하면 기존
  `--xr-overlay-forward-axis +z`로 방향을 비교한다. 이는 원인이 확정된다는 뜻은 아니다.
- `widget_ready=False`가 계속 나오면 XR UI 생성 문제다. 이 상태 로그와 먼저 발생한
  오류를 함께 전달한다. 버튼 수신 로그만으로 화면 표시 성공을 판단하지 않는다.
- `[XR REWARD TEXTURE] uploads=...`는 GPU 이미지 업로드 횟수이며, 예전 `[XR PANEL]`
  로그만 나온다면 변경 전 프로세스다. 수집기를 재시작해야 한다. 업로드 성공 역시
  실제 헤드셋 합성까지 검증한 값은 아니다. 투명하지 않은 테두리/배경까지 전혀
  보이지 않으면 이 로그와 카메라 패널의 표시 여부를 함께 전달한다.
- RGBA 텍스트 생성에는 Pillow 10.1 이상을 사용한다(현재 로컬 Isaac 환경에 설치됨).
  다른 PC에서는 프로젝트의 `rl` extra에 포함된다. RGB 카메라 센서를 끈 상태에서도
  작은 desktop viewport 업데이트를 유지해 UI 텍스처가 갱신되도록 한다.
- `bind_event_generator ... token: press`가 `[Error]`로 출력되어도 이후
  `[BUTTON] ... pressed`가 나오면 해당 버튼 콜백은 동작한 것이다. 이 메시지만으로
  연결 실패나 프로세스 중단으로 판단하지 않으며, SDK 로그를 전역으로 숨기지 않는다.
- `Body must be non-kinematic`는 무시할 로그가 아니다. 검사 모드의 고정 conveyor
  표면에 대한 불필요한 zero-velocity reset을 차단했다. 박스의 동역학이나 RL 학습
  runner는 변경하지 않는다. 같은 오류가 계속되면 다른 호출/물체도 확인해야 한다.

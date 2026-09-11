# VR 자세 저장과 eval 초기 상태

`configs/initial_states.json`에 여러 이름의 초기 상태를 저장하고,
`eval_groot.sh`와 `play_rl.sh`에서 `--initial-state NAME`으로 선택한다.
선택한 상태는 첫 reset과 매 episode reset/자동 reset에 적용된다.
RL `train_rl.sh`에도 같은 인자를 사용할 수 있다.

## 제공된 자세: quest_ready_01

2026-09-07 사용자가 VR에서 읽어 전달한 S200062 관절 36개의 실제 위치를
`quest_ready_01`로 저장했다. 양팔, 머리, 몸통, 바퀴, 내장 그리퍼 관절을 포함한다.
각도 단위는 rad이며 이름으로 매칭하므로 배열 순서에 의존하지 않는다.

이 데이터에는 **robot base pose와 box pose가 없었다.** 따라서 이 프리셋은
관절만 복원하며, base와 box는 기존 eval reset 위치를 유지한다. 바퀴 관절
각도에서 베이스 위치를 추정하지 않는다. 같은 손–박스 배치를 재현하려면
아래 capture 함수로 root와 box까지 새 프리셋에 저장한다.

## VR에서 새 자세 저장

VS Code 디버거로 `teleop.collect_quest_teleop`를 실행한 상태에서:

1. VR로 준비 자세를 만든다. A로 따라오기를 멈추고 움직임이 잦아들도록 기다린다.
2. 헤드셋을 벗고 `env.step(action)` 다음 줄에서 브레이크포인트로 멈춘다.
3. 해당 `main()` 스택 프레임의 Debug Console에서 실행한다.

```python
from kuavo_isaaclab_scene.robots.initial_states import capture_initial_state
capture_initial_state(env, "quest_ready_02", path="configs/initial_states.json")
```

기본 캡처 대상은 robot의 **모든 실제 joint position + root pose**이며,
외장 그리퍼가 있으면 그 articulation도 포함한다. 목표 관절값이 아니라
측정 관절값을 저장한다. 머리 자세도 자동 포함된다.

박스 위치와 flap joint 위치까지 함께 저장하려면:

```python
capture_initial_state(env, "middle_rack_ready", path="configs/initial_states.json", objects=("medium_box_0", "large_box_0"), description="중간 선반 앞, 양손 파지 직전")
```

`objects`에는 실제 scene에 있는 물리 객체 key만 넣는다. 지정한 객체가 없으면
저장하지 않고 오류를 낸다. Rack 등 정적 Xform의 배치·scale은 기존
`workcell_layout.json`/`rack_box_poses.json`으로 관리한다.

같은 이름이 있으면 덮어쓰지 않는다. 의도적으로 갱신하려면:

```python
capture_initial_state(env, "quest_ready_02", path="configs/initial_states.json", overwrite=True)
```

같은 파일의 다른 이름들은 보존된다. 파일 갱신은 잠금 및 atomic replace를 사용한다.
실행 중인 VR 프로세스가 이 모듈을 이미 import했다면 수정된 Python 코드 적용에는
프로세스 재시작이 필요할 수 있다.

저장 이름 확인:

```python
from kuavo_isaaclab_scene.robots.initial_states import read_states
list(read_states("configs/initial_states.json")["states"])
```

## Eval에서 사용

프로젝트 루트에서 기존 eval 명령에 다음 옵션을 추가한다.

```bash
--initial-state quest_ready_01
```

GR00T 예시(체크포인트와 기존 policy/worker 옵션은 사용하는 모델에 맞춰 유지):

```bash
./eval_groot.sh \
  --robot-model s200062 \
  --policy-profile kuavo-arm-claw \
  --checkpoint /absolute/path/to/pretrained_model \
  --initial-state quest_ready_01 \
  --episodes 5
```

`kuavo-arm-claw`는 해당 16-D arm/claw policy schema에 맞는 체크포인트용이다.
프리셋 로딩이 체크포인트의 state/action schema를 변환하지는 않는다.
GR00T의 기존 `--initial-pose` 기본값(q50 등)은 이름 있는 상태를 선택하면
`default`로 전환되고 reset 마지막에 프리셋을 적용한다.
`--initial-pose checkpoint-q50/dataset-medoid` 또는 `--initial-head-pitch-deg`를
동시에 명시하면 충돌하는 초기화이므로 오류를 낸다. 머리값은 프리셋에서 수정한다.

### GR00T의 몸체 고정

카메라 시야를 위해 head pitch를 수정할 때는 기존 RL용 `quest_ready_02`를
덮어쓰기보다 별도 eval 프리셋으로 복사한 뒤 `zhead_2_joint`를 조정한다.
로컬에서 실험한 `0.35 rad` 값은 이 변경에 포함하지 않았으며, 저장소의
`quest_ready_02` head pitch는 `−0.3871748 rad`로 유지한다. 실제 시선은
몸통 자세와 카메라 장착각에도 의존하므로 사용자가 화면에서 확인해야 한다.

`kuavo-arm-claw` / `rwh-kuavo-v2-s56` profile은 기본적으로 `--body-mode fixed`를
사용한다. 초기 상태 적용 직후 **베이스를 fixed root로, wheel/leg/knee/waist/head
관절을 해당 초기 각도 주위 ±1e-4 rad의 물리 joint limit로 고정**한다.
팔과 gripper의 구동·수동 링크 관절은 잠그지 않는다. 몸체 고정은 1초 안정화
이전부터 적용되며 반복 episode reset에서도 새 초기 자세를 다시 고정한다.
매 physics step마다 pose를 덮어쓰는 방식이 아니므로 solver 오차는 남을 수 있다.

```bash
--initial-state quest_ready_02 --body-mode fixed
```

기존처럼 몸체를 PD 목표로만 유지하는 비교 실험은 `--body-mode pd`를 명시한다.
Legacy `default` profile의 기본값은 호환성을 위해 `pd`다. 물리 고정은 실제
로봇의 균형·몸체 제어 성능을 재현하는 모드가 아니라 팔 조작 평가용 제약이다.
결과 JSON의 `body_mode`, episode별 `body_lock`과 trace의 `body_after`에
고정 대상/목표/측정 각도와 오차가 기록된다. 측정은 control-step 기준이며
자동 reset 직후의 상태는 오차 집계에서 제외한다.

S200062 `quest_ready_02`의 head 조정 전 고정 검증에서는 60초×2회 동안 knee/leg/waist/head 편차가
0.008° 미만이고 베이스 위치·회전이 유지됐다. 단, 연속 회전 바퀴는 좁은 limit를
요청해도 최대 약 6°의 잔여 회전이 관측됐다. 베이스 고정과 바퀴 회전각의 완전한
고정은 구분해야 하며, episode의 전체 관절 최대 오차에는 바퀴도 포함된다.

RL 예시(학습 때 사용한 boxes/cargo/slot/action 설정과 checkpoint manifest를 유지):

```bash
./play_rl.sh \
  --robot-model s200062 \
  --task pick \
  --checkpoint /absolute/path/to/model_1999.pt \
  --initial-state quest_ready_01 \
  --episodes 20
```

다른 파일의 프리셋도 선택할 수 있다.

```bash
--initial-states-file configs/my_eval_states.json --initial-state middle_rack_ready
```

한 실행은 선택한 하나의 프리셋을 반복 사용한다. 여러 프리셋을 비교하려면
이름을 바꿔 별도 실행한다. 결과 JSON과 RL manifest에 프리셋 이름·경로뿐 아니라
실제로 사용한 상태 데이터도 기록되므로 원본 JSON을 나중에 고쳐도 추적할 수 있다.

## 복원 규칙과 범위

- root pose는 `[x, y, z, qw, qx, qy, qz]`; 위치는 m, 회전은 단위 quaternion.
  위치는 env origin을 뺀 값으로 저장하고 복원 대상 env origin을 더한다.
  `env_0`에서 캡처한 자세를 병렬 env에 적용할 때 World 원점에 겹쳐 생성하지 않는다.
- 저장된 robot/gripper 종류가 현재 선택과 다르면 거부한다. 없는 관절과 실제
  joint limit를 벗어난 값도 거부한다. 시뮬레이터 측정의 미세한 limit 초과
  (1e-4 rad 이하)만 물리 한계로 보정한다.
- 전체 대상 검증 후 관절 위치를 쓰고 joint/root 속도를 0으로 초기화한다.
  PD 위치 목표도 같은 값, 속도 목표도 0으로 맞춘다. 정책 action의 기본 offset이나
  observation의 default joint 기준은 변경하지 않는다.
- 초기 상태는 기존 reset 이벤트 **마지막에** 적용되고 그 다음 action/command
  manager가 reset된다. 따라서 RL의 손 open reset과 base reset보다 프리셋이 우선한다.
  root가 없는 관절 전용 프리셋은 기존 RL base reset/jitter 결과를 유지한다.
- **초기 pose 복원 자체는 머리·허리 고정 기능이 아니다.** GR00T arm/claw eval은
  위 `--body-mode fixed`를 기본으로 함께 적용한다. RL에서 베이스·몸통·머리를 유지하고 팔만 제어하려면
  아래 `--control-mode arms-only`를 함께 사용한다.
  기존 GR00T legacy profile의 stabilization은 추가 그리퍼를 open으로 명령한다.
  저장된 파지를 유지하는 실험은 호환되는 arm/claw profile과 제어 구성을 사용해야 한다.
- 충돌 검사나 IK 도달 가능성을 자동으로 보장하지 않는다. 특히 다른 rack 배치·박스
  scale·그리퍼 mount에서 같은 관절값을 쓰면 접촉/관통이 달라질 수 있다.
- 이 기능은 **정지 상태 초기화**다. 속도, solver 접촉 cache, flap drive/friction,
  물체 scale, RL phase/성공 이력은 저장하지 않는다. 이미 잡은 물체의 동역학을
  정확히 이어서 재생하는 checkpoint가 아니다. RL carry/place의 predecessor
  `--reset-bank`와는 동시에 사용할 수 없다.
- mutable 파일은 checkout의 `configs/initial_states.json`이다. 배포 fallback은
  `src/kuavo_isaaclab_scene/configs/initial_states.json`이다. 새로운 프리셋을 wheel에도
  넣으려면 후자에도 반영한 뒤 빌드한다. 설치된 wheel에서는 쓰기 가능한 파일을
  `path=` / `--initial-states-file`로 명시하는 것을 권장한다.

구현: `robots/initial_states.py`; RL 연결: `rl/runners/common.py`;
GR00T 연결: `evaluation/eval_groot.py`.

## RL: 저장한 몸통·머리·베이스를 고정하고 팔만 학습

현재 `configs/rl_pick_arms_only.py`는 초기 상태를 `quest_ready_02`로 고정하고,
**오른손 한 손 flap 파지 → 6cm 상승 → 0.5초 유지**를 학습한다. 성공에 필요한
손 수와 별개로 `arms-only` action은 양팔 14개와 양 gripper 2개의 16차원이다.
GR00T의 16-D arm/claw action과는 순서·단위·제어 방식이 다르므로 서로 대체하지 않는다.

```bash
./train_flap_pick.sh --num-envs 2 --headless
./play_flap_pick.sh --checkpoint /absolute/path/to/model_1999.pt --num-envs 1
```

학습과 평가는 같은 config 및 checkpoint manifest를 유지한다. 위 전용 설정에
`--initial-state quest_ready_01`을 덧붙여 고정 프리셋을 우회하지 않는다.
초기 상태 변경은 별도 실험으로 관리하고 아래 문서를 기준으로 수정한다.

- [RL 초기 상태](RL_INITIAL_STATES.md): `quest_ready_02` 수정·캡처와 몸통 고정.
- [1단계 flap pick](RL_FLAP_PICK.md): 현재 성공 조건, action 및 실행·재개 방법.
- [배경 없는 RL 병렬 환경](RL_PARALLEL_ENVS.md): 카메라·배경 제거와 병렬 학습.
- [하위 task별 RL 학습](RL_TRAINING.md): manager별 수정 위치와 다른 task 구성.

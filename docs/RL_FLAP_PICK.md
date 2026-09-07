# 1단계: 양손 flap 상단 파지 → 6cm 상승 → 유지

이 단계는 고정된 베이스·허리·머리에서 양팔과 그리퍼만 학습한다.
랙 밖 인출, 베이스 이동, waist 회전, 컨베이어 배치는 다음 단계다.
구현은 manager-based이고 시작 설정은 `configs/rl_pick_arms_only.py`에 있다.

## 작업 정의

`grasp_mode="flap_top"`, `control_mode="arms-only"`, `required_grasp_hands=2`를 사용한다.

| 항목 | 기본값 |
|---|---|
| 왼손 대상 | `flap_right` (+X쪽 flap) |
| 오른손 대상 | `flap_left` (-X쪽 flap) |
| 접근 목표 | 각 flap 상단 중앙에서 아래로 15mm |
| 유효 접촉 영역 | flap 상단에서 아래로 30mm, 판 너비 안쪽 |
| 접촉 위치 여유 | 4mm (얇은 판의 contact offset 고려) |
| 파지 판정 | 각 손의 두 손가락이 **같은 지정 flap의 반대 면**에 접촉, 각 0.2N 초과 |
| 들어올리기 | 박스 본체 중심이 reset 기준 6cm 초과 상승 |
| 기울기 | 40° 이내 |
| 안정성 | 박스 선속도 < 0.08m/s, 각속도 < 0.35rad/s |
| 유지 시간 | 위 조건을 연속 0.5초 유지 |
| 추가 접촉 제한 | 손가락의 지정 flap 외 접촉력 < 10N |
| 내용물·선점 박스 | 모두 0개 |
| randomization | 첫 실험 OFF |

`grasp_flaps`는 **로봇 왼손, 오른손 순서**다. Rack 방향/robot 위치에 따라
이 배정을 바꿀 수 있다. 예를 들어 front/back 판을 잡으려면
`grasp_flaps=("flap_front", "flap_back")`로 설정한다.
현재는 서로 다른 두 flap을 집는 구성이다. 같은 flap의 두 모서리를 집는
작업에는 별도의 두 grasp 위치 및 contact 배정이 필요하다.

## 접촉과 flap 물리

각 손가락 ContactSensor가 자기 손에 지정된 flap의 힘과 평균 접촉 위치를 읽는다.
접촉 위치는 실제 flap rigid link frame으로 변환하여 상단 영역과 양면 접촉을
검사한다. 박스 본체만 잡거나 flap 하단을 잡은 것은 성공으로 인정하지 않는다.
접촉점이 없는 NaN 값은 실패 접촉으로 처리한다. PhysX가 보고하는 평균 접촉점
기반이므로 접촉 패치 전체의 모든 점이 상단에 있는지까지 판정하는 것은 아니다.

네 flap의 revolute joint는 열린 직립 자세(0°) 주변 `±0.5°`로 제한한다.
약한 drive(stiffness=2, damping=0.2)를 함께 사용하며 물리 joint로 하중을 전달한다.
원본 USD와 VR용 flap 설정은 수정하지 않고 RL spawn 시에만 적용한다.
손에 박스를 parent하거나 매 step 박스 pose를 옮기는 보조 파지는 없다.

박스/판 크기는 composed USD의 내부 scale과 외부 spawn scale을 모두 반영한다.
기존 단위 템플릿 좌표를 그대로 미터로 간주하지 않는다. 실제 fingertip grasp
중심과 `tool_offset`의 일치 여부는 사용자의 시뮬레이션 확인으로 보정해야 한다.
접촉 성공은 실제 센서로 판정하지만 접근 보상은 이 tool reference를 사용한다.

## Observation / Action / Reward

정책 입력은 다음이다.

- 로봇 관절 위치와 속도.
- 박스 pose: robot frame 위치 3 + quaternion wxyz 4. **박스 속도는 입력하지 않는다.**
- 손별 grasp 목표 상대 위치, flap 상대 회전, gripper 닫힘 축과 flap 법선의 정렬도.
- 지정 flap 접촉력, 양손 파지 여부, 기타 손가락 접촉력, 팔/손목/손 하우징 접촉력.
- 박스 크기, 상승 높이, 유지 시간, 누적 actuator 목표, 이전 action.

6-DoF pose의 회전을 quaternion 4개로 표현하므로 저장/입력 차원은 7이다.
박스 선속도·각속도는 **안정성 판정과 reward에만** 사용한다. 영상 입력은 없다.

Action은 왼팔 7개·오른팔 7개의 관절 목표 변화량과 두 gripper 개폐 변화량이다.
기존 `arms-only`의 몸통 lock과 초기 상태 로더를 사용한다.

보상은 손별 접근/방향 정렬 → 상단 접촉 → 양면 파지 → 상승 → 안정 유지 순으로
신호를 제공한다. 지정 flap 외 접촉과 팔/하우징 충돌에는 penalty를 부여한다.
심한 팔 접촉은 기존 unsafe termination에도 반영된다. 이 구조 자체가 실제
배포에서 무충돌을 보장하는 것은 아니며, self-collision filter와 안전 제어는
실제 모델에 맞춰 확인해야 한다.

## 설치와 실행

현재 요청에 따라 **학습/시뮬레이션 실행은 하지 않은 상태**다. 준비 중
`env_isaaclab_232`에 `rsl-rl-lib==3.1.2`, `tensordict==0.8.3` 및 관련 의존성을
설치했다. Torch 2.7.0+cu128은 유지했다.

사용자가 실행을 결정한 후 프로젝트 루트에서 아래 명령을 사용한다.
다른 컴퓨터에는 먼저 [stable 설치 가이드](INSTALL.md)를 따라
Isaac Sim 5.1.0 / Isaac Lab 2.3.2 환경과 프로젝트를 설치한다.

```bash
conda activate env_isaaclab_232
python -m pip install -e '.[rl]'
./train_flap_pick.sh
```

전용 실행 파일은 S200062 + 내장 two-finger, `pick`, `medium_box_0`,
`arms-only`, 현재 캡처된 rack/box 배치를 선택한다. 기본값은 **2 env,
2000 PPO iteration, headless**다. 2000 iteration은 학습량 설정일 뿐 성공률
80%에 도달한다는 보장은 아니다. 카메라는 기본적으로 생성하지 않는다.

`configs/rl_pick_arms_only.py`의 `INITIAL_STATE = "quest_ready_02"`로 고정한다.
첫 reset과 모든 episode reset에서 같은 root pose + 36개 관절값을 적용한다.
현재 저장된 base 위치는 env origin 기준 약 `(-0.240913, 0.066122, 0.000004)m`다.
박스 pose는 이 프리셋에 들어 있지 않으므로 `configs/rack_box_poses.json`을 쓴다.
목표 flap에 도달 가능한지/초기 접촉이 적절한지는 아직 이 RL 환경에서 검증하지 않았다.

자원 및 학습량은 실행 옵션으로 변경할 수 있다.

```bash
./train_flap_pick.sh --num-envs 4 --max-iterations 4000 --seed 42
./train_flap_pick.sh --help
```

실행 경로와 무관하게 전용 스크립트는 프로젝트의 config와 캡처 파일을 사용한다.
모델·task·박스·초기 상태·config를 바꾸는 CLI 옵션은 전용 스크립트에서 거부한다.
새 실험은 설정 파일을 복사하고 일반 `train_rl.sh`/`play_rl.sh`를 사용한다.
일반 실행기에서도 config에 `INITIAL_STATE`가 있으면 이를 따르며,
충돌하는 `--initial-state`는 오류로 처리한다. 이름은 대소문자를 구분한다.

로그/체크포인트는 기본적으로 `artifacts/rl/pick/train_<시간>_<ID>/` 아래 생성된다.
`manifest.json`에는 초기 상태의 이름뿐 아니라 실제 값도 기록되고,
`env.yaml`, `agent.yaml`에 환경/학습 설정이 저장된다.
`model_*.pt`는 100 iteration 간격과 학습 종료 시 저장된다.
Ctrl+C로 중단하면 마지막으로 저장된 checkpoint를 사용한다.

```bash
# 마지막 저장 checkpoint에서 재개 (이번 호출에서 수행할 iteration 수)
./train_flap_pick.sh --checkpoint /absolute/path/to/model_1000.pt --max-iterations 1000

# 같은 자세와 task로 평가: 기본 GUI, 1 env, 20 episode
./play_flap_pick.sh --checkpoint /absolute/path/to/model_1999.pt

# 렌더링 없는 성공률 평가
./play_flap_pick.sh --checkpoint /absolute/path/to/model_1999.pt --episodes 100 --headless

# 학습 지표
tensorboard --logdir artifacts/rl/pick
```

평가 결과는 `artifacts/rl/pick/play_<시간>_<ID>/metrics.json`에 저장된다.
`--log-dir`로 학습/평가 로그의 상위 폴더를 바꿀 수 있다.
checkpoint와 같은 폴더의 `manifest.json`도 함께 보관한다.
새 관측/접촉 정의는 예전 body-grasp/whole-body checkpoint와 호환되지 않는다.
checkpoint 옆 manifest가 잘못된 구성을 거부한다.

TensorBoard에서 `grasp_left`, `grasp_right`, `lift_height`, `hold_fraction`,
`left_target_distance`, `right_target_distance` command metric으로 어느 단계에서
막히는지 확인한다. 초기 pose부터 손–flap 거리가 줄어들지 않으면 파지 배정과
root 위치 및 tool offset을 먼저 확인한다.

## 수정 위치

| 목적 | 파일 |
|---|---|
| 전용 학습/평가 명령의 공통 기본값 | `scripts/flap_pick.sh` |
| 초기 자세의 root 및 관절 값 | `configs/initial_states.json` → `states.quest_ready_02` |
| 실험 기본값, 대상 flap, 높이·기울기·유지 시간 | `configs/rl_pick_arms_only.py` |
| 공통 task 필드 및 검증 | `rl/tasks/specs.py` |
| composed USD 크기·flap geometry | `rl/tasks/asset_geometry.py` |
| flap 제한, 센서 배치 | `rl/tasks/flap_spawn.py`, `rl/tasks/scene_cfg.py` |
| 상단/양면 접촉 판정 | `rl/mdp/flap_grasp.py` |
| 성공 유지 조건 | `rl/mdp/commands.py` |
| 관측 항목·계산 | `rl/managers/observations.py`, `rl/mdp/observations.py` |
| 보상 가중치·계산 | `rl/managers/rewards.py`, `rl/mdp/rewards.py` |

초기 자세 수정/VR 재캡처는 [RL 초기 상태 가이드](RL_INITIAL_STATES.md)를 참고한다.

물리 검증용 테스트 코드는 추가했지만 이 변경의 테스트 실행/학습 성능/실제 파지
성공률은 확인하지 않았다.

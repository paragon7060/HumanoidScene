# 1단계: 한 손 flap 상단 파지 → 6cm 상승 → 유지

이 단계는 베이스·허리·머리를 고정하고 양팔과 그리퍼를 학습한다. 지정한 한 손으로
flap을 집으며, 반대손으로 박스 본체를 받치는 동작은 허용한다.
랙 밖 인출, 베이스 이동, waist 회전, 컨베이어 배치는 다음 단계다.
구현은 manager-based이고 시작 설정은 `configs/rl_pick_arms_only.py`에 있다.
Scene은 일반/teleop 환경과 독립적인 `rl/scenes/`에서 생성한다.
[RL 전용 병렬 환경](RL_PARALLEL_ENVS.md)에 배경 제거·복제·충돌 격리 구조를 정리했다.

## 작업 정의

`grasp_mode="flap_top"`, `control_mode="arms-only"`, `required_grasp_hands=1`, `grasp_hand="right"`를 사용한다.

| 항목 | 기본값 |
|---|---|
| 왼손 | 박스 받침 허용 (flap 배정 정보: `flap_right`) |
| 오른손 대상 | `flap_left` (-X쪽 flap) |
| 접근 목표 | 선택한 flap 상단 중앙에서 아래로 15mm |
| 유효 접촉 영역 | flap 상단에서 아래로 30mm, 판 너비 안쪽 |
| 접촉 위치 여유 | 4mm (얇은 판의 contact offset 고려) |
| 파지 판정 | 선택한 손의 두 손가락이 **같은 지정 flap의 반대 면**에 접촉, 각 0.2N 초과 |
| 들어올리기 | 박스 본체 중심이 reset 후 안착 기준 6cm 초과 상승 |
| 기울기 | 40° 이내 |
| 안정성 | 박스 선속도 < 0.08m/s, 각속도 < 0.35rad/s |
| 유지 시간 | 위 조건을 연속 0.5초 유지 |
| 추가 접촉 제한 | 파지 손가락의 지정 flap 외 접촉력 < 10N; 반대손 박스 접촉 허용 |
| 주변 장애물 | 모든 로봇 rigid link의 랙·펜스·버튼·컨베이어 접촉 시 즉시 실패 (0.1N 초과) |
| 내용물·선점 박스 | 모두 0개 |
| randomization | 첫 실험 OFF |

`grasp_flaps`는 **로봇 왼손, 오른손 순서**다. Rack 방향/robot 위치에 따라
이 배정을 바꿀 수 있다. 예를 들어 front/back 판을 잡으려면
`grasp_flaps=("flap_front", "flap_back")`로 설정한다.
`grasp_hand="left"`로 바꾸면 왼손의 flap 파지를 요구하고 오른손의 받침을 허용한다.
`required_grasp_hands=2`는 기존 양손 과제를 유지한다. 이때 서로 다른 두 flap을 집는다.

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
- 박스 pose: robot frame 위치 3 + quaternion wxyz 4, 선속도 3 + 각속도 3.
- 손별 grasp 목표 상대 위치, flap 상대 회전, gripper 닫힘 축과 flap 법선의 정렬도.
- 지정 flap 접촉력, 양손 파지 여부, 기타 손가락 접촉력, 로봇 link별 주변 장애물 접촉력.
- 박스 크기, 상승 높이, 유지 시간, 누적 actuator 목표, 이전 action.

6-DoF pose의 회전을 quaternion 4개로 표현하므로 저장/입력 차원은 7이다.
안정성 판정에 필요한 박스 선속도·각속도도 관측에 포함한다. 영상 입력은 없다.

Action은 양팔 14개 관절 목표 변화량과 두 gripper 개폐 변화량, 총 16개다.
기존 `arms-only`의 몸통 lock과 초기 상태 로더를 사용한다.

보상은 선택한 손의 접근/방향 정렬 → 상단 접촉 → 양면 파지 → 상승 → 안정 유지 순으로
신호를 제공한다. 파지 손가락의 지정 flap 외 접촉에는 penalty를 부여하지만, 반대손의 박스 받침은 벌점/실패 대상이 아니다.

박스가 미끄러운 랙 위에서 흔들리지 않도록 두 보상을 추가했다.

| 항목 | 설정 | 의도 |
|---|---|---|
| `prelift_disturbance` | weight -4.0 | 안착 위치에서 밀거나 흔드는 행동 억제 |
| `stable_grasp` | weight +2.0 | 실제 flap 파지를 유지하면서 수평·회전 운동이 작은 상태 보상 |
| 수평 위치 scale | 0.02m | 밀린 뒤 정지해도 위치 오차 벌점 유지 |
| 수평/비파지 수직 속도 scale | 0.05m/s | 박스를 밀거나 튕기는 행동 억제 |
| 각속도 scale | 0.5rad/s | 회전 흔들림 억제 |
| 안착 자세 대비 회전 scale | 10° | 원래 기울어진 선반 자세를 기준으로 비교 |
| 벌점 해제 | 유효 파지 AND 0.01m 초과 상승 | 정상적으로 든 다음의 움직임 허용 |

변위 제곱 + 0.25×수평 속도 제곱 + 0.25×각속도 제곱 + 0.5×회전 오차 제곱 +
0.25×비파지 수직 속도 제곱을 각 scale로 정규화한다. 각 성분을 4로 제한한 뒤 합산하고
-4.0을 곱한다. 실제 파지 중 수직 상승에는 이 속도 벌점을 적용하지 않는다.
단순히 박스를 튕겨 띄우는 것은 파지 판정이 없으므로 벌점을 해제하지 않는다.
모든 보상률은 RewardManager가 제어 dt를 곱한다.

기준 위치·자세는 reset 안착 완료 시 한 번 저장한다. 관측에 기준 대비 위치·회전 7차원을
추가해 전체 관측은 223차원이다. 안착 전에는 두 보상을 비활성화한다. 파지 성공·충돌·
상승 조건은 그대로 유지하며, 마찰값을 바꾸는 대신 현재 물리 상태에서 보상으로 유도한다.


장애물 센서는 모든 로봇 rigid link(손가락 포함)에 하나씩 생성한다. 랙·펜스·버튼·컨베이어와
그 위 선점 물체만 필터링하고 작업 박스는 제외한다. 물리 4스텝의 힘 이력에서 최대 접촉을
검사하므로 제어 스텝 사이의 짧은 충돌도 실패다. 초기 3스텝 유예와 180N 기준은 이 과제에
적용하지 않는다. 초기 자세가 장애물과 접촉하면 첫 스텝부터 실패하므로 초기 배치를 수정해야 한다.
센서별 단일 link 배정은 [Isaac Lab의 filtered contact 제약](https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/sensors/contact_sensor/contact_sensor_cfg.html)을 따른다.
GPU의 filtered contact가 고정 장애물도 읽도록 랙과 펜스는 움직이지 않는 kinematic rigid body로 생성한다.
필터마다 환경당 rigid body 하나만 지정한다. 종료 규칙만으로 학습 정책의 무충돌을 보장하지는 않는다.
박스가 랙에 놓여 있는 지지 접촉은 유지한다. 여기서 충돌 금지는 로봇과 주변 장애물 사이에 적용된다.

손가락 표면 마찰은 공통 `configs/grippers.json`의 선택한 preset에서 읽는다.
2026-09-07 원격 `2d3764d` 기준 S200062/S56 two-finger 기본값은 **5.0 / 4.0, average**다.
RL에서 별도 상수를 덮어쓰지 않으며, 실제 값은 로그와 checkpoint 계약에 기록한다.

## 설치와 실행

CPU 회귀 테스트로 접촉·관절 고정·보상 로직을 검증했다. 현재 머신에는 `env_isaaclab_232`
환경과 PPO 의존성을 설치했고 `doctor.sh` 버전 검사를 통과했다. GPU 1번(A100 80GB)에서
PyTorch CUDA 연산과 Isaac Sim의 headless 실행을 확인했다. EULA 동의 후 실행하며,
병렬 환경별 측정 결과는 [병렬 환경 가이드](RL_PARALLEL_ENVS.md)에 기록한다.
짧은 PPO 실행은 학습 수렴이나 파지 성공률 검증과 별개다.

2026-09-07 검증: 관련 CPU·USD·그리퍼 테스트 117개 통과. S200062 rigid link 64개와
랙 collider 16개에 대한 센서 경로 포함 여부를 확인했다. USD 구조 검사는 별도
`usd-core` 바인딩을 사용했으며 Isaac Sim 5.1의 PhysX 실행 검증은 아니다.

Isaac Lab 실행 환경에서 프로젝트 루트의 아래 명령을 사용한다.
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
초기 오른손과 목표 flap의 거리는 약 0.196m다. 30 제어 스텝의 zero-action 유지에서
장애물 충돌로 종료되지 않았다. 다만 캡처된 박스 root 높이 1.14313m가 물리 안정화 후
약 1.04097m로 내려앉았다. 이전 코드는 낙하 전 높이를 기준으로 저장해 실제 상승 목표가
약 16.2cm가 되었다. 이제 reset 후 최소 0.5초 동안 동작 목표를 유지하고, 박스 속도가
0.01m/s 및 0.05rad/s 미만으로 0.2초 유지되면 그 높이를 저장한다.
이후 안정된 높이에서 6cm 상승을 평가한다. 2초 안에 안정되지 않으면 실패한다.
각 환경은 독립적으로 안착하며, 이 기간에도 로봇과 장애물의 0.1N 충돌 검사는 유지한다.
관측에는 준비 여부와 안착 경과 시간이 포함되고, 준비 전에는 파지 shaping 및 동작 비용을 주지 않는다.
실제 파지 가능성과 성공률은 추가 확인이 필요하다.

자원 및 학습량은 실행 옵션으로 변경할 수 있다.

```bash
./train_flap_pick.sh --num-envs 4 --max-iterations 4000 --seed 42
CUDA_VISIBLE_DEVICES=1 ./train_flap_pick.sh --num-envs 16384 --device cuda:0 \
  --save-interval 1000 \
  --kit_args "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
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
`model_*.pt`는 기본 1000 iteration 간격과 학습 종료 시 저장된다.
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
한 손 파지 정의와 박스 속도·장애물 접촉 관측은 예전 양손 flap/body-grasp/whole-body checkpoint와 호환되지 않는다. 새 학습을 시작해야 한다.
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
| 병렬 복제, 충돌 격리 기본값 | `rl/envs/parallel_cfg.py` |
| RL 전용 manager 환경 조립 | `rl/envs/env_cfg.py` |
| composed USD 크기·flap geometry | `rl/scenes/asset_geometry.py` |
| flap 제한, 센서 배치 | `rl/scenes/flap_spawn.py`, `rl/scenes/sensors.py` |
| 상단/양면 접촉 판정 | `rl/mdp/flap_grasp.py` |
| 장애물 접촉 이력/실패 기준 | `rl/mdp/collisions.py`, `rl/mdp/commands.py`, `rl/tasks/specs.py` |
| 성공 유지 조건 | `rl/mdp/commands.py` |
| 관측 항목·계산 | `rl/managers/observations.py`, `rl/mdp/observations.py` |
| 보상 가중치·계산 | `rl/managers/rewards.py`, `rl/mdp/rewards.py` |

초기 자세 수정/VR 재캡처는 [RL 초기 상태 가이드](RL_INITIAL_STATES.md)를 참고한다.

CPU 테스트는 물리 엔진 검증을 대체하지 않는다. 초기 자세의 도달 가능성, 단일 flap의
하중 전달, 마찰·모터 힘과 실제 파지 성공률은 Isaac Lab rollout으로 추가 확인해야 한다.

체크포인트 기본 저장 간격은 1000 PPO iteration이다. RSL-RL은 첫 업데이트와 정상 종료 시에도
저장하므로 2000회 실행에서는 통상 `model_0.pt`, `model_1000.pt`, `model_1999.pt`가 생긴다.
`--save-interval`로 간격을 바꿀 수 있다. 마찰 및 안착 기준 변경 전 checkpoint는 새 계약과 호환되지 않는다.
SIGINT/SIGTERM은 학습 루프를 중단하고 환경을 닫는다. 재실행 전 해당 PID가 GPU 목록에서 사라졌는지 확인한다.

2026-09-07 현재 실행은 16384 env, 2000 PPO iteration, save interval 1000이다.
실행 로그는 `artifacts/rl/stable_grasp/`, 실행 PID/명령은 같은 폴더의 `launch_*.json`에 기록했다.

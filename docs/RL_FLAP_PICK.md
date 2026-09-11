# 1단계: 오른손 flap 면 파지 → 6cm 상승 → 유지

현재 기본 실험은 베이스·허리·머리와 왼팔·왼손을 초기 자세로 고정하고 오른팔·오른쪽
그리퍼만 학습한다. 오른손은 `flap_left` 또는 `flap_right` 어느 쪽이든 집을 수 있다.
랙 밖 인출, 베이스 이동, waist 회전, 컨베이어 배치는 다음 단계다.
구현은 manager-based이고 시작 설정은 `configs/rl_pick_arms_only.py`에 있다.
Scene은 일반/teleop 환경과 독립적인 `rl/scenes/`에서 생성한다.
[RL 전용 병렬 환경](RL_PARALLEL_ENVS.md)에 배경 제거·복제·충돌 격리 구조를 정리했다.
[오른손 실험 상세·파지 진단·reward 튜닝](RL_RIGHT_HAND_PICK.md)에 현재 revision 3 계산식과 수정 방법을 정리했다.

## 작업 정의

`grasp_mode="flap_top"`은 기존 모드 이름이다. 현재는 `flap_contact_region="surface"`,
`control_mode="arms-only"`, `active_arm="right"`, `required_grasp_hands=1`, `grasp_hand="right"`를 사용한다.

| 항목 | 기본값 |
|---|---|
| 왼팔·왼손 | 초기 자세로 관절 lock, action 없음·접근 보상 없음 |
| 오른손 대상 | `flap_left`, `flap_right` 둘 다 허용 |
| 접근 목표 | 두 flap의 전체 넓은 면 중 tool reference와 가장 가까운 점 |
| 유효 접촉 영역 | flap 전체 면. `top_band` 옵션으로 상단 30mm만 허용 가능 |
| 접촉 위치 여유 | 4mm (얇은 판의 contact offset 고려) |
| 최초 파지 판정 | 오른손 두 손가락이 **같은 후보 flap**에 접촉, 각 0.2N 초과 + 양면 배치 |
| 파지 유지 | 같은 flap의 유효 양면 접촉이 각 0.1N 초과이면 link 원점 이동량과 무관하게 유지 |
| 접촉 누락 유예 | 0.1초 미만; 마지막 유효 접촉 대비 중점 이동 2cm·간격 증가 8mm 이내일 때만 |
| 들어올리기 | 박스 본체 중심이 reset 후 안착 기준 6cm 초과 상승 |
| 기울기 | 40° 이내 |
| 속도 제한 | pick 성공 조건에서는 선속도·각속도 제한 없음 |
| 유지 시간 | 위 조건을 연속 0.5초 유지 |
| 추가 접촉 제한 | pick 성공에서 잔여 접촉력 제한 제거; 파지 판정용 flap 접촉력은 유지 |
| 주변 장애물 | 현재 collision_constraints_enabled=False: 충돌 실패/비용 OFF, 물리·센서는 유지 |
| 내용물·선점 박스 | 모두 0개 |
| randomization | 첫 실험 OFF |

`grasp_flaps`는 **양손이 공유하는 후보 목록**이며 더 이상 손별 배정이 아니다.
예를 들어 front/back 판을 잡으려면
`grasp_flaps=("flap_front", "flap_back")`로 설정한다.
왼손 전용 실험은 `active_arm="left"`, `grasp_hand="left"`를 함께 설정한다.
양팔을 풀려면 `active_arm="both"`로 바꾼다. 양손 파지가 모두 필요하면
`required_grasp_hands=2`도 설정한다. 이때 각 손은 독립적으로 두 후보 중 하나를 잡는다.

## 접촉과 flap 물리

각 손가락 ContactSensor가 두 후보 flap의 힘과 평균 접촉 위치를 **각각** 읽는다.
접촉 위치는 실제 flap rigid link frame으로 변환하여 허용 영역과 양면 접촉을
검사한다. 평균 접촉점의 법선 방향 부호로 양면 배치를 판별하기 어려운 얇은 판에서는
두 손가락 link 원점이 판 중심의 반대쪽에 있는지도 사용한다. 이 경우에도 **같은 flap의
허용 영역에 두 유효 접촉과 접촉력**이 있어야 최초 파지를 인정한다. 박스 본체만 잡거나
두 손가락이 서로 다른 flap에 하나씩 닿은 것은 인정하지 않는다. NaN 접촉점으로 새 파지를 만들지 않는다.
PhysX가 보고하는 평균 접촉점
기반이므로 접촉 패치 전체의 모든 점을 검사하는 것은 아니다.

최초 파지와 유지 판정을 구분한다. `top_band` 모드만 파지 후 영역을 아래로 2cm 늘린다.
`surface` 모드는 전체 flap 경계+margin을 유지한다. 유지 중 접촉력
기준을 절반으로 낮춘다. 유효 접촉이 확인되면 link 원점의 중점 이동이나 간격 증가만으로
해제하지 않고 해당 배치를 새 기준으로 저장한다. 접촉이 누락된 경우에만 마지막 유효 접촉
대비 중점 이동 2cm·간격 증가 8mm 이내일 때 0.1초 미만의 유예를 허용한다
(30Hz에서 연속 2개 샘플까지). 긴 누락 뒤에는 다시
최초 파지 조건을 충족해야 한다. 단순히 박스가 올라갔다는 이유로 파지를 인정하지 않는다.
이 유예 중에는 유지 시간이 이어질 수 있으며, 이는 센서 누락에 대한 제한된 허용이다.

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
- 양손의 순간 파지 판정과 접촉 누락 시간(파지 유지 이력, 총 4차원).
- 양손별 현재 접촉 대표 후보의 one-hot ID(총 4차원). 유지 중에는 잡고 있는 flap, 미파지 시에는 최근접 flap이다.
- 박스 크기, 상승 높이, 유지 시간, 누적 actuator 목표, 이전 action.

6-DoF pose의 회전을 quaternion 4개로 표현하므로 저장/입력 차원은 7이다.
안정성 판정에 필요한 박스 선속도·각속도도 관측에 포함한다. 영상 입력은 없다.

기본 Action은 오른팔 7개 관절 목표 변화량과 오른쪽 gripper 개폐 변화량, 총 8개다.
왼팔·왼손·몸통은 초기 상태의 관절 위치 ±0.0001rad로 물리 제한하고 목표를 유지한다.
`active_arm="both"` 설정이면 기존 총 16개 action을 사용한다.

보상은 오른손의 최근접 면 접근 → 면 접촉 → 양면 파지 → 상승 → 안정 유지 순으로
신호를 제공한다. 접근 reward는 `4 * (현재 exp(-12*d) - 직전 exp(-12*d))`이고 방향 정렬로 곱하지 않는다.
방향 정렬은 여전히 observation에 있다. 왼손 접근·접촉 자체에 보상을 주지 않는다.

`orientation`은 별도 weight +0.5 항이다. 오른손이 파지 전 최근접 면 10cm 이내에
같은 flap에 대한 정렬 점수 `|닫힘축·flap법선|²`가 개선되면 +, 악화되면 -다.
거리 가중치는 직전/현재 거리 중 큰 값으로 계산한다. 정지하거나 각도 변화 없이 접근하면
정렬 보상은 0이다. 식과 전환 규칙은 [상세 가이드](RL_RIGHT_HAND_PICK.md)를 따른다.

`flap_contact`는 에피소드 최초 유효 파지에 +3을 한 번 지급한다. 단순 접촉·재파지에는
지급하지 않는다. `lift`는 파지 중 정규화 높이의 signed 변화량 ×5이며, 5.9cm에서
멈춰도 추가 보상은 0이다. 파지 전환 순간에는 기준만 갱신한다. 목표 높이에서의
`holding`과 성공 보너스는 유지한다. `flap_progress_history` 관측 11개를 추가했으므로
기존 checkpoint 대신 새 학습으로 시작한다.

박스의 불필요한 흔들림에는 기존 disturbance 비용을 유지한다.

| 항목 | 설정 | 의도 |
|---|---|---|
| `prelift_disturbance` | weight -0.25, raw cap 1 | 작은 접촉 시도를 과하게 억제하지 않음 |
| `stable_grasp` | 비활성화(None) | 선반 위에서 잡고만 있는 지속 양수 보상 제거 |
| 수평 위치 scale / deadband | 0.05m / 0.005m | 5mm까지 위치 오차 비용 0 |
| 수평/비파지 수직 속도 scale / deadband | 0.20m/s / 0.02m/s | 미세 접촉 운동 허용 |
| 각속도 scale / deadband | 1rad/s / 0.1rad/s | 회전 흔들림 허용 폭 확대 |
| 안착 자세 대비 회전 scale / deadband | 30° / 5° | 선반 안착 자세를 기준으로 비교 |
| 유효 파지 중 비용 배율 | 0.25 | 들어올리기 시작할 때 접촉 조정 허용 |
| 초기 위치·자세 벌점 | 상승 0~0.01m 구간에서 선형 감소, 그 이상은 0 | 공중에서 접촉 누락 때문에 선반 위치 오차 벌점이 재발하지 않음 |
| 운동 벌점 | 같은 높이 가중치 적용; 비파지 상태에서는 높이와 무관하게 적용 | 파지 없이 튕기거나 흔드는 행동 억제 |

각 오차/속도 크기에서 deadband를 뺀 양수 부분을 scale로 나눈 뒤 제곱한다. 위치·자세 비용 P는 변위 제곱 +
0.5×회전 오차 제곱, 운동 비용 M은 0.25×수평 속도 제곱 + 0.25×각속도 제곱 +
0.25×비파지 수직 속도 제곱이다. 각 성분을 4로 제한한 뒤 합산한다.
높이 가중치 `w = clamp(1 - max(상승 높이, 0) / 0.01, 0, 1)`에 대해
`D = P*w + M*max(w, 비파지 여부)`이다. 유효 파지 중이면 D에 0.25를 곱한 뒤
`raw = min(D, 1)`로 제한하고 weight -0.25를 적용한다. 30Hz에서 이 항은 스텝당 최저 -0.00833이다.
실제 파지 중 수직 상승에는 이 속도 벌점을 적용하지 않는다. 1cm 이상 올라간 비파지
박스에는 운동 벌점만 남고, 공중에서 정지했다고 파지/상승/성공 보상을 얻지는 않는다.
모든 보상률은 RewardManager가 제어 dt를 곱한다.

기준 위치·자세는 reset 안착 완료 시 한 번 저장한다. 관측에 기준 대비 위치·회전 7차원을
추가했고, 파지 이력 및 후보 ID도 관측한다. 전체 입력 차원은 로봇 link/관절 수와
`active_arm`에 따른 actuator 목표·이전 action 크기에 의존하므로 종전 223/227차원으로 고정하지 않는다.
안착 전에는 두 보상을 비활성화한다. 6cm 상승·0.5초 유지·기울기·속도·충돌 기준은
변경하지 않았다. 마찰값을 바꾸는 대신 현재 물리 상태에서 보상으로 유도한다.

**현재 revision 3은 후보·관측·action 계약이 변경되어 이전 checkpoint를 직접 재개/평가할 수 없다.**
아래 학습 명령에서 `--checkpoint` 없이 새 실험을 시작한다. 실행 중인 기존 프로세스에는
자동 반영되지 않는다. VR 진단은 [Quest reward 검사](RL_QUEST_REWARD_DEBUG.md)를 참고한다.


장애물 센서는 모든 로봇 rigid link(손가락 포함)에 하나씩 생성한다. 랙·펜스·버튼·컨베이어와
그 위 선점 물체만 필터링하고 작업 박스는 제외한다. 물리 4스텝의 힘 이력에서 최대 접촉을
측정한다. 현재 기본 preset은 `collision_constraints_enabled=False`로 충돌 실패와
`collision` 보상 항 전체를 비활성화한다. True로 복구하면 >20N 충돌은 첫 스텝부터 실패하며
초기 3스텝 유예와 180N 기준은 이 과제에 적용하지 않는다.
센서별 단일 link 배정은 [Isaac Lab의 filtered contact 제약](https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/sensors/contact_sensor/contact_sensor_cfg.html)을 따른다.
GPU의 filtered contact가 고정 장애물도 읽도록 랙과 펜스는 움직이지 않는 kinematic rigid body로 생성한다.
필터마다 환경당 rigid body 하나만 지정한다. 종료 규칙만으로 학습 정책의 무충돌을 보장하지는 않는다.
박스가 랙에 놓여 있는 지지 접촉 등 물리 충돌은 그대로 유지한다. 충돌 제한 OFF는 학습 전용이며 무충돌을 보장하지 않는다.

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
약 16.2cm가 되었다. 현재 기본 설정은 reset 후 0.5초 동안 동작 목표를 유지한 뒤
그 시점의 높이를 한 번 저장한다. `reset_settle_timeout=0`이면 속도 안정화 gate와
안착 타임아웃 실패를 사용하지 않는다. 따라서 그 시점에도 박스가 움직이면 기준 높이가
안착 완료 높이와 다를 수 있다. 필요하면 초기 대기 시간을 늘린다.
각 환경은 독립적으로 대기하며, 이 기간에도 장애물 힘은 측정하지만 기본 설정에서는 충돌로 실패하지 않는다.
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
| 파지 유지·접촉 누락 유예 | `rl/mdp/grasp_contact_latch.py`; 값은 `configs/rl_pick_arms_only.py` |
| 선반 위치 오차·운동 disturbance 분리 | `rl/mdp/grasp_stability.py` |
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

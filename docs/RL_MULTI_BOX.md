# 4박스 전신 RL: 단계별 학습과 전체 직접 학습

새 진입점은 `kuavo_isaaclab_scene.rl.multi_box`다. 기존 한손 RL, Quest,
VR, teleop, data collection 진입점은 이 모듈을 import하지 않는다.
공통 asset 생성·접촉 기하 계산 함수를 호출하지만 해당 함수는 변경하지 않는다.

## 범위와 기본 설정

- Isaac Sim 5.1 / Isaac Lab 2.3.2 / RSL-RL 3.1.2 PPO.
- 1단: `small_box_0`, `small_box_1`. 2단: `medium_box_0`, `large_box_0`.
- 제공된 rack-relative capture의 위치·회전·scale을 사용한다. shelf metadata가
  설정과 다르면 실행을 거부한다. 새 네 박스의 실제 배치는 GUI로 확인해야 한다.
- 카메라, factory 배경, 이동하는 다른 로봇, cargo, prefill, 버튼 task, belt 구동은 없다.
  컨베이어는 네 박스가 안정적으로 놓일 때까지 정지된 지지면이다.
- base 3 + torso 4 + 양팔 14 + head 2 + 양 gripper 2 = **25차원 action**.
- base는 기존 PlanarDrive의 root pose 적분 방식이다. 바퀴 토크·균형 제어를
  학습하는 물리적인 이동 policy는 아니다. 수동 linkage 관절도 별도 action으로 풀지 않는다.
- 양손에 서로 다른 박스를 잡는 것과 같은 박스를 양손으로 잡는 것이 모두 가능하다.
  세 박스 동시 운반을 금지하는 조건은 없지만, 손/물체 기하에 따라 가능한지는 미확인이다.
- 최초 버전은 s200062 + s200062_integrated 전용이다.

## S63 multi-box v2 grasp의 몸통 action

위 25차원 설명은 기존 4박스 경로에 해당한다. S63 `grasp-v2` PPO와
`grasp-v2-sac`는 24차원 action을 사용한다. 정책은 몸통의 로컬 앞뒤 위치와
높이 변화만 명령하고, batched IK가 `knee_joint`, `leg_joint`,
`waist_pitch_joint`를 함께 조정해 reset 때 몸통 pitch를 유지한다. 몸통
앞뒤 범위는 reset 기준 ±15 cm, 높이는 URDF nominal 기준 0~40 cm이며
명령 속도는 0.10 m/s다. base의 평면 이동과 `waist_yaw_joint`는 별도
action으로 유지한다. 충돌로 실제 몸통이 잠시 기울 수 있으므로 영상과
관절 추종 오차는 별도로 확인해야 한다.

이전 25차원 v2 checkpoint는 action/관측 차원이 달라 재개할 수 없다.
기존 Quest 성공 데모 2개는 로더가 몸통 관절 delta를 앞뒤·높이 action으로
변환하고, 독립적인 waist pitch 변화는 버린다. 따라서 새 action으로
학습한 checkpoint와 변환된 데모만 함께 사용한다.
새 보상 프로필에서는 데모의 기존 reward를 SAC critic에 섞지 않는다.
데모 관측·행동으로 actor만 초기에 모방 학습하며 비중은 20%에서 시작해
초기 SAC 업데이트 30% 동안 0까지 선형 감소한다. 그리퍼 간격 자체에는
양의 보상을 주지 않고 실제 flap 파지 이벤트에만 보상을 준다.
기본적으로 rollout 전에 데모 actor를 1000회 예열하고, 초기 탐색은
시간적으로 연속된 작은 잡음을 더한 데모 actor로 수행한다. 목표 flap 중심에서
12 cm 밖인 손의 그리퍼는 열어 둔다. 이 탐색은 충돌 없는 경로를 보증하는
플래너가 아니므로 랙·장애물 충돌과 단계별 도달률을 함께 확인해야 한다.
같은 거리 제한을 SAC 온라인 행동, critic 목표 행동, checkpoint 재생에도
적용하며, 닫기가 차단된 그리퍼 차원은 entropy 계산에서 제외한다.
기존 0=열기/1=닫기 데모 입력은 actor의 부호식 동작에 맞게 -1/+1로 변환한다.

## 실행 준비

저장소 루트에서 실행한다. 설치 환경은 [설치 문서](INSTALL.md)를 따른다.
실험마다 log와 bank 경로를 분리한다. 아래 GPU 0/1은 **서로 다른 물리 GPU**를
의미하며 각 프로세스 안에서는 `--device cuda:0`을 쓴다. GPU가 하나라면 두
프로세스가 자원을 공유하므로 env 수를 줄이거나 순차 실행한다.

```bash
conda activate env_isaaclab_232
```

이 문서의 `PICK_RUN`, `EXTRACT_RUN` 등은 실행 시 출력된 실제 run 디렉터리로
설정한다. 예시 checkpoint의 `model_1999.pt`는 실제 저장 파일명으로 바꾼다.
기존 **8-action SAC checkpoint는 이 25-action PPO 환경에 로드할 수 없다**.

## 두 실험을 병렬로 시작하기

터미널 A — 단계별 경로의 pick policy:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh staged \
  --skill pick --num-envs 256 --iterations 2000 --headless --device cuda:0 \
  --config configs/rl_multi_box.py \
  --log-dir artifacts/rl/multi_box/staged_pick_v1
```

터미널 B — 4개 전체 직접 학습:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/rl/multi_box.sh end-to-end \
  --num-envs 256 --iterations 2000 --headless --device cuda:0 \
  --config configs/rl_multi_box.py \
  --log-dir artifacts/rl/multi_box/e2e_v1
```

기본 robot pose는 layout/robot asset 초기값이다. 측정 자세로 시작하려면 pick과
전체 실험에 동일하게 `--initial-state quest_ready_02 --initial-states-file
configs/initial_states.json`을 추가한다. 이것은 초기값만 지정하며 몸을 고정하지 않는다.
한손용 자세가 낮은 선반 접근에 적절한지는 별도 확인이 필요하다.

## 단계별 학습: 실제 성공 상태를 다음 단계로 전달

학습 자체를 네 번 독립적으로 수행한다. 앞 단계 평가 중 성공 상태를 bank로
수집한 다음, 다음 단계의 `--reset-bank`에 전달한다. 실패 상태는 저장하지 않는다.
JSON bank는 root pose/velocity, joint pose/velocity, action 적분 목표값과 박스 초기
기준을 포함한다. PhysX solver/contact 캐시를 저장하는 것은 아니며 접촉은 재계산한다.

```bash
# 1. pick 평가 → 성공 상태 저장. 무작위로 네 박스 중 target을 선택한다.
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh eval \
  --strategy staged --skill pick --checkpoint "$PICK_RUN/model_1999.pt" \
  --episodes 128 --num-envs 8 --headless --device cuda:0 \
  --config configs/rl_multi_box.py \
  --snapshot-dir artifacts/rl/multi_box/banks/pick_v1

# 2. 들고 있는 상태에서 rack 밖으로 인출하기 학습
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh staged \
  --skill extract --reset-bank artifacts/rl/multi_box/banks/pick_v1 \
  --num-envs 256 --iterations 2000 --headless --device cuda:0 \
  --config configs/rl_multi_box.py --log-dir artifacts/rl/multi_box/extract_v1

# 3. extract 평가 → carry용 성공 상태
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh eval \
  --strategy staged --skill extract --checkpoint "$EXTRACT_RUN/model_1999.pt" \
  --reset-bank artifacts/rl/multi_box/banks/pick_v1 \
  --snapshot-dir artifacts/rl/multi_box/banks/extract_v1 \
  --episodes 128 --num-envs 8 --headless --device cuda:0 --config configs/rl_multi_box.py

# 4. 회전/이동하여 빈 컨베이어 슬롯 근처까지 운반하기
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh staged \
  --skill carry --reset-bank artifacts/rl/multi_box/banks/extract_v1 \
  --num-envs 256 --iterations 2000 --headless --device cuda:0 \
  --config configs/rl_multi_box.py --log-dir artifacts/rl/multi_box/carry_v1

# 5. carry 평가 → place용 성공 상태
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh eval \
  --strategy staged --skill carry --checkpoint "$CARRY_RUN/model_1999.pt" \
  --reset-bank artifacts/rl/multi_box/banks/extract_v1 \
  --snapshot-dir artifacts/rl/multi_box/banks/carry_v1 \
  --episodes 128 --num-envs 8 --headless --device cuda:0 --config configs/rl_multi_box.py

# 6. 내려놓고 손을 떼기
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh staged \
  --skill place --reset-bank artifacts/rl/multi_box/banks/carry_v1 \
  --num-envs 256 --iterations 2000 --headless --device cuda:0 \
  --config configs/rl_multi_box.py --log-dir artifacts/rl/multi_box/place_v1
```

bank는 기본 최대 128개이며, 빈 bank나 다른 환경 계약/이전 skill의 bank는
실행 전에 거부한다. 특정 박스 성공만 bank에 편중되지 않았는지 각 JSON의
`target` 0~3 분포를 확인한다. 고정된 소수 bank에서만 다음 skill을 학습하면
분포가 좁아질 수 있으므로 다양한 성공 상태를 수집한다.

## 공통 전체 평가

`skill_checkpoints.json` 파일을 아래 형태로 작성한다. 상대 경로는 JSON 위치 기준이다.

```json
{
  "pick": "/absolute/path/to/pick_run/model_1999.pt",
  "extract": "/absolute/path/to/extract_run/model_1999.pt",
  "carry": "/absolute/path/to/carry_run/model_1999.pt",
  "place": "/absolute/path/to/place_run/model_1999.pt"
}
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh eval \
  --strategy staged --skill full --skill-checkpoints skill_checkpoints.json \
  --episodes 20 --num-envs 1 --device cuda:0 --config configs/rl_multi_box.py

CUDA_VISIBLE_DEVICES=0 bash scripts/rl/multi_box.sh eval \
  --strategy end-to-end --checkpoint "$E2E_RUN/model_1999.pt" \
  --episodes 20 --num-envs 1 --device cuda:0 --config configs/rl_multi_box.py
```

`--headless`를 생략하면 GUI가 열린다. staged 전체 평가에서는 env를 리셋하지 않고
pick → extract → carry → place policy를 전환한다. 배치 후 다음 미완료 박스로
target을 바꾸고 다시 pick으로 돌아간다. end-to-end에는 강제 skill 전환이나
target 순서를 주지 않는다. 여러 박스를 동시에 들어도 물리 판정이 적용된다.

각 run의 `metrics.json`에는 success, placed, seconds, base_distance,
dual_carry_seconds가 기록된다. success_rate는 **4개 전체 완료 비율**이다.
단계별 skill 평가일 때만 해당 skill 성공률이다. 비교 시 full 평가, 동일 layout,
초기 pose, seed, episode 수, horizon을 사용한다. 학습 예산은 전체 transition 수로
맞추고 staged는 네 skill 학습량을 합산한다.

## 성공과 보상

Pick: 한 손 이상으로 대상 flap을 잡고 초기보다 6cm 상승, 기울기 40도 이내.
Extract: 파지·상승 조건과 함께 box 전체가 rack outward 경계를 벗어남.
Carry: 인출 상태에서 빈 배치 슬롯까지 25cm 이내.
Place: footprint가 벨트 안에 들어오고, box-body→belt 실제 접촉력이 0.2N 이상이며,
박스 바닥 높이도 벨트와 일치한다. 두 손이 놓고 finger tip이 박스에서 2.5cm 이상
떨어져야 한다. 선속도 0.08m/s, 각속도 0.35rad/s 미만으로 0.5초 유지한다.
전체 성공은 네 박스가 이 배치 조건을 **동시에** 만족해야 한다.

| 항목 | 기본 보상 |
|---|---:|
| 박스 최초 안정 배치 | +25 / box / episode |
| task 또는 skill 성공 | +100 |
| 박스 낙하 또는 workspace 이탈 | -50 |
| 작업 시간 | -0.2 / 초 |
| 이미 배치한 박스를 다시 불안정하게 만듦 | -2 / box / 초 |
| action 변화 / 관절 속도 | 작은 비용 |

접근·정렬·상승·인출·운반 보조 보상은 bounded potential의
`gamma * Phi(next) - Phi(previous)`다. terminal에는 potential을 0으로 처리하며
PPO gamma와 일치시킨다. action/reward는 30Hz이고 사건 보상은 manager의 dt 곱을
보정한다. 보조 보상의 정지 값은 gamma<1에서 작은 음수가 될 수 있다.
단순한 양의 상승량 누적이나 재파지 반복 보너스는 지급하지 않는다.

## 수정할 파일

모든 새 Python 파일의 기준 경로는 `src/kuavo_isaaclab_scene/rl/multi_box/`다.

| 수정할 내용 | 파일 |
|---|---|
| 실험별 박스 목록, 선반, threshold, 가중치 | `configs/rl_multi_box.py` (저장소 루트) |
| task 기본값과 유효성 검사 | `spec.py` |
| 최소 장면/센서 구성 | `env_cfg.py` |
| action 순서/scale/base 속도 | `managers/actions.py` |
| state observation | `managers/observations.py` |
| reward 항목/weight/potential | `managers/rewards.py` |
| 실패/성공/시간 제한 연결 | `managers/terminations.py` |
| reset event와 curriculum | `managers/events.py`, `managers/curriculum.py` |
| 손별·박스별 판정, 빈 슬롯, 단계 전환 | `state.py` |
| 배치 판정과 보상 중복 방지 | `kernels.py` |
| 성공 상태 저장/복원 | `reset_states.py` |
| 학습/재개/평가, checkpoint 검사 | `runner.py`, `experiments/` |

기존 잡기 threshold를 재사용하지만 별도 관리가 필요하면 `env_cfg.py`의
asset/contact용 `TaskSpec` 생성 값을 바꾼다. task 동작은 새 `MultiBoxCommand`가
담당한다. 기존 `WorkcellCommand`나 stationary pick 검증을 확장하지 않는다.

## V2 grasp SAC 접근 실패 재검토

[2026-09-30 학습 재검토](RL_V2_RECOVERY_20260930.md)에 실패 원인, 보상·SAC 설정,
데모 연결, 현재 환경에서의 IK 초기 수집, 실험 결과를 기록한다. 양손 flap 파지 및
proof-lift 성공 조건은 그대로 유지한다. 배우기 전의 IK 동작과 학습된 SAC 동작은
`rollout_policy`로 구분한다. 보상 값은 `multi_box/rewards/weights.py`,
학습 옵션은 `multi_box/experiments/train_grasp_v2_sac.py`, 데이터 혼합 및
온라인 모방은 `rl/runners/train_asymmetric_sac.py`에서 관리한다.

## 검증 범위와 다음 확인

CPU tensor 회귀 테스트와 Python/shell 구문 검사를 수행한다. 학습·Isaac Sim 실행은
이 코드 작업에 포함하지 않는다. 성공률/수렴 여부는 아직 측정하지 않았다.
먼저 작은 num-envs로 네 박스의 실제 배치, rack outward 방향, 각 단계 reset 상태의
파지 유지, 컨베이어 공간을 확인한다. outward 기본값 +local Y는 현재 rack 배치에서
robot 쪽을 향하며, rack 모델 축이 바뀌면 `rack_outward_local`을 수정해야 한다.
공유 contact geometry 함수에 대한 향후 변경은 새 실험에도 적용되므로 장기 실험은
소스·config·asset을 함께 고정한다.

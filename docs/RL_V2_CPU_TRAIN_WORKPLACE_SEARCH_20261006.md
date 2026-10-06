# CPU 물리 · fresh TRAIN에서 base 접근 위치 찾기

CPU 물리의 기존 actor 재생은 성공 사례가 있지만, fresh TRAIN256개로 추가
학습한 후에는 같은 실행의 DEV128 성적이23→12로 떨어졌다. 상단 오른쪽은
랙 충돌, 상단 왼쪽은 실제 양손 pinch 실패가 계속된다. 기본 upper 접근 위치는
왼쪽 TRAIN 성공 측정에서 가져온 후보이며 오른쪽에서도 적절하다는 증거가 없다.

다음 진단은 학습 후 악화된 actor를 승격하지 않고 CPU 학습 전 초기 actor를
고정한 채 접근 위치를 찾는다. 이전 GPU 물리의 XY/yaw 진단과 달리 이번에는
CPU PhysX의 fresh TRAIN 분포에서 네 영역을 함께 측정한다.

## 비교 방법과 범위

- 원래 sampler의 box/base/background randomization을 유지한 새 TRAIN16개:
  중간/상단·좌우 각4개. 매 사례에 동일한8개 접근 위치 후보를 적용해128 env.
- 기존 위치, 좌우9cm, 앞뒤6cm, 옆+바깥 조합, 옆+바깥+yaw±약7°를 비교한다.
  로봇의 초기 상태를 성공 자세로 바꾸지 않고, 실제 base 접근 제어의 목표만 변경한다.
- Flap은 강성1.5–2.5Nm/rad·감쇠0.15–0.25Nm·s/rad·hinge 마찰 범위에서 동적으로
  움직인다. Box를 고정하거나 episode 중 pose를 덮어쓰지 않는다.
- 후보끼리 요청된 초기 box/base/background layout은 동일하다. Flap의 물성
  추첨·contact/constructor 이력까지 bitwise로 맞춘 비교는 아니므로 작은 차이를
  base 위치만의 효과로 단정하지 않는다. 후속 fresh TRAIN에서 재확인한다.
- 성공은 양손 서로 다른 flap의 실제 pinch·안정 유지·8mm roller-clearance
  proof lift이며 rack10N·robot-only obstacle5N·self-collision OFF를 유지한다.

`--cpu-workplace-probe --base-waypoint-probe --no-training`은 CPU 물리·CPU NN의
명시적 read-only 진단이다. DEV/FINAL로 데이터를 재표기하지 않고 원래 TRAIN
seed를 유지한다. 학습 update·replay·성공 bank·measured credit bank를 추가하지
않으며 전체 network/normalizer tensor와 counter 불변을 종료 시 검사한다.
변경된 제어기에서 나온 진단 transition은 SAC의 matching Q/replay로 가져오지 않는다.

이128개는 **16사례×8후보의 탐색**이지 독립128개 일반화 평가가 아니다.
초기 무효와 충돌/timeout도 그대로 남긴다. 적절한 후보를 얻으면 새 제어 계약과
fresh Q/replay에서 SAC 학습을 하고 원래 전체 DEV128, 이후 독립 FINAL로 평가한다.
Base 접근은 측정된 목표로 이동한 뒤 유지하는 방식이며 학습된 자율 주행이 아니다.
상단 좌우 및 다양한 base/box 초기 상태의 성공은 아직 달성하지 않았다.

## 실행

```bash
conda activate env_isaaclab_232
CUDA_VISIBLE_DEVICES=0 python scripts/rl/batched_staged_goal_with_drive.py \
  --gpu 0 --physics-device cpu \
  --experiment-dir /absolute/path/to/unique-workplace-run \
  --checkpoint /absolute/path/to/CPU-initial-actor-checkpoint.pt \
  --training-manifest /absolute/path/to/CPU-training_manifest.json \
  --waves-json /absolute/path/to/fresh-TRAIN16-times8-candidates.json \
  --waypoints docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 \
  --native-seed /absolute/path/to/closed-middle-TRAIN-transitions.hdf5 \
  --native-seed /absolute/path/to/closed-upper-TRAIN-transitions.hdf5 \
  --no-training --cpu-workplace-probe --base-waypoint-probe --steps 900 \
  --eval-video-env-indices 0 32 64 96 97 101
```

GPU0은 Kit renderer 격리용이다. CPU 물리와 CPU 정책 추론을 사용하며 다른
GPU의 기존 사용자 프로세스는 유지한다. 출력은 소유자 전용 RAM 폴더에 두고
기존 Drive 연결로 닫힌 checkpoint/input 및 종료 후 로그·HDF·영상을 크기/MD5
검증한다. 인증이나 공유 권한을 변경하지 않는다.

관련 범위 검사52개 통과, 기존 CUDA integration1개는 이 CPU 진단 검사에서
skip했다. 실제 물리 성공은 테스트 통과만으로 판단하지 않고 종료 측정으로 판정한다.

[이전 실제 CPU/GPU 평가·학습 결과·영상](RL_V2_CPU_GPU_INTERIM_REPORT_20261006.md).

## 실행 중 확인한 근거

GPU3의 기존 장기 SAC는 새 TRAIN1536개와 전체 DEV5회를 모두 수행하고 exit0으로
종료했다. DEV 성공은9→4→3→2→4/128이었고 마지막 중간왼쪽1·중간오른쪽3·
상단0/0, unsafe84·timeout12·초기 무효28이었다. Actor5715/Q24906까지의 학습은
일반화 개선으로 이어지지 않았다. 종료 백업은 원래 supervisor가 처리한다.

별도로 종료된 CPU 학습의 실제 성공 TRAIN bank22개 경로에서 학습 전/후
actor를 동일한 관측에 적용했다. 마지막16개 상태의 양손 닫힘은 중간좌우 모두
144→144, 상단왼쪽61→64였다. 이 성공 상태들에서는 그리퍼가 파지를 잊거나
actor normalizer가 변했다는 증거가 없었다. 평균 body goal 변화는 normalized
좌표에서 중간좌0.00338·중간우0.00378·상단좌0.00611이었다. 이 수치는 meter나
radian이 아니며 작은 출력 차이도 폐루프 동작/충돌을 바꿀 수 있다.

학습된 Q는 같은 실제 TRAIN 상태에서 대체로 학습 후 actor의 행동을 더 높게
평가했다. Q 선호는 실제 성공 또는 그 원인의 증명이 아니다. 상단오른쪽 성공
TRAIN 경로가 bank에 없어 이 영역의 성공 동작 보존을 대조할 수 없었다.
[실제 CPU TRAIN actor/Q 진단](assets/rl_v2_CPU_TRAIN_actor_regression_audit_20261006.json).

새16사례×8접근위치의 CPU 진단은 GPU0 renderer 격리로 시작했다. 실제 writer의
소유자·실행 폴더·CUDA_VISIBLE_DEVICES=0을 확인했고 첫 rollout31steps에서
actor/Q update0·replay0·training=false였다. 입력2개와 별도 receipt는 기존
Drive 크기/MD5 검증을 마쳤다. 아직 닫힌 전체 진단 결과가 아니므로 좋은
접근 위치를 찾았거나 학습에 성공했다고 판단하지 않는다.

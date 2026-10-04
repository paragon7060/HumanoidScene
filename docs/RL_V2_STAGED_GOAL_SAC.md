# Base 접근·정지 후 남은 21개 목표 SAC

기존 whole-body SAC의 base XY/yaw와 파지 동작을 분리하는 선택적 실험이다.
중간·위 선반의 좌우 작업 위치까지 중립 팔·열린 gripper로 **실제 이동**하고,
정지 조건을 확인한 뒤 base 목표를 유지하면서 남은 관절 목표를 SAC로 학습한다.
박스는 동적이고 초기 base XY/yaw·박스·주변 박스 randomization을 유지한다.
성공/보상/충돌 기준을 완화하거나 성공 순간의 팔 자세로 reset하지 않는다.

이 실험의 [현재 물리 결과·영상](RL_V2_FOUR_REGION_SAC_20261004.md)은 별도로 갱신한다.
실행 코드나 서비스의 존재가 학습 완료 또는 일반화 성공을 뜻하지 않는다.

## 제어와 데이터 계약

- 기존 `pose-goal` 24개 목표에서 base XY/yaw 3개를 제거한 **21개 목표**다.
  Upper-body/head 17개, upright torso X/Z 2개, 양손 gripper 2개를 학습한다.
  물리 환경의 24차원 command와 기존 upright controller는 유지한다.
- Actor 480차원/critic 539차원이다. 기존 목표 관측에 held phase,
  rack 기준 held X/Y, yaw sin/cos, 정책 목표 허용 반경 6개를 추가한다.
- 실제 XY 8mm/yaw 0.02rad 이내, 선속도 0.01m/s·각속도 0.025rad/s 미만을
  15 control step 연속 확인한 뒤에만 새 replay/Q에 넣는다.
  Base 접근 중 실패도 전체 task 실패로 기록하지만 held-phase replay에는 넣지 않는다.
- 기존 24-action Q/replay/optimizer를 가져오지 않는다. 기존 학습 actor의
  팔·torso 목표 평균/normalizer만 초기화에 사용한다. 새 critic과 실제 새 replay를 쓴다.
  원본 native 성공 파일은 기존 모델의 물리 계약 audit 용도이며 새 Q seed가 아니다.
- Initial critic 2,048회는 actor 업데이트를 보류한다. 이후 critic 4회당 actor 1회,
  actor LR `1e-6`, Gaussian std `0.005` 초기/`0.001..0.02` 범위다.
  실제 held step당 critic 2회, replay 20,000행, batch 256, discount 0.999다.
- 학습 초반 frozen neural prior actor loss는 2부터 0까지 20,000 critic 진행량에
  따라 감소한다. 팔·torso goal 허용 반경은 normalized 0.05→0.15다.
  Live VR 경로, 실시간 IK teacher, simulator pinch 정보는 actor 실행에 쓰지 않는다.

기본 비교 설정은 gripper도 prior 반경에 둔다. `--free-grippers` 준비 옵션은
gripper의 prior 반경 제한만 해제한다. `--gripper-logit-scale 0.005`는 이전 actor의
gripper logit을 부드럽게 하여 닫기/열기를 탐색할 수 있게 한다. 물리 gripper는
여전히 action의 **부호**로 여닫는 binary force controller다. 작은 출력이 작은
닫힘 힘을 뜻하지 않는다. 각 손이 flap에서 12cm 이상 멀면 닫기를 차단한다.
초기 팔·torso 목표와 jaw의 열림/닫힘 결정을 원본과 대조하고 실험별 계약에 기록한다.

## 새 실험 준비

Conda `env_isaaclab_232`를 사용한다. 기존 `pose-goal` checkpoint, matching 현재 물리
native middle/upper 성공 파일, 작업 위치 JSON이 필요하다. 현재 공개 waypoint는
small box·torso extra travel 0.06m 전용이다. 다른 크기는 측정 없이 허용하지 않는다.
준비는 CPU에서 실행하며 optimizer 업데이트나 성공 데이터를 생성하지 않는다.

```bash
conda activate env_isaaclab_232
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  python scripts/rl/prepare_staged_goal_sac.py \
  --checkpoint /absolute/path/to/matching-pose-goal-checkpoint.pt \
  --native-seed /absolute/path/to/current-middle-train-success.hdf5 \
  --native-seed /absolute/path/to/current-upper-train-success.hdf5 \
  --waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --output-dir /absolute/path/to/unique-staged-initial-model \
  --free-grippers --gripper-logit-scale 0.005
```

`checkpoint_00000000.pt`와 0행 `staged_goal_experience.pt`가 만들어진다.
병렬 수집용 준비에는 `--replay-capacity 100000` 등으로 buffer를 늘릴 수 있다.
1024..2,000,000행 범위이며 기본20,000행의 이전 계약은 그대로 지원한다.
용량이 다른 새 모델은 명시적 계약에 저장되고 재개할 때 그 용량을 복원한다.
기존 다른 제어의 Q/replay를 가져오는 옵션은 아니다.
실험을 계속할 때는 checkpoint와 **같은 종료 실행의 실제 experience 파일**이 함께 필요하다.
다른 제어의 replay나 임의로 변경한 waypoint/물리 계약으로 재개하면 거부한다.

## GPU3 학습·개발 평가·독립 final

[배치 생성 안내](RL_V2_FOUR_REGION_SAC_20261004.md)에 따라 서로 분리한
TRAIN/development/final JSON과 `reference_episode_map.json`을 준비한다.
아래 관리자에서 개발 성능이 어느 구역이든 떨어지면 마지막 실제 Q/replay/진행량은
유지하고 actor만 이전 개발 checkpoint로 복구한다. Final은 optimizer 없이 실행하고
해당 경로를 후속 학습에 섞지 않는다.

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:scripts/rl \
  python scripts/rl/layout_residual_with_drive.py \
  --gpu 3 --policy-mode staged-goal \
  --experiment-dir /absolute/path/to/unique-staged-training-run \
  --layout-dir /absolute/path/to/unique-layouts \
  --reference-episode-map /absolute/path/to/unique-layouts/reference_episode_map.json \
  --layout-distribution initial-base-and-box --max-layout-depth-m 0.006 \
  --checkpoint /absolute/path/to/unique-staged-initial-model/checkpoint_00000000.pt \
  --train-count 8 --passes 4 --eval-count 8 \
  --validation-layout-dir /absolute/path/to/unique-layouts/development \
  --validation-every 4 --validation-count 4 --allowed-validation-success-drop 0 \
  --staged-base-waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --pose-student-native-seed /absolute/path/to/current-middle-train-success.hdf5 \
  --pose-student-native-seed /absolute/path/to/current-upper-train-success.hdf5 \
  --demo-dataset /absolute/path/to/reset-scenes.hdf5 \
  --training-manifest /absolute/path/to/matching-physical-manifest.json \
  --torso-extra-height-m 0.06 --steps 900 --capture-every 90 --contact-diagnostics \
  --kit_args '--/renderer/activeGpu=3 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'
```

Demo는 초기 장면과 출처 확인에 사용하며 실행 중 동작 경로를 공급하지 않는다.
독립 재생은 `replay_v2_grasp_reference.py`의 matching checkpoint/native/layout 인자에
`--staged-goal-sac --no-staged-goal-training --staged-base-waypoints ...`를 지정한다.
일반 delta SAC runner 또는 예전 `--pose-student-training`으로 실행하지 않는다.

[기존 Drive 설정](RL_GOOGLE_DRIVE.md)을 먼저 읽는다. 새 인증 없이 host-local remote를
발견하며 300초마다 checkpoint checksum을 검증하고 검증된 오래된 파일만 정리한다.
최근 2개와 최근 검증된 2개를 보호한다. Experience/HDF5/영상/로그는 writer 종료 후
업로드·검증한다. Runtime/backup 오류는 배치를 중단하며 원인 확인 없이 재시작하지 않는다.

진행은 부모 `status.json`, 개별 실행 `metrics.json`의 `staged_goal_sac`로 확인한다.
Checkpoint 번호는 **critic 업데이트 횟수**이고 actor 횟수는 별도다. Nested frozen
warm-start의 오래된 actor16910 같은 값은 새 SAC의 학습량이 아니다.
처음 critic-only 배치의 성공도 새 actor 학습 개선으로 해석하지 않는다.

## 여러 초기 배치의 병렬 wave 실행 (실험 경로)

`train_batched_staged_goal.py`는 환경마다 실제 base 정지 시점, box anchor,
held waypoint, 경과 시간을 따로 유지하며 하나의 SAC를 공유한다. 종료한 환경은
다음 중립 whole-wave reset까지 replay/normalizer에서 제외한다. Auto-reset 뒤의
관측을 종료 직전 next observation으로 쓰지 않는다. Approach 전이는 새21-goal Q에서
제외하고 실제 task 성공/실패는 모두 집계한다. Validation/final wave는 optimizer와
replay가 바뀌지 않았는지 검사한다. Live VR/IK teacher나 성공 자세 reset은 없다.

`waves.json`은 아래 형태다. 각 wave의 환경 수가 같아야 한다. `layout`은 기존
GraspLayout JSON 전체이며 `episode_index`는 **중립 초기 장면**의 원 demo 출처다.
개발 wave의 이름은 `validation`, layout 내부 split은 `holdout`이다. Train과
development/final seed가 겹치면 거부한다. 원본 demo의 reward/transition을 재사용하지 않는다.

```json
[
  {"split":"train", "layouts":[
    {"episode_index":0, "layout":{"seed":55000,"split":"train","lateral_m":-0.025}}
  ]},
  {"split":"validation", "layouts":[
    {"episode_index":0, "layout":{"seed":56000,"split":"holdout","lateral_m":-0.03}}
  ]}
]
```

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:scripts/rl \
  python scripts/rl/batched_staged_goal_with_drive.py \
  --gpu 3 --experiment-dir /absolute/path/to/unique-batched-run \
  --checkpoint /absolute/path/to/staged-checkpoint.pt \
  --native-seed /absolute/path/to/current-middle-train-success.hdf5 \
  --native-seed /absolute/path/to/current-upper-train-success.hdf5 \
  --demo-dataset /absolute/path/to/reset-scenes.hdf5 \
  --training-manifest /absolute/path/to/matching-physical-manifest.json \
  --waypoints /absolute/path/to/waypoints.json \
  --waves-json /absolute/path/to/waves.json --training
```

기존 host-local Drive remote를 발견하고 CUDA/Kit renderer를 같은 물리 장치로
격리한다. GPU3 자원이 부족하면 우리 실험만 별도 장치에 명시적으로 실행하며
다른 사용자의 프로세스를 중단하지 않는다. `--no-training`에는 TRAIN wave를 넣을 수 없다.
기존 5분 백업/검증 보존/종료 로그 검증을 그대로 쓴다. Periodic checkpoint와
종료 때 저장하는 실제 experience의 복구 가능 시점은 다르므로 실행 중 replay까지
영구 보관한다고 보장하지 않는다. Runtime 실패 traceback은 Kit 종료 전에 저장한다.

2026-10-04 첫4-env 실행은 per-environment action scale을 flatten하던 검사가
배열 shape를 거부했다. 모든 환경의 실제 scale을 각각 검증하도록 고쳤다.
다음 물리 실행은 초기 배치 guard에서 중단됐으며 해당 전이는 Q에 넣지 않았다.
환경별 reset/pose 차이를 기록하는 진단으로 이어간다. **이 문서는 코드 사용법이며
병렬 물리 성공이나 SAC 개선 완료의 증거가 아니다.** 최신 실측은 연결된 결과 문서를 본다.

후속 진단은 박스 변경/invalid reset 없이 rack–base 상대 높이가3cm 달랐다는 것을
확인했다. 단일 runner처럼 최초 reset의 물리 settling을 거친 후 기준 pose를 잡도록
수정한4-env 실행은 전체 scene guard를 통과했고 실제 base hold/정책 실행까지 진행했다.
아직 완료 성공률 전이다. Closed evaluation wave는 같은 critic counter의 checkpoint를
다시 쓰지 않아 이미 Drive에서 검증된 파일 이름을 다른 내용으로 덮어쓰지 않는다.

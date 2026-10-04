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

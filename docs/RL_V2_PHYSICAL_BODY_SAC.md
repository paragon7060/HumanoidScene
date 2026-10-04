# V2: 실제 제어명령 좌표의 SAC

2026-10-05 GPU0 비교 실험이다. 기존 GPU3 goal SAC의 성공 유지 분기는 같은
DEV128에서 `4 → 3 → 3 → 7` 성공을 기록했다. 마지막 결과는 중간오른쪽5,
중간왼쪽1, 상단왼쪽1, 상단오른쪽0이다. 일부 개선은 있지만 네 영역의 안정적인
일반화 성공은 아직 아니다. 팔 탐색 분기는 `3 → 1/128`로 개선되지 않았다.

![DEV 변화와 실제 TRAIN 데이터](assets/rl_v2_physical_body_sac_initial_20261005.png)

[측정·출처·체크섬·실행 증거](assets/rl_v2_physical_body_sac_initial_20261005.json).
기존 두 GPU3 실행은 유지했고 다른 사용자의 GPU1/2 프로세스도 그대로 두었다.

## 변경 이유와 데이터의 의미

기존 SAC의 Q 입력은 요청한 관절/torso 목표21이었다. Servo 속도와 명령이
포화되면 서로 다른 목표가 같은 실제 동작을 만들 수 있다. 원래 TRAIN 성공
상단오른쪽 경로의 body명령 원소7.8%, 상단왼쪽7.0%가 포화됐다. 이것은 학습
표현을 점검할 근거이며 모든 실패의 단독 원인으로 확인된 것은 아니다.

새 분기에서는 Q에 **실제로 실행되는 정규화된 물리 명령21**을 넣는다. 원래
전역 action24와 servo는 유지하고 base3을 제외한 열만 선택한다.

| Q 행동 | 전역 action 열 | 수 |
|---|---|---:|
| waist/양팔/head 관절 명령 | 3..17,22,23 | 17 |
| torso X/Z 명령 | 18,19 | 2 |
| 양손 jaw 열기/닫기 | 20,21 | 2 |

Base는 측정된 초기 box 위치의 선반별 waypoint로 neutral 접근한 뒤 15tick
안정 상태에서 유지한다. SAC가 base 이동까지 학습했다고 해석하지 않는다.

원래 완료된 TRAIN 성공13episode에서 literal held 전이5,737행을 복구했다.
중간오른쪽9episode/3,716행, 중간왼쪽2/820, 상단오른쪽1/603,
상단왼쪽1/598이다. 원본 action24와 pre/next actor/critic/reward/종료,
실제 양손 pinch/stable/hold/lift/safety를 검사하고 base feedback도 대조했다.
목표를 역산하거나 원본 goal Q/optimizer를 가져오지 않는다. DEV/FINAL,
변경한 물리·waypoint probe, 불완전·불안전 기록은 importer가 거부한다.
이 데이터 복원은 새 정책의 물리 성공을 뜻하지 않는다.

기존 Quest demo2와 그에 연결된 neural prior는 frozen 기반 정책으로 남는다.
학습 중 VR 궤적이나 IK teacher를 실행하지 않는다. Actor4865의 goal 정책은
frozen physical command prior로 변환하고, 새 actor는 그 명령에 residual을
더한다. Q/critic normalizer/optimizer/counter는 모두 새로 초기화한다.

## 학습 설정과 비교

| 설정 | 새 분기 |
|---|---|
| actor/critic 관측 | 480/539, held 문맥 마지막 값은 physical residual gain0.5 |
| 연속 행동 | 19차원, `clip(frozen_command + 0.5*tanh(residual), -1, 1)` |
| jaw | 근접12cm gate, 두 Bernoulli와 네 조합의 Q 기대값 |
| Actor LR | 3e-5 |
| 탐색 | AR1 rho0.98, latent Gaussian std초기0.05/최소0.02/최대0.1 |
| Q 준비 | critic 1024update 후 actor를 4critic update마다 갱신 |
| 일반 replay | 100,000행; 별도 physical 형식으로 보관 |
| 성공 데이터 | 영역별 균형 샘플, Q배치20%→5%/5000actor update |
| 성공 imitation | projected 물리 명령 MSE1.0, jaw NLL0.05 |
| 기준 정책 유지 | residual0 MSE0.2→0/5000actor update |

초기 greedy physical 명령이 frozen prior와 같은지 실제5737행에서 확인했다.
Source actor의 원래 radius0.05 문맥과 새 actor의 gain0.5 문맥은 별도로 변환하며
normalization을 보존한다. CPU restore와 native provenance/명령/학습/재개/
frozen 평가 테스트51개가 통과했다. 이는 simulator 일반화 검증을 대체하지 않는다.

S63/Leju2finger/중력보상/upright torso/PGS/원래 그리퍼 구동/동적 box/보상과
안전 기준은 같다. 양손 opposing flap pinch, 안정 hold0.25s,
corrected proof-lift8mm를 유지한다. Robot-rack10N, robot-only obstacle5N,
self-collisionOFF이며 curriculum을 추가하지 않는다.

GPU0은 `CUDA_VISIBLE_DEVICES=0`, 내부 `cuda:0`, Kit active GPU0/single GPU,
128env로 시작했다. Canonical TRAIN과 repeated DEV는 같은 배치를 쓰고,
FINAL은 실행 전에 새 seed namespace2026100500의128case를 별도 생성했다.
이 FINAL은 TRAIN/DEV와 겹치지 않고 정책 선택에 사용하지 않는다.
반복 DEV의 작은 상승을 독립 검증 성공이라고 부르지 않는다.

실행 부모는 `artifacts/rl/drive_runs/physical_body_SAC_pgs128_gpu0_20261005_063724`,
실제 child는 `batch_sac_20261005_063739_dc72fe`다. 현재 상태는 부모의
`status.json`, child의 `progress.json`/`metrics.json`과 실제 PID로 판단한다.
초기 source actor/새 checkpoint/replay/선언한 wave 입력을 기존 Drive로
크기·MD5 검증했다. 관리자는300초 검사, 최신 두 checkpoint와 검증된 두 개
보호, writer 종료 후 로그·HDF·physical replay 최종 검증을 수행한다.

## 재현 가능한 초기화와 실행

원본 성공 corpus와 선택한 frozen actor 입력은 완료된 immutable 파일을 사용한다.
새 형식과 기존 goal SAC 형식은 artifact/algorithm/replay/성공 bank 계약으로
구분하고 잘못된 재개를 거부한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  /home/seonho/miniconda3/envs/env_isaaclab_232/bin/python \
  scripts/rl/prepare_physical_body_sac.py \
  --frozen-actor /absolute/closed-input/frozen_actor_inputs.pt \
  --native-successes /absolute/closed-corpus/actual_train_successes.hdf5 \
  --success-outcomes /absolute/closed-corpus/outcomes.json \
  --source-run original_TRAIN_run_basename \
  --native-seed /absolute/lower-calibration/executed_transitions.hdf5 \
  --native-seed /absolute/upper-calibration/executed_transitions.hdf5 \
  --training-manifest /absolute/active_support_pgs_training_input.json \
  --waypoints docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --output-dir /absolute/new-physical-body-initial

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  /home/seonho/miniconda3/envs/env_isaaclab_232/bin/python \
  scripts/rl/batched_staged_goal_with_drive.py \
  --experiment-dir /absolute/unique-physical-body-experiment --gpu 0 \
  --checkpoint /absolute/new-physical-body-initial/checkpoint_00000000.pt \
  --waves-json /absolute/predeclared-TRAIN-DEV-new-FINAL/waves.json \
  --waypoints docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 \
  --training-manifest /absolute/active_support_pgs_training_input.json \
  --native-seed /absolute/lower-calibration/executed_transitions.hdf5 \
  --native-seed /absolute/upper-calibration/executed_transitions.hdf5 \
  --training --stop-on-validation-regression \
  --validation-regression-significance 0.05
```

Remote는 [Google Drive 관리](RL_GOOGLE_DRIVE.md)의 기존 연결을 재사용한다.
계정별 alias/인증은 Git에 기록하지 않는다. 일반 checkpoint uploader만으로는
종료된 HDF를 처리하지 않으므로 위 관리 entrypoint를 사용한다.

## 디스크 정리 기록과 복원

종료·Drive 검증된 원본 run `batch_sac_20261004_214747_7e35c4`의
full `executed_transitions.hdf5` 5,956,363,796bytes를 다시 검증하고 로컬에서
정리했다. 성공13episode의 전체 literal fields/초기state를 담은27MB corpus는
로컬과 Drive에 유지한다. 원본checkpoint/replay/로그, 실행 중인 파일,
다른 사용자의 파일/프로세스는 보존했다. 기록은 원본run의
`closed_hdf_offload.json`과 위 증거JSON에 있다.

Full HDF가 필요하면 기존 remote를 환경변수로 지정하고 원래run folder에
`gdrive.sh copyto`로 내려받아 정리 기록의 size/MD5와 대조한다.
이 정리는 checkpoint retention과 별개이며 종료된 우리 원본 파일에만 적용했다.

# V2: 실제 제어명령 좌표의 SAC

2026-10-05 GPU0 비교 실험이다. 기존 GPU3 goal SAC의 성공 유지 분기는 같은
DEV128에서 `4 → 3 → 3 → 7` 성공을 기록했다. 마지막 결과는 중간오른쪽5,
중간왼쪽1, 상단왼쪽1, 상단오른쪽0이다. 일부 개선은 있지만 네 영역의 안정적인
일반화 성공은 아직 아니다. 팔 탐색 분기는 `3 → 1/128`로 개선되지 않았다.

![DEV 변화와 실제 TRAIN 데이터](assets/rl_v2_physical_body_sac_initial_20261005.png)

[측정·출처·체크섬·실행 증거](assets/rl_v2_physical_body_sac_initial_20261005.json).
기존 두 GPU3 실행은 유지했고 다른 사용자의 GPU1/2 프로세스도 그대로 두었다.

07:20 후속으로 GPU0 첫 frozen DEV는4/128(중간오른쪽3/상단왼쪽1)이었다.
Actor/Q 업데이트0의 결과이므로 학습 개선은 아니다. 같은 source actor의
기존DEV7/128을 그대로 재현하지 못했고, 물리 재현의 민감성도 남아 있다.
실제 동일 CPU 관측5737행에서 원본 goal controller와 새 prior의 body/jaw
명령은 bitwise 같고 base법칙 차이는1e-5 미만임을 별도로 대조했다.

첫 TRAIN은 실제43202행/Q925까지 진행하다 `Base-plane projection is singular
or near vertical`로 종료됐다. Recorded replay의 singular next pose는1행이고
실제로 terminated였다. 현재 actor pose가 singular인 행은0이었다.
Q bootstrap을 종료행에서도 계산하다 불필요한 base solver가 실패한 것이다.
종료행의 bootstrap을0으로 바로 처리하고 body19/jaw2 계산에서 base solver를
분리했다. Live full command의 singular-base 검사는 유지한다. 실제 실패 replay를
포함한 Q/actor 업데이트가 유한했고 관련55개 테스트가 통과했다.

수정한 GPU0 재개 부모는
`physical_body_SAC_terminal_fixed_resume_pgs128_gpu0_20261005_071955`이다.
실패 writer/manager 종료와 Drive model/replay checksum을 확인하고 physical
Q925·optimizer·실제43202행을 이어간다. CPU 진단에서 수행한 업데이트는
재개 checkpoint에 저장하지 않았다. 아래 full-range 비교도 별도로 시작했다.

![실제 종료행 오류와 행동 범위](assets/rl_v2_physical_body_terminal_fix_and_support_20261005.png)

[실패·수정·행동 범위·재개 증거](assets/rl_v2_physical_body_terminal_fix_and_support_20261005.json).

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


## 성공 명령 전체를 표현하는 비교 옵션

`prepare_physical_body_sac.py --residual-gain 2`는 같은 frozen 기반 명령과
literal Q를 사용하면서 보정 가능 범위를 넓힌다. Gain0.5에서는 원본 중간왼쪽
성공 명령 원소21.1%가 policy범위 밖에 있었고, gain2에서는 네 영역5737행의
모든body명령이 범위 안에 있다. 상단오른쪽은 gain0.5에서도 범위 밖0%였으므로
이 제약을 상단오른쪽 실패 원인이라고 단정하지 않는다.

| 설정 | gain0.5 | gain2 |
|---|---:|---:|
| latent Gaussian std초기 | 0.05 | 0.0125 |
| latent std최소/최대 | 0.02/0.1 | 0.005/0.025 |
| 초기 physical std의1차 근사 | 0.025 | 0.025 |
| actor LR | 3e-5 | 7.5e-6 |
| residual0 penalty초기 | 0.2 | 3.2 |

Std와LR은gain의역수, residual0 MSE계수는gain제곱으로 조정해 초기 작은 탐색과
physical 단위의 기준 정책 유지 강도를 맞춘다. Tanh/clipping 때문에 전체
분포가 완전히 동일하다는 주장은 하지 않는다. Actor/Q/optimizer는 별도로
초기화하고 성공body imitation은 계속 실제 projected명령을 비교한다.

Full-range 부모는 `physical_body_SAC_fullrange_pgs128_gpu0_20261005_072246`이다.
GPU0/CUDA_VISIBLE_DEVICES=0/128env이며 초기전체 checkpoint/replay/audit/waves를
먼저 기존Drive로검증했다. TRAIN/DEV는 같고FINAL은 새namespace2026104500이다.
Gain0.5재개와 기존GPU3두분기,다른사용자프로세스는 유지한다. 아직새학습의
개선결과는 없으며구역별DEV로효과를판단한다.

07:10에는 종료된 원본alignedrun의4,120,512,084bytes goal replay도 기존Drive
size/MD5와writer종료를다시확인하고로컬에서정리했다. 원본최신2checkpoint/
로그,13성공literal corpus,현재physical replay는보존했다. 원본run의
`closed_replay_offload.json`에복원용size/MD5/Drive상대경로를기록했다.

## 10/05 08:04 · 실제 업데이트 시작과 frozen 영상 재생

수정한 gain0.5 재개는 첫 TRAIN에서 actor55 / critic1241 업데이트와
새 실제 전이8,908행까지 진행했다. 원래 실패한 critic925를 넘어섰고,
실제 actor 업데이트에서도 Q/actor 손실이 유한했다. 다음 학습 후 DEV는
아직 나오지 않았다. 아래 초기 DEV는 actor 업데이트0의 결과다.

| 실행 | 최근 완료 DEV | 중간왼쪽 | 중간오른쪽 | 상단왼쪽 | 상단오른쪽 |
|---|---:|---:|---:|---:|---:|
| GPU3 기존 goal SAC 성공 유지 | 4/128 | 1/32 | 3/32 | 0/32 | 0/32 |
| GPU3 episode 팔 탐색 | 4/128 | 0/32 | 3/32 | 1/32 | 0/32 |
| GPU0 physical gain0.5 초기 재평가 | 7/128 | 2/32 | 4/32 | 1/32 | 0/32 |
| GPU0 physical gain2 초기 평가 | 7/128 | 2/32 | 4/32 | 0/32 | 1/32 |

GPU3의 반복 DEV는 각각 `4 → 3 → 3 → 7 → 4`와 `3 → 1 → 4`다.
일관된 학습 개선은 아직 없다. GPU0의 두 초기 actor는 같은 frozen 기반
명령을 사용하므로, 초기 상단 성공 위치가 달라진 것을 gain 효과나 학습
효과로 해석하지 않는다. Reset에서 부적합하거나 바뀐 배치는 실패 분모에
남기고 Q에서 제외한다. 초기 gain0.5의 invalid reset도30/128이었다.
일부 원래 배경 box가 선반 밖으로 떨어지거나 충분히 안정되지 않은 것이
기록되어 있다. 최종 guard의 상태는 respawn 이전 고장 궤적을 대신하지 않는다.

![첫 실제 업데이트와 반복 DEV](assets/rl_v2_physical_body_first_updates_20261005.png)

[실제 전이·완료된 DEV·frozen 재생·디스크 검증 증거](assets/rl_v2_physical_body_first_updates_20261005.json).
Randomized base/동적 box/네 영역/보상/안전/성공 조건은 유지한다.
네 영역의 안정적인 일반화와 독립 FINAL 성공은 아직 확인되지 않았다.

학습 runner와 단일 영상 runner는 `experiments/staged_policy.py`에서 같은
artifact dispatch를 사용한다. Physical checkpoint는 body21의 명령과
`physical_body_contract`를 복원한다. 기존 goal checkpoint는 기존 형식을
유지한다. Physical checkpoint로 `--collect-train-goals`를 사용하면 거부하며,
물리 명령을 요청한 goal21로 잘못 기록하지 않는다. Frozen 재생은 실제
전후 actor/critic/replay counter가 같아야 완료되고, 영상용 HDF는 Q에 넣지 않는다.

관련57개 테스트가 통과했다. 두 gain의 실제 기록된 TRAIN 상태5,737행씩에서
frozen 평가 dispatch의 실제 body 명령과 Q action이 bitwise 같고, 초기 명령이
frozen prior와 같으며, RNG/actor/Q/replay가 바뀌지 않는 것도 확인했다.
이 CPU 검사는 simulator 성공이나 학습 개선을 뜻하지 않는다.

단일 영상은 기존 entrypoint의 `--staged-goal-sac`로도 physical artifact를
자동 구분한다. 평가 입력 manifest는 학습과 같은 설정에서 action contract만
기본 upright torso로 돌린 파일이다. Entry point가 `--torso-extra-height-m .06`을
적용한 결과가 실제 학습 contract와 정확히 같아야 한다. 이미 +6cm인 manifest에
다시 +6cm를 적용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:scripts/rl \
  /home/seonho/miniconda3/envs/env_isaaclab_232/bin/python \
  scripts/rl/replay_v2_grasp_reference.py \
  --pose-student-checkpoint /absolute/immutable/physical-checkpoint.pt \
  --no-pose-student-training --staged-goal-sac --no-staged-goal-training \
  --staged-base-waypoints docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --pose-student-native-seed /absolute/lower-calibration/executed_transitions.hdf5 \
  --pose-student-native-seed /absolute/upper-calibration/executed_transitions.hdf5 \
  --training-manifest /absolute/closed-input/baseline_entrypoint_manifest.json \
  --layout-json /absolute/declared-layout.json \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 --episode-index 1 \
  --torso-extra-height-m .06 --steps 900 --capture-every 6 --contact-diagnostics \
  --output-dir /absolute/new-unique-video-run --device cuda:0 --headless \
  --kit_args '--/renderer/activeGpu=2 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'
```

Episode0은 중간, episode1은 상단의 초기 장면을 복원하는 용도다. 평가 행동은
checkpoint에서 나오며 live VR/IK teacher를 실행하지 않는다. 기존 DEV에서
성공한 상단오른쪽 seed121306은 GPU2 단일 환경으로 영상 재생 중이다.
선택한 DEV 장면의 시각적 재현이며 새로운 FINAL이나 학습 개선으로 세지 않는다.
종료된 `policy.mp4`는 H.264 브라우저 형식으로 변환하고 기존 Drive로
영상·HDF·로그를 크기/MD5 검증한다.

### 종료된 초기화 replay의 추가 정리

실행 중인 GPU3 두 작업과 별개의 완료된 초기화 폴더에서
`staged_goal_experience.pt` 두 개, 총7,246,092,722bytes(약6.75GiB)를
Drive 크기·MD5와 다른 독자가 없음을 다시 확인한 뒤 로컬에서 정리했다.
현재 GPU3의 실제 PID가 살아 있고 각자 최근 checkpoint와 자체 replay를
갖고 있는지 확인했다. 초기화/현재 checkpoint, 실제 TRAIN 성공 corpus,
활성 HDF/로그와 다른 사용자의 파일·프로세스는 그대로 둔다.

각 초기화 폴더의 `closed_replay_offload.json`에 복원용 size/MD5를 기록하고
그 기록도 Drive 검증했다. 초기 checkpoint에서 다시 시작하려면 이 replay를
먼저 같은 폴더에 다운로드하고 검증해야 한다. 현재 실행의 최신 checkpoint로
이어갈 때는 현재 실행 폴더의 자체 replay를 사용한다.
기존6GB HDF정리와합쳐약10GB를회수했고현재여유약18GiB다.

# Base 접근·정지 후 남은 21개 목표 SAC

기존 whole-body SAC의 base XY/yaw와 파지 동작을 분리하는 선택적 실험이다.
중간·위 선반의 좌우 작업 위치까지 중립 팔·열린 gripper로 **실제 이동**하고,
정지 조건을 확인한 뒤 base 목표를 유지하면서 남은 관절 목표를 SAC로 학습한다.
박스는 동적이고 초기 base XY/yaw·박스·주변 박스 randomization을 유지한다.
성공/보상/충돌 기준을 완화하거나 성공 순간의 팔 자세로 reset하지 않는다.

이 실험의 [현재 물리 결과·영상](RL_V2_FOUR_REGION_SAC_20261004.md)은 별도로 갱신한다.
실행 코드나 서비스의 존재가 학습 완료 또는 일반화 성공을 뜻하지 않는다.

2026-10-05의 선택적 [접촉 보상 SAC](RL_V2_CONTACT_REWARD_SAC.md)는 기존 actor의
실제 동작을 보존하고 Q/replay를 새로 시작한다. 기본 보상과 이전 실행은 유지한다.

## 파지 중 관측·닫힘·접촉 진단

`train_batched_staged_goal.py --grasp-observation-audit`는 기존 정책의 입력·제어·
물리·보상·성공 판정을 유지하면서 파지 전체 구간을 기록한다. 기존 flap link/
TCP/접촉 tensor를 읽으며 sensor나 physics step을 추가하지 않는다. 고정 정책,
단일 `validation` wave,1..16환경,31..900 step,원래 background 배치에서만
허용한다. TRAIN·독립FINAL·다른 물리/배치 probe와의 혼합은 시작 전에 거부한다.

기존 frozen 실행 명령에 `--no-training --grasp-observation-audit --steps 900`를
추가하고 DEV layout만 담긴 `--waves-json`을 사용한다. Drive 관리는 같은
`batched_staged_goal_with_drive.py`와 고유 `--experiment-dir`을 사용한다.
`--gpu`와 `CUDA_VISIBLE_DEVICES`를 동일하게 지정한다.

`grasp_observation_audit.jsonl.gz`는 step 직전과 termination 계산 후 reset 직전의
실제 flap center/normal·양손 거리·production12cm close gate·배정·명령/실제 닫힘·
각 pad 힘/접촉영역/대향 여부·pinch·hold/proof lift·안전 원인을 기록한다.
Actor의 목표 ID/hand-flap relation도 reward의 실제 목표와 대조한다. 유효하지 않은
pose는 환경별로 표시하며 nonfinite 값은 strict JSON의 null로 기록한다.
종료 시 summary와 압축 trace가 닫히고 기존 finished 업로드에서 크기·MD5를 검증한다.

이 자료와 함께 수집된 transition의 source는
`grasp_observation_audit_NOT_matching_Q_replay`다. Matching Q replay로 가져오지
않으며 actor/Q/replay 평가 불변성 검사도 그대로다. 특정 성공/실패 사례를 골라
진단한 점수는 일반화 성공률이 아니다. N128 사례를 cold N16에서 재생하면
constructor·contact history도 달라지므로 동일 초기 물리 상태를 보장하지 않는다.

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

### 실제 수집량과 성공 정책 유지 항

`--normalize-prior-loss-by-radius --actor-min-replay-rows 16384`는 선택적 학습 설정이다.
Actor를 바꾸기 전에 buffer에 실제 held 전이가16,384행 있어야 한다. Critic의2,048회
warmup도 별도로 충족해야 한다. 개발/final 전이는 이 수집량에 포함하지 않는다.
Prior 항의 effective MSE weight는 `2*(1-progress)/radius²`이며 초기 radius0.05에서800이다.
Prior 감소와 radius 확장은 critic 횟수 대신 실제 actor 진행5,000회에 연결한다.
따라서 수집 중 Q만 업데이트할 때 prior가 미리 사라지지 않는다. 기존 checkpoint는
명시적으로 이 옵션을 선택하지 않으면 이전 계산과 계약을 유지한다.

`--anchor-prior-to-initial-policy`를 함께 쓰면 현재 초기 actor와 normalizer의
**고정 복사본**을 목표로 삼는다. 초기 actor 출력과 목표의 MSE는0이다. 이전 BC
정책을 기준으로 가중치만 키우면, 성공했던 초기 actor 자체를 다른 목표로 당길 수
있으므로 두 옵션을 구분한다. Snapshot은 actor만 포함하며 다른 Q/reward/replay를
가져오지 않는다. 두 경우 모두 실행 시 live VR/IK가 필요하지 않다.

개발 회귀 후 종료된 **동일21-goal 계약**의 Q/replay를 보존하면서 actor를 복구하고
위 설정으로 이동하려면 다음을 사용한다. 기존 파일을 수정하지 않고 새 폴더를 만든다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  python scripts/rl/recover_pose_goal_actor.py \
  --checkpoint /absolute/path/to/latest-closed-checkpoint.pt \
  --best-checkpoint /absolute/path/to/same-contract-validated-checkpoint.pt \
  --output-dir /absolute/path/to/new-recovery-directory \
  --replay-capacity 100000 --actor-min-replay-rows 16384 \
  --normalize-prior-loss-by-radius --anchor-prior-to-validated-policy
```

최신 critic/target/optimizer와 실제 transition tensor, actor/critic 진행 횟수는
보존하고 actor optimizer moments만 초기화한다. Validated actor의 radius와 prior
schedule 기준도 복구한다. Frozen prior snapshot은 선택한 validated actor다.
이 옵션은 물리/관측/action 계약 변경을 허용하는 bypass가 아니다.

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

병렬 runner의 완료 결과는 [CPU 지표 요약 사용법](RL_V2_CONTACT_REWARD_SAC.md#gpu-없이-완료된-학습-지표-읽기)의
`summarize_batched_staged_run.py`로 읽을 수 있다. 전체 시도 분모와 초기화 실패를
보존하며 실제 held TRAIN replay, 영역별 성공, 겹치는 안전 종료 원인을 구분한다.
이 도구는 저장된 snapshot만 읽고 현재 writer 생존 여부는 따로 확인한다.

## 여러 초기 배치의 병렬 wave 실행 (실험 경로)

선택 사례의 frozen 파지 진단과 별도로 원래 DEV128 전체를 한 장면씩 기록하는
[full distribution audit](RL_V2_CONTACT_REWARD_SAC.md#원래-dev128-전체를-한-장면씩-진단)도
지원한다. 이는 TRAIN/Q 데이터가 아니고 N128과 같은 물리 이력이라고 가정하지 않는다.

`train_batched_staged_goal.py`는 환경마다 실제 base 정지 시점, box anchor,
held waypoint, 경과 시간을 따로 유지하며 하나의 SAC를 공유한다. 종료한 환경은
다음 중립 whole-wave reset까지 replay/normalizer에서 제외한다. Auto-reset 뒤의
관측을 종료 직전 next observation으로 쓰지 않는다. Approach 전이는 새21-goal Q에서
제외하고 실제 task 성공/실패는 모두 집계한다. Validation/final wave는 optimizer와
replay가 바뀌지 않았는지 검사한다. Live VR/IK teacher나 성공 자세 reset은 없다.

초기 settling에서 일부 환경이 requested layout을 잃어 자동 재생성되면 그 환경을
`invalid_reset` 실패·0 actual replay row로 기록한다. 그 사례는 개발/final 성공률의
분모에 남고 성공 상태로 재시도하지 않는다. 같은 wave의 정상 환경은 계속 수집한다.
HDF에는 실제 action을 실행한 episode만 저장하며 wave/environment/layout JSON
attribute로 사례를 명시한다. 재생성 장면의 관측을 원래 layout의 snapshot/Q로 쓰지 않는다.

Whole-wave reset은 scene asset/contact sensor를 reset하고 이전 permanent base
wrench·joint effort/velocity target을 정리한다. Robot과 box의 teleport FK를 함께
갱신한 뒤 새 중립 자세를 action manager에 다시 잡는다. 최초60 physics substep도
매번 정상 zero-action controller를 적용해 현재 gravity/COM support를 계산한다.
이 과정에는 reward/replay/policy update가 없고 성공 상태나 base teleport 경로를
사용하지 않는다. `wave_reset_controller_contract`와 initial layout guard의
`controller_reset`에 이전/정리 후/새 중립 hold의 실제 force·torque norm을 남긴다.
이 lifecycle 수정의 효과는 같은 frozen development 배치의 반복으로 측정하며,
코드 검사만으로 불량 spawn이나 물리 발산이 해결됐다고 판단하지 않는다.

학습 wave 앞에 같은 development wave를 두고
`--stop-on-validation-regression`을 지정하면 구역별 성공 수가 감소할 때
`policy_regression` 상태로 종료한다. `--minimum-validation-region-success-rate 0.5`는
어느 구역이든 실제 성공률50% 미만일 때 다음 TRAIN wave를 시작하지 않는다.
전체 성공 수가 증가해도 다른 구역의 성능 손실을 숨기지 않는다. Development와
독립 final seed도 서로 겹칠 수 없다. 종료된 실제 Q/replay는 보존하며,
`recover_pose_goal_actor.py`로 같은 계약의 검증된 actor만 **새 폴더**에서 복구한다.
이 옵션이 오류를 자동 수리하거나 실험을 자동 재시작하는 것은 아니다.

### Binary gripper를 직접 학습하는 hybrid SAC

`prepare_staged_goal_sac.py --hybrid-grippers --free-grippers`는 **별도 fresh Q**의
`staged_base_hold_remaining_hybrid_sac_v1`을 준비한다. 나머지19개 연속 목표와
양손의 Bernoulli 열림/닫힘 정책을 함께 학습한다. 실제 gripper의 `action>0`
제어를 표현하므로 Q/replay의 jaw 값은 정확히−1/+1이다. 연속 jaw 값의 부호만
실행하면서 그 부호의0 gradient에 기대지 않는다.

현재 실제 관측에서 가능한 네 jaw 조합을 Q에 대입해 actor/target의 기대값을
정확히 계산한다. 이것은 Q의 policy expectation이고 새로운 transition 생성이나
old demo reward/Q 수입이 아니다. 각 손이 배정 flap에서12cm 이상 멀면 open만
가능하며 그 손의 categorical entropy도0이다. 연속 목표의 std/radius는 기존
설정이고, categorical entropy target은 활성 jaw당0.35nats, 초기 temperature는
0.01이다. 양손 categorical prior KL의 weight0.05는 연속 목표의 정규화 MSE와
분리했다. 재개에는 두 entropy optimizer를 포함한4개 optimizer 상태가 필요하다.
일반21-goal checkpoint/replay를 이 hybrid Q로 그대로 resume하면 거부한다.

`--exploration-correlation 0.98`는 선택적인 **TRAIN behavior** 잡음이다.
각 환경의 Gaussian latent를 AR(1)로 이어 body goal jitter를 줄인다. Hybrid jaw는
normal CDF를 통해 상관 uniform threshold를 얻는다. Feedback 중 behavior는
off-policy이고 SAC actor/target의 독립 Gaussian·Bernoulli 기대값은 유지한다.
매 중립 wave에서 환경별 잡음 이력을 새로 만들며 접근/종료한 다른 환경의 잡음은
진행시키지 않는다. 이 설정이 최소 파지 hold시간을 보장하지는 않는다.
Validation/final/정책 재생은 noise 없이 deterministic mean/최대확률 jaw를 쓴다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl \
  python scripts/rl/prepare_staged_goal_sac.py \
  --checkpoint /absolute/path/to/matching-frozen-pose-goal-model.pt \
  --native-seed /absolute/path/to/current-middle-train-success.hdf5 \
  --native-seed /absolute/path/to/current-upper-train-success.hdf5 \
  --waypoints /absolute/path/to/HumanoidScene/docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --output-dir /absolute/path/to/unique-hybrid-initial-model \
  --hybrid-grippers --free-grippers --gripper-logit-scale 0.005 \
  --normalize-prior-loss-by-radius --anchor-prior-to-initial-policy \
  --actor-min-replay-rows 32768 --replay-capacity 500000 \
  --exploration-correlation 0.98
```

새 checkpoint를 같은 batched Drive runner의 `--checkpoint`에 전달한다.
Checkpoint의 artifact type으로 일반/hybrid learner를 선택한다. 일반 delta SAC,
PPO/DPPO의 알고리즘과 기존 옵션 기본값은 변경하지 않는다.
Replay500,000행은 실제 buffer4,120,500,000bytes(약3.84GiB)이고 디스크의
experience 파일도 실제 수집량에 따라 이 크기까지 커질 수 있다.

Base stage의 거리/속도 계산과 접근 command는 벡터화했다. 각 환경의15연속
정지 step·clock·anchor·종료 mask는 유지하며, 단일 제어와 일치하는 회귀 검사를 둔다.
중도 stop한 개발 wave는 guard를 채점하지 않는다. Actor 업데이트가 같은데
구역 성능이 감소하면 `physical_reproducibility_failed`로 구분해 기록한다.
이는 여전히 실험 중단이며 실제 성능 손실을 무시하는 옵션은 아니다.

`--contact-stability-probe --no-training`은 박스가 접촉 중 비정상적인 속도로
날아가는 원인을 비교하기 위한 별도 frozen 진단이다. Physics dt1/240s·decimation8로
control dt1/30s를 유지하고 box solver64/16, body linear cap25m/s,
angular cap10,000deg/s, depenetration cap2m/s를 적용한다. Angular cap 단위는
Isaac Lab schema의 **deg/s**다. 선속도/각속도 cap은 기존 실패 기준10m/s·100rad/s
보다 높고 성공·충돌 기준은 그대로다. 이 진단은 TRAIN과 동시 사용을 거부하고
수집 출처를 matching Q replay가 아닌 것으로 표시한다. 결과가 좋아도 그 자체는
SAC 개선이 아니며, 실제 학습에는 새 dynamics 계약과 fresh Q가 필요하다.

`--tgs-zero-velocity-probe --no-training`은 같은 physics/control dt와 position
iteration을 유지하고 scene·robot·box의 TGS velocity iteration만0으로 하는 별도
frozen 진단이다. 두 solver 진단을 함께 쓰거나 TRAIN에 쓰면 거부한다.
NVIDIA는 [TGS velocity iteration 및 D6/loop-closure 제한](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html)을
안내한다. 이 문서는 현재 실패의 원인을 확정하는 근거가 아니며, 동일 실제 초기
배치의 결과로 가설을 검사한다. 성공/보상/안전 기준은 유지하고 해당 전이를 Q에 넣지 않는다.

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

128환경 후속 실행에서는 전체 batch의 박스가 동시에 멈추는 조건을 제거했다.
각 requested original layout에서 모든 활성 박스가 동일한 선속도0.01m/s·각속도
0.05rad/s 미만으로8tick 유지돼야 통과한다. 교체·numerical failure·종료·90step 뒤
미안정 환경은 실패 attempt로 기록하고 replay에서 제외한다. 정상 환경은 계속 진행하며
실패를 성공률 분모에서 빼지 않는다. Requested reset 대기 역시 실패 장면의 respawn을
기다리지 않는다. 성공·안전 조건과 randomization은 유지한다.

Experience 저장은 최근 capacity 행만 담은 소유 tensor를 직렬화한다. 빈 view나
잘라낸 view의 전체 원본 storage를 저장하지 않는다. Initial replay0행은 약11KB이며
종료 replay의 크기는 실제 보유 행 수에 따라 증가한다. 주기적 모델 checkpoint와
종료 replay를 혼동하지 않는다.

`--contact-last-probe --no-training`은 PhysX의
`solve_articulation_contact_last=True`만 바꾸는 별도 frozen 진단이다.
Physics/control dt·iteration·속도 제한·성공·안전 기준은 유지하며 실제 USD flag도
확인한다. TRAIN 및 다른 solver probe와 함께 쓰면 거부한다. NVIDIA의
[articulation 안정성 안내](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)는
접촉 순서 변경을 gripping 진단 옵션으로 설명한다. 현재 box 발산 원인 또는
개선으로 확정하지 않으며, 다른 dynamics의 전이를 matching Q에 가져오지 않는다.

단일 `replay_v2_grasp_reference.py --staged-goal-sac` 평가도 batched training과
동일한 scene/asset/controller reset·robot FK·매 substep 중립 제어·original layout
settling을 사용한다. 초기 layout guard와 reset 계약을 manifest/HDF에 남긴다.
기존 VR/다른 policy replay 경로는 유지한다.

완료 episode를 CPU에 모으는 batched runner는 `RlTransitionRecorder.append_many`로
전이 필드마다 한 번 resize/write한다. LZF 압축과64행 이하 chunk를 사용하며 실제
행/값/정렬/라벨을 바꾸지 않는다. Quest의 한 행씩 실시간 기록은 기존 `append`를
유지한다. 이미 실행 중인 프로세스에는 코드를 교체하지 않으므로 새 실행부터 적용된다.

### Hybrid actor 회귀 복구와 solver 진단

`recover_pose_goal_actor.py`의 동일-contract 복구는 hybrid SAC에서도 지원한다.
`staged_base_hold_remaining_hybrid_sac_v1`, `hybrid_goal_sac`와 binary jaw 계약,
4개 optimizer, 고정 actor normalization이 모두 같아야 한다. Writer가 종료된 뒤
최신 checkpoint와 같은 폴더의 `staged_goal_experience.pt`, 개발 평가로 선택한
best actor checkpoint를 입력한다. 최신 Q/target·critic optimizer·두 entropy
optimizer·실제 replay·update 횟수를 보존하고 actor와 actor Adam moment만
교체한다. 독립 final을 actor 선택에 쓰지 않는다. 사용 명령은 위 복구 예와 같다.

`--pgs-probe --no-training`은 PGS solver0만 선택하는 frozen 비교다. 실제 USD
`physxScene:solverType=PGS`를 확인하고 dt·iteration·velocity cap·success·safety를
유지한다. TRAIN과 다른 solver probe는 거부한다. 변경된 dynamics 진단으로
명시하며 기존 TGS Q/replay의 seed로 허용하지 않는다. 같은 정책·case·환경 수와
원점의 원래 설정으로 비교해야 하고, 채택 시 새로운 dynamics의 fresh Q가 필요하다.

### 동일 개발 baseline과 물리 수치 실패 분리

기존 guard는 구역별 성공이1개만 줄어도 중단한다. Frozen 반복4/16→5/16→5/16의
중간 좌1→2→1에도 학습을 중단할 수 있어, 다음 실행에서
`--stop-on-validation-regression --validation-regression-significance 0.05`를
선택할 수 있게 했다. Default0은 기존 strict count 방식을 유지한다.
같은 case의 초기 평가를 고정 baseline으로 삼아 성공 손실/획득을 paired exact
binomial test로 비교한다. 각 반복 k의 threshold는0.05/[k(k+1)×구역 수]다.
Best noisy repeat를 통계 null baseline으로 다시 고르지 않는다. 이 규칙은
paired 결과의 교환 가능성을 가정하며 GPU 물리 재현성을 보장하지 않는다.
Raw 구역별 하락·짝별 손실/획득·p·threshold를 모두 남기고 independent final은
선택/학습에 쓰지 않는다. Baseline floor와 성공률 분모는 유지한다.

Active layout에 nonfinite robot dynamics가 생겨 respawn됐다면 그 환경만 실패
attempt로 종료한다. Corrupt action→replacement state 행은 Q/HDF에서 제외하고,
마지막 valid 물리 기록과 numerical causes를 별도 요약한다. Healthy terminal
전이는 그대로 저장하고 학습하며, 다른 환경도 계속 진행한다. 과거 valid row의
reward/terminal flag를 사후 수정하지 않는다. Finite box speed failure는 기존
실제 terminal transition으로 유지한다. 이는 물리 발산 치료가 아닌 데이터
오염·한 환경의 오류로 전체 학습이 중단되는 문제의 수정이다.


## 별도 solver 학습의 계약

`prepare_staged_goal_sac.py --physics-solver PGS`는 기존 frozen pose-goal
actor prior만 활용해 **새 Q, 빈 replay, 빈 optimizer**를 준비한다.
`physics_dynamics`에는 PGS/solver_type0와 physics_dt1/120s, control_dt1/30s를
기록한다. 기존 초기화 예시의 옵션에 이 옵션을 추가하고 새 output을 사용한다.

Batched runner와 single staged replay는 training manifest의 같은
`physics_dynamics`를 적용하고 실제 USD solverType도 확인한다. Training 입력은
기존 baseline action contract에 이 필드를 추가한 manifest이며, runner가
상단 torso+6cm 계약을 구성한다. PGS checkpoint/replay를 TGS로 resume하거나
그 반대는 금지한다. Legacy manifest에 필드가 없으면 TGS다.

`--pgs-probe`는 기존 TGS 모델의 **frozen 비교 전용**이다. 실제 PGS 학습에
이 옵션을 사용하지 않는다. 이 진단 전이는 TGS Q/replay에 import하지 않는다.
Mass/inertia audit는 실제 initialized PhysX를 읽는 기록이며 dynamics를 변경하지 않는다.


## Frozen 월드 원점 비교

`train_batched_staged_goal.py --centered-world-probe --no-training`은 모든 clone의
월드 원점을0으로 놓는 별도 진단이다. Replicated GPU physics와 environment
collision ID 필터가 모두 필요하며, 실제 env_origins가 전부0인지 검사한다.
[설치된 SDK와 같은 공식 소스](https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/scene/interactive_scene.html)의
GPU enable_env_ids 경로를 사용한다.

일반 parallel 설정의 env_spacing≥5m 기본값은 유지한다. 이 flag는 num_envs와
초기 semantic layout, base randomization, solver, control dt, 성공/충돌 기준을
바꾸지 않지만 broadphase/좌표 정밀도의 효과를 보기 위한 frozen 진단이다.
TRAIN과 함께 사용하면 거부하며 결과는 기존 Q/replay에 import하지 않는다.
원점 자체가 현재 실패 원인이라고 확정한 것은 아니다.


## 롤러 support 기준 lift와 actor-only 새 Q 초기화

2026-10-04부터 proof lift는 settled root height 증가와 **실제 active support**
위 box underside 최소 gap의 min이다. 기본 roller deck은 bare shelf보다10mm
높으므로 이 offset을 제외한다. `--no-rack-rollers`의 offset은0이다.8mm lift와
0.25s opposing bilateral hold, 안전 기준은 그대로다.

Terminal contract에 `proof_lift_reference`와 `proof_lift_support_offset_m`를
저장하고 현재 rack 설정과 비교한다. 구 manifest나 다른 offset의 Q/replay를
resume할 수 없다. Single staged replay와 batched training 모두 비교한다.
새 `prepare_staged_goal_sac.py` output의 `manifest.json`를 training manifest로
사용한다. 이전 manifest만 복사하거나 field만 추가해 old Q를 resume하지 않는다.

구 held-goal actor를 재사용하려면 기존 초기화 명령에 `--actor-checkpoint
/absolute/path/to/matching_staged_checkpoint.pt`를 추가한다. Frozen24-goal
`--checkpoint`와 `--native-seed`는 기존 actor prior 복원 용도로만 남는다.
21-goal actor/normalizer만 이전하며 Q/target/critic normalizer/entropy/optimizer/
replay/counter는 초기 상태다. Matching goals/좌표/안전과 source contract를 검사하며
lift reference 또는 fresh-Q solver 차이만 허용한다. 현재 actor를 frozen prior로
anchor하고나서 새로운 원래 배치 rollout을 수집한다. 이는 기존 SAC의 continuation이 아니다.

과거 성공 flag에는 bare shelf gap을 사용한 false positive가 섞여 있으므로
[실제 geometry audit와 새 실행](RL_V2_FOUR_REGION_SAC_20261004.md)을 참고한다.


## 검증된 actor 복구와 닫힘 탐색 유지

2026-10-04 실제 replay 진단에서 softened jaw logits의 부호가 작은 학습 변화로
사라졌다. 아래 옵션은 **동일 solver/physical contract**의 최신 실제 Q/replay를
보존하고 검증된 actor를 복구하며 binary jaw prior와 continuous 탐색만 변경한다.
기존 명령은 기존 동작을 유지한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/recover_pose_goal_actor.py \
  --checkpoint /absolute/path/to/latest-corrected-support/checkpoint_XXXXXXXX.pt \
  --best-checkpoint /absolute/path/to/validated-actor/checkpoint_XXXXXXXX.pt \
  --output-dir /absolute/path/to/unique-recovered-run \
  --normalize-prior-loss-by-radius --anchor-prior-to-validated-policy \
  --fixed-prior-radius 0.05 --validated-jaw-prior-confidence 0.8 \
  --jaw-prior-residual-gain 20 --body-policy-std 0.001 --body-policy-std-cap 0.005
```

Goal means/actor normalizer는 검증된 actor 그대로이며 continuous log-std outputs만
초기값으로 재설정한다. Q/target/actual transition tensors/optimizer counters와 두 entropy
optimizer states는 보존한다. Frozen snapshot이 있어야 confidence 옵션을 사용할 수 있다.
Jaw sign prior는 실제 SAC가 residual로 바꿀 수 있으며 먼 flap에서는 계속 open만 허용한다.
Fixed radius를 지정하면 actor/Q update와 실제 collection 모두 같은 bound를 사용한다.
Saved config와 hybrid/goal/replay contract가 모두 matching되어야 resume한다.

복구 파일을 기존 Drive에 checksum 검증 업로드한 후 `batched_staged_goal_with_drive.py`
에서 새 run directory로 재개한다. DEV에는 gradient/replay 수집을 하지 않는다.
DEV 실패를 TRAIN 성공으로 숨기거나 randomization을 좁히지 않는다.

## Sparse background의 초기 배치 비교

`train_batched_staged_goal.py --packed-background-probe --no-training --steps 1`은
frozen reset-only 진단이다. `--waves-json`의 각 wave에
`"background_placement": "original"` 또는 `"packed"`를 넣어 같은 cases를
original→packed→original로 비교한다. 기본은 original이며 TRAIN과 함께 사용할 수 없다.

Packed mode는 selected target의 pose/type/ID와 초기 base, 모든 active masks를
유지한다. 주변 box의 앞쪽 빈 depth slot만 크기에 맞춰 채우고 sloped support
height를 맞춘다. 기존 `packed_region_depths`의 목적과 같은 reset 분포 진단이며,
실제 rollout 중의 box를 고정하거나 재배치하는 기능이 아니다. Metadata와
collection_source에 frozen/Q-import-ineligible을 기록한다. 한 tick 결과는
초기 guard 비교이고 파지 성공률이 아니다. Main learner의 기존 reset 분포는 유지한다.

### Frozen workplace 후보 비교

`train_batched_staged_goal.py --no-training --base-waypoint-probe`는 같은 개발 배치에
여러 base 진입 목표를 시험하는 진단이다. 각 wave layout row에 다음을 추가한다.

```json
{
  "episode_index": 0,
  "layout": {"seed": 123, "split": "holdout"},
  "waypoint_probe": {"name": "toward_center", "offset_xy_yaw": [0.05, 0.0, 0.0]}
}
```

`layout`은 기존의 완전한 `GraspLayout`을 사용한다; 위 예시는 추가 필드만 설명한다.
XY는 rack frame의 m, yaw는 rad이며 각 XY±0.15m/yaw±15° 이내다. 원래 neutral
initial robot/box와 strict source checkpoint/templates를 복원한 뒤 실제 base 목표만
변경한다.15 stable ticks 뒤 frozen actor를 수행한다. TRAIN/IK/시연 재생이 아니며
같은 seed를 반복할 때는 `(seed,candidate name)`으로 비교 attempt를 구분한다.
원래 randomization과 물리적 성공/안전 기준을 유지하고 metadata/Q import allowlist로
이 진단 transition을 matching SAC replay에서 제외한다. Default 학습에는 영향이 없다.

## 실제 TRAIN 성공 경험 유지

`prepare_staged_success_retention.py`는 matching hybrid held-goal SAC에서 실제로
성공했던 TRAIN episode를 별도 bank에 남겨 이후 업데이트에서 재사용한다. 원래 VR
데모의 BC prior와 별개이며, DEV/FINAL·waypoint probe·unsafe·무효 reset 자료는
bank에 넣지 않는다. 기존 checkpoint에는 이 기능이 적용되지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_staged_success_retention.py \
  --checkpoint /absolute/path/to/closed-matching-run/checkpoint_XXXXXXXX.pt \
  --best-checkpoint /absolute/path/to/dev-selected-matching-actor/checkpoint_XXXXXXXX.pt \
  --output-dir /absolute/path/to/unique-success-retention-input
```

Source manager의 writer가 종료되고 최종 Drive 백업이 검증되어야 한다. Source
checkpoint와 replay의 solver·보상·관측·action·held controller 계약도 같아야 한다.
Interrupted run도 유한한 checkpoint와 실제 성공 자료/최종 백업이 있으면 사용할 수
있지만, 그 실행 전체가 정상 종료했다고 기록하지 않는다. `/proc` 검사는 실제 host의
PID namespace에서 수행해야 한다. 체크포인트를 복사한 별도 폴더만으로는 source
writer 종료·실측 HDF 증거를 대체할 수 없다.

Native HDF 성공 궤적과 원래21-D Q rows를 raw pre/next critic·reward·termination·
elapsed clock으로 정확히 연결한다. 포화된 physical delta를 역산하여 가상 목표를
만들지 않는다. Bank에는 실제 opposing 양손 pinch/stability,0.25초 hold, 이미 roller
support가 보정된8mm proof lift를 통과한 episode의 **held-phase 전체 실측 경로**를
넣는다. 접근 phase는 포함하지 않는다.

Q의256-row batch 중 초기 약20%(51행)를 이 bank에서 샘플링하고5000 actor
updates에 걸쳐5%로 줄인다. 나머지는 원래 matching replay다. Actor에는 실제 성공
states64개에서19개 연속 목표의 MSE와 접근 gate가 허용한 손의 binary jaw BCE를
추가한다. Radius0.05에서 effective MSE weight400, jaw weight0.05이며 기존 frozen
prior와 SAC objective도 유지한다. 이는 SAC의 실제 성공 경로를 보존하는 보조 모방
손실이며, 새로운 VR 궤적이나 평가 결과를 학습 action label로 사용하는 기능이 아니다.

구역마다4096행까지 whole episode로 보관하고, 자료가 있는 구역을 균등하게 샘플링한다.
새 TRAIN wave에서 성공하면 실제 실행한21-D goals를 bank에 추가한다. 지금 초기
bank는 중간 오른쪽2episode/826행뿐이므로 다른 세 구역의 학습 자료가 충분하다고
해석하면 안 된다. `successful_train_bank`, `successful_train_replay_fraction`,
`successful_train_rows_in_Q_batch`, `success_goal_loss`, `success_jaw_loss`로 확인한다.

입력 checkpoint의 크기·MD5 검증 후 기존 `batched_staged_goal_with_drive.py`의
고유 실행 폴더에서 `--training`으로 실행한다. Checkpoint 계약이 기능을 켜며 별도의
runtime flag는 필요 없다. Source best actor/normalizer를 복구하지만 Q/replay/counter는
matching closed Q branch를 유지하므로 서로 다른 branch의 actor update 수를 단순히
비교하지 않는다. 초기 source/audit와 이후 실제 optimizer 증가량을 함께 확인한다.
Checkpoint와 종료 replay에 bank를 저장하고 DEV/FINAL에서는 optimizer/replay/bank를
수정하지 않는다. Immutable checkpoint와 최신 replay bank를 분리해 복원하므로 같은
Q counter의 이미 업로드한 파일을 덮어쓰지 않는다.

환경·randomization·진입 waypoint·성공/안전 판정은 그대로다. 기능 동작 검사는 물리적
성공률 개선의 증거가 아니며, 같은 DEV와 독립 FINAL에서 별도로 평가해야 한다.

## 그리퍼 motor drive만 비교하는 frozen 진단

`train_batched_staged_goal.py --gripper-drive-probe --no-training`은 기존 PGS/TGS
정책을 고정하고 네 개 `bar_1` 모터의 PD/drive limit만 비교한다. Source checkpoint,
solver/시간 간격, geometry/마찰/50N close-force feedforward, body/passive drives,
진입 위치, initial base·box randomization,성공·안전 조건은 유지한다.

`--waves-json`의 각 wave에는 `"gripper_drive_probe": "original"` 또는
`"gripper_drive_probe": "soft_2nm"`를 추가한다. 같은 완전한16-case DEV layouts를
세 wave에 넣어 original→soft_2nm→original 순서로 비교한다. Original은 실제
초기화된 값을 캡처해서 복원한다. Soft 후보는 stiffness100N·m/rad,
damping5N·m·s/rad, motor drive effort limit2Nm이다. 이는 검증 전 후보이며
하드웨어 정격이나 물리 발산 해결값으로 확정한 것이 아니다.

Isaac setters와 actuator-model tensors를 함께 갱신하고 PhysX에서 값을 다시 읽어
일치하지 않으면 실패한다. Arm/body/passive joint는 바꾸지 않는다. 실제 구동값은
`gripper_drive_audit.json`에 wave별로 저장하고 종료 뒤 기존 Drive 업로더가 검증한다.
명시적 flag 없는 override,TRAIN 사용,다른 waveform/waypoint/background/origin
진단의 동시 사용을 거부한다. 이 진단 전이는 matching Q/replay에 넣을 수 없다.
학습 프로세스의 구동을 동적으로 교체하는 기능이 아니며 기본 학습에는 영향이 없다.

완료된 wave의 비교 결과는 CPU에서 판독한다. 프로젝트의 기존 Python 환경을
사용하며 Isaac 앱이나 CUDA를 시작하지 않는다. 결과는 실행 writer 폴더 밖에 저장한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/rl/analyze_gripper_drive_probe.py \
  --run-dir /absolute/path/to/frozen-drive-run \
  --output /absolute/path/to/analysis/drive_comparison.json
```

최초 original/soft/original 반복을 미리 선언한 같은 전체 layouts와 실제 PhysX audit를
요구한다. 초기 무효 배치도 전체 성공률 분모에 남기고, 안전 위반 원인은 중복될 수
있음을 표시한다. 접근 거리 평균은 초기 유효·안전 terminal만 사용하여 발산한 박스의
좌표가 정상 접근을 왜곡하지 않게 한다. 같은 seed뿐 아니라 base·box·distractor를
포함한 전체 initial layout을 대조한다. 쌍별 충돌 원인 변화는 양쪽 초기 유효 사례만
비교하고 초기 유효성 변화 수도 별도로 남긴다.

마지막 original 반복이 끝나기 전에는 `comparison_complete=false`다. 일부 wave,
빠진 실패 사례,변경된 layout,TRAIN/Q replay,미검증 soft 값/원래 값 복원 실패를
완전한 비교로 인정하지 않는다. 작은 DEV 비교만으로 기본 구동값을 자동 변경하거나
independent FINAL 일반화를 주장하지 않는다.

## 에피소드 동안 유지하는 팔 탐색 편차

선택 옵션 `episode_arm_exploration`은 **TRAIN 수집 동작에만** 적용된다.
현재 Gaussian 탐색이 작을 때 매 step의 jitter를 키우는 대신, 양팔14개 목표에
에피소드별로 뽑은 일정한 latent 편차를 더한다. 실제 base 정지 후 held clock이
0→90step(3초) 동안 smoothstep으로 편차를 늘린 뒤 유지한다. 중간 선반은
표준편차0.005, 위 선반은0.02이며 표준정규 표본은 ±2로 제한한다.

이는 tanh·기존 ±0.05 prior projector **이전**의 편차다. 실제 관절 목표의
변화량이나 손끝 이동 거리를 보장하는 값은 아니다. Waist/head/torso와 binary
jaw sampler를 바꾸지 않으며, 접근 중에는 적용하지 않는다. Region은 기존 actor
관측의 selected-region one-hot만 사용한다. Actor/target 학습 분포와 entropy
목표는 그대로 유지하는 off-policy 수집 옵션이다. 평가에서는 편차를 샘플링하지
않으므로 평가 RNG와 greedy 명령도 유지한다. 환경별 ID를 유지해 다른 환경의
종료 때문에 남은 환경의 편차가 바뀌지 않으며 wave reset 때 새로 뽑는다.

이미 Drive로 전체 백업한 **닫힌 초기 입력**의 같은 Q·optimizer·실제 replay를
사용해 별도 실행을 준비한다. 현재 CLI는 `initialized_not_trained` 또는
`initialized_not_new_training` 표시가 있는 초기화 폴더만 허용하며, 실행 중인
학습 폴더를 복제하거나 수정하지 않는다. 이 초기화는 새 학습 업데이트가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/rl/prepare_staged_arm_exploration.py \
  --checkpoint /absolute/path/to/immutable-initial/checkpoint_00012588.pt \
  --verified-backup-receipt /absolute/path/to/full-backup-receipt.json \
  --output-dir /absolute/path/to/new-unique-initial
```

입력 receipt는 원본 `directory`와 `final_size_md5_verified: true`를 기록한 실제
전체 업로드 검증 결과여야 한다. 복제 전후 원본 SHA256과 새 replay tensor의
동일성을 검사하고 `episode_arm_initialization_audit.json`에 기록한다. 모델/Q/
정규화/optimizer/학습 counter는 재설정하지 않으며 실제 성공 TRAIN bank도
그대로 이어간다. 새 future collection contract와 원본 collection provenance를
따로 저장해 기존 replay를 새 탐색으로 수집한 데이터라고 표시하지 않는다.

새 모델·manifest·audit를 기존 Drive 연결로 먼저 검증하고 위의
`batched_staged_goal_with_drive.py`에 새 checkpoint를 지정한다. 옵션은 checkpoint의
계약에서 복원되며 CLI의 전역 기본 탐색을 바꾸지 않는다. 박스·base randomization,
구동·보상·안전·성공 판정과 waypoint를 유지하며, DEV/FINAL은 학습이나 성공 bank에
넣지 않는다. 초기 replay 전체 업로드가 진행 중이라면 미완료로 표시하고 로컬 입력을
보존한다. 300초 백업/검증된 checkpoint 최근2개 보관/종료 로그 검증은 기존 관리자가
수행한다. TRAIN 성공 수보다 이후 같은 DEV의 구역별 성공률로 개선을 판단한다.

## Frozen SAC의 TRAIN 목표 전이 수집

2026-10-05에는 목표 역산을 하지 않고 실제 제어명령21을 Q 행동으로 쓰는
[별도 physical-body SAC](RL_V2_PHYSICAL_BODY_SAC.md)도 추가했다.
기존 goal Q/replay와는 artifact/algorithm/행동 문맥을 구분한다. 아래 수집기의
기존 동작과 기본 goal SAC는 유지한다.

`replay_v2_grasp_reference.py --collect-train-goals`는 **실행 전에 TRAIN으로 선언한
배치**에서 frozen staged hybrid SAC가 실제로 선택한21개 목표를 기록한다.
정책/Q/optimizer를 갱신하지 않는 학습 데이터 수집이다. 기존 VR 수집·정책 평가의
기본 동작은 유지하며, 이 옵션을 명시해야 별도 데이터가 생긴다.

기존 native HDF에는 실제24개 제어 명령이 있지만, projector/관절 속도 포화 뒤의
명령에서 요청된21개 목표를 역산할 수 없다. Bounded replay에서 성공 경로가
사라지면 HDF만으로 그 Q 전이를 복원하지 않는다. 이 수집기는 **실제 pre/next
actor480·critic539 문맥, 요청 goal21, 현재 reward, termination**을 함께 보관한다.
Base 접근은 제외하고, 실제 정지 확인을 거친 held 구간의 연속 전이만 저장한다.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts/rl python scripts/rl/replay_v2_grasp_reference.py \
  --pose-student-checkpoint /absolute/path/to/immutable/checkpoint_00015602.pt \
  --no-pose-student-training --staged-goal-sac --no-staged-goal-training \
  --collect-train-goals --layout-json /absolute/path/to/declared_TRAIN_layout.json \
  --staged-base-waypoints docs/assets/rl_v2_staged_base_hold_candidates_20261004.json \
  --training-manifest /absolute/path/to/matching/baseline_entrypoint_manifest.json \
  --pose-student-native-seed /absolute/path/to/middle_native_calibration.hdf5 \
  --pose-student-native-seed /absolute/path/to/upper_native_calibration.hdf5 \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 --episode-index 1 \
  --torso-extra-height-m 0.06 --steps 900 --contact-diagnostics \
  --output-dir /absolute/path/to/unique_TRAIN_collection \
  --device cuda:0 --headless \
  --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'
```

Manifest/물리 설정은 checkpoint와 일치해야 한다. 위 single-env entrypoint의
baseline manifest는 torso 높이 설정 전의 계약이며, `--torso-extra-height-m`으로
원래+6cm 계약을 복원한다. Native calibration/demo는 기존 neural prior 초기화에
쓰며 실행 중 VR/IK 지시나 기록 action의 open-loop 재생을 사용하지 않는다.
서로 다른 물리·goal 계약은 차원이 같다는 이유로 재사용하지 않는다.

종료 시 `actual_train_goal_collection.pt`는 데이터 전용 artifact이며 학습
checkpoint가 아니다. 실패도 그대로 기록한다. 안전·초기 유효성·실제 양손
opposing pinch/stable hold0.25초·corrected proof lift8mm의 전체 조건을 만족한
완료 성공만 `successful_train_transitions`에 추가한다. 단일 pad의 강한 접촉,
일시적인 lift,미완료 시도는 성공 경험으로 세지 않는다.

CLI는 DEV/FINAL,optimizer 업데이트,live teacher,기록 action 실행과의 동시 사용을
거부한다. 저장 시에도 counter 불변,binary jaw,held phase,pre/next Q 문맥의
연속성,처음 선언한 TRAIN identity를 검사한다. 과거 평가를 나중에 TRAIN으로 바꿔
가져오는 용도로 쓰지 않는다. SAC 추가 연결은 artifact의 출처·계약·닫힌 파일
검증을 확인하고 명시적으로 수행한다. 생성만으로 실행 중인 학습에 자동 투입되지 않는다.

기존 `reference_residual_with_drive.archive_pilot`은 writer 종료 후 이 artifact도
HDF/영상/로그와 같은 로크로 업로드하고 크기·MD5를 검증한다. Drive 연결/300초
주기/최신2 checkpoint 보호는 기존 설정을 재사용한다.

기본 수집은 `greedy`다. 과거 성공 TRAIN에서 사용한 탐색을 가중치 변경 없이
확인하려면 `--train-collection-behavior checkpoint-exploration
--train-collection-seed 120352`를 추가한다. 기존 checkpoint의 Gaussian/AR1/binary
jaw sampler와 지정된 episode arm 설정을 그대로 사용하며 std/radius를 임의로
키우지 않는다. Frozen 수집의 빈 replay 때문에 greedy warmup이 적용되지 않도록
명시적으로 선택하되, normalizer/optimizer/Q/replay 업데이트는 계속0이다.
Sampling mode와 seed를 manifest/HDF/goal artifact에 기록한다. 이 옵션은
`--collect-train-goals` 없이 쓰거나 optimizer 학습과 함께 쓸 수 없다. 일반
TRAIN의 warmup과 기존 DEV/FINAL greedy/RNG 동작은 바뀌지 않는다.

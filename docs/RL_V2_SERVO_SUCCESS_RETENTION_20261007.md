# 실제 성공 제어 명령을 유지하는 SAC 보조 손실

새 보상 SAC의 실제 성공 경험에서 **그리퍼는 개선됐지만 몸체 목표가 기록된 성공
동작에서 멀어졌다.** Q를 전혀 학습하지 않는 문제가 아니라, 같은 성공 상태에서
Q를 높이는 수정 방향과 성공 동작 유지 방향이 충돌함을 확인했다. 실제 한 스텝
제어 명령과 동등한 목표 구간을 기준으로 하는 선택형 보조 손실을 구현했다.
관련 테스트와 실제 full trainer 복원·초기 동작 보존까지 확인한 뒤,
**10/07 09:15 KST에 GPU3에서 새 고유 실행을 시작했다.** 실제 writer2089063과
`CUDA_VISIBLE_DEVICES=3`을 확인했으며 초기 평가를 마치고 새 TRAIN에 진입했다.
새 학습의 파지 성능은 아직 측정하지 않았다.

박스·base·배경·단단한 동적 flap의 무작위화, 실제 양손 파지·유지·들기,
충돌10N/5N 기준과 [새 종료 보상](RL_V2_ABSORBING_GEOMETRY_20261007.md)은 유지한다.
가중치만 올리거나 VR 시연을 추가하는 변경이 아니다. 앞으로 자기 실행에서
수집한 안전한 성공 TRAIN 표본에 적용한다.

## 실제 확인한 문제

새 보상 실행의 닫힌 actor256·Q3072 체크포인트와 새 성공3경로를 사용했다.
성공 종료 동작은 bootstrap이 없으므로 해당 동작의 Q와 즉시 보상을 비교할 수 있다.

| 실제 TRAIN 성공 | 실제 종료 보상 | 같은 행동의 Qmin |
| --- | ---: | ---: |
| 중간 오른쪽 env105 | 2.608 | 2.786 |
| 중간 왼쪽 env12 | 2.613 | 2.384 |
| 중간 왼쪽 env100 | 2.619 | 2.699 |

성공 이벤트는+8이다. 종료 때 접근·접촉 등 잠재값을0으로 처리하는 항목과
비용이 함께 합산되므로 마지막 보상이+8과 같지 않다. Q는 성공 확률이 아니다.
이3건의 종료 Q가 맞는다고 전체 critic의 보정이나 미래 파지 성능이 검증되는 것은 아니다.

같은 과거 상태에서 실제 Gaussian actor 목적함수와4개 jaw 분기를 복원하고,
몸체 평균 출력 head의 경사를 비교했다. 마지막64상태의 Q 목적함수 경사는
이미 반경 보정된 성공 목표 유지 경사보다 **4.09–6.49배** 컸다. 모든12개
구역·구간·잡음 조합에서 두 경사의 cosine은 음수였다. 같은 상태의 greedy
목표는 기록된 성공 목표보다 Qmin이 더 높았지만 실제 제어 명령 차이도 더 컸다.

이는 과거 성공3경로의 첫16·마지막64상태에서 계산한 진단이며 실제 학습 때의
256개 replay minibatch, 전체 실패 경로 또는 새 물리 평가가 아니다. 별도 jaw
성공 손실·포화 정규화의 경사는 이 비교에서 제외했다. 새 실패의 유일한 원인이나
Q가 높인 새 동작이 실제로 나쁘다는 인과관계를 증명하지 않는다.
[Q·실제 보상·경사 근거](assets/rl_v2_absorbing_actor256_success_Q_gradients_20261007.json).

기존 성공 손실에는 `1 / 0.3² = 11.111…`의 반경 보정이 이미 적용돼 있다.
누락된 가중치를 추가한 것이 아니다. 작은 목표 오차가 pending target 차감과
한 스텝 제한을 거쳐 큰 제어 명령 차이가 될 수 있어 보조 손실의 좌표를 보강했다.

## 보조 손실

실제 목표와 측정된 pending drive target의 차이를 기존 한 스텝 단위로 나눈
값을 `u`라 한다. Arm·관절은 기존 joint step scale, upright torso 위치는
기존 `0.1 / 30m`를 사용한다. 실제 명령은 `clip(u, −1, 1)`이다.

기록된 안전한 성공 명령을 `c`라 할 때, 다음 오차를 사용한다.

```text
c = +1:       error = max(1 − u, 0)     # u >= 1인 모든 목표는 동등
c = −1:       error = max(u + 1, 0)     # u <= −1인 모든 목표는 동등
−1 < c < +1:  error = u − c

body_loss = 기존 실제 목표 MSE + 0.01 * mean(SmoothL1(error, 0, beta=1))
```

단순히 clipping한 두 명령의 MSE를 계산하면 잘못된 방향으로 포화된 목표에서
경사가0이다. 위 오차는 그런 목표를 성공 명령이 가능한 구간으로 되돌리는
경사를 남긴다. 같은 포화 명령을 만드는 여러 목표는 보조 손실0이며,
clipped 명령에서 임의의 역변환 목표를 만들어 라벨로 사용하지 않는다.
기존 목표 MSE와 jaw 손실은 유지한다. 큰 오차에서는 SmoothL1의 `u`에 대한
경사가 제한되며19개 연속 좌표만 포함한다.

계수0.01은 같은 저장 모델의 실제 성공 상태에서 크기를 비교한 시작 설정이다.
마지막64상태의 몸체 평균 head 유지 경사는 기존의4.53–4.92배였으며,
실제 새 학습이나 최적 계수 검증의 결과는 아니다.
[계수별 같은 상태 비교](assets/rl_v2_servo_retention_candidate_gradients_20261007.json).

## 실제 시작한 비교

소스 `0cf8c2d`와 초기 입력8개를 기존 Drive에 업로드하고 크기·MD5가 일치함을
확인했다. 고유 실행은 `batch_sac_20261007_091545_1a3cfe`이며 실제 writer와
supervisor의 소유자·명령·CUDA3, 서비스 실행 상태를 확인했다. 이후 실제 첫
DEV의 learner와 agent에서 새 성공 명령 유지 objective, 표준편차0.0075–0.03,
Q 제어 입력·새 종료 보상을 확인했다. 초기 평가 동안 actor·Q·온라인 replay와
성공/n-step 은행은0이다. 실제17개 wave도 검증한 준비 입력과 일치했다.
첫 평가 진입은 actor/Q 갱신·파지 개선을 의미하지 않는다.
[실제 첫 DEV·적용 계약](assets/rl_v2_servo_retention_first_actual_DEV_20261007.json).

전체 초기 DEV는 **23/128(중간 좌12·우6, 상단 좌5·우0)**이었다. 실제 양손
파지·유지·들기 판정과 원래 네 구역32개씩의 요청을 확인하고 동일 초기 모델을
보존했다. Actor·Q0회 모델·optimizer는 준비 입력과 같고 평가 전이를 학습에
넣지 않았다. 이후 실제 TRAIN에서 Q1,144회 갱신을 확인했다
(10/07 10:06 KST, actor0회·온라인 실제62,128전이).
이는 초기 정책의 기준과 critic 학습 진입이며 성능 개선이 아니다.
[전체 초기 평가](assets/rl_v2_servo_retention_full_initial_DEV_20261007.json).

첫 새 TRAIN128조건은 안전한 실제 양손 파지·유지·들기 성공 **8건**이었다.
중간 좌5·우2, 상단 좌1·우0이며 랙90·시간 초과30건, 초기화 무효0건이다.
실제 실행이 읽은 plan의 구역별 seed와 원래 layout 전체를 확인했다. 구역별
seed 간격이 있으므로 연속128개 정수라고 가정하지 않았다.

저장 완료된 actor0·Q1588 모델을 별도 보존해 자기 wave1의 성공8경로·3,525전이가
은행에 들어갔고 평가 데이터가0임을 확인했다. 두 Q·두 target·critic 정규화는
초기 입력에서 실제 바뀌었고 유한했다. Actor tensor는 동일하다. N-step 은행도
자기 TRAIN 성공·실패의31,584행을 포함했다. 아직 actor warmup 중이므로 새
성공 명령 유지 손실이 실제 actor minibatch에 사용됐다는 증거나 학습 후
greedy 성공률 개선은 아니다.
[첫 실제 TRAIN·닫힌 모델·은행 근거](assets/rl_v2_servo_retention_first_actual_TRAIN8_20261007.json).

이후 실제 TRAIN wave2의 actor warmup이 끝났다. Actor11회·Q2,090회 갱신
시점의 실제 actor minibatch에서 자기 성공64표본·지정 마지막 구간32표본을
사용하는 것을 확인했다. 기존 목표 MSE는0.00004278, 새 명령 구간 Huber는
0.04278이며 기존 반경 보정11.111…과 계수0.01을 합친 실제 Huber 가중치는
0.1111이다. VR/teacher BC 가중치는0이며 자기 성공8경로·3,525행, 평가0행을
확인했다. 이는 실제 새 actor 손실 사용 확인이며, 전체 물리 평가나 성공률
개선의 증거는 아니다.
[실제 actor 배치](assets/rl_v2_servo_retention_first_actual_actor_loss_20261007.json).

주기 저장은 Q1,024회마다다. Q2,048 모델은 actor0회의 warmup 경계였고,
다음 실제 Q3,072·actor256 모델을 보존했다. 실제 actor6개 tensor가 초기 입력과
달라졌고 모든 model 값이 유한하며 자기 성공8경로도 유지됐다.
저장된 명령 유지 계약도 일치했다. 여전히 실제 갱신 확인이며 전체 greedy
평가의 개선은 아니다.
[실제 저장된 갱신 actor](assets/rl_v2_servo_retention_first_updated_actor_20261007.json).

독립적인 새 TRAIN1,536조건과 원래 DEV128개를 사용한다. 네 구역은 wave마다
32개씩이며 자신의 전체 초기 DEV부터 시작한다. 다른9개 계획과 겹치지 않는
TRAIN seed를 사용하고 독립 FINAL은 사용하지 않았다. 초기 mean·jaw와
표준편차0.0075–0.03, 넓은 episode20% 팔 탐색·AR0.99는 quarter 비교와 같다.
모든 초기 학습 은행·Q 갱신·optimizer 상태는0이며 다른 실행의 성공9경로를
초기 학습 데이터로 가져오지 않았다.

원래 관리자를 통해5분마다 업로드하며 검증된 오래된 체크포인트만 정리하고
최근2개를 유지한다. 종료 로그는 writer 종료 뒤 업로드·검증한다. 전체17개
wave를 진행하도록 DEV 회귀에 따른 전역 조기 종료만 해제했다. 성공·랙10N/
장애물5N·박스 과속 등 매 동작의 종료 조건과 무작위화는 유지한다.
기존 우리 학습4개와 다른 사용자의 실행은 종료하지 않았다.
[실제 실행·입력 백업 근거](assets/rl_v2_servo_retention_actual_launch_20261007.json).

## 코드·입력·검증

새 checkpoint 형식은 `staged_actual_flap_reanchored_gentle_servo_retention_hybrid_sac_v1`이다.
[성공 보조 손실·pilot](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/servo_success_retention.py),
[공유 제어 변환](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/physical_body_actions.py),
[초기 입력 준비](../scripts/rl/prepare_servo_retention_actor.py)에 구현했다.
기존 RL 설정은 그대로이며 새 형식만 추가 항목을 사용한다. 기존 제어 명령·Q 입력은
정확히 같은 clipping 값을 반환한다. 학습·재생과 Q 영상 exporter도 새 형식을 인식한다.

관련28개 테스트가 통과했다. 실제 pending target·torso 단위, 동등한 포화 명령,
잘못 포화된 명령의 escape gradient, 실제 actor 업데이트와 replay 보존,
새/구형 objective resume 거부를 포함한다. 지표 추가 후 해당9개 테스트도 통과했다.
구형·신형 실제 입력으로 full trainer를 복원해 TRAIN123상태의 초기 평균·jaw·Gaussian,
모든 Q·target·정규화 tensor가 정확히 같고 모든 학습 은행·optimizer가0임을 확인했다.
이123상태는 검사에만 사용하며 초기 모델이나 replay에 넣지 않았다.
[최종 코드의 실제 full trainer 복원](assets/rl_v2_servo_retention_full_restore_20261007.json).

학습 지표의 `success_goal_loss`는 새 합산값이다. 원래 값과 추가 값을 구분하도록
`success_absolute_goal_MSE`, `success_servo_interval_Huber`,
`success_servo_interval_coefficient`, 실제 외부 가중치를 곱한
`success_servo_interval_weight`도 기록한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_servo_retention_actor.py \
  --initial-checkpoint /absolute/path/to/closed-quarter-inputs/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/closed-quarter-inputs/training_manifest.json \
  --waypoints /absolute/path/to/closed-quarter-inputs/waypoints.json \
  --identity-checkpoint /absolute/path/to/closed-matching-actual-TRAIN-checkpoint.pt \
  --output-dir /absolute/path/to/unique-new-input-directory
```

학습한 Q·optimizer·replay를 옮기는 resume가 아니라 미학습 quarter 입력 전용이다.
기존 mean·jaw·Gaussian·제어·보상·무작위화·성공/안전 조건을 보존하며, 실제 학습과
물리 성능은 별도 실행에서 확인해야 한다. 독립 FINAL은 사용하지 않았다.
대표 평가의 [Q 영상4개](RL_V2_EVAL_Q_REWARD_20261006.md)는 이전 정책과 당시 보상이며,
새 보조 손실이나 새 보상 영상으로 소급 표시하지 않는다.

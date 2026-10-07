# 실제 명령의 Q 근거를 보강하는 SAC 비교

## 첫 학습 후 전체 평가: 22/128

10/08 08:36 KST 확인에서 TRAIN384조건 뒤 전체 DEV128이 완료됐다.
성공은 중간 좌7·우12, 상단 좌0·우3의 **22/128**이며 랙 충돌65·시간 초과40·
초기 무효1건이다. 평가와 정확히 같은 actor699회·Q4,844회 모델의 checksum·
유한성·실제 goal 계약을 확인했다. **초기35→22회라 초기보다 개선되지 않았다.**
같은 TRAIN384 뒤 일반 SAC의19회보다3회 많지만 한 번의 평가 차이이며
flap 추첨·접촉 이력도 동일하지 않아 반복 가능한 CQL 효과로 단정하지 않는다.
양쪽이 모두 유효한 초기 비교116조건에서도35→19회였다.
현재 새 TRAIN 배치를 학습 중이며 여섯 박스 조합 전체 성공을 뜻하지 않는다.
[전체 결과·같은 평가 모델·원래 요청 비교](assets/rl_v2_support_conservative_first_learned_full_DEV22_20261008.json).

## 실제 실행 상태

2026-10-08 05:48 KST에GPU3의 별도 실행을 시작했다. 실제 writer2318510의 소유자·
고유 실행 경로·`CUDA_VISIBLE_DEVICES=3`을 확인했다. Source는main에 push한
`ddef524881a0864c2c8677526653c00ab40a0d30`이며 실행 폴더는
`GPU3_regional_support_conservative_SAC128_20261008_054847/`
`batch_sac_20261008_054847_43dca8`다. 같은 초기 정책의 전체 DEV128을 먼저 실행하고
새 TRAIN1,536조건 사이에 전체 DEV를 반복한다. 기존 두 SAC writer는 유지했다.

실제 전체 초기 DEV128을 완료했고 **35/128(중간 좌16·우13, 상단 좌0·우6)**이었다.
랙 충돌49·시간 초과33·초기 무효11건을 원래 분모에 포함했다. Actor·Q·replay0인
학습 전 기준이며 기존 regional 실행의 초기35건과 같았다. **새 Q 보강의 학습 후
성능 개선은 아직 미확인이다.** 실제 복원에 사용한 pristine 입력과 SHA256·
agent 계약을 확인해 보존했으며 runtime 모델을 직접 dump한 것이라고 표시하지
않았다. 전체 TRAIN으로 진행했고 후속 모델은 별도 관찰자로 보존한다.
[전체 초기 평가·입력 모델·원래 요청 비교](assets/rl_v2_support_conservative_full_initial_DEV35_20261008.json).

06:35 KST의 실제 TRAIN 진행은Q724회·온라인43,091행·actor0회였다. 원래
Q warmup2,048회를 진행하는 구간이다. 실제256행 Q 배치에서 관측당32개
critic 후보와 가중치1.0이 적용됐고, regularization 손실−3.5914와 합산Q손실
−1.5753은 유한했다. 이 항은 후보 평균Q−실제Q의 차이이므로 음수일 수 있으며
손실의 부호를 성공률 개선으로 해석하지 않는다. 합성 replay 행은0이고 평가
행을 학습하지 않았다. Actual actor 업데이트와 첫 학습 후 전체 DEV는 대기 중이다.

06:15 KST에 기존
구역별 실행의 두 번째 전체 DEV는23/128로 끝났다. **35→19→23/128**은 직전보다
일부 회복했지만 초기보다 낮고, 이 새 Q 보강 실행의 결과가 아니다.
[기존 실행의 두 번째 전체 평가](RL_V2_REGIONAL_INITIAL_DEV35_20261008.md).

기존 Drive 연결의`about`을 재확인했으나`invalid_grant`였다. 관리자는 기존 연결로
300초 업로드를 재시도하며 미검증 파일을 보존한다. 로컬 여유는 실행 전 root 약
163GiB·RAM 저장소 약254GiB였으며, 새 인증이나 미검증 원본 삭제는 수행하지 않았다.

## 비교를 시작한 이유

추가 SAC 업데이트가 초기 성공 동작을 유지하지 못하고 있다. 같은 전체 개발
평가에서 새 구역별 실행은35→19/128이었다. 회귀 모델의 실제TRAIN 성공 표본에서
Q 경사와 성공 명령 유지 경사가 반대였고, Q 경사의 크기가2.5..3.4배였다.
이는 성공 경로 부분집합의 진단이며 실제 production replay·Adam 갱신의 전체
원인을 확정한 것은 아니다.
[전체 평가·동작·경사 근거](RL_V2_REGIONAL_INITIAL_DEV35_20261008.md).

## 무엇을 바꾸는가

실제로 수행하지 않은 명령의 Q가 실제 TRAIN 명령보다 높아지는 정도를
critic 학습에서 억제하는 별도 옵션을 추가했다. 같은 실제TRAIN 배치의 관측에서
기존 body correction 범위 안의 uniform4개와 현재 정책 body4개를 뽑고, 각 명령의
물리적인 binary jaw4조합을 기존 근접 gate로 투영한다. 한 관측당32개 Q 질의다.
far jaw는 모두 open으로 투영되며 동일한 질의를 복제해도 정규화 평균은 같다.

두 Q 각각에 다음 항을 더한다. 온도T=1, 가중치λ=1이다.

```text
λ × mean_s[ T × logmeanexp_a(Q(s,a)/T) − Q(s,실제 수행 명령) ]
```

Q 입력은 기존 실제 servo 명령 인코더를 사용한다. 후보는 critic 질의이며
시뮬레이터 경험·reward·replay 전이를 만들어 넣지 않는다. 후보 생성에는 policy
gradient를 연결하지 않으며 일반 SAC의 actor·entropy·성공 명령 유지 손실과 실제
one-step·완료TRAIN 누적 보상 학습은 유지한다.

[CQL 원 논문](https://arxiv.org/abs/2006.04779)의 Q 정규화 방향을 참고한
**유한 후보 mixture 실험**이다. 이 온라인·제한된 후보 구현에 원 논문의 offline
성능 보장이나 정확한 값의 하한 정리를 적용했다고 주장하지 않는다.
[원 구현의 Bellman 손실과 Q 정규화](https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py).

## 초기 모델·데이터 보존과 검증

새 artifact는`staged_actual_flap_regional_support_conservative_sac_v1`이며 기존
네 구역 actor의 초기 동작·모든 Q/target·관측 정규화·optimizer와 옛 actor-only
성공TRAIN 기억12,476행을 그대로 보존한다. Actor·Q·온라인 replay·성공·누적
보상 은행은0이다. 기존에 학습된 Q·replay·평가 reward를 옮기지 않는다.
일반 SAC에는 이 penalty를 적용하지 않는다. 저장·복원·영상/Q 평가도 동일한
선택형 artifact와 계약을 확인한다.

관련22개 테스트가 통과했다. 실제 trainer를GPU3에 복원했고 CPU의 동일한
과거TRAIN1,259관측에서 초기 greedy 명령은 bit-identical이며 GPU 결과는
1e−5 이내로 일치했다. 실제64×32개 후보가 bounded·binary·finite·no-grad임을
확인했고, checkpoint·경험 round-trip과 모든 은행0을 확인했다.
[실제 trainer 복원과 초기 동작](assets/rl_v2_support_conservative_fulltrainer_preparation_20261008.json).

별도로 닫힌 우리TRAIN 성공64행에 실제 servo Q 인코더와 penalty를 적용해
두 Q의 유한한 경사와 actor gradient0을 확인했다. 이 표본은 읽기 전용 진단이며
새 학습에 넣지 않았고 optimizer를 갱신하지 않았다.
[실제 servo Q 경사 진단](assets/rl_v2_support_actual_servo_Q_gradients_20261008.json).

## 실행 범위와 전체 목표

이 비교는 원래 전체 DEV128·새TRAIN1,536조건의 small 정책을 같은 초기 동작에서
비교하기 위한 단계다. **small 성능만으로 전체 목표를 완료하지 않는다.** 중간
좌·우small/medium과 상단 좌·우small의 여섯 조합을 위한 수정된 실제TRAIN 접근
측정과 시작 조건이 겹치지 않는 재확인을 별도로 진행한다. 검증한 크기별
waypoints를 붙인 pristine regional 입력에도 같은 penalty를 연결할 수 있다.

박스·base·배경·움직이는 firm flap의 무작위화와 원래 성공·안전 기준을 유지한다.
Curriculum·박스 고정·좁힌 시작 범위·완화된 종료 조건을 추가하지 않는다.
접근 이동은 기존 제어기가 맡고 SAC는 이후 양팔·상체 목표19개와 그리퍼2개를
조정한다. 손대지 않은 독립FINAL과 여섯 조합의 안정적인 성공은 아직 미확인이다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_support_conservative_actor.py \
  --initial-checkpoint /absolute/path/pristine_regional/checkpoint_00000000.pt \
  --training-manifest /absolute/path/pristine_regional/training_manifest.json \
  --waypoints /absolute/path/pristine_regional/waypoints.json \
  --training-waves /absolute/path/original_full_waves.json \
  --output-dir /absolute/path/unique_support_initialization
```

입력의 실제 trainer 복원 뒤 기존`batched_staged_goal_with_drive.py`에 새 checkpoint·
waypoints·training-manifest·waves를 넘긴다. GPU3은`CUDA_VISIBLE_DEVICES=3`과
learner`cuda:0`으로 격리한다. 기존 Drive 연결·300초 업로드·검증된 오래된
checkpoint만 정리·최신2개 보존을 재사용한다. 인증 실패 시 미검증 파일은 보존한다.

초기 평가와 학습 후 같은 전체128요청을 비교하며 초기 배치 무효와 실패도
분모에 포함한다. 실험 연결·테스트·경사 유한성은 파지 성능 개선의 증거가 아니다.

# 학습된 초기 몸체 목표를 기준으로 SAC 조정 범위 확대

이 제어 변경 전에는 상단 오른쪽 파지를 성공하지 못했다. 종료된 훈련 기록에서는 왼손이
충분히 접근하지 못했고, 학습 후 팔 출력이 범위 끝에 몰려 실제 목표의 탐색
변화가 작아졌다. [기구학 진단](RL_V2_LOCAL_GOAL_RANGE_DIAGNOSIS_20261007.md)은
학습된 초기 body goal을 기준으로 조정 범위를 넓혀볼 근거를 제시했다.
그 계산은 접촉·충돌을 검증한 물리 성공이 아니며 IK 해를 학습 데이터에 넣지 않는다.

## 변경한 제어 방식

별도 체크포인트 형식 `staged_actual_flap_reanchored_body_correction_hybrid_sac_v2`를
추가했다. 기존 설정과 이전 replay의 의미는 유지한다.

1. 실제-flap actor689의 **19개 몸체 목표**를 동결한 기준으로 사용한다.
   팔·상체·머리의 실제 제어 목표를 관측에서 계산하는 신경망이다.
2. 새 SAC는 기준 목표 주변의 **정규화 반경0.30** 안에서 조정한다.
   실제 단위는 기존 관절별 center·scale로 변환한다. 관절의 물리 한계는 유지한다.
   경계에서는 양쪽 여유가 작은 만큼 반경을 줄이며 clipping으로 확률 질량을 만들지 않는다.
3. 새 body mean 출력19개를0으로 시작해 초기 greedy 몸체 목표를 보존한다.
   기존 trunk·그리퍼 logits·logstd와 actor 정규화를 유지하고,
   radius 관측 값의 변경만 정규화 평균에 반영한다.
4. 양손의 독립 열림/닫힘과 원래 jaw prior·production near gate는 유지한다.
   접촉 없이 닫는 명령만으로 보상을 받지 않는다.

기존0.15 제어가 demo에서 준비한 nominal 목표 주변을 조정했다면,
새 제어는 **이미 학습된 actual-flap 목표 주변**을 조정한다.
명목 목표의 반경만 두 배로 바꾸는 방법과 다르다.
기존0.15 경로는 기본으로 유지하며 새 형식은 명시적으로 초기화해야 한다.

## 학습 상태와 검증

제어 방식이 달라지므로 Q·target Q·critic 정규화·entropy·네 optimizer·replay·
성공 bank를 새로 시작한다. Source의 actor와 actor-only anchor만 사용하며
과거 transition을 새 의미의 action으로 재표시하지 않는다.

- 관련 CPU 테스트 **50개**: 기존 경로 보존, 초기 몸체·jaw 보존,
  넓어진 sampling과 고정 경계의 entropy, 새 replay·네 optimizer 복원,
  기존 controller/replay의 잘못된 재개와 Q가 섞인 anchor 거부를 확인한다.
- 실제 성공 TRAIN 관측 **538개**: 이전 목표21개와 새 초기 greedy 목표가
  bit 단위로 같고 gripper 명령 불일치0개다. 보존된 실제 controller telemetry로
  decode한 물리 명령24개도 같다. 이 확인에 DEV·FINAL 관측은 쓰지 않았다.
- 저장 후 다시 복원한 checkpoint에서도538개 목표·jaw가 같다.
  Q 영상 도구가 새 controller를 정확히 복원하고 두 critic에 대응하는 목표를 사용한다.
- 초기 actor·Q 업데이트0회, replay·성공 bank0개, optimizer state0개다.
  위 검증은 학습 구동 조건 확인이며 새 파지 성공이나 성능 개선이 아니다.

[실제 입력·저장 복원 근거](assets/rl_v2_reanchored30_initial_control_identity_20261007.json).

## 실제 비교 계획

**첫 TRAIN128개 탐색에서 상단 오른쪽 실제 성공1건을 확인했다.** 구역별
성공은3/3/1/1, 전체8/128이다. 상단 오른쪽 env47·seed2250700311에서 서로
다른 flap의 양손 파지를0.267초 유지하고 지지면에서3.35cm 들었다.
접촉 품질1.0, 안전 위반 없음, 유효한 박스 quaternion과 정상 속도를 확인했다.
Base 시작점은 구역 기준 좌우+5.38cm·바깥쪽+5.94cm·yaw−3.29°였으며,
박스·base·배경·동적 flap 무작위화와 성공·충돌 기준을 유지했다.

이 성공은 새 actor 갱신 전에 초기 정책의 stochastic 탐색에서 얻었다.
새 greedy 평가 성공률이나 이미 학습한 일반화 성능으로 세지 않는다.
어떤 탐색 성분이 성공을 만들었는지 또는 제어 변경만의 효과인지 단정하지 않는다.
실제 파지 단계590행이 TRAIN 성공 bank에 유지됐고 네 구역을 균형 있게
샘플링한다. 구역별4,096행을 따로 보관해 다른 구역의 성공이 이를 밀어내지 않는다.
다음 DEV128에서 정책이 이 성공을 재현하는지 확인한다.
[실제 파지·변경된 시작점·성공 경험 보존](assets/rl_v2_reanchored30_first_upper_right_TRAIN_success_20261007.json).

이후 actor290/Q3206의 닫힌 체크포인트에서 같은 성공590개 TRAIN 상태를
확인했다. 실제 명령과의 근접 그리퍼 불일치가17→0회로 줄었지만 정규화 몸체
목표 차이는 개선되지 않았다. 후속 정책을 물리에 실행한 결과가 아니므로
새 평가 성공으로 세지 않는다. [동일 성공 상태의 학습 변화·그림·Q 해석](RL_V2_SUCCESS_PATH_LEARNING_20261007.md).

GPU3에서 같은128환경·원래 DEV128요청·별도의 새 TRAIN1,536조건을 사용한다.
박스·base 시작 위치·배경·단단한 동적 flap 무작위화와 성공·안전 기준을 유지한다.
제어 반경과 기준 정책의 차이를 제외하면 정밀 보상25mm, 팔 episode bias20%,
그리퍼 조합 탐색30%, body mean3 완화, 실제 TRAIN n-step16 credit 설정을 유지한다.
독립 FINAL 배치는 개발 비교에 사용하지 않는다.

기존 연결로 고유 RAM 실행 입력을 먼저 백업·검증한 뒤 관리자를 실행한다.
체크포인트는5분마다 업로드하고 checksum 검증이 끝난 오래된 것만 정리한다.
최신2개 및 형식별 최신 검증2개를 보호하며 종료 후 닫힌 로그·데이터도 검증한다.
입력 백업이나 이 문서의 존재만으로 학습 시작·성공을 판단하지 않는다.

**10/07 00:55 KST에 GPU3 비교 실행을 시작했다.** 입력8개의 기존 Drive
크기·MD5 검증 후 고유 RAM 실행 폴더를 만들었다. Writer2701747의 소유자·
실행 폴더·`CUDA_VISIBLE_DEVICES=3`을 확인했고 기존 비교 writer3개는 유지했다.
실행 시작 시에는 초기 환경 준비 단계였으며, 새 학습의 전체 평가나 성능 개선은 아직 확인 전이다.
Source commit은`88ba2a5`다. [실행 시작 근거](assets/rl_v2_reanchored30_GPU3_startup_20261007.json).

초기화 후 실제 DEV0 평가151step까지 진행했다. 새 controller·반경0.30·38차원
실제 flap 관측·정밀 capture25mm가 연결됐고 actor/Q 업데이트·온라인 전이·replay·
성공 bank는 모두0이었다. TRAIN용 팔·jaw 탐색의 카운터도0으로 평가에 들어가지 않았다.
이는 실제 제어 진입 확인이며 전체128건 결과나 새 학습 성능은 아직 아니다.
[첫 실제 평가 제어 확인](assets/rl_v2_reanchored30_first_actual_DEV_progress_20261007.json).

**전체 초기 DEV0는23/128(중간 좌12·우6, 상단 좌5·우0)**이었다.
23건 모두 실제 양손 파지·유지·들기와 안전 조건을 확인했다. 초기화 무효11건도
원래128개 분모에 남겼다. 나머지는 랙 충돌69·시간 초과25건이며, 상단 오른쪽은
초기 유효29건 중 랙 충돌24·시간 초과5건이었다. Actor·Q 업데이트와 replay는
평가 종료까지0이었다. 원래 DEV128 배치 요청과 보상·flap 물성 범위는 같지만
flap의 실제 추첨과 solver 이력까지 일치한 비교라고 가정하지 않는다.
다른 초기 평가29/128을 이 실행의 기준으로 사용하지 않으며, 새 학습의 개선은
**자신의 초기23/128**과 비교한다. [전체 초기 평가](assets/rl_v2_reanchored30_full_initial_DEV_20261007.json).

이후 실제 TRAIN1 step421에서 **Q720회·held TRAIN42,917전이**를 확인했다.
127에피소드 중21개에 추가 팔 탐색이 적용됐고 실제 수집6,840행에 반영됐다.
반경0.30·정밀 보상·body mean3·그리퍼 탐색·실제 n-step16 설정은 연결됐다.
Q warmup2,048회 중1,328회가 남아 actor는 아직0회다. 이전 Q·replay를 가져오지
않았으며, 이 수집·Q 업데이트 확인을 물리 성공률 개선으로 세지 않는다.
[첫 실제 TRAIN·Q warmup](assets/rl_v2_reanchored30_first_real_TRAIN_Q_20261007.json).

TRAIN2 step301에서는 Q2,094회·실제 held96,346행을 거쳐 **actor12회**로
실제 정책 갱신을 시작했다. Q·수집 warmup을 모두 통과했고 actor 손실은 유한했다.
Mean3 설정은 적용됐고 현재 배치의 최대 평균 절댓값0.051·포화 좌표0개였다.
실제 TRAIN n-step16 신호와 실제 성공 경험의 약20% 샘플링도 연결됐다.
이는 학습 연결 확인이며 첫 학습 후 전체 평가 성능은 아직 확인 전이다.
목표 엔트로피 바닥의 가상 진단은 현재 실행에 적용하지 않았다.
[첫 실제 actor 갱신](assets/rl_v2_reanchored30_first_real_actor_updates_20261007.json).

## 첫 학습 후 전체 개발 평가

새 TRAIN384조건 후 actor692·Q4816의 원래128개 평가에서는
**23→4/128**로 떨어졌다. 네 구역 성공은 중간 좌3·우0, 상단 좌1·우0이며
랙 충돌96·낙하3·과도한 들기1·시간 초과23·초기화 무효1건이었다.
탐색 중 상단 오른쪽을 성공했어도 greedy 정책이 재현하지 못했다.
[전체 평가 근거](assets/rl_v2_reanchored30_first_full_DEV_after384_20261007.json).

이 결과와 성공 상태의 몸체 목표 경사를 토대로
[실제 성공15개를 연결하고 파지 직전 actor 표본을 보강한 새 비교](RL_V2_SUCCESS_ACTOR_TAIL_20261007.md)를
실행했다. 보상·무작위화·안전 기준을 유지하며 별도 실행의 결과와 구분한다.

## 이후 전체 개발 평가

TRAIN768조건 뒤 두 번째 평가는 **15/128(중간 좌6·우7, 상단 좌1·우1)**이었다.
모든128조건이 유효했고 랙 충돌63·낙하13·시간 초과37건이었다.
[두 번째 전체 평가](assets/rl_v2_reanchored30_second_full_DEV_after768_20261007.json).

TRAIN1,152조건 뒤 세 번째 평가는 **23/128(중간 좌13·우8, 상단 좌1·우1)**이었다.
전체 흐름은 **23→4→15→23/128**로 자신의 초기 성공 수와 같아졌지만 넘지는 못했다.
상단 양쪽의1/32 성공을 안정적인 일반화로 보지 않는다. 초기화 무효1건도 원래128개
분모에 남겼으며 랙 충돌63·낙하10·과도한 들기1·시간 초과30건이었다.

평가에 일치하는 actor3,114·Q14,502 체크포인트를 종료 전에 별도 보존하고,
실제 양손 접촉·유지·지지면 clearance·안전 조건을23건 모두 확인했다.
랙 충돌63건은 모두 base 접근 이후 파지 단계였다. 상단 오른쪽 충돌29건의
종료 순간 최대 힘 링크는 오른쪽 그리퍼 본체26·오른팔3건이었다.
이는 첫 접촉이나 전체 접촉 이력에 대한 원인 판정은 아니다.
남은 TRAIN384조건과 마지막 개발 평가를 이어가며 독립 FINAL은 남긴다.
[세 번째 전체 평가·일치 모델·종료 단계](assets/rl_v2_reanchored30_third_full_DEV_after1152_20261007.json).

## 사용법

이전 실제-flap actor와 일치하는 manifest·waypoint·native seed를 사용한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_actual_flap_reward_actor.py \
  --checkpoint /absolute/path/to/learned-actual-flap-checkpoint.pt \
  --training-manifest /absolute/path/to/matching-training-manifest.json \
  --waypoints /absolute/path/to/matching-waypoints.json \
  --native-seed /absolute/path/to/middle-native-seed.hdf5 \
  --native-seed /absolute/path/to/upper-native-seed.hdf5 \
  --body-controller reanchored30 --initialization-seed 20261007 \
  --output-dir /absolute/path/to/unique-reanchored-inputs
```

Runner는 새 checkpoint의 형식으로 controller를 선택한다. 실제-flap TRAIN
옵션과 Drive 관리자는 기존 진입점을 사용한다. Q 영상 exporter도 같은
형식을 인식한다. 기본0.15 checkpoint는 새 형식의 continuation으로 받지 않는다.

초기화: [prepare_actual_flap_reward_actor.py](../scripts/rl/prepare_actual_flap_reward_actor.py).
제어·형식: [actual_flap_reanchored_sac.py](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/actual_flap_reanchored_sac.py).

# 학습된 초기 몸체 목표를 기준으로 SAC 조정 범위 확대

상단 오른쪽 파지는 아직 성공하지 못했다. 종료된 훈련 기록에서는 왼손이
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
현재 초기 환경 준비 단계이며 새 학습의 전체 평가나 성능 개선은 아직 확인 전이다.
Source commit은`88ba2a5`다. [실행 시작 근거](assets/rl_v2_reanchored30_GPU3_startup_20261007.json).

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

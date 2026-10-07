# 실제 정책 동작과 팔 탐색을 함께 수집하는 SAC

성공 보상+64 비교의 첫 TRAIN256조건에서는 안전한 성공이1건뿐이었다.
같은 실행의 초기 greedy DEV는15/128이었다. TRAIN과 DEV의 시작 조건이 달라
잡음만이 원인이라고 단정할 수 없지만, 실제 성공 경험을 충분히 수집하는지도
점검해야 한다. 학습된 동작을 실행하는 경로를 늘리는 선택형 수집 옵션을 추가했다.

## 변경한 수집 방법

`--body-behavior arm20-explore-rest-greedy`를 선택하면 실제 TRAIN episode 중
20%는 기존의 넓은 팔 탐색을 사용하고, 나머지는 **현재 학습 중인 actor의
greedy 몸체 목표·binary 그리퍼**로 움직인다. VR 시연이나 teacher 경로를
재생하는 방법이 아니다. Actor가 갱신되면 greedy 경로의 정책도 함께 바뀐다.

| 경로 | 동작 |
| --- | --- |
| 탐색 episode 약20% | 기존 AR0.99 Gaussian, 팔 bias·90 held-step ramp, joint-jaw 탐색 유지 |
| 나머지 episode 약80% | 같은 actor의 평균 몸체 목표와 greedy 그리퍼, 추가 잡음 없음 |
| 모드 선택 | 파지 단계에 들어갈 때 기존 팔 탐색 선택을 재사용하고 episode 끝까지 유지 |
| 평가 | 기존 greedy DEV 그대로, 평가 전이를 학습 bank에 넣지 않음 |
| SAC 학습 | stochastic actor 목적함수·Q target·손실 유지; 실제 실행한 TRAIN 전이를 학습 |

새20% 추첨을 기존20% 팔 선택과 독립적으로 곱하지 않는다. 그렇게 하면 넓은
팔 탐색이4%로 줄기 때문이다. 기존 `ramped-arm-bias20` 및 옵션을 생략한
수집 동작은 유지한다. 박스·base·주변 박스·단단한 동적 flap의 무작위화,
성공+64·10cm 상대 낙하 판정·다른 보상·충돌10N/5N·제어 반경0.30은 그대로다.
Curriculum은 추가하지 않았다.

## 복원과 최소 검사

관련45개 검사가 통과했다. Greedy 몸체·binary jaw가 정확하고 추가 잡음이
탐색 환경에만 적용되며, 끝난 환경을 제외해도 남은 episode의 모드는 유지된다.
기존 RNG의20% 팔 선택·bias·ramp와 기본 경로도 검사했다.

실제 full training pilot을 GPU/Isaac 없이 복원했다. 실제 TRAIN1,240상태의
초기 actor4337 목표·그리퍼, 모든 모델·네 optimizer·물리 계약이 같고,
actor/Q 카운터와 온라인·성공·n-step reward bank가0이었다. 실제 관측64개로
수집 함수를 확인했을 때 **greedy50개·탐색14개**가 선택됐다. 이는 함수 호출
검사이며 새 물리 학습이나 성공률 결과가 아니다.
[실제 복원·계획·수집 함수 근거](assets/rl_v2_greedy80_fulltrainer_and_plan_20261007.json).

새 TRAIN1,536조건은 seed2350700000으로 다른14개 계획과 겹치지 않는다.
원래 DEV128개를 구역별32개씩 사용하는17-wave 계획이며 독립 FINAL은 남긴다.
새 실행의 실제 시작·성능은 아래 실행 확인에 별도로 기록한다.

## 사용법과 지표

새 Q·optimizer·reward bank가 빈 입력만 아래 준비 도구에 사용한다.
학습한 Q나 기존 reward 전이를 다른 계약으로 바꿔 재개하는 도구가 아니다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_greedy_collection_actor.py \
  --initial-checkpoint /absolute/path/to/pristine-success64-drop10-inputs/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/pristine-success64-drop10-inputs/training_manifest.json \
  --waypoints /absolute/path/to/pristine-success64-drop10-inputs/waypoints.json \
  --output-dir /absolute/path/to/unique-new-inputs
```

기존 batched staged SAC 실행에 `--body-behavior arm20-explore-rest-greedy`를
지정한다. `TRAIN_policy_mode_statistics`에 선택된 탐색/greedy episode·held-row
수를 기록하고 닫힌 episode의 JSON·HDF에 `collection_policy_mode`를 남긴다.
Manifest/checkpoint/replay에는 수집 설정과 기존 통계가 함께 보존된다.
학습된 Gaussian 표준편차를 줄이거나 그리퍼를 강제로 닫는 변경은 아니다.

## 실행 확인

**10/07 13:29 KST에 GPU3에서 별도 고유 실행을 시작했다.** 실제 writer694668과
supervisor의 소유자·명령·`CUDA_VISIBLE_DEVICES=3`, 서비스 실행 상태와 새 수집
옵션을 확인했다. 기존 우리 writer5개와 다른 사용자의 프로세스는 유지했다.
[실제 시작·명령·백업 범위](assets/rl_v2_greedy80_actual_launch_20261007.json).

입력 checkpoint·계약7개는 기존 Drive 연결로 크기·MD5를 검증했다. 실행도
checkpoint·계약·닫힌 로그만300초마다 검증·업로드하며 최신2개를 유지한다.
Raw replay·HDF·영상은 로컬에 남긴다. 실제 환경 초기화·첫 DEV 및 실제 TRAIN
모드 적용은 확인 전이며 새 성능도 아직 없다. 초기 동작을 더 실행한다고 네
구역의 일반화나 대표30/128을 넘는 성능이 보장되는 것은 아니다.

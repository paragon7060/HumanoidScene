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
Raw replay·HDF·영상은 로컬에 남긴다. 이후 **실제 첫 DEV에 진입했다.** 초기화된
환경의10cm 낙하 조건과 reward manager의+64, 실제 agent·checkpoint·17개
wave의 계약 일치, 새 수집 설정과 actor/Q·학습 은행0을 확인했다. 평가 동안
수집 모드의 episode/row 수는0이었다. 이후 전체 초기 DEV128은 **13/128
(중간 좌6·우6, 상단 좌1·우0)**으로 완료됐고 실제 TRAIN에 진입했다. 박스 낙하39·
랙 충돌53·시간 초과12·초기 무효11건을 원래 분모에 포함했다. 일치하는
초기 actor/Q0 모델과 원래128조건·실제 양손 파지·유지·들기 성공을 확인했다.
수집 방식이 적용되기 전의 기준 성능이며 학습 개선은 아니다. 다른 실행과
reset 물리가 같다고 가정하지 않는다. 학습 후 새 성능은 아직 확인 전이다.
[첫 실제 DEV·환경·agent 적용](assets/rl_v2_greedy80_first_actual_DEV_20261007.json).
[전체 초기 평가·일치 모델](assets/rl_v2_greedy80_full_initial_DEV_20261007.json).

**14:11 KST에는 실제 TRAIN 수집 모드도 확인했다.** 첫95개 episode 추첨에서
탐색14개·현재 정책의 greedy 동작81개였고, 파지 단계의 수집 행은 각각225·1,319개였다.
20%는 episode 선택 확률이며 이 부분 관찰의 비율이 정확히20%라는 뜻은 아니다.
기존 팔 탐색과 같은 선택 마스크·실제 agent/manifest 설정·누적 행 통계의 일치를
확인했다. Q62회·온라인 TRAIN1,618행이며 actor는 아직0회였다. 성공·n-step
학습 은행에 평가 행이 없었다. 설정이 실제 수집에 적용된 확인이며, 전체 첫
TRAIN128의 성공률이나 이후 greedy 성능이 개선됐다는 결과는 아니다.
[실제 TRAIN 모드·계약·Q 갱신](assets/rl_v2_greedy80_actual_TRAIN_mixture_20261007.json).

**첫 전체 TRAIN128을 마친 결과는 안전한 성공13건**이었다. 중간 좌5·우6,
상단 좌2·우0이며 모든 초기 배치가 유효했다. 현재 actor의 greedy 동작109개
episode에서 성공13건·위험 종료85건·시간 초과11건, 팔 탐색19개에서는 성공0건·
위험 종료18건·시간 초과1건이었다. 서로 다른 flap의 실제 양손 pad 접촉·유지·
들기와 원래 TRAIN 요청128개를 대조했다.

닫힌 actor0·Q1594 모델의 자기 성공 은행에는13경로·5,717전이가 보존됐다.
성공 종료의 실제 보상과 bootstrap 없는 같은 critic 목표가 각각약62.995–62.997로
일치했다. 초기 actor·jaw·정규화는 그대로였다. **초기 정책으로 성공 경험을
수집한 결과이며, 학습 후 greedy 평가 개선은 아직 아니다.** 기존 Gaussian
수집의 첫 TRAIN3건과 seed가 달라 수집 변경의 인과 효과로 단정하지 않는다.
평가 전이를 학습 은행에 넣거나 진행 중 HDF/replay를 읽지 않았다.
[완료한 첫 TRAIN·모드별 결과·실제 보상 연결](assets/rl_v2_greedy80_first_closed_TRAIN_20261007.json).

## 첫 학습 후 전체 평가 · 10/07 15:50 KST

TRAIN384조건 후 전체 DEV는 **초기13→1/128(중간 좌0·우0, 상단 좌1·우0)**로
줄었다. 초기화가 양쪽 모두 유효한116조건에서도13→1건이었다. 랙77·박스
낙하32·시간 초과17·초기 무효1건을 원래 분모에 포함했고 같은 actor692/Q4814
모델을 보존했다. 성공 경험13개를 모은 뒤에도 학습 후 정책은 그 성능을 유지하지
못했다. 수집량 확대만으로 해결되지 않았으며 서로 다른 TRAIN seed의 비교이므로
greedy 수집 방식 하나가 성능 저하의 원인이라고 단정하지 않는다.
[첫 전체 학습 평가·일치 모델·짝 비교](assets/rl_v2_resetdrop_success64_greedy80_first_full_DEV_20261007.json).

## 실제 성공13경로의 학습 후 명령·Q 변화

첫 TRAIN에서 실제 실행한 초기 actor0/Q1594와 첫 DEV의 actor692/Q4814를
원래 성공13경로·5,717행에서 비교했다. 진행 중 HDF/replay는 읽지 않았다.
초기 정책이 실제 실행한 명령은 같은 물리 decoder로 다시 계산한 값과 일치했다.
다음 값은 **정규화한 제어 명령의 차이**이며 m나 rad가 아니다.

| 구역·경로 수 | 초기16동작의 몸체 명령 MAE·초기→학습 후 | 파지 직전64동작의 몸체 명령 MAE·초기→학습 후 | 파지 직전 양쪽 jaw 일치율·초기→학습 후 |
| --- | ---: | ---: | ---: |
| 중간 왼쪽·5경로 | 약0→0.319 | 약0→0.415 | 100→71.2% |
| 중간 오른쪽·6경로 | 약0→0.218 | 약0→0.311 | 100→99.2% |
| 상단 왼쪽·2경로 | 약0→0.458 | 약0→0.334 | 100→100% |

몸체 명령은 초기 접근부터 달라졌고 중간 왼쪽은 jaw도 유지되지 않았다.
같은 실제 성공 종료 명령의 보상은약62.996인데, 학습 후 Qmin 평균은 각 구역
31.97·32.79·32.30이었다. 초기 Q의3.75–5.27보다는 커졌지만 마지막 동작은
bootstrap 없이 실제 마지막 보상 자체가 목표여서 큰 오차가 남았다.
이13개 성공의 진단만으로 전체 Q 정확도를 단정하지 않는다.

학습 후 정책의 대안 명령은 이 과거 상태에서 실제 실행하지 않았다. 같은
학습 후 Q는 파지 직전 대안 명령에 기록된 명령보다 평균0.02–0.37 큰 값을
줬지만, 이 값은 추정치이며 그 대안의 실제 return이나 물리 실패 원인의 증명이
아니다. 별도의 전체 DEV13→1은 위에서 확인한 실제 성능이다.

이+64 비교와 새 시간 입력 비교는 기존 절대 목표 Q 입력을 유지한다.
현재 최고33/128의 성공 유지 정책은 실제 servo 명령을 Q에 입력하는 별도
계약이다. 진단에서는 두 모델의 원래 Q 입력을 그대로 두고, 물리 decoder는
명령 차이를 재는 데만 사용했다. 단순 수집 확대 외에도 **성공 가치의 전파와
초기 접근을 포함한 동작 유지**를 검토해야 한다.
[닫힌 동일 경로·두 정확한 모델·명령과 Q 근거](assets/rl_v2_greedy80_actor692_first13_success_path_diagnostic_20261007.json).

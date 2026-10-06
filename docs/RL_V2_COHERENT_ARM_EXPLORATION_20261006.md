# 양손 파지 SAC: 연속적인 팔 자세 탐색 비교

## 문제와 가설

최신 주 학습은 같은 개발 평가에서 15→29→22/128 성공했다.
상단 오른쪽은 모두 실패했다. 업데이트한 정책의 새 TRAIN 배치16개에
진입 위치8후보를 적용한 실제 비교도 상단 오른쪽 성공0이었다.
유효30경로의 안전한 관측에서 왼손의 실제 자격을 갖춘 pinch는0회였다.
이는 닫기 명령만의 문제가 아니라 팔·손 자세와 실제 접촉도 남았다는 근거다.
최근접 거리나 rigid TCP의 명목상 축만으로 실제 파지를 판단하지 않는다.

기존 연속 관절 탐색의 초기 pre-tanh 표준편차는0.08이며 허용 범위는0.03..0.12다.
예를 들어 관측된 왼팔 첫 관절의 가용 반경16.78°에서 국소적인 탐색 폭은
약1.34°이고, 손목 마지막 관절 반경5.35°에서는 약0.43°다.
이는 tanh 평균이0 근처인 경우의 근사이며 도달 불가능성의 증명이 아니다.
허용된 관절 목표 영역 안에서도 기존 자세에서 멀리 떨어진 연속 동작을
충분히 시도하지 않았을 가능성을 비교한다.

## 이번 변경

`--body-behavior ramped-arm-bias20`은 **훈련 에피소드의20%를 확률적으로 선택**한다.
선택한 환경은 양팔14관절의 pre-tanh 공간에 한 번 뽑은 방향을 유지한다.
편향은 표준편차0.8의 독립 Gaussian으로 뽑아 절댓값1.6으로 제한한다.
held 파지 시작 후90 제어 스텝에 걸쳐 `u²(3−2u)`로 천천히 반영한다.
이는 갑자기 매 스텝 전혀 다른 자세를 내는 탐색이 아니다.
종료된 actor1899의 상단 오른쪽 랙 충돌27경로는 held 시작334..552스텝 이후
종료됐다(중앙값361). 이 진단에서는90스텝 ramp가 적용되기 전에 종료된 사례가
없었다. 새 탐색의 충돌 감소 효과를 입증하는 수치는 아니다.

| 항목 | 처리 |
| --- | --- |
| 선택되지 않은80% 에피소드 | 기존 행동 분포 유지 |
| 선택된20% 에피소드 | 에피소드당 고정 방향의 팔 편향을 점차 반영 |
| 기존 관절·그리퍼 잡음 | AR1 상관0.99와 joint-jaw epsilon30 유지 |
| base·머리·torso | 추가 편향0; 원래 제어와 탐색 유지 |
| 관절 목표의 허용 범위 | 기존 anchor와 대칭 가용 반경0.15 유지 |
| 학습 대상 | 실제로 실행한 투영된 목표와 실제 관측·보상 |
| actor·Q target·평가 | 추가 팔 편향을 사용하지 않음 |
| 기본값 | 미지정 또는`off`; 기존 동작 유지 |

20%는 에피소드 선택 확률이다. 종료 시점이 다르므로 수집된 전이의20%라는 뜻은 아니다.
지표에 선택 에피소드 수와 선택된 에피소드의 전이 수를 각각 기록한다.
옵션을 켜면 난수 사용이 추가되므로 기존과 같은 난수열을 재생하는 비교는 아니다.
평가에는 탐색 옵션을 적용하지 않으며, 명시적 frozen noisy 진단에도 적용하지 않는다.
구형480차원·반경0.05의 `episode_arm_exploration` 설정과 별도 구현이다.

## 비교 설계

정밀 보상 비교의 **학습 전 초기 체크포인트**에서 분기한다.
actor689에서 옮긴 동작과 fresh Q·target Q·정규화·4개 optimizer를 그대로 복제한다.
새 실행의 actor/Q 카운터는0, replay와 성공 bank는 비어 있다.
기존 학습의 Q나 새 성공 사례를 이 비교에 추가하지 않는다.

정밀 포획 보상은 그대로 유지한다: 가중치0.5, 거리 스케일25mm,
양손 합성 `0.25(sL+sR)+0.5·min(sL,sR)`.
나머지 보상과 16스텝 실제 결과를 사용하는 Q 학습, 실제 TRAIN 성공 재사용도 유지한다.
박스·base·배경 무작위화, flap 강성·마찰, 작업 위치, 성공과 충돌 기준은 변경하지 않는다.
curriculum과 실시간 VR/IK 교사는 추가하지 않는다.

초기화 검증은 실제 TRAIN 관측538개에서 관절 목표와 그리퍼 명령이 동일함을 확인한다.
학습·anchor·초기화 정책을 포함한 모델·정규화178개 텐서와4개 optimizer 상태,
모든 기존 체크포인트·replay 필드도 직렬화 후 동일해야 한다.
이538개는 제어 동일성을 확인하는 입력일 뿐 새 학습의 replay에 넣지 않는다.

새 TRAIN 시작 조건1,536개를 생성하며 seed origin은2210600000이다.
기존 주 학습·정밀 비교·최신 작업 위치 진단의 TRAIN seed와 겹치지 않음을 확인한다.
원래 개발 평가128개 요청을 학습 전과 새 TRAIN384개마다 반복한다.
초기화 실패를 포함한 원래128개를 분모로 유지한다. 독립 FINAL은 사용하지 않는다.
기존 실행은 유지하고 새 GPU3 실행을 별도 RAM 폴더에서 관리한다.
CUDA 격리, 기존 Drive 연결·5분 주기·검증된 checkpoint만 정리하는 보관 규칙을 따른다.

21:18 KST에 별도 실행을 시작했다. 소유자·GPU3 CUDA mask·실제 manifest의
20% 팔 탐색 설정을 확인했다. 입력8개는 Drive 크기·MD5 검증을 마쳤다.
실제 초기 개발 평가 첫 스텝에서 actor/Q 업데이트0, replay0,
팔 탐색 선택·수집 통계0을 확인했다. 평가에 새 탐색이 적용되지는 않았다.
초기화·출력 동일성과 실제 실행 근거는
[입력·실행 검증](assets/rl_v2_precision25_armbias20_initialization_20261006.json)에 있다.

이 비교에서 성공률 개선을 확인한 상태는 아니다. 같은 초기 정책의 정밀 보상 비교와
전체128개 개발 평가를 비교하고, 상단 오른쪽 파지·랙 충돌·중간 선반 성능 유지도 확인한다.

첫 실제 새 TRAIN384조건 이후 전체 평가는 **21/128**이었다.
구역별 중간 좌13·우6, 상단 좌2·우0으로 시작29/128(13/8/8/0)을 넘지 못했다.
21건 모두 실제 양손 파지·hold·proof lift와 안전 조건을 확인했다.
초기화 무효1건도 원래128개 분모에 유지했다. 나머지는 랙 충돌73·박스 낙하7·
과도한 들기2·시간 초과24건이다. 상단 오른쪽32건은 모두 랙 충돌이었다.
정밀 보상 단독 비교의 첫19/128보다2건 많지만, 별도의 TRAIN 조건과
flap 물성 추첨을 사용하므로 팔 탐색의 인과적 개선을 입증하지 않는다.

새 TRAIN768조건 이후 두 번째 전체 평가는 **18/128(중간 좌8·우7, 상단 좌3·우0)**이었다.
초기29/128과 첫21/128을 넘지 못했다. 초기 무효1건을 분모에 포함했고 실제 양손
파지·hold·들기 성공18건을 확인했다. 실패는 랙 충돌59·박스 낙하9·과도한 들기1·
시간 초과40·초기화 무효1건이다. 상단 오른쪽32건은 모두 랙 충돌이었다.
기존0.15 비교는 계속하며 [초기 동작 기준·반경0.30 비교](RL_V2_REANCHORED_BODY_CONTROLLER_20261007.md)를
별도 fresh Q·replay로 시작했다. [두 번째 전체 평가 근거](assets/rl_v2_armbias20_full_DEV_after768_20261007.json).
[첫 전체 평가 근거](assets/rl_v2_armbias20_first_learned_DEV_after384_20261006.json).

## 구현과 사용

- sampler: `src/kuavo_isaaclab_scene/rl/multi_box/experiments/body_behavior_exploration.py`
- pilot·체크포인트·replay provenance: `actual_flap_residual_sac.py`
- 수집·manifest·outcome 기록: `scripts/rl/train_batched_staged_goal.py`
- 초기 입력 분기·동일성 증명: `scripts/rl/prepare_actual_flap_body_behavior.py`

초기화 도구는 종료된 초기 입력을 받아 고유한 새 출력 폴더를 만든다.
`--verification-inputs-checkpoint`는 기존 성공 TRAIN의 관측을 읽는 용도이며,
그 checkpoint의 Q·optimizer·replay는 가져오지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_actual_flap_body_behavior.py \
  --checkpoint "$TASK_ARM_SOURCE/checkpoint_00000000.pt" \
  --training-manifest "$TASK_ARM_SOURCE/training_manifest.json" \
  --waypoints "$TASK_ARM_SOURCE/waypoints.json" \
  --verification-inputs-checkpoint "$TASK_ARM_VERIFICATION" \
  --output-dir "$TASK_ARM_INPUTS"
```

새 managed/batched actual-flap TRAIN 실행의 기존 인자에
`--body-behavior ramped-arm-bias20`을 더한다.
`batched_staged_goal_with_drive.py`는 이 인자를 child에 전달한다.
옵션은 actual-flap TRAIN만 허용하며 frozen backend·작업 위치 진단과 혼합하면 거부한다.
같은 설정으로 재개할 때는 checkpoint·replay의 설정과 활성화 출처가 일치해야 한다.

현재 수집 통계는 `progress.json`의 `learner.body_behavior_statistics`에서 확인한다.
`episodes_drawn`, `biased_episodes`, `held_collection_rows`, `biased_episode_rows`를 기록한다.
설정과 활성화 출처는 checkpoint·replay·manifest·TRAIN outcome에도 남긴다.

관련 테스트54개 통과, CUDA 단위 통합 테스트1개는 환경 요구로 생략했다.
실제 실행에서는 소유자·CUDA mask·manifest와 초기 평가의 카운터 불변을 별도로 확인한다.
상세 보상과 Q 영상은 [보상·Q 설명](RL_V2_EVAL_Q_REWARD_20261006.md),
작업 위치 실패 근거는 [진단 기록](RL_V2_LEARNED_WORKPLACE_PRECISION_20261006.md)에서 볼 수 있다.

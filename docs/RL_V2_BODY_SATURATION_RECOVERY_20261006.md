# SAC 몸체 출력 포화 완화 비교

## 추가 학습의 결과

주 학습은 새 TRAIN1,536조건과17wave를 오류 없이 마쳤다.
같은 원래128요청에서 성공은 **15→29→22→20→18/128**이며 마지막 구역별
성공은 중간 왼쪽12, 중간 오른쪽4, 상단 왼쪽2, 상단 오른쪽0이다.
18건 모두 실제 양손 파지·0.25초 유지·8mm 이상 들기 증거를 확인했다.
나머지는 랙 충돌64, 박스 낙하5, 과도한 들기1, 시간 초과40이다.
초기화는128건 모두 유효했으며 상단 오른쪽32건은 모두 랙 충돌이다.
[전체 마지막 결과](assets/rl_v2_region_sac_final_DEV_after1536_20261006.json).
같은 설정으로 주 학습을 재시작하지 않았다. 종료 로그·데이터의 기존 Drive
최종 업로드와 검증도 완료했다. [백업 완료](assets/rl_v2_region_sac_final_backup_verified_20261006.json).

거리 스케일25mm와 약한 손을 강조한 정밀 보상 비교의 첫 학습 후 결과는
**29→19/128**이며 구역별11/5/3/0이다. 초기 평가의29건은 학습 전의 동작이다.
초기 유효 요청117→128로 늘었지만 성공은 감소했다. 실패는 랙 충돌77,
과도한 들기3, 박스 낙하1, 시간 초과28이다.
[정밀 보상 첫 학습 후 결과](assets/rl_v2_precision25_first_learned_DEV_after384_20261006.json).
새 TRAIN768조건 후 두 번째 전체 평가도9/128(6/3/0/0)이었다.
초기화는128건 모두 유효했고 랙 충돌67·낙하5·과도한 들기1·시간 초과46건이었다.
[두 번째 결과](assets/rl_v2_precision25_second_learned_DEV_after768_20261006.json).
팔 탐색 단독 비교의 새 TRAIN384조건 후 첫 평가도21/128(13/6/2/0)로
학습 전29/128을 넘지 못했다. [팔 탐색 결과](assets/rl_v2_armbias20_first_learned_DEV_after384_20261006.json).

두 비교 모두 같은 DEV 시작 배치 요청을 반복했다. flap 성질과 solver 이력까지
동일한 반복은 아니며 독립 FINAL 평가도 아니다. 정밀 보상을 바꾼 것만으로
성공이 개선됐다는 근거는 없다.

## 관측한 정책 변화

완전히 종료된 actor1899의 상단 오른쪽 TRAIN 진입 진단30경로에서
동일한 관측을 actor1899와 최신 actor3106에 입력했다.
안전한 왼손 near 관측216개에서 닫기40→0, 오른손 near1,678개에서는
379→367이었다. 왼손 닫기 평균 확률도0.122→0.044로 낮아졌다.
원래 source의 실제 왼손 qualified pinch는0이었다.

같은 관측에서 actor3106의 body pre-tanh 평균 절댓값 중앙값은1.73,
최대10.15였고, 좌표43.9%가 절댓값2 이상이었다.
이는 **과거의 같은 상태에 대한 출력 비교**이며 실제 새3106 경로의 파지가 아니다.
Q branch 순위도 이 과거 후보 상태에서의 추정이며 새 성공률로 해석하지 않는다.
[관측·정책 비교 근거](assets/rl_v2_actor3106_closed_TRAIN_behavior_20261006.json).

tanh는 큰 입력에서 출력이±1에 붙는다. 예를 들어 FP32의 평균10에서는
tanh 출력의 gradient가0이 될 수 있어 Q나 실제 성공 목표의 gradient가
약해진다. 같은 Gaussian 표준편차라도 실제 관절 목표 변화는 작아진다.
이 진단은 학습 실패의 유일한 원인을 증명하지 않지만 완화할 최적화 문제다.

반면 새 팔 탐색 비교의 **초기 actor689**는 종료된 실제 자기 TRAIN 관측에서
팔 평균 절댓값 중앙값0.445, 절댓값2 이상0.19%였다.
4,096개 관측·좌표당64개 분포 샘플에서 tanh 출력 표준편차의 중앙값은
가용 반경 대비 기존 잡음9.14%, 팔 bias20의 선택 에피소드51.86%였다.
약5.7배 넓어진다. 이 수치는 한 상태의 분포 진단이며 실제 새 경로의 성공이 아니다.
[초기 팔 탐색 범위 근거](assets/rl_v2_armbias20_initial_tanh_coverage_20261006.json).
새 팔 탐색의 첫 실제 TRAIN에서도128개 중24개 에피소드가 선택됐고,
GPU에서 global env ID·518차원 입력·실제 투영된 목표 수집이 진행됨을 확인했다.

## 추가 옵션

`--body-saturation-penalty mean3-soft`는 actual-flap SAC의 TRAIN actor 손실에
작은 보조 손실을 더한다. 기본값은 미지정/`off`다.

```text
excess = max(abs(pre_tanh_body_mean) - 3, 0)
L_body = 0.001 × mean_rows(sum_active(excess²) / max(active_count, 1))
active = available_affine_goal_radius > 1e-8
```

절댓값3의 tanh 출력은약0.995이고 원래 가용 반경의99.5%까지 다룬다.
이 수치는 hard limit가 아니다. 평균은 여전히3을 넘을 수 있으며, 몸체 목표의
허용 범위와 투영·policy 표준편차·Q target 계산식은 그대로다.
평균의 과도한 크기에 직접 gradient를 줘 tanh가 포화된 경우에도 회복 신호를 만든다.
가용 반경0인 좌표는 손실에서 제외한다.

보상·실제 성공 labels·jaw 손실·실행 제어는 그대로 사용한다.
훈련 밖 평가에는 보조 업데이트를 하지 않는다. 새 checkpoint/replay에는
설정과 활성화 출처를 따로 기록하고, 기존 tensor·replay·4개 optimizer 상태를 유지한다.

지표에는 다음을 추가한다.

- `body_saturation_loss`, `body_saturation_weight`, `body_saturation_limit`
- `body_saturation_active_coordinates`, `body_saturation_saturated_coordinates`
- `body_active_abs_mean_max`

구현은 `experiments/body_saturation.py`, `actual_flap_residual_sac.py`와
`algorithms/hybrid_goal_sac.py`의 opt-in hook이다.
일반 hybrid SAC는 기본 hook이`None`이고 추가 forward나 손실이 없다.
관련 테스트71개가 통과했다(CUDA 통합 단위 테스트1개 생략).
포화된 tanh의 gradient0에서도 새 손실이 유한한 회복 gradient를 내고,
내부·고정 좌표에는 gradient0이며, 평가 명령·모델·optimizer·replay 보존과
resume provenance를 검증했다.

## 비교 실행

정밀 보상+팔 bias20 초기 입력과 같은 actor·fresh Q·normalizer·optimizer에서
분기한다. **추가 actor 손실만 활성화**해 팔 bias20 단독 실행과 비교한다.
실제538개 TRAIN 관측의 greedy 목표와 전체178개 모델 tensor가 같아야 한다.
새 실행은 actor/Q 카운터0, replay·성공 bank0에서 시작한다.

새 TRAIN seed origin2220600000의1,536조건과 원래 DEV128요청을 쓴다.
다른 세 실행의 TRAIN seed와 겹치지 않음을 확인한다.
박스·base·배경·flap 무작위화, waypoint와 보상, 성공·충돌 기준은 동일하다.
기존 세 실행을 유지하고 GPU3의 고유한 RAM 실행 폴더를 사용한다.
기존 Drive 연결로 입력을 검증하고5분 주기·최근2개 보관·종료 후 로그 검증을 따른다.

초기 입력 생성 도구에`--body-saturation-penalty mean3-soft`를 더하고,
managed/batched TRAIN 실행에도 같은 인자를 전달한다.
보조 손실 없는 기존 checkpoint로의 묵시적인 변경이나 mismatched replay 재개는 허용하지 않는다.

2026-10-06 22:07 KST에 실제 비교 실행을 시작했다.
실행 입력8개는 기존 Drive 연결에서 크기와 MD5를 확인했다.
초기538개 실제 TRAIN 관측의 greedy 목표와178개 모델·normalizer tensor,
네 optimizer 및 원본 checkpoint/replay 필드를 그대로 보존했다.
실행 manifest의 `ramped-arm-bias20`·`mean3-soft`와 실제 writer의
`CUDA_VISIBLE_DEVICES=3`을 확인했다. 첫 DEV0 step31에서 actor·Q 업데이트·
replay가 모두0이고 팔 탐색의 에피소드·전이 통계도 모두0이었다.
기존 세 학습의 소유 프로세스·실행 폴더·GPU 제한도 확인했고 종료하지 않았다.
백업 오류 없이 첫 주기 업로드가 확인됐으며, 원래 입력 검증과 실제 초기 상태는
[실행 검증](assets/rl_v2_precision25_armbias20_mean3_initialization_20261006.json)에 있다.

초기 전체DEV0는29/128(중간 좌13·우8, 상단 좌8·우0)이었다.
초기 무효11건도 분모에 남겼고 실제 양손 파지·hold·proof lift 판정29건을
확인했다. 팔 탐색 단독 비교와 원래 DEV128요청이 모두 같으며 시작 성능도 같다.
Flap 물성 추첨과 접촉 history까지 완전히 일치했다고 가정하지 않는다.
이는 보조 손실로 학습하기 전 옮긴 동작의 결과다.
첫 실제 TRAIN step271에서Q416회·held24,127행을 확인했고, 추가 팔 탐색은
127에피소드 중30개에 적용됐다. Actor는 Q2,048회·수집32,768행을 기다려
아직0회였다. [전체 초기 평가·실제 수집 근거](assets/rl_v2_mean3_initial_full_DEV_and_real_TRAIN_20261006.json).

23:33 KST 실제 TRAIN3 step361에서 actor442회·Q3,816회·held177,424행을 확인했다.
Q2,048회와 수집32,768행 warmup을 통과했고 actor의 보조 손실은
`8.4286e-7`, 가중치0.001, 평균 기준3이었다. 배치의 active4,864좌표 중
11개가 기준을 넘었으며 최대 평균 절댓값은4.322였다. 작은 손실의 실제 연결을
확인한 기록이다. 첫 학습 후 전체 개발 평가나 물리 개선은 아직 측정하지 않았다.
[실제 actor 업데이트 근거](assets/rl_v2_mean3_real_actor_auxiliary_updates_20261006.json).

새 TRAIN384조건 이후 첫 학습 후 전체 개발 평가는 **24/128(16/8/0/0)**이었다.
24건 모두 실제 양손 파지·hold·proof lift와 안전 조건을 확인했다.
초기화 무효1건도 분모에 유지했다. 실패는 랙 충돌82·박스 낙하2·과도한 들기12·
시간 초과7건이었다. 상단 오른쪽32건은 모두 랙 충돌이며 상단 왼쪽도0/32였다.
팔 탐색 단독의 첫21/128보다3건 많지만 별도 TRAIN 조건과 flap 추첨을 사용했고,
초기29/128이나 대표 최고30/128은 넘지 못했다. 포화 완화의 인과적 성능 개선을
입증하거나 상단 파지 문제를 해결했다고 볼 수 없다. 다음 훈련 배치는 계속 진행한다.
[첫 학습 후 전체 평가](assets/rl_v2_mean3_first_learned_DEV_after384_20261007.json).

후속 [실제 동작·제어 목표 진단](RL_V2_PHYSICAL_EXPLORATION_DIAGNOSIS_20261007.md)에서는
종료된 주 학습의 상단 오른쪽 실패가 오른손 그리퍼 몸체의 랙 충돌에 몰린 것과
작은 탐색 폭을 확인했다. [종료 TRAIN의 목표 범위 비교](RL_V2_LOCAL_GOAL_RANGE_DIAGNOSIS_20261007.md)는
더 넓은 변화 범위와 학습된 초기 body goal을 함께 검토할 근거를 제시한다.
기구학 해는 실제 성공으로 세지 않았다. 후속 [초기 동작 기준·반경0.30 제어](RL_V2_REANCHORED_BODY_CONTROLLER_20261007.md)를
구현하고 실제 TRAIN538개에서 초기 목표·jaw·물리 명령 보존을 확인했다.
기존 비교는 유지하며 새 Q·replay로 별도 GPU3 실행을 시작했다. 새 물리 성공은 아직 확인 전이다.

후속 [종료된 TRAIN의 Q·경사 진단](RL_V2_CLOSED_TRAIN_Q_AND_GRADIENT_20261006.md)에서는
일치하는 동결 critic이 실제 랙 충돌 직전의 음의 보상을 반영하는 것과,
후속 actor의 포화된 팔 좌표에서 Q-only 경사가 약해진 것을 확인했다.
원래 전체 개발 평가나 새 포화 완화 학습의 성공률을 측정한 결과는 아니다.

# 실제 TRAIN 성공 궤적의 파지 직전 구간을 더 자주 학습

새 제어의 첫 학습 후 전체 개발 평가는 **23→4/128**로 떨어졌다. 네 구역은
중간 좌3·우0, 상단 좌1·우0이며, 유효한 시도127개 중 랙 충돌96·낙하3·
과도한 들기1·시간 초과23건이었다. 초기화 무효1건도 원래128개 분모에 남겼다.
[전체 평가 근거](assets/rl_v2_reanchored30_first_full_DEV_after384_20261007.json).
상단 오른쪽을 훈련 탐색 중 잡았지만 평가에서 재현하지 못했으므로 성공한 정책으로
보지 않는다. 반복한 개발 평가는 독립 FINAL 평가도 아니다.

## 왜 바꾸는가

[성공 경로의 학습 진단](RL_V2_SUCCESS_PATH_LEARNING_20261007.md)에서 실제 성공
상태에 대한 그리퍼 명령은 개선됐지만 몸체 목표 오차는 커졌다. Actor692의
과거 상단 오른쪽 성공 상태590개에서 목표 MSE는0.001381, 마지막64개에서는
0.003125였다. 마지막 구간의 성공 보조 손실 경사 크기는 Q 경사의 중앙값0.70배였고
대체로 서로 반대 방향이었다. 이는 특정 과거 상태의 부분 경사 분석이며 Q가
틀렸다는 증명이나 새 표본 방식이 성공한다는 증거는 아니다.

긴 성공 궤적 전체에서 균일하게 뽑으면 실제 접촉·닫기·들기 구간을 자주 보지
못할 수 있다. 다음 비교에서는 **actor 성공 보조 학습만** `tail64-half`로 바꾼다.

| 설정 | 기존 | 새 비교 |
| --- | --- | --- |
| Actor 성공 표본64개 | 성공 경로 전체에서 균일 추출 | 32개는 마지막64스텝, 32개는 전체 경로 |
| 구역 균형 | 존재하는 네 구역을 균형 있게 추출 | 유지; 각 구역 지정 tail 표본8개 |
| 성공 목표·그리퍼 정답 | 실제 실행한 목표·열림/닫힘 | 유지 |
| Q 성공 표본 | 전체 성공 경로에서 추출, 비율20→5% | 유지 |
| 보상·실패·성공 기준 | 실제 접촉·유지·들기, 기존 안전 기준 | 유지 |

전체 경로에서 뽑는 나머지32개도 우연히 마지막 구간에 속할 수 있다. 정확히 절반의
관측만 마지막 구간에 속한다는 뜻은 아니다. 구간 길이가64보다 짧으면 전체를 쓴다.
기본 옵션은 기존 균일 추출이며 새 checkpoint의 명시된 계약만 새 방식을 선택한다.

## 실제 훈련 경험을 연결하는 방법

모델과 Q는 이전 새 실험의 **학습 전 초기화**에서 시작한다. 학습 후 actor692·Q4816의
신경망·optimizer·온라인 replay는 가져오지 않는다. 해당 체크포인트 안에 닫혀 저장된
**실제 안전한 TRAIN 성공15개, 6,899전이**만 가져온다.

| 구역 | 성공 궤적 | 실제 전이 |
| --- | ---: | ---: |
| 중간 왼쪽 | 8 | 3,291 |
| 중간 오른쪽 | 3 | 1,230 |
| 상단 왼쪽 | 3 | 1,788 |
| 상단 오른쪽 | 1 | 590 |

이는 두 VR 시연의 추가 복제가 아니라 실제 정책 탐색에서 얻은 경험이다. 초기 온라인
replay는0이며 실제 훈련에서 새 전이를 수집한다. n-step16 보조 Q의 초기 은행도
이 성공 경로에서 구성하므로 **초기에 가져온 실패 궤적은0개**다. 새 훈련에서 생긴
실패는 계속 수집·사용한다. 실행 중인 HDF/replay를 읽거나 성공·행동·보상을
다시 붙이지 않으며 DEV·FINAL·가상 IK 결과는 학습에 넣지 않는다.

초기 actor·Q·optimizer tensor와 직렬화된 경험이 정확히 유지됨을 확인했다.
실제 TRAIN 관측698개에서도 초기 deterministic 몸체 목표와 그리퍼 출력이 동일했다.
관련 고유 테스트64개가 통과했으며, 이는 데이터·표본·복원 배선 검증이지 물리 성능이 아니다.
[초기화 검증](assets/rl_v2_success_actor_tail_initialization_20261007.json).

## 실행과 비교

초기화 입력은 기존 Drive 연결로 크기·MD5를 검증한 뒤 관리 실행에 사용한다.
별도 TRAIN1,536조건, 원래 개발 평가128개로 비교하며 새 초기 평가를 자신의 기준으로
기록한다. 구체적인 flap 추첨과 solver 이력이 같은 것으로 가정하지 않는다.
박스·base·주변 박스·단단하지만 움직이는 flap의 무작위화와 충돌 기준을 유지한다.
보상 식·가중치는 [Q 영상과 보상 설명](RL_V2_EVAL_Q_REWARD_20261006.md)을 참고한다.

관리 진입점은 `batched_staged_goal_with_drive.py`이며 learner는
`CUDA_VISIBLE_DEVICES=3` / `cuda:0`, 물리 장면은 기존 CPU PGS 계약이다.
새 실험은 자신만의 RAM 실행 폴더와 기존 Drive 업로더(300초, 검증 후 최신2개 보존)를
사용한다. 종료된 writer의 로그도 검증한다. 기존 사용자 파일·프로세스는 건드리지 않는다.
준비·학습 시작만으로 개선을 주장하지 않으며 학습 후 전체 개발 평가와 최종 독립 평가가 필요하다.

**10/07 03:33 KST에 새 비교 실행을 시작했다.** 실제 writer3847594의 owner·실행
폴더·CUDA3 마스크를 확인했으며 당시에는 장면 초기화 중이었다. 코드348558e를
main에 push한 뒤 실행했다. [실행·입력 백업 근거](assets/rl_v2_success_actor_tail_actual_start_20261007.json).
기존 다른 사용자 프로세스는 건드리지 않았고 이전 비교3개도 유지했다.

기존 출력 포화 완화(mean3) 비교의 새 TRAIN1,152조건 후 전체 평가도
**18/128(중간 좌12·우6, 상단 양쪽0)**으로 확인했다. 흐름29→24→12→18이며
초기29건을 넘지 못했다. 새 tail 표본 방식과 구분하는 대조 기록이다.
[기존 mean3의 전체 평가](assets/rl_v2_mean3_full_DEV_after1152_20261007.json).

## 실제 연결과 남은 실패 위치 확인

새 실행은 초기화를 통과해 첫 DEV128개 평가에 들어갔다. 실제 learner JSON에서
actor·Q 업데이트0, 온라인 replay0, 성공 은행15궤적/6,899행과 같은 크기의
n-step16 은행, `tail64-half` 계약을 확인했다. 이 초기 평가는 학습 전 성능이며
새 actor의 파지 개선을 뜻하지 않는다.
[실제 평가 진입·은행 연결](assets/rl_v2_success_actor_tail_first_actual_DEV_20261007.json).

닫힌 TRAIN 성공6,899행의 몸체 정답을 동일한 actor 기준점과 현재 affine 목표
범위에 대입했을 때 **범위를 벗어난 정답은0행**이었다. 이는 명령 좌표의 범위
확인이며 새로운 물리 성공이나 충돌 없는 경로를 계산한 것은 아니다.
현재 성공 경로를 균일하게 뽑을 때 마지막64스텝의 예상 비율은 상단 오른쪽10.8%,
상단 왼쪽10.7%, 중간 양쪽약15.6%였다. 새 방식에서는 각각약55.4%,55.4%,57.8%가
된다. 전체 경로 절반에서도 마지막 구간이 뽑힐 수 있어 지정된50%보다 높다.
[실제 정답 범위·표본 비율 근거](assets/rl_v2_actual_success_goal_feasibility_20261007.json).

이전 DEV4의 랙 충돌96건에서 **실패 종료 동작의 최대 힘 링크**를 집계했다.
오른쪽 그리퍼 본체70·왼쪽 본체20·왼팔4번째 링크4·오른팔7번째 링크2건이었다.
그리퍼 본체가90/96건을 차지하며 종료 시 힘의 중앙값은40.7N이었다. 모든 이전
접촉을 복원한 통계는 아니다. 기존10N 안전 기준과 그리퍼 본체 충돌 판정은 유지한다.
새 비교에서 파지 직전 손의 위치·자세를 더 자주 학습하고 동일한 실패 원인 집계로
실제 변화가 있는지 확인한다.
[종료 시 충돌 링크 근거](assets/rl_v2_reanchored30_DEV4_rack_peak_links_20261007.json).

## 학습 전 전체 평가와 실제 Q 갱신

새 실행의 첫 전체 평가는 **23/128(중간 좌12·우6, 상단 좌5·우0)**으로 끝났다.
실제 안전한 양손 접촉·유지·들기23건을 확인했다. 나머지는 랙 충돌69·시간 초과25·
초기화 무효11건이며 무효도 원래128개 분모에 포함한다. 당시 actor·Q 갱신과 온라인
replay가0이어서 성공 은행을 연결한 초기 정책의 기준이다. 이후 성능은 이 초기23건과
비교하되 실제 flap 추첨·solver 이력까지 같다고 가정하지 않는다.
[새 전체 초기 평가](assets/rl_v2_success_actor_tail_full_initial_DEV_20261007.json).

이후 첫 TRAIN에서 **Q956회 갱신**을 확인했다. 256개 Q 표본 중 실제 성공 경험51개,
별도의 실제16스텝 표본64개·가중치0.1이 연결됐고 one-step·보조 Q 손실은 유한했다.
온라인 replay에는 새 실제 TRAIN 전이만 들어가며 진행 중인 VR 재생·모방 손실은 없다.
Actor는 Q warmup 중이어서 성공 표본64개/지정 tail32개를 사용한 **실제 actor 갱신은
아직 시작 전**이다. Q 손실 감소만으로 파지 개선을 주장하지 않는다.
[실제 TRAIN·Q 연결](assets/rl_v2_success_actor_tail_first_real_TRAIN_Q_20261007.json).

기존 팔 탐색20% 비교는 새 TRAIN1,536조건을 끝내고 마지막 개발 평가가
**18/128(중간 좌10·우8, 상단 양쪽0)**이었다. 흐름은29→21→18→22→18이며 초기29건을
넘지 못했다. Writer의 정상 종료와 actor4321·Q19332의 유한한 체크포인트를 확인했다.
기존 관리자의 최종 업로드가 진행 중이므로 별도 업로더를 중복 실행하지 않는다.
이는 반복한 마지막 개발 평가이며 독립 FINAL은 사용하지 않았다.
[팔 탐색 비교의 마지막 평가](assets/rl_v2_armbias20_final_full_DEV_after1536_20261007.json).

준비 예시(Isaac conda의 Python을 사용):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_actual_success_actor_tail.py \
  --initial-checkpoint /absolute/path/to/fresh-reanchored-checkpoint.pt \
  --matching-train-checkpoint /absolute/path/to/protected-matching-TRAIN-checkpoint.pt \
  --matching-train-proof /absolute/path/to/immutable-capture-proof.json \
  --training-manifest /absolute/path/to/matching-training-manifest.json \
  --waypoints /absolute/path/to/matching-waypoints.json \
  --output-dir /absolute/path/to/new-unique-input-directory
```

준비 도구는 실제 owner·파일 SHA256·일치하는 제어/보상/관측 계약·성공 근거를 확인하며,
기존 파일을 덮어쓰거나 훈련을 자동 시작하지 않는다.
코드: [준비 도구](../scripts/rl/prepare_actual_success_actor_tail.py),
[표본 은행](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/staged_train_success.py),
[actor 연결](../src/kuavo_isaaclab_scene/rl/multi_box/experiments/staged_goal_sac.py).

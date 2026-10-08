# 실제 관절 범위에서 학습하는 별도 SAC

## 최신 결과: TRAIN768 뒤 전체 DEV4/128, 개선 없음

2026-10-08 20:05 KST 두 번째 학습 후 전체 평가는 **성공4·안전 위반49·
시간 초과75·초기 무효0/128회**였다. 초기14→첫 평가8→두 번째4회이며 중형은0회다.
안전 위반은랙 충돌48·낙하1회다. Actor1,909·Q9,682회의 실제 모델·optimizer와
평가 종료 모델을 대조했고 전체 tensor는 유한했다. 공통 초기 유효115조건도
14→4회였다. 동일 요청이어도 flap/contact 이력은 완전히 같지 않다.
[전체 결과·실제 모델 대조](assets/rl_v2_URDF_second_learned_full_DEV8_20261008.json).

기존 긴 학습은 유지하면서, [실제 SAC 수집의 속도 제한 진단](RL_V2_MEASURED_SERVO_DIAGNOSTICS_20261008.md)을
추가했다. 다른 제어기가 만든 과거 경로에서 경사 차단을 발견했으므로 현재 SAC
상태에서도 확인한다. 시간이나 VRAM 증가만으로 해결된다고 판단하지 않는다.

아래8/128회 기록은 첫 학습 후 결과이며 최신4/128회와 구분한다.

## 첫 학습 후 전체 DEV8/128, 명령 유지·탐색 단위 보강 준비

2026-10-08 18:20 KST 첫 TRAIN384조건 뒤 같은 전체 DEV128은 **성공8·랙 충돌46·
시간 초과74·초기 무효0회**였다. 초기14회보다 낮고 공통 유효115조건도14→8회다.
중간 왼쪽small1회·상단 오른쪽small6회·상단 왼쪽small1회이며 나머지는0회다.
실제 actor698·Q4,840회 모델·optimizer와 평가 종료 모델이 같고 유한함을
확인했다. 중형은0회이며 목표를 달성하지 않았다. 기존 긴 실행은 계속하면서
[명령 유지와 팔 탐색 단위를 보강한 선택형 SAC](RL_V2_URDF_SERVO_GUARD_20261008.md)를
준비했다. 새로운 물리 rollout·학습 개선은 아직 증명하지 않았다.
[전체 첫 학습 후 평가와 같은 실제 모델](assets/rl_v2_URDF_first_learned_full_DEV4_20261008.json).

아래14회·16개 TRAIN 경로·actor256/Q3,072 기록은 각각 당시 초기 평가와 학습 검증이다.
최신 학습 후 평가8회와 구분한다.

2026-10-08 16:38 KST에 초기 전체 DEV128이 완료됐다. **성공14·안전 위반45·
시간 초과56·초기 무효13회**, numerical failure0이다. Actor/Q 업데이트0인
**학습 전 기준**이며 기존 실행의11회나 별도 교사1회를 학습 개선으로 비교하지
않는다. 성공은 실제 양손 opposing flap 파지·0.25초 유지·8mm clearance 기준이다.

| 선반·방향 | 크기 | 요청 | 성공 | 안전 위반 | 시간 초과 | 초기 무효 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 중간 왼쪽 | small | 16 | 9 | 7 | 0 | 0 |
| 중간 오른쪽 | small | 16 | 0 | 0 | 13 | 3 |
| 중간 왼쪽 | medium | 16 | 0 | 16 | 0 | 0 |
| 중간 오른쪽 | medium | 16 | 0 | 5 | 10 | 1 |
| 상단 왼쪽 | small | 32 | 0 | 0 | 28 | 4 |
| 상단 오른쪽 | small | 32 | 5 | 17 | 5 | 5 |
| 전체 | 여섯 조합 | 128 | 14 | 45 | 56 | 13 |

안전 위반은랙 충돌43·과도한 들기1·박스 낙하1회다. 원래 전체128요청·현재
물리/보상/성공/안전/DR·격리 소스728개를 대조했다. 실제 평가 종료에서 저장된
모델·normalizer74개 tensor가 준비한 입력 모델과 모두 같고 frozen 구역별 소스와
body anchor도 같았다. Checkpoint 전체256개 tensor는 유한했다. 따라서 초기
모델 일치는 더 이상 미확인 상태가 아니다. 저장 파일의 SHA는 metadata 차이로
입력과 다르며 모델 값의 일치와 파일 checksum을 따로 확인했다.
[첫 전체 결과·실제 저장 모델·128개 결과](assets/rl_v2_URDF_regional_goal_first_full_DEV_20261008.json).

16:40 KST에 동일한 writer가 첫 TRAIN128조건으로 넘어갔다. 첫 학습 후 전체
DEV는 새 TRAIN384조건 뒤다. 중형·상단 왼쪽의0회가 남아 있으며 목표는 아직
달성하지 않았다. 전체 개선과 독립FINAL을 확인하면서6144조건 계획을 계속한다.

16:48 KST 첫TRAIN step361에서 실제held 전이35,912개·Q598회 갱신과 유한한
업데이트 손실을 확인했다. Q2,048회 warmup이 먼저이므로 actor갱신0은 정상이다.
현재 성공 은행과 평가 import는0이며 VR/teacher BC 갱신 가중치도0이다.
업데이트 뒤 전체 모델의 유한성은 다음 실제 저장 checkpoint에서 확인한다.
[실제 첫TRAIN/Q 갱신과 확인 범위](assets/rl_v2_URDF_initial_DEV14_first_TRAIN_20261008.json).

16:53 KST 실제 저장된Q1,024회 checkpoint를 별도로 보존했다. 두 Q 네트워크가
초기 값에서 실제 바뀌었고, 전체 모델/normalizer/optimizer316개 tensor는 유한했다.
Q optimizer의20개 parameter state 모두 실제Adam step1,024였으며 actor와
actor normalizer29개 tensor는 원래 값과 같았다. 초기Q warmup 동안 예상한
동작이다. Frozen source/body anchor와 격리 소스728개도 보존했다.
학습 로그만으로 모델 갱신을 추정하지 않았으며 첫 학습 후 전체 DEV는 아직 전이다.
[실제Q1,024 모델·optimizer·동결 actor 대조](assets/rl_v2_URDF_first_Q1024_actual_model_20261008.json).

17:06 KST에 첫 TRAIN128조건의 종료 체크포인트를 대조했다. **성공16·안전
위반51·시간 초과61·초기 무효0회**였다. 성공은 중간 왼쪽small9회·상단
오른쪽small6회·상단 왼쪽small1회이며 중형과 중간 오른쪽은0회다. Greedy
97조건에서16회 성공했고 연속 팔 탐색31조건에서는 성공0·안전 위반23·시간
초과8회였다. 당시 actor갱신0·Q1,590회였으며 TRAIN 시작 조건은 DEV와 다르다.
따라서16회를 학습 후 평가 성공률이나 초기14회 대비 개선으로 해석하지 않는다.

실제로 성공한16개 경로의7,666개 전이가 성공 은행에 들어갔다. 모든 행의
518D/578D 관측과 새 bounded21D 목표·binary jaw·실제 opposing 파지/유지/들기
증거, production jaw gate와 servo encoder를 확인했다. 실제 종료 checkpoint의
모델·optimizer·성공 행428개 tensor가 유한했고 Q optimizer는 모두 실제
1,590회 갱신이었다. CPU Q 복원도 정확히 일치했다. 성공 경로 입력에서 greedy
목표의 최대 오차2.56e-6과 servo 평균 오차4.92e-6은 CPU/GPU 수치 차이이며
정책 갱신으로 해석하지 않는다.

완료한 TRAIN128조건의 실제 종료까지 계산한 return도 수집됐으며 평가 데이터
유입은0이다. 학습 로그에서 성공 전이의 Q 표본 약20%와 실제 종료 return
보조64행·가중치0.1·bootstrap0을 확인했다. 실제 성공 유지 손실은 이 새
TRAIN 성공 행을 사용한다. VR/교사 BC 가중치는0이고 과거 좌표의 성공 은행을
가져오지 않았다. Return 은행의 보고·수집 수는 대조했지만 실행 중인 전체
replay/HDF를 읽어 재구성하지 않았다.
[첫 TRAIN 전체 결과·실제 성공 은행·모델 대조](assets/rl_v2_URDF_first_closed_TRAIN1590_data_verified_20261008.json).

17:19 KST 실제 저장 모델에서 **actor256회·Q3,072회 갱신**을 확인했다. Actor
가중치가 초기 값에서 바뀌었고 모든 actor/Q optimizer parameter의 실제 Adam
step이 각각256/3,072였다. 저장된 모델·normalizer·optimizer·성공 행506개
tensor는 유한했다. Actor LR1e-6, 관측 normalizer와 고정 기준 actor/body anchor,
격리 소스728개는 유지됐다. 실제 actor 갱신에 TRAIN 성공64행과 Q 성공 표본,
종료 return 보조64행·bootstrap0이 사용됐다. 새 성공 은행은 구역 균형을 유지한
전체 경로 uniform 표본이며 이전 actor-memory 실행의 tail32 표본을 가져오지
않는다. 이는 **실제 SAC 정책이 갱신되고 있다는 증거**다. 첫 학습 후 전체
DEV4는 아직 진행 전이므로 성공률 개선이나 일반화 성공의 증거는 아니다.
[첫 실제 actor 모델·optimizer·성공 학습 연결](assets/rl_v2_URDF_first_actor_Q3072_actual_model_20261008.json).

같은 모델을 CPU에서 정확히 복원해 **이전 실제 성공 경로의 입력**을 다시
질의했다. 중간 왼쪽small/상단 오른쪽small/상단 왼쪽small의 몸체 목표 평균
오차는0.00617/0.00715/0.00406인 반면, 정규화된 실제 servo 명령 평균 오차는
0.271/0.264/0.161이었다. 작은 목표 변화가 production step 제한에서 큰 명령
변화로 이어질 수 있으므로 성공 동작 보존을 추가로 살펴야 한다. 이는 과거
입력에 대한 정적 측정이며 실제 새 경로의 충돌·성공 여부를 단정하지 않는다.
전체 DEV에서 성공이 감소하면 이 명령 변화와 Q/성공 유지 손실의 균형을
함께 분석한다. 목표 오차만 작다고 성공 경로가 유지됐다고 판단하지 않는다.

두 번째 TRAIN128조건은 성공7·안전 위반56·시간 초과65·초기 무효0회였다.
성공은 중간 왼쪽small5회·상단 오른쪽small2회이며 탐색27조건의 성공은0이다.
Actor가 수집 중 갱신됐고 시작 조건도 첫 TRAIN과 다르므로16→7을 동일한
모델/조건의 성능 추세로 해석하지 않는다. 중형과 탐색 성공0, 정적 명령 변화는
다음 전체 평가에서 확인할 위험 신호다. 현재 세 번째 TRAIN을 이어가며 종료
뒤 첫 학습 후 전체 DEV4를 수행한다. 이 기록은 종료 결과 metadata 대조이고
두 번째 TRAIN 종료 checkpoint 전체를 추가로 감사한 것은 아니다.
[두 번째 TRAIN의 종료 결과와 판단 범위](assets/rl_v2_URDF_second_closed_TRAIN128_metadata_20261008.json).

### 같은 초기 정책의 대표 영상

[중간 왼쪽 small 실제 성공 영상](assets/rl_v2_URDF_initial_DEV_middle_left_small_success_20261008.mp4)은
base 시작 lateral+18.8cm·outward+12.0cm·yaw+15.0도인 무작위 조건의 한 사례다.
접근 이동은 기존 별도 제어기이며 양팔/몸체/jaw는 현재 정책이 제어한다. 실제
양손 opposing 파지0.267초·clearance45.18mm, 안전 위반0이었다. 초기 모델의
한 경로이며 학습 개선이나 전체 일반화 성능을 뜻하지 않는다.

[중간 왼쪽 medium 실제 실패 영상](assets/rl_v2_URDF_initial_DEV_middle_left_medium_unsafe_20261008.mp4)은
랙 충돌로 끝났고 양손 파지는없었다. 종료 때 손–flap 거리는2.48/5.07cm다.
이는 종료 순간의 측정이며 첫 충돌 원인이나 전체 접근 이력으로 단정하지 않는다.
두 영상은 실제 측정된 몸체/flap pose와 reset 전 terminal frame을 기록하며
H264/avc1·yuv420p·faststart 및 별도 전체 decode를 확인했다.
[영상 형식·실제 모델·결과 대조](assets/rl_v2_URDF_initial_DEV_media_20261008.json).

Notion 중간 보고의 native video2개로 업로드하고 caption·초기 기준14/128과
실제TRAIN 시작을 재확인했다. 기존 media63개와 native table5개의 내용·순서를
보존했으며 현재 media는65개다.
[Notion 영상과 기존 자료 보존 검증](assets/rl_v2_URDF_initial_DEV14_Notion_native_videos_verified_20261008.json).

17:25 KST Notion의 첫 TRAIN 성공16경로·7,666행과 실제 actor256/Q3,072 갱신,
정적 servo 명령 차이·첫 학습 후 전체 DEV 미확인을 갱신하고 다시 읽어 대조했다.
기존 native media65개·table5개의 내용과 순서를 보존했다.
[학습 진행 기록·기존 Notion 자료 보존 검증](assets/rl_v2_URDF_first_TRAIN16_actor256_Notion_verified_20261008.json).

## 방법과 변경 범위

기존 여섯 조합 SAC의 전체 DEV가11→10→7→6/128로 내려간 뒤 준비한 선택형
학습 경로다. 목표의 팔14좌표를 데모의 관절 범위 대신 알려진 S63 URDF의
중간점·절반 범위로 정규화한다. 관절 끝에는0.01rad 여유를 둔다. 나머지 몸체·
torso·head·jaw 목표 정규화, 원래 physical decoder와 속도/명령 제한은 유지한다.

기존 구역별 actor는 자기의 **원래 좌표**에서 실행한 뒤 물리적 몸체 목표를
새 좌표로 변환하는 고정 기준이다. 이 기준 주변에서 새 SAC가 보정을 학습한다.
새 actor body mean은0으로 시작하고 원래 jaw logits와 네 구역 routing은 보존한다.
Gaussian은 기존 quarter 설정을 사용한다. 같은 정규화 폭도 실제 팔 범위에서는
더 큰 물리적 목표 차이를 표현한다. 원래 관절 제한 밖의 기준 목표가 있으면
알려진URDF 여유로만 제한한다. 실제244개 종료 입력에서는 이 제한이0행이었다.

**교사 진단의 goal·접촉 센서 제어 규칙을 SAC로 가져오지 않는다.** 고정 소스의
actor만 기준으로 사용하며 그 정책의 학습된 Q·replay·reward·entropy·optimizer를
새 학습으로 옮기지 않는다. Actor/replay의 실제 실행 goal은 새 bounded[-1,1]
좌표이고 Q는 같은 goal을 기존 decoder로 변환한 실제 servo 명령으로 평가한다.
새 artifact/goal contract가 기존 좌표의 체크포인트와 replay 복원을 거부한다.

## 새 장기 실행 설정

| 항목 | 설정 |
| --- | --- |
| 관측/행동 | actor518·critic578·연속 body19+binary jaw2; midpoint 관계38D 유지 |
| 팔 목표 | S63 URDF14관절, 양끝0.01rad 여유; 보정 반경0.30 |
| Actor | 네 구역 head, 고정 소스 기준, actor LR1e-6 |
| 경험 버퍼 | 200만 전이, 약16.5GiB; Q와 성공/return bank는 비어 있는 상태에서 시작 |
| TRAIN 수집 | 20% 연속 팔 탐색, 나머지 greedy; 기존 production gate 안의 joint jaw30% 탐색 |
| 학습 | 기존 servo Q·TRAIN episode-return 보조 Q·새 실제 성공 유지 손실 |
| 새 장기 계획 | fresh TRAIN6,144조건; TRAIN384조건마다 같은 전체 DEV128 |
| 범위 | 중간 좌우small/medium 각16, 상단 좌우small 각32로 매 배치128 |
| 무작위화 | 기존 box/base/배경/firm dynamic flap 유지; 새 TRAIN seed와 기존 DEV/FINAL 분리 |
| 성공/안전 | opposing pads5N, hold0.25s, proof lift8mm; 랙10N·장애물5N·drop10cm; selfOFF |

2026-10-08 16:00 KST에 GPU3에서 별도 장기 실행을 시작했다. 16:21 KST에는
실제 writer3758401·CUDA_VISIBLE_DEVICES=3·격리 소스728개와 전체 runtime
계약을 확인했다. 첫 전체 DEV0의 step361이며 actor/Q/online/replay는0이다.
당시 **첫 전체 평가와 학습 후 성능은 미완료**였다. 초기화 입력 모델은 유한했고
실제 평가 종료 때 저장되는 모델과의 일치는 위16:38 KST 결과에서 확인했다.

실제 버퍼는200만, writer의 GPU 메모리는17,561MiB다. 기존 중력 보상18관절과
원래128개 DEV 요청·여섯 조합 배분·물리/관측/보상/성공/안전 계약을 대조했다.
관리 폴더는`GPU3_URDF_regional_goal_long_SAC128_20261008_155959`, 실행 폴더는
`batch_sac_20261008_160000_2d2cbc`다. 초기 환경의 무효13회도 원래128개 분모에
포함하며 성공이나 평가 개선을 미리 주장하지 않는다.

기존 Drive 연결·300초/최근2개 유지와16:20 KST 체크포인트/계약 범위의 검증을
확인했다. 현재 로그·raw HDF/replay는 로컬에 남고 종료 로그 검증은 writer 종료
뒤 수행한다. 시작 확인 당시 root 약160GiB·shm 약229GiB의 여유가 있었다.
전체65배치의 예상 실행 시간은 기존 속도 기준28~40시간이며 성공 달성 시간을
보장하는 추정치는 아니다. 새6144조건과 버퍼는 새 실행에 적용했고 기존
GPU3의500k 버퍼/3072조건을 실행 중에 바꾸지 않았다.
[실제 시작·원래 계약·백업 확인](assets/rl_v2_URDF_regional_goal_SAC_actual_startup_20261008.json).

Notion 중간 보고도 새 실제 실행·200만 버퍼·6144조건과 기존 SAC의 최신
11→10→7→6→10회 결과로 갱신했다. 기존 native media63개와 table5개의 내용·
순서가 동일하고 첫 전체 DEV0·학습 개선은 미확인으로 표시한 것을 재확인했다.
[Notion 갱신·자료 보존 확인](assets/rl_v2_URDF_regional_goal_SAC_Notion_verification_20261008.json).

## 초기화 사용법

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl OMP_NUM_THREADS=1 \
python scripts/rl/prepare_urdf_regional_goal_sac.py \
  --source-checkpoint /absolute/path/to/original-regional/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/training_manifest.json \
  --waypoints /absolute/path/to/size_waypoints.json \
  --waves-json /absolute/path/to/disjoint-training-waves.json \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 \
  --native-seed /absolute/path/to/closed-lower-native/executed_transitions.hdf5 \
  --native-seed /absolute/path/to/closed-upper-native/executed_transitions.hdf5 \
  --replay-capacity 2000000 \
  --output-dir /absolute/path/to/unique-new-initialization
```

입력은 소유한 일반 파일이어야 하고 출력은 새 폴더여야 한다. 원래 region actor
체크포인트와 동일한 물리·waypoint 계약을 요구한다. 새 checkpoint·빈 experience·
manifest·검증 기록을 만들고 실제 학습 진입점에서 사용할 복원까지 검사한다.
훈련과 영상 재생은 같은 staged dispatch를 사용한다. 실제 실행은 기존
[Drive 관리자](RL_GOOGLE_DRIVE.md)를 재사용하고 CUDA_VISIBLE_DEVICES=3,
고유 run·300초 검사·검증된 오래된 checkpoint만 정리·최근2개 보호를 적용한다.
Raw HDF/replay는 체크포인트·로그 전용 백업 범위에서 로컬에 유지한다.

## 확인한 범위

관련28개 검사가 실제 decoder의 pending target·servo 제한·지역별 actor 갱신·
동결 소스 보존·신규 좌표 및 기존 dispatch 보존을 통과했다. 실제 초기 체크포인트를
학습용으로 복원해 모델·optimizer·contract가 정확히 일치하고 Q/online/성공/
return bank0임을 확인했다. Q 영상용 CPU 복원도 같은 전체 모델로 일치했다.

정상 종료한 교사 진단의 유효115경로에서 정착·접근·삽입·들기 입력244개를
선택했다. 새 actor의 초기 물리 목표가 원래 구역별 정책과 일치하고 팔 제한
추가 clamp는0행이었다. Jaw와 비팔 physical 명령을 보존하고 모든 새bounded
goal의 정확한 decoder를 확인했다. 원래/새 모델과 종료 HDF는 변경하지 않았다.
이 검사는 해당 과거 상태의 질의이며 새 rollout·접촉·성공을 증명하지 않는다.

[실제 초기화·학습 재개 계약](assets/rl_v2_URDF_regional_goal_initialization_20261008.json),
[실제 종료 입력244개와 명령 보존](assets/rl_v2_URDF_regional_goal_closed_inputs_20261008.json).

기존 일반 PPO/SAC/DPPO 설정이나 고정 교사 v3/v4의 기본값은 변경하지 않는다.
같은 전체 DEV128의 **교사 없는 실제 SAC 결과**와 아직 사용하지 않은 독립FINAL로
성과를 판단한다. 중형·상단의 접근/정렬·동시 파지는 여전히 풀어야 하는 문제이며
팔 범위를 넓혔다는 사실만으로 성공할 것이라 단정하지 않는다.

# 실제 관절 범위에서 학습하는 별도 SAC

## 최신 결과: 초기 전체 DEV14/128, 새 TRAIN 시작

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

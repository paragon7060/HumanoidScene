# 실제 관절 범위에서 학습하는 별도 SAC

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

## 준비한 설정

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

준비와 실제 실행은 구분한다. 2026-10-08 15:51 KST까지 확인한 것은 초기화·
복원·명령 일치이며 **새 물리 평가·학습 성능은 아직 미확인**이다.
새6144조건과 버퍼는 새 실행의 설정이고 기존 활성 GPU3의500k 버퍼/3072조건을
실행 중에 변경하지 않았다. 기존 실행은 격리 소스와 다음 전체 평가를 유지한다.

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

# 바닥에 떨어진 박스가 시간 초과로 기록되던 판정 수정

박스가 바닥까지 떨어져도 기존 파지 환경은 실패 감점을 주지 않는 경우가 있었다.
박스 중심 높이를 `12cm 미만`으로만 검사했기 때문이다. 옆으로 누운 박스의
중심이 12cm 위에 남으면 낙하 판정이 빠진다. **Grasp에서 초기 정착 높이보다
10cm 넘게 내려가면 실패하도록 수정했다.** 성공 조건이나 충돌 기준은 완화하지 않았다.

## 확인한 실제 사례

종료 표본을 보강한 비교의 첫 학습 후 전체 DEV4에서 상단 왼쪽 env122는
857동작 뒤 시간 초과로 끝났다. 박스 중심은 **z=0.1229799986m**, 랙 지지면
간격은 **−1.6551m**였다. 양손–flap 표면 거리는1.6007m·1.8975m였고 실제
파지는 없었다. 위치·속도는 유한하고 다른 안전 위반도 없었다. 기록된
`box_drop=false, unsafe=false, time_out=true`는 기존12cm 검사가 이 낙하를
누락했음을 보여준다. 해당 평가의 actor691/Q4812도 일치 모델로 보존했다.

완료된13개 관리 실행의 JSON만 살펴보면, 랙 간격−0.5m 미만·중심 높이
12–20cm·낙하 판정false인 후보178건이 있었고167건은 안전한 시간 초과로
기록됐다. TRAIN·DEV와 서로 관련된 실행을 포함하는 원인 탐색용 집계다.
모든 낙하를 찾은 결과나 독립적인 성능 통계가 아니며, 모든 시간 초과의
원인이 낙하라고 해석하지 않는다. 활성 HDF·replay는 읽지 않았다.
[원본 종료 상태와 집계](assets/rl_v2_grasp_drop_guard_bug_20261007.json).

## 수정과 호환성

| 항목 | 적용 |
| --- | --- |
| 새 grasp 낙하 조건 | 현재 중심 z − 자신의 초기 정착 중심 z < −0.10m |
| 기존 조건 | 원점 대비 중심 z<0.12m 또는 유효하지 않은 상태도 계속 실패 |
| 초기화 중 | 정착이 완료된 이후에만 판정 |
| 실패 보상 | 기존 −12와 성공 보상 억제를 유지하고 즉시 실패 종료 |
| Carry·place·full | 운반대에 내려놓는 정상 동작을 위해 새 상대 낙하 조건 미적용 |
| 계산 | 이미 측정한 초기 높이·현재 높이의 비교. 새 센서·물리 계산 없음 |
| 과거 staged 학습·영상 | 기록에 새 표식이 없으면 당시의12cm 조건으로 복원 |
| 새 Q·reward bank | 새 계약으로 시작. 과거 낙하 라벨·Q를 그대로 재개하지 않음 |
| Frozen 초기 actor | 검토한10cm 조건만 actor 호환성에서 분리. Q·replay 계약은 엄격히 유지 |

공통 [box_drop.py](../src/kuavo_isaaclab_scene/rl/multi_box/geometry/box_drop.py)에서
조건과 계약을 관리한다. 표준 SAC·PPO의 새 학습, staged SAC 및 Quest grasp
보상 확인에 연결했다. `MultiBoxSpec.max_box_drop_height` 기본값은0.10m이며
`None`은 과거 조건이다. 새 SAC/PPO checkpoint 호환성은 조건 변경을 거부한다.
새 replay에서 실패나 시간 초과 라벨을 몰래 바꾸지 않는다.

## 새 학습 입력과 검증

성공 이벤트+64 비교의 학습 전 입력에서 기존 actor4337의 동작만 보존하고,
Q·optimizer·온라인/성공/n-step reward 은행이 모두0인 상태에 새 낙하 계약을
적용했다. 전체 trainer를 실제로 복원했으며 기존 TRAIN1,240개 관측에서
몸체 목표·양쪽 binary jaw가 같고 모든 모델·네 optimizer가 정확히 일치했다.
실제 reward manager와 초기화된 환경에서도 기록한 낙하 제한을 검사하도록 했다.
관련 최소 검사 **49개 통과**, 별도 GPU 통합 검사1개는 실행하지 않았다.
[Full trainer·새 계획](assets/rl_v2_reset_drop_fulltrainer_and_plan_20261007.json).

박스·base·주변 박스·단단한 동적 flap 무작위화와 파지 조건, 탐색 설정은
그대로다. 새 TRAIN1,536조건은 seed2340700000으로 다른13개 계획과 겹치지
않고 원래 DEV128개·17wave를 유지한다. 독립 FINAL은 사용하지 않았다.
위 full trainer 증거는 실행 전 입력 검증이다. 그 후 **10/07 12:41 KST에
GPU3에서 별도 SAC를 시작했고12:48에 첫 전체 DEV 진입을 확인했다.** 실제
writer3997160·supervisor·서비스·`CUDA_VISIBLE_DEVICES=3`와 GPU learner를
확인했다. 초기화된 환경의10cm 낙하 조건과 reward manager의+64 로그,
env·manifest·실제 agent·초기 checkpoint의 계약 일치, actor/Q·모든 학습
은행0을 확인했다. 시작 당시 기존 네 학습은 유지했고 전체 초기 DEV와
학습 후 개선은 아직 측정 전이었다. [실제 시작·Drive 설정](assets/rl_v2_success64_reset_drop_actual_launch_20261007.json),
[실제 환경·보상·agent 복원과 첫 DEV](assets/rl_v2_success64_reset_drop_first_actual_DEV_20261007.json).

이후 전체 초기 DEV128은 **13/128(중간 좌6·우6, 상단 좌1·우0)**으로 완료됐다.
새 판정의 박스 낙하39·랙 충돌53·시간 초과12·초기화 무효11건을 원래 분모에
포함했다. Actor/Q0회의 초기 성능이며 학습 후 개선을 뜻하지 않는다. 기존
판정의15건과 실제 reset 물리도 동일하다고 가정하지 않는다. 새 TRAIN·Q
갱신을 시작했고 이후 자신의13/128과 비교한다.
[전체 초기 평가·모델 계약](assets/rl_v2_success64_resetdrop_initial_full_DEV_20261007.json).

첫 실제 TRAIN128조건에서는 **안전한 성공3건(중간 좌1·우2, 상단 양쪽0)**,
랙70·낙하35·시간 초과20건이었다. 초기 무효는0이다. 저장이 끝난 actor0·
Q1594 모델을 따로 보존했고 자기 성공3경로·1,206행과 원래 TRAIN 배치를
확인했다. 세 경로의 마지막 실제 reward와 같은 critic의 terminal target은
각각62.9948·62.9967·62.9968로 같았다. Bootstrap0·reward clipping 없음을
확인했고 두 Q·두 target은 실제 갱신됐으며 actor·jaw·actor normalizer는 초기와
같았다. 아직 actor warmup 중 얻은 탐색 성공으로, 학습 후 greedy 개선은 아니다.

초기 DEV의 낙하39건과 첫 TRAIN의35건 모두 마지막 단계가 `held_grasp`였다.
첫 TRAIN의 낙하 종료는 control step548–642에서 발생했다. Base 접근 중에
끝난 실패는 아니지만, 마지막 단계만으로 처음 접촉한 순간이나 낙하 원인을
확정하지 않는다. 새 판정 때문에 시작부터 불가능해졌다고 단정하거나 낙하
기준을 다시 완화하지 않고, 실제 접촉·양손 파지 경험과 이후 전체 평가를 본다.
[첫 실제 TRAIN·모델·성공 target·낙하 단계](assets/rl_v2_resetdrop_success64_first_actual_TRAIN3_20261007.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_reset_drop_actor.py \
  --initial-checkpoint /absolute/path/to/pristine-fresh-Q-input/checkpoint_00000000.pt \
  --training-manifest /absolute/path/to/matching-input/training_manifest.json \
  --waypoints /absolute/path/to/matching-input/waypoints.json \
  --output-dir /absolute/path/to/unique-reset-drop-inputs
```

이 명령은 이미 학습한 Q나 reward 전이가 들어 있는 입력을 거부한다.
학습 시작은 별도 관리 진입점으로 수행한다. 기존 Drive로 checkpoint·계약·로그만
300초마다 크기·MD5 검증하고 최신2개 및 검증된 최신2개를 보호한다.
Raw replay·HDF는 로컬에 남긴다. 종료 로그는 writer 종료 후 검증한다.

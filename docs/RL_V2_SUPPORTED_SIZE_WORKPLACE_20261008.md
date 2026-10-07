# 지원하는 박스 크기의 실제 TRAIN 접근 위치 측정

현재 구역별 초기 정책의34/128회 성공은 small 박스만의 개발 평가다.
원래 목표에는 다른 박스 크기와 base·박스·배경·움직이는 flap의 무작위화도
포함한다. 현재 정책의 성능을 전체 목표 달성으로 보지 않는다.

| 위치 | 측정할 실제 박스 크기 W × D × H |
| --- | --- |
| 중간 좌·우 | small 26.6 × 18.5 × 13cm / medium 32 × 22 × 18.5cm |
| 상단 좌·우 | small 26.6 × 18.5 × 13cm |

## 무엇을 추가했는가

`GraspLayout.target_box_type`은 선택형 reset 설정이다. 크기를 생략하면 기존
직렬화와 동작이 그대로다. 명시하면 실제 지원 asset의 type·dimensions·pool을
선택한다. 큰 footprint를 의미상 반 선반 안에 배치하되 원래2..4cm 안쪽
perturbation과 yaw·depth·base·배경 범위를 줄이거나 clamp하지 않는다.
박스 몸체 root가 바닥면에 가까운 현재 USD wrapper의 위치 기준도 보존한다.
기록된 시연 action·성공·reward를 다른 크기의 데이터로 재라벨링하지 않는다.

기존 파지 단계는 측정한 박스 크기가 다르면 실행을 거부한다. 이 검사를
일반 학습에서 유지하며, 명시적 고정 TRAIN 진단에만
`--unmeasured-size-workplace-probe`를 추가했다. 실제 medium을 쓰더라도
small의 접근 위치를 medium의 측정된 성공 위치라고 표시하지 않는다.
원래 측정 source template·actor·Q·normalizer·optimizer 계약은 유지하고
새 크기의 후보는`measured_success: false`로 기록한다.

같은16개 TRAIN 초기 조건에8개 접근 후보를 비교한다. 각 구역4조건이며
중간은small2·medium2, 상단은small4다. 원래 후보도 포함한다. 물리 요청은
128개지만 독립적인 초기 조건은16개다. 후보 사이 flap 추첨과 접촉 이력은
같다고 가정하지 않는다. 성공·충돌·초기 무효·시간 초과를 모두 기록하고
중간의 두 크기 결과를 각각2조건씩 따로 집계한다. 섞은4조건의 성공 수는
크기별 학습 waypoint 근거로 사용할 수 없다.

이 진단은900스텝·모델 고정·TRAIN 전용이다. DEV·FINAL·학습 모드·지원하지
않는 크기·상단medium을 거부하며, Q/replay 전이를 가져오지 않는다.
지역 actor의 전체198개 모델·normalizer tensor가 바뀌지 않아야 한다.
기존 단일 actor의178개 검사도 유지하고 단순 최소 개수 검사로 완화하지 않는다.

## 사용법

CPU 준비 도구는 다음과 같다. 아래 경로는 모두 고유 실행의 예시다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_region_grasp_layouts.py \
  --output-dir /absolute/path/new_mixed_layouts --seed-origin 4880000 \
  --train-per-region 4 --development-per-region 32 --eval-per-region 32 \
  --mixed-middle-box-types

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/rl python scripts/rl/prepare_size_workplace_probe.py \
  --layout-recipe /absolute/path/new_mixed_layouts/recipe.json \
  --training-manifest /absolute/path/compatible_source/training_manifest.json \
  --output /absolute/path/new_inputs/mixed_size_TRAIN16_candidates8.json
```

실제 진단은 기존 관리 진입점에`--no-training --cpu-workplace-probe
--base-waypoint-probe --unmeasured-size-workplace-probe`를 전달한다.
`--steps 900`을 사용하고 호환된 고정 checkpoint·원래 waypoints와 manifest를
지정한다. `--physics-device cpu`와 별개로 렌더러 GPU를`--gpu`와
`CUDA_VISIBLE_DEVICES`로 제한한다. 예시는 GPU0을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rl/batched_staged_goal_with_drive.py \
  --experiment-dir /absolute/path/unique_size_probe --gpu 0 --physics-device cpu \
  --checkpoint /absolute/path/compatible_source/checkpoint.pt \
  --training-manifest /absolute/path/compatible_source/training_manifest.json \
  --waypoints /absolute/path/compatible_source/waypoints.json \
  --waves-json /absolute/path/new_inputs/mixed_size_TRAIN16_candidates8.json \
  --demo-dataset /absolute/path/demos.hdf5 \
  --native-seed /absolute/path/lower_seed.hdf5 --native-seed /absolute/path/upper_seed.hdf5 \
  --no-training --cpu-workplace-probe --base-waypoint-probe \
  --unmeasured-size-workplace-probe --steps 900 --checkpoint-log-backup-only
```

원래 writer의 정상 종료 후`summarize_cpu_workplace_probe.py`로 집계한다.
결과의`promising_by_region_and_box_type_for_fresh_TRAIN_recheck`를 참고하되
새 크기의 실제 TRAIN 재확인과 크기별 계약을 거친 뒤 별도 SAC를 시작해야 한다.
일반 학습에 진단 flag를 붙여 검사를 우회하지 않는다.

## 지금 확인한 범위

관련77개 검사와 실제 구역별 전체 trainer의 복원을 통과했다. 새128개
neutral reset·stage 구역과 medium 후보32개를 준비했고 원래 모델·optimizer·
관측 정규화·source 시연을 보존했다. 새 보상·Q/replay 은행은0이다.
이는 준비 및 연결 검사이며 실제 medium 파지 성공은 아직 측정 전이다.
별도 DEV128·FINAL128은 준비했지만 사용하지 않았다.
[실제 입력·복원 근거](assets/rl_v2_regional_mixed_size_TRAIN_workplace_preparation_20261008.json).

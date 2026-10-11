# SAC 성공 보호 후보의 내부 교정 탐색

목표는 box/base/flap randomization과 실제 양손 opposing two-pad pinch·hold0.25초·
proof-lift·안전 기준을 유지한 성공이다. 이전 실행과 기존 성공 모델·영상은 보존한다.
이번 변경은 보호 표본의 갱신 정체를 다루며 물리 성공률 개선의 증거가 아니다.

## 실제 정체와 원인 범위

GPU3 candidate-repair 실행의 실제 초기 DEV는15/128이었다. 새 production 갱신65개에서
중간 왼쪽47개·상단 오른쪽20개만 실제 parameter를 옮겼고 상단 오른쪽의 뒤29개는
모두 이동0이었다. 전체 actor counter 증가에는 성공 없는 두 구역의 갱신도 들어가므로
이 숫자만으로 성공 구역이 정상적으로 갱신됐다고 설명하지 않는다.

상단 오른쪽은 가장 작은1/1024 후보에서도 같은 tail64 오른팔 한도 위반이 반복됐다.
실제 저장 GPU3 모델(actor512/Q4,096)의 복사본에서 최대 위반 행이 교정마다 달라지고,
float32 절대 관절 목표의 분해능이 한도에 대한 비율 약0.00030~0.00119인 상태를 확인했다.
같은 정규화 목표의 작은 변화가 같은 실제 float32 관절 목표로 계산되기도 했다.
그러나 명령 분해능2ULP만큼 내부 여유를 더한 진단에서는 갱신이 여전히 막혔다.
**정밀도만을 유일한 원인으로 확정하지 않는다.** 제한된 반복에서 여러 최대 행을 교정하는
방향·교정량도 영향을 줬다. 해당 여유 방식은 실제 학습 코드에 추가하지 않았다.

## 변경한 부분

몸체 한도만 위반한 후보에서는 기존 교정 방향의1/2/4배를 내부에서 비교한다.
매번 원래 투영 Adam proposal의 구역별 norm 안으로 제한하고, 보호 jaw의 오답 수를
우선한 뒤 고정 한도에 대한 상대 위반 합이 줄어든 후보를 고른다. 선택된 후보에도
기존 구역별1~1/1024 축소·마지막 전체 cohort·행/손별 greedy jaw 검사를 적용한다.
**내부 비교 점수는 보상이나 성공 조건이 아니며 위반 후보를 학습 갱신으로 수용하지 않는다.**

Jaw loss/label도 위반한 교정은 기존 한 번의 교정을 유지한다. 구역별 원래 Adam norm,
실제 이동 norm, 교정 시도 수·선택 gain을 기록한다. 전체 갱신이 거절되면 구역 scale도
실제 원복된0으로 기록하고 Adam/entropy의 기존 거절 처리를 유지한다.

수정 파일은 `success_cohort_guard.py`, `success_update_guard.py`와 진단 도구·회귀 검사다.
CPU PhysX/GPU3 CUDA learner·controller·reward·DR·curriculum 없음·성공·안전 계약은
그대로다. 보호 명령 예산(goal0.001, servo 접근0.02/tail0.01, jaw BCE0.002)은 늘리지 않는다.
기존 고정 경로·phase별 batch와 float64 보호 손실을 유지한다.

## 최소 검증 결과와 한계

실제 완료 TRAIN provenance를 대조한 저장 모델의 복사본으로 두 번 비교했다.
같은 RNG·Adam·TRAIN 성공 Q256행·보호64행을 사용했고 모든 고정 한도와 전체
보호 jaw를 확인했다. 원본 모델과 한도는 그대로였다.

| GPU3 복사본 제안 | 기존 중간 왼쪽 norm | 변경 후 | 기존 상단 오른쪽 norm | 변경 후 |
|---|---:|---:|---:|---:|
| 첫 번째 |0.0000303|0.0000613|0|0.0000446|
| 두 번째 |0.0000154|0.0000622|0|0.0000465|

변경 후 상단 오른쪽 scale은 두 번 모두0.125였다. 원래 구역별 Adam norm
0.0003566/0.0003722보다 실제 이동이 작았다. 실제 body mean/jaw logit도 바뀌었다.
두 경우 모두 중간 왼쪽이 갱신돼 기존/변경판 모두 entropy는 갱신했으므로,
이번 결과를 entropy가 처음 재개됐다는 근거로 쓰지 않는다. 성공 없는 두 구역은 진단 Q
표본이 없어 이동0이었다. production에서 이 구역이 멈춘다는 뜻은 아니다.

추가한 두 회귀 검사는 부족한 교정량의 회복과 몸체 오차 감소를 위한 보호 jaw 변경
금지를 실제 Adam/encoder/전체 보호 검사로 확인한다. 관련27개 검사가 통과했다.
GPU 비교는 성공 은행만 사용했고 production 혼합 replay와 teacher/episode-return 보조
critic loss를 재구성하지 않았다. 저장된 TRAIN만 사용했으며 현재 raw replay/HDF와
DEV/독립 FINAL을 갱신 데이터로 쓰지 않았다. **실제 성공 보존·개선은 새 TRAIN과 같은
전체 DEV128으로 다시 확인해야 한다.**

[GPU3 실제 TRAIN provenance·동일 원본·구역별 norm·전체 보호 검사](assets/rl_v2_body_restoration_GPU3_disposable_20261011.json).

재현할 때에는 신뢰할 수 있는 기존 frozen source와 완료 TRAIN metrics를 사용한다.
GPU3의 기존 UUID·single-GPU namespace·사설 runtime을 유지한다.

```bash
python scripts/rl/audit_candidate_feasibility_repair.py \
  --checkpoint /path/to/immutable/checkpoint.pt \
  --metrics /path/to/completed/source/metrics.json \
  --baseline-source-root /path/to/trusted/frozen/baseline \
  --device cuda:0 --output /path/to/unique/audit.json
```

## 실제 적용과 다음 확인

현재 진행 배치의 실제128개 TRAIN 결과를 먼저 기록하고 안전하게 전환한다.
새 고유 GPU3 실행에서 같은 계약의 유한 모델·Q·replay·Adam·성공 cohort를 이어간다.
이번 새 초기 DEV를 실제 측정하며 이전15회나14회를 대신 넣지 않는다. 새 TRAIN
seed는 기존 실행과 겹치지 않고 같은6개 구역/크기·box/base/flap DR을 유지한다.
새 실제 보호 구역의 이동0 비율·body/jaw 이동·critic/entropy를 확인하고, 초기/중간/종료
DEV128의 실제 성공·invalid reset·안전 원인·timeout·수치 오류를 비교한다.
독립 FINAL은 수정·탐색에 사용하지 않는다. 목표는 미달성이다.

[이전 실제 초기·첫 갱신·실패 영상](RL_V2_SERVO_TUBE_SUCCESS_GUARD_20261011.md).

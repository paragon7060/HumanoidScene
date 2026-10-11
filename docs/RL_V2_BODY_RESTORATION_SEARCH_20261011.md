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

**10/11 23시 정정:** 아래 복사본 비교에서는 복원 Adam의 CPU step tensor 공유가
발견됐다. 당시 ‘같은 Adam’ 통제 주장은 철회하며 숫자를 변경의 인과적 효과로 사용하지
않는다. 디스크 원본과 실제 TRAIN/DEV 결과는 별개다. optimizer를 독립 복원하도록
수정하고 같은 critic·Adam norm을 확인해 다시 비교한 결과와 적용 범위는
[새 수치 진단·수정 기록](RL_V2_QUANTIZED_GUARD_REPAIR_20261011.md)에 있다.

실제 완료 TRAIN provenance를 대조한 저장 모델의 복사본으로 두 번 비교했다.
같은 RNG·TRAIN 성공 Q256행·보호64행을 사용했고 모든 고정 한도와 전체
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

## 직전 완료 TRAIN128과 실제 전환

Candidate-repair 실행의 첫 TRAIN128은 실제 성공9/128, invalid reset1, unsafe58,
timeout60, numerical failure0이었다. 유효127개를 별도 표시하되 분모128을 유지한다.
9회 모두 실제 양손 opposing two-pad pinch·hold0.2667초·proof-lift·안전을 확인했다.

| 구역·크기 | 성공/요청 | 초기 유효 | 안전 실패 | timeout |
|---|---:|---:|---:|---:|
| 중간 왼쪽small |5/16|16|11|0|
| 중간 오른쪽small |0/16|16|1|15|
| 상단 왼쪽small |0/32|32|0|32|
| 상단 오른쪽small |4/32|31|15|12|
| 중간 왼쪽medium |0/16|16|16|0|
| 중간 오른쪽medium |0/16|16|15|1|

상단 오른쪽의 초기화 실패1개와 별개로 상단 왼쪽32개는 모두 유효timeout,
양쪽medium은 유효16개 중16/15개가 안전 종료였다. 초기화뿐 아니라 policy 문제가
남았다. 안전58회는rack55·drop3이며, rack55회 모두held_grasp 단계였다. 종료 시
최대힘은220.74N이었다. 종료 시 최대힘 몸체는 오른손 gripper base33·왼팔4번째link11·
왼손 gripper base6·오른팔4번째link3·오른팔7번째link2였다. 첫 접촉/전체 접촉 이력은 아니다.

새 actor404회·critic1,614회에서 중간 왼쪽324/404(80.2%)·상단 오른쪽384/404(95.0%)는
실제 이동0이었다. 성공 없는 두 구역은 모든404회 갱신했고 보호 baseline 한도 초과는0이었다.
성공 구역의 반복 정체 때문에 현재 설정을 계속 반복하지 않고 개선판으로 전환했다.
첫 완료 TRAIN 결과와 실패 원인을 기록한 뒤 우리 writer에만 정상 종료 신호를 보냈다.
다음 TRAIN은step1에서 중단되어 완료 성공률을 계산하지 않는다. 최종 전체 DEV는 없으므로
초기 DEV15회와 TRAIN9회를 전후 성능 비교나 lost/gained success로 쓰지 않는다.

[완료된 첫 TRAIN128·6조건·안전 원인·실제 새 갱신](assets/rl_v2_candidate_repair_first_TRAIN128_20261011.json).

## 실패 구역에서 실제 제어 추종과 접근을 분리

새 초기 평가가 진행되는 동안 직전 candidate-repair의 **학습 전 DEV128**의 닫힌 기록을
CPU로 분석했다. 실제 출발 model counter는manifest의actor447/Q3,836으로 대조했다.
전체128개 중 유효115경로를 읽었고, 각 경로의 종료 전60행에서만 관절 오차를 계산했다.
Controller telemetry는절대 목표가 아니라`pending target − measured joint position`
(raw416:436)이다. 실제 decoder와 같은 의미로 계산했고 DEV를 학습 전이나 optimizer
진단에 넣지 않았다.

| 조건·종료 종류 | 경로 수 | 팔 tracking RMS 중앙값 L/R(rad) | TCP–배정 flap 표면거리 중앙값 L/R(m) |
|---|---:|---:|---:|
| 중간 오른쪽small · 안전timeout |13|0.00249 /0.000435|0.238 /1.076|
| 상단 왼쪽small · 안전timeout |28|0.000565 /0.000891|0.0854 /0.0392|
| 중간 오른쪽medium · 안전timeout |7|0.00299 /0.000529|0.213 /0.931|
| 중간 왼쪽medium · 충돌 |16|0.0200 /0.0224|0.0333 /0.0335|

안전timeout 세 조건은 종료 전에는 명령을 잘 추종하면서도 목표 flap에서 먼 자세를
유지했다. 따라서 이 구간은 접근 명령·탐색/학습의 문제를 점검할 근거다. 이 통계로
초반 전체 경로·torque·중력보상·torso의 정상 상태를 증명하지는 않는다. 충돌 경로의
오차는 접촉 반응과 함께 측정됐으므로 제어 실패의 독립적인 원인으로 해석하지 않는다.

지금 실행의 보호 교정 효과와 실제 새 TRAIN을 먼저 확인한다. 이후 정체가 지속된다면
성공 없는 구역의 실제 실행 표본에서 Q-to-goal gradient 전달·도달 거리 변화·탐색 범위를
대조해 접근 문제에 맞는 작은 수정을 선택한다. 이미 확인한 servo clipping 진단과
이번 추종 진단을 구분한다. 현재 learner/물리/reward/DR은 이 분석 때문에 바꾸지 않았다.

[6조건·실제115경로·명령 추종·종료 거리와 분석 범위](assets/rl_v2_candidate_repair_DEV128_tracking_20261011.json).

## 실제 적용과 다음 확인

10/11 21:26 KST GPU3 새 고유 실행
`GPU3_URDF_body_restoration_SAC1024_20261011_212605`를 시작했다.
실제 run은`batch_sac_20261011_212605_6a2d4a`, writer1171480·supervisor1171443이다.
Frozen source는`758d6a3838fb8060c740cec1ed0113f8107febcb`이다.

직전 writer·supervisor 종료, 유한 model/optimizer, 최종 Drive 검증·닫힌 logs를 확인했다.
의도적 중단의 원래exit1/interrupted를 보존했다. 실제 source model SHA256은
`5f4ab3ef6d844345c897d963b215599a6dac7205775c73c63f2e7649d41bc74d`이다.
Raw replay/HDF는 업로드하거나 삭제하지 않았다. 기존 최선 모델·평가 영상도 보존했다.

새 DEV의 실제 learner는actor851/Q5,450·replay250,000·TRAIN 성공은행7,534행
(중간 왼쪽10경로3,963행·상단 오른쪽6경로3,571행)·고정 cohort4경로2,013행을 복원했다.
이 누적 counter를 새 학습량으로 세지 않는다. 최신 성공은행에서 이미 빠진 첫4경로의
actor obs/action·손별 보호 정답·고정 한도는 이전 모델과 tensor가 동일했다.
Global replay capacity만으로 해당 경로 전체가 eviction됐다고 주장하지 않는다.
상속된 최초 actor-only 초기화의`old_Q_or_replay_imported:false` 같은 설명은
이번 같은 CPU 계약의 재개 상태가 아니다. 실제 복원된 row 수·counter와 restore 코드로 판단했다.

새 TRAIN seed는 기존 실행과 겹치지 않는1,024조건이다. 중간/상단·왼쪽/오른쪽·small/medium의
같은6조건과 box/base/flap DR을 유지한다. 이번 초기/중간/종료 DEV는 동일128개 요청 조건이다.
새 초기 성공률은 아래 실제10/128로 확인했다. 이전15회/14회를 대신 넣지 않는다. GPU3 UUID와
private namespace에서`nvidia3`/`renderD131`만 노출됨을 확인했다. PGS·physics dt1/120·
control dt1/30·curriculum0 계약을 유지하고 GPU1·2 또는 다른 사용자 실행은 건드리지 않았다.

첫300초 백업 검증이 완료됐고 오류는 없었다. 기존 checksum 업로드·최근2개 checkpoint·
writer 종료 후 logs 검증을 유지한다. 빈 CUDA 마스크의 CPU 감시1179294가 실제 실행되며,
새 writer·supervisor 종료 및 최종 Drive 검증 뒤 같은 대화에서 한 번 후속 점검한다.
기존 감시721069는 종료했다. 이 연결은 기존 Codex 서비스와 서버가 계속 실행돼야 한다.

[실제 초기화·재개·첫 cohort 보존·GPU 격리·백업·후속 점검](assets/rl_v2_body_restoration_SAC_actual_startup_20261011.json).

## 이번 출발 모델의 실제 초기 DEV128

**초기 성공10/128(7.8%), initial-valid115, invalid reset13, unsafe57, timeout48,
numerical failure0이다.** 성공10회는 실제 양손 opposing two-pad pinch·hold·proof-lift·
안전을 만족한 경로만 센 것이다. 새 TRAIN 갱신 전 actor851/Q5,450의 평가이며,
원본 종료 모델과77개 model tensor가 모두 동일하다. 이번 수정판의 학습 효과가 아니다.

| 구역·크기 | 성공/요청 | 초기 유효 | 초기 무효 | 안전 실패 | timeout |
|---|---:|---:|---:|---:|---:|
| 중간 왼쪽small |7/16|16|0|9|0|
| 중간 오른쪽small |0/16|13|3|0|13|
| 상단 왼쪽small |0/32|28|4|1|27|
| 상단 오른쪽small |3/32|27|5|17|7|
| 중간 왼쪽medium |0/16|16|0|16|0|
| 중간 오른쪽medium |0/16|15|1|14|1|

동일128 요청 조건의 직전 초기 actor447/Q3,836(15/128)과 공통 유효115개에서
lost6·gained1이다. 중간 왼쪽small은9→7(lost3/gained1), 상단 오른쪽small은6→3
(lost3/gained0), 나머지4조건은0→0이다. Unsafe45→57·timeout55→48이었다.
**실제 reset의 물리 상태까지 같다고 주장하거나, 서로 다른 run의 초기 평가를 같은
run의 최종 평가로 설명하지 않는다.** 이전 학습을 거친 출발 모델에서 성공 퇴행이 보였다.

Unsafe57은rack55·drop2이며 rack55회 모두held_grasp 단계였다. 종료 시 최대힘은
314.57N이다. 종료 시 최대힘 몸체는 오른손 gripper base25·왼팔4번째link15·
오른팔4번째link8·왼손 gripper base5·왼팔7번째link1·오른팔7번째link1이었다.
초기화 실패와 별개로 중간 오른쪽small의 유효13개 모두timeout, 중간 왼쪽medium의
유효16개 모두rack이므로 policy 문제가 남는다. 이 최대힘 통계는 첫 접촉 이력이 아니다.

이번 초기 모델은`artifacts/rl/protected_dev_models_20261011/` 아래 별도로 보존했다.
SHA256은`4a3fa67140fbf9c42fd285fefceb966c87628db4a438f861ae5f27113b17b9c4`이다.
이전15회 성공 모델과 영상도 보존한다. 이후 같은 run의 중간/최종 DEV는 이번 실측10회를
기준으로 비교하고, 이전 최선15회와의 비교는 별도로 표시한다.

[이번 실제128개·6조건·초기 모델 checksum·lost/gained](assets/rl_v2_body_restoration_initial_DEV128_20261011.json).

## 첫 TRAIN 진행 중 실제 갱신 확인

10/11 22:24 KST 첫 TRAIN128의step241에서 새 actor95회·critic380회를 확인했다.
누적actor946/Q5,830에서 출발actor851/Q5,450을 뺀 값이다. TRAIN wave는 진행 중이므로
완료 성공률을 계산하지 않는다. 고정 cohort4경로2,013행과 TRAIN 성공은행7,534행을 유지했다.

| 구역 | 실제 이동/새95제안 | 이동0 | 이동0 비율 |
|---|---:|---:|---:|
| 중간 왼쪽 |82/95|13|13.7%|
| 상단 오른쪽 |12/95|83|87.4%|
| 중간 오른쪽 · 성공 cohort 없음 |95/95|0|0%|
| 상단 왼쪽 · 성공 cohort 없음 |95/95|0|0%|

직전 완료 TRAIN의 중간 왼쪽 이동0은80.2%였고 이번 부분 구간에서는13.7%로 줄었다.
서로 다른 학습 구간·상태·Q이므로 통제된 효과 비교는 아니다. 상단 오른쪽은 여전히
87.4%에서 이동0이며 최근 보호 검사에서 tail64 팔 명령 한도가 막았다. 모든 actor
counter 증가를 양쪽 성공 구역의 갱신으로 설명하지 않는다. 실제 이동0은 갱신으로 세지 않는다.

Baseline 고정 한도 초과는0이고, 내부 교정gain1/2/4가 실제 선택됐다. 성공 없는 구역은
계속 갱신되며 보호 대기 상태가 아니다. Critic과 entropy도 갱신했다: 현재 연속alpha
0.000007527·이산alpha0.013140·Gaussian std 평균0.008829이다. Parameter norm 기록은
body mean/jaw 출력 변화나 실제 성공 보존의 보장이 아니다. 다음 완료 TRAIN와 같은
전체 DEV128에서 실제 성공·안전 실패를 확인한다. 상단 오른쪽 정체가 지속되면 닫힌
실제 TRAIN/model을 사용해 위반 행·관절과 교정 방향을 분석한 뒤 작은 개선을 선택한다.

22:21의300초 Drive 검증은 완료됐고 backup error는 없었다. 체크포인트·계약·닫힌
로그만 백업하며 현재 raw replay/HDF는 읽거나 업로드·삭제하지 않았다.

[새 갱신95회·구역별 실제 이동·최신 보호 후보·critic/entropy·백업](assets/rl_v2_body_restoration_first_TRAIN_guard_snapshot_20261011.json).

Production 보호 구역의 이동0 비율·선택 교정gain·원래 Adam norm과 실제 이동·body/jaw 변화·
critic/entropy를 확인한다. 초기/중간/종료 DEV128의 실제 성공·invalid reset·안전 원인·timeout·
수치 오류를 비교하며 공통 유효 조건의 lost/gained success를 별도 표시한다.
표본의 보호 검사가 실제 성공 보존을 보장하지 않는다. 독립 FINAL은 수정·탐색에 사용하지 않는다.
목표는 미달성이다. Notion의 기존 image57(외부1 포함)/video47·table5개와 기존 미디어 주소를 보존하고
최신 상태와 수정·근거만 갱신했다.

[이전 실제 초기·첫 갱신·실패 영상](RL_V2_SERVO_TUBE_SUCCESS_GUARD_20261011.md).

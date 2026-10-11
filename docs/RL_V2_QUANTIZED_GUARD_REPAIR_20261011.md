# SAC 갱신 정체의 수치 원인과 다음 수정

GPU3의 body-restoration 실행은 계속 진행한다. **첫 전체 TRAIN128은 실제 성공10/128,
초기 유효128·무효0·unsafe69·timeout49·numerical failure0이다.** 10회 모두 현재
정책의 greedy 수집에서 나왔으며 상단 왼쪽small에서도2회 성공했다. 양손 opposing
two-pad pinch·hold0.2667초·proof-lift·안전을 확인한 경로만 셌다. 학습 중 actor가
바뀌므로 고정된 마지막 모델의 성공률이 아니다. 초기 DEV10/128과 다른 요청 조건의
TRAIN10/128을 직접 성능 비교로 쓰지 않는다. 같은128조건의 중간/종료 DEV가 필요하다.
이전 최선15회 모델·영상도 보존한다.

[전체 TRAIN128·수집 방식·은행/cohort·모델 checksum](assets/rl_v2_body_restoration_first_full_TRAIN128_20261011.json).
23:04의[부분 TRAIN 기록](assets/rl_v2_body_restoration_partial_TRAIN_20261011.json)은 역사적 진행 snapshot이다.

## 첫 전체 TRAIN의 실제 변화

| 구역·크기 | 성공/요청 | 초기 유효 | 안전 실패 | timeout |
|---|---:|---:|---:|---:|
| 중간 왼쪽small |7/16|16|9|0|
| 중간 오른쪽small |0/16|16|1|15|
| 상단 왼쪽small |2/32|32|9|21|
| 상단 오른쪽small |1/32|32|19|12|
| 중간 왼쪽medium |0/16|16|16|0|
| 중간 오른쪽medium |0/16|16|15|1|

Greedy 수집108조건에서성공10·unsafe58·timeout40, 접촉 탐색20조건에서성공0·
unsafe11·timeout9였다. 서로 다른 조건의 수집 결과이며 두 방식의 통제 비교가 아니다.
상단 왼쪽의 env38/seed64000209와env78/seed64000219는 실제 greedy 성공이었다.
은행은7,534→8,591행, 고정 cohort는4경로2,013행→6경로3,042행으로 늘었다.
새 상단 왼쪽2경로1,029행이 등록됐고 원래4경로의 obs/action·고정 명령 한도와
기존 보호 jaw 정답은 checkpoint에서 보존됨을 확인했다. 미래 성공의 보장은 아니다.

새 actor403회·critic1,612회 중 중간 왼쪽87회, 상단 오른쪽50회만 실제 parameter가
움직였다. 각각316/403·353/403(78.4%·87.6%)은 이동0이었다. 이 wave 동안 성공
cohort가 없던 중간 오른쪽·상단 왼쪽은403회 모두 움직였다. 첫95제안에서 중간 왼쪽의
이동0이13.7%였던 사실을 이후에도 정상 갱신이 유지됐다는 근거로 쓰지 않는다.
새 상단 왼쪽 cohort는 wave 종료 뒤 등록돼 그 구역의 다음 보호 갱신은 아직 별도 확인이 필요하다.

안전69회는rack67·drop2다. Rack은 모두held_grasp 단계이며 종료 시 최대힘241.81N이다.
최대힘 몸체는 오른손 gripper base30·왼팔4번째link16·왼손 gripper base14·
오른팔7번째link4·오른팔4번째link2·왼팔7번째link1이었다. 첫 접촉/전체 접촉 이력은 아니다.
초기화128개가 모두 유효했는데도 오른쪽small과medium 실패가 남아 policy 문제를 확인했다.

actor1254/Q7062 완료 저장 모델77개 tensor는 모두 유한했고 SHA256
`f0cb64d7ac3732043af41347c61e82bbe3fd5add3c124b06fac703d2bbc007d7`으로 별도 로컬 보존했다.
이 모델은 새 구역 성공을 수집한 TRAIN 실행의 저장점이며 최종 DEV 개선 모델로 설명하지 않는다.
새 구역의 실제 greedy 성공이 있어 현재 전체 학습을 유지한다. 준비한 fallback은 다음 실행용이다.

## 원인과 비교 과정에서 발견한 오류

저장된 actor1024/Q6144의 상단 오른쪽 보호 후보가 가장 작은1/1024에서도
같은 tail64 왼팔 한도에 막혔다. 복사본에서 교정 norm은 약2.35e-10이었다.
위반 관절의 float32 실제 목표 분해능은2.98e-8rad, 한도 대비 비율은약0.000149였다.
교정이 너무 작아 계산된 실제 관절 목표가 바뀌지 않는 사례를 재현했다.
여러 최대 오차 행·관절이 교대로 바뀌는 문제도 있어 정밀도만을 유일한 원인으로
확정하지 않는다. 보호 손실의 float64 계산만으로 실제 float32 목표의 경계가 사라지지 않는다.

별도로 Adam 복원 코드가 저장 state의 CPU tensor를 공유하는 결함을 확인했다.
동일 checkpoint에서 복원한 첫 learner의 갱신이 저장 Adam step을1→2로 바꿨다.
CPU learner에서는 moment도 공유됐다. 세 가지 SAC 형태에서 수정 전 재현됐고,
복원 시 optimizer state를 deep-copy하도록 고쳤다. 디스크 원본과 현재 writer를
변경한 결함은 아니며 물리적 학습 실패의 원인으로 단정하지 않는다.

**이전 disposable 비교의 ‘동일 Adam’ 통제는 확인되지 않았다.**
[body-restoration 기록](RL_V2_BODY_RESTORATION_SEARCH_20261011.md)의 기존 숫자는
역사적 진단으로 남기되 변경의 인과적 효과를 입증하는 짝지은 비교로 쓰지 않는다.
같은 방식의 이전 candidate-repair 진단도 이 한계를 가진다. 실제 TRAIN/DEV 결과,
디스크 모델 checksum과 각 후보의 전체 보호 검사 통과는 별개다.

## 준비한 작은 수정

이미 수용된 보통 Adam 후보는 그대로 유지한다. 기존 교정·축소로 거절된 구역에만,
**몸체 한도만 위반한 경우** 실제 관절 목표 분해능의2ULP 안쪽을 목표로 추가 교정한다.
이 여유는 내부 교정량에만 더한다. 등록된 한도·보호 jaw 정답·고정 경로는 바꾸지 않는다.
원래 투영 Adam norm 안에서1/2/4배를 탐색하고 기존1~1/1024 축소와 마지막 전체
cohort 검사를 모두 적용한다. Jaw loss나 행·손별 정답을 위반한 후보에 이 fallback을
적용하지 않는다. 실제 이동0은 갱신으로 세지 않는다.

물리·controller·reward·box/base/flap randomization·성공·안전·curriculum 없음은 유지한다.
고정 goal/servo/jaw 예산도 늘리지 않는다. **현재 writer는 frozen758d6a3 코드를 사용하므로
이 새 fallback이 실행 중이라고 설명하지 않는다.** 새 구역의 실제 TRAIN 성공이 나왔으므로
현재 수집을 먼저 유지하고, 실제 전체 결과를 검토한 뒤 다음 실행의 적용을 결정한다.

수정 파일은 `sac.py`, `asymmetric_sac.py`, `success_cohort_guard.py`,
`success_update_guard.py`다. Hybrid SAC의 optimizer 복원도 상속한 수정이 적용된다.

## 독립 Adam으로 다시 비교한 범위

현재 완료 저장 checkpoint의 복사본과 **이미 종료·Drive 검증된 이전 TRAIN**의
replay250,000행·성공은행7,534행만 사용했다. Mixed Q256행 중43행이 실제 성공은행이며
actor 보호64행과 measured-return 보조 critic64행을 사용했다. 현재 raw replay/HDF와
DEV/독립 FINAL은 읽지 않았다. 이는 현재 writer의 Q·RNG·새 TRAIN 분포를 재현한 실험이 아니다.

같은 draw의 두 복사본에서 입력 optimizer가 변하지 않았고 critic tensor checksum과
구역별 투영 Adam norm이 정확히 같았다. 원래758d6a3 guard와 새 fallback을 비교했다.

| 복사본 제안 | 기존 중간 왼쪽 norm | fallback | 기존 상단 오른쪽 norm | fallback |
|---|---:|---:|---:|---:|
|1|0|0|0|0.000343973|
|2|0|0.000114259|0|0.000336157|

![저장 모델 복사본의 실제 parameter 이동. 파지 성공률 개선이 아님](assets/rl_v2_body_quantum_fallback_disposable_20261011.png)

두 제안 모두 고정 전체 body/jaw 검사·행/손별 greedy 정답을 통과했고 원래 Adam norm을
넘지 않았다. 상단 오른쪽의 실제 body mean/jaw logit도 변했다. 성공 cohort가 없는
두 구역의 갱신은 비교한 두 판에서 정확히 같았다. 이는 모든 제안이 수용된다거나
실제 파지 성공이 보존·개선된다는 보장이 아니다.

관련 최소 회귀 검사32개가 통과했다. 저장 optimizer 불변·두 복원 learner의 독립성,
이미 수용된 후보의 유지, 잘못된 보호 jaw에 몸체 fallback 금지를 포함한다.
[원본 checksum·같은 critic/Adam·실제 이동·전체 보호 검사](assets/rl_v2_body_quantum_fallback_disposable_20261011.json).

## 다음 실제 확인과 저장

10/12 새 상단 왼쪽이 등록된 actor1254/Q7062·cohort6경로3,042행 모델에서도
독립된 두 복사본 제안을 확인했다. 현재 완료 저장 모델과 공개된 완료 TRAIN metadata를
대조했고, mixed replay/return 보조 표본은 이전 종료·검증된 TRAIN만 사용했다.
같은 critic·투영 Adam·입력 optimizer 불변을 확인했다. 현재 writer의 새 분포 재현은 아니다.

| 제안 | 기존 중간 왼쪽 norm | fallback | 기존 상단 오른쪽 norm | fallback |
|---|---:|---:|---:|---:|
|1|0|0.000207523|0|0.000154131|
|2|0|0.000207099|0|0.000154201|

기존에 수용된 상단 왼쪽과 성공 cohort 없는 중간 오른쪽의 실제 parameter 이동은
두 판에서 정확히 같았다. 고정 전체 한도·행/손별 jaw 정답을 모두 통과하고 실제
body mean/jaw logit 변화도 확인했다. **물리 성공률 개선은 아니다.**
동시 CUDA 진단의 무제한 경합 위험은 자동 검토에서 거부됐다. GPU3의 실제 여유
73,996MiB 확인과 진단 자체의2GiB allocator 상한·여유 감소 시 진단만 종료·180초
제한을 둔 방식은 허용돼 완료됐다. 실제 Torch 최대 allocated309MiB/reserved332MiB이며
driver/context 추가 메모리는 이 수치에 포함되지 않는다. 기존 writer는 계속 진행했다.

[새 실제 TRAIN10경로 provenance·6cohort·같은 critic/Adam·전체 보호 검사](assets/rl_v2_quantized_body_guard_completed_TRAIN1254_disposable_20261012.json).

**실제 CPU 순차 전환 대기 PID2006098이 실행 중이다.** 두 번째 TRAIN128이 모두
완료됐을 때, 기존 두 보호 구역이 각각100제안 이상이고 이동0 비율75% 이상이면
등록된 우리 writer만 정상 종료한다. PID 시작 시각·명령 hash·고유 run을 재확인하며
감독자나 다른 사용자 프로세스에 신호를 보내지 않는다. 이 수치는 운영상의 정체
판단이며 task의 성공 기준이 아니다. 갱신이 회복됐으면 기존 학습을 유지한다.

기존 writer·supervisor 종료와 원래 Drive 최종 checksum 검증 뒤에만 수정판의
새 고유 GPU3 실행을 시작한다. 완료 모델/Q/replay/optimizer/은행/cohort를 이어받고,
새 TRAIN256조건(2배치128)·동일 DEV128의 초기/종료로 물리 효과를 먼저 확인한다.
Box/base/flap DR·reward·물리·성공·안전 기준과 curriculum 없음은 유지한다.
Source의 최종 DEV가 없으면 있다고 말하지 않으며 새 초기 DEV를 실제로 측정한다.
새 GPU 학습은 아직 시작하지 않았다. 파일의 존재만으로 실행을 추정하지 않는다.

00:34 KST의 두 번째 TRAIN511/900스텝에서 actor는1254→1484·critic7062→7984다.
이230제안 중 기존 중간 왼쪽과 상단 오른쪽은 모두 실제 이동0이다. 새 상단 왼쪽은
71회 이동·159회0, 성공 없는 중간 오른쪽은230회 이동이다. 고정 baseline 한도 초과0·
첫 성공 보호 대기false·cohort6경로3,042행을 확인했다. 두 entropy 계수도 첫 배치 종료와
달라졌다. 전체 수용 counter 증가를 각 보호 구역의 정상 갱신으로 설명하지 않는다.

종료·Drive 검증된 **이전 TRAIN** replay의8,192개 저장 slot을 CPU로 조회했다.
첫 TRAIN 완료 모델의 고정 normalizer에서 추가한 손–flap 중점 상대관계38D와 선택
박스 위치는4구역 모두 clipping0이며 구역 routing도 모든 표본에서 보존됐다.
일부 기존 one-hot·회전 좌표는 잘렸지만 이 표본에서 상대관계가 사라졌다는 근거는 없다.
현재 writer의 새로운 분포나 실패 원인 전체를 확정하는 진단이 아니며 활성 replay/HDF와
DEV/독립 FINAL은 읽지 않았다. 물리·보상·관측 계약을 추가로 바꾸지 않는다.
[종료 TRAIN 표본·normalizer·4구역 clipping 확인](assets/rl_v2_closed_TRAIN_observation_clipping_20261012.json).

[실제 CPU worker·조건부 종료 범위·검증 후 순차 시작](assets/rl_v2_quantized_body_guard_transition_waiter_20261012.json).

다음 TRAIN에서6조건별 성공·invalid reset·unsafe·timeout·numerical failure를 확인한다.
상단 왼쪽2경로가 등록된 뒤에도 실제 보호 갱신과 탐색이 유지되는지 확인한다.
현재 run의 초기 DEV10/128은 역사적 기준으로 남긴다. 새 run은 자체 초기 DEV128을
실제로 측정한 뒤 동일128요청 조건의 종료 평가와 비교하며, 공통 유효 조건의
lost/gained와 안전 실패 단계·최대힘·몸체를 별도 표시한다. 독립 FINAL로 조정하지 않는다.

GPU3 UUID와 single-GPU namespace를 유지한다. GPU1·2와 다른 사용자 프로세스는
건드리지 않는다. 기존300초 Drive checksum 검증·최근2개 checkpoint·종료 logs 검증과
CPU 완료 후속 점검을 유지한다. 현재 raw replay/HDF는 업로드·삭제하지 않는다.
목표는 미달성이다.

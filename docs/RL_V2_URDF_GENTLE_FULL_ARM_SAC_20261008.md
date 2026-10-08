# 실제 관절 명령에 맞춘 full-arm SAC 탐색

2026-10-08에는 장기 학습을 유지하면서 팔 탐색 폭을 조정한 별도 SAC를 준비했다.
안정적인 학습 개선과 여섯 구역/크기 모두의 양손 파지는 아직 달성하지 못했다.

## 바꾼 이유

같은 full-arm 초기 정책의 실제 TRAIN128 수집에서 큰 팔 탐색27회는 성공0·안전
위반21·시간 초과6회였고 greedy90회에서는 성공13회였다. Actor 업데이트0인
수집이며, 학습 후 평가나 동일 접촉 이력을 비교한 성공률은 아니다.
[실제 경로·모델 측정](assets/rl_v2_actual_measured_TRAIN128_closed_gradient_20261008.json).

Full-arm 목표 범위는 실제 관절의 한 step 이동 범위보다 훨씬 크다. 같은 실제
상태에서 bias 표준편차0.2도 명령 차이 중앙값0.66–0.82로 컸다. 새0.01에서는
0.048–0.065였고, 이동 한도의 절반을 넘게 바뀌는 비율은0.5–1.6%였다.
Gaussian/AR1/jaw의 전체 물리 경로를 측정한 것은 아니다.
[16개 같은 방향·실제 decoder 비교](assets/rl_v2_actual_closed_TRAIN_bias_units_20261008.json).

## 설정

`--body-behavior arm20-gentle-rest-greedy`는 TRAIN episode bias의 표준편차만
0.8→0.01, 상한1.6→0.02로 바꾼다. 원래 옵션과 기본값은 유지한다.
20% 선택·90step 점진 진입·기존 Gaussian/AR1 및 jaw 탐색·나머지80% greedy 수집과
학습 가능한 전체 팔 관절 범위는 그대로다. DEV에서는 이 수집 bias를 쓰지 않는다.

원래 TRAIN6,144·DEV128·65배치, 여섯 구역/크기 배분, replay200만을 사용한다.
중간 좌/우small·medium 각16개, 위 좌/우small 각32개를 매 배치 유지한다.
CPU에서 원래 full-arm 초기 actor/normalizer와 동결 기준을 그대로 복원하고
Q·replay·성공 은행은0에서 시작한다. 진단의 학습 행을 가져오지 않는다.
[실제 CPU 초기화·계약 검증](assets/rl_v2_URDF_gentle_full_arm_initial_verified_20261008.json).
관련54개 CPU 테스트가 통과했다.

S63·leju twofinger·body19+jaw2·actor518/critic578·중력 보상18관절·upright torso,
박스/base/배경/firm dynamic flap 무작위화와 reward를 유지한다. 성공은 양손
opposing pad5N·0.25초 유지·8mm 들기, 랙10N·장애물5N·drop10cm·selfOFF다.
새 실행은 읽기 전용 `--policy-servo-diagnostics`를 켜 구역/크기별 명령 제한을 기록한다.

## 실행과 판단

GPU3 전용 고유 실행·동결 소스로 추가한다. 기존 네 장기 실행을 중단하지 않는다.
실제 writer·CUDA_VISIBLE_DEVICES=3·VRAM·시작 계약·기존 Drive 백업 확인은
준비 완료와 구분해 추가 기록한다. 이 파일 작성 시 새 물리 실행은 아직 시작하지 않았다.

자기 초기 전체DEV128과 TRAIN384 단위 후속 DEV를 비교하며 중형과 오른쪽도 따로
확인한다. 초기 무효도 성공률 분모에 포함한다. 독립FINAL은 사용하지 않았고 목표는
미달성이다. 기존 Drive 연결로300초 업로드·크기/MD5 검증·최신2개 보호를 사용한다.
Raw HDF/replay는 로컬에 남고 마지막 로그는 writer 종료 후 검증한다.

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

21:33 KST에 GPU3 전용 고유 실행을 시작했다. writer1,560,881의 소유자·실제run·
CUDA_VISIBLE_DEVICES=3과동결 소스`a1ee62b53fb9fadc6a780ceb7daffdc00fb89e88`의
Python736개를 대조했다. 21:39 KST에 실제 관측518/578·body19+jaw2·작은 bias·
전체URDF 범위·읽기 전용 진단·원래65배치/6,144TRAIN/DEV128·중력 보상18관절·
원래 물리/보상/성공/안전/DR를 확인했다. 실제 초기 입력257개tensor는 유한하며
actor/Q/replay는0으로 전체 초기DEV에 들어갔다. 첫 Drive 백업은21:39:11 KST에
검증됐다. 새 writer는17,561MiB였고 기존 네 실행을 포함한 GPU3는약75.5GB,
여유5.5GB였다. VRAM 증가가 성능 개선을 의미하지 않는다.
[실제 시작·계약·첫 백업](assets/rl_v2_URDF_gentle_full_arm_SAC_actual_startup_20261008.json).

관리 폴더는`GPU3_URDF_gentle_full_arm_long_SAC128_20261008_213310`, 실제run은
`batch_sac_20261008_213311_2b86dc`다. 기존 네 학습은 중단하지 않았다. 22:13 KST에 전체 초기DEV는성공11·안전 위반48·시간 초과56·초기 무효13회,
수치 오류0으로 끝났다. 실제 저장 모델/normalizer75개는 이 실행의 지정 초기 모델과
같고 전체257tensor는 유한했다. 원래 full-arm 초기11과 같은 결과이며 bias가
없는 DEV 기준이다. Actor/Q0인 초기 평가를 학습 개선으로 해석하지 않는다.
[전체 초기 결과·같은 모델](assets/rl_v2_URDF_gentle_full_arm_first_full_DEV_20261008.json).
이후 새 TRAIN 수집을 진행하며 학습 후 성능은 아직 미확인이다. CPU 관찰기는 각 전체 평가와
정확히 같은 저장 모델을 보존하고 초기 모델·전체 결과를 대조한다.

자기 초기 전체DEV128과 TRAIN384 단위 후속 DEV를 비교하며 중형과 오른쪽도 따로
확인한다. 초기 무효도 성공률 분모에 포함한다. 독립FINAL은 사용하지 않았고 목표는
미달성이다. 기존 Drive 연결로300초 업로드·크기/MD5 검증·최신2개 보호를 사용한다.
Raw HDF/replay는 로컬에 남고 마지막 로그는 writer 종료 후 검증한다.


## 첫 실제 TRAIN: 작은 탐색에서 성공3/25경로

22:58 KST metadata 확인에서 첫 TRAIN128은 성공14·안전 위반42·시간 초과72·
초기 무효0회였다. Greedy103조건은성공11·안전 위반29·시간 초과63회,
coherent 탐색25조건은**성공3·안전 위반13·시간 초과9회**였다. 같은 요청의
넓은 탐색 비교 첫 TRAIN은 greedy103/탐색25조건에서 각각성공9/0회였다.
실제 flap 추첨·접촉·표본 이력까지 같은 paired 경로는 아니므로 인과 효과나
학습 후 전체평가 개선을 확정하지 않는다. 모든14성공은 원래 opposing pads·
hold·proof lift·안전 기준을 만족했고 중간 왼쪽small10/상단 오른쪽small4회다.
중형과 나머지 구역은0이며 독립FINAL은 미사용이다.

이어 진행 중인 SAC의 actual actor/Q 갱신과 실제 성공 TRAIN14경로6,304행의
은행 연결을 확인했다. 성공 replay 비율은 초기20%에서 약19%로 감소 중이며
평가 행은0이었다. Live HDF/replay나 은행 payload는 읽지 않았다. 탐색에서
성공 경험을 얻기 시작한 것은 확인했지만 학습 후 자기 전체DEV11회보다
좋아졌는지는 첫 학습 후 전체 평가에서 판단해야 한다.
[128요청·수집 모드별 결과·성공 기준·실제 학습 은행 metadata](assets/rl_v2_gentle_full_arm_first_TRAIN_mode_outcomes_20261008.json).

# 실제 성공 관절 명령을 유지하는 full-arm SAC 보강

2026-10-08 22:13 KST, 전체128평가에서 uniform14→8→4→1, full-arm11→10회다.
박스/base/배경/firm dynamic flap 무작위화와 양손 파지·들기·안전 기준을 유지한다.
중형과 일부 구역은 성공0이며 목표는 미달성이다. 기존 다섯 장기 학습은 계속한다.

## 측정한 문제

Full-arm actor695/Q4,828 모델에 자기 실제 성공 TRAIN16경로/7,556행을 질의했다.
Q와 성공 동작 유지의 몸체 방향이5,014행에서 반대였고 경로 마지막64행에서
유지/Q 경사 크기의 episode 중앙값은0.090이었다. 실제 성공은 중간 왼쪽10·상단
오른쪽6경로이며 아직 성공 데이터가 없는 구역을 해결한 결과는 아니다.
[현재 sampler·실제 명령 Q·같은 성공 은행](assets/rl_v2_actual_full_arm_DEV4_success_Q_retention_20261008.json).

같은 경로의 마지막64행에서 계수0.1·0.3·1·3을 비교했다. 팔의 source local scale을
포함한 실제 몸체 신경망 경사에서 성공 명령 유지 방향의 경로 수는11·13·16·16/16,
유지/Q 경사 비율 중앙값은0.059·0.177·0.590·1.768이었다. 잠재 평균 좌표에서
합성 갱신이 실제 명령 일치 손실을 늘리는 행은442·240·40·0/1,024였다.
**전체 replay의 혼합 batch·Gaussian·jaw·entropy·regularization·실제 optimizer
step/물리 성공을 측정한 것은 아니다.** Q가 틀렸다거나 성공 모방이 무조건
우세해야 한다는 증거로 해석하지 않는다. 측정 후보 중 모든 경로의 신경망 몸체
방향이 유지 쪽이었던 가장 작은 계수1을 별도 비교로 선택했다.
[같은 모델·상태에서의 후보 비교](assets/rl_v2_actual_full_arm_success_servo_coefficients_20261008.json).

실제 모델에서는 환경/learner 할인율0.999가 같고 entropy backup은OFF다.
이 두 설정의 불일치가 현재 회귀 원인은 아니다. Q/actor lr3e-4/1e-6, 보상 스케일1,
entropy와 jaw 계수·sampling은 그대로 유지한다.

## 선택형 변경과 검증

초기화 옵션 `--servo-retention-profile full-arm-tail64-strong-servo`를 추가했다.
실제 성공 TRAIN의 관절 명령 구간 Huber 계수만0.1→1로 바꾼다. 원래 목표 MSE와
바깥 goal weight, jaw 유지 손실, 성공64표본 중 절반의 마지막64행 선택, Q/return
표본·보상·행동 공간·velocity 제한은 그대로다. 같은 모터 명령을 내는 포화 목표
구간을 허용하며 잘못된 방향의 포화에는 되돌아오는 경사를 남긴다.

별도 artifact/validator/학습 dispatch/Q 영상 복원 경로로 구분한다. 기존 profile과
기본0.1 계약은 그대로이며, 서로 다른 유지 목적의 체크포인트 혼용을 거부한다.
관련56개 CPU 검사에서 같은 초기 행동·Gaussian·밀도·entropy·Q target과 실제
강한 유지 update/재개를 확인했다. 실제 CPU 초기화·저장·SAC 재개·Q 영상 복원,
기존 actor/normalizer29·frozen 기준과 모든 원래 입력 보존을 추가 대조했다.
[실제 초기화 검증](assets/rl_v2_URDF_strong_gentle_full_arm_initial_verified_20261008.json).

작은 탐색0.01/0.02·선택20%·90step ramp와 원래 TRAIN6,144/DEV128·65배치·
replay200만을 사용한다. Q·replay·성공/return 은행은0에서 시작하며 진단에 사용한
16개 성공 행·평가 행·과거 Q를 새 초기화에 넣지 않는다. 여섯 구역/크기와 모든
무작위화·S63/twofinger·중력 보상18관절·upright torso·양손5N/0.25초/8mm·
랙10N/장애물5N/drop10cm/selfOFF를 유지하고 curriculum은 없다.

## 실행과 판단

GPU3의 기존 다섯 실행은 유지한다. 사용자가 허용한 다른 여유 GPU에서 비교할 수
있도록 GPU1의 실제 기존51,690MiB 점유 프로세스 하나와29,454MiB 여유를 확인했다.
해당 프로세스는 건드리지 않았다. 새17.6GB 실행을 추가해도약11.9GB 여유가 남는다.
실제 새 실행·초기 전체평가·첫 Drive 백업은 준비와 구분해 추가 기록한다.
현재 이 기록의 보강 정책은 실제 물리 학습을 아직 시작하지 않았다.

자기 전체 초기DEV128과 TRAIN384마다의 전체DEV를 비교하며 중형/오른쪽/상단도
별도로 판단한다. 성능 개선 전에는 독립FINAL을 사용하거나 목표를 완료하지 않는다.
기존 Drive 연결·300초 백업·크기/MD5 검증·최신2개 보호를 사용하고 raw HDF/replay와
현재 로그는 유지한다. 종료 로그는 writer가 멈춘 뒤 업로드·검증한다.

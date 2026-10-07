# 여섯 박스 조합의 SAC 학습 진행

2026-10-08 08:46 KST 기준이다. 기존 small 정책의 평가35→19→23→30/128은
직전보다 회복했지만 초기보다 낮다. Medium 파지가 없다는 전체 목표의
부족한 범위를 유지하고 새 실제 학습을 시작했다.

## 새 학습 설정

| 항목 | 설정과 확인된 범위 |
| --- | --- |
| 로봇 | S63, Lejuclaw twofinger, 기존 중력보상 |
| 배치 | 중간 좌/우 small·medium 각16개, 상단 좌/우 small 각32개, 총128개 |
| 제어 | 별도 base 접근·정지 제어기, 이후 몸통·팔19개 목표와 두 jaw는 SAC |
| 무작위화 | 기존 박스·base 시작 위치·배경·움직이는 firm flap 범위 유지 |
| 학습 | 새 TRAIN3,072조건,384조건마다 같은 전체 DEV128 반복 |
| 성공 | 실제 양손 opposing flap 접촉·0.25초 유지·8mm 지지면 clearance |
| 안전 | 기존 랙10N·주변 장애물5N·reset 대비 낙하10cm, self-collision OFF |
| 최종 평가 | 독립FINAL128은 준비만 했으며 미사용 |
| 장치 | 실제 writer3987840, CUDA_VISIBLE_DEVICES=3, learner cuda:0 |

고유 관리 폴더는`GPU3_supported_size_failed_bootstrap_SAC128_20261008_084131`이다.
기존 두GPU3 학습과GPU0의 새 접근 후보 재확인을 유지했다. 시작 전GPU3 여유는
70,962MiB, root 약162GiB였으며 다른 사용자의 프로세스·파일을 건드리지 않았다.

## 왜 이 경로가 필요한가

기존 크기별 준비는 모든 조합에서 성공 또는 안전한 접근 후보가 먼저 있어야
했다. 실제 발견·새 TRAIN 재확인의 선택한 중간 왼쪽medium 후보는4회 모두
충돌했다. 기존 strict 준비 경로는 그대로 두고, **실패를 실패로 보존하며 새
경험을 수집하는 별도 선택형 초기화**를 추가했다. 성공0·unsafe4 및
`measured_success=false`를 유지한다. 실패 조건을 완화하거나 medium을
학습 범위에서 제외하지 않았다.

전체 정책 복원에서 nested actual-flap anchor가 새 크기별 목표를 원래 regional
출처와 비교해 거부하던 오류를 고쳤다. 검증된 새 계약의 원래 regional 출처로
actor-only anchor만 비교한다. 새 Q/replay의 전체 goal 계약은 크기별 목표를
유지해 old Q와 섞이지 않는다. 모델·정규화·optimizer·anchor와 기존 actor-only
TRAIN 기억12,476행은 보존하며 calibration·DEV·VR 행을 새 Q/reward에 넣지 않는다.

관련86개 테스트와 실제 전체 trainer·첫128개 구역/크기 stage 복원을 통과했다.
Actor·Q·온라인 replay·성공·누적 보상 은행은0으로 시작한다. 초기 기준 전체
DEV128을 먼저 수행하고 이후 실제 TRAIN 경험으로 업데이트한다. **복원 및
프로세스 시작 검사는 물리적 파지 성공의 증거가 아니다.**
[실제 초기화·PID·격리·원래 조건 근거](assets/rl_v2_all6_failed_bootstrap_SAC_start_20261008.json).

소스는 격리 commit`8b061dbee5e2c8fbb2a9a09def81f99fc0133a27`이다. 진행 중인
원래 접근 비교의 같은 소스 검사를 보호하기 위해 별도 worktree에서 실행한다.
Main 통합은 해당 비교의 정상 종료와 원래 소스 검증 후 진행한다. 기존 checkout
수정 파일은 보존한다.

## 함께 확인한 원인과 적용 여부

- 몸체 기울기 제어를 부드럽게 한 진단은 상단 떨림을 줄였지만 전체 파지4/128은
  그대로였고 안전 위반이 늘어 본 학습에는 적용하지 않았다.
- 실제 관측의 박스 종류 정규화만 바꿔 계산한32개 명령은 변화가 없었다.
  이 수정만으로 medium 실패를 해결할 근거가 없어 배포하지 않았다.
- 별도 Q 보강 SAC의 첫 학습 후 전체 평가는22/128로 초기35회보다 낮다.
  일반 SAC의 같은 TRAIN384 뒤19회와의 단일 차이를 확실한 개선으로 보지 않는다.

[접촉·몸체·정규화와 실제 영상](RL_V2_CONTACT_AND_BASE_DIAGNOSIS_20261008.md),
[Q 보강 비교의 실제 전체 평가](RL_V2_Q_SUPPORT_CONSERVATIVE_20261008.md).

## 새 접근 후보 비교

대각선 이동·방향을 포함한8개 후보의 frozen TRAIN16×8 진단도 정상 종료했다.
전체128요청은 성공5·안전 위반29·시간 초과78·초기 무효16회이며 medium 성공은0이다.
유효한 medium26회 중13회는 충돌 없이 시간 초과했다. 모델·normalizer198개와
actor/Q/replay0 고정 검사를 통과했다. 새 seed의 같은 후보 재확인 writer3980667은
GPU0에서 실제 실행 중이다. 후보 반복의 결과를 SAC 성능이나 독립적인 일반화
점수로 해석하지 않는다.
[전체 종료·여섯 조합 결과·고정 검사](assets/rl_v2_diagonal_size_discovery_closed_20261008.json).

## 보관 상태

원래 checkout의 Drive wrapper·supervisor와 기존 인증을 재사용한다. 300초
업로드·크기/MD5 검증·최신2개 및 최신 검증2개 보존 규칙이 같다. 08:43 KST
실제 접속 검사는`invalid_grant`였으므로 업로드 완료를 주장하거나 미검증
파일을 삭제하지 않는다. 새 인증은 만들지 않았다. Checkpoint/log 전용 백업에서
raw replay·HDF·영상은 로컬에 남으며 여유 공간을 확인하면서 보존한다.

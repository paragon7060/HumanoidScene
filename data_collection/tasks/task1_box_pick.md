# Task 1: 박스 꺼내기

## 목표와 범위

Ready → 양손 pregrasp → 접근·파지 → rack 밖 인출 → 필요시 lift/정렬 → 안정적으로 유지.

뒤돌기, 이동, 내려놓기, 버튼 누르기는 제외한다. Pull-first/lift-first 순서는 고정하지 않고 실제 box 전체 형상과 선반·턱의 여유 공간으로 정한다.

초기 검증 대상은 캡처된 중간 랙·눈높이 선반(1-based shelf 2)의 `MediumBox_0`이다. 첫 실행은 이 한 위치에서 랜덤화를 끄고, 이후 검증된 범위만 YAML로 확장한다.

## 단계별 검증

1단계는 현재 robot/hand 조합, 한 환경, 한 box 종류·위치에서 양손 pregrasp까지만 한다. 손을 닫거나 box를 꺼내지 않는다. 계획 성공과 실제 도달 성공을 분리하고, target box 비접촉·settling 이후 비의도 이동도 검사한다.

2단계는 양손 물리 파지, box 전체 형상의 rack 이탈, 파지 유지와 안정화를 모두 확인한다. 중심점 이동만으로 성공 판정하지 않는다. 미끄러짐·flap 움직임을 감시한다. 파지 후 양손은 공통 box 목표와 진행률을 사용하되 접촉 안정성을 별도로 검증한다.

Reset 이후 teleport, 관절 상태 덮어쓰기, fixed joint/grasp assist로 동작이나 파지를 위조하지 않는다.

## 손목 pitch 접근 규칙

양팔 `zarm_l7_joint`/`zarm_r7_joint`는 같은 물리 pitch 목표를 사용한다. 현재
`configs/task1.yaml`의 첫 검증 schedule은 `full_target_rad=0.65`와
`transit_fraction=0.33`이다.

1. Ready 자세에서 양쪽을 약 `0.65×0.33` rad로 먼저 맞춘다.
2. 그 공통 partial pitch를 유지한 채 flap 법선 방향 pregrasp까지 들어간다.
3. pregrasp에서 위치를 고정하고 양쪽을 `0.65` rad로 회전시킨다.
4. 이후 grasp runner는 같은 full-pitch 자세와 각 flap의 실제 법선 방향을 유지한 채
   grasp point로 접근하고 claw를 닫는다.

공통인 것은 관절 pitch 목표이고, 좌우 TCP quaternion은 flap 법선과 손 방향이
서로 반대이므로 별도로 계산한다. 현재 `scripts/task1_pregrasp_smoke.py`는
1–3번과 위치/관절 진단까지만 수행한다. transit에서는 orientation weight `0`과
q7 bounded direct correction을 사용하고, pregrasp staging에서 원래 orientation
weight를 복원한다. claw close·pull·lift와 성공 판정은 아직 수집 runner 범위가 아니다.

현재 실행 상태:

- 실행 진입점: `scripts/task1_pregrasp_smoke.py`
- 산출물: 지정한 output run 아래 `pregrasp_smoke.json`과 3개 RGB PNG
- 판정 범위: pregrasp 위치 추종과 q7 phase 진단만 기록하며, `execution_success`가
  true여도 pick 성공이나 dataset episode 성공으로 승격하지 않는다.
- Kanu GPU4 최신 실행은 `task1_wrist_schedule_smoke_20260908_directq7`에서
  위치 게이트를 통과했지만 contact 센서가 없어 `pregrasp_verified`는 보류한다.
- 실제 grasp·pull·lift와 접촉/인출 검증은 다음 단계로 남아 있다.

## 설정과 실패 처리

`../configs/task1.yaml`에서 box 종류/위치, grasp annotation, 성공 기준, 랜덤화 활성 여부·종류·범위를 관리한다. 네 종류(small/medium/large/xlarge)를 표현하되 최초 검증은 하나로 제한한다. Randomization의 위치/회전 범위에는 단위와 기준 frame을 명시하고, 활성화할 때 누락 범위를 거부한다.

Grasp annotation은 candidate ID, asset/scale, 좌우 reference link, pose, 접근축/frame, pregrasp 거리, 손 열림/닫힘 목표, validation_status를 포함한다. 움직이는 flap은 현재 flap frame을 사용한다. 기하 기반 후보와 물리 검증 완료값을 구분한다.

도달 불가, 충돌, 추종 오차, timeout, 미끄러짐/낙하를 구분해 기록한다. 임계값은 실행 전에 설정하며 실패를 성공으로 바꾸려고 완화하지 않는다.

구체적인 grasp 후보, pull/lift 경로, 물리 성공 임계값은 아직 미확정이다. 현재
GPU smoke는 위치·q7 추종만 입증했으므로 contact/충돌 검증이 연결되기 전까지
`pregrasp_verified`와 `pick_success`는 `NOT_RUN`으로 유지한다.

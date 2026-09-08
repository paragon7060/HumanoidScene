# Task 1: 박스 꺼내기

## 목표와 범위

Ready → 양손 pregrasp → 접근·파지 → rack 밖 인출 → 필요시 lift/정렬 → 안정적으로 유지.

뒤돌기, 이동, 내려놓기, 버튼 누르기는 제외한다. Pull-first/lift-first 순서는 고정하지 않고 실제 box 전체 형상과 선반·턱의 여유 공간으로 정한다.

## 단계별 검증

1단계는 현재 robot/hand 조합, 한 환경, 한 box 종류·위치에서 양손 pregrasp까지만 한다. 손을 닫거나 box를 꺼내지 않는다. 계획 성공과 실제 도달 성공을 분리하고, target box 비접촉·settling 이후 비의도 이동도 검사한다.

2단계는 양손 물리 파지, box 전체 형상의 rack 이탈, 파지 유지와 안정화를 모두 확인한다. 중심점 이동만으로 성공 판정하지 않는다. 미끄러짐·flap 움직임을 감시한다. 파지 후 양손은 공통 box 목표와 진행률을 사용하되 접촉 안정성을 별도로 검증한다.

Reset 이후 teleport, 관절 상태 덮어쓰기, fixed joint/grasp assist로 동작이나 파지를 위조하지 않는다.

## 설정과 실패 처리

`../configs/task1.yaml`에서 box 종류/위치, grasp annotation, 성공 기준, 랜덤화 활성 여부·종류·범위를 관리한다. 네 종류(small/medium/large/xlarge)를 표현하되 최초 검증은 하나로 제한한다. Randomization의 위치/회전 범위에는 단위와 기준 frame을 명시하고, 활성화할 때 누락 범위를 거부한다.

Grasp annotation은 candidate ID, asset/scale, 좌우 reference link, pose, 접근축/frame, pregrasp 거리, 손 열림/닫힘 목표, validation_status를 포함한다. 움직이는 flap은 현재 flap frame을 사용한다. 기하 기반 후보와 물리 검증 완료값을 구분한다.

도달 불가, 충돌, 추종 오차, timeout, 미끄러짐/낙하를 구분해 기록한다. 임계값은 실행 전에 설정하며 실패를 성공으로 바꾸려고 완화하지 않는다.

구체적인 대상 box, grasp 수치, 성공 임계값은 아직 미확정이다. 실행 코드·명령·성공 증거는 아직 없다.

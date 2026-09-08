# Box Pick 데이터 수집

HumanoidScene/Isaac 환경에서 box GT 6D를 이용해 양손 접근 경로를 계획하고, 실제 물리 파지·인출을 검증한 뒤 시연을 저장한다. RoboTwin 2.0은 구조 참고이며 SAPIEN 환경을 이식하지 않는다.

## 구조

- [tasks/task1_box_pick.md](tasks/task1_box_pick.md): 무엇을 할지, 성공·실패 조건.
- [robot/README.md](robot/README.md): 로봇 모델, 좌표계, 충돌 환경, planner와 실행.
- [collection/README.md](collection/README.md): 반복 수집, 저장 항목, 10Hz와 시간 정렬.
- `configs/`: task1 / robot / collection YAML 설정 초안.
- `references/`: 사용자가 제공한 원본 MD 2개. 원문은 수정하지 않는다.

이 폴더는 문서·설정용이다. 구현 코드는 저장소의 `src/kuavo_isaaclab_scene/`, 실행 진입점은 기존 launcher 관례, 테스트는 `tests/`를 따른다. 새 모듈의 실제 위치와 실행 명령은 구현 후 해당 문서에 연결한다.

## 구현 순서와 현황

| 단계 | 완료 기준 | 상태 |
|---|---|---|
| 문서·설정 | 역할별 문서와 미확정 값 구분 | 작성 완료; YAML은 실행용 아님 |
| 1. Pregrasp | 모델/FK 일치, 경로 실제 추종, 양손 도달, 금지 접촉 없음 | CPU 사전 검사 구현·98 tests 통과; planner/world/drive 및 Isaac 검증은 남음 |
| 2. 파지·인출 | 한 종류·한 위치에서 물리 파지, 완전 인출, 안정화 | 후속 |
| 3. 기록 | 3종 action, 관측/GT, 시간·schema·resume 검증 | 후속 |
| 4. 확대 | 네 종류·배치 변화, 조건별 성공률/실패 원인 기록 | 후속 |

첨부 문서의 우선순위는 RoboTwin2 문서 > 최초 BoxPick 문서다. 이후 사용자가 명시적으로 바꾼 결정이 최우선이다. 원본의 구현 지시문은 설계 자료이며, 이 문서 생성이 모든 단계의 실행 승인을 뜻하지 않는다.

## 결정과 미확정 값

확정: Task1은 꺼내 안정적으로 들기까지, HumanoidScene 유지, 10Hz 데이터, 풍부한 관측/GT, action 3종 분리. LeRobot v3를 저장 목표로 두되 실제 writer 연결과 저장 경로는 검증한다.

코드 조사로 해결: 현재 robot/hand, 관절 순서·차원, TCP, asset/scene, planner 호환 버전, control/physics 주기.

실험으로 보정: grasp 후보, 접근 거리, 경로 속도, FK·추종·성공 허용 오차. 검증 전 숫자를 정답으로 고정하지 않는다.

본격 수집 전 YAML에서 설정: 카메라 해상도·목표 에피소드 수·고용량 raw/depth/segmentation 기록은 `configs/collection.yaml`, 배치/랜덤화 범위는 `configs/task1.yaml`. 기존 카메라 기본값은 사용자 확정값이 아니다. 향후 실행기는 이 YAML 값을 사용하며 코드에 수집 조건을 고정하지 않는다.

YAML의 `null`은 미확정이고 `draft: true`는 실행 불가 초안이다. CPU inspect는 초안을 읽되 실행 완료로 취급하지 않는다. Collection/randomization 설정 검증 함수는 구현했으며, 실제 수집 runner 연결은 후속이다. 실행 단계에 필요한 값이 빠졌으면 오류로 중단해야 한다. 실제 데이터 metadata에는 null/TBD 차원을 쓰지 않는다.

## 현재 구현된 범위 (2026-09-08)

`src/kuavo_isaaclab_scene/planning/`에 CPU 관절 이름 매핑, URDF tree FK, body/flap 기준 pregrasp 변환, YAML/산출물 경로 검증, 소스 사전 검사 CLI를 추가했다. 기존 로봇/손/초기 상태 테스트를 포함해 98개 통과했다. CPU FK는 Pinocchio와 3개 자세 × 양팔에서 비교했다. 이 결과는 Isaac 실제 TCP 일치나 충돌 안전성의 증거가 아니다.

실행한 사전 검사: `/home/work/mntvol/data/outputs/pregrasp_source_audit_20260908_01/preflight.json`.
테스트 결과: `/home/work/mntvol/data/outputs/pregrasp_cpu_tests_20260908_03.junit.xml`.
재현 명령과 모델 조사 내용은 [robot/README.md](robot/README.md)에 둔다.

cuRobo는 현재 Isaac 환경에 미설치이며 조사 시 GPU 0–3은 기존 작업이 사용 중이었다. GPU 작업은 시작하지 않았다. 다음은 cuRobo 버전/모델 설정, live USD collider adapter, 관절 drive runner, 실제 pregrasp 검증이다. GPU 사용 범위 확인 후 진행한다.

## 보관 및 참고

Repo에는 코드·설정·문서만 둔다. 생성 데이터·영상·로그·진단 결과는 사용자 지정 루트 `/home/work/mntvol/data/outputs` 아래 `<run_name>/`에 저장한다. `configs/collection.yaml`의 `output_root`와 `run_name`으로 경로를 구성하며, 기존 실행을 덮어쓰지 않는다. GPU와 run 이름은 실행 전 확인한다. commit/push 및 대량 수집은 아직 하지 않았다.

- HumanoidScene 문서 생성 기준 commit: `753e62d364c95785fc95d038133612470ff40c3a`.
- [RoboTwin task 예시](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/lift_pot.py)
- [로봇/planner 계층](https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs/robot)
- [반복 수집 코드](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/collect_data.py)

위 main 링크는 변경될 수 있다. 실제 planner 구현 시 참고 commit과 dependency 버전을 고정해 기록한다.

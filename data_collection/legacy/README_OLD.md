# Box Pick 데이터 수집

HumanoidScene/Isaac 환경에서 box GT 6D를 이용해 양손 접근 경로를 계획하고, 실제 물리 파지·인출을 검증한 뒤 시연을 저장한다. RoboTwin 2.0은 구조 참고이며 SAPIEN 환경을 이식하지 않는다.

## 구조

- [tasks/task1_box_pick.md](tasks/task1_box_pick.md): 무엇을 할지, 성공·실패 조건.
- [robot/README.md](robot/README.md): 로봇 모델, 좌표계, 충돌 환경, planner와 실행.
- [collection/README.md](collection/README.md): 반복 수집, 저장 항목, 10Hz와 시간 정렬.
- `configs/`: task1 / robot / collection YAML 설정 초안.
- `task1_box_pick_web.py`: 기존 텔레옵 preview를 수정하지 않고 Task1 pregrasp를
  현재 브라우저 화면에 실시간 스트리밍하는 전용 실행 파일.
- `references/`: 사용자가 제공한 원본 MD 2개. 원문은 수정하지 않는다.

기존 범용 웹 preview는 그대로 둔다. Task1 전용 동작처럼 데이터 수집에만
필요한 로직은 이 폴더의 실행 파일에서 격리한다. 테스트는 `tests/`를 따른다.

Kanu 예시(브라우저 WebSocket은 기존 포트 8765를 사용):

```bash
cd /home/seonho/HumanoidScene
CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src \
  python data_collection/task1/pose_editor.py \
  --headless --device cuda:0 --bridge-host 127.0.0.1 --bridge-port 8765 \
  --no-camera-preview --no-domain-randomization --robot-model s200062 \
  --stereo-eye-width 848 --stereo-eye-height 480 \
  --head-camera-width 848 --head-camera-height 480 \
  --wrist-camera-width 848 --wrist-camera-height 480 \
  --jpeg-quality 95 --stream-fps 10 --pregrasp \
  --pregrasp-distance-m 0.10 --pregrasp-grasp-depth-m 0.015 \
  --pregrasp-steps 300 --pregrasp-initial-state task1_ready_bent \
  --pregrasp-settle-steps 120
```

`--pregrasp`를 켜면 서버가 먼저 대기하고, 브라우저가 연결된 첫 simulation
step에서 초기 자세·settle·IK 이동을 시작한다. 따라서 연결 전 one-shot으로
움직여 화면에서 놓치는 구간이 없다. 영상/산출물은 repo가 아니라
`/home/work/mntvol/data/outputs` 또는 실행 시 지정한 output lane에 둔다.

### Task1 수동 관절 자세 편집기

이 모드는 정적인 initial pose만 만들고 검사한다. 수집 episode를 기록하지 않으며,
직접 joint state를 적용하므로 여기서 생긴 이동을 expert action으로 사용하지 않는다.
고정 3인칭 카메라에 q1~q7 물리축을 실린더로 표시한다(왼팔 청록, 오른팔 분홍,
마지막 선택 관절 노랑).

```bash
CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src \
  python data_collection/task1/pose_editor.py \
  --headless --device cuda:0 --robot-model s200062 \
  --joint-pose-editor --pregrasp-initial-state task1_ready_bent \
  --bridge-host 127.0.0.1 --bridge-port 8765
python -m http.server 8080 --directory data_collection
```

`http://127.0.0.1:8080/task1_pose_editor.html`에서 전후·좌우 4개 시점을 바꾸고
양팔 q1~q7을 하나씩 조정한다. 화면에는 gripper local `-Z`와 robot-base 하향축
사이의 각도 및 TCP base-frame 위치가 계속 표시된다. `서버 로그에 출력`은 현재
14개 arm 값을 로그로 내보내고, `JSON 복사`는 preset 파일을 수정하지 않고 값을
클립보드에 복사한다.

## 구현 순서와 현황

| 단계 | 완료 기준 | 상태 |
|---|---|---|
| 문서·설정 | 역할별 문서와 미확정 값 구분 | 작성 완료; YAML은 실행용 아님 |
| 1. Pregrasp | 모델/FK 일치, 경로 실제 추종, 양손 도달, 금지 접촉 없음 | CPU 계약·정적 검증 통과; Kanu GPU4 staged smoke에서 위치 게이트 통과; contact 센서 부재로 물리 검증은 보류 |
| 2. 파지·인출 | 한 종류·한 위치에서 물리 파지, 완전 인출, 안정화 | 후속 |
| 3. 기록 | 3종 action, 관측/GT, 시간·schema·resume 검증 | LeRobot v3/AV1 writer smoke 통과; 자동 Task1 runner 연결은 후속 |
| 4. 확대 | 네 종류·배치 변화, 조건별 성공률/실패 원인 기록 | 후속 |

첨부 문서의 우선순위는 RoboTwin2 문서 > 최초 BoxPick 문서다. 이후 사용자가 명시적으로 바꾼 결정이 최우선이다. 원본의 구현 지시문은 설계 자료이며, 이 문서 생성이 모든 단계의 실행 승인을 뜻하지 않는다.

## 결정과 미확정 값

확정: Task1은 꺼내 안정적으로 들기까지, HumanoidScene 유지, 10Hz 데이터, 풍부한 관측/GT, action 3종 분리다. 첫 검증은 중간 랙·눈높이의 `medium` `MediumBox_0`(shelf 2), 랜덤화 없음, IK-first로 진행한다. RGB 정책 카메라는 `3x480x848`이며 목표 feature는 `head_cam_h`, `wrist_cam_l`, `wrist_cam_r`다. 첫 수집 목표는 성공 10개(최대 20회 시도)다. LeRobot v3/AV1 저장을 목표로 두되 실제 writer 연결과 인코더는 검증한다. Kanu의 LeRobot 0.6.0에서 요청 key와 CHW metadata `[3, 480, 848]`, 10 fps, AV1/yuv420p를 1프레임 smoke로 확인했다.

코드 조사로 해결: 현재 robot/hand, 관절 순서·차원, TCP, asset/scene, planner 호환 버전, control/physics 주기.

실험으로 보정: grasp 후보, 접근 거리, 경로 속도, FK·추종·성공 허용 오차. 검증 전 숫자를 정답으로 고정하지 않는다.

본격 수집 전 YAML에서 설정: 카메라 해상도·목표 에피소드 수·고용량 raw/depth/segmentation 기록은 `configs/collection.yaml`, 배치/랜덤화 범위는 `configs/task1.yaml`. 기존 카메라 기본값은 사용자 확정값이 아니다. 향후 실행기는 이 YAML 값을 사용하며 코드에 수집 조건을 고정하지 않는다.

YAML의 `null`은 미확정이고 `draft: true`는 실행 불가 초안이다. CPU inspect는 초안을 읽되 실행 완료로 취급하지 않는다. Collection/randomization 설정 검증 함수와 camera key/layout/video config 선택은 구현했지만, 실제 Task1 grasp·pull·lift 및 10개 성공 episode 수집 runner 연결은 후속이다. 실행 단계에 필요한 값이 빠졌으면 오류로 중단해야 한다. 실제 데이터 metadata에는 null/TBD 차원을 쓰지 않는다.

## 현재 구현된 범위 (2026-09-08)

`src/kuavo_isaaclab_scene/planning/`에 CPU 관절 이름 매핑, URDF tree FK, body/flap 기준 pregrasp 변환, YAML/산출물 경로 검증, 소스 사전 검사 CLI를 추가했다. 전용 IsaacLab 환경에서 전체 테스트는 `387 passed, 1 skipped`(`pxr.Usd` 미설치)였고, CPU FK는 Pinocchio와 3개 자세 × 양팔에서 비교했다. `scripts/task1_pregrasp_smoke.py`는 flap 로컬 면 법선과 월드→로봇 루트 좌표 변환을 사용해 Isaac headless 검증을 수행하고 PNG/JSON을 남긴다. 기존 baseline 실행에서 두 손 위치 오차는 1.9/5.2 cm, box translation은 0.02 mm였지만 보수적인 3 cm 게이트는 통과하지 못했다. 이는 pregrasp 후보 보정 자료이며 grasp·extraction 성공이나 금지 접촉 부재의 증거가 아니다.

현재 smoke에는 `configs/task1.yaml:wrist_pitch` schedule도 연결했다. 양쪽 q7를 먼저
partial pitch(`0.33×0.65` rad)로 맞춘 뒤 pregrasp까지 이동하고, pregrasp에서
full pitch(`0.65` rad)로 staging한다. 각 phase의 목표/실제 q7와 위치 오차는
`pregrasp_smoke.json`에 기록된다. transit phase에서는 위치 IK가 손목 방향과
충돌하지 않도록 orientation weight를 `0`으로 두고, q7에만 bounded direct correction을
적용한다. 이 단계는 아직 claw close·pull·lift를 수행하지 않으므로 데이터셋 성공
episode로 집계하지 않는다.

Kanu 실행은 `OMNI_KIT_ACCEPT_EULA=Y`를 사용해 EULA 질문 없이 수행했다. 최신
GPU4 산출물은 Kanu의
`/home/seonho/outputs/HumanoidScene/task1_wrist_schedule_smoke_20260908_directq7/`
에 그대로 보관한다(로컬로 복사하지 않는다). `pregrasp_smoke.json`에서
`execution_success=true`, 최대 양손 위치 오차 `0.01210 m`, transit q7 오차
`0.01899/0.01143 rad`, full-stage q7 오차 `0.01298/0.02836 rad`, box 이동
`0.000026 m`를 확인했다. gripper는 열림 상태이고 contact 센서가 등록되지 않아
`pregrasp_verified`/pick 성공으로 승격하지 않는다.

실행한 사전 검사: `/home/work/mntvol/data/outputs/pregrasp_source_audit_20260908_01/preflight.json`.
테스트 결과: `/home/work/mntvol/data/outputs/pregrasp_cpu_tests_20260908_03.junit.xml`.
재현 명령과 모델 조사 내용은 [robot/README.md](robot/README.md)에 둔다.

cuRobo는 현재 Isaac 환경에 미설치이며 조사 시 GPU 0–3은 기존 작업이 사용 중이었다. Kanu GPU4에서만 smoke를 실행했다. 다음은 grasp annotation 확정, live USD collider/contact adapter, 관절 drive 기반 grasp·pull·lift runner, 실제 contact 포함 pregrasp 게이트 확정이다.

## 보관 및 참고

Repo에는 코드·설정·문서만 둔다. 기본 출력 루트는 `configs/collection.yaml`의
`/home/work/mntvol/data/outputs`이며, Kanu smoke처럼 해당 mount가 없는 호스트에서는
명시한 `/home/seonho/outputs/HumanoidScene/<run_name>/`에 보관한다. 기존 실행을
덮어쓰지 않고, GPU와 run 이름을 실행 전에 확인한다. 이번 staged wrist 변경은
`72a619c`로 commit/push됐고 대량 수집은 아직 하지 않았다.

- HumanoidScene 문서 생성 기준 commit: `753e62d364c95785fc95d038133612470ff40c3a`.
- [RoboTwin task 예시](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/lift_pot.py)
- [로봇/planner 계층](https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs/robot)
- [반복 수집 코드](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/collect_data.py)

위 main 링크는 변경될 수 있다. 실제 planner 구현 시 참고 commit과 dependency 버전을 고정해 기록한다.

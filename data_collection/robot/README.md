# 로봇·경로 계획·실행

## 책임

현재 관절 상태 + 현재 scene → box/flap 기준 양손 목표 → 충돌 고려 joint trajectory → 관절 drive 실행 → 실제 상태 검증.

## 구현 전 확인

- 실제 실행 설정의 robot/hand, USD/URDF, joint names/order/limits/mimic, wrist와 TCP.
- 고정 base/torso/head 상태, 손 벌림과 충돌 형상, planner-simulator 관절 매핑.
- Isaac/Python/Torch/CUDA/cuRobo 버전과 호환 API. 공유 환경을 임의 업그레이드하지 않는다.
- Physics/control/planner interpolation 주기, actuator, smoothing/clamp 전달 경로.

값은 `../configs/robot.yaml`에 기록한다. 이름만 보고 모델을 교체하거나 arm/hand 차원을 추정하지 않는다.

## 좌표와 충돌

Pose는 `[x,y,z,qw,qx,qy,qz]`, 위치 m. World/robot base/box body/flap/좌우 TCP frame을 실제 link/prim 이름으로 기록한다. Box root/geometry center/COM/flap을 구별하고 scale과 env_origin을 보존한다.

`T_world_grasp = T_world_reference @ T_reference_grasp`.
Planner FK와 Isaac TCP를 여러 유효 관절 자세에서 비교해 위치·회전 오차를 기록한다.

Planning world는 rack 선반·기둥·턱, target/주변 box와 flap, 몸통·손·반대팔을 포함한다. 물리 collider와 대응시키며 rack 전체를 빈 공간 없는 단일 cuboid로 만들지 않는다. 동적 pose는 계획 직전 갱신한다. 충돌 margin/제외 pair/샘플 해상도를 기록하며 충돌 검사를 꺼서 통과시키지 않는다.

## 실행과 통과 기준

초기 구현은 box 6D 목표를 양팔 IK로 joint target으로 바꾸고 Isaac 관절 drive에 전달한다. 이후 cuRobo를 붙이더라도 EE로 변환한 뒤 teleop IK로 다른 경로를 만들지 않는다. 재표본화/clamp/smoothing으로 바뀐 명령은 다시 한계·충돌 검증하고 계획값과 전달값을 분리한다.

양팔 동시 목표 지원은 실제 backend에서 확인한다. 미지원이면 1단계 비접촉 접근에 한해 반대팔 충돌을 포함한 순차 계획을 검토한다. 이를 양팔 협응 파지 지원으로 보고하지 않는다.

Stale plan, 큰 추종 오차, 예상 밖 접촉, timeout은 중단한다. 양손 위치·방향·도달 유지·관절 제한·금지 접촉·target box 이동을 확인한다.

상태는 imports_ok / unit_tests_passed / fk_validation_passed / planning_success / execution_success / pregrasp_verified로 구분한다. 이 단계에는 pick_success가 없다.

기존 Quest/평가 기능을 유지하고 planner를 lazy import한다. CPU 코드 검증과 실제 물리 검증 상태를 아래처럼 분리한다.

## 조사 결과와 구현 현황 (2026-09-08)

- HumanoidScene 기본 선택은 `s200062 + s200062_integrated`. `robots/robot_model.py`, `configs/grippers.json`, `envs/teleop_env.py`에서 확인했다. 실행 환경의 모델 선택 변수는 설정되지 않았다. 다른 실제 로봇과 자동 호환된다는 뜻은 아니다.
- URDF 기준 root는 `base_link`, 팔은 좌우 `zarm_[lr]1_joint`부터 `7_joint`까지 총 14개. Runtime articulation index는 아직 미검증이다. 손 drive는 좌우 f/b bar_1 총 4관절이며 정책 hand command 차원과 별개다.
- 팔 체인의 고정 대상 조상은 knee/leg/waist_pitch/waist_yaw 4관절이다. 현재 실제 자세를 가져와 lock해야 하며 임의 0으로 채우지 않는다.
- S200062 two-finger의 좌우 `zarm_[lr]7_end_effector`는 main과 동일하게 마지막 손목 링크로부터 z=-0.17m인 기존 EEF 프레임을 보존한다. 실제 Task1 TCP는 `configs/grasp_reference_points.json`의 닫힌 양손가락 캘리브레이션으로 계산된 `endeffector_center`이며, 기존 EEF 기준 좌/우 각각 약 `[+0.000184, -/+0.001111, -0.055498] m`의 고정 offset을 적용한다. `rl/tasks/specs.py`의 별도 tool offset을 중복 적용하지 않는다. Task1 웹 확인용 pregrasp는 box 좌우 flap의 상단 grasp 쌍을 먼저 계산하고, 각 점에서 robot-base +Z 방향으로 설정 높이만큼 올린 위치를 사용한다. 좌우 바깥 법선 방향으로 벌리는 점은 pregrasp로 사용하지 않는다.
- `zarm_[lr]7_joint`의 URDF 축은 양쪽 모두 parent `zarm_[lr]6_link`의 +Y이고 범위는 ±0.6981317 rad이다. Task1 pregrasp smoke는 이 축을 월드로 변환해 좌우 공통 pitch schedule을 만들고, IK null-space target과 q7 bounded direct correction에도 같은 목표를 넣는다. transit에서는 position-first로 orientation weight를 0으로 두며, pregrasp staging에서 복원한다. 좌우 TCP orientation은 flap 법선에 맞춰 각각 계산한다.
- 기존 manager 환경은 physics 120Hz / decimation 4 (control 30Hz). 신규 실행기의 주기는 아직 정하지 않았으며 데이터 저장 10Hz와 구별한다.
- 패키지 URDF에는 collision 요소 없는 링크 49개가 있다. 이것이 USD physics collider 부재를 뜻하지는 않는다. URDF만으로 planner 충돌 모델 완성으로 처리하지 않는다.
- ROS 원본 `biped_s200062.urdf`는 `camera_base_joint`에서 존재하지 않는 `camera_base` 링크를 참조한다(정의된 링크는 `head_camera_base`). 전체 tree 검증이 실패해 원본과의 동일성은 UNKNOWN으로 남겼다. 원본 저장소는 수정하지 않았다.
- RwH `RwH/configs/data/stage2_box_convert.yaml`은 `platform_type: 5w`, `eef_type: leju_claw`, `train_hz: 10`을 정의한다. Dataset 차원을 sim drive 차원으로 복사하지 않는다. 이 파일의 해상도도 이번 수집의 확정값으로 가져오지 않았다.

조사 commit: HumanoidScene `753e62d364c95785fc95d038133612470ff40c3a`, RwH `51c17908255cf1e3b9263173a420c344bb922b7e`, ROS `4dfa28baee30059a7e16fd64a8940a0c3a6e66b2`. 미커밋 상태와 URDF/config hash는 preflight JSON에 별도로 기록한다. RwH의 기존 미커밋 변경은 보존했다.

현재 interpreter의 distribution metadata: Torch 2.7.0+cu128, NumPy 1.26.0, pin 2.7.0, isaacsim 5.1.0.0, isaaclab 0.54.2. 마지막 값은 설치 distribution metadata이며 Isaac Lab release/tag와 동일하다고 가정하지 않는다. cuRobo import spec은 없었다. [cuRobo 설치 문서](https://curobo.org/get_started/1_install_instructions.html)를 확인했으며 실제 설치 버전과 Python 3.11/Torch 호환성 검증은 남아 있다.

## CPU 검사 재현

아래 CLI는 Isaac/GPU를 시작하지 않는다. 기존 run 이름을 다시 쓰면 오류로 거부한다. `--run-name`은 새 이름으로 지정한다.

```bash
cd /home/work/workspace/HumanoidScene
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  /home/work/.local/miniconda3/envs/env_isaaclab_232/bin/python \
  -m kuavo_isaaclab_scene.planning.inspect \
  --config-dir data_collection/configs \
  --humanoid-repo /home/work/workspace/HumanoidScene \
  --rwh-repo /home/work/workspace/RwH-Kuavo_V2 \
  --ros-repo /home/work/workspace/kuavo-ros-opensource \
  --upstream-urdf /home/work/workspace/kuavo-ros-opensource/src/kuavo_assets/models/biped_s200062/urdf/biped_s200062.urdf \
  --run-name pregrasp_source_audit_next
```

실제 통과한 테스트 묶음은 `tests/test_box_pregrasp_foundation.py`, `tests/test_robot_model.py`, `tests/test_gripper_config.py`, `tests/test_initial_states.py`다(98 passed). CPU tree FK와 Pinocchio를 양팔 각각 3개 유효 자세에서 비교해 matrix absolute tolerance 1e-10을 통과했다. 테스트 산출물과 임시 파일은 `/home/work/mntvol/data/outputs` 아래에 두고 pytest cache/bytecode 생성을 끄고 실행했다.

구현 파일: `planning/geometry.py`, `planning/robot_model.py`, `planning/config.py`, `planning/inspect.py`. 임의 torso 0 적용, 잘못된 quaternion/scale, 누락/중복 joint, 잘못된 YAML·범위·산출물 경로를 거부한다.

현재 구현된 실행 경로는 `scripts/task1_pregrasp_smoke.py`의 고정 `MediumBox_0`
양팔 IK·관절 drive pregrasp smoke이며, staged q7 pitch와 phase 진단까지 포함한다.
아직 구현하지 않은 것은 cuRobo backend/양팔 방식, live scene collision adapter,
grasp annotation loader, claw close·pull·lift executor 및 validator runner다. 따라서
pregrasp smoke의 위치 추종 결과와 물리적인 `pregrasp_verified`/`pick_success`는
서로 분리해 기록한다.

## IK backend 결정 (2026-09-08)

로봇과 동일한 IK를 기준으로 삼을 때는 `kuavo-ros-opensource`의
`motion_capture_ik`/`kuavo_humanoid_sdk.arm_ik`를 source of truth로 사용한다.
이 경로는 Kuavo의 14-DoF 양팔 모델·버전별 URDF·Drake `plantIK`와 연결되어
있으며, `/arms_ik_node` 및 IK ROS service가 필요하다.

`letools_opensource`는 이 솔버를 대체하지 않는다. LeTools의
`check_ik_accessibility`는 ROS의 `/mobile_manipulator_ik_accessibility_check`를
호출하는 상위 adapter이고, EE world/local 명령도 최종적으로 Kuavo SDK/ROS로
전달한다. 따라서 수집기의 solver로 직접 선택하지 않고, 실기/ROS runtime에서
명령을 보낼 때의 선택적 client layer로 둔다.

현재 Kanu IsaacLab 환경에는 `rospy`, `kuavo_humanoid_sdk`, `pydrake`와 ROS
master가 없어 이 backend를 아직 live simulator에 연결하지 않았다. 검증되지 않은
호출로 현재 pregrasp를 바꾸지 않으며, ROS `motion_capture_ik` version 62
(`s200062`) sidecar 또는 Drake standalone adapter가 준비된 뒤 다음 순서로
전환한다: robot-base pose 입력 → Kuavo IK의 14개 rad 반환 → Isaac FK 잔차/관절
순서 검증 → 실행 backend 활성화. 그 전까지 현재 IsaacLab DLS는 baseline으로
유지하고, SDK IK 결과가 실제로 반환된 경우에만 비교 결과를 기록한다.

단, 모델/scene 자산 자체는 이미 HumanoidScene에 있다. `assets/kuavo_s200062/urdf`
아래의 `biped_s200062.source.urdf`, `urdf/drake/biped_v3_arm.urdf`,
`biped_v3_full.urdf`, `Larm.urdf`, `Rarm.urdf`를 Kuavo ROS 저장소의 같은 경로와
대조했고 SHA-256이 일치했다. 따라서 standalone 경로에서는 ROS/LeTools 없이
이 scene의 Drake URDF를 그대로 읽고 Kuavo `plantIK` 솔버와 Drake runtime만
붙이면 된다. 현재 scene에 없는 것은 `plantIK` 구현/공유 라이브러리와 Drake
runtime이며, 이를 받기 전에는 Kuavo IK를 사용했다고 기록하지 않는다.

Kanu GPU4의 `task1_wrist_schedule_smoke_20260908_directq7` 실행은 최대 위치 오차
`0.01210 m`와 full-stage q7 오차 `0.01298/0.02836 rad`를 기록해 위치 실행
게이트를 통과했다. 이 smoke에는 contact 센서와 claw/pull/lift가 없으므로 물리적
pregrasp 또는 pick 성공으로 해석하지 않는다.

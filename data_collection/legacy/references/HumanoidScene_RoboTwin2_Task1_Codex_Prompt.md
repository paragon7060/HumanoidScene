# HumanoidScene — RoboTwin 2.0 방식의 Task 1 자동 시연 생성

> Codex 구현 지시서 · 2026-09-08
>
> **환경은 HumanoidScene을 유지한다. RoboTwin 2.0의 물체 파지 정보 → task script → motion planner → 물리 실행 → 성공 검증 구조를 참고한다.**
>
> **이번 구현 범위는 1단계: Kuavo–planner 연결 + rack 충돌 환경 + pregrasp 도달 검증이다.**
> 파지·인출·대량 데이터 수집은 후속 단계다. 아래에 전체 목표와 데이터 계약을 남기지만, 한 번에 전부 구현하지 않는다.

## 0. 기존 프롬프트와의 관계

이 문서는 앞서 작성한 `HumanoidScene_Task1_BoxPick_Codex_Prompt.md`의 **궤적 생성 방식과 구현 순서를 수정·구체화**한다.

유지하는 결정:
- 최종 Task 1은 rack에서 box를 꺼내 안정적으로 들고 있는 상태까지다.
- Box GT 6D와 분석에 필요한 simulator 정보는 풍부하게 저장한다.
- `action.ee_target`, `action.arm_joint_target`, `action.hand_target`을 각각 저장한다.
- 모든 vector feature의 각 dim 이름을 `meta/info.json`에 명시한다.
- 좌표계·단위·값의 의미는 `meta/feature_semantics.json`에 기록한다.

수정하는 결정:
- 전체 동작을 단순한 waypoint 보간 + teleop IK로 해결한다고 가정하지 않는다.
- `pull → lift` 순서를 미리 고정하지 않는다. 실제 rack 여유 공간과 접촉 상태에 따라 결정한다.
- Box 본체와 flap의 상대 자세가 변할 수 있으므로 grasp reference link를 명시한다.
- 접근 구간은 RoboTwin식 planner 기반으로 검증한다. 파지 이후의 양팔 협응 제어는 별도 단계로 구현한다.

두 문서가 충돌하면 **이 문서의 범위·제어·검증 지시를 우선**한다.

---

## 1. 목표와 범위

### 최종 목표

```text
Rack 앞 ready pose
→ pregrasp 접근
→ 양손 파지
→ rack 밖으로 box 인출
→ 필요시 lift / 자세 정렬
→ 안정화
→ Task 1 시연 저장
```

뒤돌기, 이동, rail/conveyor에 놓기, 버튼 누르기, 전체 task chaining은 구현 대상이 아니다.

### 이번 1단계에서 반드시 구현할 것

```text
현재 repository / robot contract 조사
→ Kuavo용 planner 모델 설정
→ 실제 scene과 일치하는 planning world 구성
→ box/flap 기반 양손 pregrasp target 생성
→ 충돌을 고려한 접근 경로 계획
→ Isaac의 실제 관절 drive로 실행
→ 도달·추종·충돌·target box 비접촉 검증
```

이번에는 손을 닫아 box를 잡거나 꺼내지 않는다. Task 1 성공 데이터셋을 만들었다고 보고하지 않는다.
실행 로그·진단용 trajectory·영상 저장은 가능하지만, 이를 성공한 pick demonstration으로 분류하면 안 된다.

### 환경 선택

첫 구현은 한 robot/hand 조합, 한 환경, 한 box 종류, 한 rack 위치로 제한한다.
Base·torso·head는 명시한 ready 상태로 유지한다. 팔과 손은 물리 drive를 통해 움직인다.
초기 reset 이후 robot/EE/box pose를 순간이동시키거나 관절 상태를 직접 덮어써 이동을 구현하지 않는다.

Robot/hand는 현재 실행 설정에서 실제 사용 중인 조합을 우선 확인한다. 임의로 다른 모델로 교체하지 않는다.
Real 대상 모델과 현재 sim 모델이 다르면 차이를 문서화한다. 모델의 이름만으로 호환성을 가정하지 않는다.

---

## 2. 먼저 읽을 코드와 공식 참고 자료

### 작업 저장소

- https://github.com/paragon7060/HumanoidScene

우선 확인할 후보 경로다. 현재 checkout에서 이동·변경 여부를 확인하고 실제 구현을 기준으로 판단한다.

```text
src/kuavo_isaaclab_scene/teleop/urdf_arm_ik.py
src/kuavo_isaaclab_scene/teleop/teleop_ik.py
src/kuavo_isaaclab_scene/teleop/collect_quest_teleop.py
src/kuavo_isaaclab_scene/rl/scenes/asset_geometry.py
src/kuavo_isaaclab_scene/rl/mdp/flap_grasp.py
src/kuavo_isaaclab_scene/rl/mdp/commands.py
src/kuavo_isaaclab_scene/rl/tasks/specs.py
src/kuavo_isaaclab_scene/recording/teleop_lerobot_recorder.py
src/kuavo_isaaclab_scene/recording/lerobot_writer_worker.py
docs/QUEST_URDF_IK.md
docs/QUEST_SELF_COLLISION.md
docs/RL_FLAP_PICK.md
```

### RoboTwin 2.0

- Repository: https://github.com/RoboTwin-Platform/RoboTwin
- 양손 파지·상승 task 예시: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/lift_pot.py
- Planner wrapper: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/robot/planner.py
- Robot 실행 계층: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/robot/robot.py
- Task 공통 계층: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py
- 새 embodiment: https://robotwin-platform.github.io/doc/usage/new-embodiment.html
- Task API: https://robotwin-platform.github.io/doc/usage/API.html
- 수집 흐름: https://robotwin-platform.github.io/doc/usage/collect-data.html

### cuRobo

- 공식 문서: https://curobo.org/
- Isaac Sim / multi-arm 예제: https://curobo.org/get_started/2b_isaacsim_examples.html

**참고 방식:** task와 planner를 분리하는 설계를 가져온다. SAPIEN 기반 RoboTwin 전체를 이식하거나 환경을 갈아타지 않는다.
RoboTwin의 로봇별 좌표 보정값, 테이블 위치, 손 방향, collision 설정을 Kuavo에 그대로 복사하지 않는다.
읽은 참고 코드의 commit과 실제 사용하는 라이브러리 버전을 기록한다.

---

## 3. 코드 수정 전 조사 결과

아래를 짧은 문서로 남긴 뒤 구현한다. 이미 코드에서 확인 가능한 것을 사용자에게 다시 묻지 않는다.

| 조사 항목 | 확인할 내용 |
|---|---|
| 현재 checkout | branch, commit, 기존 미커밋 변경 |
| Sim 환경 | Isaac Sim/Lab, Python, CUDA, PyTorch 버전 |
| Robot | 실제 USD/URDF, robot model, hand 구성 |
| 관절 | arm/hand joint names, 순서, 위치·속도 제한, mimic 관계 |
| EE | wrist link, 실제 TCP, wrist→TCP 변환 |
| 고정 부위 | base/torso/head의 현재 자세와 유지 방법 |
| Planner | 설치된 cuRobo 버전, RoboTwin API와의 호환성, 양팔 지원 방식 |
| Scene | rack·box·flap의 pose, scale, 물리 collider, 활성 instance |
| 실행기 | 관절 target 전달 경로, smoothing/clamping, physics/control 주기 |

현재 버전과 맞는 API를 확인한 뒤 dependency를 고정한다. 단순히 최신 cuRobo를 설치하면 된다고 가정하지 않는다.
공유 환경의 패키지를 무분별하게 upgrade/downgrade하지 않는다. 필요시 격리 환경과 설치 지침을 제공한다.

기존 변경을 reset/overwrite하지 않는다. 요청하지 않은 push, 외부 문서 수정, 실제 로봇 실행은 하지 않는다.

---

## 4. Kuavo planner 모델과 좌표계

### 모델 설정

실제 사용하는 robot/hand 조합에 대해 다음을 설정한다.

- URDF와 필요한 mesh 경로, base link, 양손 EE/TCP link.
- Planner active joints와 simulator joint indices의 명시적 mapping.
- 관절 한계, 속도·가속도 제한, backend에서 지원하는 jerk 제한.
- 고정된 torso/head/hand 자세 및 그 자세의 충돌 형상.
- 팔·손·몸통·반대팔을 포함한 collision geometry와 self-collision 설정.
- 사용하지 않는 관절의 lock value와 실제 sim 상태의 일치 검증.

`J_arm=14` 같은 값은 코드에서 확인 후 확정한다. 단순 인덱스 slicing으로 관절 mapping을 추측하지 않는다.
손의 실제 벌림 폭이 바뀌면 collision geometry도 갱신하거나 안전한 보수적 형상을 사용한다.

### 좌표 규약

문서와 로그에 아래 frame을 구체적인 link/prim 이름으로 명시한다.

```text
W: simulator world
R: robot reference / planner base
B: target box body reference
F: selected flap reference
E_L, E_R: left/right TCP
```

`T^A_B`는 B frame의 좌표를 A frame으로 옮기는 변환이다.

```text
T_world_grasp = T_world_reference @ T_reference_grasp
T_robot_grasp = inverse(T_world_robot) @ T_world_grasp
```

Pose 저장 규약은 `[x, y, z, qw, qx, qy, qz]`, 위치는 m로 한다.
내부 API가 `xyzw`나 Euler를 사용하면 경계에서 명시적으로 변환한다.
Joint 단위는 실제 API를 확인해 변환하고 저장 규약을 문서화한다. degree/radian을 혼동하지 않는다.

**Box root, geometry center, COM(질량중심), flap frame을 같은 것으로 취급하지 않는다.**
Center offset과 asset scale을 반영한다. 복수 환경을 나중에 지원할 수 있도록 `env_origin`도 보존한다.

### FK 검증

동일한 q를 사용해 planner FK와 Isaac 실제 TCP pose를 여러 유효 자세에서 비교한다.
좌우 position error와 rotation error를 수치로 기록한다. 초기 한 자세만 맞는 것으로 충분하다고 판단하지 않는다.
검증 임계값은 config에 노출하고 사전에 명시한다. 실패를 감추려고 임계값을 임의로 넓히지 않는다.

---

## 5. Planning world: rack, 주변 box, robot

Sim에 있는 물체가 planner에 자동 반영된다고 가정하지 않는다. Scene에서 planning world를 만드는 adapter를 구현한다.

### 포함할 것

- Rack 선반 판, 기둥, 전면 턱 등 접근에 영향을 주는 부품.
- 활성 target box와 주변 box, 실제 움직이는 flap.
- Robot 몸통·고정 부위·반대팔과 손.
- 접근 범위에 존재하는 다른 구조물.

시각 mesh보다 **실제 physics collider와의 대응**을 우선 확인한다.
Rack 전체를 하나의 꽉 찬 cuboid로 근사해 선반 사이 빈 공간을 없애지 않는다.
불필요하게 세밀한 형상 대신 필요한 여유 공간과 장애물 경계를 보존하는 단순 형상을 사용할 수 있다.

동적인 물체 pose는 계획 직전 갱신한다. Reset 이후 오래된 planning world를 재사용하지 않는다.
충돌 margin·근사 오차·제외한 collider를 기록한다.

### 허용 접촉

1단계의 목표는 **target box와 접촉하지 않는 pregrasp**다. 손가락–box 접촉을 미리 허용할 필요가 없다.
초기 box–rack 지지 접촉과 robot–rack 충돌을 구분한다.
충돌 예외는 pair별로 명시하고 근거를 기록한다. 전체 self-collision/환경 충돌 검사를 꺼서 성공시키지 않는다.

후속 grasp/extraction 단계에서는 phase별 허용 접촉이 필요하다. 이번에는 그 확장 지점만 마련한다.
Planner에서 held object를 attached collision geometry로 표현하는 것과 sim에서 물체를 강제로 붙이는 것은 다르다.
후속 데이터 수집에서도 fixed joint, teleport, grasp assist로 실제 파지를 위조하지 않는다.

### 검증

Planner 형상을 Isaac scene 위에 선택적으로 겹쳐 표시할 수 있게 한다.
명백한 obstacle-intersecting 목표는 실패해야 한다.
충돌 검사가 이산 샘플링이면 해상도와 한계를 기록한다. 연속 충돌 안전성을 확인하지 않고 보장한다고 쓰지 않는다.

---

## 6. Box별 grasp annotation과 pregrasp 생성

대상 type은 대화에서 정한 `small`, `medium`, `large`, `xlarge`다. 현재 asset/layout 정의와 대조한다.
설정 형식은 네 type을 담을 수 있게 만들되, 1차 물리 검증은 한 type만 한다.

### Config에 필요한 정보

```text
box_type / asset identifier
candidate_id
grasp_reference_link_left / right
left_grasp_pose_reference / right_grasp_pose_reference
left_approach_axis / right_approach_axis
left_pregrasp_distance / right_pregrasp_distance
hand_open_target / hand_close_target
validation_status
```

Grasp는 point가 아니라 orientation까지 포함한 pose다. Approach axis를 어느 frame에서 표현하는지도 명시한다.

- Flap이 body에 대해 고정돼 있을 때만 body 기준 고정 grasp 관계를 사용한다.
- Flap이 움직이면 현재 flap pose와 flap-local grasp를 사용한다.
- 어느 실험에서 flap을 lock했는지, 가동 범위가 얼마인지 기록한다. 임의로 lock 상태를 바꾸지 않는다.
- Geometry 기반 후보는 가능하지만 `validated grasp`로 표시하지 않는다.
- Pregrasp 도달 성공은 파지 성공 증거가 아니다.

검증되지 않은 grasp 숫자를 정답처럼 hard-code하지 않는다. 기하 기반 후보·수동 설정·검증 완료값을 구분한다.
같은 type이라도 scale/hand가 달라지면 재검증해야 한다. Type 문자열만으로 동일성을 가정하지 않는다.

---

## 7. 경로 계획과 실제 실행

### 기본 경로

```text
Measured joint state + current planning world
→ candidate pregrasp targets
→ cuRobo joint trajectory
→ execution validation / resampling
→ simulator joint drive targets
→ measured state feedback
```

cuRobo가 만든 joint trajectory를 다시 EE로 바꿔 teleop IK로 재생성하지 않는다.
그렇게 하면 검사한 경로와 실제 명령 경로가 달라질 수 있다.

기존 teleop 코드에서는 joint mapping, actuator 연결, frame 검사 등을 재사용한다.
기존 IK를 사용하는 fallback이 있으면 명시적으로 구분하고 planner 결과와 혼동하지 않는다.

### 양팔

사용하는 backend에서 복수 EE goal과 두 팔을 포함한 충돌 모델을 실제로 지원하는지 확인한다.
가능하면 full robot 모델에서 양팔 목표를 함께 계획한다.

그 구성이 아직 지원되지 않으면 1단계의 비접촉 접근에 한해 순차 계획을 사용할 수 있다.
이때도 움직이지 않는 팔을 장애물로 반영하고 전체 로봇 경로를 검증한다.
두 팔을 각각 독립 계획한 뒤 시간만 맞추는 것으로 inter-arm collision 검증을 대신하지 않는다.
순차 접근 성공을 후속 양팔 파지·협응 인출 지원으로 보고하지 않는다.

### 실행 규칙

- Physics dt, control dt, planner interpolation dt를 명확하게 구분한다.
- Control 주기에 맞춰 trajectory를 재표본화한 뒤 한계·충돌 조건을 다시 검사한다.
- Clamp/smoothing이 target을 변경한다면 요청값과 실제 전달값을 구분해 기록한다.
- 현재 상태가 planner 시작 상태와 너무 다르면 stale plan을 실행하지 않는다.
- 목표 오차가 큰데 시간만 지났다는 이유로 성공 처리하지 않는다.
- 도달 불가능한 target을 조용히 다른 위치로 투영하고 원래 목표에 성공한 것으로 처리하지 않는다.
- 심한 tracking error, 예상치 못한 접촉, timeout 발생 시 실행을 중단하고 이유를 기록한다.

성공 조건은 양손의 위치·방향 허용 오차, 안정적인 도달 유지, 관절 제한 준수, 금지 접촉 없음이다.
초기 physics settling 이후 기준으로 target box의 이동과 의도하지 않은 접촉도 검사한다.
단순히 planner가 `success=True`를 반환한 것은 **planning success**일 뿐 **execution success**가 아니다.

---

## 8. 1단계 산출물과 CLI

현재 repository 관례에 맞게 이름을 조정하되 다음 책임을 분리한다.

```text
Robot planner configuration
Scene → planning-world adapter
Box/flap grasp annotation loader
Planner interface
Joint-trajectory executor
Headless pregrasp validation runner
Validation report / tests / usage documentation
```

예시 파일 구조이며 그대로 만들 의무는 없다.

```text
configs/planning/kuavo_<model_hand>.yml
configs/grasp/box_grasps.json
src/kuavo_isaaclab_scene/planning/{robot_model,world,planner,executor}.py
scripts/validate_box_pregrasp.py
docs/BOX_PICK_PLANNING.md
```

CLI는 최소한 아래 설정을 명시할 수 있어야 한다.

```text
robot-model / hand
scene-config
target-box
grasp-config
seed
headless / device
output directory
debug collision visualization
```

문서에는 **실제로 구현·검증한 CLI의 정확한 smoke-test 명령**을 적는다.
존재하지 않는 launcher를 이미 사용할 수 있는 것처럼 제시하지 않는다.
Import 시 planner 의존성이 기존 Quest 수집·GR00T 평가까지 강제되지 않도록 lazy import / 명확한 오류를 사용한다.

---

## 9. 후속 단계의 데이터 계약 — 유지할 요구사항

이 절은 이번 단계에서 전체 recorder를 구현하라는 뜻이 아니다.
Planner/executor 로그와 후속 recorder를 연결할 때 정보가 유실되지 않도록 지금 인터페이스를 설계한다.

### 세 가지 action을 각각 저장

| Feature | Shape | 의미 |
|---|---|---|
| `action.ee_target` | `[14]` | 해당 control step의 양손 TCP 실행 목표 pose |
| `action.arm_joint_target` | `[J_arm]` | 같은 step에 실제 전달한 arm joint target |
| `action.hand_target` | `[G_cmd]` | 같은 step에 실제 전달한 hand/gripper command |

`action.ee_target`은 최종 waypoint를 모든 frame에 반복 저장하는 항목이 아니다.
Joint trajectory 구간에서는 **실제 전달한 joint target의 FK로 계산한 TCP 목표**로 정의하고 출처를 기록할 수 있다.
고정 관절·hand 상태·TCP offset도 FK 계산과 일치해야 한다.
다음 정보는 별도로 보존한다.

```text
privileged.segment_goal_ee_pose        # 구간 최종 목표
privileged.requested_ee_pose           # 존재하는 경우: IK/필터 전 요청 pose
control.planned_arm_joint_target       # 실제 전달 target과 달라질 수 있는 계획값
control.target_modified                # 변경 여부
```

EE-space 제어와 joint-space 제어의 생성 방식·오차를 metadata에 남긴다.
실제 관측 관절값을 command label로 대체하지 않는다.
이 세 feature를 저장한다고 모든 policy loader가 자동 지원하는 것은 아니다.
학습용 vector 조립은 후속 adapter에서 명시적으로 처리하고 저장 원본을 익명 vector 하나로 합치지 않는다.

### Observation / GT

```text
observation.images.head / left_wrist / right_wrist
observation.joint_position / joint_velocity
observation.hand_state
observation.ee_pose
observation.robot_root_pose_w

privileged.box_pose_w / box_pose_robot
privileged.box_velocity_w
privileged.target_box_id
privileged.target_box_pose_w / target_box_pose_robot
privileged.left_grasp_pose_box / right_grasp_pose_box   # body 기준일 때
privileged.flap_pose_w / flap_joint_position           # flap 구성에 맞춰 추가
privileged.phase
privileged.finger_contact_force
privileged.is_grasped
privileged.target_box_delta_pose
privileged.success
```

Grasp reference가 flap이면 feature 이름·semantics에 실제 frame을 반영한다.
Hand state dimension `G_state`와 command dimension `G_cmd`는 다를 수 있다. 같다고 가정하지 않는다.
Velocity의 표현 frame과 기준점(root/COM), EE의 TCP definition을 명시한다.
센서가 없으면 0을 실제 측정값처럼 채우지 말고 validity/missingness를 표시한다.

모든 활성 box를 저장할 경우 instance 순서를 고정하고 episode별 ID mapping을 남긴다.
Box 수가 변하면 fixed-capacity padding + valid mask 또는 별도 schema를 사용한다.
새 episode에서 dim의 의미가 조용히 바뀌면 안 된다.

### 시간 정렬

```text
o_t = 명령 적용 전 관측
u_t = [t, t + control_dt) 구간에 전달할 실제 명령
physics execution
o_(t+1), transition outcome
```

Camera/state/command의 실제 timestamp와 갱신 주기를 보존한다.
오래된 영상 frame을 새 영상인 것처럼 표시하지 않는다.
Control decimation이 있다면 raw control 기록과 학습용 resampling을 구분한다.
Terminal success는 어떤 transition 이후의 결과인지 정의한다.

### info.json / semantics

LeRobot writer가 표준 `meta/info.json`을 생성하도록 한다. 임의로 불완전한 info 파일을 쓰지 않는다.
Vector feature의 `dtype`, 정수 `shape`, 실제 dim 순서의 `names`를 모두 제공한다.
Flat vector에서는 `len(names) == shape[0]`를 검사한다.
Joint/channel names는 실제 모델/API에서 가져온다. 생산용 metadata에 `G`, `TBD`, `...`를 남기지 않는다.

`meta/feature_semantics.json`에는 다음을 넣는다.

```text
schema version
feature description / source / policy visibility
reference frame / TCP / link identity
units / quaternion order / absolute-vs-delta
value ranges / hand open-close meaning
phase enum mapping
sampling and timestamp semantics
robot / asset / planner configuration identity
```

GT는 저장 여부와 policy input 여부를 분리한다. `privileged.*` 접두사만으로 leakage가 방지되는 것은 아니므로
학습 loader는 명시적인 feature allowlist를 사용하게 한다.

Resume에서는 dtype/shape뿐 아니라 dim names/order, units, frames, robot/hand contract, schema version도 비교한다.
기존 LeRobot `observation.state` 같은 feature를 이름만 바꾸거나 의미를 조용히 덮어쓰지 않는다.
LeRobot 버전별 표준 metadata 파일은 writer에 맡기고, episode custom metadata는 기존 sidecar를 확장한다.

---

## 10. 후속 구현 순서 — 이번 작업 범위 밖

| 단계 | 목표 | 통과 조건 |
|---|---|---|
| **1 — 이번** | Planner 연결 + rack world + pregrasp | FK 일치, 실제 도달, 충돌/접촉 검증 |
| 2 | 한 box·한 위치의 grasp/extraction | 물리적인 양손 파지·완전 인출·안정화 |
| 3 | Rich dataset recorder 연결 | 3종 action, GT, dim names, 시간 정렬 검증 |
| 4 | 4종 box / 위치 변화 확대 | 조건별 생성 성공률·실패 원인 기록 |

파지 후에는 box 목표 궤적 하나와 양손 공통 진행률을 사용한다.
안정적으로 파지한 시점의 box–TCP 관계를 저장한 뒤 양손 목표를 함께 생성한다.
이는 파지 관계가 거의 고정된다는 근사이므로 flap 움직임·미끄러짐을 감시해야 한다.

Rack 인출은 box 중심 이동 거리만으로 성공 판정하지 않는다.
Box 전체 형상이 선반·턱·기둥을 벗어났는지와 파지 유지·안정화를 확인한다.
Lift-first/pull-first/mixed 경로는 geometry로 결정한다.

대량 수집에서는 모든 시도의 결과를 기록하고, 성공 시연 선택과 benchmark 평가 seed 선택을 분리한다.
성공한 expert seed에서만 policy를 평가하지 않는다.

---

## 11. 테스트와 완료 기준

### Simulator 없이 가능한 테스트

- Pose composition / inverse, quaternion 순서 변환, 비영점 root-center offset.
- Joint-name/index mapping, 누락/중복 joint 거부, 단위 변환.
- Annotation reference와 pregrasp offset 계산.
- Frame/shape/name metadata consistency와 incompatible resume 거부.
- Planner/executor 상태 전이와 오류 전파.

### GPU / Isaac 실행이 필요한 테스트

- 여러 유효 q에서 planner FK와 Isaac TCP 일치.
- 빈 공간 경로 실제 추종.
- Rack collider와 planner geometry 정렬.
- 알려진 충돌 목표 거부 및 반대팔 간섭 처리.
- 선택 box의 양손 pregrasp 실제 도달.
- 도달 실패·timeout·target 수정이 성공으로 둔갑하지 않는지 검사.
- Target box 비접촉, settling 이후 비의도 이동 검사.

### 단계별 상태를 분리해서 보고

```text
imports_ok
unit_tests_passed
fk_validation_passed
planning_success
execution_success
pregrasp_verified
```

`pick_success`는 이번 범위에 없다.
Sim/GPU/dependency가 없어 실행하지 못했다면 mock 결과를 물리 검증처럼 보고하지 않는다.
가능한 구현·단위 테스트까지 완료하고, 미검증 항목과 재현 명령·필요 환경을 명확히 남긴다.
오류를 숨기는 fallback이나 충돌 검사 비활성화로 완료 처리하지 않는다.

---

## 12. Codex 최종 보고 형식

1. 실제로 확인한 robot/hand, dependency versions, 참고 code commits.
2. 변경 파일과 재사용한 기존 기능.
3. Planner joint mapping, base/TCP/world/box/flap 좌표 규약.
4. Planning world에 포함·제외한 geometry와 collision 검사 한계.
5. 양팔 처리 방식: 동시 계획인지, 제한된 순차 접근인지.
6. 실행한 테스트와 실제 측정값: FK/추종/최종 pose 오차, 충돌·접촉 상태.
7. 정확한 설치·smoke-test·진단 로그 확인 명령.
8. 이번에 하지 않은 것과 후속 파지·인출 단계에 남은 작업.

### 마지막 작업 지시

**현재 repo를 먼저 조사하고, 필요한 기반이 있으면 재사용해서 1단계까지만 구현하라.**
**RoboTwin 방식을 참고하되 Kuavo 모델·Isaac physics·rack geometry에 맞춰 검증하라.**
**IK 도달 가능성, 경로 계획 성공, 실제 물리 실행 성공, 파지 성공을 서로 다른 것으로 취급하라.**

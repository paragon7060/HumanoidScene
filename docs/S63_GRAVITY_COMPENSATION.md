# S63 + Leju twofinger 중력 보상과 PD 설정

기본 실행 모델은 S63 + `leju-twofinger`다. 명시적으로 지정하려면:

Leju claw의 각 손 패키지는 D405를 포함해 1 kg이다. 링크별 질량 비율과 CoM을
유지하며 질량과 관성을 함께 조정했다. URDF·USD 및 runtime spawn 모두 같은 값을 쓴다.

```bash
./run_scene.sh --robot-model s63 --gripper leju-twofinger
./quest_collector.sh collect --robot-model s63 --gripper leju-twofinger --rl-reward-debug 1
```

## 공통 보상 경로

`envs/scene_physics.py`에서 모든 Kuavo 모델의 `ArticulationCfg.class_type`을 공통
`GravityCompensatedArticulation`으로 선택한다. standalone scene, manager environment,
RL 학습·reward debug·평가, Quest collect가 이 asset 설정을 공유한다.

매 물리 스텝의 명령 쓰기 때 PhysX의 `get_gravity_compensation_forces()`로 현재 로봇
전체 자세의 중력 토크를 구한다. 포함된 claw/camera의 질량도 로봇 동역학에 들어간다.
그중 knee, leg, waist_pitch, waist_yaw와 양팔 각각 7개 관절에 보상을 적용한다.
wheel, head, gripper motor 및 passive four-bar 관절에는 별도 보상을 적용하지 않는다.
arms-only 등으로 물리 관절 범위가 잠긴 관절도 제외한다.

S63의 기본 `--dynamics-profile auto`는 `s63-body-id`로 해석된다. 몸통 네 관절과 양팔
모두에서 제어 tick마다 계산한 `M(q)qdd_des+C(q,dq)+G(q)`가 단순 `G(q)`를 대체한다.
물리 substep 사이에는 계산값을 재사용한다. 적용 대상은 `wbc_acceleration_profile()`이
게인을 정의한 관절과 정확히 일치하므로, 게인만 있고 적용되지 않는 관절은 생기지 않는다.

몸통을 포함하는 이유는 감쇠 때문이다. 고정 joint PD(`stiffness=400, damping=40`)에서
ready 자세 기준 축 관성은 `knee_joint` 28.2, `leg_joint` 19.2, `waist_pitch_joint`
4.23 kg·m²이고, 감쇠비는 각각 0.19 / 0.23 / 0.49다. 랙에 부딪히거나 박스를 집어
하중이 바뀌면 `knee_joint`가 0.6 Hz로 5초 넘게 흔들린다. 가속도 task를 더하면 유효
게인이 `Kp+I*30`, `Kd+I*6.2`가 되어 감쇠비가 0.57~0.70, 정착 시간이 약 1초로 줄고,
팔 자세나 파지 하중이 바뀌어도 감쇠비가 일정하게 유지된다.

`--dynamics-profile s63-arm-id`는 몸통을 gravity-PD로 되돌린 비교용 profile이고,
`--dynamics-profile gravity`를 지정하면 모든 선택 관절이 아래 기존 식을 사용한다.
가속도 상한은 기존 값을 유지하며, 토크는 URDF가 정의한 구동 effort 한계로 제한된다.

현재 구동은 PhysX implicit force drive이므로 중력 토크를 별도 외력으로 더하지 않고,
solver에 전달하는 목표에만 `g(q)/Kp`를 더한다:

```text
논리 목표: q_cmd                    ← 사용자/RL 목표, 보상으로 변경하지 않음
solver 목표: q_cmd + g(q)/Kp
drive 토크: Kp*(q_cmd-q) + Kd*(v_cmd-v) + g(q)
```

PD와 중력 보상이 같은 drive 안에서 기존 force limit에 제한된다. 사용자/RL 목표,
IK 기준, 기록·관측의 `joint_pos_target`, action manager의 목표에는 보정이 누적되지 않는다.
solver 목표는 물리 관절 범위 밖에 있을 수 있지만 실제 관절 범위와 solver 제약은 유지된다.
이를 미리 관절 범위에 clamp하면 관절 경계에서 중력 보상이 사라지므로 clamp하지 않는다.
일반 teleop 팔 제어에 있던 중력 position bias는 공통 보상이 켜진 asset에서 0으로 만든다.
S200062와 S56에도 같은 경로를 적용한다. S56은 다리 12관절·waist_yaw·양팔 14관절을
선택한다. 모델별 검증 및 추가 확인 사항은 [공통 중력 보상 검토](ROBOT_GRAVITY_COMPENSATION.md).

PhysX inverse dynamics도 접촉·마찰·별도 물체 payload를 포함하지 않는다. 손에 잡힌 박스의 무게가
자동으로 로봇 중력 모델에 합쳐지는 것은 아니다. 모델 오차, 외력, 토크 한도 때문에
자세 오차가 남을 수 있다. 현재 팔 profile은 fixed-root computed-torque 근사이며 실제 S63의
task/contact constraint를 푸는 full QP WBC 구현은 아니다.
[NVIDIA inverse dynamics 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/extensions/runtime/source/omni.physics.tensors/docs/inverse_dynamics.html).

## PD 설정

`configs/s63_servo.json`이 S63의 공통 simulation PD 설정이다.
현재 값은 기존 시뮬레이션 gain을 유지한 초기값이며 실제 모터 단위 변환이 검증된 값은 아니다.
이 모델은 reward debug/Quest env에서 8000/200 등의 추가 PD 보강을 덮어쓰지 않는다.

| actuator | stiffness (N·m/rad) | damping (N·m·s/rad) |
|---|---:|---:|
| height_axis: knee·leg·waist_pitch | 400 | 40 |
| arms: 양팔 14개 | 220 | 22 |
| upper_body: waist_yaw·head | 120 | 15 |

관절별 값이 확인되면 scalar 대신 joint regex별 dictionary를 사용할 수 있다:

```json
"arms": {
  "stiffness": {"zarm_[lr][1-4]_joint": 220.0, "zarm_[lr][5-7]_joint": 120.0},
  "damping": {"zarm_[lr][1-4]_joint": 22.0, "zarm_[lr][5-7]_joint": 15.0}
}
```

위 숫자는 형식 예시다. 모든 해당 관절을 포함하는 regex를 지정한다.
별도 파일은 `KUAVO_S63_SERVO_CONFIG=/absolute/path/servo.json`으로 선택한다.
파일 변경은 새 실행부터 적용되며, RL 학습·디버깅·수집·평가에서 같은 파일을 사용한다.
`--arm-stiffness`, `--arm-damping`은 일반 Quest collect에서 명시적으로 지정한 경우만
공통 팔 profile을 덮어쓴다. RL reward debug는 기존처럼 이 collector 옵션을 사용하지 않는다.

## 초기 자세와 진단

기본 flap pick 실험은 S63용 `s63_leju_ready_01`을 사용한다. 이 자세는 시뮬레이션용으로
준비한 자세이며 실물/VR 측정값이 아니다. base는 기존 S200062 `quest_ready_02`의
저장 root pose를 사용하며, torso는 S63 장착 offset을 보정해 같은 위치·방향을
재현한다. S63 팔·머리 초기값과 원본 S200062 프리셋은 유지한다.
[재현 기준과 관절값](RL_INITIAL_STATES.md#s200062-basetorso-재현)을 참고한다.

실행 중 robot asset에서 아래 tensor로 보상을 확인할 수 있다:

```python
robot = env.scene["robot"]
robot.gravity_compensation_torque   # 선택된 관절의 계산 중력 토크
robot.gravity_compensation_bias     # solver-only g/Kp 보정
robot.inverse_dynamics_torque       # s63-arm-id의 M*qdd+C+G
robot.dynamics_profile              # s63-arm-id 또는 gravity
robot.data.joint_pos_target         # 보정되지 않은 논리 목표
```

`computed_torque`와 `applied_torque`는 implicit actuator의 근사 추정치이며 실제 출력 토크
측정값이 아니다. 실물 PD·제어 계층 확인 결과는 [로봇 제어 확인 기록](REAL_ROBOT_CONTROL_AUDIT.md).

## 시뮬레이션 검증

### 현재 1 kg claw 검증 (2026-09-15)

PhysX의 실제 rigid-body mass를 조회해 각 claw의 14개 링크 합계가 양손 모두
1.00000012 kg(float32 오차)임을 2개 병렬 환경에서 확인했다.
`s63_leju_ready_01`, physics 120 Hz/control 30 Hz, 180 무입력 control steps(6초),
CPU simulation에서 논리 목표가 유지되고 모든 관절 상태가 유한함을 검사했다.

| 모드 | 몸통 최대 오차(rad) | 왼팔 최대 오차(rad) | 오른팔 최대 오차(rad) |
|---|---:|---:|---:|
| whole-body | 0.007590 | 0.002036 | 0.002077 |
| right-arm only | 0.0001002 | 0.0001009 | 0.001439 |

right-arm only에서 잠긴 몸통과 왼팔의 보상 토크는 0이다.
Quest 양팔 absolute/downward/responsive IK 경로에 합성 컨트롤러 pose를 넣은
90 control steps도 관절 범위 초과와 이중 중력 보상 없이 완료했다.
manager/evaluation, Quest collect, RL reward debug의 공통 asset class 선택을 확인했다.
실제 Quest 렌더링·장치 입력이나 실물 로봇의 검증 결과는 아니다.

### 이전 0.740 kg claw 비교 기록

아래 표는 질량을 1 kg으로 변경하기 전 0.740 kg claw에서의 검증 기록이다.
S63 + Leju, 동일한 초기 자세와 위 PD 값에서 무입력 180 제어 스텝(6초)을 비교했다.
RL whole-body, physics 120 Hz/control 30 Hz, CPU simulation에서의 최대 목표 각도 오차다.

| 보상 | 몸통 | 왼팔 | 오른팔 |
|---|---:|---:|---:|
| OFF | 0.5000 rad (28.65°) | 0.02877 rad (1.65°) | 0.02876 rad (1.65°) |
| ON | 0.00754 rad (0.43°) | 0.00196 rad (0.11°) | 0.00200 rad (0.11°) |

오른팔 전용 모드에서도 2개 병렬 환경, 180 스텝을 확인했다. 잠긴 몸통/왼팔의 보상은
0이고, 오른팔 최대 오차는 0.00138 rad였다. 논리 목표가 반복 쓰기로 변하지 않는 것도 확인했다.
Quest RL 제어 어댑터에 합성 컨트롤러 pose를 넣은 90 스텝은 URDF 관절 범위 초과 없이
완료했다. 실제 Quest 장치와 실물 로봇의 동작 검증을 대신하지 않는다.

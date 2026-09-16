# Real-to-sim / sim-to-real 검토와 실물 보정 계획

검토 기준: 2026-09-15 checkout의 공통 gravity writer, RL action/observation/config,
기존 SSH 제어 확인 기록 및 `real_robot_calibration/s63_gripper/reports/session_02.md`.
이번 검토에서 실물 동작 명령, 서비스 호출 또는 모터 설정 변경은 하지 않았다.
몸통/팔의 실측 출력 토크·응답 시험은 아직 없다. 모델별 실측값을 다른 모델에 그대로 이식하지 않는다.

## 먼저 맞춰야 하는 구조

| 우선순위 | 현재 차이/문제 | 전이 영향 | 필요한 변경/확인 |
|---|---|---|---|
| P0 | 실물 WBC 가속도/QP + motor loop, sim implicit torque PD+g | 같은 q 목표와 숫자 gain이어도 동적 응답이 다름 | 배포 인터페이스를 q 목표→기존 WBC로 정의하고 동일 경로의 폐루프 응답을 sim에서 식별. 또는 동일 WBC를 sim에 사용 |
| P0 | actor가 정확한 box/flap pose, velocity, finger force, raw/held grasp flag, contact flap index를 받음 | 실물에서 관측 벡터를 그대로 만들 수 없음 | 각 observation에 센서/추정기/단위/주기/age를 대응. 불가능한 값은 critic 전용 또는 학습부터 센서 기반 추정치로 변경 |
| P0 | base는 `write_root_pose_to_sim()`으로 이동; arms-only는 실제 limit를 ±1e-4로 잠금 | wheel slip, base 관성, 몸통 compliance/반력, S56 균형이 사라짐 | stationary arm 실험과 mobile 실험을 구분. 실물 이동 제어기를 반영하고 배포 조건과 body hold 방식을 맞춤 |
| P0 | `collision_constraints_enabled=False`가 현재 flap config의 기본 | 장애물 접촉을 피하는 정책으로 검증된 상태가 아님 | 충돌 조건을 포함한 재평가 및 실물 WBC/기존 충돌 제한 확인. sim reward predicate를 실물 안전 제어기로 취급하지 않음 |
| P1 | actuator class/gravity 모드, PD, mass/inertia, physics dt 등이 checkpoint 호환 hash에 포함되지 않음 | 물리 구성이 바뀌어도 기존 checkpoint 검사가 통과할 수 있음 | actuator/compensation revision, asset hash, servo 값/파일 hash, control/physics Hz 및 센서 정의를 manifest/호환성 검사에 추가 |

현재 `FlapPickPolicyCfg.enable_corruption=False`이므로 해당 policy 입력은 noise/delay를
모델링하지 않는다. generic policy의 ±0.003 rad noise가 flap policy에도 적용된다고
가정하면 안 된다. `joint_pos_rel/joint_vel_rel`은 robot joint 배열 전체를 사용하므로
passive finger bar 각도까지 실물에서 encoder로 측정할 수 있는지도 확인해야 한다.
측정할 수 없는 passive state는 보정한 기구 FK로 추정하거나 actor 입력 구성을 바꾼다.

### 실물에 보내는 명령

현재 sim의 `g/Kp`는 **solver 내부 보정**이다. 실물에 이 보정된 q를 보내지 않는다.
Kuavo WBC가 이미 중력을 포함한 토크를 계산하므로, 같은 중력 토크를 다시 더하면 중복된다.
논리 q 목표와 WBC가 기대하는 v 목표만 기존 제어 경로에 넘기는 방식을 우선 검토한다.
`/joint_cmd.tau`를 실제 토크 측정값 또는 추가로 넣어야 하는 gravity torque로 해석하지 않는다.
실제 사용하는 SDK/WBC 서비스의 제어권·명령 의미부터 확정한다.

`JointDeltaTargets`는 policy step마다 `q_target += action*scale`을 수행한다.
예를 들어 0.02 rad/action을 30 Hz로 적분하면 최대 0.6 rad/s다. 500 Hz callback에서
같은 action을 매번 적분하면 10 rad/s가 된다. policy 적분은 30 Hz에서 한 번만 수행하고
500 Hz loop에는 이미 계산된 목표를 유지/보간한다. 500 Hz는 이전 실물 설정값이며
실측 주기는 따로 기록한다. filter, v 목표, stale-command 처리와 재개 시 integrator 상태도 맞춘다.

## 실물 테스트로 보정할 수 있는 항목

| 항목 | 필요한 실험/데이터 | sim에 반영할 값 | 식별 한계 |
|---|---|---|---|
| joint zero/축 부호/순서 | 여러 자세에서 encoder와 외부 TCP/링크 pose 비교 | encoder offset, URDF axis/transform, joint map | TCP offset 하나로 모든 관절 오차를 대신 보정할 수 없음 |
| TCP·hand-eye·camera | 고정 기준점 및 여러 손목 자세, 실제 패드 접촉점, camera timestamp | tool/mount SE(3), intrinsics/distortion/depth scale, observation delay | FK로 생성한 pose끼리 비교하면 URDF 오차를 검증하지 못함 |
| 중력 질량/COM/payload | 여러 정적 자세 + 양방향 접근 + 교정된 joint effort 또는 known-load 차분 | mass/first moment, nominal payload, gravity residual | 정적 로그로 inertia/armature는 식별 불가; 모든 per-link mass/COM이 각각 유일하게 식별되는 것도 아님 |
| 폐루프 servo 응답 | 기존 WBC/gain을 유지한 작은 step, ramp, 완만한 sine, 서로 다른 자세 | equivalent bandwidth, damping, steady error, delay/filter | q_cmd/q만으로 물리 motor Kp/Kd·마찰·관성을 각각 분리할 수 없음 |
| 마찰/정지 오차/방향성 | 동일 경로의 저속 정·역방향 및 속도별 effort 비교 | Coulomb/viscous/stiction 또는 방향 의존 acceptance/hysteresis | 단일 정지 오차를 항상 gravity bias로 보정하지 않음 |
| inertia/armature/coupling | free-space의 여러 속도·가속도, single-joint 이후 multi-joint, calibrated effort | inertia, reflected rotor inertia, coupling response | 상호 결합/모델 구조가 틀리면 과도한 gain으로 맞춰도 다른 자세에서 틀림 |
| latency/jitter/filter | command 생성→WBC 수신→motor/state timestamp, 실제 movement onset | command hold/delay, observation age, filter state | 네트워크 receipt 시각, timestamp alignment, 순수 actuator delay를 혼동하거나 중복 적용하지 않음 |
| limit/derating | 문서·설정·지원되는 motor feedback 및 운용 범위 로그 | q/v/a, torque-current-speed envelope, continuous limit, torque slew, temperature derating | peak torque 시험으로 지속 한도를 정하지 않음. config max_current와 Nm를 직접 비교하지 않음 |
| gripper 기구/응답 | % sweep 양방향, 좌우 pad 내면 폭, 속도/effort별 반복 | nonlinear %→jaw/angle LUT, coupling, stop tolerance, delay/speed | max opening 한 점으로 전체 기구 전달을 선형 가정하지 않음 |
| contact/box/conveyor | 같은 box/flap/pad, known mass, force gauge, slip/lift/placement, belt/roller 속도 | pad geometry, friction, compliance, box/flap hinge parameters, belt response | 전류나 단일 파지 성공만으로 각 finger 정상력·마찰계수를 따로 알 수 없음 |

실물 EC_MASTER의 몸통 `Kd`는 velocity-loop P register이며 WBC Kp/Kd는 가속도 gain이다.
이 값을 그대로 torque stiffness/damping에 복사하는 방식 대신, 먼저 전체 폐루프 응답을
맞출 수 있다. 물리 motor gain이 필요하면 encoder/current/gear/torque constant와
active driver 내부 제어식을 별도로 확인해야 한다.
[실물 제어 계층/값 확인 기록](REAL_ROBOT_CONTROL_AUDIT.md).

PhysX inverse dynamics는 damping, joint friction, contact를 포함하지 않는다.
따라서 중력 보상 후 남는 정지 오차가 곧 질량 오차라고 판단하면 안 된다.
[NVIDIA inverse dynamics 설명](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/extensions/runtime/source/omni.physics.tensors/docs/inverse_dynamics.html).

### 기존 S63 gripper 측정에서 이미 확인된 차이

기존 session 02 보고서 기준이며 이번 검토에서 원본 126,771행을 다시 분석한 결과는 아니다.

- 사용자 실물 관찰: 0%=열림, 100%=닫힘. 명목 signed action은 `1-2*p/100`.
- 후속 session 03에서 왼손 닫는 방향 0/25/50/75/100의 폭은
  90/78/51/38/0 mm, 여는 방향 75/50/25는 18/40/68 mm로 기록됐다.
  한 cycle이며 오른손, 여는 방향 0, 내측/외측면 기준 및 폭 측정의 정확한 시각은 미확인이다.
- 증가 방향의 중간 목표는 약 −6%p, 감소 방향은 약 +6%p에서 멈춤. 상수 offset 하나로 맞출 수 없음.
- speed=25, effort=1 A 조건에서 관찰 10→90% 이동 시간은 약 0.24–0.35초.
  순수 command latency나 torque gain을 식별한 값은 아님.
- `/leju_claw_state`에 실물 feedback과 visualizer의 all-zero Reached 값이 함께 발행됨.
  합쳐진 메시지 빈도를 motor sampling rate로 사용하거나 zero 행을 그대로 fit하면 잘못된 모델이 된다.
  recorder에 publisher callerid/connection provenance를 남기고 근본적으로 별도 topic을 사용하는 것이 우선이다.
  all-zero 제거 규칙은 정상 실물 0도 버릴 수 있어 일반화하지 않는다.
- 약 6% stop band는 SDK의 precision setting과 비슷하지만, software acceptance인지
  mechanical friction/backlash인지 확정되지 않았다. 작은 목표 변화·반복·운용 설정 확인으로 구분한다.

현재 Leju 모델은 donor S200062 기구와 simulation-estimated 약 0.74 kg/hand를 사용한다.
S63 실물과 동일한 기구/질량이라는 실측 검증은 없다. % 로그는 두 motor crank의 각도나
jaw force를 측정한 데이터가 아니므로 scalar gain만 바꾸는 것으로 기구 보정을 대신하지 않는다.
[측정 결과](../real_robot_calibration/s63_gripper/reports/session_02.md),
[후속 폭 측정](../real_robot_calibration/s63_gripper/reports/session_03.md),
[기존 측정 도구/절차](../real_robot_calibration/s63_gripper/README.md).

## 측정 순서와 보정 방법

1. **출처·시간·단위 먼저 확정.** hardware feedback을 visualizer/명령과 분리하고
   joint name/sign/zero, rad vs deg, velocity/current/effort 단위, mode, active gain/WBC flag를 기록한다.
2. **정적 기구/중력.** 외부 TCP, pad 폭, 실제 hand/box mass부터 측정한다.
   여러 hold 자세를 정·역방향 접근해 gravity와 방향 의존 오차를 분리한다.
3. **동적 servo.** 기존 실물 제어기를 바꾸지 않은 작은 반복 motion을 기록하고,
   같은 논리 command와 initial pose를 sim에 replay한다. 먼저 latency/bandwidth를 맞춘 뒤
   effort 데이터가 있을 때 inertia/friction을 단계적으로 식별한다.
4. **동일 접촉 task.** 실제 pad와 box/flap을 맞춰 접촉 위치, 닫힘 폭, 힘, slip,
   lift/hold를 비교한다. 파지력을 calibrated gauge로 측정하지 않았다면 current를 N으로 취급하지 않는다.
5. **held-out 검증.** 보정에 사용하지 않은 자세·속도·하중·좌우 및 여러 반복에서 다시 비교한다.
   기본 stationary hold 이후 body/base 이동을 포함한 실험으로 넓힌다.
6. **policy 입력/명령 일치.** real estimator와 같은 noise/filter/age로 학습·재평가한다.
   actor가 unavailable contact truth를 사용하지 않는지, action 적분/reset/관절 순서가 같은지 확인한다.

기록할 최소 채널은 q_cmd/v_cmd/tau_ff, measured q/v와 가능하면 calibrated current/torque,
command/state 각각의 hardware/ROS/receipt timestamp, control mode/gains, joint order,
base IMU/odometry, payload, gripper speed/effort/feedback, 외부 TCP 및 box pose다.
`/joint_cmd`만 있으면 요청 값을 알 수 있지만 실제 motor torque/gain을 검증할 수 없다.
q_cmd/q만 있어도 폐루프 tracking은 맞출 수 있으나 물리 파라미터 식별에는 제한이 있다.

pose별 고정값을 찍어 넣는 대신 물리적으로 의미 있는 작은 parameter set부터 fit한다.
정적 모델은 identifiable base inertial parameters/first moments를 고려하고,
positive mass 및 positive-definite inertia 같은 물리 제약을 유지한다.
평가 지표는 joint/TCP RMSE와 최대 오차, static bias, 10–90% time/overshoot,
latency/jitter, saturation duration, gripper width/force/slip, task 성공률/접촉 빈도다.
운용 목적과 실물 반복성·센서 오차를 기준으로 허용 오차를 정한다.

기록된 command replay와 단계적 parameter fit, held-out validation은 NVIDIA system
identification 설명과 같은 접근이다. 최신 문서의 새 extension 설치를 이 checkout에서
확인한 것은 아니며, 우선 현재 Isaac Lab 2.3.2 환경에서 offline replay로 진행할 수 있다.
[NVIDIA system identification 설명](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_robot_setup_sysid.html).

## 중력 randomization과 남은 확인

현재 PhysX g query는 randomized physical mass를 매번 정확히 사용해 정적 gravity model
오차를 상쇄한다. 실제 nominal robot/payload 모델과 sensor q/IMU로 계산하는 보상을
sim에서도 사용하고 physical model/observation/delay를 독립적으로 랜덤화하는 방식을 검토한다.
현재 implicit-only writer는 `DelayedPDActuator`를 그대로 대체 지원하지 않는다.
implicit command buffer를 추가할지, explicit PD+g의 전체 torque clamp 모델을 따로
구현할지 결정해야 하며, 지연 모델과 filter를 두 번 적용하지 않는다.

기존 확인 사항도 유효하다: S63 bare의 camera mass/inertia 경고,
S200062 measured pose의 약 1.94° shoulder steady error, external hand/payload 제외,
fixed-root의 balance 미재현 및 torque telemetry가 estimate인 점.
[모델별 중력 검증](ROBOT_GRAVITY_COMPENSATION.md).

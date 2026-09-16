# 실제 로봇 제어 확인 — 2026-09-15

`lab@192.168.0.22`에 키 인증으로 접속해 코드·ROS 파라미터를 읽기 전용으로 확인했다.
로봇 구동 명령, 서비스 호출, 파라미터 변경, 원격 파일 수정은 수행하지 않았다.

## 실행 설정과 확인 범위

- ROS `/robot_version`: `63`.
- 실행 launch: `humanoid_controllers/load_kuavo_real_wheel.launch`, `joystick_type:=h12`, `start_way:=auto`.
- `/wbc_frequency`, `/sensor_frequency`: 각각 `500` Hz (설정값이며 실측 주파수는 아님).
- ROS `/taskFile`: `/home/lab/hb/kuavo-ros-opensource/src/humanoid-wheel-control/humanoid_wheel_interface/config/kuavo_s63/task.info`.
- ROS `/urdfFile`: `/home/lab/hb/kuavo-ros-opensource/src/kuavo_assets/models/biped_s63/urdf/biped_s63.urdf`.
- 활성 설정이 가리키는 작업공간: `/home/lab/hb/kuavo-ros-opensource`, HEAD `f709b69b9`, 기존 미커밋 변경 있음.
- 별도 `/home/lab/kuavo-ros-opensource`도 있으나 HEAD `1f7b43651`이며 활성 task 파일 경로와 다르다.
- `/hardware/is_ready=0`, `/hardware/ready_to_start=1`.
- `/joint_cmd`의 publisher/subscriber는 `/nodelet_manager`. 5초간 한 메시지 읽기를 시도했으나 수신하지 못했다.
- 실행 프로세스 `/proc/5231/maps`는 읽기 권한이 없어, 로드된 바이너리와 현재 소스의 일치 여부는 확인하지 못했다.

이하 동작 설명은 활성 설정 경로의 소스와 ROS에 로드된 설정을 근거로 한다.
실시간 모터 레지스터 값, 서비스로 전환 가능한 플래그의 현재 내부 상태, 실제 출력 토크는 미확인이다.

## 중력과 자세 제어

원격 `src/humanoid-wheel-control/humanoid_wheel_wbc/src/WbcBase.cpp`:

- `updateMeasured()`는 측정된 `q`, `v`로 `pinocchio::crba()` 및 `pinocchio::nonLinearEffects()`를 호출한다.
- 동역학 제약은 `M*ddq - J^T*f - S^T*tau = -nle` 형태다. `nle`에는 중력 및 코리올리/원심력 항이 들어간다.
- 코드에서 M의 base/몸통/팔 사이 일부 결합 블록을 0으로 만든다. 완전한 결합 동역학을 그대로 사용하는 구조는 아니다.
- 몸통·팔 PD는 토크가 아니라 목표 관절 가속도를 계산한다: `ddq_des=Kp*(q_des-q)+Kd*(v_des-v)`.
- 몸통 목표 가속도는 ±20, 팔은 ±300으로 제한한다.

원격 `src/humanoid-wheel-control/humanoid_wheel_wbc/src/WeightedWbc.cpp`:

- WBC는 동역학 및 토크 제한을 제약으로 하고, 몸통/팔 목표 가속도와 torso 작업을 비용으로 하는 QP를 푼다.
- 최종 토크는 중력 항을 포함한 이 동역학 제약을 만족하도록 계산된다. 단순 PD 토크만 전달하는 방식이 아니다.

원격 `src/humanoid-control/humanoid_controllers/src/humanoidController_wheel_wbc.cpp`:

- 약 1364행: `wheel_wbc_->update()`.
- 약 1372행: 해의 끝에서 몸통 4관절+양팔 14관절 토크 추출.
- 약 1388–1409행: 각 관절의 목표 `q`, `v`, WBC `tau`, `control_modes=2`를 구성.
- `replaceDefaultEcMotorPdoGait()`가 드라이버 종류별 모터 PD 값을 채운 후 `/joint_cmd` 및 공유 메모리에 전달.

목표 자세가 바뀌지 않는 정지 상태에서도 중력을 상쇄하는 토크는 필요하다.
시뮬레이션에는 목표 자세를 유지하는 제어와 중력 보상을 공통 로봇 구동 계층에서 적용하는 것이 적절하다.
다만 `PD 토크+g(q)`만으로 실제 WBC의 가속도 제어·동역학·작업 제약까지 동일해지는 것은 아니다.

## PD는 서로 다른 두 계층

### WBC 목표 가속도 PD — 활성 task.info의 real 설정

몸통 순서: knee, leg, waist_pitch, waist_yaw.
팔은 왼팔/오른팔 각각 zarm 1–7 순서이며 두 팔 값이 같다.

| 대상 | Kp | Kd |
|---|---|---|
| 몸통 4관절 | `[30,30,30,30]` | `[6.2,6.2,6.2,6.2]` |
| 각 팔 7관절 | `[300,64,64,300,70,70,70]` | `[18,12,12,40,30,30,30]` |
| VR용 각 팔 7관절 | `[400,400,400,400,100,100,100]` | `[36,36,36,36,18,18,18]` |

task 파일의 `vrArmAccelTask.useVrArmAccelTask=false`가 시작 기본값이다.
제어기에 이 플래그를 전환하는 콜백이 있으므로 파일 값만으로 현재 실행 중 내부 상태까지 단정할 수 없다.
WBC의 Kp/Kd는 목표 가속도용이므로 Isaac의 토크 기반 stiffness/damping과 직접 호환되지 않는다.

### 모터에 전달하는 PD — ROS /kuavo_configuration에서 추출

| 대상 | Kp | Kd |
|---|---|---|
| 몸통 EC_MASTER 4개 | `[300,300,260,160]` | `[2150,2300,2150,2150]` |
| 각 팔 RUIWO 7개 | `[270,210,256,180,120,120,120]` | `[20,10,10,10,6,6,6]` |
| head RUIWO 2개 | `[10,10]` | `[4,3]` |

로드된 설정의 `useVrArmKpKd=false`. 런타임 전환 콜백 `/enable_vr_arm_kpkd`가 별도로 존재한다.
EC_MASTER PD는 설치된 하드웨어 인터페이스에서 `int32_t` 형태로 취급된다.
드라이버 내부 구현/단위 변환이 확인되지 않아, 특히 몸통 Kd=2150 등을 Isaac damping에 그대로 넣으면 안 된다.

## 토크 제한도 계층별로 다름

| 출처 | 몸통 4개 | 각 팔 7개 |
|---|---|---|
| WBC task.info 토크 제한 | `[668.53,668.53,618.48,618.48]` | `[85,75,57,75,14.1,14.1,14.1]` |
| 로드된 JSON joint_torque_limits | `[467.6,467.6,186.9,186.9]` | `[46.67,52.5,39.9,52.5,9.87,9.87,9.87]` |

제어 명령의 `tau_max`에는 하드웨어 설정의 `max_current`도 사용한다.

따라서 기록에서 팔 `tau_max=18`이 관측돼도 이를 18 Nm 관절 토크 한계로 해석하면 안 된다. Isaac PhysX의 `effort_limit_sim`은 N·m 단위의 관절 hard clamp이므로, S63에는 별도로 설정된 `joint_peak_torque_limits`를 관절별로 적용한다. 팔 1–7은 각 side `[66.67, 75, 57, 75, 14.1, 14.1, 14.1] Nm`다. 기존 sim의 팔 공통 `100 Nm`는 실물 근거가 없는 기본값이었다.
이 값들을 모두 같은 토크 한도로 해석할 수 없으며, 실제 제한 위치와 전류/토크 변환 확인이 필요하다.

## 시뮬레이터에 반영할 때

1. 몸통과 양팔에 전체 로봇 자세 기반 중력 보상을 공통 적용한다.
2. 기존 teleop 팔의 중력 position bias와 중복되지 않도록 통합한다.
3. 실제 모터 PD 단위와 제어식을 확인한 뒤 대응하는 시뮬레이터 값을 설정한다.
4. 실제와 동일한 계층에서 최종 토크 및 전류/속도 제한을 적용한다.
5. 기본 시뮬레이션도 S63 + Leju twofinger로 맞췄다. S200062를 선택하는 경우 모델별 질량·무게중심·모터 설정을 구분한다.
6. 실물과 같은 응답이 필요하면 중력 보상만 아니라 WBC와 저수준 모터 제어의 구분도 반영한다.

## 추가로 확인한 드라이버 자료

원격 `/home/lab/EC_Master_Tools/src/Sharelib/User/ObjectDiction.h`에서
`JOINT_CSP_KP=0x3500`은 위치 루프 P, `JOINT_CSV_KP=0x3504`는 속도 루프 P로 정의한다.
`dirver_lunbi2_4m.json` 설명에는 각각 ×10, ×1000 스케일이 명시되어 있다.
설치된 `EC_Master/.../EcDemoApp.h`의 PDO도 `position_kp`, `velocity_kp` 필드를 가진다.
따라서 몸통 `joint_kd`를 일반적인 토크 PD의 damping으로 해석할 근거는 없다.
정확한 토크 기반 등가 gain에는 위치/속도 루프 제어식과 전류·엔코더·감속기 변환이 필요하다.
위 자료는 설치된 SDK의 정적 정의이며 현재 모터 SDO를 직접 읽은 결과는 아니다.

활성 팔 구성은 C++ RUIWO SDK다. 설치된 `ruiwo_controller_cxx/include/ruiwo_actuator.h`
및 `leju_claw_driver/ruiwoSDK.h`에는 `run_ptm_mode(pos,vel,kp,kd,torque)` 형식이 있다.
함께 설치된 Python SDK에는 gain 범위와 packet 변환도 있지만 그 코드가 활성 C++ SDK와
동일하다고 검증되지는 않았다. 제조사 `Motorevo Driver User Guide`와 활성 SDK의
관절 출력축 토크 단위·제어식 확인이 필요하다. 장치 초기화/모터 gain 변경 도구는 실행하지 않았다.

## 이후 사용자 시험의 명령 관측

사용자가 수행한 `real_robot_calibration/s63_body_arms/data/timing_fix_20260915_234217/run_20260915_234345.jsonl`에는 새로운 `/joint_cmd` 5238개가 기록됐다. 해당 로그의 gain 배열은 위 로드 설정의 몸통/팔/head 값과 일치하고 `control_modes`는 모두 2였다. 관측된 메시지의 `tau_max`는 몸통 40, 팔 18, head 18이며 `tau_ratio`는 모두 1이었다. 이 다섯 배열은 전체 관측 메시지에서 각각 동일했다. 이는 ROS에 송신한 값의 확인이고 실제 모터 register readback·출력축 토크 단위 검증은 아니다. `tau_max`를 이 숫자의 N·m 제한으로 해석하지 않는다.

팔 14개 중 12개 시험 구간 완료와 정지 추종 오차는 [선별 결과](../real_robot_calibration/s63_body_arms/reports/screen_20260915.md)에 정리했다. 소각도 q 비교만으로 WBC와 모터 루프를 각각 식별하거나 mass/COM을 확정하지 않는다. Gripper는 별도 작업 결과를 통합한다.

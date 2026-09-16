# S63 real-like control 보정 계획

## 목표와 현재 판정

대상은 기본 S63 + Leju twofinger의 몸통·양팔이다. 실물과 최대한 비슷한 폐루프 응답을 만들되, 실물 WBC의 QP·접촉 최적화 전체를 simulation에 복제하지 않는다. 최종 제어는 GPU tensor 연산으로 벡터화하고 별도 ROS node, CPU IK/WBC process, 매 step 최적화 solver를 추가하지 않는다.

2026-09-16 robot-only 재검증에서 `PD+G`는 팔 pooled RMSE 0.706°/worst 1.241°/포화 0%였다. 기존 G를 live `M qdd_des+C+G`로 대체하면 0.652°/0.985°로 개선됐고 별도 저속 자세에서도 0.149°에서 0.133°로 개선됐다. 몸통은 0.104°에서 0.237°로 악화되므로 몸통 gravity-PD와 팔 live inverse-dynamics를 조합한 `s63-arm-id`를 S63 `auto` 기본값으로 선택했다. 실물 WBC tau가 G를 대체하는 `recorded-total`은 1.951°/4.013°로 악화됐고 `G+기록 tau`는 중력 중복 가능성이 있어 제외한다. 계산량은 256환경에서 0.266 ms/호출로 충분히 작다. 상세 수치는 [WBC 계산량·실물 유사도 판정](../real_robot_calibration/s63_body_arms/reports/wbc_load_and_fidelity_20260916.md)에 있다.

현재 완료된 항목은 공통 중력 보상, 실물 기반 관절별 torque cap, 실물 VR command/state 기록, q/v motor-target replay다. 몸통 replay의 sim-real 위치 RMSE는 0.070–0.154°다. 팔 robot-only replay도 pooled 0.706°/worst 1.241°로 통과했으므로 자유공간 actuator gain 보정은 완료 판정한다.

## 검증한 후보 제어 구조

실물 mode 2의 경량 근사는 다음 식을 목표로 한다.

```text
tau_total = Kp (q_cmd - q) + Kd (dq_cmd - dq)
          + s_g G(q) + s_c C(q,dq) + s_m M(q) ddq_cmd
          + friction(q,dq)
tau_applied = clamp(tau_total, -tau_peak, +tau_peak)
```

- `G(q)`: 현재 구현된 PhysX gravity compensation을 유지한다.
- `C(q,dq)`, `M(q)`: PhysX inverse-dynamics tensor API를 사용한다.
- `ddq_cmd`: 30 Hz command의 velocity 차분을 제한·필터링해 만든다.
- `s_g/s_c/s_m`: 실물 모델 차이를 흡수하는 소수의 group scale이다. 처음에는 1로 둔다.
- PD, feedforward, friction을 합친 전체 토크가 동일한 관절별 physical cap을 공유한다.
- implicit force drive에서는 feedforward를 solver-only position bias로 표현해 외력이 cap을 우회하지 않게 한다.

이는 contact Jacobian, task QP, constraint optimization을 매 step 푸는 full WBC가 아니다. fixed-root 팔에 필요한 rigid-body inverse dynamics만 사용한다. 몸통은 gravity-PD를 유지하고 팔에만 적용하며, `--dynamics-profile gravity`가 비교/복구 경로다.

## 단계 1: 기록된 WBC torque의 의미 확인

추가 실물 동작 없이 성공한 60초 VR 로그를 사용한다.

1. replay plan에 `/joint_cmd.tau`, kp/kd, mode와 torque cap provenance를 보존한다.
2. 두 가설을 별도 replay한다.
   - `recorded-total`: 기록된 tau가 gravity를 포함한 전체 feedforward다.
   - `gravity-plus-recorded`: 기록된 tau가 gravity 외 residual이다.
3. real q/v와의 오차, 정지 bias, 포화 시간으로 두 가설을 판정한다.
4. 단위 scale은 몸통/팔 group별 하나부터 fit한다. 관절별 임의 scale은 첫 단계에서 금지한다.
5. 기록된 tau를 넣어도 팔 오차가 줄지 않으면 driver 단위 또는 명령 합산 의미가 미확정인 것으로 남기고 runtime 구현에 사용하지 않는다.

이 단계의 recorded-tau replay는 진단 전용이다. 최종 RL/VR 실행이 로그에 의존하지 않는다.

## 단계 2: 경량 inverse-dynamics feedforward

단계 1에서 feedforward의 효과가 확인된 경우에만 공통 robot articulation에 다음을 구현한다.

1. 기존 `G(q)`에 PhysX의 `get_coriolis_and_centrifugal_compensation_forces()`를 추가한다.
2. `get_generalized_mass_matrices()`의 몸통·팔 block과 `ddq_cmd`를 곱한다. 행렬 역산은 하지 않는다.
3. command-rate counter를 두어 `M/C/G`를 30 Hz에서 갱신하고 physics substep 사이에는 cache한다.
4. 모든 feedforward를 `tau_ff/Kp`의 solver bias로 합쳐 implicit drive hard cap 안에 둔다.
5. 잠긴 관절, zero stiffness, reset 직후, stale command에는 dynamic feedforward를 0으로 만든다.
6. acceleration spike를 막기 위해 `ddq_cmd` limit, one-pole filter와 reset seed를 설정한다.

접촉 force를 만드는 full WBC, Jacobian QP, 별도 balance controller는 이 fixed-root 보정에 넣지 않는다. 추후 floating-base 보행을 지원할 때 별도 작업으로 다룬다.

## 단계 3: actuator와 inertial parameter fit

구조를 먼저 고정한 뒤 작은 파라미터 집합만 순서대로 fit한다.

1. **command timing:** motor target→state 구간만 사용해 actuator를 fit한다. VR→motor 약 197 ms는 별도 command-pipeline 항목으로 둔다.
2. **Kp/Kd:** 현재 공통 220/22를 관절 group dictionary로 분해한다. 실물 register gain을 그대로 복사하지 않고 폐루프 q/v 응답으로 맞춘다.
3. **armature/friction:** rise/settle/방향 차이에 Kp/Kd만으로 설명되지 않는 부분을 fit한다.
4. **mass/COM/inertia:** 자세에 따라 반복되는 gravity/dynamic residual이 있을 때만 물리 제약과 CAD prior 안에서 조정한다. Leju hand 1 kg과 카메라 질량은 유지한다.
5. 좌우 값은 held-out 구간에서도 좌우 차이가 반복될 때만 분리한다.

동시에 모든 값을 fit하지 않는다. 순서는 timing → Kp/Kd → armature/friction → inertial parameter다. 각 단계에서 이전 단계보다 validation error가 줄지 않으면 변경을 되돌린다.

## 단계 4: command pipeline 일치

실물 VR→motor 경로의 약 197 ms에는 네트워크만이 아니라 IK filter, controller filter와 mode handling이 포함된다. motor servo 보정과 섞지 않는다.

- Quest teleop에는 실물 `/kuavo_arm_traj`→`/joint_cmd` 응답과 같은 filter/rate/limit을 적용한다.
- RL action이 실물에서 동일 WBC target 경로를 통과한다면 같은 filter와 frame delay를 학습·평가 모두에 적용한다.
- 실물 배포에서 사용하지 않는 VR transport delay는 일반 RL actuator에 넣지 않는다.
- safety stop과 reset은 delay queue를 우회하고 즉시 적용한다.

## 계산량 예산과 선택 기준

현재 gravity-only를 기준으로 동일 headless scene에서 1, 64, 256 environments를 각각 측정한다.

| profile | 계산 | 사용 조건 |
|---|---|---|
| gravity-PD | 기존 `G + PD` | 비교 baseline |
| real-like ID | 30 Hz `G+C+M·ddq + PD` | 기본 후보 |
| cached/diagonal | 30 Hz cached G/C와 identified diagonal inertia | full mass-matrix 후보가 예산 초과 시 |

수용 기준은 64 env step time 증가 10% 이하, 256 env 15% 이하, GPU memory 증가 5% 이하로 둔다. solver iteration 수와 physics rate는 늘리지 않는다. full mass-matrix가 기준을 넘으면 동일 로그에서 diagonal effective inertia를 식별해 O(N joint) 경로로 내린다. 정확도가 유의하게 나빠지는 경우에만 full tensor 경로를 유지한다.

## 정확도와 안전 수용 기준

- 몸통 q RMSE: 관절별 0.2° 이하 유지.
- 팔 q RMSE: 관절별 1.0° 이하 목표, worst joint 2.0° 이하.
- 정지 drift: 0.2° 이하.
- velocity RMSE, 10–90% rise time, overshoot와 saturation duration이 baseline보다 모두 악화되지 않음.
- physical torque cap을 초과하거나 cap 값을 fit 변수로 사용하지 않음.
- NaN, reset jump, stale-command torque, logical target 누적 bias가 없음.
- 보정에 쓰지 않은 VR 구간과 기존 소각도/정지 기록에서 기준을 다시 통과함.

목표를 달성하지 못하면 torque cap이나 관절 limit을 완화하지 않는다. 남은 오차가 driver 내부 제어, 접촉 WBC 또는 미측정 torque telemetry 때문인지 보고하고, 계산량을 늘리는 다음 구조는 개선량과 benchmark를 함께 제시한 뒤 선택한다.

## 실행 순서와 결과물

1. replay schema에 tau/gain/mode를 추가하고 기존 원본 hash 검사를 유지한다.
2. recorded-total / gravity-plus-recorded 진단 replay를 실행한다.
3. 결과가 유효하면 30 Hz inverse-dynamics prototype을 구현한다.
4. `gravity-PD`, `real-like ID`, 필요시 `cached/diagonal`의 정확도와 step time을 같은 조건에서 비교한다.
5. 선택한 구조에서 제한된 actuator/inertial 파라미터를 offline fit한다.
6. held-out validation 후 `configs/s63_servo.json`과 별도 real-like control profile에 확정값·출처·단위를 저장한다.
7. Quest collect, RL reward debug, manager/evaluation에서 같은 profile 선택과 torque cap을 검증한다.

추가 실물 동작은 단계 1–6에 필요하지 않다. 기존 로그만으로 단위/구조가 판정되지 않을 때만 motor torque telemetry를 읽기 전용으로 추가 수집한다. Gripper 보정은 별도 작업 결과를 통합하며 이 계획에서 중복 시험하지 않는다.

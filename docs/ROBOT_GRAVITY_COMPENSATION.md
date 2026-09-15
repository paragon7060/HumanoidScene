# 모든 Kuavo 모델의 공통 중력 보상과 검토 결과

## 적용 범위

`envs/scene_physics.configure_robot_asset_physics()`의 모델별 분기 **이전**에
`configure_gravity_compensation()`을 호출한다. standalone scene, manager env,
Quest collect 및 RL scene은 모두 `GravityCompensatedArticulation`을 사용한다.
기존 모델/그리퍼 기본 선택은 유지한다.

| 모델 | 보상 관절 | 손 옵션 |
|---|---|---|
| S63 | knee/leg/waist_pitch/waist_yaw + 양팔 14개 = 18 | Leju 및 bare/external 설정 |
| S200062 | knee/leg/waist_pitch/waist_yaw + 양팔 14개 = 18 | integrated 및 기존 손 설정 |
| S56 | 다리 12개/waist_yaw + 양팔 14개 = 27 | QiangNao, transplanted twofinger, bare/external 설정 |

wheel, head, hand motor 및 passive linkage는 선택하지 않는다. arms-only의 좁은
물리 limit로 잠긴 몸통/다리/비활성 팔에도 보정을 주지 않는다. separate external
hand의 자체 articulation에는 몸통/팔 writer를 적용하지 않는다.

매 physics write에서 PhysX의 전체 robot gravity tensor를 조회하고 선택 관절의
`g(q)/Kp`를 **solver 목표에만** 더한다. 논리 `joint_pos_target`, IK command 및 action
manager 목표는 유지되므로 보정이 누적되지 않는다. `joint_drive_props.drive_type="force"`를
명시해 torque gain 식을 유지한다. acceleration drive에는 같은 단위 식을 쓸 수 없다.
PD와 보상은 하나의 implicit drive force cap을 공유한다.

S63은 `configs/s63_servo.json`, 다른 모델은 기존 asset의 PD 값을 사용한다.
Quest와 reward debug의 임시 800/50, 8000/200 gain 보강은 보상 모델에서 건너뛴다.
S56의 per-motor force/velocity cap, armature 0.05 및 friction 0.02는 유지한다.
teleop IK에 있던 팔 position gravity bias는 공통 writer가 활성화되면 0이다.
[S63 설정과 구동식](S63_GRAVITY_COMPENSATION.md).

## 검증

아래 수치는 Leju claw를 한쪽당 1 kg으로 변경하기 전의 검증 기록이다.
현재 Leju 패키지는 D405를 포함해 각 1 kg이며, 링크 질량/관성을 함께 스케일하고
spawn/build에서 같은 값을 적용한다. 이후 1 kg 검증 결과는 별도로 기록한다.
현재 1 kg claw에서는 2개 환경의 6초 무입력 전신 검사에서 몸통 최대 오차
0.007590 rad, 왼팔 0.002036 rad, 오른팔 0.002077 rad였다.
오른팔 전용과 합성 Quest 양팔 IK도 확인했다.
[현재 1 kg 검증 상세](S63_GRAVITY_COMPENSATION.md#현재-1-kg-claw-검증-2026-09-15).

초기 상태를 명시적으로 복원한 뒤 두 환경에서 120 Hz physics 720스텝(6초),
무입력 명령을 유지했다. 목표 tensor 불변, joint state 유한성 및 보상 관절 수를 검사했다.
아래는 준비한 비영점 body/leg/arm 자세에서의 최대 목표 각도 오차이며, 전체 작업 공간이나
임의의 하중·자세에 대한 보장은 아니다.

| 모델/손 | device | 몸통/다리 최대 오차(rad) | 양팔 최대 오차(rad) |
|---|---|---:|---:|
| S200062 integrated | CPU | 0.00628 | 0.00148 |
| S63 Leju | GPU cuda:0 | 0.00754 | 0.00199 |
| S63 bare | CPU | 0.00658 | 0.00499 |
| S56 twofinger | CPU | 0.000156 | 0.000195 |
| S56 QiangNao | CPU | 0.000156 | 0.000266 |
| S56 bare | CPU | 0.000156 | 0.000210 |

S56 RL whole-body는 별도 준비한 `s56_twofinger_ready_01`에서 180 control steps를
확인했다. 기존 S200062 측정 자세를 S56에 적용하려던 config 참조를 수정했다.
이 자세는 fixed root simulation용이며 실제 측정값 또는 지면 균형 자세가 아니다.
S56 오른팔 전용도 두 환경에서 180스텝을 확인했다. 잠긴 다리/waist/왼팔의 보상은 0,
오른팔 최대 오차는 0.000169 rad였다.

S200062의 **기존 측정 자세** `quest_ready_02`에서는 다음 잔여 오차가 있다.

| RL 작업 공간, 180 control steps | 몸통 | 왼팔 | 오른팔 |
|---|---:|---:|---:|
| 보상 OFF | 0.44366 rad | 0.11466 rad | 0.15317 rad |
| 보상 ON | 0.00529 rad | 0.03399 rad | 0.00736 rad |

왼쪽 첫 어깨 관절 `zarm_l1_joint`는 장애물을 제거한 동일 자세에서도 최대
0.033995 rad, 6초 후 약 0.033871 rad(1.94°)의 잔여 오차가 있었다.
해당 무접촉 검사에서 모든 보상 관절은 force drive였고, 계산 중력 토크/force cap의
최대 비율은 약 0.236이었다. 작업 공간 장애물이나 중력 항 단독의 토크 포화가 원인이라고
단정할 근거는 없다. 기본 준비 자세 검사는 0.03 rad, 이 측정 자세 검사는 0.05 rad를
상한으로 검사하며, 측정 자세의 오차는 해결된 것으로 취급하지 않는다.

## 추가로 확인할 사항

1. **실물 gain과 토크 단위.** 실제 몸통의 `Kd` 필드는 velocity-loop P raw register다.
   WBC acceleration gain과 motor gain도 다르다. 실제 `/joint_cmd` 및 피드백 로그,
   active driver의 단위/기어비/torque constant 확인 후 joint-output Kp/Kd로 변환해야 한다.
   현재 PD는 real-equivalent로 검증된 값이 아니다.
2. **S63 bare 관성.** 실제 실행에서 `/Robot/head_camera_base`에 negative mass 및
   invalid inertia 경고가 발생했고 PhysX가 작은 구의 관성으로 대체했다. 자세 유지 검사는
   통과했지만 이 경로의 계산 중력 토크를 실물과 비교하기 전에 USD/URDF 관성을 확인해야 한다.
   모든 모델에서 일부 EEF visual reference 경고도 남아 있다. 이번 변경은 원본 asset을 수정하지 않는다.
3. **S200062 측정 자세의 잔여 어깨 오차.** joint friction, closed-link constraints,
   solver force 및 실제 joint torque를 분리해서 확인해야 한다. 원인은 아직 규명하지 않았다.
   높은 Kp로 오차를 숨기거나 측정 자세를 처진 값으로 바꾸지 않았다.
4. **하중과 external hand.** 별도 박스 payload/접촉/마찰은 robot gravity query에
   포함되지 않는다. external hand attachment는 `excludeFromArticulation=True`인
   FixedJoint로 연결되므로 별도 hand 질량 역시 robot gravity 모델에 자동 포함되지 않는다.
   nominal payload나 TCP wrench를 추가하는 방식이 필요하다.
5. **free-base 보행/균형.** 현재 모든 패키지 모델은 fixed root 경로로 검증했다.
   floating root는 명시적으로 거부한다. S56의 고정된 torso 아래 다리 보상이 실제 지면에서
   상체를 지탱하는 균형 제어를 검증하는 것은 아니다. floating base에는 접촉력을 포함한
   WBC/inverse dynamics가 필요하다.
6. **domain randomization.** 현재 매번 PhysX에서 구하는 g는 랜덤화된 실제 simulation
   질량을 안다. 따라서 질량 변경에 따른 정적 중력 모델 오차를 상쇄한다. 실물처럼 nominal
   model로 보상하면서 physical mass를 랜덤화할지 별도 결정이 필요하다. 질량 랜덤화의
   동적 관성 효과까지 사라지는 것은 아니다.
7. **포화와 진단.** 모든 자세와 박스 무게에서 PD+g가 force cap 아래인지 추가 확인한다.
   `gravity_compensation_torque`는 요청된 계산값이다. `computed_torque`/
   `applied_torque`는 implicit drive 추정치이며 실측 토크가 아니다. 별도 external effort를
   더하면 그 외력은 drive cap 바깥이므로 이 보상 방식과 구분해야 한다.
8. **기존 RL/수집 제약.** gravity asset 지원과 flap reward의 손 센서/파지 지원은 별개다.
   현재 RL contact presets는 S200062 integrated, S56 twofinger, S63 Leju만 지원한다.
   QiangNao/bare/external 손을 flap RL로 사용하는 변경은 포함하지 않았다. 기존 정책은
   effective servo dynamics가 달라졌으므로 성능을 다시 평가해야 한다. 실제 Quest 장치와
   real robot motion은 이번 검증에서 실행하지 않았다.

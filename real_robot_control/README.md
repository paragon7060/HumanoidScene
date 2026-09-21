# S63 real robot control

VR teleoperation으로 수집한 상태 궤적 또는 RL policy가 만든 정규화 action을
실물 S63의 기존 wheel-WBC 팔 제어 경로에 연결하는 독립 폴더다. Isaac Sim을
실행하지 않으며, 실물에는 논리적 관절 목표만 보낸다. 시뮬레이터의 중력 보상,
PD gain, torque 또는 `/joint_cmd`를 복사하지 않는다.

2026-09-21 로컬에 설정된 로봇 SSH 대상에서 읽기 전용으로 다시 확인한 활성 계약은 다음과 같다.

- 상태: `/sensors_data_raw`, `kuavo_msgs/sensorsData`, 500 Hz, body 4 + left arm 7 + right arm 7 + head 2, rad/rad/s.
- 팔 목표: `/kuavo_arm_traj`, `sensor_msgs/JointState`, left 7 + right 7.
- 활성 `ArmController::storeMode2Target()`은 `JointState.position`과 `velocity`를 degree/degree/s로 받은 뒤 rad/rad/s로 변환한다. 일반적인 `JointState` 관례와 다르므로 `ros1_runtime.py`에서만 이 변환을 수행한다.
- 외부 팔 모드: `/change_arm_ctrl_mode`, mode 2=external, mode 0=현재 명령 고정.
- Leju claw: `/control_robot_leju_claw`, 실측 계약 0%=open, 100%=closed. 데이터 계약은 0=open, 1=close다.

## 구조

```text
real_robot_control/
├── config/s63.json                 # 실제 URDF 한계, 순서, 속도/감시 한계
├── kuavo_real_control/
│   ├── contract.py                 # S63 고정 계약 검증
│   ├── prepare.py                  # VR HDF5 / RL action 변환
│   ├── trajectory.py               # portable NPZ 형식
│   ├── safety.py                   # limit/rate/accel/tracking watchdog
│   └── ros1_runtime.py             # ROS 상태, WBC mode, arm/claw 출력
├── prepare_teleop_hdf5.py
├── prepare_rl_actions.py
├── make_hold_trajectory.py
├── inspect_trajectory.py
├── run_robot.py
└── tests/
```

`NPZ`는 로봇 PC에 없는 `h5py`나 PyTorch를 요구하지 않는다. HDF5/checkpoint 관련
처리는 workstation에서 끝내고, 로봇 PC에는 검증된 NPZ와 이 폴더만 복사한다.

## 1. VR 데이터 준비

Quest 수집기의 Cartesian EE action을 그대로 실물에 보내지 않는다. 기본 변환은
`self_collision_safe_joint_target`에서 정확한 14개 팔 관절을 이름으로 선택한다. 즉
bounded IK와 self-collision guard를 지난 논리 목표가 명령 원본이고,
`robot_joint_position`은 비교용 측정 상태다. safe target이 없는 이전 HDF5에 한해서만
측정 상태로 fallback하며 metadata에 source를 명시한다.

성공으로 끝난 episode만 기본 허용한다. `success=false`는 변환을 거부하며
`--allow-unsuccessful`은 offline 검토용 artifact에만 쓴다. 이렇게 만든 artifact도
`deployment_ready=false`라 live runner가 거부한다.

```bash
python3 real_robot_control/prepare_teleop_hdf5.py \
  datasets/kuavo_quest_example.hdf5 \
  real_robot_control/trajectories/quest_demo_00000.npz \
  --episode demo_00000

python3 real_robot_control/inspect_trajectory.py \
  real_robot_control/trajectories/quest_demo_00000.npz
```

변환기는 원본을 실기 VR 로그에서 관측한 replay 운용 cap인 1.5 rad/s의 90% 이하로
느리게 만들고 30 Hz로 다시
표본화한다. 같은 acceleration supervisor를 offline으로 통과시켜 source 종료 시점에
limiter가 뒤처진 채 궤적이 잘리는 것을 막는다. 원본/출력 시간, 최대 속도·가속도,
filter 오차와 source SHA256은 NPZ metadata에 남는다. joint safe limit을 넘으면 clamp하지
않고 변환을 실패시킨다. 원본 timing만 비교하려면 `--keep-source-timing`을 쓸 수 있지만
그 결과는 live 실행할 수 없다.

기록된 sim 자세와 현재 실물 자세의 첫 프레임 차이가 관절별 0.25 rad를 넘으면 실행기는
거부한다. 이는 sim 궤적을 임의 offset해 실물에 맞추지 않기 위한 의도적인 제한이다.

2026-09-21 새 수집분의 offline 검사 결과:

- `kuavo_quest_20260921-203936-881819_190d02f4.hdf5`: 성공 episode지만
  `zarm_l1_joint`가 39 sample에서 0.03 rad safe margin을 넘고 일부는 URDF hard limit에
  도달했으므로 변환을 거부했다. 자동 clamp하지 않는다.
- `kuavo_quest_20260921-204112-770396_ff0ab300.hdf5`: 성공 episode, 최소 self-collision
  거리 0.00300086 m, safe target 선택. 원본 최대 2.83494 rad/s를 2.62494배 retime하여
  30 Hz 313 sample, 10.400초 deployment artifact로 검증했다. 출력 최대 속도
  1.06535 rad/s, 최대 가속도 8.0 rad/s²다. 이는 offline 명령 검증이며 실물 안전이나
  task 성공을 입증하지 않는다.

두 번째 episode의 첫 자세는 기록된 `s63_leju_vr_collect_01` 초기 팔 자세와 최대
2.41279 rad(왼팔 2번, 138.243°) 차이가 난다. HDF5의 첫 measured state도 최대
2.40765 rad 차이가 나므로 변환 과정에서 생긴 차이가 아니라 녹화를 누른 시점에 이미
로봇이 초기 preset에서 이동해 있었던 것이다. 따라서 실물이 그 초기 자세에 있으면
0.25 rad start gate가 mode 전환 전에 실행을 거부한다. 이 차이를 허용하도록 gate를
넓히지 않는다. 다음 수집에서는 초기 자세 hold부터 실제 이동 경로를 episode에 포함하거나,
별도로 시뮬레이션에서 충돌 검증된 transition trajectory를 만들어야 한다.

## 2. RL action 준비

입력은 `(N,14)` 정규화 팔 action 또는 `(N,16)`의 팔 14 + binary gripper 2다.
`.npy`, `.npz`, JSONL을 지원한다. `.npz`/JSONL의 기본 key는 `action`이다.

```bash
python3 real_robot_control/prepare_rl_actions.py rollout_actions.npy \
  real_robot_control/trajectories/rl_rollout_01.npz --hz 30
```

각 action은 policy sample마다 정확히 한 번 `q_target += action * 0.035 rad`로 적분된다.
500 Hz sensor callback마다 적분하지 않는다. scale은 sim과 실물 config 양쪽에서 같아야
실행된다.

현재 flap/multi-box PPO actor는 정확한 box/flap pose, contact/force, grasp flag 등
실물에서 아직 생성하지 못하는 observation을 포함한다. 이 코드는 그 값을 0이나 상수로
위조해 checkpoint를 직접 실행하지 않는다. 먼저 동일 센서 observation으로 재학습하거나
별도 perception adapter를 검증한 뒤, 그 policy가 출력한 14/16차원 action을 위 계약으로
연결해야 한다. 사전 계산한 open-loop action은 정책의 폐루프 성능 검증이 아니다.

## 3. 로봇 PC에서 dry-run

먼저 폴더와 하나의 NPZ를 로봇 PC의 별도 작업 위치로 복사한다.
활성 `$KUAVO_ROBOT_WORKSPACE` checkout은 수정하지 않는다.

```bash
ssh "$KUAVO_ROBOT_SSH"
source /opt/ros/noetic/setup.bash
source "$KUAVO_ROBOT_WORKSPACE/devel/setup.bash"
export ROS_MASTER_URI=http://kuavo_master:11311
export ROS_IP=<robot-ip>

cd /path/to/HumanoidScene-real-control
python3 make_hold_trajectory.py trajectories/hold_1s.npz
python3 run_robot.py trajectories/hold_1s.npz --dry-run-speed 10
python3 inspect_trajectory.py trajectories/rl_rollout_01.npz
python3 run_robot.py trajectories/rl_rollout_01.npz --dry-run-speed 10
```

기본 실행은 publisher/service를 만들지 않는 dry-run이다. live state, robot version,
joint count, enable/disabled state, 첫 자세, 모든 limit/action/시간 규칙을 검사하고 로그만
남긴다. dry-run의 shadow state는 명령을 따라간다고 가정하므로 실제 추종 성능 시험은 아니다.

## 4. 실제 동작

작업 공간에서 사람과 장애물을 치우고, 하드웨어 E-stop 담당자를 둔다. Dashboard,
Quest/IK, trajectory demo 등 `/kuavo_arm_traj`를 낼 수 있는 다른 조작을 중지한다.
실행기는 시작 전 0.5초와 실행 중 실제 foreign message가 하나라도 들어오면 중단한다.
예를 들어 Dashboard jog가 열려 실제 메시지를 내는 동안에는
`Foreign /kuavo_arm_traj command received`로 dry-run부터 실패한다. 해당 조작을 운영자가
정상 종료한 뒤 새 로그 이름으로 다시 확인하며, 이 도구가 다른 node를 자동 종료하지 않는다.

팔만:

```bash
python3 run_robot.py trajectories/rl_rollout_01.npz \
  --enable-motion --confirm S63_CLEAR_AND_ESTOP_READY
```

팔과 Leju claw:

```bash
python3 run_robot.py trajectories/rl_rollout_01.npz \
  --enable-motion --enable-gripper \
  --confirm S63_CLEAR_AND_ESTOP_READY
```

실행 순서는 현재 state 동기화 → 전체 trajectory prevalidation → mode 2 → 0.75초 현재
자세 hold → 3초 quintic first-pose approach → 0.5초 settle 및 실제 추종 오차 검사 →
운영자의 `PLAY` 입력을 기다리며 first pose hold → 30 Hz source → 0.5초 마지막 자세
hold → mode 0이다. source 중에는 같은 terminal에 `STOP` Enter로 중단할 수 있고
Ctrl+C도 context cleanup을 거쳐 mode 0을 요청한다. terminal 입력은 하드웨어 deadman이나
E-stop이 아니다. 정상 종료나 예외에서 mode 1(home/auto-swing)로 자동 복귀시키지 않는다.
mode 0 전환도 하드웨어 E-stop을 대신하지 않는다.

## 안전 경계

아래 수치는 제조사 정격이나 실물 식별 결과가 아니다. 2026-09-16 실기 VR 로그의
`/joint_cmd` 절대 속도 p99=1.495663 rad/s, 절대 가속도 p99=7.704488 rad/s²를 반올림해
replay 운용 cap으로 정한 값이다. `config/s63.json`에 근거 로그와 수치를 함께 기록한다.
활성 컨트롤러의 `/arm_move_spd=1.2`는 mode 2 진입 시 첫 외부 목표로 이동하는 전환
속도이며 연속 replay cap이 아니다. 실제 WBC와 모터의 허용 속도·가속도는 별도 계측과
제조사 자료로 확정해야 한다. URDF velocity는 simulation/model 값이므로 이 cap의
근거로 사용하지 않는다.

- URDF hard limit 안쪽 0.03 rad만 사용하며, 범위를 벗어난 target은 clamp하지 않고 중단한다.
- 명령 속도 1.5 rad/s, 가속도 8.0 rad/s², source step 0.18 rad로 제한한다.
- 첫 자세 오차 0.25 rad를 넘으면 시작하지 않는다.
- first-pose approach 뒤 실측 오차가 0.10 rad를 넘으면 `PLAY` 단계로 가지 않는다.
- 명령–측정 오차 0.35 rad가 0.25초 지속되거나 state가 0.1초 stale이면 중단한다.
- action, timestamp, joint order, scale, NaN/Inf, binary gripper를 실행 전에 검증한다.
- 성공 Quest episode와 deployment-ready retiming metadata가 없으면 live 실행을 거부한다.
- `/joint_cmd.tau`, motor gain, gravity torque는 발행하지 않는다.

이 운용 cap은 관측 로그를 벗어나지 않기 위한 값이며 실물 성공을 보장하지 않는다. 첫 live 시험은
0 action 또는 현재 자세 근처의 단일 관절 소각도 궤적으로 수행하고 로그와
`/sensors_data_raw`를 대조한 뒤 범위를 넓힌다.

## 다음 검증 순서

1. `s63_leju_vr_collect_01`에서 artifact 첫 자세까지의 transition을 만들고 S63+Leju
   self-collision model로 전 구간을 검사한다. 현재 큰 첫 자세 차이 때문에 이 단계 전에는
   start gate를 완화하지 않는다.
2. transition과 10.4초 replay를 Isaac dynamics에서 연속 실행해 target clearance,
   실제 joint tracking, base 안정성과 gripper 간섭을 확인한다.
3. 운영자가 로봇 PC에서 dry-run을 실행해 live joint order/state, foreign publisher,
   enable 상태를 확인한다. 이 저장소 작업에서는 로봇 명령을 원격 실행하지 않는다.
4. 실물은 hold → 현재 자세 주변 단일 관절 소각도 → 팔 replay(gripper off) → gripper
   포함 replay 순서로 확대하고, 각 단계에서 로그를 보존한다.

## 테스트

ROS와 실물 동작 없이 core/format/converter를 검사한다.

```bash
conda activate env_isaaclab_232
python3 -m unittest discover -s real_robot_control/tests -v
```

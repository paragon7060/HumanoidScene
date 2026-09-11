# 공통 endeffector_center — S200062 integrated hand

## 정의와 보정값

`configs/grasp_reference_points.json`은 좌우 대응 손가락을 로컬 Y 반사 후 평균낸 값입니다.
f/b 형상 전체는 완전 대칭이 아니므로 두 손가락끼리는 억지로 평균내지 않았습니다.
수동 보정 원본은 `configs/grasp_reference_points_manual.json`에 보존했습니다.
배포 기본값은 `src/kuavo_isaaclab_scene/configs/grasp_reference_points.json`에도 포함됩니다.

| 점 | 로컬 X (mm) | 로컬 Y (mm) | 로컬 Z (mm) |
|---|---:|---:|---:|
| left f | -29.141679 | -0.782802 | -65.602375 |
| left b | 29.509662 | -0.439710 | -65.402815 |
| right f | -29.141679 | 0.782802 | -65.602375 |
| right b | 29.509662 | 0.439710 | -65.402815 |

서로 다른 손가락 프레임의 숫자를 바로 평균내지 않습니다. 각 점을 같은 프레임으로 변환한 뒤
중점을 계산합니다. 두 가지 중심을 구분합니다.

- `endeffector_center`: 닫힘 명령의 **nominal closed TCP**. URDF와 기존 four-bar 수동관절
  해를 이용해 닫혔을 때의 중점을 계산하고, 원래 EEF에 대한 고정 오프셋으로 둡니다.
  방향은 원래 EEF의 방향을 유지합니다. IK/관측/기록/reaching이 사용하는 안정적인 기준입니다.
- `midpoint_w`: 실제 시뮬레이션 손가락 링크 pose로 갱신되는 **실시간 두 점의 중점**.
  개폐·접촉·물리 오차에 따라 고정 TCP와 달라질 수 있습니다.

현재 close motor command가 0 rad일 때 원래 `zarm_l/r7_end_effector` 기준 고정 위치(m):

```text
left:  (0.000183991523, -0.001111255819, -0.055497895149)
right: (0.000183991523, +0.001111255819, -0.055497895149)
rotation: identity relative to original EEF
```

이 값은 실제 무부하 닫힘을 실행·측정한 값이 아니라 **기구학 기준값**입니다.
현재 선택점은 nominal close에서도 약 5.11mm 떨어져 있으므로, 두 점이 완전히 겹칠 것을
기대하지 마세요. 실시간 중점과 고정 TCP가 근접하는지를 확인합니다.
물체를 잡은 채로 닫으면 목표 각도에 도달하지 못할 수 있습니다.

## VR에서 확인

기존 simulator를 종료한 뒤 저장소 루트에서:

```bash
./quest_collector.sh collect --rl-reward-debug --rl-grasp-markers \
  --rl-config configs/rl_inspect_endeffector_centers.py \
  --rl-endeffector-centers --no-rl-collision-view
```

- 왼손 f/b: 주황/분홍 구체
- 오른손 f/b: 청록/파랑 구체
- **빨간 큐브: closed endeffector_center (IK/reward 기준)**
- **흰 구체: 실시간 두 점의 중점**
- HUD: `Live-center L/R` 차이(mm), `Tip separation L/R` 거리(mm)
- G로 표시 토글. `--no-rl-endeffector-centers`는 기존 paired-goal 표시 경로로 돌아갑니다.
- `--rl-grasp-calibration` 편집 모드와 동시에 쓰지 마세요. 편집 모드가 우선합니다.

A로 진행 후 양손을 움직이고 열고 닫아 확인합니다. 위 명령은 전용 검사 config에서
`active_arm="both"`로 양팔/그리퍼만 제어 가능하게 합니다. 몸통/머리/베이스는 고정이며
reward는 원래 오른손 task를 유지합니다. 검사 config를 학습용으로 사용하지 마세요.
`--rl-config`를 생략하면 기존 오른팔 전용 설정으로 왼손은 고정됩니다.
기본 학습 config의 왼손 고정을 해제하지 않았습니다.
마커는 x-ray가 아니며 손/물체에 가려질 수 있습니다. 빨간 큐브와 흰 구체가 겹치면
닫힘 기준과 실시간 중점이 가까운 상태입니다.

## 모든 환경에서 접근하는 공통 API

```python
from kuavo_isaaclab_scene.robots.end_effector import get_end_effector_frames

frames = get_end_effector_frames(scene["robot"])
closed_pose_w = frames.center_pose_w  # (num_envs, 2, 7), xyz + quaternion wxyz
closed_pose_b = frames.center_pose_b  # robot root 기준, 같은 shape
tips_w = frames.tips_w               # (num_envs, 2, 2, 3), left/right, f/b
midpoint_w = frames.midpoint_w       # (num_envs, 2, 3)
# 환경 로컬 위치: closed_pose_w[..., :3] - scene.env_origins[:, None, :]
# 호출 후 scene["robot"].endeffector_center로도 같은 객체 접근 가능
```

이 객체는 rigid body가 아닌 **읽기용 좌표 계산 API**입니다. `scene["endeffector_center"]`라는
별도 articulation이 추가된 것은 아닙니다. 위치를 제어하려면 해당 TCP를 목표로 IK/joint
action을 보내야 하며, 마커 prim을 옮겨 로봇을 이동시키지 않습니다.

일반 scene/manager/RL의 S200062 spawn에는 원래 EEF 아래에 비물리 Xform도 생성됩니다:

```text
<robot root>/zarm_l7_end_effector/endeffector_center
<robot root>/zarm_r7_end_effector/endeffector_center
```

원래 EEF와 `_1`, `_2` 프레임, 손가락 collision, 카메라/그리퍼 장착 위치는 보존합니다.
Fabric 사용 중에는 USD transform 읽기보다 위 tensor API가 현재 물리 pose를 읽는 기준입니다.

## 적용 범위 / 의도적으로 유지한 계약

| 기능 | 적용 |
|---|---|
| 일반 scene·manager scene | spawn에 closed center Xform 추가; 공통 API로 접근 |
| Quest / browser PersistentTeleopIKAction | closed TCP offset 포함 pose 및 Jacobian, URDF IK 일치 검증 |
| Quest 목표 mapping·손속도 검사 | closed TCP pose 기준 |
| HDF5 / LeRobot 신규 수집 | EEF pose와 Cartesian action이 closed TCP 기준; episode metadata에 전체 정의 기록 |
| GR00T eval bridge | `bridge.end_effector_pose_w` 제공; 기존 joint state/action 차원·joint 의미는 변경하지 않음 |
| RL reaching / flap_reaching / hand-flap observation / button reach | 기존 추가 -12cm proxy 대신 closed TCP |
| RL approach_rack | base/yaw navigation 유지 + closed TCP 접근 보상 `approach_reaching` (weight 1) 추가 |
| RL 파지 양쪽 판정·alignment·slip | 링크 원점 대신 보정된 실제 움직이는 두 점 사용; 접촉점/힘은 계속 PhysX 센서 기준 |
| RL VR 표시 | 같은 공통 API의 네 점·실시간 중점·고정 TCP 표시 |
| reset bank / checkpoint 계약 | 실제 offsets와 closed TCP 정의를 포함; 기존 정의와 불일치하면 거부 |

`approach_reaching = exp(-6*d)`, d는 필요한 손의 TCP와 목표점 거리 평균이며 approach phase에만
활성화됩니다. 기존 stationary flap pick에서는 이 항목을 비활성화합니다. flap reaching은
TCP 거리의 `exp(-12*d)` 최고 점수를 갱신할 때만 weight 4로 진전 보상을 줍니다.
자세한 누적 보상·reset 규칙은 [오른손 파지 가이드](RL_RIGHT_HAND_PICK.md)를 따릅니다. 기존 base navigation의
성공/충돌 조건을 손의 근접성만으로 대체하지 않았습니다.

모든 `zarm_*7_end_effector` 문자열을 새 이름으로 치환하지 않았습니다. 원래 프레임은
관절/장착/기구학 기준으로 여전히 필요합니다. 로봇이나 정책이 **joint action**을 쓰는
경우에는 EEF 변경을 이유로 joint 값이나 차원을 바꾸지 않습니다.

## 호환성과 재실행

- **보정 대상은 현재 S200062 integrated hand뿐**입니다. S56/S63/Allegro 등에 수치를 자동 적용하지 않습니다.
- 보정 파일 변경 후에는 소비 프로세스를 재시작합니다. 이미 실행 중인 학습에 파일을 실시간 재로딩하지 않습니다.
- 기존 checkpoint를 같은 정의로 취급해 재개하지 마세요. observation 의미와 reward/파지 기준이 바뀌었습니다.
- 기존 LeRobot dataset에 새 TCP 정의를 섞어 append하는 것은 거부합니다. 새 dataset 경로를 사용하세요.
- 사용자 지정 파일은 프로세스 시작 전에 `KUAVO_GRASP_REFERENCE_POINTS=/absolute/path/points.json`으로 선택합니다.
- 패키지 배포 시 새 보정값을 기본값으로 만들려면 루트 configs와 package configs를 함께 갱신합니다.
- `robots/end_effector.py`가 공통 정의·기구학·tensor API, `rl/debug/endeffector_markers.py`가 표시입니다.
- 현재 검증은 CPU 기구학/미분 Jacobian 테스트와 문법 검사이며, 실제 VR/물리 테스트는 아직 하지 않았습니다.

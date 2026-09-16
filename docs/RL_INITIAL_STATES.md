# RL 모델별 초기 자세

전용 `train_flap_pick.sh`와 `play_flap_pick.sh`는
`configs/rl_pick_arms_only.py`에서 모델별 `INITIAL_STATE`를 선택한다.
기본 S63 + `leju-twofinger`는 `s63_leju_ready_01`을 사용한다. 이 자세는
시뮬레이션용으로 준비한 값이며 실물 또는 VR 측정값이 아니다.
S200062를 명시적으로 선택하면 기존 측정 자세 `quest_ready_02`를 사용한다.
S56 + `s56_twofinger`는 별도 시뮬레이션용 `s56_twofinger_ready_01`을 사용한다.
이 자세는 fixed root 조건이며 지면에서 균형을 잡는 보행 자세가 아니다.
학습 실행법과 task 설정은 [flap 파지 학습](RL_FLAP_PICK.md)에 있다.

## S200062 base·torso 재현

`s63_leju_ready_01`은 `quest_ready_02`의 base `root_pose`를 그대로 저장한다.
따라서 S63의 arms-only/whole-body pick 및 pick_place에서도 reset 마지막에
저장된 위치·방향을 복원하고, 박스 접근 위치로 자동 배치한 값을 덮어쓴다.
base XY는 −0.240912810 / +0.066121511 m이며 전체 quaternion도 함께 복사했다.

Torso 기준은 팔이 붙는 `waist_yaw_link`의 위치·방향이다. S63는 하부 고정
프레임과 waist 장착 offset이 달라서, S200062 관절각을 그대로 사용하면
torso가 X −4.69 mm, Z +45.20 mm 어긋난다. base를 유지하고 S63의 세 pitch
관절을 역기구학으로 계산해 같은 torso pose가 되도록 설정했다.

| S63 관절 | 초기 각도 (rad) |
|---|---:|
| knee_joint | 0.23065070098376964 |
| leg_joint | −0.47266920953812963 |
| waist_pitch_joint | 0.26300891006078403 |
| waist_yaw_joint | −0.00017328046669717878 |

S63 팔·머리 초기 관절값과 gripper 설정은 유지한다. 이 값은 S200062 저장
자세를 시뮬레이션에서 재현한 결과이며, S63 실물 측정 자세는 아니다.
원본 `quest_ready_02`는 변경하지 않는다. 모델별 torso FK 일치와 S63 관절
제한을 검사하며, headless 2환경에서 실제 reset 후 root와 torso pose를 검증했다.

## 직접 수정

`configs/initial_states.json`의 `states.s63_leju_ready_01.assets.robot`을 수정한다.
S200062에서는 `states.quest_ready_02.assets.robot`을 수정한다.

- `joint_positions`: 관절 이름별 각도, 단위 rad. 예: `zhead_1_joint`,
  `zhead_2_joint`, `waist_pitch_joint`, `waist_yaw_joint`, `zarm_l1_joint`.
- `root_pose`: `[x, y, z, qw, qx, qy, qz]`. 위치는 env origin 기준 m,
  회전은 정규화된 quaternion(wxyz). 바퀴 각도로 base 위치를 추정하지 않는다.
- `robot_model`과 `gripper`는 실제 캡처 모델을 유지한다. 다른 모델용 자세를
  이름만 바꿔서 재사용하지 않는다.

각 실행 시작 시 파일을 읽는다. 실행 중 JSON을 수정해도 이미 시작한 학습에는
반영되지 않으므로 다음 실행부터 적용한다. 실행별 manifest에 사용한 값이 남는다.
패키지 배포본에도 반영하려면 `src/kuavo_isaaclab_scene/configs/initial_states.json`을
동일하게 갱신한다. checkout에서는 루트 `configs` 파일을 사용한다.

## VR에서 다시 캡처

VS Code에서 실행 중인 teleop의 `env.step(action)` 다음 줄에 중단점을 걸고,
`env`가 있는 프레임의 Debug Console에서 실행한다. 이 작업은 파일을 덮어쓴다.

```python
from kuavo_isaaclab_scene.robots.initial_states import capture_initial_state
capture_initial_state(
    env, "s63_leju_ready_01",
    path="/absolute/path/to/kuavo_isaaclab_scene/configs/initial_states.json",
    overwrite=True,
)
```

실제 측정한 모든 robot joint와 root pose를 저장한다. 별도 articulation인
그리퍼가 있으면 함께 저장한다. 다른 이름의 프리셋은 보존된다.
기본 캡처에는 박스 pose가 없으며, 이번 실험은 캡처 배치
`configs/rack_box_poses.json`을 별도로 사용한다.

이전 자세를 보관하려면 새 이름으로 캡처하고 `INITIAL_STATE`를 바꾼다.
일반 `train_rl.sh`/`play_rl.sh`에서는 `--initial-state NAME`과
`--initial-states-file JSON`도 지원하지만 config가 이름을 고정했다면 일치해야 한다.
`--reset-bank`와 초기 자세는 동시에 사용할 수 없다.

## 복원과 고정의 범위

reset 이벤트 마지막에 프리셋을 적용하고 속도와 PD 속도 목표를 0으로 만든다.
그 다음 arms-only action이 몸통 관절을 고정하고 팔/손 목표를 초기화한다.
base는 fixed root이고 wheel/knee/leg/waist/head는 초기 각도 주위 ±1e-4 rad의
물리 limit와 PD 목표로 제한한다. solver 오차까지 0이라고 보장하지 않는다.
팔과 그리퍼는 정책이 움직인다. 관절 이름, 모델, 실제 물리 limit를 검사하며
측정의 미세한 limit 초과(1e-4 rad 이하)만 경계로 보정한다.

초기화는 정지 상태 복원이다. 접촉 cache, 속도, 마찰, 배치 scale, 학습 진행
상태를 저장하는 snapshot이 아니며 충돌/도달 가능성을 자동 검증하지 않는다.

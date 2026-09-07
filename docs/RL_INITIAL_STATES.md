# RL 초기 자세: quest_ready_02

전용 `train_flap_pick.sh`와 `play_flap_pick.sh`는
`configs/rl_pick_arms_only.py`의 `INITIAL_STATE = "quest_ready_02"`를 사용한다.
학습 실행법과 task 설정은 [flap 파지 학습](RL_FLAP_PICK.md)에 있다.

## 직접 수정

`configs/initial_states.json`의 `states.quest_ready_02.assets.robot`을 수정한다.

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
    env, "quest_ready_02",
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

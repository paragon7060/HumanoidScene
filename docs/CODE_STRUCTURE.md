# 코드 구조와 개발 위치

Python 코드는 `src/kuavo_isaaclab_scene/` 아래에서 기능별로 나눈다.
이 문서의 Python 경로는 모두 해당 패키지를 기준으로 한다.
폴더 구분은 코드의 역할 구분이며, USD의 prim 계층이나 `scene["rack"]` 같은
Isaac Lab scene key를 변경하지 않는다.

## 1. 기능별 위치

| 폴더 | 책임 | 주요 파일 |
|---|---|---|
| `envs/` | Scene 생성, manager 환경, MDP, 공통 물리 설정 | `scene.py`, `manager_env.py`, `manager_mdp.py`, `teleop_env.py`, `scene_physics.py`, `contact_physics.py`, `task_system.py` |
| `rl/` | 하위 task별 manager 환경, PPO 학습/평가 | `tasks/`, `managers/`, `mdp/`, `agents/`, `runners/` |
| `teleop/` | Meta Quest 연결, tracking → action 변환, 안전 제어, 수집 실행 | `quest_openxr.py`, `quest_runtime.py`, `teleop_mapping.py`, `teleop_body_action.py`, `collect_quest_teleop.py` |
| `robots/` | Robot/gripper asset 선택과 설정, 관성, linkage, 물리 카메라 장착 위치 | `robot_model.py`, `gripper_config.py`, `gripper_runtime.py`, `robot_inertials.py`, `twofinger_linkage.py`, `wrist_camera_mount.py` |
| `workcell/` | Rack/box 배치, 좌표 변환, pose 캡처, rack–conveyor 간격, flap 마찰 설정 | `workcell_layout.py`, `rack_box_layout.py`, `capture_layout.py`, `capture_rack_box_poses.py`, `workcell_gap.py`, `box_flap_friction.py` |
| `display/` | 카메라 프레임 추출, 화면 표시, stereo 보정/합성, XR panel, 영상 저장 | `camera_frames.py`, `camera_viewports.py`, `stereo_camera_calibration.py`, `stereo_compositor.py`, `xr_camera_overlay.py`, `xr_control_status.py`, `eval_video.py` |
| `recording/` | HDF5/LeRobot 데이터 기록과 별도 writer process | `teleop_recorder.py`, `teleop_lerobot_recorder.py`, `lerobot_writer_worker.py` |
| `evaluation/` | GR00T action 변환, online/offline 평가, metrics, 별도 policy process | `eval_groot.py`, `groot_lerobot_bridge.py`, `groot_policy_worker.py`, `offline_eval_groot.py`, `offline_policy_eval.py`, `eval_metrics.py` |
| `core/` | 공통 경로와 Isaac Sim/Lab runtime 버전 검사 | `paths.py`, `runtime_compat.py` |
| `assets/` | 실제 배포하는 USD/URDF/mesh/texture 원본 및 파생 asset | 기존 파일 경로 유지 |
| `configs/` | wheel에 포함되는 기본 JSON 설정 | `grippers.json`, `workcell_layout.json`, `rack_box_poses.json` |

패키지 루트의 `__init__.py`에는 Gym 환경 등록만 둔다. 새 하위 패키지의
`__init__.py`에서는 runtime 모듈을 자동으로 import하지 않는다.

## 2. 어떤 파일을 수정해야 하나?

### Scene / RL 환경

- 배치 확인용 standalone scene과 직접 simulation loop: [`envs/scene.py`](../src/kuavo_isaaclab_scene/envs/scene.py)
- Manager-based scene/actuator/action/observation/reward 설정: [`envs/manager_env.py`](../src/kuavo_isaaclab_scene/envs/manager_env.py)
- Reset, randomization, observation, reward, termination 함수: [`envs/manager_mdp.py`](../src/kuavo_isaaclab_scene/envs/manager_mdp.py)
- Manager 환경 실행 예제: [`envs/run_manager_env.py`](../src/kuavo_isaaclab_scene/envs/run_manager_env.py)
- 공통 robot/box 물리 설정: [`envs/scene_physics.py`](../src/kuavo_isaaclab_scene/envs/scene_physics.py), [`envs/contact_physics.py`](../src/kuavo_isaaclab_scene/envs/contact_physics.py)
- Standalone conveyor/button task 상태 전이: [`envs/task_system.py`](../src/kuavo_isaaclab_scene/envs/task_system.py)

새 하위 task별 RL 작업은 `rl/tasks/`, `rl/managers/`, `rl/mdp/`에서 시작한다.
[RL 학습 가이드](RL_TRAINING.md)에 각 manager의 수정 위치와 PPO 실행법이 있다.
기존 robustness 환경은 `envs/manager_env.py`와 `envs/manager_mdp.py`를 유지하며,
standalone도 별도 실행 방식이다.

### Meta Quest teleop

- OpenXR 수집 진입점: `teleop/collect_quest_teleop.py`
- 로컬/브라우저 preview: `teleop/preview_quest_local.py`, `teleop/preview_quest_browser.py`
- WebSocket tracking protocol 및 browser action 조합: `teleop/browser_teleop_bridge.py`, `teleop/browser_teleop_control.py`
- OpenXR adapter / runtime 점검: `teleop/quest_openxr.py`, `teleop/quest_runtime.py`
- 손·팔·머리·몸통 mapping: `teleop/teleop_mapping.py`, `teleop/teleop_hand_mode.py`, `teleop/teleop_body.py`
- IK / body action term: `teleop/teleop_ik.py`, `teleop/teleop_body_action.py`
- Tracking 손실 안전 처리: `teleop/teleop_safety.py`
- 수집 scene 단순화 / 사용자 scene 설정: `teleop/teleop_scene.py`, `teleop/teleop_scene_config.py`

Quest의 **Isaac Lab 환경 설정**은 [`envs/teleop_env.py`](../src/kuavo_isaaclab_scene/envs/teleop_env.py)에
있다. 입력 장치 처리는 `teleop/`, 데이터 기록은 `recording/`, 화면 표시는
`display/`로 분리한다. CloudXR의 TypeScript bridge는 기존
`integrations/cloudxr/`에 유지한다.

### Asset과 카메라

원본 USD나 URDF를 바꾸는 작업은 `assets/`, 로봇 모델/손 선택과 runtime 설정은
`robots/`, rack/box 배치 설정은 `workcell/`에서 한다. Hand/box collision의
공통 USD spawn override는 `envs/contact_physics.py`에 있다.

카메라 **장착 위치**는 `robots/wrist_camera_mount.py`와 `robots/robot_model.py`,
Isaac Lab **sensor 생성 설정**은 `envs/`, 카메라 **프레임 처리/표시**는
`display/`에서 관리한다.

저장소 루트의 `configs/`는 편집하는 배포 설정이고, 패키지 안의 `configs/`는
wheel용 fallback이다. `scripts/build_wheel.sh`는 배포 설정을 fallback에 복사한
뒤 패키징한다. Asset 변환·설치 도구는 기존 `scripts/`에 둔다.

## 3. 실행법과 Python 경로 변경

기존 루트 shell 명령과 CLI 인자는 유지한다.

```bash
./run_scene.sh --robot-model s63
./run_manager_env.sh --num-envs 1 --steps 240
./preview_quest_local.sh --robot-model s200062
./collect_quest_teleop.sh --xr-runtime-json /path/to/openxr_cloudxr.json
./eval_groot.sh --help
```

Shell launcher를 쓰면 Python 환경 선택, `PYTHONPATH`, 기존 startup 옵션이 함께
설정된다. 직접 `python -m`을 사용할 경우 설치된 패키지 또는 `PYTHONPATH=src`가
필요하고, 모듈명은 다음 새 경로를 사용한다.

| 루트 launcher | Python 모듈 (`python -m` 뒤) |
|---|---|
| `run_scene.sh` | `kuavo_isaaclab_scene.envs.scene` |
| `run_manager_env.sh` | `kuavo_isaaclab_scene.envs.run_manager_env` |
| `preview_quest_local.sh` | `kuavo_isaaclab_scene.teleop.preview_quest_local` |
| `preview_quest_browser.sh` | `kuavo_isaaclab_scene.teleop.preview_quest_browser` |
| `collect_quest_teleop.sh` | `kuavo_isaaclab_scene.teleop.collect_quest_teleop` |
| `capture_layout.sh` | `kuavo_isaaclab_scene.workcell.capture_layout` |
| `capture_rack_box_poses.sh` | `kuavo_isaaclab_scene.workcell.capture_rack_box_poses` |
| `set_workcell_gap.sh` | `kuavo_isaaclab_scene.workcell.workcell_gap` |
| `eval_groot.sh` | `kuavo_isaaclab_scene.evaluation.eval_groot` |
| `offline_eval_groot.sh` | `kuavo_isaaclab_scene.evaluation.offline_eval_groot` |
| `doctor.sh` | `kuavo_isaaclab_scene.core.runtime_compat` |
| `quest_doctor.sh` | `kuavo_isaaclab_scene.teleop.quest_runtime` |

사용자 Python 코드의 import도 새 경로로 변경한다. 예를 들어:

```python
# 기존: from kuavo_isaaclab_scene.manager_env import KuavoRobustWorkcellEnvCfg
# Isaac-dependent config는 AppLauncher 초기화 이후 import한다.
from kuavo_isaaclab_scene.envs.manager_env import KuavoRobustWorkcellEnvCfg
from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.core.paths import ASSET_DIR
```

이전 flat Python 모듈 경로용 alias/wrapper는 두지 않는다. 별도로 저장한
`--scene-config` Python 파일, notebook, 외부 실행 스크립트가 이전 경로를
import한다면 해당 경로도 바꿔야 한다. 대부분 기존 파일명 앞에 위 표의 폴더만
추가하며, 역할을 명확히 하려고 이름도 바꾼 파일은 다음 세 개다.

| 이전 파일명 | 새 패키지 상대 경로 |
|---|---|
| `teleop_contacts.py` | `envs/contact_physics.py` |
| `teleop_inertials.py` | `robots/robot_inertials.py` |
| `teleop_camera.py` | `display/camera_frames.py` |

Gym ID는 그대로다.

- `Isaac-Kuavo-RobustWorkcell-v0` → `kuavo_isaaclab_scene.envs.manager_env:KuavoRobustWorkcellEnvCfg`
- `Isaac-Kuavo-QuestTeleop-v0` → `kuavo_isaaclab_scene.envs.teleop_env:KuavoQuestTeleopEnvCfg`

## 4. 경로와 import 규칙

- Asset/config 경로는 `core.paths`를 사용한다. 각 파일에서 `__file__.parents[n]`으로
  패키지 루트를 따로 추측하지 않는다.
- `KUAVO_CONFIG_DIR`, `KUAVO_WORKCELL_LAYOUT`, `KUAVO_RACK_BOX_POSES`의 기존 동작은 유지한다.
- 평가 결과의 기본 위치는 checkout에서는 저장소의 `artifacts/`, wheel 설치에서는
  실행 작업 폴더의 `artifacts/`다. 명시적인 출력 CLI 경로가 있으면 그것을 사용한다.
- 같은 하위 패키지끼리는 `.module`, 다른 기능 패키지에서는 `..group.module`로
  import한다. 외부 코드에서는 완전한 `kuavo_isaaclab_scene.group.module` 경로를 쓴다.
- `envs/scene.py` 같은 실행 모듈을 공용 API로 import하지 않는다. CLI parsing과
  AppLauncher 실행이 있는 모듈은 launcher를 통해 실행한다.
- 하위 패키지의 `__init__.py`에는 scene 생성이나 Isaac/Omniverse 모듈의 eager import를
  추가하지 않는다. CLI 설정 적용 → AppLauncher → Isaac-dependent import 순서를 지킨다.
- GR00T worker는 `evaluation.groot_policy_worker` 모듈을 별도 Python으로 실행하며,
  `core.paths.PACKAGE_IMPORT_ROOT`를 자식 process의 `PYTHONPATH`에 전달한다.
- LeRobot writer는 `recording/teleop_lerobot_recorder.py`와 같은 폴더의
  `lerobot_writer_worker.py`를 `python -I`로 직접 실행한다. 이 worker에 패키지 상대
  import를 추가하면 isolated 실행이 깨지므로 독립 실행을 유지한다.

폴더 변경 시 실행기, Gym 등록 문자열, worker 실행 경로, 테스트 import, 문서의
경로를 같이 갱신한다. 단위 테스트는 기존 `tests/`에 유지한다.

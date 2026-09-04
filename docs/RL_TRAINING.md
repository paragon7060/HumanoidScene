# 하위 task별 Manager-based RL 학습

새 학습 코드는 `src/kuavo_isaaclab_scene/rl/`에 있다. 기존 standalone,
Quest teleop, GR00T 평가의 환경 설정을 덮어쓰지 않고, 공통 workcell asset만
재사용한다. 환경은 `isaaclab.envs.ManagerBasedRLEnv`, 학습기는 RSL-RL PPO다.
구성 API는 저장소의 Isaac Lab v2.3.2와
[공식 RSL-RL 실행 예제](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2/scripts/reinforcement_learning/rsl_rl)를 기준으로 한다.

## 1. 구현 범위와 전제

- 기본 robot/hand: `s200062` + `s200062_integrated`. `s56` + `s56_twofinger`도
  같은 접촉 센서/손 preset을 사용한다. S63·Allegro·QiangNao 등 다른 손은
  finger/tool sensor와 action 설정을 먼저 추가해야 하며 현재는 명시적으로 거부한다.
- 이동은 **고정 root의 평면 XY/yaw 이동 추상화**다. 팔·허리·박스·접촉은 물리
  시뮬레이션을 사용하지만 실제 wheel dynamics나 이족 보행을 학습하는 것은 아니다.
  바퀴 회전 animation도 RL planar action에서는 별도로 구동하지 않는다.
- 기본 actor는 joint/object/contact/cargo 상태를 보는 **state-based policy**다.
  이미지 기반 policy나 sim-to-real 성능을 의미하지 않는다.
- 박스를 손에 붙이는 fixed joint, suction, 강제 box pose 이동으로 grasp를 대신하지 않는다.
- Robot self-collision은 기존 asset 설정을 유지한다. 현재 기본 모델은 self-collision을
  끄므로 실제 안전 동작 검증 전에는 linkage별 collision filter와 self-collision도 보완해야 한다.
- 컨베이어는 유효 버튼 입력 후 지지된 박스에 진행 방향 속도를 주는 추상화다.
  실제 conveyor surface-velocity physics extension은 아니다.
- 기본 RL scene에서는 factory 배경, RTX 카메라, 기존 legacy tote/cargo, 이동 AMR을
  제외하고 선택한 실제 USD 박스와 그 내부의 자유 물체를 사용한다. 기존 scene/Quest의
  해당 기능은 그대로 유지된다. 카메라는 `--enable_cameras`로 추가할 수 있지만
  PPO observation에 이미지를 넣어주지는 않는다.
- 이 변경에서 GPU rollout/학습은 실행하지 않았다. 성공률 80%, 3개/11초 성능은
  아직 검증된 결과가 아니며 reset·grasp·reward 튜닝과 별도 평가가 필요하다.

## 2. 학습할 task 선택

| `--task` | 목표 / 성공 조건 | 초기 상태 |
|---|---|---|
| `approach_rack` | 목표 box 앞까지 이동하고 heading 정렬 | 원래 robot 위치 주변 XY/yaw randomization |
| `pick` | 양쪽 opposing finger 접촉이 있는 손으로 box를 기준 높이보다 들어 올리고 유지 | Rack 앞; 또는 선택적으로 approach 성공 bank |
| `carry` | Grasp·lift·upright를 유지하면서 빈 conveyor slot 앞으로 이동·회전 | **pick 성공 bank 필수** |
| `place` | Conveyor 내부에 box 전체 footprint를 지지시키고, 다른 box와 겹치지 않게 놓고 손을 뗀 뒤 안정화 | **carry 성공 bank 필수** |
| `press_button` | 선택한 모든 box가 conveyor에 있고 실제 ButtonJoint를 누른 뒤 belt 작동 시간까지 유지 | **place 성공 bank 필수** |
| `full` | 각 box에 대해 approach → pick → carry → place를 반복하고 마지막에 button | 원래 rack 배치; 한 개의 phase-conditioned policy 학습 |

조건은 기본 0.3초 연속 충족해야 한다. 기본 lift 기준은 0.10 m, 목표 heading
오차는 0.18 rad, navigation 오차는 0.10 m다. 내부 cargo 이탈, floor drop,
몸통/팔의 큰 접촉력, workspace 이탈은 실패로 처리한다. 주요 값은 `tasks/specs.py`다.

기본 action 순서는 base XY/yaw 3 → waist yaw/양팔 15 → height 3(S200062만)
→ 좌/우 gripper 각 1이다. S200062는 23-D, S56은 20-D다. 입력은 `[-1, 1]`로
제한하며, base는 속도, 관절은 control step당 목표 증분, gripper는 open/close
목표 증분이다. 손 action의 양수는 열기, 음수는 닫기이고 0은 목표 유지다.
팔·허리 목표도 누적 후 유지하며, physics decimation마다 증분을 중복 적용하지 않는다.
기존 Quest/GR00T action schema와 같지 않으므로 checkpoint/action을 그대로 섞지 않는다.

Dense reward는 초당 reward rate이며, `stage_completed`, `success`, `failure`는
일회성 bonus/penalty다. 후자는 Isaac Lab RewardManager의 `dt` 곱을 상쇄하므로
설정 weight 자체가 해당 step에 지급된다. 목표 직전에 멈춰 dense reward만 얻는
행동이 유리해지지 않도록 이 차이를 고려해 weight를 조절한다.

`pick`의 기본값은 **한 손의 두 opposing finger**가 실제 box Body에 접촉하는 것이다.
두 손 grasp를 요구하려면 `required_grasp_hands=2`로 바꾼다. `full`은 박스를
하나씩 처리한다. **동시에 2~3개를 집는 multi-object grasp policy는 아직 포함하지 않는다.**

`full`은 별도로 학습하는 단일 policy이며, 학습한 5개 checkpoint를 자동으로
이어 붙이는 hierarchical policy executor가 아니다. 각 skill을 튜닝하고
reset distribution을 확보한 뒤 전체 task 학습에 활용하는 출발 구조다.

## 3. 설치와 첫 학습

기존 Isaac Sim 5.1.0 / Isaac Lab v2.3.2 환경을 사용한다. 새 conda 환경은 필요 없다.

```bash
cd /absolute/path/to/HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
python -m pip install -e '.[rl]'
```

RL extra는 `rsl-rl-lib==3.1.2`와 TensorBoard를 설치한다. `isaaclab_rl`이 없다면
먼저 기존 Isaac Lab 설치를 완성하거나 같은 tag의 확장을 설치한다.

```bash
python -m pip install -e .external/IsaacLab-v2.3.2/source/isaaclab_rl
```

먼저 작은 병렬 수로 시작한다. 학습 실행기는 기본 8개지만, PC 메모리/VRAM 문제가
있었다면 아래처럼 2개부터 늘리는 편이 안전하다. Headless에서도 PhysX/GPU 메모리는 사용한다.

```bash
./train_rl.sh --task approach_rack \
  --robot-model s200062 --gripper s200062_integrated \
  --boxes small_box_0 --num-envs 2 --headless --max-iterations 2000
```

기본 배포 `configs/rack_box_poses.json`의 `small_box_0`를 사용한다. 선택한 box가
rack에 배치된 것으로 정의되어 있지 않으면 실행 전에 오류를 낸다.
캡처 대신 자동 배치를 쓰려면 기존 옵션을 그대로 사용한다.

```bash
./train_rl.sh --task pick --boxes small_box_0 \
  --rack-boxes '2:small' --ignore-captured-box-poses \
  --num-envs 2 --headless
```

목적별 실행기는 `train_rl.sh`와 `play_rl.sh`이고, Python 진입점은 각각
`kuavo_isaaclab_scene.rl.runners.train`, `kuavo_isaaclab_scene.rl.runners.play`다.
새 task들은 이 실행기에서 `ManagerBasedRLEnv`로 직접 구성한다. 기존 Gym ID와
`run_manager_env.sh`는 기존 robustness 환경을 계속 가리킨다.

## 4. 성공 상태를 다음 단계로 넘기기

처음에는 **한 box**에 대해 다음 순서로 학습하는 것이 단순하다. 아래 명령은
동시에 실행하지 않는다. 다음 단계로 넘어가기 전에 bank에 실제 성공 JSON이
생겼는지 확인한다. 성공이 없다면 앞 단계의 reward/reset/contact를 먼저 수정한다.

```bash
# 1) box 집기: approach 정책 없이도 rack 앞 reset으로 학습 가능
./train_rl.sh --task pick --num-envs 2 --headless \
  --snapshot-dir artifacts/rl_banks/pick

# 2) 집은 상태로 이동·회전
./train_rl.sh --task carry --num-envs 2 --headless \
  --reset-bank artifacts/rl_banks/pick \
  --snapshot-dir artifacts/rl_banks/carry

# 3) 내려놓기와 손 떼기
./train_rl.sh --task place --num-envs 2 --headless \
  --reset-bank artifacts/rl_banks/carry \
  --snapshot-dir artifacts/rl_banks/place

# 4) 모든 box 배치 완료 후 버튼 누르기
./train_rl.sh --task press_button --num-envs 2 --headless \
  --reset-bank artifacts/rl_banks/place
```

이 명령들은 모두 같은 기본 box/model/layout/prefill/cargo 설정을 사용한다.
처음 명령에 `--robot-model`, `--boxes`, `--rack-box-poses`, `--prefill`, `--cargo-per-box`,
`--config` 등을 추가했다면 이후 단계에도 동일한 환경 설정을 사용한다.

Bank는 scene의 root pose/velocity, joint position/velocity를 env-local 좌표로 저장한다.
물리 엔진의 contact solver cache 자체를 저장하지는 않는다. 복원 직후에는
접촉을 다시 계산하며, 회복되지 않는 grasp는 성공으로 인정되지 않는다.
관절/box/배치/크기 등 호환성 hash와 predecessor task를 검사하므로 서로 다른
실험의 state를 잘못 섞어 쓰지 않는다. 파일은 JSON이며 episode 자동 reset
**이전**에 저장한다. `--max-snapshots` 기본값은 실행당 100개다.

`approach_rack`의 성공 상태도 저장할 수 있다. `pick --reset-bank .../approach`
사용은 선택 사항이다. `carry/place/press_button`에는 bank를 생략할 수 없다.

## 5. 평가 / 학습 재개

저장 경로는 `artifacts/rl/<task>/train_<timestamp>_<id>/`이고, 그 아래에 다음이 생긴다.

- `model_*.pt`: PPO checkpoint
- `manifest.json`: robot/box/layout/action/observation 호환성 정보
- `env.yaml`, `agent.yaml`: 실제 실행 설정
- TensorBoard event 파일

아래의 `/absolute/path/to/model_2000.pt`는 실제로 생성된 checkpoint로 바꾼다.
`manifest.json`도 checkpoint 옆에 있어야 한다. PyTorch checkpoint는 신뢰하는 파일만 로드한다.

```bash
# 화면을 보며 pick 평가, 성공 상태도 추가 수집
./play_rl.sh --task pick --num-envs 1 --episodes 20 \
  --checkpoint /absolute/path/to/model_2000.pt \
  --snapshot-dir artifacts/rl_banks/pick

# carry는 평가에도 동일한 grasp 초기 상태가 필요
./play_rl.sh --task carry --headless --num-envs 1 --episodes 100 \
  --checkpoint /absolute/path/to/carry/model_2000.pt \
  --reset-bank artifacts/rl_banks/pick

# 같은 task 학습을 이어서 실행
./train_rl.sh --task pick --headless --num-envs 2 \
  --checkpoint /absolute/path/to/pick/model_2000.pt --max-iterations 1000

tensorboard --logdir artifacts/rl
```

평가 결과는 별도 `play_<timestamp>_<id>/metrics.json`에 성공률과 episode별 시간,
실패 여부로 저장된다. 이전 학습을 재개할 때도 새 로그 폴더를 만들므로 기존
기록을 덮어쓰지 않는다. `--no-randomization`은 환경 randomization과 observation
noise를 끈다. Checkpoint의 action 의미/차원과 환경 구성이 다르면 거부한다.

## 6. 여러 box / 전체 task

```bash
./train_rl.sh --task full \
  --boxes small_box_0,small_box_1,medium_box_0 \
  --prefill 1 --slots 4 --slot-pitch 0.52 \
  --num-envs 2 --headless --max-iterations 5000 \
  --snapshot-dir artifacts/rl_banks/full
```

처리 순서는 `--boxes` 순서다. 지정하지 않은 box는 RL scene에서 제외한다.
`--boxes all`은 캡처/설정된 rack box를 전부 선택하며, 수용 공간이 충분해야 한다.

Conveyor 후보 slot은 실제 surface의 로컬 X축 기준이다. Box의 회전된 footprint와
이미 놓인 box/prefill을 검사하여 빈 후보를 고른다. 직접 다른 box를 미는 물리
상호작용은 가능하지만 **밀기 전용 skill/reward는 없다**. Slot 수만 늘려 실제
공간 부족을 해결할 수는 없다. `--slots`, `--slot-pitch`, box 크기와 layout을 함께 조절한다.

`full --snapshot-dir`은 단계 성공 상태를 `approach_rack/`, `pick/`, `carry/`,
`place/` 하위 폴더로 저장한다. `place/`에는 모든 선택 box를 배치한 상태만 저장한다.
이 폴더를 같은 여러-box 환경의 `press_button --reset-bank`로 사용할 수 있다.
단일-box bank는 다른 여러-box 환경으로 옮겨 쓸 수 없다.

## 7. 무엇을 어느 파일에서 고치나?

아래 경로는 `src/kuavo_isaaclab_scene/rl/` 기준이다.

| 고칠 내용 | 설정 파일 | 실제 계산/동작 구현 |
|---|---|---|
| Task 시간 제한, 거리/각도/lift/접촉 threshold, cargo/prefill | [`tasks/specs.py`](../src/kuavo_isaaclab_scene/rl/tasks/specs.py) | `mdp/commands.py` |
| Scene asset 선택, sensor 추가, cargo 크기/배치 | [`tasks/scene_cfg.py`](../src/kuavo_isaaclab_scene/rl/tasks/scene_cfg.py) | `mdp/events.py` |
| 모든 manager 조립, simulation dt/decimation | [`tasks/env_cfg.py`](../src/kuavo_isaaclab_scene/rl/tasks/env_cfg.py) | `ManagerBasedRLEnv` |
| 팔/허리 action joint 순서·scale, base 속도/가속도, 손 속도 | [`managers/actions.py`](../src/kuavo_isaaclab_scene/rl/managers/actions.py) | [`mdp/actions.py`](../src/kuavo_isaaclab_scene/rl/mdp/actions.py) |
| Observation 구성·noise·scale | [`managers/observations.py`](../src/kuavo_isaaclab_scene/rl/managers/observations.py) | `mdp/observations.py` |
| 목표 위치, 접촉 body, phase 관리 | [`managers/commands.py`](../src/kuavo_isaaclab_scene/rl/managers/commands.py) | [`mdp/commands.py`](../src/kuavo_isaaclab_scene/rl/mdp/commands.py) |
| Reward weight / 항목 활성화 | [`managers/rewards.py`](../src/kuavo_isaaclab_scene/rl/managers/rewards.py) | [`mdp/rewards.py`](../src/kuavo_isaaclab_scene/rl/mdp/rewards.py) |
| Reset, 물성 randomization, conveyor 진행 | [`managers/events.py`](../src/kuavo_isaaclab_scene/rl/managers/events.py) | `mdp/events.py` |
| 성공·실패·timeout 종료 항목 | [`managers/terminations.py`](../src/kuavo_isaaclab_scene/rl/managers/terminations.py) | `mdp/terminations.py`; 물리 predicate는 `mdp/commands.py` |
| Reset randomization 증가 속도 | [`managers/curriculum.py`](../src/kuavo_isaaclab_scene/rl/managers/curriculum.py) | `mdp/curriculum.py` |
| 성공 상태 저장 / reset bank schema | [`managers/recorders.py`](../src/kuavo_isaaclab_scene/rl/managers/recorders.py) | `mdp/reset_bank.py` |
| PPO network, learning rate, horizon, batch, entropy | [`agents/ppo_cfg.py`](../src/kuavo_isaaclab_scene/rl/agents/ppo_cfg.py) | `runners/train.py` |
| CLI, 학습 재개 검사, 평가 loop | [`runners/common.py`](../src/kuavo_isaaclab_scene/rl/runners/common.py) | `runners/train.py`, `runners/play.py` |

현재 root frame 고정 이동을 실제 mobile-base controller로 바꾸려면
`mdp/actions.py`의 `PlanarDrive`부터 교체한다. 이 경우 free-base asset, 바퀴/다리
actuator와 collision, balance reward/termination도 함께 설계해야 한다.

### 실행별 설정 파일 사용

공통 파일을 매번 수정하지 않으려면 [`configs/rl_example.py`](../configs/rl_example.py)를
복사해 실험별로 관리한다.

```bash
./train_rl.sh --task pick --config configs/rl_example.py --headless --num-envs 2
```

```python
from dataclasses import replace

def configure_task(spec):
    # scene를 만들기 전에 task 조건 변경
    return replace(spec, lift_height=0.12, required_grasp_hands=2,
                   hold_seconds=0.5, episode_length_s=20.0)

def configure(env_cfg, agent_cfg):
    # scene 구성 후 manager/학습기 설정 변경
    env_cfg.rewards.reaching.weight = 3.0
    env_cfg.actions.base.velocity_limits = (0.20, 0.20, 0.60)
    env_cfg.events.arm_mass = None
    agent_cfg.algorithm.learning_rate = 2e-4
```

이 파일은 임의 Python 코드가 실행되는 **신뢰된 로컬 설정 파일**이다.
`configure_task()`에서 task를 바꾸고 `configure()`에서 manager를 조절한다.
Action scale/순서를 바꾼 checkpoint는 이전 action 의미와 호환되지 않으므로
새 실험으로 학습한다. Reward weight 변경은 같은 action schema에서 재개 가능하다.

## 8. 직접 확인할 순서

1. `--num-envs 1`로 robot/box/cargo가 올바르게 시작하는지 확인한다.
2. Body bounds로 만든 grasp 목표와 실제 finger/tool frame이 맞는지 확인한다.
3. 먼저 `approach_rack`, 다음 `pick`을 조정한다. Bank 없는 carry/place를 공중에
   box만 spawn하는 방식으로 우회하지 않는다.
4. 짧은 성공 기록으로 reset bank 복원 시 실제 grasp가 유지되는지 확인한다.
5. `play_rl.sh --episodes 100` 등으로 고정 seed/설정을 기록하고 평가한다.
6. 그 뒤 box 종류·수, randomization, 이동 장애물, 이미지 observation을 확장한다.

선택하지 않은 box·AMR을 제외한 초기 학습 환경의 결과를 곧바로 전체 공장
robustness로 해석하지 않는다. 성공률과 처리 시간은 평가 결과로 판단한다.

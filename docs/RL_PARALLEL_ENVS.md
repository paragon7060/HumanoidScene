# 배경 없는 RL 전용 병렬 환경

RL 환경은 일반 scene/VR 환경과 별도로 구성한다. Factory 배경, 돌아다니는
사람·로봇, 사용하지 않는 staging box와 legacy tote는 학습 scene에 생성하지 않는다.
로봇, 작업 대상 박스, rack, 버튼·fence와 필요한 충돌 지오메트리만 사용한다.
일반 scene과 Meta Quest teleop 환경의 구성은 이 변경으로 바꾸지 않는다.

이 구현은 **ManagerBasedRLEnv + InteractiveScene의 단일 프로세스 벡터 환경**이다.
`--num-envs`개 작업 공간을 GPU에서 함께 step하고 PPO batch를 수집한다.
프로세스를 여러 개 실행하거나 GPU 여러 장을 사용하는 분산 학습 옵션은 아니다.

## 코드 구분

아래 경로는 프로젝트 루트 기준이다. Python package 아래 구조는 다음과 같다.

```text
src/kuavo_isaaclab_scene/
├── envs/                       # 일반 작업 환경 / teleop용 환경
├── teleop/                     # Meta Quest 연결과 조작
└── rl/
    ├── envs/
    │   ├── env_cfg.py          # RL 전용 manager-based 환경 조립
    │   └── parallel_cfg.py     # 환경 간격·복제·충돌 격리 설정
    ├── scenes/
    │   ├── scene_cfg.py        # 독립적인 RL InteractiveSceneCfg 조립
    │   ├── robot.py            # 학습 로봇과 gripper
    │   ├── workcell.py         # rack, 버튼, fence, 단순 conveyor 지지체
    │   ├── boxes.py            # task에서 사용하는 박스/내용물
    │   ├── sensors.py          # 학습에 필요한 접촉 센서와 선택적 카메라
    │   ├── layout.py           # 캡처 배치와 env-local 좌표 처리
    │   ├── asset_geometry.py   # composed USD의 물리 크기 계산
    │   └── flap_spawn.py       # RL 전용 flap hinge 제한
    ├── managers/              # action/observation/reward/event 등의 설정
    ├── mdp/                   # manager에서 호출하는 계산과 reset 함수
    ├── tasks/                 # 하위 task 정의와 이전 import 경로 호환
    ├── agents/                # PPO 설정
    └── runners/               # 학습/평가 진입점
```

일반 환경을 전부 만든 뒤 배경 객체를 제거하는 방식이 아니라,
`rl/scenes/scene_cfg.py`에서 학습에 필요한 객체만 조립한다.
공유 로봇·박스 asset과 캡처 파일은 재사용하지만 scene 구성은 분리되어 있다.

RL conveyor는 외부의 장식용 conveyor asset 대신 **단순한 cuboid 지지체**를
사용한다. 실제 rack과 버튼·fence는 작업 정의 및 충돌 관계를 위해 유지한다.
RL scene이 장식 배경이나 network conveyor asset을 별도로 내려받지 않도록 한다.
필요한 로봇·작업 asset은 사전에 로컬에 준비해야 한다. 카메라는 기본적으로 끈다.

## 학습 실행

먼저 [설치 가이드](INSTALL.md)에 따라 환경을 준비하고 프로젝트 루트에서 실행한다.

```bash
conda activate env_isaaclab_232
python -m pip install -e '.[rl]'

# 기존 기본값: 2 env, headless, 2000 PPO iteration
./train_flap_pick.sh

# 메모리 상황을 확인한 뒤 환경 개수 증가
CUDA_VISIBLE_DEVICES=1 ./train_flap_pick.sh --num-envs 16384 --device cuda:0 \
  --save-interval 1000 \
  --kit_args "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"

# 추가 확장 예시: 성능/메모리 측정값에 근거한 권장치는 아님
./train_flap_pick.sh --num-envs 16 --env-spacing 8.0 --max-iterations 4000
```

Flap pick의 작업 정의는 그대로다.

- `quest_ready_02`의 root pose 및 관절값으로 reset한다.
- 베이스·허리·머리를 고정하고 양팔 14개 관절 + 그리퍼 2개, 총 16개 action을 쓴다.
- 박스 안착 후 오른손으로 지정 flap의 상단을 집고, 안정된 높이보다 6cm 들어 올려 0.5초 유지한다. 왼손 받침은 허용한다.
- 로봇의 랙·펜스·버튼·컨베이어 접촉은 0.1N 초과 시 초기 유예 없이 실패 처리한다.
- 내용물·선점 박스·domain randomization은 이 첫 실험의 설정에서 꺼져 있다.

작업/초기 상태/접촉 정의는 [Flap pick 가이드](RL_FLAP_PICK.md), 다른 하위 task는
[RL 학습 가이드](RL_TRAINING.md)를 참고한다. 일반 `train_rl.sh`/`play_rl.sh`도
같은 RL 전용 scene과 병렬 설정을 사용한다.

환경 수를 늘리면 physics state, contact buffer와 PPO rollout이 함께 증가한다.
실제 PPO 업데이트를 포함한 짧은 실행으로 메모리 최대값을 측정하며 점진적으로 늘린다.
특히 접촉 센서와 카메라는 메모리 사용량에 영향을 준다. `--num-envs` 증가가
항상 처리량 증가를 뜻하지 않으며, GPU 메모리 부족 시 먼저 환경 수를 줄인다.

## 병렬 평가

```bash
# 동일한 task를 4개 환경에서 렌더링 없이 평가
./play_flap_pick.sh \
  --checkpoint /absolute/path/to/model_1999.pt \
  --num-envs 4 \
  --env-spacing 8.0 \
  --headless
```

모든 환경은 같은 layout과 초기 상태 프리셋을 사용하지만, 로봇/박스의 물리 상태,
action, reward, 종료 조건 및 episode reset은 환경별로 분리된다. Randomization을
끄면 초기 조건도 같으므로 병렬 평가 자체가 다양한 조건의 robustness 검증은 아니다.
manifest에는 scene profile `minimal_rl_v1`과 병렬 실행 설정을 기록한다.
이전 일반-scene 기반 RL checkpoint/reset bank는 profile이 달라 로더에서 거부한다.
새 profile에서 학습한 checkpoint는 같은 task/config로 환경 수를 바꿔 평가할 수 있다.
환경 수·간격·device는 실행 기록에는 남지만 물리 task 계약 hash에는 넣지 않는다.

## 복제와 좌표계

`rl/envs/parallel_cfg.py`의 기본 설정은 다음과 같다.

| 설정 | 기본값 | 의미 |
|---|---|---|
| `num_envs` | 실행기/CLI에서 지정 | 동시에 생성할 환경 수 |
| `env_spacing` | 8.0m, 최소 5.0m | 환경 원점 사이의 간격 |
| `replicate_physics` | `True` | 동일한 물리 구성을 복제 |
| `filter_collisions` | `True` | 서로 다른 환경의 객체 간 충돌을 필터링 |
| `clone_in_fabric` | `False` | USD 기반 spawn/후처리 경로 사용 |
| 공유 바닥의 `collision_group` | `-1` | 모든 환경에서 사용할 공용 충돌 바닥 |

환경별 객체는 `/World/envs/env_<번호>/...` 아래에 생성한다. 캡처한 rack/box
배치와 `quest_ready_02`의 root pose는 **env origin 기준 좌표**로 사용하고,
world 좌표에 쓸 때 해당 `env.scene.env_origins[env_id]`를 더한다.
예를 들어 로봇 위치를 비교할 때 world 좌표가 환경마다 다른 것은 정상이다.
같은 초기 자세인지 비교하려면 root world position에서 각 env origin을 뺀다.

새 reset/event를 작성할 때도 이 구분을 유지한다.

- 입력된 `env_ids`만 reset하고, 다른 환경의 상태를 덮어쓰지 않는다.
- world 위치를 작성할 때 해당 환경 원점 offset을 정확히 한 번 적용한다.
- 환경별 객체의 prim path에 `{ENV_REGEX_NS}`를 사용한다.
- episode 진행도/성공 유지 시간 등은 환경 수 차원의 tensor로 보관한다.

`env_spacing`의 최소값은 현재 workcell을 위한 입력 제한이다. Rack/fence 배치를
더 넓히거나 새로운 task 구조물을 추가했다면 필요한 간격도 다시 검토한다.
여러 복제 환경의 world 좌표를 한 캡처 파일에 섞어서 저장하지 않는다.

## 무엇을 어디서 수정할까?

| 수정 목적 | 파일 또는 옵션 |
|---|---|
| 환경 개수, GPU 선택 | `--num-envs`, `--device` |
| 환경 간격 | `--env-spacing`, `src/kuavo_isaaclab_scene/rl/envs/parallel_cfg.py` |
| RL에 어떤 객체를 생성할지 | `src/kuavo_isaaclab_scene/rl/scenes/scene_cfg.py` |
| Rack/fence/button/conveyor 충돌 구성 | `src/kuavo_isaaclab_scene/rl/scenes/workcell.py` |
| RL 로봇, gripper 구성 | `src/kuavo_isaaclab_scene/rl/scenes/robot.py` |
| Task 박스와 flap spawn | `src/kuavo_isaaclab_scene/rl/scenes/boxes.py` |
| 접촉 센서 또는 선택적 카메라 | `src/kuavo_isaaclab_scene/rl/scenes/sensors.py` |
| 캡처 배치 적용 및 좌표 변환 | `src/kuavo_isaaclab_scene/rl/scenes/layout.py` |
| Manager 연결, physics timestep | `src/kuavo_isaaclab_scene/rl/envs/env_cfg.py` |
| 관측/action/보상/종료 조건 | `src/kuavo_isaaclab_scene/rl/managers/`, `rl/mdp/` |
| Flap pick 실험값 | `configs/rl_pick_arms_only.py` |
| PPO network, rollout, optimizer | `src/kuavo_isaaclab_scene/rl/agents/ppo_cfg.py` |
| 일반 환경/Meta Quest 구성 | `src/kuavo_isaaclab_scene/envs/`, `teleop/` |

## 2026-09-07 실행 측정

물리 GPU 1: A100 80GB. 다른 사용자의 작업이 약 29GB를 사용 중이다.
`CUDA_VISIBLE_DEVICES=1`로 CUDA를 격리하고 `--device cuda:0`을 사용했다.
Omniverse 렌더러 번호는 별도이므로 `--/renderer/activeGpu=1`도 지정했다.
검사 PID가 다른 GPU를 점유하지 않는지 확인하고, 매 실행이 끝난 뒤 메모리 반환을 확인했다.

| 환경 수 | PPO 업데이트 | 최대 프로세스 VRAM | 전이/초 | 결과 |
|---:|---:|---:|---:|---|
| 2 | 20 | 2961MiB | 2.51 | 초기 버전 학습·timeout reset 통과 |
| 512 | 3 | 3889MiB | 569.62 | 최신 마찰 5.0/4.0, 안착 기준, PPO 통과 |
| 8192 | 2 | 16517MiB | 6240.86 | 안착·충돌 격리·PPO 통과 |

2개 환경 결과는 마찰 및 안착 기준 변경 전 실행이다. 512개 환경의 안착 후 높이 오차는
최대 3.58e-7m였다. 8192개 환경에서도 같은 오차를 확인했다. 0번 환경에만 랙을
접촉시키자 0번만 즉시 실패했고 나머지 8191개 환경의 장애물 힘은 0이었다.
충돌 검사는 학습 전 별도 상태에서 수행하고 원상 복구 후 PPO를 실행했다.
초기화 시간은 512개 약 22초, 8192개 약 182초였으며 표의 처리량에는 초기화를 포함하지 않는다.

원본 로그와 `probe.json`은 `artifacts/rl/diagnostics/`에 있다. Headless, 카메라 없음,
16 actions, rollout 32, PPO 4 minibatches/5 epochs 조건이다. 512개 실행의 관측은 216차원이다.
측정은 동시 사용자 부하에 따라 달라지며, 짧은 실행으로 파지 성공률·학습 수렴을 보장하지 않는다.

사용자 지정 최종 학습 환경 수는 **16384**다. 26624개 검사는 초기화 중 사용자 요청으로
중단했으므로 완료된 메모리 한계 측정값으로 취급하지 않는다. 새 흔들림 억제 보상과
223차원 관측을 적용한 학습은 `artifacts/rl/stable_grasp/`에 기록한다.

16384개 최종 학습은 첫 PPO 업데이트(524288 전이)를 완료했고 처리량은 약 8347 전이/초였다.
이때 관측한 프로세스 VRAM은 29399MiB(약 28.7GiB)이며, 전체 학습의 최대값은 아니다.
`CUDA_VISIBLE_DEVICES=1`과 GPU UUID를 확인했고 `model_0.pt`의 모든 모델 텐서는 유한했다.
검증 상세는 `artifacts/rl/stable_grasp/train_20260907_223932_c32c48/verification.json`에 있다.

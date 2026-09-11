# SAC · Diffusion Policy · DPPO

기존 PPO와 같은 `WorkcellRLEnvCfg`를 사용한다. 오른손으로 flap 상단을 집어
6 cm 들고 유지하며, 반대손의 박스 지지는 허용한다. 로봇과 rack/주변 장애물의
접촉 임계값 0.1 N, 초기 안정화, 공통 손가락 마찰 5.0/4.0,
들기 전 박스 이동·흔들림 보상은 `configs/rl_pick_arms_only.py`와 공통 MDP에서 온다.
새 실행기는 이 조건을 따로 복제하거나 완화하지 않는다.

## 파일 위치

```text
src/kuavo_isaaclab_scene/rl/
├── algorithms/                 # Isaac import 없는 PyTorch 계산
│   ├── common.py               # 관측 정규화, GAE, gradient 검사
│   ├── sac.py                  # SAC 모델, optimizer, replay
│   ├── diffusion.py            # DDPM BC 및 denoising chain
│   └── dppo.py                 # denoising MDP의 PPO 업데이트
├── data/episodes.py            # 정확한 RL 관측/action episode HDF5
├── envs/terminal_observation.py # 자동 reset 직전 관측 보존
└── runners/
    ├── alternatives.py         # 공통 CLI, GPU 격리, Isaac 시작·종료
    ├── train_sac.py            # SAC rollout/update
    ├── collect_diffusion.py    # PPO 성공 trajectory 수집
    ├── pretrain_diffusion.py   # 시뮬레이터 없는 diffusion 사전학습
    ├── train_dppo.py           # BC checkpoint → DPPO / 재개
    ├── evaluate_alternative.py # SAC/BC/DPPO 평가
    └── storage.py              # 계약 확인, checkpoint 보관, scalar 기록
scripts/rl/flap_pick.sh         # 새 방법들의 단일 launcher
tests/test_rl_alternatives.py   # CPU 수치·실행 경로 검증
```

기존 `agents/ppo_cfg.py`, `runners/train.py`, 루트 `train_flap_pick.sh`는 PPO용이다.
기존 `common.parse_args`에는 새 runner가 CLI를 추가하는 선택적 hook만 추가했다.
실험 산출물은 코드 폴더가 아닌 `artifacts/rl/` 아래에 둔다.

## 환경과 GPU

기존 conda `env_isaaclab_232`의 Python 3.11 / torch 2.7.0 / Isaac Lab 2.3.2를
사용한다. 설치된 `.[rl]`과 h5py 외에 skrl, diffusers, LeRobot를 추가 설치할 필요는 없다.
공통 환경 설정과 PPO teacher 수집 때문에 기존 RSL-RL 의존성은 유지한다.

```bash
conda activate env_isaaclab_232
bash scripts/rl/flap_pick.sh --help
```

아래 `N`은 사용 가능한 **물리 GPU 번호**로 바꾼다. 시뮬레이터 실행기는
`CUDA_VISIBLE_DEVICES`에 숫자 하나만 허용한다. CUDA는 `cuda:0`, Omniverse
renderer는 물리 GPU `N`으로 설정하고 multi-GPU를 끈다. renderer 인자는
직접 중복 지정하지 않는다. 기본 병렬 수는 짧은 확인용 **2 env**다.
현재 PPO의 16384 env 실행을 자동 중지하거나 다른 GPU의 프로세스를 정리하지 않는다.

## 1. SAC

먼저 짧게 확인한다.

```bash
CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh sac \
  --num-envs 2 --max-iterations 2 --rollout-steps 32 \
  --learning-starts 4 --batch-size 16 --updates-per-step 1 \
  --replay-capacity 256 --log-dir artifacts/rl/sac_smoke
```

큰 병렬 학습의 설정 예시다. GPU에서 실제 측정한 최대 병렬 수를 의미하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh sac \
  --num-envs 16384 --max-iterations 2000 \
  --replay-capacity 1000000 --replay-device cpu \
  --batch-size 1024 --updates-per-step 16 --learning-starts 100000 \
  --save-interval 1000 --keep-checkpoints 2 --log-dir artifacts/rl/sac
```

SAC는 tanh Gaussian actor, twin Q, target Q의 Polyak 업데이트, 자동 entropy
조절을 사용한다. 관측 통계는 online으로 갱신하고 replay에는 원래 관측을 저장한다.
기본 lr=3e-4, gamma=0.99, tau=0.005, alpha 초기값=0.1이다.
수식 기준은 [SAC 공식 설명](https://spinningup.openai.com/en/latest/algorithms/sac.html)이다.

`--learning-starts`는 **전체 transition 수**이며 그 전에는 uniform random action을
사용한다. `--updates-per-step`은 **vector step마다** 수행하는 minibatch 수다.
환경 수만 늘리면 transition당 업데이트가 줄어든다. 예시의 sample reuse 비율은
`16 × 1024 / 16384 = 1`이다. 실제 속도와 성공률을 보며 조절해야 한다.

현재 223-D 관측, 16-D action에서 100만 transition replay는 약 **1.73 GiB**다.
기본은 CPU RAM이며 `--replay-device cuda:0`으로 GPU에 둘 수 있다. 모델·시뮬레이터·
학습 minibatch 메모리는 별도다. Replay 용량은 env별 용량이 아니다.

SAC `--checkpoint`는 모델·정규화·optimizer·target Q를 복원한다. 디스크 절약을 위해
replay는 저장하지 않으므로 재개 시 replay와 random warmup을 새로 시작한다.
시뮬레이터 상태와 RNG를 그대로 복원하는 bit-exact resume는 아니다.

## 2. 성공 trajectory 수집

성공하는 **현재 계약의 PPO teacher checkpoint**가 필요하다. checkpoint 옆의
`manifest.json`에서 robot, action scale/order, 관측, task 계약을 검사한다.

```bash
CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh collect \
  --checkpoint /path/to/ppo_run/model_1000.pt \
  --num-envs 8 --episodes 100 --collect-max-steps 20000 \
  --log-dir artifacts/rl/diffusion_demos
```

출력 `collect_<시간>_<ID>/episodes.hdf5`에는 성공한 **완전한 episode만** 저장한다.
대기 중 action은 0으로 기록한다. 충돌 실패나 timeout episode를 성공 데이터로
사용하지 않는다. 지정 step 안에 목표 개수를 모으지 못하면 실패 상태와 실제 개수를
`collection.json`에 쓰고 이미 모은 성공 데이터는 보존한다. 수집 RAM을 제한하기 위해
최대 64 env를 허용한다. 초기 PPO가 아직 집기에 성공하지 못하면 이 단계에서 먼저
teacher를 개선해야 한다. 무작위 정책을 성공 demonstration처럼 만들지 않는다.

HDF5 형식은 `format_version=1`, `action_encoding=manager_normalized_incremental_v1`,
JSON `manifest` 속성과 `episode_XXXXXX/{obs,action}` dataset이다. `obs[t]`는
`action[t]` **실행 전** 관측이며 각 group에는 `success`, `complete` 속성이 있다.

## 3. Diffusion Policy 사전학습

```bash
bash scripts/rl/flap_pick.sh pretrain \
  --dataset /path/to/collect_run/episodes.hdf5 \
  --output artifacts/rl/diffusion_bc/run_01 \
  --device cpu --steps 10000 --batch-size 256 \
  --horizon 4 --denoising-steps 20 --save-interval 1000
```

새 output 폴더를 지정한다. GPU 사전학습은 앞에 `CUDA_VISIBLE_DEVICES=N`을
붙이고 `--device cuda:0`을 사용한다. 이 단계는 Isaac을 시작하지 않는다.
MLP epsilon prediction, cosine DDPM schedule, demonstration 관측 정규화를 사용한다.
Action chunk는 episode 경계를 넘지 않는 full window만 사용한다.
관측은 현재 RL의 **223-D state**, action은 **16-D 증분 제어**이며 영상 입력은 없다.

## 4. Diffusion → DPPO RL

```bash
CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh dppo \
  --checkpoint artifacts/rl/diffusion_bc/run_01/checkpoint_00010000.pt \
  --num-envs 2 --max-iterations 10 --rollout-steps 32 \
  --batch-size 256 --critic-warmup 5 \
  --save-interval 1000 --log-dir artifacts/rl/dppo_smoke
```

처음 5 iteration은 critic만 학습하고 이후 diffusion network를 업데이트한다.
DPPO checkpoint를 다시 `--checkpoint`로 넘기면 actor/critic/optimizer와 iteration을
복원한다. `--max-iterations`는 재개 후 추가 iteration 수다. 재개 시 저장된 DPPO
optimizer 설정을 사용하므로 `--epochs`, `--batch-size`, `--critic-warmup`은 신규 BC→RL
시작 때 적용된다. BC 정규화 통계는 RL 중 고정한다.

이 구현은 [DPPO 논문](https://arxiv.org/abs/2409.00588)과
[공식 구현](https://github.com/irom-princeton/dppo)을 기준으로 작성한 **state/DDPM variant**다.
최종 action에 Gaussian을 씌워 PPO라고 부르는 방식이 아니다.

- 매 환경 step에서 전체 denoising chain과 기존 transition log probability를 저장한다.
- 업데이트 때 각 관측마다 denoising step을 균등 표본추출해 PPO ratio를 계산한다.
- Denoising 단계별 discount와 0.001→0.1 clip schedule, KL 조기 중단을 적용한다.
- 마지막 단계에도 std floor=0.1을 사용한다. Gaussian latent는 자르지 않고
  환경으로 보내는 action만 `[-1,1]`로 제한한다.
- 전체 chunk의 **합산 joint log likelihood**를 사용한다. 공식 코드의 log density
  clipping/평균 처리와는 다르며, DDIM·일부 denoising 단계만 fine-tune하는 기능은 없다.
- Horizon=4를 예측해 첫 action 하나만 실행하고 매 control step 다시 계획한다.
  따라서 reset 후 이전 episode의 나머지 action을 실행하지 않는다. 30 Hz 제어 주기와
  환경 reward/discount의 시간 단위가 유지된다.

32 rollout × 16384 env × 21 chain state × 4 horizon × 16 action × float32는
chain만 **2.625 GiB**, 관측·확률·가치 등을 더한 고정 rollout은 약 **3.1 GiB**다.
이외에 시뮬레이터와 gradient minibatch 공간이 필요하다. PPO의 메모리 측정을
SAC/DPPO에 그대로 적용하거나 VRAM 여유를 모두 채우지 않는다.

## 5. 평가·저장·검증 범위

```bash
CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh play \
  --checkpoint /path/to/native_run/checkpoint_00001000.pt --num-envs 2 --episodes 20
```

SAC는 actor 평균을 사용한다. Diffusion은 고정 zero prior에서 noise 없는 mean path를
사용한다. 학습과 같은 확률적 diffusion 평가에는 `--stochastic-eval`을 지정한다.
각 실행 폴더에 `manifest.json`, `env.yaml`, `agent.yaml`, `status.json`을 쓰고,
학습 scalar는 `metrics.jsonl`, 평가 성공·충돌·소요시간은 `metrics.json`에 쓴다.
`status.json`이 `complete`인지 확인한다. Kit 종료가 shell exit code를 덮을 수 있어
로그와 상태 파일을 함께 확인해야 한다.

학습 checkpoint 기본 간격은 **1000 iteration**, 마지막 iteration에도 저장하고
해당 실행 폴더의 최근 **2개만** 남긴다. `--keep-checkpoints`로 변경할 수 있다.
큰 replay와 전체 rollout, 영상은 checkpoint에 넣지 않는다.

2026-09-07 구현 검증: 신규 CPU 테스트 18개와 기존 RL 관련 117개, 총 **135개 통과**.
CPU에서 SAC·DPPO 실제 학습 loop/재개, diffusion BC entrypoint,
Gaussian likelihood, denoising likelihood 재계산, clipping, timeout bootstrap,
episode window, GPU CLI 격리, checkpoint 보관을 검사했다.
GPU 1 추가 시뮬레이션은 기존 PPO/타 사용자 작업과의 경합 때문에 자동 승인 검토가
거부하여 **새 SAC/DPPO의 실제 Isaac 실행·수렴·성공률·최대 env 수는 아직 검증하지 않았다.**

```bash
PYTHONPATH=src python -m pytest tests/test_rl_alternatives.py -q
```

## LeRobot 연결 범위

이 경로는 **자체 state Diffusion Policy → DPPO 연결**이다. 기존 LeRobot
DiffusionPolicy나 GR00T checkpoint를 직접 import하는 adapter는 아니다.
기존 Quest/LeRobot 데이터의 절대 joint target과 joint-state 관측만으로는 현재 RL의
223-D 접촉·flap·안정화·action 누적 상태를 자동 복원할 수 없다.
기존 데이터를 단순히 reshape하거나 관절 차분만 취해 사용하지 않는다.

현재 제공한 연결 경로는 같은 RL 환경에서의 PPO teacher rollout → HDF5 → DDPM BC
→ DPPO이며 관측·action·물리 계약을 전 구간 검사한다. 영상 LeRobot 정책의 직접
fine-tuning에는 별도로 동일 observation encoder, action 변환, denoising scheduler
및 transition likelihood adapter가 필요하다.

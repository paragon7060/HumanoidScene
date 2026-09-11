#!/usr/bin/env bash
# Alternative algorithms use the same pinned flap-pick experiment as PPO.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
method="${1:---help}"
if [[ "$method" == "--help" || "$method" == "-h" ]]; then
  cat <<'USAGE'
Usage: CUDA_VISIBLE_DEVICES=N bash scripts/rl/flap_pick.sh {sac|collect|dppo|play} [options]
       bash scripts/rl/flap_pick.sh pretrain --dataset PATH --output NEW_DIR [options]
Simulation defaults: 2 envs, headless, cuda:0; one explicitly masked physical GPU.
Pinned task: right-hand flap pick, optional opposite-hand support, 0.1 N obstacle limit.
SAC: --max-iterations N --replay-capacity N --batch-size N --updates-per-step N
Collect: --checkpoint PPO.pt --episodes N --collect-max-steps N (successful episodes only)
Pretrain: offline state-conditioned DDPM; --device cpu (default) or masked cuda:0
DPPO: --checkpoint diffusion.pt --max-iterations N --critic-warmup N
Play: --checkpoint native.pt --episodes N [--stochastic-eval]
Storage: --save-interval 1000 --keep-checkpoints 2; replay is never checkpointed.
Details: docs/RL_ALTERNATIVES.md
USAGE
  exit 0
fi
shift
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
if [[ "$method" == "pretrain" ]]; then
  exec env PYTHONUNBUFFERED=1 PYTHONPATH="${PROJECT_DIR}/src" \
    "$ISAACLAB_PYTHON" -m kuavo_isaaclab_scene.rl.runners.pretrain_diffusion "$@"
fi
case "$method" in sac|collect|dppo|play) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
for argument in "$@"; do
  case "${argument%%=*}" in
    --method|--task|--robot-model|--gripper|--boxes|--control-mode|--config|--initial-state|--initial-states-file|--reset-bank|--workcell-layout|--rack-box-poses|--rack-boxes|--ignore-captured-box-poses|--cargo-per-box|--prefill)
      echo "Pinned option: ${argument%%=*}. Use the alternatives Python module for another experiment." >&2
      exit 2 ;;
  esac
done
exec env TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="${PROJECT_DIR}/src" \
  KUAVO_CONFIG_DIR="${PROJECT_DIR}/configs" \
  "$ISAACLAB_PYTHON" -m kuavo_isaaclab_scene.rl.runners.alternatives \
  --method "$method" --num-envs 2 --headless --device cuda:0 "$@" \
  --robot-model s200062 --gripper s200062_integrated \
  --task pick --boxes medium_box_0 --control-mode arms-only \
  --config "${PROJECT_DIR}/configs/rl_pick_arms_only.py" \
  --initial-states-file "${PROJECT_DIR}/configs/initial_states.json" \
  --workcell-layout "${PROJECT_DIR}/configs/workcell_layout.json" \
  --rack-box-poses "${PROJECT_DIR}/configs/rack_box_poses.json" \
  --cargo-per-box 0 --prefill 0 --no-randomization

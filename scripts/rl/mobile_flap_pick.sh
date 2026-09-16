#!/usr/bin/env bash
# Single-box SAC/DPPO experiment with 25 base/torso/arm/head/hand actions.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
method="${1:?Usage: CUDA_VISIBLE_DEVICES=N bash scripts/rl/mobile_flap_pick.sh sac|play [options]}"
shift
case "$method" in sac|dppo|play) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
exec env TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="${PROJECT_DIR}/src" \
  KUAVO_CONFIG_DIR="${PROJECT_DIR}/configs" \
  "$ISAACLAB_PYTHON" -m kuavo_isaaclab_scene.rl.runners.alternatives \
  --method "$method" --num-envs 2 --headless --device cuda:0 \
  --robot-model "${KUAVO_ROBOT_MODEL:-s63}" \
  --gripper "${KUAVO_GRIPPER:-leju-twofinger}" "$@" \
  --task pick --boxes medium_box_0 --control-mode whole-body \
  --config "${PROJECT_DIR}/configs/rl_pick_whole_body.py" \
  --initial-states-file "${PROJECT_DIR}/configs/initial_states.json" \
  --workcell-layout "${PROJECT_DIR}/configs/workcell_layout.json" \
  --rack-box-poses "${PROJECT_DIR}/configs/rack_box_poses.json" \
  --cargo-per-box 0 --prefill 0 --no-randomization --no-rack-rollers

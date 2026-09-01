#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"

RWH_ROOT="${RWH_KUAVO_V2_DIR:-${PROJECT_DIR}/artifacts/huggingface/Whalswp_RwH-Kuavo_V2}"
CHECKPOINT="${RWH_KUAVO_V2_CHECKPOINT:-${RWH_ROOT}/stage1/checkpoints/checkpoint-40K/pretrained_model}"
TASK="${RWH_KUAVO_V2_TASK:-pick up the box}"
CONTROL_HZ="${RWH_KUAVO_V2_CONTROL_HZ:-10}"
LEROBOT_RUNTIME="$(resolve_groot_n15_python)"

if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
  printf '%s\n' \
    "RwH-Kuavo V2 checkpoint is missing: ${CHECKPOINT}/model.safetensors" \
    "Download it first with: ./download_rwh_kuavo_v2_checkpoint.sh" >&2
  exit 1
fi

exec "${PROJECT_DIR}/scripts/eval_groot.sh" \
  --policy-profile rwh-kuavo-v2-s56 \
  --robot-model s56 \
  --gripper s56_twofinger \
  --checkpoint "${CHECKPOINT}" \
  --lerobot-python "${LEROBOT_RUNTIME}" \
  --local-files-only \
  --allow-checkpoint-key-mismatch \
  --no-domain-randomization \
  --control-hz "${CONTROL_HZ}" \
  --task "${TASK}" \
  "$@"

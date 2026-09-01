#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="${RWH_KUAVO_V2_DIR:-${PROJECT_DIR}/artifacts/huggingface/Whalswp_RwH-Kuavo_V2}"
REVISION="d6687fa613e2847c67d38a161d1de847bc7b235f"
CHECKPOINT_DIR="${LOCAL_DIR}/stage1/checkpoints/checkpoint-40K/pretrained_model"

if ! command -v hf >/dev/null 2>&1; then
  printf '%s\n' \
    "The Hugging Face 'hf' CLI is required." \
    "Install it in the active environment with:" \
    "  python -m pip install --upgrade huggingface_hub" >&2
  exit 1
fi

hf download Whalswp/RwH-Kuavo_V2 \
  --revision "${REVISION}" \
  --include 'stage1/checkpoints/checkpoint-40K/pretrained_model/*' \
  --local-dir "${LOCAL_DIR}" \
  --max-workers 4

if [[ "${RWH_SKIP_BASE_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  hf download nvidia/GR00T-N1.5-3B
fi

for required_file in config.json model.safetensors; do
  if [[ ! -f "${CHECKPOINT_DIR}/${required_file}" ]]; then
    printf 'Checkpoint download is incomplete; missing %s\n' \
      "${CHECKPOINT_DIR}/${required_file}" >&2
    exit 1
  fi
done

printf '%s\n' \
  "Checkpoint ready:" \
  "${CHECKPOINT_DIR}"

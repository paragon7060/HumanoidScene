#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
LEROBOT_PYTHON="$(resolve_lerobot_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${LEROBOT_SRC:-}" ]]; then
  PACKAGE_PATH="${LEROBOT_SRC}:${PACKAGE_PATH}"
fi

exec env TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="${PACKAGE_PATH}" \
  "${LEROBOT_PYTHON}" -m kuavo_isaaclab_scene.evaluation.offline_eval_groot "$@"

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
exec env TERM=xterm PYTHONUNBUFFERED=1 \
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  KUAVO_CONFIG_DIR="${KUAVO_CONFIG_DIR:-${PROJECT_DIR}/configs}" \
  "${ISAACLAB_PYTHON}" -m kuavo_isaaclab_scene.rl.runners.train "$@"

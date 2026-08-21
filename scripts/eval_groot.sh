#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${LEROBOT_SRC:-}" ]]; then
  PACKAGE_PATH="${LEROBOT_SRC}:${PACKAGE_PATH}"
fi
RUNTIME_CONFIG_DIR="${KUAVO_CONFIG_DIR:-${PROJECT_DIR}/configs}"

exec env TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="${PACKAGE_PATH}" \
  KUAVO_CONFIG_DIR="${RUNTIME_CONFIG_DIR}" \
  "${ISAACLAB_PYTHON}" -m kuavo_isaaclab_scene.eval_groot "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_CONFIG_DIR="${KUAVO_CONFIG_DIR:-${PROJECT_DIR}/configs}"

if [[ -z "${LEROBOT_PYTHON:-}" ]]; then
  DETECTED_LEROBOT_PYTHON="$(resolve_lerobot_python 2>/dev/null || true)"
  if [[ -n "${DETECTED_LEROBOT_PYTHON}" ]]; then
    export LEROBOT_PYTHON="${DETECTED_LEROBOT_PYTHON}"
  fi
fi

exec env TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH="${PACKAGE_PATH}" \
  KUAVO_CONFIG_DIR="${RUNTIME_CONFIG_DIR}" \
  "${ISAACLAB_PYTHON}" -m kuavo_isaaclab_scene.collect_quest_teleop "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_CONFIG_DIR="${KUAVO_CONFIG_DIR:-${PROJECT_DIR}/configs}"

# Pure geometry/JSON utility: do not start Isaac Sim or require a GPU.
exec env PYTHONPATH="${PACKAGE_PATH}" KUAVO_CONFIG_DIR="${RUNTIME_CONFIG_DIR}" \
  "${ISAACLAB_PYTHON}" -m kuavo_isaaclab_scene.workcell.workcell_gap "$@"

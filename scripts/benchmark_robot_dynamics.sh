#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
benchmark_python="$(resolve_isaaclab_python)"
require_supported_runtime "$benchmark_python"
exec env TERM=xterm PYTHONUNBUFFERED=1 \
  PYTHONPATH="${KUAVO_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  KUAVO_CONFIG_DIR="${KUAVO_PROJECT_DIR}/configs" \
  "$benchmark_python" "${KUAVO_PROJECT_DIR}/scripts/benchmark_robot_dynamics.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
replay_python="$(resolve_isaaclab_python)"
require_supported_runtime "$replay_python"
exec env TERM=xterm PYTHONUNBUFFERED=1 \
  PYTHONPATH="${KUAVO_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  KUAVO_CONFIG_DIR="${KUAVO_PROJECT_DIR}/configs" \
  "$replay_python" "${KUAVO_PROJECT_DIR}/scripts/replay_joint_candidate.py" "$@"

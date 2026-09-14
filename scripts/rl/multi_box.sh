#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:---help}"
case "$mode" in
  staged) module=train_staged ;;
  end-to-end) module=train_end_to_end ;;
  eval) module=evaluate ;;
  *) echo 'Usage: bash scripts/rl/multi_box.sh {staged|end-to-end|eval} [options]';
     echo 'See docs/RL_MULTI_BOX.md'; exit 0 ;;
esac
shift
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
cd "${PROJECT_DIR}"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="${PROJECT_DIR}/src" \
  KUAVO_CONFIG_DIR="${PROJECT_DIR}/configs" \
  "${ISAACLAB_PYTHON}" -m "kuavo_isaaclab_scene.rl.multi_box.experiments.${module}" "$@"

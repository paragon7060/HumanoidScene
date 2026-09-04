#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"

ALLOW_UNSUPPORTED=0
if [[ "${1:-}" == "--allow-unsupported" ]]; then
  ALLOW_UNSUPPORTED=1
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--allow-unsupported]\n' "$0" >&2
  exit 2
fi

ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Python executable: %s\n' "${ISAACLAB_PYTHON}"
printf 'OS/architecture:  %s / %s\n' "$(uname -s)" "$(uname -m)"
printf 'GLIBC:            %s\n' "$(ldd --version 2>&1 | head -n 1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  if GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null)"; then
    printf '%s\n' "${GPU_INFO}" | sed 's/^/GPU:              /'
  else
    printf '%s\n' 'GPU:              nvidia-smi could not query the driver'
  fi
else
  printf '%s\n' 'GPU:              nvidia-smi not found'
fi

ARGS=()
if [[ ${ALLOW_UNSUPPORTED} -eq 1 ]]; then
  ARGS+=(--allow-unsupported)
fi
env PYTHONPATH="${PACKAGE_PATH}" "${ISAACLAB_PYTHON}" \
  -m kuavo_isaaclab_scene.core.runtime_compat "${ARGS[@]}"

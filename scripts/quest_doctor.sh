#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"

ALLOW_UNSUPPORTED=0
QUEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-unsupported)
      ALLOW_UNSUPPORTED=1
      shift
      ;;
    --require-runtime)
      QUEST_ARGS+=("$1")
      shift
      ;;
    --xr-runtime-json)
      if [[ $# -lt 2 ]]; then
        printf '%s\n' '--xr-runtime-json requires a path.' >&2
        exit 2
      fi
      QUEST_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      printf 'Usage: %s [--allow-unsupported] [--require-runtime] [--xr-runtime-json PATH]\n' "$0" >&2
      exit 2
      ;;
  esac
done

DOCTOR_ARGS=()
if [[ ${ALLOW_UNSUPPORTED} -eq 1 ]]; then
  DOCTOR_ARGS+=(--allow-unsupported)
fi
"${PROJECT_DIR}/scripts/doctor.sh" "${DOCTOR_ARGS[@]}"

ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
env PYTHONPATH="${PACKAGE_PATH}" "${ISAACLAB_PYTHON}" \
  -m kuavo_isaaclab_scene.teleop.quest_runtime "${QUEST_ARGS[@]}"

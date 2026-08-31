#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_DIR="${CLOUDXR_JS_SAMPLES_DIR:-${PROJECT_DIR}/.external/cloudxr-js-samples}/simple"
exec "${ISAACLAB_PYTHON:-python3}" "${PROJECT_DIR}/scripts/serve_quest_browser.py" \
  --directory "${SAMPLE_DIR}/build" "$@"

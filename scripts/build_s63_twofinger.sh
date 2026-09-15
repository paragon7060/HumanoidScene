#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_DIR="$(resolve_isaaclab_dir)"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
ASSET_DIR="${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets"
VARIANT_DIR="${ASSET_DIR}/kuavo_s63_twofinger"
"${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/build_s63_twofinger_urdf.py"
env TERM=xterm "${ISAACLAB_PYTHON}" "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${VARIANT_DIR}/urdf/kuavo_s63_twofinger.urdf" \
    "${VARIANT_DIR}/usd/kuavo_s63_twofinger_fixed.usd" \
    --fix-base --joint-stiffness 400 --joint-damping 40 --headless --device cpu
"${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/finalize_twofinger_usd.py" \
    "${VARIANT_DIR}/usd/kuavo_s63_twofinger_fixed.usd" \
    --claw-config "${ASSET_DIR}/leju_claw_two_finger/config.json"
rm -f "${VARIANT_DIR}/usd/config.yaml"

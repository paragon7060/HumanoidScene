#!/usr/bin/env bash
# Build only independent claws; do not regenerate or modify host robot USDs.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_DIR="$(resolve_isaaclab_dir)"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"
CLAW_DIR="${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/leju_claw_two_finger"
"${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/extract_leju_claw.py"
mapfile -t CLAW_ACTUATOR < <("${ISAACLAB_PYTHON}" -c \
    'import json,sys; c=json.load(open(sys.argv[1]))["actuator"]; print(c["stiffness"]); print(c["damping"])' \
    "${CLAW_DIR}/config.json")
CLAW_STIFFNESS="${CLAW_ACTUATOR[0]}"
CLAW_DAMPING="${CLAW_ACTUATOR[1]}"
for side in left right; do
    env TERM=xterm "${ISAACLAB_PYTHON}" \
        "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
        "${CLAW_DIR}/urdf/leju_claw_${side}.urdf" \
        "${CLAW_DIR}/usd/${side}/leju_claw_${side}.usd" \
        --joint-stiffness "${CLAW_STIFFNESS}" --joint-damping "${CLAW_DAMPING}" \
        --headless --device cpu
    "${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/finalize_twofinger_usd.py" \
        "${CLAW_DIR}/usd/${side}/leju_claw_${side}.usd" \
        --sides "${side:0:1}" --claw-config "${CLAW_DIR}/config.json"
    # Machine-local converter metadata is not part of the distributable asset.
    rm -f "${CLAW_DIR}/usd/${side}/config.yaml"
done

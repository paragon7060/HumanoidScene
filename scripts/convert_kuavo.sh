#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_DIR="$(resolve_isaaclab_dir)"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo5/kuavo5.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo5/usd/kuavo5_fixed.usd" \
    --fix-base \
    --joint-stiffness 400 \
    --joint-damping 40 \
    --headless

echo "Generated fixed-base Kuavo USD."

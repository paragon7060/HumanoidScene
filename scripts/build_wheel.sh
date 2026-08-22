#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_CONFIG_DIR="${PROJECT_DIR}/src/kuavo_isaaclab_scene/configs"

# Keep the standalone wheel fallback synchronized with the deployment config.
cp "${PROJECT_DIR}/configs/workcell_layout.json" "${PACKAGE_CONFIG_DIR}/workcell_layout.json"
cp "${PROJECT_DIR}/configs/rack_box_poses.json" "${PACKAGE_CONFIG_DIR}/rack_box_poses.json"
cp "${PROJECT_DIR}/configs/grippers.json" "${PACKAGE_CONFIG_DIR}/grippers.json"

exec "${ISAACLAB_PYTHON}" -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${PROJECT_DIR}/dist" \
  "${PROJECT_DIR}"

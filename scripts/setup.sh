#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--check-only]\n' "$0" >&2
  exit 2
fi

ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
PACKAGE_PATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Isaac Lab Python: %s\n' "${ISAACLAB_PYTHON}"
require_supported_runtime "${ISAACLAB_PYTHON}"

if [[ ${CHECK_ONLY} -eq 0 ]]; then
  "${ISAACLAB_PYTHON}" -m pip install --no-build-isolation -e "${PROJECT_DIR}"
fi

env PYTHONPATH="${PACKAGE_PATH}" "${ISAACLAB_PYTHON}" -c \
  'import gymnasium, h5py, isaaclab, numpy, websockets; print("Python dependencies: OK")'
env PYTHONPATH="${PACKAGE_PATH}" "${ISAACLAB_PYTHON}" -c \
  'from kuavo_isaaclab_scene.core.paths import ASSET_DIR; required=("Rack.usd", "SmallBox.usd", "MediumBox.usd", "LargeBox.usd", "XLargeBox.usd"); missing=[name for name in required if not (ASSET_DIR/name).is_file()]; assert not missing, f"Missing assets: {missing}"; print(f"Packaged USD assets: OK ({ASSET_DIR})")'
env PYTHONPATH="${PACKAGE_PATH}" "${ISAACLAB_PYTHON}" -c \
  'from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings; cfg=load_gripper_settings(); print(f"Gripper config: OK ({cfg.name}, {cfg.active_sides})")'

if LEROBOT_CHECK="$(resolve_lerobot_python 2>/dev/null)"; then
  printf 'LeRobot Dataset v3 Python: %s\n' "${LEROBOT_CHECK}"
else
  printf '%s\n' 'LeRobot Dataset v3 Python: not configured (optional for HDF5/scene use)'
fi

printf '%s\n' 'Setup check completed.'

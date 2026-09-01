#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_DIR="$(resolve_isaaclab_dir)"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/urdf/kuavo_s56.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/usd/kuavo_s56_fixed.usd" \
    --fix-base \
    --joint-stiffness 400 \
    --joint-damping 40 \
    --headless

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/urdf/kuavo_s63.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/kuavo_s63_fixed.usd" \
    --fix-base \
    --joint-stiffness 400 \
    --joint-damping 40 \
    --headless

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/urdf/biped_s200062.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/usd/kuavo_s200062_fixed.usd" \
    --fix-base \
    --joint-stiffness 400 \
    --joint-damping 40 \
    --headless

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/robotiq_2f85/urdf/robotiq_2f85.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/robotiq_2f85/usd/robotiq_2f85.usd" \
    --joint-stiffness 100 \
    --joint-damping 10 \
    --headless

for usd_path in \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/usd/kuavo_s56_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/kuavo_s63_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/usd/kuavo_s200062_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/robotiq_2f85/usd/robotiq_2f85.usd"; do
    if [[ ! -f "${usd_path}" ]] || [[ "$(stat -c '%s' "${usd_path}")" -lt 1024 ]]; then
        printf 'USD conversion failed or produced an empty stage: %s\n' "${usd_path}" >&2
        exit 1
    fi
done

# The URDF importer writes conversion metadata with machine-specific absolute
# paths and transient mesh USD directories. They are not referenced by the
# composed USDs and must not be included in the offline asset package.
rm -f \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/robotiq_2f85/usd/config.yaml"
for mesh_name in base_mount base coupler driver follower pad silicone_pad spring_link; do
    rm -rf "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/robotiq_2f85/meshes/2f85/${mesh_name}_tmp"
done

echo "Generated fixed-base Kuavo S200062/S63/S56 and Robotiq 2F-85 USDs."

#!/usr/bin/env bash
# Generate the offline Isaac input without modifying the official model.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63"
test -f "${ASSET_DIR}/urdf/biped_s63.urdf"
# Path-only mechanical adaptation: geometry, joints, frames and colors stay intact.
sed 's|package://kuavo_assets/models/biped_s63/meshes/|../meshes/|g' \
    "${ASSET_DIR}/urdf/biped_s63.urdf" > "${ASSET_DIR}/urdf/kuavo_s63.urdf"

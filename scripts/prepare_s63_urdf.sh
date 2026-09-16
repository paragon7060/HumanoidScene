#!/usr/bin/env bash
# Generate the offline Isaac input without modifying the official model.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63"
test -f "${ASSET_DIR}/urdf/biped_s63.urdf"
# Use the S63 hand-pitch shells instead of the source noHand meshes.
# Preserve the official source, joints, frames, collision shapes and colors.
sed -e 's|package://kuavo_assets/models/biped_s63/meshes/|../meshes/|g' \
    -e 's|l_hand_pitch_noHand.STL|l_hand_pitch.STL|g' \
    -e 's|r_hand_pitch_noHand.STL|r_hand_pitch.STL|g' \
    "${ASSET_DIR}/urdf/biped_s63.urdf" > "${ASSET_DIR}/urdf/kuavo_s63.urdf"

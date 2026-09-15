#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
ISAACLAB_DIR="$(resolve_isaaclab_dir)"
ISAACLAB_PYTHON="$(resolve_isaaclab_python)"
require_supported_runtime "${ISAACLAB_PYTHON}"

"${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/build_s56_twofinger_urdf.py"
bash "${PROJECT_DIR}/scripts/prepare_s63_urdf.sh"

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
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/urdf/kuavo_s56_bare.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_bare/usd/kuavo_s56_bare_fixed.usd" \
    --fix-base \
    --joint-stiffness 400 \
    --joint-damping 40 \
    --headless

env TERM=xterm "${ISAACLAB_PYTHON}" \
    "${ISAACLAB_DIR}/scripts/tools/convert_urdf.py" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/urdf/kuavo_s56_twofinger.urdf" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_twofinger/usd/kuavo_s56_twofinger_fixed.usd" \
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

"${ISAACLAB_PYTHON}" "${PROJECT_DIR}/scripts/finalize_twofinger_usd.py"
bash "${PROJECT_DIR}/scripts/build_s63_twofinger.sh"

for usd_path in \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56/usd/kuavo_s56_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_bare/usd/kuavo_s56_bare_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_twofinger/usd/kuavo_s56_twofinger_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/kuavo_s63_fixed.usd" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/usd/kuavo_s200062_fixed.usd"; do
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
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_bare/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s56_twofinger/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s63/usd/config.yaml" \
    "${PROJECT_DIR}/src/kuavo_isaaclab_scene/assets/kuavo_s200062/usd/config.yaml"

echo "Generated fixed-base Kuavo S200062/S63/S56/S56-twofinger USDs."

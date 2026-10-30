#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_DIR}/scripts/_common.sh"
COLLISION_PYTHON="$(resolve_isaaclab_python)"
# Keep Isaac Sim's NumPy/Torch intact; never upgrade shared dependencies.
"${COLLISION_PYTHON}" -c 'import numpy, scipy, trimesh'
exec "${COLLISION_PYTHON}" -m pip install --no-cache-dir --no-deps --only-binary=:all: \
  --target "${PROJECT_DIR}/.external/self-collision" python-fcl==0.7.0.8

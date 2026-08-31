#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/versions/stable.env"

ENV_NAME="${KUAVO_DEFAULT_CONDA_ENV}"
ISAACLAB_CHECKOUT="${PROJECT_DIR}/.external/IsaacLab-${KUAVO_ISAAC_LAB_TAG}"
REUSE_ENV=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Install the repository's pinned GA stack into a new conda environment:
  Isaac Sim ${KUAVO_ISAAC_SIM_VERSION}
  Isaac Lab ${KUAVO_ISAAC_LAB_TAG}
  Python ${KUAVO_PYTHON_VERSION}

Options:
  --env-name NAME       Conda environment name (default: ${ENV_NAME})
  --isaaclab-dir PATH  Isaac Lab source checkout (default: ${ISAACLAB_CHECKOUT})
  --reuse-env           Reuse an existing environment; never delete it
  --dry-run             Print the resolved installation plan only
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { printf '%s\n' '--env-name requires a value' >&2; exit 2; }
      ENV_NAME="$2"
      shift 2
      ;;
    --isaaclab-dir)
      [[ $# -ge 2 ]] || { printf '%s\n' '--isaaclab-dir requires a value' >&2; exit 2; }
      ISAACLAB_CHECKOUT="$2"
      shift 2
      ;;
    --reuse-env)
      REUSE_ENV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  printf '%s\n' \
    'This installer targets Linux x86_64. The repository OpenXR/Quest path is not supported on aarch64.' >&2
  exit 1
fi

GLIBC_VERSION="$(ldd --version 2>&1 | sed -n '1s/.* //p')"
if [[ "$(printf '%s\n' "2.35" "${GLIBC_VERSION}" | sort -V | head -n 1)" != "2.35" ]]; then
  printf 'GLIBC %s is too old for Isaac Sim pip packages; 2.35+ is required.\n' \
    "${GLIBC_VERSION}" >&2
  exit 1
fi

printf '%s\n' \
  'Pinned stable installation plan:' \
  "  conda environment: ${ENV_NAME}" \
  "  Python:            ${KUAVO_PYTHON_VERSION}" \
  "  Isaac Sim:         ${KUAVO_ISAAC_SIM_VERSION}" \
  "  Isaac Lab:         ${KUAVO_ISAAC_LAB_TAG}" \
  "  PyTorch:           ${KUAVO_TORCH_VERSION} (${KUAVO_PYTORCH_INDEX_URL})" \
  "  Isaac Lab source:  ${ISAACLAB_CHECKOUT}"

if [[ ${DRY_RUN} -eq 1 ]]; then
  printf '%s\n' \
    '' \
    'Dry run only; no environment or files were changed.' \
    "Run without --dry-run to install, then: conda activate ${ENV_NAME}"
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  printf '%s\n' 'conda was not found. Install Miniconda/Anaconda, then rerun this script.' >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  ENV_EXISTS=1
fi
if [[ ${ENV_EXISTS} -eq 1 && ${REUSE_ENV} -ne 1 ]]; then
  printf '%s\n' \
    "Conda environment '${ENV_NAME}' already exists." \
    'Use --reuse-env after confirming it is the intended environment; nothing was deleted.' >&2
  exit 1
fi
if [[ ${ENV_EXISTS} -eq 0 ]]; then
  conda create -y -n "${ENV_NAME}" "python=${KUAVO_PYTHON_VERSION}" pip
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade "pip>=25.3" setuptools "wheel==0.45.1"
python -m pip install \
  "isaacsim[all,extscache]==${KUAVO_ISAAC_SIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade \
  "torch==${KUAVO_TORCH_VERSION}" \
  "torchvision==${KUAVO_TORCHVISION_VERSION}" \
  --index-url "${KUAVO_PYTORCH_INDEX_URL}"

if [[ -d "${ISAACLAB_CHECKOUT}/.git" ]]; then
  CHECKOUT_TAG="$(git -C "${ISAACLAB_CHECKOUT}" describe --tags --exact-match 2>/dev/null || true)"
  if [[ "${CHECKOUT_TAG}" != "${KUAVO_ISAAC_LAB_TAG}" ]]; then
    printf 'Existing Isaac Lab checkout is at %s, expected %s: %s\n' \
      "${CHECKOUT_TAG:-an untagged commit}" "${KUAVO_ISAAC_LAB_TAG}" "${ISAACLAB_CHECKOUT}" >&2
    exit 1
  fi
elif [[ -e "${ISAACLAB_CHECKOUT}" ]]; then
  printf 'Isaac Lab target exists but is not a Git checkout: %s\n' \
    "${ISAACLAB_CHECKOUT}" >&2
  exit 1
else
  mkdir -p "$(dirname "${ISAACLAB_CHECKOUT}")"
  git clone --depth 1 --branch "${KUAVO_ISAAC_LAB_TAG}" \
    https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_CHECKOUT}"
fi

# Install the Isaac Lab core extension required by this workcell directly.
# The upstream wrapper also installs mimic/RL/notebook packages and opens the
# VS Code/EULA bootstrap even with `--install none`; those extras are not
# required by this repository and introduce unbounded transitive dependencies.
# flatdict 4.0.1 still imports pkg_resources from setuptools during its build.
# Limit the isolated build environment without downgrading runtime packages.
python -m pip install \
  --build-constraint "${PROJECT_DIR}/versions/build-constraints.txt" \
  --editable "${ISAACLAB_CHECKOUT}/source/isaaclab"
# Re-assert Isaac Sim's exact runtime requirements after Isaac Lab resolves
# ONNX/OpenXR dependencies from the current package index.
python -m pip install \
  "onnx==${KUAVO_ONNX_VERSION}" \
  "typing_extensions==4.12.2" \
  "psutil==5.9.8"
python -m pip install --no-build-isolation -e "${PROJECT_DIR}"

export ISAACLAB_PYTHON="${CONDA_PREFIX}/bin/python"
export ISAACLAB_DIR="${ISAACLAB_CHECKOUT}"
"${PROJECT_DIR}/scripts/doctor.sh"

printf '%s\n' \
  '' \
  'Installation complete. In a new shell run:' \
  "  conda activate ${ENV_NAME}" \
  '  export ISAACLAB_PYTHON="$(command -v python)"' \
  "  export ISAACLAB_DIR=\"${ISAACLAB_CHECKOUT}\"" \
  '  ./setup.sh --check-only' \
  '  ./run_scene.sh --prefill 2'

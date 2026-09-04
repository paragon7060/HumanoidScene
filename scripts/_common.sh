#!/usr/bin/env bash

# Shared, side-effect-free launcher helpers. This file is sourced by scripts.

KUAVO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_isaaclab_python() {
  local candidates=()
  local candidate

  if [[ -n "${ISAACLAB_PYTHON:-}" ]]; then
    candidates+=("${ISAACLAB_PYTHON}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  candidates+=(
    "${HOME}/anaconda3/envs/env_isaaclab_232/bin/python"
    "${HOME}/miniconda3/envs/env_isaaclab_232/bin/python"
    "${HOME}/miniforge3/envs/env_isaaclab_232/bin/python"
    "${HOME}/anaconda3/envs/env_isaaclab/bin/python"
    "${HOME}/miniconda3/envs/env_isaaclab/bin/python"
    "${HOME}/miniforge3/envs/env_isaaclab/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    # Do not import AppLauncher here: the first Isaac Sim import displays the
    # NVIDIA EULA prompt. Runtime discovery and doctor checks must remain
    # non-interactive, so distribution metadata is sufficient at this stage.
    if [[ -x "${candidate}" ]] && "${candidate}" -c \
      'from importlib.metadata import version; version("isaaclab")' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' \
    "Unable to find a Python environment containing Isaac Lab." \
    "Activate the Isaac Lab conda environment or set:" \
    "  export ISAACLAB_PYTHON=/absolute/path/to/python" >&2
  return 1
}

require_supported_runtime() {
  local python_exe="$1"
  local package_path="${KUAVO_PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  local args=(--quiet)

  if [[ "${KUAVO_ALLOW_UNSUPPORTED_RUNTIME:-0}" == "1" ]]; then
    args+=(--allow-unsupported)
  fi
  if ! env PYTHONPATH="${package_path}" "${python_exe}" \
    -m kuavo_isaaclab_scene.core.runtime_compat "${args[@]}"; then
    printf '%s\n' \
      'Install the pinned stable stack with:' \
      '  ./install_isaaclab_stable.sh' \
      'For a deliberate temporary override only:' \
      '  export KUAVO_ALLOW_UNSUPPORTED_RUNTIME=1' >&2
    return 1
  fi
}

resolve_lerobot_python() {
  local candidates=()
  local candidate

  if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
    candidates+=("${LEROBOT_PYTHON}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=(
    "${HOME}/anaconda3/envs/lerobot060_groot/bin/python"
    "${HOME}/miniconda3/envs/lerobot060_groot/bin/python"
    "${HOME}/miniforge3/envs/lerobot060_groot/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c \
      'from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == "v3.0"' \
      >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' \
    "Unable to find a LeRobot Dataset v3 Python environment." \
    "Set LEROBOT_PYTHON before recording LeRobot data:" \
    "  export LEROBOT_PYTHON=/absolute/path/to/python" >&2
  return 1
}

resolve_groot_n15_python() {
  local candidates=()
  local candidate

  if [[ -n "${LEROBOT_PYTHON:-}" ]]; then
    candidates+=("${LEROBOT_PYTHON}")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=(
    "${HOME}/anaconda3/envs/lerobot_050_groot/bin/python"
    "${HOME}/miniconda3/envs/lerobot_050_groot/bin/python"
    "${HOME}/miniforge3/envs/lerobot_050_groot/bin/python"
    "${HOME}/anaconda3/envs/lerobot_050/bin/python"
    "${HOME}/miniconda3/envs/lerobot_050/bin/python"
    "${HOME}/miniforge3/envs/lerobot_050/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c \
      'import lerobot; from lerobot.policies.groot.modeling_groot import GrootPolicy; assert tuple(map(int, lerobot.__version__.split(".")[:2])) < (0, 6)' \
      >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' \
    "Unable to find a LeRobot 0.5.x environment with GR00T N1.5 support." \
    "Set LEROBOT_PYTHON to its Python executable." >&2
  return 1
}

resolve_isaaclab_dir() {
  local candidates=()
  local candidate

  if [[ -n "${ISAACLAB_DIR:-}" ]]; then
    candidates+=("${ISAACLAB_DIR}")
  fi
  candidates+=(
    "${KUAVO_PROJECT_DIR}/.external/IsaacLab-v2.3.2"
    "${KUAVO_PROJECT_DIR}/../IsaacLab"
    "${HOME}/IsaacLab"
    "${HOME}/isaaclab"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/scripts/tools/convert_urdf.py" ]]; then
      printf '%s\n' "$(cd "${candidate}" && pwd)"
      return 0
    fi
  done

  printf '%s\n' \
    "Unable to locate the IsaacLab source checkout required for URDF conversion." \
    "Set ISAACLAB_DIR=/absolute/path/to/IsaacLab." >&2
  return 1
}

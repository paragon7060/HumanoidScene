#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/NVIDIA/cloudxr-js-samples.git"
UPSTREAM_COMMIT="29941936e90234a06847ba1c209d70f60b6b59bd"
TARGET_DIR="${CLOUDXR_JS_SAMPLES_DIR:-${PROJECT_DIR}/.external/cloudxr-js-samples}"
PATCH_FILES=(
  "${PROJECT_DIR}/integrations/cloudxr/cloudxr-js-samples-local-isaaclab.patch"
  "${PROJECT_DIR}/integrations/cloudxr/cloudxr-js-samples-quest-cert.patch"
  "${PROJECT_DIR}/integrations/cloudxr/cloudxr-js-samples-kuavo-safe-defaults.patch"
)
BRIDGE_FILE="${PROJECT_DIR}/integrations/cloudxr/LocalIsaacLabBridge.ts"
PATCH_ONLY=0

if [[ "${1:-}" == "--patch-only" ]]; then
  PATCH_ONLY=1
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--patch-only]\n' "$0" >&2
  exit 2
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
  git -C "${TARGET_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
fi

CURRENT_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
  printf 'CloudXR sample must be at %s, found %s in %s\n' \
    "${UPSTREAM_COMMIT}" "${CURRENT_COMMIT}" "${TARGET_DIR}" >&2
  exit 1
fi

patch_marker_present() {
  case "$(basename "$1")" in
    cloudxr-js-samples-local-isaaclab.patch)
      grep -q "LocalIsaacLabBridge" "${TARGET_DIR}/simple/src/main.ts"
      ;;
    cloudxr-js-samples-quest-cert.patch)
      grep -q 'id="certLink" href="#" target="_self"' "${TARGET_DIR}/simple/index.html"
      ;;
    cloudxr-js-samples-kuavo-safe-defaults.patch)
      grep -q "kuavo-cloudxr-safe-defaults-v1" "${TARGET_DIR}/simple/src/main.ts"
      ;;
    *)
      return 1
      ;;
  esac
}

for PATCH_FILE in "${PATCH_FILES[@]}"; do
  if patch_marker_present "${PATCH_FILE}" \
      || git -C "${TARGET_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    printf 'Patch already applied: %s\n' "$(basename "${PATCH_FILE}")"
  else
    git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}"
    git -C "${TARGET_DIR}" apply "${PATCH_FILE}"
  fi
done
install -m 0644 "${BRIDGE_FILE}" "${TARGET_DIR}/simple/src/LocalIsaacLabBridge.ts"

printf 'CloudXR browser integration: %s/simple\n' "${TARGET_DIR}"
if [[ ${PATCH_ONLY} -eq 1 ]]; then
  exit 0
fi

if ! command -v npm >/dev/null 2>&1; then
  printf '%s\n' 'npm is required to install the CloudXR.js sample.' >&2
  exit 1
fi
if [[ -z "${CLOUDXR_NPM_TGZ:-}" || ! -f "${CLOUDXR_NPM_TGZ}" ]]; then
  printf '%s\n' \
    'Set CLOUDXR_NPM_TGZ to the NVIDIA-provided CloudXR.js package:' \
    '  export CLOUDXR_NPM_TGZ=/absolute/path/to/nvidia-cloudxr-6.2.0.tgz' >&2
  exit 1
fi

npm --prefix "${TARGET_DIR}/simple" install "${CLOUDXR_NPM_TGZ}"
printf '%s\n' \
  'Browser client is ready. Start it with:' \
  "  npm --prefix \"${TARGET_DIR}/simple\" run dev-server"

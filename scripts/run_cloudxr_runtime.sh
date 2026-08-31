#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${CLOUDXR_RUNTIME_DIR:-${PROJECT_DIR}/.external/cloudxr-runtime}"
SOURCE="${PROJECT_DIR}/integrations/cloudxr/runtime_service.cpp"
BINARY="${PROJECT_DIR}/.external/bin/kuavo-cloudxr-service"
if [[ ! -f "${RUNTIME_DIR}/include/cxrServiceAPI.h" || ! -f "${RUNTIME_DIR}/libcloudxr.so" ]]; then
  printf '%s\n' 'Extract the CloudXR Linux SDK into .external/cloudxr-runtime or set CLOUDXR_RUNTIME_DIR.' >&2
  exit 1
fi
mkdir -p "$(dirname "${BINARY}")"
if [[ ! -x "${BINARY}" || "${SOURCE}" -nt "${BINARY}" || "${RUNTIME_DIR}/include/cxrServiceAPI.h" -nt "${BINARY}" ]]; then
  BUILD_OUTPUT="$(mktemp "${BINARY}.XXXXXX")"
  trap 'rm -f "${BUILD_OUTPUT}"' EXIT
  "${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
    -I "${RUNTIME_DIR}/include" "${SOURCE}" -L "${RUNTIME_DIR}" \
    -Wl,-rpath,"${RUNTIME_DIR}" -lcloudxr -o "${BUILD_OUTPUT}"
  chmod 755 "${BUILD_OUTPUT}"
  mv -f "${BUILD_OUTPUT}" "${BINARY}"
  trap - EXIT
fi
export NV_CXR_OUTPUT_DIR="${NV_CXR_OUTPUT_DIR:-${PROJECT_DIR}/artifacts/cloudxr-logs}"
mkdir -p "${NV_CXR_OUTPUT_DIR}"
exec "${BINARY}" "$@"

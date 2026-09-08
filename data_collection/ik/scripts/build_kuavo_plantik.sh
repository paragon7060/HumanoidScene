#!/usr/bin/env bash
set -euo pipefail

runtime_root=${1:-${KUAVO_IK_RUNTIME:-}}
if [[ -z "$runtime_root" ]]; then
  echo "usage: $0 <external-runtime-root>" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build_root="$runtime_root/build"
mkdir -p "$build_root"

cmake_args=(
  -S "$repo_root/data_collection/ik/native"
  -B "$build_root"
  "-DKUAVO_IK_RUNTIME=$runtime_root"
)
if [[ -n "${DRAKE_DIR:-}" ]]; then
  cmake_args+=("-Ddrake_DIR=$DRAKE_DIR")
fi
cmake "${cmake_args[@]}"
cmake --build "$build_root" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
echo "built $build_root/kuavo_plantik_server"


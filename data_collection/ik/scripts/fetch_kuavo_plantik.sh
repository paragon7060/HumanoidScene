#!/usr/bin/env bash
set -euo pipefail

runtime_root=${1:-${KUAVO_IK_RUNTIME:-}}
if [[ -z "$runtime_root" ]]; then
  echo "usage: $0 <external-runtime-root>" >&2
  exit 2
fi

source_repo="https://github.com/LejuRobotics/kuavo-ros-opensource.git"
source_commit="1797481d4639a76afbdc035f705acbf522fceafa"
tmp_root=$(mktemp -d "${TMPDIR:-/tmp}/kuavo_plantik_fetch.XXXXXX")
cleanup() { rm -rf "$tmp_root"; }
trap cleanup EXIT

git clone --quiet --filter=blob:none --no-checkout "$source_repo" "$tmp_root/repo"
git -C "$tmp_root/repo" checkout --quiet "$source_commit" -- \
  src/manipulation_nodes/motion_capture_ik/include/plantIK.h \
  src/manipulation_nodes/motion_capture_ik/lib/libplantIK.so

mkdir -p "$runtime_root/include" "$runtime_root/lib"
install -m 0644 "$tmp_root/repo/src/manipulation_nodes/motion_capture_ik/include/plantIK.h" \
  "$runtime_root/include/plantIK.h"
install -m 0755 "$tmp_root/repo/src/manipulation_nodes/motion_capture_ik/lib/libplantIK.so" \
  "$runtime_root/lib/libplantIK.so"
cat > "$runtime_root/manifest.txt" <<EOF
source_repo=$source_repo
source_commit=$source_commit
header_sha256=$(sha256sum "$runtime_root/include/plantIK.h" | awk '{print $1}')
library_sha256=$(sha256sum "$runtime_root/lib/libplantIK.so" | awk '{print $1}')
EOF
echo "installed Kuavo plantIK runtime at $runtime_root"
cat "$runtime_root/manifest.txt"


#!/usr/bin/env bash
# Project-local rclone; credentials stay in an ignored, private directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
umask 077
AUTH_DIR="$ROOT/.external/rclone-auth"
mkdir -p "$AUTH_DIR"
chmod 700 "$AUTH_DIR"
if [[ -f "$AUTH_DIR/rclone.conf" ]]; then
  chmod 600 "$AUTH_DIR/rclone.conf"
fi
if [[ ! -x "$ROOT/.external/rclone/rclone" ]]; then
  echo 'Install the official rclone binary at .external/rclone/rclone; see docs/RL_GOOGLE_DRIVE.md.' >&2
  exit 1
fi
exec "$ROOT/.external/rclone/rclone" --config "$AUTH_DIR/rclone.conf" "$@"

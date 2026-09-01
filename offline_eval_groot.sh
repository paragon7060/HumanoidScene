#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_DIR}/scripts/offline_eval_groot.sh" "$@"

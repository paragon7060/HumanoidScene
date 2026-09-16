#!/usr/bin/env bash
# Read-only setup for four representative joints; does not run --send.
set -euo pipefail
CAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAL_SESSION="$CAL_DIR/data/response_$(date +%Y%m%d_%H%M%S_%N)"
CAL_TOOL="$CAL_DIR/calibrate.py"

python3 "$CAL_TOOL" capture --output "$CAL_SESSION/ready.json"
python3 "$CAL_TOOL" plan --capture "$CAL_SESSION/ready.json" --kind static \
  --joint zarm_r6_joint zarm_l5_joint zarm_r5_joint zarm_r7_joint \
  --amplitude-deg 0.5 --ramp-s 4 --hold-s 3 \
  --output "$CAL_SESSION/response.json"
python3 "$CAL_TOOL" check --plan "$CAL_SESSION/response.json" \
  --tracking-deg 0.5 --watch-s 30 --log "$CAL_SESSION/check.jsonl"

printf '\n읽기 전용 준비 완료. 현장 정지·제어권·동작 공간을 확인한 뒤, 5분 이내에 실행할 실물 명령(115초):\n'
printf 'python3 %q run --plan %q --tracking-deg 0.5 --log %q --send\n' \
  "$CAL_TOOL" "$CAL_SESSION/response.json" "$CAL_SESSION/run.jsonl"
printf '세션: %s\n' "$CAL_SESSION"

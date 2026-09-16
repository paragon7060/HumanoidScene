#!/usr/bin/env bash
# Read-only preparation; never publish robot targets or change WBC mode.
set -euo pipefail
CAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAL_SESSION="$CAL_DIR/data/right_finish_$(date +%Y%m%d_%H%M%S_%N)"
CAL_TOOL="$CAL_DIR/calibrate.py"

python3 "$CAL_TOOL" capture --output "$CAL_SESSION/ready.json"
python3 "$CAL_TOOL" plan --capture "$CAL_SESSION/ready.json" --kind static \
  --joint zarm_r6_joint zarm_r7_joint --amplitude-deg 0.25 \
  --ramp-s 2 --hold-s 2 --output "$CAL_SESSION/right_67.json"
python3 "$CAL_TOOL" check --plan "$CAL_SESSION/right_67.json" \
  --tracking-deg 0.5 --watch-s 30 --log "$CAL_SESSION/check.jsonl"

printf '\n읽기 전용 검사가 완료됐습니다. 현장에서 정지·제어권을 확인한 뒤, 5분 이내에 실행할 실물 명령(34초):\n'
printf 'python3 %q run --plan %q --tracking-deg 0.5 --log %q --send\n' \
  "$CAL_TOOL" "$CAL_SESSION/right_67.json" "$CAL_SESSION/run.jsonl"
printf '세션: %s\n' "$CAL_SESSION"

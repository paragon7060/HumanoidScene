#!/usr/bin/env bash
# Foreground-only collector services; never launches or stops another process.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_DIR}/.external/quest-collector/session.env"
if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 3 ]] || { echo 'Usage: quest_collector.sh [--config FILE] COMMAND' >&2; exit 2; }
  CONFIG="$2"
  shift 2
fi
COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "${COMMAND}" in
  help|--help|-h)
    printf '%s\n' \
      'Usage: ./quest_collector.sh [--config FILE] info|check|runtime|web|collect [extra arguments]' \
      'Prepare first: ./setup_quest_collector.sh --host <PC LAN IPv4> --download-runtime' \
      'Run in order: runtime -> web -> Quest CONNECT -> collect (three separate terminals).' \
      'check does not start services or Isaac Sim. Services stay in the foreground; Ctrl+C stops them.'
    exit 0 ;;
  info|check|runtime|web|collect) ;;
  *) printf 'Unknown command: %s\n' "${COMMAND}" >&2; exit 2 ;;
esac
[[ -f "${CONFIG}" ]] || { printf 'Missing config: %s\nRun ./setup_quest_collector.sh first.\n' "${CONFIG}" >&2; exit 1; }
# The setup tool creates this local shell file with permissions 0600.
source "${CONFIG}"
: "${ISAACLAB_PYTHON:?}" "${XR_RUNTIME_JSON:?}" "${CLOUDXR_RUNTIME_DIR:?}"
: "${CLOUDXR_HOST:?}" "${CLOUDXR_CERTIFICATE:?}" "${CLOUDXR_KEY:?}"
: "${CLOUDXR_JS_SAMPLES_DIR:?}" "${QUEST_COLLECTOR_WEB_PORT:?}"
cd "${PROJECT_DIR}"

# The configured IP is fixed at setup time. After a Wi-Fi/router change it can
# silently point at an address this PC no longer owns, while the certificate
# still matches that old IP. Binding a throwaway UDP socket is a read-only test:
# 0 = assigned to this PC, 1 = not assigned, 2 = could not determine.
host_assignment() {
  python3 - "$1" <<'PY'
import errno, socket, sys
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind((sys.argv[1], 0))
except OSError as exc:
    sys.exit(1 if exc.errno == errno.EADDRNOTAVAIL else 2)
PY
}

report_stale_host() {
  local level="$1" current
  current="$(ip -4 -o address show scope global 2>/dev/null | awk '{split($4, a, "/"); printf "  %s  %s\n", $2, a[1]}' || true)"
  printf '[%s] Configured CLOUDXR_HOST %s is not assigned to this PC (network changed?).\n' "${level}" "${CLOUDXR_HOST}" >&2
  if [[ -n "${current}" ]]; then
    printf 'Current IPv4 addresses (pick the one on the same network as Quest):\n%s\n' "${current}" >&2
  else
    printf 'Check current addresses with: ip -4 -brief address\n' >&2
  fi
  printf 'Update the collector config (keeps the old config/certificate as backup):\n' >&2
  printf '  ./setup_quest_collector.sh --host <NEW_PC_IP> --isaaclab-python %q' "${ISAACLAB_PYTHON}" >&2
  [[ "${QUEST_COLLECTOR_WEB_PORT}" == 8443 ]] || printf ' --web-port %q' "${QUEST_COLLECTOR_WEB_PORT}" >&2
  [[ -z "${LEROBOT_PYTHON:-}" ]] || printf ' --lerobot-python %q' "${LEROBOT_PYTHON}" >&2
  printf ' --update-config\nThen re-trust the new certificates on Quest. See docs/QUEST_COLLECTOR_SETUP.md section 3.\n' >&2
}

HOST_STATUS=0
if [[ "${COMMAND}" != collect ]]; then
  host_assignment "${CLOUDXR_HOST}" || HOST_STATUS=$?
fi
if [[ "${COMMAND}" == info ]]; then
  printf 'Config: %s\nQuest page: https://%s:%s\nBackend: Manual Input IP:Port\nRuntime: %s:49100 (WSS)\nSDK: %s\nPython: %s\n' \
    "${CONFIG}" "${CLOUDXR_HOST}" "${QUEST_COLLECTOR_WEB_PORT}" "${CLOUDXR_HOST}" "${CLOUDXR_RUNTIME_DIR}" "${ISAACLAB_PYTHON}"
  [[ "${HOST_STATUS}" -ne 1 ]] || report_stale_host WARN
  exit 0
fi
case "${HOST_STATUS}" in
  1) report_stale_host ERROR; exit 1 ;;
  2) printf '[WARN] Could not verify that %s is assigned to this PC; continuing.\n' "${CLOUDXR_HOST}" >&2 ;;
esac
for REQUIRED_FILE in "${XR_RUNTIME_JSON}" "${CLOUDXR_CERTIFICATE}" "${CLOUDXR_KEY}"; do
  [[ -f "${REQUIRED_FILE}" ]] || { printf 'Missing file: %s\n' "${REQUIRED_FILE}" >&2; exit 1; }
done
if [[ "${COMMAND}" == check || "${COMMAND}" == runtime || "${COMMAND}" == web ]]; then
  openssl x509 -in "${CLOUDXR_CERTIFICATE}" -noout -checkend 0
  openssl x509 -in "${CLOUDXR_CERTIFICATE}" -noout -checkip "${CLOUDXR_HOST}"
fi
case "${COMMAND}" in
  check)
    [[ $# -eq 0 ]] || { echo 'check takes no extra arguments.' >&2; exit 2; }
    bash "${PROJECT_DIR}/scripts/run_cloudxr_runtime.sh" --check
    bash "${PROJECT_DIR}/scripts/quest_doctor.sh" --require-runtime
    [[ -f "${CLOUDXR_JS_SAMPLES_DIR}/simple/build/index.html" ]] || { echo 'Missing collector web build.' >&2; exit 1; }
    printf '%s\n' '[OK] Host IP on this PC, files, certificate, SDK loading and Isaac/OpenXR metadata checked. No service or simulator was started.' \
      'Not checked: whether Quest can reach this PC (same Wi-Fi, client isolation, firewall).' ;;
  runtime)
    exec bash "${PROJECT_DIR}/scripts/run_cloudxr_runtime.sh" \
      --host "${CLOUDXR_HOST}" --certificate "${CLOUDXR_CERTIFICATE}" --key "${CLOUDXR_KEY}" "$@" ;;
  web)
    exec bash "${PROJECT_DIR}/scripts/run_quest_browser.sh" \
      --host "${CLOUDXR_HOST}" --port "${QUEST_COLLECTOR_WEB_PORT}" \
      --certificate "${CLOUDXR_CERTIFICATE}" --key "${CLOUDXR_KEY}" "$@" ;;
  collect)
    # Whole-body reward debugging has a purpose-built collection preset. Keep
    # these arguments before "$@" so an explicit caller option still wins.
    RL_REWARD_DEBUG_MODE=""
    COLLECT_ARGS=("$@")
    for ((ARG_INDEX = 0; ARG_INDEX < ${#COLLECT_ARGS[@]}; ARG_INDEX++)); do
      ARG_VALUE="${COLLECT_ARGS[ARG_INDEX]}"
      case "${ARG_VALUE}" in
        --rl-reward-debug=*)
          RL_REWARD_DEBUG_MODE="${ARG_VALUE#*=}" ;;
        --rl-reward-debug)
          NEXT_INDEX=$((ARG_INDEX + 1))
          if [[ "${COLLECT_ARGS[NEXT_INDEX]:-}" == "0" || "${COLLECT_ARGS[NEXT_INDEX]:-}" == "1" || "${COLLECT_ARGS[NEXT_INDEX]:-}" == "2" ]]; then
            RL_REWARD_DEBUG_MODE="${COLLECT_ARGS[NEXT_INDEX]}"
            ARG_INDEX="${NEXT_INDEX}"
          else
            # argparse treats the option without a value as the arms-only mode.
            RL_REWARD_DEBUG_MODE="0"
          fi ;;
      esac
    done
    REWARD_DEBUG_DEFAULTS=()
    INITIAL_STATE_DEFAULTS=()
    if [[ "${RL_REWARD_DEBUG_MODE}" == "1" || "${RL_REWARD_DEBUG_MODE}" == "2" ]]; then
      REWARD_DEBUG_DEFAULTS=(
        --controller-mapping absolute
        --absolute-orientation downward
        --arm-response responsive
        --rl-task pick_place
        --rack-rollers
        --no-quest-camera-overlay
        --no-camera-preview
        --no-wrist-cameras
        --no-head-camera
        --no-rl-obstacle-collision
        --arm-orientation-weight 0.5
      )
    elif [[ -z "${RL_REWARD_DEBUG_MODE}" ]]; then
      INITIAL_STATE_DEFAULTS=(--initial-state s63_leju_vr_collect_01)
    fi
    printf '%s\n' '[START] Connect Quest to the Runtime first; this command starts Isaac Sim and records only after an explicit start.'
    exec bash "${PROJECT_DIR}/scripts/collect_quest_teleop.sh" \
      --robot-model s63 --input-mode controllers --device cpu --control-hz 30 \
      --xr-resolution-scale 1.0 --scene-detail compact --render-quality performance \
      --no-desktop-render --no-camera-preview --no-head-camera --wrist-cameras --no-record-depth \
      --controller-mapping scaled --position-gain 1.1 --dataset-format hdf5 \
      --max-episodes 0 --episode-seconds 0 --no-auto-start \
      "${INITIAL_STATE_DEFAULTS[@]}" "${REWARD_DEBUG_DEFAULTS[@]}" "$@" ;;
esac

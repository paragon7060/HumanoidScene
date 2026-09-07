#!/usr/bin/env bash
# One reproducible experiment; generic train_rl/play_rl remain configurable.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:?Expected train or play}"
shift
case "${mode}" in
  train) defaults=(--num-envs 2 --max-iterations 2000 --headless) ;;
  play) defaults=(--num-envs 1 --episodes 20) ;;
  *) printf 'Expected train or play, got: %s\n' "${mode}" >&2; exit 2 ;;
esac
for argument in "$@"; do
  case "${argument%%=*}" in
    -h|--help)
      printf '%s\n' \
        "Usage: ./${mode}_flap_pick.sh [runner options]" \
        'Pinned: pick / s200062 / s200062_integrated / medium_box_0 / arms-only.' \
        'Pose: INITIAL_STATE in configs/rl_pick_arms_only.py (quest_ready_02).' \
        'Grasp: right hand on flap_left; opposite-hand box support allowed; 14 arm + 2 gripper actions.' \
        'Train defaults: --num-envs 2 --max-iterations 2000 --headless.' \
        'Checkpoint default: every 1000 iterations plus the final model; override --save-interval N.' \
        'Play defaults: --num-envs 1 --episodes 20; --checkpoint PATH is required.' \
        'Allowed examples: --num-envs N --env-spacing M --seed N --device cuda:0 --log-dir PATH --checkpoint PATH.' \
        'Train --checkpoint resumes; play --checkpoint evaluates. Use generic train_rl.sh/play_rl.sh for another experiment.'
      exit 0 ;;
    --task|--robot-model|--gripper|--boxes|--control-mode|--config|--initial-state|--initial-states-file|--reset-bank|--workcell-layout|--rack-box-poses|--rack-boxes|--ignore-captured-box-poses|--cargo-per-box|--prefill)
      printf 'This launcher pins %s. Use train_rl.sh/play_rl.sh for a different experiment.\n' "${argument%%=*}" >&2
      exit 2 ;;
  esac
done
# Resolve experiment files independently of the caller's working directory.
# Pin the config directory so an unrelated teleop shell setting cannot select a different library.
exec env KUAVO_CONFIG_DIR="${PROJECT_DIR}/configs" \
  bash "${PROJECT_DIR}/scripts/${mode}_rl.sh" \
  "${defaults[@]}" "$@" \
  --robot-model s200062 --gripper s200062_integrated \
  --task pick --boxes medium_box_0 --control-mode arms-only \
  --config "${PROJECT_DIR}/configs/rl_pick_arms_only.py" \
  --initial-states-file "${PROJECT_DIR}/configs/initial_states.json" \
  --workcell-layout "${PROJECT_DIR}/configs/workcell_layout.json" \
  --rack-box-poses "${PROJECT_DIR}/configs/rack_box_poses.json" \
  --cargo-per-box 0 --prefill 0 --no-randomization

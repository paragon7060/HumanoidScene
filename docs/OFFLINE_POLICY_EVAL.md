# Offline policy regression evaluation

`offline_eval_groot.sh` runs a GR00T checkpoint over recorded LeRobot Dataset
observations and compares its decoded actions with the recorded actions. This
adopts the useful offline-regression part of LeTools while retaining this
project's policy-runner boundary and dataset schema checks.

It is a pipeline diagnostic, not a task-success score. A low action error can
detect checkpoint, preprocessing, normalization, feature-order, or action-unit
regressions. It cannot prove that a policy succeeds in closed loop, because
multiple actions can be valid for the same observation and recorded
observations do not reflect the policy's own state distribution.

## Required schema

The dataset features must match the checkpoint exactly. In particular, the
current online GR00T evaluator expects the documented 47-D state and 17-D
manager-action schema by default. The raw Quest collector's 16-D Cartesian IK
command is not interchangeable with that action. Retarget/export the raw
demonstrations before training or offline evaluation.

The evaluator fails early when an input key is absent or the predicted and
recorded action dimensions differ.

## Run

Use a current LeRobot Dataset v3 environment. The launcher resolves the same
LeRobot Python setup used by the other GR00T scripts:

```bash
./offline_eval_groot.sh \
  --checkpoint outputs/kuavo_groot_n17/checkpoints/last/pretrained_model \
  --dataset-root /absolute/path/to/kuavo_rack_to_conveyor \
  --repo-id YOUR_ORG/kuavo_rack_to_conveyor \
  --episodes 0,1,2 \
  --device cuda \
  --output-dir artifacts/offline_eval/checkpoint_last
```

Useful diagnostic options:

```bash
# Fast schema/loader smoke test
./offline_eval_groot.sh ... --max-frames 20

# Supply the checkpoint's task instruction when the dataset has none or when
# testing a deliberate language override
./offline_eval_groot.sh ... --task 'Move every open box to the conveyor.'
```

## Outputs

The output directory contains:

- `metrics.json`: global and per-action-dimension MSE, MAE, RMSE, maximum
  absolute error, inference count, and mean/p95 inference latency.
- `actions.npz`: predicted actions, targets, errors, episode indices, and
  inference latency samples for programmatic analysis.
- `actions.csv`: one row per frame for spreadsheet inspection.

Use offline regression as a gate before expensive simulation rollouts, then use
`eval_groot.sh` for the authoritative closed-loop success rate, completion time,
failure reasons, reward/progress, saturation, and inference timing.

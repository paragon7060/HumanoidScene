# GR00T N1.7 / LeRobot evaluation

`eval_groot.sh` connects a trained LeRobot GR00T N1.7 checkpoint to the
manager-based Kuavo workcell. It reads the four Isaac cameras and controlled
joint state, executes a decoded action chunk through Isaac Lab, and writes
per-episode success, completion time, progress, reward, inference latency, and
action-clipping metrics to JSON.

The summary also reports a 95% Wilson confidence interval for success rate,
termination-reason counts, and aggregate mean/p95 inference latency. These
statistics make comparisons across checkpoints less sensitive to a small
episode sample and show whether failures are task timeouts or safety events.

## Fixed LeRobot schema

Training and evaluation must use the same feature names, dimensions, units,
and order. The recommended schema is:

| LeRobot key | Shape | Representation |
|---|---:|---|
| `observation.state` | `(31,)` | 15 Kuavo manager-coordinate joints + 16 Leju claw joint offsets |
| `observation.images.head` | `(3,H,W)` | RGB float after LeRobot conversion |
| `observation.images.waist` | `(3,H,W)` | RGB float after LeRobot conversion |
| `observation.images.left_wrist` | `(3,H,W)` | RGB float after LeRobot conversion |
| `observation.images.right_wrist` | `(3,H,W)` | RGB float after LeRobot conversion |
| `task` | text | natural-language instruction |
| `action` | `(17,)` | 15 Kuavo manager actions + left/right binary gripper |

The first 15 axes are:

```text
waist_yaw_joint,
zarm_l1_joint, zarm_l2_joint, zarm_l3_joint, zarm_l4_joint,
zarm_l5_joint, zarm_l6_joint, zarm_l7_joint,
zarm_r1_joint, zarm_r2_joint, zarm_r3_joint, zarm_r4_joint,
zarm_r5_joint, zarm_r6_joint, zarm_r7_joint
```

Manager coordinates use:

```text
joint_target_rad = default_joint_position + action_scale * action
action_scale = [1.0 for waist, 0.45 for each arm joint]
```

The default action then appends `left_gripper, right_gripper`; positive opens
and negative closes. `--gripper none` restores the legacy 15-D action and
15-D state. Collection, training, and evaluation must use the same preset.

This makes state and action share one coordinate system, which is also the
safest representation if GR00T is trained with relative actions. If a dataset
instead stores absolute joint targets in radians, evaluate with
`--state-mode joint_position --action-mode joint_position`. For delta-radian
actions use `--action-mode joint_delta`.

The Quest collector records a 16-D command: 14-D Cartesian differential IK/head
plus two binary grippers. That representation is not interchangeable with this
17-D joint-manager policy schema. Retarget/export demonstrations into the
schema above before GR00T training; do not point an IK-action checkpoint at
this evaluator.

## LeRobot version and installation

GR00T N1.7 support requires a LeRobot installation that provides the `groot`
extra and `lerobot.policies.groot`. Check the interpreter selected by
`ISAACLAB_PYTHON`; older LeRobot revisions cannot load N1.7.

Install a current version in the Isaac Lab Python environment:

```bash
"${ISAACLAB_PYTHON}" -m pip install --upgrade "lerobot[groot]"
```

If the old editable checkout still shadows that installation, keep it intact
and use a separate current LeRobot checkout:

```bash
export LEROBOT_SRC=/absolute/path/to/current/lerobot/src
```

The launcher prepends `LEROBOT_SRC` to `PYTHONPATH`. Confirm the selected
version before starting Isaac Sim:

```bash
LEROBOT_SRC=/absolute/path/to/current/lerobot/src \
  "${ISAACLAB_PYTHON}" -c \
  'from lerobot.policies.groot.modeling_groot import GrootPolicy; print("N1.7 ready")'
```

Official references:

- <https://github.com/huggingface/lerobot/blob/main/docs/source/groot.mdx>
- <https://github.com/huggingface/lerobot/blob/main/docs/source/bring_your_own_policies.mdx>

## Training checkpoint requirements

Use GR00T N1.7 (`policy.type=groot`) with base model
`nvidia/GR00T-N1.7-3B` and embodiment tag `new_embodiment`. A typical LeRobot
training configuration uses an action chunk such as 16 steps:

```bash
lerobot-train \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --dataset.repo_id=YOUR_ORG/kuavo_rack_to_conveyor \
  --output_dir=outputs/kuavo_groot_n17
```

Use the exact flags supported by the installed LeRobot revision. For eval,
point `--checkpoint` at its `pretrained_model` directory, which should include
the model config/weights and saved policy pre/postprocessor files. Those
processors are essential because they restore the dataset normalization and
decode GR00T output into dataset action units.

## Smoke test without GR00T

This exercises camera acquisition, the LeRobot-shaped observation, 17-D action
application, termination lookup, and metrics writing without loading the 3B
model:

```bash
cd HumanoidScene
./eval_groot.sh --mock-policy --headless --episodes 1 --max-steps 5
```

It intentionally holds the nominal pose and is not expected to succeed.

## Evaluate a trained checkpoint

```bash
./eval_groot.sh \
  --checkpoint outputs/kuavo_groot_n17/checkpoints/last/pretrained_model \
  --headless \
  --episodes 50 \
  --actions-per-inference 8 \
  --metrics-out artifacts/eval/n17_50episodes.json
```

The default captured rack-box poses select the standalone USD boxes on the
rack. Success requires every selected box to be on the conveyor and a physical
green-button joint travel of at least 6 mm. The JSON `success_rate` is the task
success rate; `mean_success_time_s` can be compared to the 11-second target.
Use at least 20 episodes for a quick check and preferably 50 or more for an 80%
claim across randomized starts.

Before a full simulation run, use the recorded-dataset regression evaluator to
check policy loading, preprocessing, normalization, action units, and latency:

```bash
./offline_eval_groot.sh \
  --checkpoint /path/to/pretrained_model \
  --dataset-root /path/to/lerobot/dataset \
  --repo-id YOUR_ORG/kuavo_rack_to_conveyor \
  --max-frames 100
```

See [`OFFLINE_POLICY_EVAL.md`](OFFLINE_POLICY_EVAL.md). Offline action error is
diagnostic only; the closed-loop `success_rate` from this simulator evaluator
remains the authoritative task score.

For a deterministic diagnostic run:

```bash
./eval_groot.sh --checkpoint /path/to/pretrained_model \
  --no-domain-randomization --episodes 5
```

For a different rack layout, the evaluator accepts the same layout arguments
as the existing launchers:

```bash
./eval_groot.sh --checkpoint /path/to/pretrained_model \
  --rack-boxes '1:small*2;2:medium,large;3:xlarge' \
  --ignore-captured-box-poses --episodes 20
```

## Camera names and GPU memory

Checkpoint image feature names must match training. Map custom names by
repeating `--camera-map`:

```bash
./eval_groot.sh --checkpoint /path/to/pretrained_model \
  --camera-map observation.images.front=robustness_camera \
  --camera-map observation.images.hand_left=left_wrist_camera \
  --camera-map observation.images.hand_right=right_wrist_camera
```

Only map cameras that were present in training. The default map uses head,
waist, left wrist, and right wrist. Non-headless mode can show these as small
Isaac Sim windows; disable them with `--no-camera-preview` when they are not
needed.

Isaac Sim RTX rendering and a 3B policy on one GPU can exhaust VRAM and cause
other GPU-accelerated desktop applications to close. Prefer headless eval and,
when two GPUs are available, split them:

```bash
./eval_groot.sh --checkpoint /path/to/pretrained_model \
  --device cuda:0 --policy-device cuda:1 --headless --no-camera-preview
```

On one GPU, keep the checkpoint camera set minimal and use the same modest
camera resolution used during collection. Do not remove a visual key that the
checkpoint declares; the evaluator fails early and reports the missing key.

## Action safety and diagnostics

Decoded manager actions are clamped to `[-1, 1]` by default. The JSON records
`mean_action_saturation`; a large value indicates a units/statistics mismatch
or an overly restrictive clamp. Change the limit with `--action-clip VALUE`,
or use `--action-clip 0` only after verifying the checkpoint's action units.

GR00T relative-action checkpoints are executed with
`predict_action_chunk()`, then the entire chunk is postprocessed before local
queueing. This is required because N1.7's single-step `select_action()` path
does not decode cached relative chunks against a stable observation.

The base S63 and S56 USDs have no articulated fingers, but their default
runtime preset adds two 8-joint Robotiq 2F-85-based Leju claw articulations.
Select S56 with `--robot-model s56`. The evaluator validates the configured
policy action dimension (17 by default, 15 with `--gripper none`) before
execution and records `robot_model` in the metrics JSON. Use the same robot,
preset, and state/action schema for collection, training, and eval.

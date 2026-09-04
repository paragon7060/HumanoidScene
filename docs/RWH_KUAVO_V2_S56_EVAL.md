# RwH-Kuavo V2 checkpoint-40K evaluation on S56

This profile evaluates the GR00T N1.5 checkpoint at
`Whalswp/RwH-Kuavo_V2/stage1/checkpoints/checkpoint-40K` without installing
LeRobot into the Isaac Lab environment.

## Quick start

```bash
cd /absolute/path/to/HumanoidScene
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"

./download_rwh_kuavo_v2_checkpoint.sh
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 \
  --video-out artifacts/eval/rwh_s56_pick_box.mp4 \
  --metrics-out artifacts/eval/rwh_s56_pick_box.json
```

The dedicated launcher supplies `pick up the box` as the default LeRobot
`task` and now selects `--gripper s56_twofinger`, matching the dataset's scalar
claw representation and wrist-mounted D405 rig more closely. A later
`--task 'another instruction'` or `--gripper s56_qiangnao` argument overrides
the corresponding default.

## Training time base

The checkpoint was trained from
[`LejuRobotics/LET-KUAVO-VLA-1.0-Dataset`](https://huggingface.co/datasets/LejuRobotics/LET-KUAVO-VLA-1.0-Dataset).
Its LeRobot v3 `meta/info.json` declares dataset FPS 10, and all three video
features also declare `video.fps=10`. The dedicated launcher therefore defaults
to `--control-hz 10`. With the checkpoint's 16-action chunk, one chunk spans
1.6 seconds and the policy replans at 0.625 Hz when the entire chunk is used.

Isaac physics remains at 120 Hz; evaluation applies each policy action for 12
physics steps. `--video-fps` only changes MP4 playback metadata and must not be
used to compensate for a control-rate mismatch. To override the time base for
an ablation, pass `--control-hz` explicitly or export
`RWH_KUAVO_V2_CONTROL_HZ`. The requested rate must divide 120 Hz exactly.

## Runtime split

- Isaac Lab runs with `env_isaaclab_232`.
- GR00T N1.5 runs in a separate LeRobot 0.5.x Conda process.
- The two processes exchange observations and action chunks over a local IPC socket.

The GR00T worker validated on this machine reports LeRobot 0.5.1. A compatible
checkout that reports the requested 0.5.2 can be selected explicitly:

```bash
export LEROBOT_PYTHON=/absolute/path/to/lerobot-0.5.2-env/bin/python
```

The LeRobot environment must import `lerobot.policies.groot` and report a 0.5.x
version. It does not need Isaac Sim. The Isaac environment must contain the
installed project and Isaac Lab, but it does not need LeRobot. This separation
avoids incompatible NumPy, PyTorch, and simulator dependencies.

The saved checkpoint has one tied LLM embedding serialization key that the
local 0.5.1 model does not register separately. The dedicated launcher uses
non-strict loading for this compatible extra key; all feature and action
dimensions are still validated.

## Exact policy schema

State and action both use this 16-D order and raw joint radians:

```text
zarm_l1..zarm_l7, left_claw, zarm_r1..zarm_r7, right_claw
```

`left_claw` and `right_claw` use `0=open, 1=closed`. The S56 profile:

- holds the simulator's waist joint outside the policy schema;
- clamps the 14 absolute-radian arm targets only to the S56 soft joint limits,
  then converts them exactly to Isaac manager coordinates;
- compresses each selected integrated hand pose to one continuous claw state;
- linearly interpolates every selected hand joint from the continuous predicted
  claw value. The legacy signed endpoints remain `+1=open`, `-1=closed`.

The profile disables the generic manager-coordinate `[-1, 1]` clip by default.
`mean_action_saturation` therefore reports policy values corrected by robot
joint/claw limits rather than the old arbitrary `default +/- 0.45 rad` window.
An explicit `--action-clip VALUE` can still be used for an ablation.

At every reset, the profile loads `observation.state.q50` from the selected
checkpoint's preprocessing statistics and uses its 14 arm angles as the ready
pose. Both claws start fully open. Pass `--initial-pose default` to reproduce
the old all-zero arm reset.

For a task-relevant demonstration start, pass `--initial-pose dataset-medoid`.
This preset is one complete, real `frame_index == 0` state rather than a
per-joint synthetic median. It is the robust medoid of 2,485 episode starts
sampled from six parcel/box-related tasks in
`LejuRobotics/LET-KUAVO-VLA-1.0-Dataset` (revision
`61e3256ac917721c3c9d2098db28b1f5efc2d54a`): task
`078-scan-cardboard-parcels`, episode 234. Both claws start open. The preset
also initializes `zhead_2_joint` at +25 degrees so both hands remain in the
initial head-camera frame. The S56 pitch limit is +30 degrees. You can
override the preset with `--initial-head-pitch-deg DEGREES`.

This option changes only `robot.init_state.joint_pos`. It does not adjust the
head or wrist camera extrinsics, FOV, link hierarchy, or gripper geometry.

Images are supplied in this checkpoint-key order:

```text
observation.images.head_cam_h    -> robustness_camera
observation.images.wrist_cam_l   -> left_wrist_camera
observation.images.wrist_cam_r   -> right_wrist_camera
```

Each image is RGB `3x480x848`. The launcher configures those dimensions automatically.

With the default `s56_twofinger` preset, the complete S200062 gripper and D405
URDF subtrees are transplanted onto the S56 wrist frames. The camera sensors
attach to the physical `l_d405_camera` and `r_d405_camera` links and add only the
camera-body-to-ROS-optical rotation. The optional `s56_qiangnao` preset instead
uses a virtual unobstructed rig because its source asset has no published D405
links. If the training dataset used a separately measured calibration, its
extrinsics still take precedence over either packaged transform.

Only the 14 arm joints and the two hands are policy-controlled. The S56 torso
root is fixed, its 12 leg joints and two head joints remain at their default
targets, and waist yaw is held at its current position. Scene objects remain
physical and may move after contact.

## Download

The fine-tuned checkpoint is about 5.1 GiB. The GR00T N1.5 base model is also
required when it is not already in the Hugging Face cache.

```bash
./download_rwh_kuavo_v2_checkpoint.sh
```

This also populates the GR00T N1.5 base-model cache. Set
`RWH_SKIP_BASE_MODEL_DOWNLOAD=1` only when that base model is already cached.

The fine-tuned checkpoint is pinned to Hub revision
`d6687fa613e2847c67d38a161d1de847bc7b235f` and stored under ignored
`artifacts/huggingface/`.

For a shared model cache or a checkpoint stored elsewhere:

```bash
export HF_HOME=/shared/huggingface
export RWH_KUAVO_V2_CHECKPOINT=/models/RwH-Kuavo_V2/checkpoint-40K/pretrained_model
./eval_rwh_kuavo_v2_s56.sh --headless --episodes 1
```

## Smoke test

Check the S56 scene and adapter without loading GR00T:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --mock-policy \
  --headless --no-camera-preview --no-domain-randomization \
  --episodes 1 --max-steps 5
```

Check one real inference and one queued action:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview --no-domain-randomization \
  --episodes 1 --max-steps 2 \
  --metrics-out artifacts/eval/rwh_kuavo_v2_s56_checkpoint40k_smoke.json
```

## Headless policy and scene video

Headless recording preserves the exact three RGB observations passed to the
policy and, by default, adds a fixed side view for diagnosing whole-body motion.
One MP4 contains the synchronized horizontal layout:

```text
head_cam_h | wrist_cam_l | wrist_cam_r | scene_side
```

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 \
  --video-out artifacts/eval/rwh_s56_pick_box.mp4 \
  --trace-out artifacts/eval/rwh_s56_pick_box_trace.json
```

`--trace-out` records each decoded 16-D joint/claw target, the policy state
immediately before and after the simulation step, and the adapted 17-D manager
command. Use it to distinguish a policy that does not command a grasp from an
actuator or gripper that fails to track a valid command. On a terminal step,
`state_after_is_auto_reset` is true and `state_after` is the automatically reset
state; omit that sample from tracking-error calculations.

At the native checkpoint resolution the output is H.264/yuv420p at
`3392x480`, 10 FPS. `--video-height 360` makes a smaller review copy without
changing the policy input resolution. `--video-fps` changes playback metadata,
not the evaluation control rate. Existing videos are never replaced unless
`--overwrite-video` is specified.

Use `--no-video-scene-view` for the former three-camera-only output. For another
workcell layout, reposition the fixed view without changing policy observations:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --video-out artifacts/eval/custom_view.mp4 \
  --scene-camera-eye -0.2 -3.2 1.8 \
  --scene-camera-target -0.2 0.45 1.0
```

For multiple episodes:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview --episodes 5 \
  --video-out artifacts/eval/rwh_s56_pick_box.mp4
```

This writes `rwh_s56_pick_box_ep000.mp4` through
`rwh_s56_pick_box_ep004.mp4`, and records every absolute path in the metrics
JSON. Install FFmpeg with `libx264` support before using `--video-out`.

To watch the robot live instead, omit `--headless` and enable the previews:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --episodes 1 --max-steps 240 --camera-preview
```

## Full evaluation

The profile now defaults to `pick up the box`. The checkpoint repository
contains `task_index=0` statistics but not the original training text; override
the task if the dataset used a different spelling or instruction.

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 50 \
  --task 'pick up the box' \
  --video-out artifacts/eval/rwh_kuavo_v2_s56_checkpoint40k.mp4 \
  --metrics-out artifacts/eval/rwh_kuavo_v2_s56_checkpoint40k_50ep.json
```

The dedicated launcher defaults to `--no-domain-randomization` for deterministic
compatibility checks. After the nominal task succeeds, append
`--domain-randomization` to run a robustness evaluation; the final CLI option
overrides the launcher's default.

The current manager environment's built-in `success` condition still describes
the rack-to-conveyor task: process every configured box and press the green
button. A `pick up the box` rollout video is useful for behavior inspection,
but that manager success rate is not a valid pick-only benchmark until a
pick-specific success detector is configured.

## Deployment overrides

No user-specific absolute path is baked into the launchers. Configure another
machine with environment variables instead of editing scripts:

| Variable | Purpose |
|---|---|
| `ISAACLAB_PYTHON` | Python executable of the Isaac Lab Conda environment |
| `LEROBOT_PYTHON` | Python executable of a LeRobot 0.5.x GR00T environment |
| `RWH_KUAVO_V2_DIR` | Download root used by the checkpoint downloader |
| `RWH_KUAVO_V2_CHECKPOINT` | Existing local `pretrained_model` directory |
| `RWH_KUAVO_V2_TASK` | Default instruction used by the dedicated launcher |
| `RWH_KUAVO_V2_CONTROL_HZ` | Control/observation rate; defaults to dataset-matched 10 Hz |
| `HF_HOME` | Shared or relocated Hugging Face cache |
| `KUAVO_CONFIG_DIR` | Deployment-specific gripper/layout configuration directory |

The resolver recognizes common `anaconda3`, `miniconda3`, and `miniforge3`
locations, but explicit interpreter paths are recommended in services and CI.

Use different GPUs when available:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --device cuda:0 --policy-device cuda:1 \
  --headless --no-camera-preview --episodes 1
```

Custom scene camera names can retain the checkpoint feature names:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --camera-map head_cam_h=my_head_camera \
  --camera-map wrist_cam_l=my_left_wrist_camera \
  --camera-map wrist_cam_r=my_right_wrist_camera \
  --headless --episodes 1
```

The video follows the order of these repeated mappings. A checkpoint with a
different state/action dimension or feature order needs a new policy profile;
do not force it through this 16-D adapter.

## Validated rollout status

### After the physical four-bar fix (2026-09-02)

The actual checkpoint-40K rollout completed normally after the S56/S200062
[closed-loop correction](GRIPPER_CONFIGURATION.md#s200062--s56-physical-four-bar-closure).
This was not a mock-policy test. Reproduction from the repository root:

```bash
conda activate env_isaaclab_232
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 --seed 42 \
  --control-hz 10 --actions-per-inference 16 \
  --video-out artifacts/eval/rwh_s56_fourbar_fixed_24s.mp4 \
  --metrics-out artifacts/eval/rwh_s56_fourbar_fixed_24s.json \
  --trace-out artifacts/eval/rwh_s56_fourbar_fixed_trace_24s.json
```

Use new output filenames when repeating an experiment. The task remained
`pick up the box`, with checkpoint-q50 arm reset, open claws, 120 Hz physics,
no domain randomization and the unchanged 16-D policy/17-D manager interface.

| Check | Post-fix result |
|---|---|
| Completion | exit 0; 240 steps / 24.0 s; time limit |
| Recorded video | 240 frames, 2544 x 480, 10 FPS; head / left wrist / right wrist |
| Box lift | no successful lift observed in sampled video frames |
| Grippers | both claws received and followed closing commands |
| 14-arm next-step raw-target MAE / p95 | 0.01555 / 0.04276 rad |
| Left / right claw next-step MAE | 0.01234 / 0.01003 on the normalized 0-1 scale |
| Mean action saturation | 4.88%, including claw clipping |
| Existing manager progress / reward | 0.0 / 83.96320 |
| Policy inference calls | 15 |

The right-claw closing target exceeded 0.8 at 11.1-13.1 s and from 22.8 s
onward; the earlier trace had almost no right-claw closure. Tracking statistics
exclude the final auto-reset state and compare claws with clipped [0,1]
targets. Actions and states were finite, and signed manager-to-claw conversion
matched the clipped targets exactly. The trace does not include passive-link
pin gaps or contact forces; those are covered separately by the linkage tests.

The earlier 0.01754-rad arm MAE came from a separate pre-fix trace run, not
the pre-fix video run. Different closed-loop trajectories and one episode per
comparison do not establish a systematic policy improvement. The correction
fixes linkage motion, but this rollout still does not demonstrate successful
picking. Existing manager success also requires the broader conveyor/button
task, so its 0/1 success result is not a pick-only success rate.

The launcher again selected conda `lerobot_050_groot` with LeRobot **0.5.1**;
exact training-runtime parity with 0.5.2 remains unverified. Non-strict loading
reported the same unexpected tied embedding key. Existing tokenizer regex,
imported visual reference and sensor inertia warnings remain. The checkpoint's
saved embodiment routing was not changed. A mid-run GPU snapshot attributed
7441 MiB to this Isaac process and 8830 MiB to its policy worker (~15.89 GiB
combined); this is a snapshot, not a measured peak or minimum VRAM requirement.

### Historical rollout before the physical four-bar fix

**Historical, before the physical four-bar fix.** The recordings named
`corrected_twofinger` removed the leftover dexterous meshes but still had
fixed central bars. Their arm tracking results do not validate gripper
mechanics. The S56/S200062 closed-loop fix is documented in
[Gripper configuration](GRIPPER_CONFIGURATION.md#s200062--s56-physical-four-bar-closure);
the post-fix rollout is reported above.

Validation on 2026-09-02 used the generated S56 articulation with the complete
S200062 two-finger/D405 subtrees, checkpoint q50 arm reset, three native camera
observations, 10 Hz control, 16 actions per inference, and no domain
randomization:

```bash
./eval_rwh_kuavo_v2_s56.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 \
  --video-out artifacts/eval/rwh_s56_pick_box.mp4 \
  --metrics-out artifacts/eval/rwh_s56_pick_box.json \
  --trace-out artifacts/eval/rwh_s56_pick_box_trace.json
```

Observed result:

| Check | Result |
|---|---|
| Simulation duration | 24.0 s / 240 control steps |
| Box contact or lift | not observed |
| Left claw | closed from approximately 4-19 s and tracked the command |
| Right claw | remained open |
| 14-arm next-step target MAE | approximately 0.0175 rad |
| 14-arm target-error p95 | approximately 0.0458 rad |
| Existing manager task progress | 0.0; not a valid pick-only success metric |

An 8-second ablation with `--actions-per-inference 1` produced a larger arm
trajectory and closed the left claw earlier, but still produced no task
progress. Executing the checkpoint's default 16-step chunk is therefore not the
sole explanation for failure in this scene; this pre-fix ablation does not
exclude action-chunk effects after the mechanism correction.

The trace checks action-order, commanded claw interpolation, and arm tracking,
but does not validate passive linkage closure. The subsequently discovered
fixed-bar defect invalidates the earlier inference that the entire gripper
mechanism was correct. It also does not validate grasp contact because the
commanded hand did not reach a box.

### Remaining validation gaps

The post-fix rollout does not resolve these high-priority gaps:

1. Runtime parity. The validated machine automatically selected a functional
   LeRobot 0.5.1 GR00T environment. A separate environment reporting 0.5.2 was
   present but lacked Transformers, PEFT, and Flash Attention and could not load
   the policy. Exact 0.5.2 reproduction and strict checkpoint loading remain a
   release gate. Non-strict loading currently reports one tied LLM embedding
   serialization key as unexpected.
2. Target-task coverage. The checkpoint `train_config.json` names the local
   merged dataset `rwh-kuavo-v2-stage1`; it is not enough to infer that the
   language `pick up the box` and the current scene were in fine-tuning data.
   The public LET-KUAVO dataset contains many industrial tasks and is used here
   as pretraining data, not as a pick-task success specification.
3. Visual domain. The simulated head view was substantially darker than the
   checkpoint's saved training-image quantiles, and the wrist images contained
   large dark/gripper regions. Match exposure, gamma, white balance, FOV,
   camera extrinsics and object visibility before investing in cosmetic texture
   detail. PBR material and lighting variation should follow sensor matching.
4. Evaluation semantics. Implement a pick-specific detector based on opposing
   jaw contact, object height/lift, release from the support surface and a hold
   duration before reporting a pick success rate.
5. Asset hygiene. Empty imported visual references and invalid head sensor
   mass/inertia warnings do not explain the tracked arm behavior, but should be
   removed before a release asset is declared clean.

The checkpoint preprocessor stores `embodiment_tag=new_embodiment` and maps it
to ID `31`. Evaluation loads the saved preprocessor and only overrides its
device, so ID 31 is retained exactly. This is the correct routing ID for this
checkpoint; it does not compensate for different robot link geometry, camera
calibration, or end-effector physics.

Evaluation artifacts are written below ignored `artifacts/eval/` and are not
committed. Metrics and trace JSON are intended to accompany experiment reports.

For a new robot or checkpoint, follow
[Robot model evaluation pipeline](ROBOT_MODEL_EVAL_PIPELINE.md) instead of
weakening the S56 profile validation.

## GPU memory

Measured on one RTX 4090 with one S56 environment and the three native
`480x848` policy cameras:

| Component | Observed VRAM |
|---|---:|
| S56 Isaac Sim and RTX cameras | approximately 6.1-7.5 GiB |
| GR00T N1.5 worker | approximately 8.8 GiB |
| Stable combined use | approximately 16.9 GiB |

A 24 GiB GPU is recommended for single-GPU deployment. A 16 GiB GPU has
insufficient margin for this native-resolution configuration. Desktop UI and
camera preview can consume additional memory; headless mode is preferred for
batch evaluation.

## Troubleshooting

- `Unable to find a LeRobot 0.5.x environment`: set `LEROBOT_PYTHON` to the
  exact Conda Python executable and verify the GR00T import there.
- `Checkpoint ... model.safetensors is missing`: run the downloader or set
  `RWH_KUAVO_V2_CHECKPOINT` to the local `pretrained_model` directory.
- `ffmpeg is required`: install FFmpeg or omit `--video-out`.
- `Video already exists`: choose a new path or intentionally pass
  `--overwrite-video`.
- CUDA out of memory: close other GPU applications, use headless mode, or split
  Isaac and policy across `cuda:0` and `cuda:1`.

# Robot model evaluation pipeline

This repository separates reusable evaluation infrastructure from
checkpoint-specific policy adapters. A new robot can reuse simulation startup,
the LeRobot subprocess, video recording, metrics and traces, but it must satisfy
an explicit robot/action/camera contract.

## Pipeline layers

Python file paths below are relative to `src/kuavo_isaaclab_scene/`.
See [code structure](CODE_STRUCTURE.md) for the full package map.

| Layer | Responsibility | Main files |
|---|---|---|
| Robot asset | Complete USD/URDF, body and joint names, default pose | `robots/robot_model.py`, `assets/` |
| End effector | Preset, hand joints, open/close convention, actuator gains | `configs/grippers.json`, `robots/gripper_runtime.py` |
| Hand feedback | Independent left/right measured joint and normalized claw views | `robots/gripper_io.py` |
| Isaac manager | Physics, 15-D upper body plus configured hand actions | `envs/manager_env.py`, `envs/scene_physics.py` |
| Policy profile | State/action order, units, limits, camera keys | `evaluation/groot_lerobot_bridge.py`, `evaluation/eval_groot.py` |
| Policy worker | Load LeRobot in a separate Conda environment and return chunks | `evaluation/groot_policy_worker.py` |
| Evidence | Metrics JSON, synchronized camera MP4 and optional per-step trace | `evaluation/eval_metrics.py`, `display/eval_video.py` |

The generic entry point is `eval_groot.sh`. A checkpoint-specific root wrapper,
such as `eval_rwh_kuavo_v2_s56.sh`, should only supply validated defaults and
must leave the generic evaluator reusable.

## Compatibility boundary

Do not decide compatibility from the robot name alone.

### Reusable without a new policy adapter

A registered robot normally reuses the default pipeline when all of these are
true:

- every controlled joint resolves in the same order and uses the same units;
- the environment manager action dimension matches the checkpoint output;
- gripper actions have the same meaning and range;
- every checkpoint camera key maps to a sensor with the expected shape;
- the checkpoint's state/action preprocessing matches the selected modes.

Changing only mesh detail or inertial estimates does not normally require a new
profile when the public contract above stays unchanged. It still requires an
Isaac spawn/step test and camera inspection.

### Shared 16-D arm/claw profile

Select `--policy-profile kuavo-arm-claw` to use the shared adapter with the
selected robot and two configured hands. The existing `rwh-kuavo-v2-s56` CLI
profile retains its S56-only defaults and delegates to the same adapter.
Both profiles enforce:

```text
state/action: left arm 7, left_claw, right arm 7, right_claw
units:        arm radians; claw 0=open, 1=closed
images:       head_cam_h, wrist_cam_l, wrist_cam_r at 3x480x848
```

The manager receives 17 actions: waist, left arm 7, right arm 7, left/right
gripper. The waist is held at its current position. Arm targets are clamped to
the selected robot's soft limits; claw commands interpolate the selected
preset's open/close joint poses continuously. Camera resolution defaults to
848x480 but can be explicitly overridden; body mounts/extrinsics remain
model-specific. S200062 does not inherit the S56-only 25-degree head preset.

Both arm/claw profiles now default to `--body-mode fixed`: the articulation
root is fixed, and wheel/leg/knee/waist/head joints are physically constrained
within ±1e-4 rad of their reset pose before stabilization. Arms and all hand
linkage joints remain free. `--body-mode pd` opts into the previous body-drive
behavior; it does not guarantee a stationary torso under load. Legacy profiles
default to `pd`. See [initial states](INITIAL_STATES.md) for reset semantics and
body constraint diagnostics. These constraints isolate manipulation; they do
not validate real-robot balance or torso controllers.

`GripperView` resolves integrated hand joints on `scene["robot"]` and external
hands on `scene["left_gripper"]` / `scene["right_gripper"]`. Both expose the
same measured `joint_state()` and `claw_state()` API. This is a **logical**
separation: no USD articulation, four-bar loop constraint or camera parent is
changed. Missing enabled hands/joints or inconsistent manager dimensions fail
at startup rather than silently dropping state. The current state, not the
last command, is used after every simulation step and reset.

The `default` profile still exposes hand joints individually. For S200062 and
S56 two-finger this is now 19-D state (15 upper-body + 2 left + 2 right), fixing
the previous integrated-hand omission. With manager actions it has 17 action
values **and 17 names**. External-hand all-joint state layout is preserved;
its dimension can differ. Absolute/delta default-profile actions remain 15-D
upper-body only and hold grippers open. Checkpoints trained against the old
incomplete 15-D state must not be padded silently; select the correct profile.
Metrics include `state_names`, `policy_action_names`, `manager_action_names`
and `gripper_views` to make the actual mapping inspectable. Metrics format v3
and trace format v2 identify this corrected schema; manager joint labels now
use the actual joint names (e.g. `waist_yaw_joint`). Read the name arrays rather
than assuming 15/17 state dimensions or stripping suffixes implicitly.

Shared feature dimensions do not prove physical policy transfer. Another
robot still requires validation of joint meaning, limits, hand calibration,
camera extrinsics and checkpoint preprocessing. Robotiq/QiangNao and two-finger
hands share an interface, not identical grasp mechanics.

To verify actual left/right open, close and intermediate feedback over two
reset cycles without loading a checkpoint (activate the Isaac Lab conda env):

```bash
python scripts/verify_gripper_io.py --headless \
  --robot-model s200062 --gripper s200062_integrated \
  --output artifacts/eval/s200062_gripper_io.json
```

Use a new output path for each run. The diagnostic fails if tracking error
exceeds 0.1 claw units or state/action names do not match tensor dimensions.
It is an I/O integration check, not a grasp-success evaluation.

Local validation on 2026-09-07 (30 control steps per pose, two resets):

| Model / hand | Default state / policy state / manager action | Maximum claw tracking error | Result |
|---|---|---|---|
| S200062 / integrated two-finger | 19 / 16 / 17 | 0.00290 | Pass |
| S56 / two-finger | 19 / 16 / 17 | 0.00310 | Pass |
| S63 / external Robotiq | 31 / 16 / 17 | 0.56089 | Feedback mapping works; physical tracking fails |

S63's existing preset produced measured closure around 0.44 for a 1.0 close
target in this workcell/reset pose. Its physical tracking/preset must be
investigated separately before treating it as a validated replacement. The
interface refactor does not fix or recalibrate that asset. Reports are in
`artifacts/eval/{s200062,s56,s63}_gripper_io_20260907.json` (ignored by Git).

An additional Stage 1 checkpoint run reached model load, the 16D schema check
and the 1-second initial hold, but its first inference ran out of GPU memory
while a separate Quest teleoperation/CloudXR session was active. No checkpoint
rollout success is claimed from that attempt; free sufficient VRAM before
repeating the full eval. The other session was not stopped.

### Requires a new profile

Create a new policy profile when any of these change:

- joint count, order, units, absolute/delta representation or normalization;
- single claw versus multi-joint hand semantics;
- image feature names, order, count or shape;
- base/leg/head actions included in the policy;
- checkpoint-specific reset pose or preprocessing.

Never pad, truncate, reorder or mirror a checkpoint action silently merely to
make dimensions pass.

## New robot onboarding checklist

1. Add the source-derived URDF, meshes and generated USD below `assets/`.
2. Register a `RobotModelSettings` entry with spawn height, integrated hand,
   head camera body, wrist camera bodies and transforms.
3. Add or select a gripper preset. Define every active joint exactly once and
   verify that open/close commands move in the intended direction.
4. Convert assets with `scripts/convert_kuavo.sh` or a dedicated reproducible
   builder. Do not hand-edit generated USD layers.
5. Run a gripperless articulation when replacing an integrated hand. A filename
   such as `nohand.STL` is not evidence that the mesh actually lacks fingers.
6. Spawn and step the bare and configured variants in Isaac Sim.
7. Render head and wrist observations and verify optical-axis convention,
   handedness, target visibility, FOV and resolution.
8. Add the policy profile or explicitly prove that an existing one matches.
9. Run a mock-policy smoke test before loading a multi-gigabyte checkpoint.
10. Run one real chunk, then a full rollout with MP4, metrics and trace.
11. Validate task-specific success semantics; manager reward alone is not a
    universal manipulation metric.
12. Run all unit tests and `git diff --check` before release.

Example smoke sequence:

```bash
./run_manager_env.sh \
  --robot-model MODEL --gripper PRESET \
  --headless --num-envs 1 --steps 1

./eval_groot.sh \
  --robot-model MODEL --gripper PRESET \
  --mock-policy --headless --no-camera-preview \
  --episodes 1 --max-steps 5
```

Use a checkpoint-specific launcher for the real rollout and record all three
evidence outputs:

```bash
./checkpoint_eval_wrapper.sh \
  --headless --no-camera-preview \
  --episodes 1 --max-steps 240 \
  --video-out artifacts/eval/rollout.mp4 \
  --metrics-out artifacts/eval/rollout.json \
  --trace-out artifacts/eval/rollout_trace.json
```

## What the trace proves

`--trace-out` stores decoded policy actions, policy state immediately before and
after each step, adapted manager commands, clipping/saturation and terminal
state. It can distinguish:

- a policy that never commands approach or closure;
- incorrect action ordering or claw sign;
- joint-limit clipping;
- a controller that does not track a valid target.

It cannot prove collision quality or grasp stability unless the hand reaches
and contacts the object. Terminal steps are auto-reset by `ManagerBasedRLEnv`;
omit entries where `state_after_is_auto_reset=true` from tracking-error
statistics.

## Visual-domain guidance

Photorealism is not the direct target. Match the policy's sensor distribution
in this order:

1. camera pose, handedness, optical axis, FOV and target framing;
2. exposure, gamma, white balance, black level and clipping;
3. object scale, shape, color and foreground/background contrast;
4. material roughness, specular response, lighting and shadows;
5. bounded texture, lighting and camera domain randomization.

An embodiment ID selects a learned routing embedding; it does not repair visual
or kinematic mismatch. Load the ID saved with the checkpoint, log it at startup,
and never substitute another existing embodiment ID as a calibration shortcut.

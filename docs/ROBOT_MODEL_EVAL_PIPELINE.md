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

### Requires extending an existing profile

The `rwh-kuavo-v2-s56` profile is intentionally limited to S56 with an
integrated `s56_twofinger` or `s56_qiangnao` hand. It enforces:

```text
state/action: left arm 7, left_claw, right arm 7, right_claw
units:        arm radians; claw 0=open, 1=closed
images:       head_cam_h, wrist_cam_l, wrist_cam_r at 3x480x848
```

Another robot may be physically similar but is not accepted automatically.
Extend the profile only after its joint order, limits, gripper convention and
camera calibration have been verified. Keep the old model path working and add
tests for every allowed model/gripper combination.

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

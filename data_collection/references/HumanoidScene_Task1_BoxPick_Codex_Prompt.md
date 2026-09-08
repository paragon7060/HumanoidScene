# HumanoidScene Task 1 — Box Pick Dataset Auto-Collection Implementation Prompt

Repository:
- https://github.com/paragon7060/HumanoidScene

## Goal

Implement an **automatic dataset collection pipeline for Task 1 only: picking/extracting a box from the rack**.

Do **not** implement:
- turning around,
- locomotion to the rail/conveyor,
- placing the box,
- button pressing,
- full-task chaining.

The current scope is strictly:

> **Rack-aligned robot → approach box → grasp → pull box out of rack → lift/stabilize → episode success**

The generated dataset should be useful for:
1. training a VLA/policy later,
2. offline debugging/analysis,
3. future 6D-conditioned policy experiments,
4. sim-to-real comparison,
5. re-deriving different action representations without recollecting data.

Therefore, **store as much simulator ground-truth and control information as reasonably possible**, but keep simulator-only GT clearly separated from policy observations.

---

# 1. First inspect the existing repository

Before editing code, inspect and reuse the current HumanoidScene structure.

Relevant existing components already present in the repo include:

- rack box types:
  - `small`
  - `medium`
  - `large`
  - `xlarge`

- task/phase structure:
  - `approach_rack`
  - `pick`
  - `carry`
  - `place`
  - `press_button`

- box geometry extraction:
  - box center
  - half-size
  - flap geometry

- box/task pose utilities

- `box_pose_only()` style robot-relative 6D pose computation

- contact / grasp detection

- Meta Quest teleoperation recorder

- HDF5 + LeRobot Dataset v3 writer

- per-frame data such as:
  - robot joint position
  - robot joint velocity
  - EE pose
  - camera RGB
  - box root pose
  - action

- LeRobot writer:
  - `src/kuavo_isaaclab_scene/recording/teleop_lerobot_recorder.py`
  - `src/kuavo_isaaclab_scene/recording/lerobot_writer_worker.py`

- task / geometry code:
  - `src/kuavo_isaaclab_scene/rl/tasks/specs.py`
  - `src/kuavo_isaaclab_scene/rl/mdp/observations.py`
  - `src/kuavo_isaaclab_scene/rl/mdp/commands.py`
  - `src/kuavo_isaaclab_scene/rl/mdp/flap_grasp.py`
  - `src/kuavo_isaaclab_scene/rl/scenes/asset_geometry.py`

Do not reimplement functionality that already exists.

---

# 2. Important design principle

Separate:

## Policy-visible observation

Things that a real robot policy can actually observe, e.g.

- RGB
- robot joint state
- hand state
- EE state, if we decide to use it later

from:

## Privileged simulator information

Things available only from Isaac Sim / scripted expert, e.g.

- exact box 6D pose
- target grasp pose
- box velocity
- contact force
- task phase
- exact grasp success

**All simulator-only GT must be stored, but must be clearly namespaced as `privileged.*`.**

Do not silently feed simulator GT into the policy observation.

---

# 3. Box pose convention

A 6-DoF pose should be stored numerically as:

```text
[x, y, z, qw, qx, qy, qz]
```

Use this convention consistently unless the repository already has a canonical convention that must be preserved.

If the repository uses another quaternion order internally, convert explicitly at dataset serialization boundaries and document it.

Store both:

```text
privileged.box_pose_w
privileged.box_pose_robot
```

where:

```text
T_robot_box = inverse(T_world_robot) * T_world_box
```

Do not rely only on world-frame pose.

---

# 4. Box types and grasp definition

There are four geometry types:

```text
small
medium
large
xlarge
```

We want a **box-type-specific grasp configuration**, defined in the **box local frame**.

Do not make separate grasp parameters for every box instance if multiple instances share the same geometry type.

Conceptually:

```text
small:
    left_grasp_pose_box
    right_grasp_pose_box

medium:
    left_grasp_pose_box
    right_grasp_pose_box

large:
    left_grasp_pose_box
    right_grasp_pose_box

xlarge:
    left_grasp_pose_box
    right_grasp_pose_box
```

Each grasp pose should be a full pose, not position only:

```text
[x, y, z, qw, qx, qy, qz]
```

Prefer deriving sensible defaults from existing box/flap geometry if possible, but make the final values configurable.

Do not bury these values inside control logic.

Put them in a clear config structure/file so they can later be tuned manually.

---

# 5. Grasp/world transform

Given:

```text
T_world_box
T_box_grasp_left
T_box_grasp_right
```

compute:

```text
T_world_grasp_left  = T_world_box * T_box_grasp_left
T_world_grasp_right = T_world_box * T_box_grasp_right
```

The system should use the current GT box pose every control step or at a clearly defined planning point.

Be explicit about which behavior is used.

---

# 6. Task 1 trajectory

Implement a simple deterministic scripted expert for dataset generation.

Suggested phase structure:

```text
0 = pregrasp
1 = approach
2 = grasp
3 = pull
4 = lift
5 = stabilize
```

Use an enum or named constants rather than magic integers.

Expected sequence:

## Phase 0 — pregrasp

Move both EEs to a safe pregrasp offset from the target grasp pose.

The offset should be configurable, not hard-coded deep inside the controller.

Example concept:

```text
pregrasp = grasp_pose + offset_along_approach_axis
```

## Phase 1 — approach

Move from pregrasp to the exact configured left/right grasp poses.

Use smooth interpolation / bounded per-step motion.

Do not teleport the hands.

## Phase 2 — grasp

Close the hand/gripper.

Wait until a robust grasp/contact condition is satisfied.

Reuse existing contact/grasp logic where possible.

## Phase 3 — pull

Extract the box outward from the rack while maintaining the grasp relation.

The pull direction should be defined relative to the rack/box/workcell geometry, not as an unexplained world-axis magic number.

Use a configurable pull distance.

## Phase 4 — lift

Lift the extracted box by a configurable amount.

Initial candidate:
- around 0.10 m

Do not assume this exact value if existing geometry indicates another safer value.

## Phase 5 — stabilize

Hold the box for a short dwell time and require reasonable pose/velocity stability.

Initial candidate:
- around 0.30 s

The repository already contains similar concepts such as `lift_height`, `hold_seconds`, settle speed, angular speed, etc. Reuse them where appropriate.

---

# 7. Task success

A successful Task 1 episode should mean more than “the fingers touched the box”.

At minimum consider:

- intended box is grasped,
- box was extracted from its initial rack position,
- box was lifted,
- box remains reasonably upright,
- box linear/angular velocity is sufficiently low during stabilization,
- grasp is still maintained at the end,
- no drop/failure condition occurred.

Store the exact success criteria in code and document them.

Also store:

```text
success
failure_reason
```

for analysis.

Useful failure reasons could include:

```text
ik_failure
reach_timeout
grasp_timeout
lost_grasp
box_dropped
extraction_timeout
lift_timeout
unstable
collision
episode_timeout
```

Use repository-native failure semantics if already available.

---

# 8. Dataset schema

Implement/extend the recorder so Task 1 can save the following information.

Dimensions that depend on robot configuration must be derived dynamically.

Do **not** hard-code `J`, `J_arm`, or `G` unless the actual model contract proves fixed.

---

## 8.1 Observation

```text
observation.images.head
    shape: [H, W, 3]

observation.images.left_wrist
    shape: [Hw, Ww, 3]

observation.images.right_wrist
    shape: [Hw, Ww, 3]

observation.joint_position
    shape: [J]

observation.joint_velocity
    shape: [J]

observation.hand_state
    shape: [G]

observation.ee_pose
    shape: [14]
```

`observation.ee_pose` order:

```text
left_x
left_y
left_z
left_qw
left_qx
left_qy
left_qz

right_x
right_y
right_z
right_qw
right_qx
right_qy
right_qz
```

Preserve actual robot joint names.

---

# 9. Store all three action representations separately

Do not collapse them into one anonymous vector.

Store:

## 9.1 EE target

```text
action.ee_target
shape: [14]
```

Meaning:

```text
left desired EE absolute pose
right desired EE absolute pose
```

Order:

```text
left_x
left_y
left_z
left_qw
left_qx
left_qy
left_qz

right_x
right_y
right_z
right_qw
right_qx
right_qy
right_qz
```

Use an explicit reference frame and document it.

Prefer matching the real Kuavo control convention.

---

## 9.2 Arm joint target

```text
action.arm_joint_target
shape: [J_arm]
```

This is the actual arm joint target produced by IK/controller.

Use real joint names as dimension names.

Do not call them generically `joint_1`, `joint_2`, etc. if actual names are available.

---

## 9.3 Hand target

```text
action.hand_target
shape: [G]
```

Store the actual hand/gripper command sent to the simulated robot.

Again, use actual channel/joint names.

Document:
- value units,
- valid range,
- meaning of open/close values.

---

# 10. Consider also storing EE delta action

If easy and unambiguous, additionally store:

```text
action.ee_delta
```

but this is optional.

The three mandatory action groups are:

```text
action.ee_target
action.arm_joint_target
action.hand_target
```

The important point is that later we should be able to choose which action representation to train without recollecting data.

---

# 11. Privileged simulator features

Store the following whenever available.

## 11.1 All box poses

```text
privileged.box_pose_w
shape: [N_box, 7]

privileged.box_pose_robot
shape: [N_box, 7]
```

If the LeRobot feature schema requires flattened vectors, flatten consistently and provide dimension names such as:

```text
box_0_x
box_0_y
box_0_z
box_0_qw
...
box_1_x
...
```

---

## 11.2 Box velocity

```text
privileged.box_velocity_w
shape: [N_box, 6]
```

Per box:

```text
vx
vy
vz
wx
wy
wz
```

---

## 11.3 Target box identity

```text
privileged.target_box_id
shape: [1]
```

Also keep convenient duplicated target-only features:

```text
privileged.target_box_pose_w
shape: [7]

privileged.target_box_pose_robot
shape: [7]
```

This duplication is intentional for easier analysis.

---

## 11.4 Box-local configured grasp poses

```text
privileged.left_grasp_pose_box
shape: [7]

privileged.right_grasp_pose_box
shape: [7]
```

These are the type-specific grasp references.

---

## 11.5 Task phase

```text
privileged.phase
shape: [1]
```

Suggested semantic labels:

```text
pregrasp
approach
grasp
pull
lift
stabilize
```

The numeric mapping must be documented.

---

## 11.6 Contact and grasp state

Store as much useful contact information as practical, e.g.

```text
privileged.finger_contact_force
privileged.is_grasped
```

If possible store actual force vectors or per-finger scalar force rather than only the final boolean.

---

## 11.7 Initial/current box relation

Store episode-level:

```text
target_box_initial_pose_w
```

and preferably frame-level:

```text
privileged.target_box_delta_pose
```

where conceptually:

```text
delta_T = inverse(T_world_box_initial) * T_world_box_current
```

This will later make extraction-distance and motion analysis easy.

---

## 11.8 Success

Store:

```text
privileged.success
```

per frame if useful, plus episode-level success/failure metadata.

---

# 12. Dataset-level metadata: `meta/info.json`

This is important.

Every vector feature must contain:

- dtype
- shape
- human-readable dimension names

Example:

```json
"action.ee_target": {
  "dtype": "float32",
  "shape": [14],
  "names": [
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz"
  ]
}
```

For joint/action vectors, names must reflect the actual robot channels.

Example:

```json
"action.arm_joint_target": {
  "dtype": "float32",
  "shape": [J_arm],
  "names": [
    "<actual_left_arm_joint_name_0>",
    "...",
    "<actual_right_arm_joint_name_n>"
  ]
}
```

Do not leave `names: null` for vectors when channel semantics are available.

Use LeRobot Dataset v3 standard fields and avoid adding arbitrary custom fields directly into `info.json` if they are outside the expected schema.

---

# 13. Add `meta/feature_semantics.json`

In addition to LeRobot's standard `meta/info.json`, create a sidecar:

```text
meta/feature_semantics.json
```

Purpose:
- coordinate frame definitions,
- units,
- quaternion order,
- absolute vs delta semantics,
- value ranges,
- feature descriptions.

Example structure:

```json
{
  "pose_convention": {
    "position_unit": "meter",
    "quaternion_order": "wxyz"
  },

  "observation.ee_pose": {
    "description": "Measured left/right end-effector pose",
    "reference_frame": "world",
    "position_unit": "meter",
    "quaternion_order": "wxyz"
  },

  "action.ee_target": {
    "description": "Desired absolute left/right end-effector target",
    "reference_frame": "world",
    "position_unit": "meter",
    "quaternion_order": "wxyz",
    "representation": "absolute"
  },

  "action.arm_joint_target": {
    "description": "Arm joint targets produced by the IK/controller",
    "unit": "radian"
  },

  "action.hand_target": {
    "description": "Command sent to the left/right hand/gripper",
    "unit": "<derive from actual hand controller>",
    "range": "<derive from actual hand controller>"
  },

  "privileged.box_pose_w": {
    "description": "Ground-truth box root pose from Isaac Sim",
    "reference_frame": "world",
    "position_unit": "meter",
    "quaternion_order": "wxyz"
  },

  "privileged.box_pose_robot": {
    "description": "Ground-truth box pose relative to robot base",
    "reference_frame": "robot_base",
    "position_unit": "meter",
    "quaternion_order": "wxyz"
  }
}
```

Generate this file automatically when the dataset is created.

If it already exists on resume, validate compatibility where practical.

---

# 14. Episode metadata

Use the existing episode metadata sidecar mechanism if possible.

Useful episode-level metadata:

```text
episode_id
task = "pick_box_from_rack"

robot_model
joint_names
action_names

target_box_id
target_box_type

target_box_initial_pose_w

box_types
box_geometry
    center
    half_size

rack_pose_w

success
failure_reason

fps
control_dt
seed
```

If an existing file such as:

```text
meta/kuavo_episode_metadata.jsonl
```

already exists, extend/reuse it rather than inventing unnecessary duplicate formats.

---

# 15. Real-robot compatibility

The final dataset should be designed with eventual real Kuavo use in mind.

The real control/data pipeline contains concepts such as:

```text
head RGB
wrist RGB
robot state
hand state
arm joint target
hand target
EE pose
IK outputs
```

Therefore:

- use real joint/channel names where possible,
- preserve action decomposition,
- avoid simulator-specific action representations unless clearly marked,
- keep GT 6D as privileged information,
- keep camera/state/action tensors compatible with future real data conversion.

The scripted expert is allowed to use exact simulator box 6D pose.

The trained policy should not be forced to depend on GT box 6D.

---

# 16. No hard-coded dimensions when avoidable

The following must be derived from the actual loaded robot/model/controller:

```text
J
J_arm
G
joint_names
arm_joint_names
hand/channel names
```

Do not assume S200062/S56/S63 dimensions are identical.

The implementation should work with the active HumanoidScene robot model when feasible.

If a feature truly cannot be made model-agnostic, document the limitation explicitly.

---

# 17. Configuration

Make key expert parameters configurable, including at least:

```text
box-type grasp poses
pregrasp offset
approach speed / interpolation rate
pull distance
pull direction convention
lift distance
hold/stabilization time
position tolerance
orientation tolerance
grasp timeout
motion timeout
settle linear velocity
settle angular velocity
contact/grasp threshold
```

Avoid hidden magic constants.

Prefer a dedicated config/dataclass/YAML/JSON consistent with the repository style.

---

# 18. Dataset collection CLI

Add a clear command/script for automatic Task 1 collection.

Example desired user experience:

```bash
./collect_box_pick_dataset.sh \
  --robot-model <...> \
  --episodes 100 \
  --rack-boxes <...> \
  --dataset-root datasets/box_pick_v1 \
  --dataset-format both
```

Exact CLI naming may follow current repo conventions.

Useful options:

```text
--episodes
--target-box / --box-type / --boxes
--dataset-root
--repo-id
--dataset-format
--headless
--device
--seed
--save-failed
--grasp-config
```

If selecting boxes randomly, record the selected target in episode metadata.

---

# 19. Randomization

For v0.1, prioritize correctness over aggressive randomization.

However, the pipeline should be structured so future randomization is easy.

Potential future randomization dimensions:

```text
box rack position
box small translation jitter
box yaw/rotation jitter
robot initial posture
camera noise
friction
lighting
```

Do not make broad randomization a prerequisite for the first implementation.

---

# 20. Testing

Add tests where practical.

At minimum verify:

## Schema

- all required features exist,
- shapes match,
- names length == flattened vector dimension,
- no unnamed vector channels when names are available,
- `info.json` is valid LeRobot v3 metadata,
- `feature_semantics.json` is created.

## Transform correctness

Check:

```text
T_world_grasp ≈ T_world_box * T_box_grasp
```

for known synthetic poses.

Check robot-relative pose transforms.

## Dataset resume

Existing dataset resume should:
- validate feature shape/dtype,
- reject incompatible schemas,
- preserve metadata.

## Expert rollout smoke test

Run one short headless episode and verify:

```text
pregrasp -> approach -> grasp -> pull -> lift -> stabilize
```

and confirm all required fields are recorded.

---

# 21. Logging/debug visualization

Add concise debug logging for:

```text
target box
box type
phase transition
current grasp error
contact/grasp status
pull displacement
lift height
success/failure reason
```

If easy, provide optional visualization markers for:
- left/right grasp target,
- pregrasp target,
- current box frame.

Do not require visualization for headless collection.

---

# 22. Implementation philosophy

Please follow these rules:

1. Inspect existing code before adding new abstractions.
2. Reuse current HumanoidScene recorder/task/contact/geometry utilities.
3. Keep changes modular.
4. Do not break existing Quest teleoperation collection.
5. Do not break existing GR00T evaluation.
6. Preserve LeRobot v3 compatibility.
7. Avoid duplicate sources of truth for box geometry.
8. Avoid hard-coded robot dimensions.
9. Explicitly document coordinate frames.
10. Save rich GT data, but keep it under `privileged.*`.
11. Keep all three action representations separately.
12. Prefer actual joint/channel names in metadata.
13. Add tests for transforms and schema.
14. Do not implement Task 2 or locomotion in this task.

---

# 23. Desired final output from you

After implementing, report:

## A. Files changed

List each file and what changed.

## B. Dataset schema

Print the final exact feature schema with:
- name,
- shape,
- dtype,
- dimension names.

## C. Coordinate conventions

Explicitly state:
- world frame,
- robot/base frame,
- EE frame,
- box frame,
- quaternion order,
- units.

## D. Action contract

Explicitly state the final dimensions and names of:

```text
action.ee_target
action.arm_joint_target
action.hand_target
```

Do not guess these dimensions before inspecting the actual robot/controller.

## E. Task 1 expert behavior

Summarize the implemented phase transitions and success criteria.

## F. Commands

Provide:
- one smoke-test command,
- one example dataset collection command.

## G. Validation

Report which tests/smoke checks passed.

## H. Remaining uncertainties

If any real-robot convention cannot be determined confidently from this repository, clearly flag it rather than inventing a value.

---

# 24. Key conceptual summary

The intended architecture is:

```text
Isaac Sim GT box 6D
        ↓
box type
        ↓
box-local left/right grasp pose
        ↓
world left/right EE target
        ↓
scripted smooth trajectory
        ↓
IK/controller
        ↓
arm_joint_target + hand_target
        ↓
robot simulation
        ↓
RGB + robot state + EE state
        ↓
LeRobot/HDF5 dataset
```

Store simultaneously:

```text
policy observation
expert action representations
controller targets
privileged GT
episode metadata
```

The main objective is:

> Collect a rich Task 1 box-picking dataset once, while preserving enough information that we can later train with EE actions, joint actions, GT-assisted experiments, or sim-to-real analysis without recollecting the simulation dataset.

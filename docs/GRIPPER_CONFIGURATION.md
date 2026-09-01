# Gripper configuration

The default `s200062` robot contains its own two-finger grippers. The
`s200062_integrated` preset controls four linkage joints per side directly on
`scene["robot"]`; it does not spawn another hand asset. S56 supports two
complete integrated variants: `s56_qiangnao` with ten physical joints per hand,
and `s56_twofinger`, which transplants the S200062 four-bar grippers and their
physical D405 links. `--gripper none` selects a third generated S56 articulation
with bare S200062 wrist shells and no hand geometry. Only the
`--robot-model s63` comparison mode defaults to the external Robotiq-based
Leju claws from the 2026 OpenLET challenge model.

## Run

The full S200062 model is enabled by default:

```bash
./run_scene.sh
./run_manager_env.sh --num-envs 1 --steps 100000
```

Run either integrated S56 hand, compare an external-Robotiq model, or disable
gripper action channels:

```bash
./run_scene.sh --robot-model s63 --gripper robotiq_2f85
./run_scene.sh --robot-model s56 --gripper s56_qiangnao
./run_scene.sh --robot-model s56 --gripper s56_twofinger
./run_scene.sh --robot-model s56 --gripper none
./run_manager_env.sh --robot-model s200062 --gripper none --num-envs 1
```

`s56_twofinger` is not layered on top of the QiangNao USD. Selecting it swaps
the complete robot articulation to the generated
`assets/kuavo_s56_twofinger/usd/kuavo_s56_twofinger_fixed.usd`. The QiangNao
links are absent, the S200062 bare wrist replaces the S56 source mesh named
`*_hand_pitch_nohand.STL` (that misleading mesh actually contains a complete
five-finger hand), and the original S200062 linkage/camera subtrees attach
directly to it. The wrist sensors use the transplanted physical D405 frames
with only the ROS optical-axis correction. `--gripper none` selects
`assets/kuavo_s56_bare/usd/kuavo_s56_bare_fixed.usd` for geometry-level removal;
it is not a visibility toggle on the QiangNao articulation.

The standard manager action order is:

```text
waist_yaw(1), left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)
```

That is 17 dimensions with the default two hands and 15 dimensions with
`--gripper none`. A positive gripper value opens the hand and a negative value
closes it. Meta Quest and browser teleoperation append two channels to the
existing 14-D bimanual IK/head command, producing a 16-D action. Pinch distance
at or below 0.055 m closes the corresponding hand.

S56 is a fixed-root biped rather than a wheel-base model. Its source home pose
places the torso root at 0.98 m, so the launcher applies that height
automatically. Teleoperation keeps the existing six-channel body command
schema, but the three telescopic-height channels are no-ops for S56.

## Change an external S63 gripper or mount pose

Edit the deployment file [`configs/grippers.json`](../configs/grippers.json),
or preserve it and pass a different file:

```bash
./run_scene.sh --robot-model s63 --gripper custom \
  --gripper-config /data/workcell/grippers.json
```

Each enabled preset defines:

- `usd_path`: default hand USD; `${KUAVO_PACKAGE_ASSET_DIR}` resolves to this
  package's checked-in asset directory and ordinary relative paths resolve
  beside the JSON file;
- `attachment_mount_body`: rigid mount prim inside the hand USD;
- `joint_names`, `default_joint_pos`, `open_command`, `close_command`;
- implicit actuator effort, stiffness, damping, and friction;
- `sides.left` and `sides.right`, each with an optional USD/mount override and
  a Kuavo `robot_mount_body`, local translation, and `(w,x,y,z)` rotation.

For example, after visually calibrating a left-hand mount in Isaac Sim, copy
its local pose into:

```json
{
  "robot_mount_body": "zarm_l7_end_effector",
  "robot_mount_pos": [0.0, 0.0, 0.0],
  "robot_mount_rot": [1.0, 0.0, 0.0, 0.0]
}
```

The mount Xform appears at:

```text
/World/envs/env_0/Kuavo/zarm_l7_end_effector/KuavoGripperMount
/World/envs/env_0/Kuavo/zarm_r7_end_effector/KuavoGripperMount
```

Pause simulation before editing. Record the mount's local transform relative
to its end-effector parent, update the JSON, and restart; edits made only in a
composed running stage are not automatically written back to configuration.

## Packaged Robotiq 2F-85 / Leju claw assets

The shared left/right USD is generated from a PhysX tree-articulation port of
OpenLET's challenge MJCF. The source revision, meshes, original MJCF, URDF port,
and generated USD are all repository-local:

```text
${KUAVO_PACKAGE_ASSET_DIR}/robotiq_2f85/mjcf/robotiq_2f85.xml
${KUAVO_PACKAGE_ASSET_DIR}/robotiq_2f85/urdf/robotiq_2f85.urdf
${KUAVO_PACKAGE_ASSET_DIR}/robotiq_2f85/usd/robotiq_2f85.usd
```

They need no network access. Each source hand has two closed-loop four-bar
branches driven through one tendon. URDF does not encode that MuJoCo
tendon/equality loop, so the PhysX port keeps eight revolute tree joints and one
binary action sends synchronized driver/coupler/spring/follower targets. The
pad-only box colliders remain available for object contact, while internal
self-collision is disabled to avoid fighting those synchronized targets.

Rebuild the robot and gripper USDs after changing their URDFs:

```bash
export ISAACLAB_DIR=/absolute/path/to/IsaacLab-v2.3.2
./scripts/convert_kuavo.sh
```

The converter first regenerates `kuavo_s56_bare.urdf` and
`kuavo_s56_twofinger.urdf` from the untouched S56 and S200062 source-derived
URDFs, then converts all S56 variants. Do not edit the generated URDFs by hand;
change the generator or donor asset and rebuild.

The robot and gripper assets are repository-local. The workcell environment
keeps its original NVIDIA warehouse and Digital Twin conveyor runtime assets.

## LeRobot and GR00T compatibility

New Quest demonstrations store gripper joint state and the two binary gripper
actions in the dataset schema. Keep the same preset for collection, training,
and evaluation:

```bash
./collect_quest_teleop.sh --robot-model s63 --gripper robotiq_2f85 --dataset-format lerobot \
  --lerobot-python "$LEROBOT_PYTHON"

./eval_groot.sh --robot-model s63 --gripper robotiq_2f85 --checkpoint /path/to/pretrained_model
```

Use `--gripper none` to evaluate an older 15-D manager checkpoint. The evaluator
rejects a checkpoint whose action dimension does not match the selected preset
before executing its output.

## Visual check delegated to the operator

Use a short GUI run and inspect both palms before collecting data:

```bash
./preview_quest_local.sh --robot-model s63 --gripper robotiq_2f85 --steps 600
```

Confirm that each palm faces the box, fingertips point in the intended reach
direction, and neither hand intersects the forearm. If not, adjust only the
affected side's `robot_mount_pos`/`robot_mount_rot` and relaunch.

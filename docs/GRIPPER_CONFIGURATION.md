# Gripper configuration

The default `allegro` preset adds one independently controlled Allegro Hand to
each Kuavo wrist. Both `scene.py` and the manager-based environment use the
same configuration. The manager scene exposes the hands as
`scene["left_gripper"]` and `scene["right_gripper"]`; the fixed-joint group is
`scene["gripper_attachments"]`.

## Run

Allegro is enabled by default:

```bash
./run_scene.sh
./run_manager_env.sh --num-envs 1 --steps 100000
```

Select it explicitly, or restore the old handless action schema:

```bash
./run_scene.sh --gripper allegro
./run_manager_env.sh --gripper none --num-envs 1
```

The standard manager action order is:

```text
waist_yaw(1), left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)
```

That is 17 dimensions with the default two hands and 15 dimensions with
`--gripper none`. A positive gripper value opens the hand and a negative value
closes it. Meta Quest and browser teleoperation append two channels to the
existing 14-D bimanual IK/head command, producing a 16-D action. Pinch distance
at or below 0.035 m closes the corresponding hand.

## Change gripper or mount pose

Edit the deployment file [`configs/grippers.json`](../configs/grippers.json),
or preserve it and pass a different file:

```bash
./run_scene.sh --gripper custom --gripper-config /data/workcell/grippers.json
```

Each enabled preset defines:

- `usd_path`: default hand USD; local relative paths are resolved relative to
  the JSON file, while `${ISAAC_NUCLEUS_DIR}` selects official Isaac assets;
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
  "robot_mount_rot": [0.5, 0.5, -0.5, -0.5]
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

## Allegro asset note

The default USD is NVIDIA's official Wonik Robotics Allegro Hand asset:

```text
${ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd
```

It is supplied by the installed Isaac asset library and is not redistributed
in this repository. The official preset available in Isaac Lab is right-hand
geometry. This project initially uses that geometry on both wrists. For a true
mirrored left hand, set `sides.left.usd_path` to a compatible left-hand USD and
override `attachment_mount_body` if its prim name differs.

For an offline deployment, place an authorized hand USD and all of its
dependencies in your own asset directory and reference that local path in a
custom preset.

## LeRobot and GR00T compatibility

New Quest demonstrations store gripper joint state and the two binary gripper
actions in the dataset schema. Keep the same preset for collection, training,
and evaluation:

```bash
./collect_quest_teleop.sh --gripper allegro --dataset-format lerobot \
  --lerobot-python "$LEROBOT_PYTHON"

./eval_groot.sh --gripper allegro --checkpoint /path/to/pretrained_model
```

Use `--gripper none` to evaluate an older 15-D manager checkpoint. The evaluator
rejects a checkpoint whose action dimension does not match the selected preset
before executing its output.

## Visual check delegated to the operator

Use a short GUI run and inspect both palms before collecting data:

```bash
./preview_quest_local.sh --gripper allegro --steps 600
```

Confirm that each palm faces the box, fingertips point in the intended reach
direction, and neither hand intersects the forearm. If not, adjust only the
affected side's `robot_mount_pos`/`robot_mount_rot` and relaunch.

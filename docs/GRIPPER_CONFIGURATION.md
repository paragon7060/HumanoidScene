# Gripper configuration

The default `s200062` robot contains its own two-finger grippers. The
`s200062_integrated` preset controls two motor crank joints per side directly on
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

## S200062 / S56 physical four-bar closure

The `s200062_integrated` and `s56_twofinger` presets drive only
`{side}_f_bar_1_joint` and `{side}_b_bar_1_joint`. Central `bar_4` links are
revolute, not welded to the housing. Each finger is connected to its matching
`bar_4` by a passive hinge marked `excludeFromArticulation`, closing the loop
without making an unsupported cyclic articulation tree. `bar_3` and `bar_4`
have zero position stiffness; never command them with the motor's angle.
This follows the [Isaac Sim closed-loop joint pattern](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html).

Model fidelity is explicit: the source URDF's rigid `bar_1+bar_2` and
`bar_3+finger` subassemblies are retained. This is a physically closed,
one-DOF four-bar approximation per jaw, not a full reproduction of every
revolute joint and soft equality in the source MuJoCo model. The source
MJCF helper endpoints do not coincide with the supplied CAD pin holes.
The implemented finger anchor is `(-0.0125, 0, -0.021)` m and the follower
anchor is `(-0.0006261, 0, -0.0499953)` m for the front jaw; back jaws mirror X.
They coincide at the zero-pose assembly and match the mesh holes within the
source coordinate rounding. Meshes are not moved independently of rigid bodies.

The validated motor range remains front `[-0.25, 0]`, back `[0, 0.25]` rad.
Reset uses circle-intersection kinematics to initialize passive joints on the
assembled branch. Runtime motion is solved by PhysX, including under contact;
the analytic helper is not a visual animation or a passive-joint servo.
Bar collisions are still omitted; contact is modeled on housing, wrist and
finger meshes. This is not a validated detailed force/transmission model.

Existing custom JSON presets that still drive `bar_3` must remove it from
`joint_names`, `default_joint_pos`, `open_command` and `close_command`.
An actionable error prevents silently using those legacy commands. The RwH
16-D arm/claw interface and 17-D manager commands are unchanged, but generic
raw-joint observations/recordings may change shape and must be versioned.

The converter finishes both packaged USDs with a small reproducible root-layer
override. To reapply it without rebuilding mesh layers:

```bash
conda activate env_isaaclab_232
python scripts/finalize_twofinger_usd.py
```

Physical regression checks (3 open/close/reset cycles at 120 Hz, both hands):

```bash
python scripts/verify_twofinger_linkage.py --robot-model s56 --headless \
  --output artifacts/diagnostics/s56_fourbar_cycles.json \
  --video-out artifacts/diagnostics/s56_fourbar_verified.mp4
python scripts/verify_twofinger_linkage.py --robot-model s200062 --headless \
  --output artifacts/diagnostics/s200062_fourbar_cycles.json
python scripts/verify_twofinger_linkage.py --robot-model s56 --headless \
  --contact-probe --cycles 2 --output artifacts/diagnostics/s56_fourbar_contact_verified.json
```

The check fails if any closure gap reaches 1 mm or any central link moves less
than 0.35 rad. On 2026-09-02 the 1,440-step runs measured maximum gaps of
0.00193 mm (S56) and 0.00244 mm (S200062), with about 0.414 rad follower travel.
The S56 blocking-contact test (960 steps) measured a 0.0739 mm maximum gap
while the kinematic blocks generated up to 51 N of simulated contact force.
Contact mode requires measured force above 1 N and follower travel above
0.05 rad, since the blocks deliberately prevent full closure. These force
values are numerical test loads, not calibrated real-hardware grip forces.
The inspection video uses temporary blue jaw / orange central-bar materials;
the packaged robot's materials are unchanged.
These are linkage tests, not a policy grasp-success claim. A subsequent actual
checkpoint-40K evaluation ran for 24 s at 10 Hz with both claws following their
commands, but still showed no successful box lift. See the
[post-fix rollout results](RWH_KUAVO_V2_S56_EVAL.md#after-the-physical-four-bar-fix-2026-09-02)
for reproduction and limitations. Older `corrected_twofinger` artifacts only
refer to removing the leftover dexterous-hand meshes, before physical closure.
The existing RwH launcher also completed two 5-step mock episodes with the
16-D policy schema and 17-D manager action intact. Mock success/failure values
are wiring checks, not model performance measurements.

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

# Task1 Canonical Data-Collection Pipeline and Waist-Aware Planning

Date: 2026-09-11

## Objective

Make the successful `endeffector_center` bimanual grasp/pull pipeline the only
active Task1 implementation under `data_collection`, archive superseded work as
read-only history, and extend planning/execution from arm-only 14 DoF to both
arms plus safe torso freedom.

The immediate manipulation objective is to replace the successful but visibly
rear-biased `+76 mm` side-flap grasp with the front-most collision-free grasp
that still completes:

```text
approach -> close -> lift -> pull -> hold
```

## Proven Baseline

The migration baseline is commit
`b2ce832aa768361651dd71d9690fdb941dbe7e86` and its Kanu run
`physical_v2_main_default_verified_params`.

The baseline contract is:

- S200062 with `s200062_integrated` two-finger grippers.
- Planner and runtime TCP: `endeffector_center`.
- Main calibrated closed-center offsets.
- Gripper motor actuator: effort `100`, stiffness `4000`, damping `400`.
- Finger material friction: static `20`, dynamic `16`, combine mode `average`.
- Simultaneous bimanual endpoint and approach planning.
- Successful physical close, 10 cm lift, approximately 19 cm robotward pull,
  and stable final hold.
- Existing successful target: nominal flap grasp plus base-frame offset
  `[+0.076, 0, +0.015] m`, tool-down angle `56 deg`.

The old result remains a comparison baseline. It is not the desired final grasp
location because its target lies about 30 mm from the rear end of the side flap.

## Scope

### Included

- Reorganize active Task1 code into a package under `data_collection/task1`.
- Move superseded code, draft configurations, prompt snapshots, and historical
  documentation under `data_collection/legacy`.
- Extract reusable collision and plan-contract logic from the old sequential
  planner before archiving that planner.
- Add a canonical joint-layout contract shared by endpoint, approach, retreat,
  execution, and reports.
- Generate waist-aware endpoints and complete approach paths.
- Replay waist-aware plans in Isaac Lab and produce a verified MP4/report.
- Search for a more central grasp before allowing the old `+76 mm` target.
- If fixed-height 16 DoF has no valid front grasp, search additional torso
  heights while preserving the existing coupled height mechanism.

### Excluded

- Changing the policy-facing action contract. It remains arm 14D plus claw 2D.
- Training, dataset conversion, or model changes.
- Independent unconstrained optimization of `knee_joint`, `leg_joint`, and
  `waist_pitch_joint`.
- Changing S200062 TCP calibration, gripper strength, friction, rack geometry,
  or box geometry.
- Claiming attached-object collision safety from the current retreat IK alone.

## Active Layout

The active directory becomes:

```text
data_collection/
  README.md
  task1/
    __init__.py
    contract.py
    collision.py
    pose_editor.py
    endpoint.py
    approach.py
    retreat.py
    execute.py
    ui/
      editor.html
  legacy/
    README.md
    configs/
    docs/
    prompts/
    task1_cumotion_collision_plan.py
    task1_cumotion_replay_video.py
```

`scripts/task1_cartesian_lift_pull_plan.py` and
`scripts/task1_cumotion_grasp_pull_smoke.py` remain thin command wrappers so
existing shell workflows have stable entry points. Their implementation moves
to `data_collection.task1.retreat` and `data_collection.task1.execute`.

### File migration

| Current path | Destination or replacement |
| --- | --- |
| `task1_box_pick_web.py` | `task1/pose_editor.py` |
| `task1_pose_editor.html` | `task1/ui/editor.html` |
| `task1_cumotion_bimanual_rmpflow_endpoint.py` | `task1/endpoint.py` |
| `task1_cumotion_bimanual_plan.py` | `task1/approach.py` |
| reusable functions in `task1_cumotion_collision_plan.py` | `task1/collision.py` |
| old sequential CLI in `task1_cumotion_collision_plan.py` | `legacy/task1_cumotion_collision_plan.py` |
| `task1_cumotion_replay_video.py` | `legacy/task1_cumotion_replay_video.py` |
| `configs/*` | `legacy/configs/*` |
| `collection/README.md`, `robot/README.md`, `tasks/*` | `legacy/docs/*` |
| `references/*` | `legacy/prompts/*` |

No compatibility import modules remain at the old `data_collection.task1_*`
paths. Tests and active callers are updated to the package paths. Git history
and `legacy/README.md` provide provenance.

## Canonical Plan Contract

`task1/contract.py` owns all joint ordering and plan validation. No stage may
check only `len(joint_names) == 14` or slice a vector with unnamed constants.

The canonical waist-aware order is:

```text
waist_pitch_joint
waist_yaw_joint
zarm_l1_joint ... zarm_l7_joint
zarm_r1_joint ... zarm_r7_joint
```

Generated waist-aware plans contain 16 values per waypoint. Every artifact
records:

- `schema_version: 2`
- exact `joint_names`
- `controlled_groups`
- `tcp_frame: endeffector_center`
- `tool_frames`
- `torso_height_m`
- baseline and terminal waist values
- source commit and input-artifact paths
- planner and collision-model settings

Approach and retreat plans must have byte-for-byte identical joint-name order
before composition or execution. A 14-DoF plan is accepted only when the caller
explicitly requests the proven arm-only baseline mode; new generation defaults
to the 16-DoF contract.

## Torso Model

### Primary search: fixed-height 16 DoF

The primary run keeps torso height at 0.25 m. `TeleopBodyMapper` supplies the
coupled `knee_joint`, `leg_joint`, and upright-reference `waist_pitch_joint`
values for that height. `knee_joint` and `leg_joint` remain fixed. The planner
then optimizes:

- both arms: 14 DoF;
- waist pitch residual around the height-derived reference;
- waist yaw.

Safe planning windows intersect physical URDF limits with:

- waist pitch: reference plus or minus 20 degrees;
- waist yaw: plus or minus 15 degrees around the centered reference.

RMPflow updates are clamped to those windows, and endpoint scoring penalizes
waist deviation and total c-space movement. This prevents a collision-free but
needlessly twisted solution.

### Fallback search: effective torso 3 DoF

If the fixed-height search produces no acceptable front grasp, run the same
16-DoF solver at torso heights `0.20`, `0.25`, `0.30`, and `0.35 m`. Each height
is converted through `TeleopBodyMapper`, so knee/leg/waist compensation remains
on the valid height manifold.

Across the candidates, the effective torso choices are:

```text
height h + waist pitch residual + waist yaw
```

This supplies three useful torso freedoms without exposing the raw four-joint
chain as independent variables. Candidate selection includes height change in
the posture cost.

## Planning and Execution Flow

### 1. Snapshot

`task1.pose_editor` captures the settled box/flap geometry, robot state,
physical joint limits, collision model, calibrated center TCP, and torso-height
reference. The target box must match the snapshot at execution within the
existing position/orientation gates.

### 2. Endpoint search

`task1.endpoint` loads one XRDF c-space containing both waist joints and both
arms. It assigns both `endeffector_center` target frames in the same RMPflow
solve.

The primary candidate grid is:

- rearward/base-X offsets: `+0.030`, `+0.040`, `+0.050 m`;
- tool-down angles: `45`, `50`, `56 deg`;
- flap line offset: centered unless a bounded local adjustment is required.

Candidates are hard-rejected for joint-window violation, world collision,
self-collision, TCP position error above 3 mm, or orientation error above 1
degree. Among valid candidates, selection prioritizes a forward grasp, then
collision clearance, then total posture motion and waist deviation. The old
`+0.076 m` target is diagnostic fallback only and cannot win the normal search.

### 3. Approach

`task1.approach` plans in the same 16-DoF c-space. Collision spheres and task
constraints are indexed by joint name rather than fixed arm offsets. The waist
may move continuously from the snapshot pose to the selected endpoint.

The emitted path must pass dense world/self-collision validation and preserve
the current simultaneous-motion and maximum-joint-step checks.

### 4. Close and retreat

At grasp, the selected waist pitch/yaw are held fixed. `task1.retreat` solves
the two arm chains for lift and pull while carrying the unchanged waist values
in every 16-DoF waypoint. This isolates the purpose of waist freedom to reach
the better grasp and avoids moving the torso while holding an object.

Retreat reports continue to state explicitly that the current Cartesian IK
does not validate attached-object collision.

### 5. Physical execution

`task1.execute` resolves plan joint IDs from the canonical contract and sends
position targets to both waist and arm joints. Gripper motor commands remain a
separate four-joint group. Tracking reports contain:

- maximum arm error;
- maximum waist-pitch error;
- maximum waist-yaw error;
- box motion by phase;
- TCP positions using `endeffector_center`;
- grasp retention and motor obstruction.

The runner preserves the proven baseline execution mode as an explicit option
for A/B diagnosis. The final waist-aware result records whether approach replay
was kinematic-direct or position-servo; one must not be reported as the other.

## Legacy Policy

`data_collection/legacy` is not a supported production import surface and is
excluded from active test discovery. Its README records for every moved item:

- original path;
- purpose;
- reason archived;
- canonical replacement, if any;
- baseline commit that preserves an executable historical version.

Legacy files are not silently repaired when active interfaces change. This
keeps historical evidence intact and prevents obsolete planners from appearing
to be supported.

## Tests

CPU tests cover:

- exact 14- and 16-DoF joint contracts and explicit baseline opt-in;
- rejection of mixed plan orders and malformed waypoint widths;
- torso-height manifold generation;
- safe waist-window intersection and clamp behavior;
- waist-aware XRDF contents;
- endpoint scoring preference for front grasps;
- approach distance weights and arm-index lookup by name;
- retreat output carrying a constant waist pose;
- runner joint resolution and separate tracking metrics;
- all moved imports and the absence of active references into `legacy`.

Repository validation includes targeted Task1 tests, the prior 94-test TCP and
gripper set, `git diff --check`, and an import/reference scan. Pre-existing
optional `python-fcl` and unrelated upstream failures remain reported
separately rather than being attributed to this change.

## Kanu Validation and Acceptance

Run in the clean Kanu worktree without modifying the dirty primary checkout or
the existing GPU4 editor lane. Use an available isolated GPU lane.

A new run is accepted only if all of the following hold:

- source commit and remote checkout match;
- target uses `endeffector_center`;
- generated approach and retreat share the 16-DoF contract;
- chosen rearward offset is at most `+0.050 m`;
- endpoint and dense approach are world/self-collision free;
- approach box translation is at most 3 mm;
- both grippers show motor obstruction;
- retreat box translation is at least 30 mm;
- robotward pull is positive and visually extracts the box;
- final-hold box motion is at most 10 mm;
- hand-box retention drift is at most 50 mm;
- MP4 frame count/duration are valid;
- a contact sheet confirms the grasp is visibly forward of the old `+76 mm`
  baseline and retained through final hold.

If no candidate passes, retain all reports and state which gate failed. Do not
weaken collision or retention gates merely to produce a video.

## Delivery

The implementation is committed on the isolated `codex/` branch and pushed
only after tests and Kanu verification. The handoff includes:

- active/legacy file map;
- selected torso height, waist pose, grasp offset, and angle;
- plan and runtime validation metrics;
- local MP4/report/contact-sheet links;
- Google Drive link only when the user requests an upload, with metadata read
  back after upload;
- known limitations, especially attached-object collision validation.

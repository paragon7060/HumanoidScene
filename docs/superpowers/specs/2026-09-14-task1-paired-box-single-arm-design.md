# Task1 Paired-Box Single-Arm Physical Feasibility

Date: 2026-09-14

## Objective

Add a canonical paired-box mode to the existing Task1 pipeline and use it for
the first physical feasibility run of the `MMS-left` shelf-2 layout. The left
S200062 integrated gripper must grasp the two adjacent inner flaps of
`MediumBox_0` and `MediumBox_1`, lift and pull both boxes together with one arm,
hold them, and produce an MP4 plus a machine-readable `report.json` whether the
run passes or fails.

This is a paired-box physical-retention experiment. Planner completion alone
is not success.

## Proven Baseline and Inputs

The implementation reuses the current canonical Task1 components:

- robot model `s200062` with the `s200062_integrated` gripper;
- calibrated planner and runtime TCP `endeffector_center`;
- the existing smoothed approach and lift/pull/hold execution flow;
- the arm-only 14-DoF joint schema;
- the existing gripper actuator and contact-material parameters;
- rack pose input `configs/rack_box_poses_task1_mms_left.json`.

The rack pose file defines robot-view order
`MediumBox_0, MediumBox_1, SmallBox_0`, paired boxes
`MediumBox_0, MediumBox_1`, active arm `left`, and grasp mode
`adjacent_inner_flaps`.

## Scope

### Included

- Add a versioned paired-target scenario contract while keeping all existing
  single-box scenario manifests valid and behaviorally unchanged.
- Resolve and snapshot two target boxes and their facing inner side flaps.
- Generate an arm14 endpoint, approach, and retreat in which the left arm is
  active while the right arm and waist remain fixed.
- Keep the rack, the third box, both target-box bodies, and all non-selected
  flaps in collision checking.
- Track both boxes independently through settle, approach, close, retreat, and
  final hold.
- Apply aggregate paired-box acceptance only after every per-box gate passes.
- Run the `MMS-left` experiment on an audited Kanu GPU lane and preserve its
  MP4, `report.json`, plans, logs, and provenance.

### Excluded

- Training, dataset conversion, policy, or action-contract changes.
- Changes to the TCP calibration, gripper effort/stiffness/damping, friction,
  box geometry, rack geometry, or shelf pose.
- Waist motion, right-arm motion, bimanual grasping, or an attached-object
  collision claim.
- Weakening any existing approach, tracking, retention, or hold gate.
- Running the other `SSM-left`, `MSS-right`, and `SMM-right` layouts before the
  first `MMS-left` result is evaluated.
- Requiring full rack extraction; stable partial extraction is sufficient for
  this feasibility run.

## Scenario Contract

Existing schema-version-1 single-box manifests continue to use
`scene.target_box` and retain their current validation and CLI translation.
The new paired-box manifest uses schema version 2 and declares:

```json
{
  "scene": {
    "rack_box_poses": "configs/rack_box_poses_task1_mms_left.json",
    "paired_boxes": ["medium_box_0", "medium_box_1"],
    "pair_grasp": "adjacent_inner_flaps",
    "clear_same_shelf_boxes": false
  },
  "planning": {
    "cspace_dof": 14,
    "include_waist": false,
    "active_arm": "left"
  },
  "execution": {
    "active_gripper": "left"
  }
}
```

The lowercase names are runtime scene keys. Loading must cross-check them
against the corresponding `pair_pick.paired_boxes` entries in the rack pose
file instead of silently relying on list position or name casing. Pair mode
rejects duplicate boxes, mixed box sizes, non-adjacent boxes, an arm mismatch,
or any layout whose `pair_gap_m` is not the expected touching arrangement.

`clear_same_shelf_boxes` is false because all three shelf-2 boxes must remain in
the scene for the experiment. The existing single-box requirement that clears
same-shelf obstacles is unchanged.

## Pair Geometry and Grasp Target

For each target box, the snapshot resolves the body pose and both side-flap
colliders. It selects the facing flap pair from world geometry: among the four
side-flap combinations, the chosen pair has the smallest horizontal separation
and mutually facing normals. The resolver does not hard-code that a particular
asset flap name is always the inner flap.

Each selected flap contributes its current upper-edge grasp point using the
existing grasp-depth convention. The left `endeffector_center` target is the
midpoint of those two points. Its closing axis spans the facing-flap stack, and
its forward/tool-down orientation follows the existing Task1 TCP convention.
The snapshot records:

- ordered runtime box keys and source rack-pose names;
- each box body pose in the robot base frame;
- selected flap body path, grasp point, normal, and dimensions;
- pair grasp midpoint and closing axis;
- initial box-center separation and pair center;
- active arm/gripper and inactive-arm joint reference.

The endpoint is valid only if both selected flap points are within the gripper's
declared capture geometry and the target pose satisfies the current TCP error
and joint-limit gates.

## Planning and Collision Boundary

The plan keeps the canonical 14 joint names so it remains compatible with the
arm14 executor. Only the seven left-arm joints may differ from the snapshot;
all seven right-arm joints are held byte-for-byte at their initial values in
every endpoint, approach, and retreat waypoint. Waist and torso state remain at
the current fixed arm14 baseline.

World collision contains the rack, `SmallBox_0`, both MediumBox bodies, and all
their flaps. During ordinary approach samples no box geometry is omitted. At
the terminal contact stage, collision relaxation may omit only the two resolved
inner-flap collider paths; both box bodies and every other flap remain active.
The emitted approach must pass the existing dense world/self-collision,
joint-limit, terminal-error, and maximum-joint-step validation. Existing path
shortcut and resampling are reused without loosening their limits.

Retreat uses the current lift then robotward-pull sequence. It operates only on
the left arm, carries the fixed right-arm joints in every waypoint, and makes no
claim that Cartesian retreat planning validates the two attached box bodies.

## Physical Execution and Measurements

Only the left gripper closes. The executor samples both target bodies and the
left `endeffector_center` in every phase. A single target-box proxy is forbidden
for pair acceptance.

References and measurements are defined as follows:

- Settled pose: each box body pose after settle and before approach.
- Close reference: box centers, pair-center position, inter-box center
  distance, and left-TCP-to-pair-center vector at the end of close/closed hold.
- Robotward progress: signed displacement of each box center from its settled
  pose along the existing Task1 robotward axis.
- Pair-separation drift: absolute change in center-to-center distance from the
  close reference, evaluated across retreat and final hold.
- Hand-to-pair-center drift: maximum change of the
  left-TCP-to-pair-center vector from the close reference across retreat and
  final hold.
- Final-hold motion: displacement of each box center from the first to the last
  final-hold sample.

The report contains per-box time series summaries, per-box pass flags, pair
metrics, phase summaries, source commit, scenario/rack-pose hashes, plan hashes,
GPU/lane metadata, and a top-level `paired_box_success`.

## Acceptance Contract

`paired_box_success` is true only when all of the following are true:

- approach replay is collision-free and each MediumBox stays within the
  existing configured pre-close motion limit;
- arm tracking remains within the existing arm14 tolerance;
- each MediumBox achieves at least `0.05 m` robotward progress;
- maximum pair-separation drift is at most `0.01 m`;
- maximum left-hand-to-pair-center drift is at most `0.05 m`;
- final-hold motion is at most `0.01 m` for each MediumBox;
- the active left gripper meets the existing obstruction/closure evidence gate;
- required phases complete and the MP4 is finalized successfully.

Partial extraction may pass. Full extraction is reported separately and is not
required. If either box slips, fails the progress gate, or disappears from
tracking, the aggregate result fails even if the other box passes. No threshold
may be changed after observing the run.

## Artifacts and Execution Lane

Use a unique, non-overwriting run root, with the same relative structure on
Kanu and after local retrieval, for example:

```text
task1_paired_box_mms_left_20260914_v1/
  snapshot/
  endpoint/
  approach_plan.json
  retreat_plan.json
  physical_v1/
    report.json
    task1_mms_left_single_arm.mp4
    run.log
```

Before launch, record the exact Kanu checkout commit, scenario and rack-pose
hashes, CUDA device, competing Isaac processes, tmux/session name, log path,
and output path. Deployment requires local/remote source-hash parity for the
files used by the run. A process exit or an MP4 alone is not completion:
`report.json` must be written and the MP4 must pass `ffprobe` validation.

Failures are retained with the same artifact set and reported as failures. A
planner success, attractive video, or motion of only one box must not be
described as paired-box physical success.

## Implementation Surface

Expected production changes are limited to:

- `data_collection/task1/scenario.py` for schema/version validation and CLI
  translation;
- `data_collection/task1/pose_editor.py` for pair-aware snapshot geometry;
- `data_collection/task1/collision.py` for explicit two-flap selection and
  narrowly scoped contact omission;
- `data_collection/task1/endpoint.py`, `approach.py`, and `retreat.py` for the
  active-arm/fixed-inactive-arm contract;
- `data_collection/task1/execution_contract.py` and `execute.py` for pair
  measurement and acceptance;
- one `MMS-left` experimental scenario manifest and focused tests.

Existing thin command wrappers may forward new arguments but must not gain a
second implementation path.

## Verification Plan

Before Kanu execution, tests must cover:

- schema-1 single-box regression and schema-2 paired-box validation;
- rack-pose/runtime-name cross-checks and malformed-pair rejection;
- geometry-based inner-flap selection, midpoint, normal, and capture-width
  validation under translated/rotated box poses;
- collision omission of exactly the two selected flap paths at terminal contact
  and no omission during ordinary approach;
- fixed right-arm and waist values across all arm14 waypoints;
- per-box progress, pair-separation, retention, final-hold, missing-tracking,
  and aggregate acceptance edge cases;
- executor CLI translation and report serialization;
- all existing scoped Task1 and gripper regression tests.

The live sequence is:

1. Audit Kanu GPU/process/output lanes and verify deployment hashes.
2. Capture and statically validate the `MMS-left` pair snapshot.
3. Generate endpoint, smoothed approach, and lift/pull retreat plans.
4. Validate plan contracts and collision results before physics.
5. Execute once with the predeclared thresholds and preserve all artifacts.
6. Retrieve artifacts, verify hashes and MP4 integrity, and evaluate
   `report.json` without changing gates.

Only the resulting physical evidence determines whether the first `MMS-left`
single-arm paired-box attempt passed.

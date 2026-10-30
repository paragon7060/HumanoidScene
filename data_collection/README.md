# Task1 canonical data-collection pipeline

The supported pipeline is `data_collection.task1`:

- `pose_editor.py` and `ui/editor.html`: authoring and live inspection
- `endpoint.py`: simultaneous waist-pitch/yaw + both-arm endpoint search
- `approach.py`: collision-aware synchronized approach
- `retreat.py`: lift and robotward pull construction
- `execute.py`: physical close, lift, pull, hold, and video/report capture
- `contract.py` and `collision.py`: shared plan and collision contracts

The default planning mode is both arms only: left arm q1-q7 followed by right
arm q1-q7, for 14DoF total. Waist-assisted planning is an opt-in comparison;
when enabled, `waist_pitch_joint` and `waist_yaw_joint` precede the two arms for
16DoF total. Torso height is a linked outer-search parameter; knee and leg are
not optimized independently.

Supported script entry points are:

- `scripts/task1_cartesian_lift_pull_plan.py`
- `scripts/task1_cumotion_grasp_pull_smoke.py`
- `scripts/task1_run_scenario.py`: validate a checked-in scenario manifest,
  reject cross-DoF plans, and replay the physical run with an annotated report

## Default approach smoothing and replay speed

The supported approach planner shortcuts a constrained-RRT result only through
segments that pass the full collision gate, then densifies those shortcut
segments for execution. Raw RRT detours are not replayed directly.

For `--kinematic-direct-approach-render`, the normal-speed video default is 73
evenly spaced approach frames. Every collision-checked plan waypoint is still
applied to the robot; only camera capture is sampled, so the safe path is not
skipped. Use `--kinematic-approach-render-frames N` only when a comparison needs
an explicit playback duration. Physical close, retreat, and final-hold timing
is unaffected.

The Task1 Drive layout keeps one current verified MP4 and its report at the top
of each verified scenario folder. Superseded MP4s are deleted instead of being
mixed with the current representative videos; JSON reports remain the source
of acceptance metrics and provenance. A category without a strict verified
result stays empty rather than receiving a partial or failed video labeled as
verified.

The frozen bimanual single-box manifests live directly under
`configs/task1/scenarios/`. Evidence-backed single-arm comparisons live under
`configs/task1/scenarios/verified/single_arm/`:

| Bimanual mode | Planning DoF | Verification state |
| --- | ---: | --- |
| default arms only | arm14 | `verified` (`stable_bimanual_retention`) |
| waist-assisted comparison | arm14 + waist16 | `verified` (`stable_bimanual_retention`) |

| Active hand | Planning DoF | Verification state |
| --- | ---: | --- |
| left | arm14 | `verified_partial_extraction` |
| left | arm14 + waist16 | `verified_partial_extraction` |
| right | arm14 | `verified_failure` (`PHYSICAL_TRACKING_FAILURE`) |
| right | arm14 + waist16 | `verified_partial_extraction` |

All four manifests reuse `configs/rack_box_poses_task1_centered.json`; exact box
coordinates are not duplicated inside each scenario. The verified folder means
the outcome has artifact evidence, not that every outcome passed. Generated
endpoint, approach, retreat, report, frame, and video files remain run artifacts
rather than checked-in config; manifests retain their report/video paths and
SHA256 evidence.

Superseded drafts and tools are retained under `legacy/` for history only and
are neither supported imports nor executable entry points. See
`legacy/MANIFEST.md`.

Bimanual physical success requires collision-free approach, no box motion
during approach, obstruction on both grippers, box motion during retreat, and a
stable final hold. A single-arm comparison instead requires obstruction on the
selected gripper and the configured oriented-box partial-extraction thresholds;
rigid hand-box retention is diagnostic rather than a single-arm pass gate. Both
use TCP `endeffector_center`. Verified S200062 settings remain effort 100,
stiffness 4000, damping 400, and finger static/dynamic friction 20/16.

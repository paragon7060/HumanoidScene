# Task1 canonical data-collection pipeline

The supported pipeline is `data_collection.task1`:

- `pose_editor.py` and `ui/editor.html`: authoring and live inspection
- `endpoint.py`: simultaneous waist-pitch/yaw + both-arm endpoint search
- `approach.py`: collision-aware synchronized approach
- `retreat.py`: lift and robotward pull construction
- `execute.py`: physical close, lift, pull, hold, and video/report capture
- `contract.py` and `collision.py`: shared plan and collision contracts

The canonical planning order is `waist_pitch_joint`, `waist_yaw_joint`, left
arm q1-q7, then right arm q1-q7. Normal endpoint runs therefore have 16DoF.
The 14DoF arm-only path is an explicitly selected baseline. Torso height is a
linked outer-search parameter; knee and leg are not optimized independently.

Supported script entry points are:

- `scripts/task1_cartesian_lift_pull_plan.py`
- `scripts/task1_cumotion_grasp_pull_smoke.py`

Superseded drafts and tools are retained under `legacy/` for history only and
are neither supported imports nor executable entry points. See
`legacy/MANIFEST.md`.

Physical success requires collision-free approach, no box motion during
approach, obstruction on both grippers, box motion during retreat, stable final
hold, and TCP `endeffector_center`. Verified S200062 settings remain effort 100,
stiffness 4000, damping 400, and finger static/dynamic friction 20/16.

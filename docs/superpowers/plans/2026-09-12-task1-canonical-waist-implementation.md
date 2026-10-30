# Task1 Canonical Waist-Aware Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `data_collection` to one supported Task1 pipeline, preserve superseded files under non-importable `legacy/`, and produce a verified `endeffector_center` pick/pull using both arms plus waist pitch/yaw.

**Architecture:** `data_collection.task1` owns plan contracts, collision helpers, endpoint search, approach planning, retreat construction, execution, and the pose editor. The two scripts under `scripts/` remain CLI wrappers. Planning defaults to ordered 16DoF `[waist_pitch, waist_yaw, left_arm_7, right_arm_7]`; 14DoF is an explicit baseline. Torso height stays on the knee/leg linkage manifold and is searched outside the solver only if the 0.25m run cannot find a front grasp.

**Tech Stack:** Python, pytest, NumPy, cuMotion/Isaac Sim on Kanu, Git.

**Spec:** `docs/superpowers/specs/2026-09-11-task1-canonical-pipeline-waist-design.md`

## Global Constraints

- Preserve the user's dirty primary checkout and Kanu GPU4 editor lane.
- Keep TCP frame `endeffector_center` and verified gripper dynamics/friction unchanged.
- Do not optimize raw knee and leg independently.
- Do not upload externally unless the user asks.
- Every behavior change starts with a failing focused test.

---

## Task 1: Canonical joint-plan contract

- [ ] Add failing tests in `tests/test_task1_plan_contract.py` for exact 16DoF order, explicit 14DoF baseline, shape/name rejection, split/compose round trips, safe waist bounds, and constant-waist expansion.
- [ ] Add `data_collection/task1/__init__.py` and `data_collection/task1/contract.py` with `ARM_JOINT_NAMES`, `WAIST_JOINT_NAMES`, `WAIST_ARM_JOINT_NAMES`, `PlanLayout`, `layout_for_joint_names`, `validate_plan`, `split_trajectory`, `compose_waist_arm`, and `safe_waist_bounds`.
- [ ] Run `pytest -q tests/test_task1_plan_contract.py` and commit the passing contract.

## Task 2: Canonical package and history-only archive

- [ ] Add failing import/manifest tests proving only `data_collection.task1` is supported and `legacy/` is not a package.
- [ ] Move the active editor, endpoint, approach, retreat, and executor implementations into `data_collection/task1/`; extract shared collision helpers into `collision.py`.
- [ ] Move superseded configs, prompts, docs, replay code, and sequential planners into `data_collection/legacy/` without `__init__.py`; add a dated manifest mapping every moved path to its replacement or archival reason.
- [ ] Update active tests/imports and `data_collection/README.md`; remove executable references to archived paths.
- [ ] Run focused import/collision/editor tests and commit the reorganization.

## Task 3: 16DoF endpoint search

- [ ] Add failing tests for the default 16DoF c-space, `--arm-only-baseline`, safe waist limits, front-biased target grid X offsets `0.03/0.04/0.05`, Z offset `0.015`, angles `45/50/56`, and metadata schema v2.
- [ ] Refactor `data_collection/task1/endpoint.py` so waist pitch/yaw and both arms are solved simultaneously by default, while arm-only mode is opt-in.
- [ ] Emit target candidate, torso height, ordered joint names, selected waist posture, collision margins, and terminal TCP errors in metadata.
- [ ] Run the endpoint tests and commit.

## Task 4: Name-driven 16DoF approach planning

- [ ] Add failing tests that reject hard-coded 14-column assumptions and prove shoulder/waist weights are resolved by joint name.
- [ ] Refactor `data_collection/task1/approach.py` to preserve the endpoint plan's exact joint order and to use waist-aware smoothness/regularization weights.
- [ ] Validate every segment with the shared plan contract and collision checks.
- [ ] Run approach and collision tests and commit.

## Task 5: Waist-preserving retreat and execution

- [ ] Add failing tests that prepend the selected waist posture to arm-only lift/pull rows and keep it constant through close, lift, pull, and hold.
- [ ] Implement `data_collection/task1/retreat.py` and `execute.py`; make `scripts/task1_cartesian_lift_pull_plan.py` and `scripts/task1_cumotion_grasp_pull_smoke.py` thin wrappers.
- [ ] Resolve drive IDs from plan joint names, command waist and arms together during approach, then report waist and arm tracking errors separately.
- [ ] Preserve the verified effort/stiffness/damping/friction settings and `endeffector_center` assertions.
- [ ] Run retreat/execution tests and commit.

## Task 6: Local regression and hygiene

- [ ] Run all targeted Task1 tests, then the full test suite and `python -m compileall data_collection/task1 scripts`.
- [ ] Scan active code/docs for imports or commands referencing archived modules.
- [ ] Run `git diff --check`, inspect `git status --short`, and commit any final cleanup.

## Task 7: Kanu planning and verified video

- [ ] Push the exact branch and update only `/home/seonho/worktrees/HumanoidScene_main_tcp_video_20260911` to the tested SHA.
- [ ] Run the focused tests and compile checks on Kanu.
- [ ] Audit GPU/process/port state and leave GPU4 untouched.
- [ ] First run fixed torso height `0.25m` with the 16DoF front-target grid. If no valid front grasp exists, retry heights `0.20/0.25/0.30/0.35m` while keeping knee/leg linked.
- [ ] Generate approach and retreat plans, then execute close/lift/pull/hold with physical validation enabled.
- [ ] Require: collision-free approach, both grippers obstructed, object follows retreat, final hold stable, TCP `endeffector_center`, and expected gripper dynamics/friction.
- [ ] Inspect the MP4 with `ffprobe` and a contact sheet; compare grasp X/rear-edge distance against the successful baseline.
- [ ] Copy plans, reports, logs, MP4, hashes, and exact Git SHA into a new local artifact directory; confirm both worktrees' status and Kanu lane isolation.


# Task1 Paired-Box Single-Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and physically evaluate an `MMS-left` mode that grasps two adjacent MediumBoxes with the left hand, records both boxes independently, and emits a verified MP4 plus `report.json`.

**Architecture:** Schema-version-2 scenarios add an explicit pair contract while schema-version-1 single-box scenarios remain unchanged. Snapshot and collision code resolve the two inner flap paths geometrically; planners solve only the selected 7-DoF arm and compose it into the canonical 14-DoF layout with the inactive arm fixed. The executor samples both box bodies and applies pure paired-retention acceptance before writing artifacts.

**Tech Stack:** Python 3.11, NumPy/SciPy, PyTorch, Isaac Lab, NVIDIA cuMotion 1.1.0, pytest, JSON, ffmpeg/ffprobe, Git, SSH to Kanu.

**Spec:** `docs/superpowers/specs/2026-09-14-task1-paired-box-single-arm-design.md`

## Global Constraints

- Use exactly `configs/rack_box_poses_task1_mms_left.json` for the first run.
- Use S200062, `s200062_integrated`, and TCP `endeffector_center` unchanged.
- Store canonical arm14 paths; only the left seven joints may move.
- Keep the right arm and waist fixed through planning and execution.
- Keep both target boxes, `SmallBox_0`, the rack, and all non-selected flaps in collision geometry.
- Omit only the two resolved inner-flap paths, and only for terminal contact.
- Require at least `0.05 m` robotward progress from each MediumBox.
- Require pair-separation drift `<= 0.01 m`.
- Require left-hand-to-pair-center drift `<= 0.05 m`.
- Require final-hold motion `<= 0.01 m` for each MediumBox.
- Preserve existing pre-close motion, arm tracking, and motor-obstruction gates.
- Stable partial extraction may pass; full extraction is reported separately.
- Preserve MP4/report/logs for a physical failure; one-box motion is not pair success.
- Work only in `/home/minje/.codex/worktrees/humanoidscene-main-tcp-video-20260911`; do not alter the dirty user checkout.

## File Structure

- `scenario.py`: validate schema 2 and translate pair executor arguments.
- `contract.py`: centralize active-arm indexing and 7D-to-14D composition.
- `collision.py`: select adjacent flaps geometrically and omit exact collider paths.
- `pose_editor.py`: capture pair body/flap/TCP snapshot data.
- `endpoint.py`: solve one active TCP and emit a fixed-inactive-arm endpoint.
- `approach.py`: plan/validate 7D active motion and emit arm14 waypoints.
- `retreat.py`: lift/pull only the active arm.
- `execution_contract.py`: compute pair metrics and aggregate acceptance.
- `execute.py`: track both bodies and finalize MP4/report evidence.
- `configs/task1/scenarios/paired_medium_boxes_mms_left_arm14_experimental_v1.json`: freeze the first run's thresholds.

---

### Task 1: Version-2 Pair Scenario

**Files:**
- Modify: `data_collection/task1/scenario.py`
- Create: `configs/task1/scenarios/paired_medium_boxes_mms_left_arm14_experimental_v1.json`
- Modify: `tests/test_task1_scenario_config.py`

**Interfaces:**
- Consumes: `pair_pick` from the rack-pose JSON.
- Produces: validated `scene.paired_boxes`; executor arguments `--scenario-path`, `--paired-boxes`, `--no-clear-same-shelf-boxes`, and `--pair-separation-drift-max-m`.

- [ ] **Step 1: Write failing pair-manifest tests**

```python
PAIR_CONFIG = DEFAULT_SCENARIO_DIR / "paired_medium_boxes_mms_left_arm14_experimental_v1.json"

def test_mms_left_pair_scenario_matches_pose_contract():
    scenario = load_scenario_config(PAIR_CONFIG)
    assert scenario["schema_version"] == 2
    assert scenario["scene"]["paired_boxes"] == ["medium_box_0", "medium_box_1"]
    assert scenario["scene"]["clear_same_shelf_boxes"] is False
    assert scenario["planning"]["active_arm"] == "left"
    assert scenario["execution"]["active_gripper"] == "left"
    assert scenario["execution"]["pair_separation_drift_max_m"] == 0.01

def test_pair_scenario_forwards_pair_arguments(tmp_path):
    argv = build_physical_executor_argv(
        load_scenario_config(PAIR_CONFIG),
        approach_plan=tmp_path / "approach.json",
        retreat_plan=tmp_path / "retreat.json",
        output=tmp_path / "report.json",
        video_out=tmp_path / "video.mp4",
    )
    index = argv.index("--paired-boxes")
    assert argv[index + 1:index + 3] == ["medium_box_0", "medium_box_1"]
    assert argv[argv.index("--scenario-path") + 1] == str(PAIR_CONFIG.resolve())
    assert "--no-clear-same-shelf-boxes" in argv
    assert argv[argv.index("--pair-separation-drift-max-m") + 1] == "0.01"
```

Add rejection cases for duplicate targets, mixed sizes, a non-adjacent pair,
active-arm mismatch, nonzero `pair_gap_m`, and `scene.target_box` in schema 2.

- [ ] **Step 2: Prove the new tests fail**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_scenario_config.py -k pair
```

Expected: FAIL because schema 2 and its manifest are absent.

- [ ] **Step 3: Implement exact schema-2 validation**

Map checked-in instance labels to runtime scene keys with a strict regex:

```python
_INSTANCE_PREFIX_TO_SCENE = {
    "SmallBox": "small_box", "MediumBox": "medium_box",
    "LargeBox": "large_box", "XLargeBox": "xlarge_box",
}

def _instance_name_to_scene_key(value: str) -> str:
    match = re.fullmatch(r"(SmallBox|MediumBox|LargeBox|XLargeBox)_(\d+)", value)
    if match is None:
        raise ValueError(f"unsupported rack-box instance name: {value!r}")
    return f"{_INSTANCE_PREFIX_TO_SCENE[match.group(1)]}_{match.group(2)}"
```

For schema 2, load the referenced rack JSON and require exact agreement with
`pair_pick.paired_boxes`, `active_arm`, `grasp=adjacent_inner_flaps`, and
`pair_gap_m=0.0`. Require `clear_same_shelf_boxes=false`, arm14, no waist, and
`active_gripper=active_arm`. Keep the schema-1 branch unchanged.

- [ ] **Step 4: Create the full experimental manifest**

Copy all execution and gripper numeric values from
`verified/single_arm/single_medium_box_left_arm14_verified_v1.json`. Change the
scene to MMS-left pair mode, add `pair_separation_drift_max_m: 0.01`, and set:

```json
"verification": {
  "state": "experimental_unverified",
  "expected_passed": null,
  "acceptance_mode": "paired_box_partial_extraction",
  "reference_commit": null,
  "reference_report": null,
  "reference_report_sha256": null,
  "reference_video": null,
  "reference_video_sha256": null,
  "result": null
}
```

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_scenario_config.py tests/test_task1_pair_pick_pose_configs.py
git add data_collection/task1/scenario.py configs/task1/scenarios/paired_medium_boxes_mms_left_arm14_experimental_v1.json tests/test_task1_scenario_config.py
git commit -m "Add paired-box Task1 scenario contract"
```

### Task 2: Active-Arm Composition Contract

**Files:**
- Modify: `data_collection/task1/contract.py`
- Modify: `tests/test_task1_plan_contract.py`

**Interfaces:**
- Produces: `active_arm_indices(active_arm: str) -> tuple[int, ...]`; `compose_active_arm14(active_arm, active_waypoints, reference_arm_q) -> np.ndarray`; `inactive_arm_is_fixed(active_arm, arm14_waypoints, reference_arm_q) -> bool`.

- [ ] **Step 1: Write failing tests**

```python
def test_compose_left_arm14_keeps_right_reference_exactly():
    reference = np.arange(14, dtype=float)
    active = np.asarray([[0.1] * 7, [0.2] * 7])
    result = compose_active_arm14("left", active, reference)
    np.testing.assert_allclose(result[:, :7], active)
    np.testing.assert_array_equal(result[:, 7:], np.repeat(reference[None, 7:], 2, axis=0))

def test_compose_active_arm14_rejects_wrong_width():
    with pytest.raises(ValueError, match="seven columns"):
        compose_active_arm14("left", [[0.0] * 6], [0.0] * 14)

def test_inactive_arm_fixed_detects_one_changed_right_joint():
    reference = np.arange(14, dtype=float)
    path = np.repeat(reference[None, :], 2, axis=0)
    assert inactive_arm_is_fixed("left", path, reference) is True
    path[1, 9] += 1e-12
    assert inactive_arm_is_fixed("left", path, reference) is False
```

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_plan_contract.py -k active_arm14
```

Expected: FAIL with missing helper imports.

- [ ] **Step 3: Implement the shared helper**

```python
def active_arm_indices(active_arm: str) -> tuple[int, ...]:
    try:
        start = {"left": 0, "right": 7}[active_arm]
    except KeyError as error:
        raise ValueError(f"unsupported active arm: {active_arm!r}") from error
    return tuple(range(start, start + 7))

def compose_active_arm14(active_arm, active_waypoints, reference_arm_q) -> np.ndarray:
    active = np.asarray(active_waypoints, dtype=float)
    reference = np.asarray(reference_arm_q, dtype=float)
    if active.ndim != 2 or active.shape[1] != 7:
        raise ValueError("active-arm trajectory must have seven columns")
    if reference.shape != (14,):
        raise ValueError("reference_arm_q must contain 14 values")
    if not np.isfinite(active).all() or not np.isfinite(reference).all():
        raise ValueError("active-arm inputs must be finite")
    result = np.repeat(reference[None, :], active.shape[0], axis=0)
    result[:, active_arm_indices(active_arm)] = active
    return result

def inactive_arm_is_fixed(active_arm, arm14_waypoints, reference_arm_q) -> bool:
    path = np.asarray(arm14_waypoints, dtype=float)
    reference = np.asarray(reference_arm_q, dtype=float)
    if path.ndim != 2 or path.shape[1] != 14 or reference.shape != (14,):
        raise ValueError("inactive-arm check requires an Nx14 path and 14-value reference")
    active = set(active_arm_indices(active_arm))
    inactive = tuple(index for index in range(14) if index not in active)
    expected = np.repeat(reference[None, inactive], path.shape[0], axis=0)
    return bool(np.array_equal(path[:, inactive], expected))
```

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_plan_contract.py
git add data_collection/task1/contract.py tests/test_task1_plan_contract.py
git commit -m "Add fixed-inactive-arm trajectory contract"
```

### Task 3: Inner-Flap Geometry and Collision Boundary

**Files:**
- Modify: `data_collection/task1/collision.py`
- Modify: `tests/test_task1_cumotion_collision_plan.py`

**Interfaces:**
- Produces: `paired_inner_flap_geometry(box_flaps, *, capture_width_m: float) -> dict`; explicit `allowed_target_flap_paths` support in `collision_world_config`.

- [ ] **Step 1: Write failing geometry and omission tests**

```python
def test_paired_inner_flap_geometry_selects_closest_opposed_pair():
    flaps = {
        "medium_box_0": [
            {"path": "/MediumBox_0/box/flap_left", "grasp_point_b_m": [0.6, 0.20, 1.1], "inward_normal_b": [0, 1, 0]},
            {"path": "/MediumBox_0/box/flap_right", "grasp_point_b_m": [0.6, 0.00, 1.1], "inward_normal_b": [0, -1, 0]},
        ],
        "medium_box_1": [
            {"path": "/MediumBox_1/box/flap_left", "grasp_point_b_m": [0.6, -0.01, 1.1], "inward_normal_b": [0, 1, 0]},
            {"path": "/MediumBox_1/box/flap_right", "grasp_point_b_m": [0.6, -0.21, 1.1], "inward_normal_b": [0, -1, 0]},
        ],
    }
    result = paired_inner_flap_geometry(flaps, capture_width_m=0.03)
    assert result["selected_flap_paths"] == [
        "/MediumBox_0/box/flap_right", "/MediumBox_1/box/flap_left"
    ]
    np.testing.assert_allclose(result["grasp_midpoint_b_m"], [0.6, -0.005, 1.1])
```

Add failures for non-opposed normals, ambiguous equal-distance pairs, missing
candidates, duplicate paths, and gap above capture width. Add a collision test
asserting only two exact paths are removed and all bodies/other flaps remain.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_collision_plan.py -k 'paired_inner or explicit_pair'
```

- [ ] **Step 3: Implement deterministic selection**

Normalize normals, enumerate the four cross-box combinations, require normal
dot product `<= -cos(10 degrees)`, sort by horizontal point distance and paths,
and reject a best-score tie within `1e-6 m`. Return both points, midpoint,
normalized point-to-point closing axis, separation, normals, and paths.

- [ ] **Step 4: Implement exact contact omission**

Extend `collision_world_config` with `allowed_target_flap_paths=()`. With
contact disabled, remove nothing. With two explicit paths, require two unique
existing collider paths and remove only exact matches. With no explicit paths,
preserve the existing single-box `MediumBox_0` behavior.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_collision_plan.py tests/test_gripper_collision.py
git add data_collection/task1/collision.py tests/test_task1_cumotion_collision_plan.py
git commit -m "Resolve paired inner flaps for Task1"
```

### Task 4: Pair-Aware Snapshot

**Files:**
- Modify: `data_collection/task1/pose_editor.py`
- Create: `tests/test_task1_pair_snapshot_source.py`

**Interfaces:**
- Consumes: `--paired-boxes medium_box_0 medium_box_1 --active-arm left`.
- Produces: `pose_editor_state.target_mode=paired`, both body poses, pair midpoint/axis/paths, and the inactive arm reference.

- [ ] **Step 1: Write a failing source-level CLI contract test**

```python
def test_pose_editor_declares_pair_snapshot_arguments():
    source = Path("data_collection/task1/pose_editor.py").read_text()
    tree = ast.parse(source)
    option_names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--paired-boxes", "--active-arm"} <= option_names
```

Do not import `pose_editor.py` in base-Python tests: it intentionally launches
Isaac at module import. Geometry shape/rejection is already covered in Task 3;
the real serialized runtime schema is validated on Kanu in Task 9.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_pair_snapshot_source.py
```

- [ ] **Step 3: Add pair snapshot CLI and Isaac resolution**

Add `--paired-boxes` with `nargs=2` and `--active-arm` with left/right choices.
Require both together and require planning-snapshot mode. Resolve both scene
assets and one `Body` id per asset; transform their body poses and both side
flap candidates into the robot-base frame.

- [ ] **Step 4: Serialize exact pair state**

Call `paired_inner_flap_geometry` with
`resolve_gripper_settings().pinch_close_threshold_m` (`0.055 m` in the checked-in
S200062 preset) as the capture-width gate, then write:

```python
{
    "target_mode": "paired",
    "paired_box_keys": ["medium_box_0", "medium_box_1"],
    "paired_box_body_poses_b": {"medium_box_0": pose0, "medium_box_1": pose1},
    "pair_grasp": {
        "active_arm": "left",
        "selected_flap_paths": paths,
        "grasp_points_b_m": points,
        "grasp_midpoint_b_m": midpoint,
        "closing_axis_b": axis,
        "initial_center_separation_m": separation,
        "capture_width_m": capture_width,
    },
    "reference_arm_q_rad": arm14,
}
```

Keep every current single-box key and browser-editor behavior unchanged in
single mode.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_pair_snapshot_source.py tests/test_task1_canonical_layout.py
git add data_collection/task1/pose_editor.py tests/test_task1_pair_snapshot_source.py
git commit -m "Capture paired-box Task1 planning snapshots"
```

### Task 5: One-Arm Endpoint

**Files:**
- Modify: `data_collection/task1/endpoint.py`
- Modify: `tests/test_task1_cumotion_bimanual_rmpflow_endpoint.py`

**Interfaces:**
- Consumes: Task 4 pair state and Task 2 composition helper.
- Produces: arm14 `plan.json` with `active_arm=left`, `active_solver_dof=7`, one TCP target, and fixed right-arm values.

- [ ] **Step 1: Write failing target-selection tests**

```python
def test_pair_endpoint_uses_left_tcp_and_pair_midpoint():
    inputs = endpoint_targets_from_editor_state({
        "target_mode": "paired",
        "pair_grasp": {"active_arm": "left", "grasp_midpoint_b_m": [0.55, -0.15, 1.12], "closing_axis_b": [0, 1, 0], "selected_flap_paths": ["/a", "/b"]},
    })
    assert inputs["active_arm"] == "left"
    assert inputs["tool_frames"] == [CENTER_TOOL_FRAMES["left"]]
    np.testing.assert_allclose(inputs["nominal_centers_b_m"], [[0.55, -0.15, 1.12]])
```

Also assert that serialized terminal columns 7-13 exactly equal the captured
right-arm reference.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_bimanual_rmpflow_endpoint.py -k pair
```

- [ ] **Step 3: Implement the pair branch**

Keep the current two-frame single-box branch unchanged. For pair mode, build a
single-arm XRDF with `collision.xrdf(side="left", ...)`, add only the left
center tool frame, solve from the captured left seven joints, and target the
pair midpoint/axis. Compose the selected 7-vector into arm14 with
`compose_active_arm14`.

Build the terminal collision world with
`allow_target_flap_contact=true` and the two exact
`pair_grasp.selected_flap_paths`. The endpoint inspector therefore relaxes only
those two contact colliders while keeping both box bodies and all other
obstacles.

Record `joint_names=ARM_JOINT_NAMES`, `cspace_dof=14`, `active_solver_dof=7`,
`inactive_arm_fixed=true`, the right-arm reference, pair geometry, and exact
terminal-contact flap paths.

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_bimanual_rmpflow_endpoint.py tests/test_task1_plan_contract.py
git add data_collection/task1/endpoint.py tests/test_task1_cumotion_bimanual_rmpflow_endpoint.py
git commit -m "Plan paired-box endpoint with one active arm"
```

### Task 6: One-Arm Approach and Retreat

**Files:**
- Modify: `data_collection/task1/approach.py`
- Modify: `data_collection/task1/retreat.py`
- Modify: `tests/test_task1_cumotion_bimanual_plan.py`
- Modify: `tests/test_task1_retreat_contract.py`

**Interfaces:**
- Consumes: Task 5 endpoint with canonical arm14 terminal pose and 7D active metadata.
- Produces: arm14 approach/retreat paths with invariant right-arm columns.

- [ ] **Step 1: Write failing invariance tests**

```python
def test_pair_path_keeps_right_arm_fixed():
    reference = np.arange(14, dtype=float)
    active = np.asarray([[0.0] * 7, [0.1] * 7])
    full = compose_active_arm14("left", active, reference)
    np.testing.assert_allclose(full[:, :7], active)
    np.testing.assert_array_equal(full[:, 7:], np.repeat(reference[None, 7:], 2, axis=0))
    assert inactive_arm_is_fixed("left", full, reference) is True
```

Add the equivalent retreat assertion and reject pair plans that request waist16
or have no active-arm metadata.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_bimanual_plan.py tests/test_task1_retreat_contract.py -k 'pair or active_arm'
```

- [ ] **Step 3: Implement the approach branch**

Plan, shortcut, densify, and collision-check only the left 7D c-space using the
single active TCP. Construct both a full-obstacle inspector and a
terminal-contact inspector that omits the two explicit flap paths. The planner
may use the terminal-contact world to reach the grasp, but final validation
must check every `path[:-1]` sample with the full-obstacle inspector and only
`path[-1]` with the terminal-contact inspector. Compose to arm14 after
validation and hard-fail with `inactive_arm_is_fixed` if any right-arm column
differs from the reference.

- [ ] **Step 4: Implement the retreat branch**

Lift and robotward-pull only the left TCP, preserving its grasp rotation when
requested. Do not execute the current right-first/left-follow coupling. Compose
each left 7D row into arm14 and use the average initial X center of both boxes
for rack-edge calculations. Record `attached_object_collision_validated=false`.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_cumotion_bimanual_plan.py tests/test_task1_retreat_contract.py tests/test_task1_plan_contract.py
git add data_collection/task1/approach.py data_collection/task1/retreat.py tests/test_task1_cumotion_bimanual_plan.py tests/test_task1_retreat_contract.py
git commit -m "Keep inactive arm fixed in paired-box motion"
```

### Task 7: Pure Pair Metrics

**Files:**
- Modify: `data_collection/task1/execution_contract.py`
- Modify: `tests/test_task1_execution_contract.py`

**Interfaces:**
- Produces: `paired_retention_metrics(...) -> dict`; `paired_box_acceptance(...) -> tuple[bool, str, dict]`; `executor_target_keys(paired_boxes, clear_same_shelf_boxes) -> tuple[str, ...]`.

- [ ] **Step 1: Write passing and one-box-slip tests**

```python
def test_paired_retention_metrics_require_both_boxes_to_follow_tcp():
    metrics = paired_retention_metrics(
        box_keys=("medium_box_0", "medium_box_1"),
        settled_positions={"medium_box_0": [0.60, 0.10, 1.0], "medium_box_1": [0.60, -0.10, 1.0]},
        closed_positions={"medium_box_0": [0.60, 0.10, 1.0], "medium_box_1": [0.60, -0.10, 1.0]},
        closed_tcp_position=[0.60, 0.0, 1.1],
        retreat_hold_samples=[
            {"box_positions_b_m": {"medium_box_0": [0.54, 0.10, 1.04], "medium_box_1": [0.54, -0.10, 1.04]}, "active_tcp_position_b_m": [0.54, 0.0, 1.14]},
        ],
    )
    assert metrics["box_robotward_progress_m"] == pytest.approx({"medium_box_0": 0.06, "medium_box_1": 0.06})
    assert metrics["pair_separation_drift_max_m"] == pytest.approx(0.0)
    assert metrics["hand_pair_center_drift_max_m"] == pytest.approx(0.0)
```

Add cases for one stationary box, 2 cm separation drift, 6 cm hand drift, one
box moving 2 cm during final hold, failed motor obstruction, and a missing box
sample.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_execution_contract.py -k paired
```

- [ ] **Step 3: Implement metrics and gates**

Compute pair center as the mean of two finite XYZ values. Reference both center
distance and TCP-relative pair-center vector at close; take maximum drift over
retreat plus final hold. Compute robotward progress as `settled_x-final_x` for
each box. Reject duplicate/missing keys and empty sample lists.

`paired_box_acceptance` returns true only when approach contact-free, tracking
within tolerance, both progress values pass 0.05 m, both final-hold values pass
0.01 m, separation passes 0.01 m, hand drift passes 0.05 m, and left motor
obstruction passes. Return every named sub-gate for the report. Do not change
the existing `physical_acceptance` function.

Add `executor_target_keys` here rather than in `execute.py`, because the latter
starts Isaac during import. It returns `("medium_box_0",)` for legacy mode,
returns the two validated keys only when same-shelf clearing is false, and
rejects duplicate pair keys.

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_execution_contract.py
git add data_collection/task1/execution_contract.py tests/test_task1_execution_contract.py
git commit -m "Add paired-box physical acceptance metrics"
```

### Task 8: Pair Executor and Artifact Finalization

**Files:**
- Modify: `data_collection/task1/execute.py`
- Modify: `tests/test_task1_execution_contract.py`
- Modify: `tests/test_task1_scenario_config.py`
- Modify: `tests/test_task1_video.py`
- Create: `tests/test_task1_execute_pair_source.py`

**Interfaces:**
- Consumes: pair CLI, snapshot, arm14 plans, and Task 7 metrics.
- Produces: finalized MP4 and report with `paired_box_success`, `per_box`, `pair_metrics`, `acceptance_gates`, hashes, and both-box samples.

- [ ] **Step 1: Write failing parser/selection/report tests**

```python
def test_pair_selection_retains_both_targets_and_third_box():
    targets = executor_target_keys(
        paired_boxes=("medium_box_0", "medium_box_1"),
        clear_same_shelf_boxes=False,
    )
    assert targets == ("medium_box_0", "medium_box_1")

def test_execute_declares_pair_options_and_report_fields():
    source = Path("data_collection/task1/execute.py").read_text()
    tree = ast.parse(source)
    option_names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert {"--scenario-path", "--paired-boxes", "--clear-same-shelf-boxes", "--pair-separation-drift-max-m"} <= option_names
    assert '"paired_box_success"' in source
    assert '"box_body_positions_b_m"' in source
```

Require pair mode to reject the wrong gripper, clearing same-shelf boxes,
duplicate keys, and nonpositive separation threshold. Pure metrics must fail if
any sample lacks either target; the source-level test avoids importing
`execute.py`, which intentionally starts Isaac during import.

- [ ] **Step 2: Prove failure**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_execution_contract.py tests/test_task1_scenario_config.py tests/test_task1_video.py tests/test_task1_execute_pair_source.py -k 'pair or paired'
```

- [ ] **Step 3: Add pair CLI without changing defaults**

Add `--scenario-path` as a required `Path` for pair mode, `--paired-boxes` with two values,
`--clear-same-shelf-boxes/--no-clear-same-shelf-boxes` with default true, and
`--pair-separation-drift-max-m` default 0.01. Single mode remains
`medium_box_0` with same-shelf parking.

- [ ] **Step 4: Track both target bodies in every phase**

Resolve an ordered mapping of both assets/body IDs. Store
`box_body_positions_b_m` keyed by runtime name in each pair sample. At settle,
compare both poses against the snapshot. At approach, require each box to stay
within the existing motion threshold. Capture both close/final poses and left
TCP reference.

- [ ] **Step 5: Serialize pair acceptance and preserve failures**

Write:

```python
{
    "passed": paired_box_success,
    "paired_box_success": paired_box_success,
    "acceptance_mode": "paired_box_partial_extraction",
    "target_mode": "paired",
    "paired_boxes": ["medium_box_0", "medium_box_1"],
    "active_gripper": "left",
    "per_box": per_box_metrics,
    "pair_metrics": pair_metrics,
    "acceptance_gates": gates,
    "source_commit": source_commit,
    "scenario_sha256": scenario_sha256,
    "rack_box_poses_sha256": rack_sha256,
    "approach_plan_sha256": approach_sha256,
    "retreat_plan_sha256": retreat_sha256,
    "samples": samples,
}
```

Compute `scenario_sha256` from `--scenario-path`, the rack hash from the
resolved `--rack-box-poses`, and the two plan hashes from their exact bytes.
Resolve `source_commit` with `git rev-parse HEAD` before Isaac execution begins.

Finalize video before the final report. A gate failure still writes MP4/report.
An encoding failure writes `passed=false`, `artifact_status=video_error`, the
error text, and frame directory; it cannot report pair success.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_execution_contract.py tests/test_task1_scenario_config.py tests/test_task1_video.py tests/test_task1_execute_pair_source.py tests/test_gripper_runtime.py tests/test_gripper_io.py
git add data_collection/task1/execute.py tests/test_task1_execution_contract.py tests/test_task1_scenario_config.py tests/test_task1_video.py tests/test_task1_execute_pair_source.py
git commit -m "Execute and report paired-box Task1 runs"
```

### Task 9: Regression, Kanu Run, and Video Retrieval

**Files:**
- Verify: all committed source and tests.
- Create on Kanu: `/home/seonho/outputs/HumanoidScene/task1_paired_box_mms_left_20260914_v1/`
- Retrieve to: `/home/minje/.codex/worktrees/humanoidscene-main-tcp-video-20260911/.artifacts/task1_paired_box_mms_left_20260914_v1/`

**Interfaces:**
- Produces: source-matched logs/plans/report/MP4 and local/remote hashes.

- [ ] **Step 1: Run the full scoped local gate**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_task1_*.py tests/test_gripper_collision.py tests/test_gripper_config.py tests/test_gripper_io.py tests/test_gripper_runtime.py tests/test_urdf_arm_ik.py
git diff --check
git status --short --branch
```

Expected: all tests PASS, diff check exits 0, and no uncommitted source/test changes.

- [ ] **Step 2: Audit Kanu before mutation**

```bash
ssh -o BatchMode=yes kanu 'bash -s' <<'REMOTE'
set -euo pipefail
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo pid=,stat=,etime=,cmd= | grep -E 'Isaac|task1/(pose_editor|endpoint|approach|retreat|execute)' | grep -v grep || true
git -C /home/seonho/worktrees/HumanoidScene_main_tcp_video_20260911 status --short --branch
test ! -e /home/seonho/outputs/HumanoidScene/task1_paired_box_mms_left_20260914_v1
REMOTE
```

Select and record one GPU with no competing Isaac process and enough memory;
do not launch if no lane is free.

- [ ] **Step 3: Push and deploy the exact commit**

```bash
git fetch origin codex/data-collection
git merge-base --is-ancestor origin/codex/data-collection HEAD
git push origin HEAD:codex/data-collection
ssh -o BatchMode=yes kanu 'bash -s' <<'REMOTE'
set -euo pipefail
repo=/home/seonho/worktrees/HumanoidScene_main_tcp_video_20260911
git -C "$repo" fetch origin codex/data-collection
git -C "$repo" merge --ff-only origin/codex/data-collection
git -C "$repo" status --short --branch
git -C "$repo" rev-parse HEAD
REMOTE
```

Compare SHA256 for every changed production file, the scenario, and MMS-left
rack pose; stop on mismatch.

- [ ] **Step 4: Capture pair snapshot**

Run `data_collection.task1.pose_editor` in snapshot-only mode with the audited
GPU, MMS-left rack pose, `--paired-boxes medium_box_0 medium_box_1`, and
`--active-arm left`. Write only under the unique run root. Require two settled
poses, two unique flap paths, finite midpoint/axis, and active arm left.

- [ ] **Step 5: Generate and validate plans**

Run endpoint, approach, and retreat with the checked-in S200062 URDF and current
collision/step limits. Save exact commands and logs. Require status SUCCESS,
canonical 14 joint names, active solver 7D, exact right-arm invariance, no
world/self collision, and the existing maximum joint-step gate.

- [ ] **Step 6: Execute one immutable physical attempt**

Build arguments from the checked-in pair scenario and run on the same GPU.
Write `physical_v1/report.json`, `physical_v1/task1_mms_left_single_arm.mp4`,
and `physical_v1/run.log`. Do not change thresholds after viewing the result. A
diagnostic rerun uses `physical_v2` and the same acceptance limits.

- [ ] **Step 7: Verify evidence**

```bash
jq '{passed,paired_box_success,paired_boxes,active_gripper,per_box,pair_metrics,acceptance_gates,video_encoding}' "$TASK1_RUN_ROOT/physical_v1/report.json"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration -of default=noprint_wrappers=1 "$TASK1_RUN_ROOT/physical_v1/task1_mms_left_single_arm.mp4"
sha256sum "$TASK1_RUN_ROOT/physical_v1/report.json" "$TASK1_RUN_ROOT/physical_v1/task1_mms_left_single_arm.mp4"
```

Require both box entries and all seven gates. Report failure if either box
progress is below 5 cm, separation exceeds 1 cm, hand drift exceeds 5 cm,
either hold motion exceeds 1 cm, tracking/obstruction fails, or MP4 is invalid.

- [ ] **Step 8: Retrieve and hand off**

Copy only this run into the explicit local artifact root without touching the
dirty checkout's tracked files. Verify local/remote report and MP4 SHA256
parity. Do not stage `.artifacts`. In the final response, render the MP4 from
its absolute path and link the report, scenario, and rack-pose JSON.

- [ ] **Step 9: Confirm branch state**

```bash
git status --short --branch
git log --oneline --decorate -10
git ls-remote --heads origin codex/data-collection
```

Confirm the remote branch points to the implementation commit, the canonical
worktree is clean, the dirty user checkout was not altered, and the physical
claim matches only `report.json` evidence.

import numpy as np
import pytest
import yaml

from data_collection.task1_cumotion_bimanual_plan import (
    ARM_JOINT_NAMES,
    TOOL_FRAMES,
    bimanual_xrdf,
    densify_path,
    joint_space_path_length,
    planner_yaml,
    rack_width_constraint_in_base,
    rack_width_coordinates_m,
    rack_width_max_violation_m,
    rack_width_task_space_limits,
    shortcut_path,
    synchronized_seed_path,
)


def test_bimanual_xrdf_has_one_14dof_cspace_and_two_tools():
    sphere = {"center": [0.1, 0.0, 0.0], "radius": 0.02}
    data = yaml.safe_load(
        bimanual_xrdf(
            {name: 0.0 for name in ARM_JOINT_NAMES},
            {"zarm_l2_link": [sphere], "zarm_r2_link": [sphere]},
            {"zarm_l2_link": [sphere], "zarm_r2_link": [sphere]},
        )
    )

    assert data["cspace"]["joint_names"] == ARM_JOINT_NAMES
    assert data["tool_frames"] == TOOL_FRAMES
    assert len(data["cspace"]["acceleration_limits"]) == 14


def test_planner_config_weights_all_14_joints():
    data = yaml.safe_load(
        planner_yaml(len(ARM_JOINT_NAMES), seed=42, step_size=0.03)
    )

    assert data["distance_metric_weights"] == [8.0] + [1.0] * 6 + [8.0] + [1.0] * 6
    assert data["cspace_planning_params"]["exploration_fraction"] == 0.5
    assert data["seed"] == 42
    assert data["step_size"] == 0.03


def test_planner_config_accepts_rack_front_workspace_corridor():
    limits = [[0.1, 0.8], [0.1, 0.35], [0.4, 1.4]]

    data = yaml.safe_load(planner_yaml(14, task_space_limits=limits))

    assert data["task_space_limits"] == limits


def test_densify_path_limits_each_joint_step():
    path = np.asarray([[0.0, 0.0], [0.025, -0.011], [0.03, 0.0]])
    dense = densify_path(path, 0.01)

    assert np.allclose(dense[0], path[0])
    assert np.allclose(dense[-1], path[-1])
    assert np.max(np.abs(np.diff(dense, axis=0))) <= 0.01 + 1e-12


def test_shortcut_path_prefers_direct_segment_when_clear():
    path = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    shortcut = shortcut_path(path, 0.1, lambda _q: False)

    np.testing.assert_allclose(shortcut, [[0.0, 0.0], [1.0, 1.0]])
    assert joint_space_path_length(shortcut) == pytest.approx(np.sqrt(2.0))


def test_shortcut_path_keeps_required_collision_avoidance_knot():
    path = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def in_collision(q):
        return 0.35 < q[0] < 0.65 and 0.35 < q[1] < 0.65

    shortcut = shortcut_path(path, 0.05, in_collision)

    np.testing.assert_allclose(shortcut, path)


def test_synchronized_seed_path_uses_equal_arm_progress():
    report = {
        "arms": {
            "left": {"sample_q_rad": [[0] * 7, [1] * 7, [2] * 7]},
            "right": {"sample_q_rad": [[10] * 7, [14] * 7]},
        }
    }
    initial = np.asarray([0] * 7 + [10] * 7, dtype=float)
    terminal = np.asarray([2] * 7 + [14] * 7, dtype=float)

    path = synchronized_seed_path(report, initial, terminal)

    assert path.shape == (3, 14)
    np.testing.assert_allclose(path[1], [1] * 7 + [12] * 7)
    np.testing.assert_allclose(path[[0, -1]], [initial, terminal])


def test_rack_width_constraint_is_transformed_to_robot_base():
    root_pose_w = np.asarray([1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    constraint = {
        "center_w_m": [1.0, 1.5, 1.0],
        "axis_w": [0.0, 1.0, 0.0],
        "half_width_m": 0.5,
    }

    center_b, axis_b, half_width_m = rack_width_constraint_in_base(
        root_pose_w, constraint
    )

    np.testing.assert_allclose(center_b, [0.0, -0.5, 1.0])
    np.testing.assert_allclose(axis_b, [0.0, 1.0, 0.0])
    assert half_width_m == 0.5


def test_rack_width_gate_checks_both_eefs_and_only_width_axis():
    positions = np.asarray(
        [
            [[10.0, 0.25, -5.0], [-8.0, -0.49, 9.0]],
            [[20.0, 0.50, 12.0], [-6.0, -0.57, -7.0]],
        ]
    )
    coordinates = rack_width_coordinates_m(
        positions, np.zeros(3), np.asarray([0.0, 1.0, 0.0])
    )

    np.testing.assert_allclose(coordinates, [[0.25, -0.49], [0.50, -0.57]])
    assert rack_width_max_violation_m(coordinates[:1], 0.5) == 0.0
    assert rack_width_max_violation_m(coordinates, 0.5) == pytest.approx(0.07)


def test_axis_aligned_rack_width_guides_cumotion_task_space():
    limits = rack_width_task_space_limits(
        np.asarray([0.5, -0.06, 1.0]), np.asarray([0.0, -1.0, 0.0]), 0.5255
    )

    np.testing.assert_allclose(
        limits, [[-1.5, 1.5], [-0.5855, 0.4655], [-0.5, 2.5]]
    )
    assert rack_width_task_space_limits(
        np.zeros(3), np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0]), 0.5
    ) is None

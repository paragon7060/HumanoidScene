import numpy as np
import pytest
import yaml

from data_collection.task1_cumotion_bimanual_plan import (
    ARM_JOINT_NAMES,
    TOOL_FRAMES,
    bimanual_xrdf,
    corridor_max_violation_m,
    densify_path,
    joint_space_path_length,
    planner_yaml,
    shortcut_path,
    task_urdf_with_joint_bounds,
    workspace_corridor_bounds,
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


def test_workspace_corridor_reports_outward_tcp_excursion():
    start = np.asarray([[0.15, 0.25, 0.47], [0.15, -0.25, 0.47]])
    target = np.asarray([[0.69, 0.27, 1.34], [0.69, -0.05, 1.34]])
    bounds = workspace_corridor_bounds(start, target, 0.08)
    positions = np.stack((start, target))

    assert corridor_max_violation_m(positions, bounds) == 0.0
    positions[1, 1, 1] = -0.40
    assert corridor_max_violation_m(positions, bounds) == pytest.approx(0.07)


def test_task_urdf_narrows_only_requested_joint_limits():
    urdf = """<robot name="r">
      <joint name="q1" type="revolute"><limit lower="-2" upper="2"/></joint>
      <joint name="q2" type="revolute"><limit lower="-3" upper="3"/></joint>
    </robot>"""

    bounded = task_urdf_with_joint_bounds(urdf, {"q1": (0.1, 0.9)})

    assert 'name="q1"' in bounded
    assert 'lower="0.1" upper="0.9"' in bounded
    assert 'name="q2"' in bounded
    assert 'lower="-3" upper="3"' in bounded

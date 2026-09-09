import numpy as np
import yaml

from data_collection.task1_cumotion_bimanual_plan import (
    ARM_JOINT_NAMES,
    TOOL_FRAMES,
    bimanual_xrdf,
    densify_path,
    planner_yaml,
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
    data = yaml.safe_load(planner_yaml(len(ARM_JOINT_NAMES)))

    assert data["distance_metric_weights"] == [1.0] * 14
    assert data["cspace_planning_params"]["exploration_fraction"] == 0.5


def test_densify_path_limits_each_joint_step():
    path = np.asarray([[0.0, 0.0], [0.025, -0.011], [0.03, 0.0]])
    dense = densify_path(path, 0.01)

    assert np.allclose(dense[0], path[0])
    assert np.allclose(dense[-1], path[-1])
    assert np.max(np.abs(np.diff(dense, axis=0))) <= 0.01 + 1e-12

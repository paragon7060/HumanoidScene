import yaml

from data_collection.task1_cumotion_bimanual_plan import (
    ARM_JOINT_NAMES,
    TOOL_FRAMES,
    bimanual_xrdf,
    rmpflow_yaml,
)


def test_bimanual_xrdf_has_one_14dof_cspace_and_two_tools():
    sphere = {"center": [0.1, 0.0, 0.0], "radius": 0.02}
    xrdf_text, controllers = bimanual_xrdf(
        {name: 0.0 for name in ARM_JOINT_NAMES},
        {"zarm_l2_link": [sphere], "zarm_r2_link": [sphere]},
        {"zarm_l2_link": [sphere], "zarm_r2_link": [sphere]},
    )
    data = yaml.safe_load(xrdf_text)

    assert data["cspace"]["joint_names"] == ARM_JOINT_NAMES
    assert data["tool_frames"] == TOOL_FRAMES
    assert len(data["modifiers"]) == 2
    assert len(controllers) == 2


def test_rmpflow_config_registers_all_collision_controllers():
    controllers = [
        {"name": "left_collision", "radius": 0.03},
        {"name": "right_collision", "radius": 0.04},
    ]
    data = yaml.safe_load(rmpflow_yaml(len(ARM_JOINT_NAMES), controllers))

    assert len(data["joint_limit_buffers"]) == 14
    assert data["body_collision_controllers"] == controllers

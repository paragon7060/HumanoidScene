"""Mobile flap configuration and actual integrator behavior without Isaac startup."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import torch
import pytest
import argparse

from kuavo_isaaclab_scene.rl.mdp.settling import gate_actions
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec
from kuavo_isaaclab_scene.rl.action_spaces import add_action_space_argument, select_task_action_space
from kuavo_isaaclab_scene.rl.multi_box._legacy_spec import MultiBoxSpec

ROOT = Path(__file__).resolve().parents[1]


def test_shared_action_selection_overrides_experiment_and_keeps_grasp_goal():
    parser = argparse.ArgumentParser()
    add_action_space_argument(parser)
    config = runpy.run_path(str(ROOT / "configs/rl_pick_whole_body.py"))
    original = config["configure_task"](task_spec("pick"))
    for choice, mode, arm in (("right-arm", "arms-only", "right"),
                             ("all-joints", "whole-body", "both")):
        args = parser.parse_args(["--action-space", choice])
        selected = select_task_action_space(original, args.action_space)
        selected.validate()
        assert (selected.control_mode, selected.active_arm) == (mode, arm)
        assert selected.grasp_hand == "right" and selected.required_grasp_hands == 1
        assert selected.lift_height == original.lift_height
    assert select_task_action_space(original, None) is original
    with pytest.raises(ValueError, match="requires base motion"):
        select_task_action_space(task_spec("full"), "right-arm")
    MultiBoxSpec(strategy="staged", skill="pick", action_space="right-arm").validate()
    with pytest.raises(ValueError, match="requires base motion"):
        MultiBoxSpec(action_space="right-arm").validate()


def production_method(class_name, method_name):
    # Execute the unmodified method body; avoid simulator-only base classes.
    path = ROOT / "src/kuavo_isaaclab_scene/rl/mdp/actions.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = {"torch": torch, "gate_actions": gate_actions}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


def test_mobile_config_keeps_lift_goal_and_releases_both_arms_and_torso():
    cfg = runpy.run_path(str(ROOT / "configs/rl_pick_whole_body.py"))
    spec = cfg["configure_task"](task_spec("pick", box_names=("medium_box_0",)))
    spec.validate()
    assert spec.control_mode == "whole-body" and spec.active_arm == "both"
    assert spec.grasp_hand == "right" and spec.required_grasp_hands == 1
    assert spec.lift_height == .06 and spec.hold_seconds == .5
    assert not spec.randomization and spec.collision_constraints_enabled
    assert spec.obstacle_contact_force == pytest.approx(.1)
    env = NS(task=spec, actions=NS(base=NS(), upper_body=NS(), height=NS(), head=NS(),
        left_gripper=NS(), right_gripper=NS()), rewards=NS(prelift_disturbance=NS(),
        orientation=NS(params={})))
    cfg["configure"](env, NS())
    assert env.actions.base.velocity_limits == (.15, .15, .5)
    assert env.actions.upper_body.scale["waist_yaw_joint"] == .01
    assert env.actions.height.scale == .015
    assert (ROOT / "configs/rl_pick_whole_body.py").read_bytes() == (
        ROOT / "src/kuavo_isaaclab_scene/configs/rl_pick_whole_body.py").read_bytes()


def test_mobile_settling_holds_base_and_joint_targets_before_control():
    command = NS(settling=NS(ready=torch.tensor([False, True])))
    env = NS(cfg=NS(task=NS(reset_settle_seconds=.5, reset_bank=None, grasp_mode="flap_top")),
        step_dt=.1, command_manager=NS(get_term=lambda _: command))
    drive = NS(_env=env, _raw=torch.zeros(2, 3), _velocity=torch.full((2, 3), .2),
        _scale=torch.ones(3), _accel=torch.ones(3))
    production_method("PlanarDrive", "process_actions")(drive, torch.ones(2, 3))
    assert drive._velocity[0].eq(0).all()
    torch.testing.assert_close(drive._velocity[1], torch.full((3,), .3))
    # A measured pose outside the softer margin must not jump on zero action.
    physical = torch.tensor([[[-1., 1.]], [[-1., 1.]]])
    joint = NS(_env=env, _raw_actions=torch.zeros(2, 1), _targets=torch.full((2, 1), .9),
        _scale=.02, _joint_ids=[0], _processed_actions=torch.zeros(2, 1),
        _asset=NS(data=NS(joint_pos_limits=physical, soft_joint_pos_limits=physical * .8)))
    integrate = production_method("JointDeltaTargets", "process_actions")
    integrate(joint, torch.ones(2, 1))
    torch.testing.assert_close(joint._targets, torch.tensor([[.9], [.92]]))
    before = joint._targets.clone()
    integrate(joint, torch.zeros(2, 1))
    torch.testing.assert_close(joint._targets, before)

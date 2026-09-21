"""Single-environment v2 orchestration with replaceable sensor sources."""

from dataclasses import replace

import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import HighLevelSelection, SKILL_IDS
from kuavo_isaaclab_scene.rl.multi_box.runtime import MultiBoxDeployableRuntime
from kuavo_isaaclab_scene.rl.multi_box.state.perception import PerceptionFrame
from kuavo_isaaclab_scene.rl.multi_box.state.schema import DeployableBoxState, DeployableRobotState


def _poses(*shape):
    result = torch.zeros(*shape, 7)
    result[..., 3] = 1.0
    return result


class Source:
    def __init__(self, value):
        self.value = value
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return self.value


def _sources():
    active = torch.zeros(1, 12, dtype=torch.bool)
    active[0, :2] = True
    size = torch.zeros(1, 12, 3)
    size[0, :2] = torch.tensor([0.266, 0.185, 0.130])
    pose = _poses(1, 12)
    pose[0, :2, 2] = 0.30
    pose[0, 1, 0] = 0.50
    frame = PerceptionFrame(
        boxes=DeployableBoxState(
            active=active, box_type_id=torch.zeros(1, 12, dtype=torch.long),
            rack_region_id=torch.zeros(1, 12, dtype=torch.long),
            size_m=size, pose_world=pose, pose_confidence=active.float()),
        rack_pose_world=_poses(1), conveyor_pose_world=_poses(1))
    robot = DeployableRobotState(
        joint_pos=torch.zeros(1, 20), joint_vel=torch.zeros(1, 20),
        base_pose_world=_poses(1), base_twist_world=torch.zeros(1, 6),
        tcp_pose_world=pose[:, :1].expand(-1, 2, -1).clone(),
        gripper_position=torch.ones(1, 2), gripper_command=torch.ones(1, 2))
    return Source(frame), Source(robot)


def _move(perception, robot, target, z, *, closed=True):
    box_pose = perception.value.boxes.pose_world.clone()
    box_pose[0, target, 2] = z
    perception.value = replace(perception.value, boxes=replace(
        perception.value.boxes, pose_world=box_pose))
    robot.value = replace(
        robot.value,
        tcp_pose_world=box_pose[:, target:target + 1].expand(-1, 2, -1).clone(),
        gripper_position=torch.ones(1, 2) if closed else torch.zeros(1, 2))


def _step(runtime):
    return runtime.step(previous_action=torch.zeros(1, 25), dt=0.1)


def _finish_one(runtime, perception, robot, target):
    _move(perception, robot, target, 0.309)
    for _ in range(3):
        result = _step(runtime)
    assert result.state.control.current_skill.item() == SKILL_IDS["carry"]
    _move(perception, robot, target, 0.115 + 0.005 * 0.130)
    result = _step(runtime)
    assert result.state.control.current_skill.item() == SKILL_IDS["place"]
    _move(perception, robot, target, 0.015 + 0.005 * 0.130, closed=False)
    for _ in range(5):
        result = _step(runtime)
    assert result.state.control.completed_box.item() == target
    assert result.state.control.needs_target.item()
    assert result.state.placement.placed[0, target]


def test_runtime_selects_each_box_once_and_waits_when_all_are_placed():
    perception, robot = _sources()
    runtime = MultiBoxDeployableRuntime(
        num_envs=1, device="cpu", perception_source=perception, robot_source=robot)
    first = _step(runtime)
    assert first.selected_box.item() == 0
    assert first.state.control.target_box.item() == 0
    assert first.actor_observation.target_box.item() == 0
    assert perception.read_count == robot.read_count == 1
    _finish_one(runtime, perception, robot, 0)

    next_box = _step(runtime)
    assert next_box.selected_box.item() == 1
    assert next_box.state.control.target_box.item() == 1
    assert next_box.state.placement.placed[0, 0]
    _move(perception, robot, 1, 0.30)
    _step(runtime)  # Capture the closed-hand relative pose for the new target.
    _finish_one(runtime, perception, robot, 1)

    idle = _step(runtime)
    assert idle.selected_box.item() == -1
    assert idle.waiting_without_box.item()
    assert idle.state.control.needs_target.item()
    assert idle.state.placement.placed[0, :2].all()

    restored_perception, restored_robot = _sources()
    perception.value, robot.value = restored_perception.value, restored_robot.value
    runtime.reset()
    restarted = _step(runtime)
    assert restarted.selected_box.item() == 0
    assert not restarted.state.placement.placed[0, 0]
    assert restarted.state.control.current_skill.item() == SKILL_IDS["grasp"]


def test_high_level_selector_is_replaceable_and_receives_actor_observation():
    class LastSelectable:
        def select(self, observation, selectable_boxes, waiting_env_ids):
            assert observation.needs_target[waiting_env_ids].all()
            assert observation.box_mask[waiting_env_ids, :2].all()
            return HighLevelSelection(torch.tensor([1]))

    perception, robot = _sources()
    runtime = MultiBoxDeployableRuntime(
        num_envs=1, device="cpu", perception_source=perception, robot_source=robot,
        selector=LastSelectable())
    result = _step(runtime)
    assert result.selected_box.item() == 1
    assert result.state.control.target_box.item() == 1

"""CPU-only checks for the v2 design skeleton."""

from dataclasses import replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box import (
    MultiBoxTaskSkeleton,
    PendingTaskDesignError,
    SemanticPipeline,
    TaskSemantics,
)
from kuavo_isaaclab_scene.rl.multi_box.hierarchy import SkillDispatch, SkillRouter
from kuavo_isaaclab_scene.rl.multi_box.reset_bank import ResetSnapshot
from kuavo_isaaclab_scene.rl.multi_box.skills import SkillRegistry
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


class Provider:
    def __init__(self): self.calls = []
    def evaluate(self, context): return context
    def reset(self, context, env_ids): self.calls.append("reset")
    def update(self, context): self.calls.append("state"); return "state"
    def build(self, context, task_state): self.calls.append("observation"); return "observation"
    def compute(self, context, task_state, success): self.calls.append("reward"); return "reward"


class Policy:
    def __init__(self, value):
        self.value = value

    def act(self, observation, selected_box, env_ids):
        return torch.full((len(env_ids), 3), self.value)


def test_scene_contract_has_no_default_timeout_or_task_semantics():
    skeleton = MultiBoxTaskSkeleton()
    skeleton.validate_scene()
    assert skeleton.spec.episode_seconds is None
    assert skeleton.spec.workspace_radius == 1.5
    assert skeleton.spec.collision_constraints_enabled
    with pytest.raises(PendingTaskDesignError, match="success, state, observation, reward"):
        skeleton.validate_training()
    with pytest.raises(ValueError, match="timeout"):
        replace(skeleton.spec, episode_seconds=0).validate()


def test_training_requires_all_semantics_and_all_three_skills():
    provider = Provider()
    semantics = TaskSemantics(provider, provider, provider, provider)
    skeleton = MultiBoxTaskSkeleton(semantics=semantics)
    with pytest.raises(ValueError, match="grasp, carry, place"):
        skeleton.validate_training()
    skeleton.skills = SkillRegistry(
        grasp=Policy(0), carry=Policy(1), place=Policy(2))
    skeleton.validate_training()


def test_semantic_pipeline_only_defines_call_order():
    provider = Provider()
    class Success:
        def evaluate(self, context):
            provider.calls.append("success")
            return "success"
    pipeline = SemanticPipeline(TaskSemantics(Success(), provider, provider, provider))
    pipeline.reset(None, torch.tensor([0]))
    step = pipeline.step(None)
    assert provider.calls == ["reset", "state", "success", "observation", "reward"]
    assert (step.task_state, step.success, step.observation, step.reward) == (
        "state", "success", "observation", "reward")


def test_high_level_action_rejects_inactive_boxes_and_router_dispatches():
    active = torch.zeros(3, 12, dtype=torch.bool)
    active[0, 2] = active[1, 4] = active[2, 9] = True
    selection = SkillDispatch(
        box_index=torch.tensor([2, 4, 9]),
        skill_index=torch.tensor([0, 1, 2]),
    )
    selection.validate(active)
    registry = SkillRegistry(grasp=Policy(1), carry=Policy(2), place=Policy(3))
    actions = SkillRouter(registry).act(None, selection, active)
    assert torch.equal(actions[:, 0], torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="inactive"):
        SkillDispatch(torch.tensor([1, 4, 9]), selection.skill_index).validate(active)


def test_reset_snapshot_is_format_neutral_detached_and_versioned():
    snapshot = ResetSnapshot("grasp", {"robot/joint_pos": torch.ones(2, 3)})
    clone = snapshot.cpu_copy()
    assert clone.source_skill == "grasp"
    assert clone.tensors["robot/joint_pos"].device.type == "cpu"
    assert clone.tensors["robot/joint_pos"].data_ptr() != snapshot.tensors["robot/joint_pos"].data_ptr()
    with pytest.raises(ValueError, match="detached"):
        ResetSnapshot("carry", {"bad": torch.ones(1, requires_grad=True)}).validate()

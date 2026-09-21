"""Isaac sources wired to the deployable v2 single-step runtime."""

from __future__ import annotations

import torch

from ..runtime import HighLevelBoxSelector, MultiBoxDeployableRuntime
from .isaac_perception import IsaacScenePerceptionAdapter
from .isaac_robot_proprio import IsaacRobotProprioAdapter


class IsaacDeployableStateAdapter:
    """Opt-in v2 runtime; call reset after each Isaac environment reset."""

    def __init__(self, env, *, selector: HighLevelBoxSelector | None = None):
        self.perception = IsaacScenePerceptionAdapter(env)
        self.robot = IsaacRobotProprioAdapter(env)
        self.runtime = MultiBoxDeployableRuntime(
            num_envs=env.num_envs, device=env.device,
            perception_source=self.perception, robot_source=self.robot,
            selector=selector)

    def reset(self, env_ids=None) -> None:
        self.runtime.reset(env_ids)

    def step(self, *, previous_action: torch.Tensor, dt: float):
        return self.runtime.step(previous_action=previous_action, dt=dt)

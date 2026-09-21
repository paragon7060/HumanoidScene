"""Isaac scene pose source for the deployable v2 perception interface."""

from __future__ import annotations

import torch

from ..scene.spawn import physical_asset_names
from .perception import PerceptionFrame, simulated_perception_frame


class IsaacScenePerceptionAdapter:
    """Use simulator truth temporarily, behind a replaceable perception API."""

    def __init__(self, env):
        self.env = env
        self.asset_names = physical_asset_names()

    def read(self) -> PerceptionFrame:
        env = self.env
        physical_poses = torch.stack(
            [env.scene[name].data.root_pose_w for name in self.asset_names], dim=1)
        return simulated_perception_frame(
            active=env._multi_box_active,
            box_type_id=env._multi_box_box_type_ids,
            rack_region_id=env._multi_box_region_ids,
            pool_id=env._multi_box_pool_ids,
            physical_box_poses_world=physical_poses,
            rack_pose_world=env.scene["rack"].data.root_pose_w,
            conveyor_pose_world=env.scene["conveyor_surface"].data.root_pose_w,
        )

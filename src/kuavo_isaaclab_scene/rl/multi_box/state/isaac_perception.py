"""Isaac scene pose source for the deployable v2 perception interface."""

from __future__ import annotations

import torch

from ..scene.spawn import physical_asset_names
from ..geometry.pose import quat_apply
from .perception import PerceptionFrame, simulated_perception_frame


class IsaacScenePerceptionAdapter:
    """Use simulator truth temporarily, behind a replaceable perception API."""

    def __init__(self, env):
        self.env = env
        self.asset_names = physical_asset_names()
        self.flap_ids = self.flap_centers = None
        if env.cfg.multi_box.flap_pose_source == "articulated":
            # Static asset geometry only. No privileged grasp/contact adapter
            # is imported by this replaceable perception source.
            from ...scenes.asset_geometry import box_geometry
            names = ("flap_right", "flap_left")
            ids, centers = [], []
            for name in self.asset_names:
                asset = env.scene[name]
                body_ids, _ = asset.find_bodies(names, preserve_order=True)
                if len(body_ids) != 2:
                    raise ValueError(f"{name} lacks two perceived flap panels")
                geometry = box_geometry(getattr(env.cfg.scene, name), names)
                ids.append(body_ids)
                centers.append([geometry.flaps[key].center for key in names])
            self.flap_ids = ids
            self.flap_centers = torch.tensor(centers, device=env.device, dtype=torch.float32)

    def read(self) -> PerceptionFrame:
        env = self.env
        physical_poses = torch.stack(
            [env.scene[name].data.root_pose_w for name in self.asset_names], dim=1)
        physical_flaps = None
        if self.flap_ids is not None:
            position = torch.stack([
                env.scene[name].data.body_link_pos_w[:, ids]
                for name, ids in zip(self.asset_names, self.flap_ids, strict=True)
            ], dim=1)
            quaternion = torch.stack([
                env.scene[name].data.body_link_quat_w[:, ids]
                for name, ids in zip(self.asset_names, self.flap_ids, strict=True)
            ], dim=1)
            physical_flaps = torch.cat((
                position + quat_apply(quaternion, self.flap_centers[None].expand_as(position)),
                quaternion), dim=-1)
        return simulated_perception_frame(
            active=env._multi_box_active,
            box_type_id=env._multi_box_box_type_ids,
            rack_region_id=env._multi_box_region_ids,
            pool_id=env._multi_box_pool_ids,
            physical_box_poses_world=physical_poses,
            rack_pose_world=env.scene["rack"].data.root_pose_w,
            conveyor_pose_world=env.scene["conveyor_surface"].data.root_pose_w,
            physical_flap_center_poses_world=physical_flaps,
        )

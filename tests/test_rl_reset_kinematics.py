"""Reset FK must repair descendants without changing physical/control state."""

from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics import (
    refresh_teleported_articulations,
)


def test_gpu_partial_reset_publishes_links_and_preserves_other_environments():
    class Asset:
        num_joints = 2

        def __init__(self):
            self.data = SimpleNamespace(joint_pos=torch.zeros(3, 2),
                _body_link_pose_w=SimpleNamespace(timestamp=100.))
            self.root = torch.tensor([10., 20., 30.])
            self.children = torch.tensor([0., 20., 0.])
            self.dirty = torch.zeros(3, dtype=torch.bool)
            self.velocity = torch.tensor([1., 2., 3.])
            self.drive = torch.tensor([4., 5., 6.])

        def write_joint_position_to_sim(self, q, env_ids):
            self.dirty[env_ids] |= (q != self.data.joint_pos[env_ids]).any(-1)
            self.data.joint_pos[env_ids] = q

    assets = [Asset(), Asset()]

    def fk():
        for asset in assets:
            asset.children[asset.dirty] = asset.root[asset.dirty]
            asset.dirty.zero_()

    env = SimpleNamespace(device='cuda:0',
                          sim=SimpleNamespace(physics_sim_view=SimpleNamespace(
                              update_articulations_kinematic=fk)))
    assert refresh_teleported_articulations(env, assets, torch.tensor([0]))
    for asset in assets:
        assert asset.children.tolist() == [10., 20., 0.]
        assert torch.equal(asset.data.joint_pos, torch.zeros(3, 2))
        assert asset.velocity.tolist() == [1., 2., 3.]
        assert asset.drive.tolist() == [4., 5., 6.]
        assert asset.data._body_link_pose_w.timestamp == -1


def test_cpu_and_empty_reset_need_no_physics_view():
    assert not refresh_teleported_articulations(
        SimpleNamespace(device='cpu'), [object()], torch.tensor([0]))
    assert not refresh_teleported_articulations(
        SimpleNamespace(device='cuda:0'), [object()], torch.empty(0, dtype=torch.long))

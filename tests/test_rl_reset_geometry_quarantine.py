"""A corrupt surrounding box cannot crash the other original reset cases."""
from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_settling import IsaacResetSettling
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def test_original_layout_geometry_marks_corrupt_quaternions_failed_without_repairing_physics():
    spec=MultiBoxSpec()
    rack=torch.zeros(4,7);rack[:,3]=1
    env=SimpleNamespace(num_envs=4,device='cpu',common_step_counter=0,
        cfg=SimpleNamespace(multi_box=spec),scene={'rack':SimpleNamespace(data=SimpleNamespace(root_pose_w=rack))})
    adapter=IsaacResetSettling(env)
    position,quaternion=logical_cells(spec)[0].local_pose('small')
    poses=torch.tensor([(*position,*quaternion)]).repeat(4,1)
    poses[1,3]=float('nan');poses[2,3:]=0.;poses[3,3:]=1e30
    original=poses.clone()
    kinds=torch.zeros(4,dtype=torch.long);regions=kinds.clone()
    assert adapter._footprint_in_region(poses,kinds,regions).tolist()==[True,False,False,False]
    assert adapter._on_assigned_shelf(poses,kinds,regions).tolist()==[True,False,False,False]
    torch.testing.assert_close(poses,original,equal_nan=True)
    # Invalid rack geometry is also a failure, not a substituted valid scene.
    rack[0,3:]=0.
    assert not adapter._footprint_in_region(poses,kinds,regions).any()
    assert not adapter._on_assigned_shelf(poses,kinds,regions).any()

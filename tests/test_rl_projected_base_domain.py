import math
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.geometry.projected_base import (
    planar_projection_determinant, outside_projected_base_domain,
    BASE_PLANE_MIN_ABS_DETERMINANT, BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT,
)
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import matrix6


def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def test_world_up_dot_matches_actual_sixD_decoder_for_arbitrary_rotated_racks():
    generator=torch.Generator().manual_seed(81)
    root=torch.randn(200,4,generator=generator,dtype=torch.float64)
    rack=torch.randn(200,4,generator=generator,dtype=torch.float64)
    relative=quat_matrix(root).transpose(-1,-2)@quat_matrix(rack)
    measured=_rotation_matrix(torch.stack([matrix6(r) for r in relative])).transpose(-1,-2)
    expected=torch.linalg.det(measured[:,:2,:2])
    torch.testing.assert_close(planar_projection_determinant(root,rack),expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(outside_projected_base_domain(expected),
                       outside_projected_base_domain(planar_projection_determinant(root,rack)))


def test_one_finite_near_vertical_base_is_rejected_without_rejecting_other_environments():
    angles=torch.tensor([0.,math.pi/2-.01,.15],dtype=torch.float64)
    root=torch.stack((torch.cos(angles/2),torch.zeros_like(angles),torch.sin(angles/2),torch.zeros_like(angles)),-1)
    rack=torch.zeros_like(root);rack[:,0]=1
    determinant=planar_projection_determinant(root,rack)
    assert outside_projected_base_domain(determinant).tolist()==[False,True,False]
    torch.testing.assert_close(determinant,angles.cos())
    assert torch.isfinite(root).all()


def test_invalid_quaternion_does_not_silently_become_a_valid_planar_pose():
    root=torch.tensor([[0.,0.,0.,0.],[float('nan'),0.,0.,0.],[1.,0.,0.,0.]])
    rack=torch.tensor([[1.,0.,0.,0.]]).expand(3,-1)
    assert outside_projected_base_domain(planar_projection_determinant(root,rack)).tolist()==[True,True,False]
    with pytest.raises(ValueError):outside_projected_base_domain(torch.zeros(2),0.)


def test_safety_leaves_a_float32_roundoff_margin_before_decoder_singularity():
    determinant=torch.tensor([.0999,.10005,.1002,-.10005,-.1002])
    assert outside_projected_base_domain(determinant,BASE_PLANE_MIN_ABS_DETERMINANT).tolist()==[True,False,False,False,False]
    assert outside_projected_base_domain(determinant,BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT).tolist()==[True,True,False,True,False]


def test_replay_resume_drops_unsupported_old_next_state_without_inventing_terminal_labels():
    from prepare_projected_base_guard_resume import valid_replay_rows
    raw=torch.zeros(3,518);raw[:,71:77]=matrix6(torch.eye(3))
    following=raw.clone()
    following[1,71:77]=matrix6(torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]))
    rows=dict(actor_obs=raw,next_actor_obs=following,reward=torch.tensor([.1,.2,.3]),
              terminated=torch.tensor([False,False,True]))
    assert valid_replay_rows(rows,chunk=2).tolist()==[True,False,True]
    assert rows['reward'].tolist()==pytest.approx([.1,.2,.3])
    assert rows['terminated'].tolist()==[False,False,True]

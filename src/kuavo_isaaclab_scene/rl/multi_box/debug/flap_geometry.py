"""Measured flap geometry diagnostics; never consumed by the actor or reward."""
import torch

from ..geometry.grasp import estimated_flap_center_poses, nominal_flap_geometry
from ..geometry.pose import quat_apply


def compare_flap_centers(box_pose, link_poses, local_centers, normal_axes,
                         box_size, box_type, tcp_pose):
    """Compare physical panel midpoints with the actor's neutral estimates.

    Caller supplies already validated selected-body measurements. TCP distances
    remain a full hand/flap matrix, so a changed assignment cannot hide an error.
    These are diagnostics, not reconstructed old training observations.
    """
    n=len(box_pose)
    shapes=[(box_pose,(n,7)),(link_poses,(n,2,7)),(local_centers,(n,2,3)),
            (normal_axes,(n,2)),(box_size,(n,3)),(box_type,(n,)),(tcp_pose,(n,2,7))]
    if any(value.shape!=shape or not torch.isfinite(value).all() for value,shape in shapes):
        raise ValueError('Flap comparison requires finite aligned box/panel/TCP measurements')
    for quaternion in [box_pose[...,3:],link_poses[...,3:],tcp_pose[...,3:]]:
        if not torch.allclose(quaternion.norm(dim=-1),torch.ones_like(quaternion[...,0]),atol=1e-3,rtol=0):
            raise ValueError('Flap comparison requires valid unit quaternions')
    if ((normal_axes<0)|(normal_axes>2)).any():
        raise ValueError('Panel normal axes must be0,1,2')
    nominal,_=estimated_flap_center_poses(box_pose,box_size,box_type,tcp_pose)
    nominal=nominal[:,0]
    actual=link_poses.clone()
    actual[...,:3]+=quat_apply(link_poses[...,3:],local_centers)
    _,_,nominal_axes=nominal_flap_geometry(box_size,box_type)
    actual_normals=quat_apply(actual[...,3:],torch.nn.functional.one_hot(normal_axes,3).to(actual))
    nominal_normals=quat_apply(nominal[...,3:],torch.nn.functional.one_hot(nominal_axes,3).to(nominal))
    angle=torch.acos((actual_normals*nominal_normals).sum(-1).abs().clamp(0,1))
    return dict(actual_center_pose_world=actual,nominal_center_pose_world=nominal,
        center_error_m=(actual[...,:3]-nominal[...,:3]).norm(dim=-1),
        normal_error_rad=angle,
        hand_actual_center_distance_m=(tcp_pose[:,:,None,:3]-actual[:,None,:,:3]).norm(dim=-1),
        hand_nominal_center_distance_m=(tcp_pose[:,:,None,:3]-nominal[:,None,:,:3]).norm(dim=-1))

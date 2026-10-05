"""The existing absolute-goal decoder's planar coordinate domain."""
import torch

BASE_PLANE_MIN_ABS_DETERMINANT = .1
BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT = BASE_PLANE_MIN_ABS_DETERMINANT + 1e-4


def projected_base_safety_contract():
    return dict(name='measured_projected_base_domain_workspace_failure_v1',
                minimum_absolute_planar_determinant=BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT,
                decoder_minimum_absolute_planar_determinant=BASE_PLANE_MIN_ABS_DETERMINANT,
                float32_rotation_roundoff_margin=1e-4,
                threshold_source='unchanged_absolute_goal_decoder_domain',
                outcome='unsafe_workspace_limit_with_pre_reset_terminal_observation',
                reward='existing_workspace_limit_event',
                finite_failed_transition_kept_for_Q=True,partial_reset_only=True,
                observations_and_success_conditions_unchanged=True)


def planar_projection_determinant(root_quaternion, rack_quaternion):
    """det(R_root_to_rack[:2,:2]) from normalized world up vectors, wxyz."""
    if root_quaternion.shape != rack_quaternion.shape or root_quaternion.shape[-1] != 4:
        raise ValueError('Matching wxyz robot and rack quaternion batches required')
    def up(q):
        norm=q.norm(dim=-1,keepdim=True)
        valid=torch.isfinite(q).all(-1)&(norm.squeeze(-1)>1e-8)
        w,x,y,z=(q/norm.clamp_min(1e-8)).unbind(-1)
        vector=torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),-1)
        return vector,valid
    root,root_valid=up(root_quaternion);rack,rack_valid=up(rack_quaternion)
    determinant=(root*rack).sum(-1)
    return torch.where(root_valid&rack_valid,determinant,torch.full_like(determinant,float('nan')))


def outside_projected_base_domain(determinant,minimum=BASE_PLANE_MIN_ABS_DETERMINANT):
    if not 0<minimum<1:
        raise ValueError('Projection minimum must be within0..1')
    return ~torch.isfinite(determinant)|(determinant.abs()<minimum)

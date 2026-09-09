"""Keep the resting box still until a physical flap grasp lifts it clear."""

import torch


def grasp_stability_scores(delta, orientation, reference_orientation, velocity, grasped, height, spec):
    # Rest-pose error matters near the shelf, not after liftoff. Do not switch
    # it back on because one filtered contact sample vanished in mid-air.
    rest_weight = (1 - height.clamp_min(0) / spec.grasp_lift_clearance).clamp(0, 1)
    def cost(value, deadband, scale):
        return ((value - deadband).clamp_min(0) / scale).square()
    displacement = cost(delta[..., :2].norm(dim=-1), spec.prelift_position_deadband, spec.prelift_position_scale)
    horizontal_speed = cost(velocity[..., :2].norm(dim=-1), spec.prelift_speed_deadband, spec.prelift_speed_scale)
    angular_speed = cost(velocity[..., 3:].norm(dim=-1), spec.prelift_angular_deadband, spec.prelift_angular_scale)
    dot = (orientation * reference_orientation).sum(-1).abs().clamp(0, 1)
    rotation = cost(2 * torch.acos(dot), spec.prelift_rotation_deadband, spec.prelift_rotation_scale)
    ungrasped_vertical_speed = cost(velocity[..., 2].abs(), spec.prelift_speed_deadband, spec.prelift_speed_scale) * ~grasped
    # Near the shelf, displacement remains penalized even after sliding stops.
    # In the air, ungrasped motion is penalized, not displacement from the shelf.
    pose_cost = torch.stack((displacement, .5 * rotation), -1).clamp_max(4).sum(-1)
    motion_cost = torch.stack((.25 * horizontal_speed, .25 * angular_speed,
                               .25 * ungrasped_vertical_speed), -1).clamp_max(4).sum(-1)
    disturbance = pose_cost * rest_weight + motion_cost * torch.maximum(rest_weight, (~grasped).float())
    disturbance *= torch.where(grasped, spec.prelift_grasp_scale, 1.)
    disturbance = disturbance.clamp_max(spec.prelift_penalty_cap)
    stable_grasp = grasped * torch.exp(-horizontal_speed - angular_speed
                                       - displacement * rest_weight)
    return disturbance, stable_grasp

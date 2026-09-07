"""Keep the resting box still until a physical flap grasp lifts it clear."""

import torch


def grasp_stability_scores(delta, orientation, reference_orientation, velocity, grasped, height, spec):
    supported_phase = ~(grasped & (height > spec.grasp_lift_clearance))
    displacement = (delta[..., :2] / spec.prelift_position_scale).square().sum(-1)
    horizontal_speed = (velocity[..., :2] / spec.prelift_speed_scale).square().sum(-1)
    angular_speed = (velocity[..., 3:] / spec.prelift_angular_scale).square().sum(-1)
    dot = (orientation * reference_orientation).sum(-1).abs().clamp(0, 1)
    rotation = (2 * torch.acos(dot) / spec.prelift_rotation_scale).square()
    ungrasped_vertical_speed = (velocity[..., 2] / spec.prelift_speed_scale).square() * ~grasped
    # A displaced box remains penalized after it stops sliding. A genuine
    # vertical lift is allowed; pushing/tossing without a grasp never bypasses it.
    disturbance = torch.stack((displacement, .25 * horizontal_speed,
        .25 * angular_speed, .5 * rotation, .25 * ungrasped_vertical_speed), -1).clamp_max(4).sum(-1)
    disturbance *= supported_phase
    stable_grasp = grasped * torch.exp(-horizontal_speed - angular_speed
                                       - displacement * supported_phase)
    return disturbance, stable_grasp

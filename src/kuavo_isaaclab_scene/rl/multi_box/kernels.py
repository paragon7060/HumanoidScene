"""Pure tensor predicates shared by training and evaluation."""
import torch


def placement_mask(local, half, belt_half, contact, released, speed, angular, upright,
                   *, clearance, tolerance, min_force, max_speed, max_angular, min_up):
    inside = (local[..., :2].abs() + half[..., :2] <= belt_half - clearance).all(-1)
    bottom = (local[..., 2] - half[..., 2] - .015).abs() < tolerance
    supported = inside & bottom & (contact > min_force)
    # Include every box near belt height, including moving/held boxes.
    delta = (local[:, :, None, :2] - local[:, None, :, :2]).abs()
    extent = half[:, :, None, :2] + half[:, None, :, :2] + clearance
    overlap = (delta < extent).all(-1)
    overlap &= ~torch.eye(local.shape[1], device=local.device, dtype=torch.bool)[None]
    overlap &= ((local[:, :, None, 2] - local[:, None, :, 2]).abs()
                < half[:, :, None, 2] + half[:, None, :, 2])
    return (supported & released & ~overlap.any(-1) & (speed < max_speed)
            & (angular < max_angular) & (upright > min_up))


def advance_placement(valid, timer, paid, dt, duration):
    timer = torch.where(valid, timer + dt, 0.)
    complete = timer >= duration
    credit = complete & ~paid
    return timer, paid | complete, credit, complete


def potential_delta(previous, current, terminal, discount):
    # Zero absorbing-terminal potential prevents shaping credit through failure/reset.
    return discount * torch.where(terminal, 0., current) - previous

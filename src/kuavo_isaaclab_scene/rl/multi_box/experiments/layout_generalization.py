"""Reproducible layout splits and geometry retargeting of a measured path.

Only the lower-shelf small-box tube is covered by this first distribution.
Changing shelf/type is deliberately not claimed by its contract. Generated
observations describe reset poses, never synthetic Q transitions.
"""
from dataclasses import asdict, dataclass
import math

import torch

from ..demo_replay import _rotation_matrix


@dataclass(frozen=True)
class GraspLayout:
    seed: int
    split: str
    lateral_m: float
    yaw_rad: float = 0.0
    distractors: tuple[int, ...] = ()

    def validate(self):
        if self.split not in {'train', 'holdout', 'probe'}:
            raise ValueError('Unknown layout split')
        if not math.isfinite(self.lateral_m) or abs(self.lateral_m) > .08:
            raise ValueError('Lower-shelf layout lateral range is +/-8cm')
        if not math.isfinite(self.yaw_rad) or abs(self.yaw_rad) > math.radians(3):
            raise ValueError('Layout yaw range is +/-3degrees')
        if len(set(self.distractors)) != len(self.distractors) or any(i not in (5, 6, 9) for i in self.distractors):
            raise ValueError('Distractors must use separate rear/upper cells 5,6,9')
        return self

    def record(self):
        return asdict(self)


def sample_layout(seed, split):
    """Separate namespaces ensure training never consumes held-out seeds."""
    if split not in {'train', 'holdout'} or seed < 0:
        raise ValueError('Expected a nonnegative train/holdout seed')
    generator = torch.Generator().manual_seed(seed + (0 if split == 'train' else 10_000_000))
    # A 26.6cm-wide box must stay wholly inside its assigned half-shelf.
    # The measured seed has about 4.7cm inward clearance before yaw. Sampling
    # 2..4cm leaves clearance rather than triggering an invalid-reset respawn.
    lateral = -float(.02+torch.rand((), generator=generator)*.02)
    yaw = float((torch.rand((), generator=generator)*2-1)*math.radians(1))
    count = int(torch.randint(0, 4, (), generator=generator))
    candidates = torch.tensor([5, 6, 9])[torch.randperm(3, generator=generator)]
    return GraspLayout(seed, split, lateral, yaw, tuple(sorted(candidates[:count].tolist())))


def yaw_matrix(angle, like):
    c, s = math.cos(angle), math.sin(angle)
    return like.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix6(rotation):
    return rotation[..., :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def validate_layout_footprints(actor, *, margin_m=.002):
    """Reject invalid box poses before starting physics, using reset bounds.

    This is a spawn check, not a replacement for measured collision or reset
    termination. All active boxes must fit their assigned semantic half-shelf.
    """
    from ....workcell.workcell_layout import (
        scale, RACK_SHELF_CENTER_LOCAL_X_RAW, RACK_SHELF_WIDTH_RAW, RACK_RAW_BOUNDS_M,
    )
    from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
    sx, sy, _ = scale('rack')
    center = RACK_SHELF_CENTER_LOCAL_X_RAW*sx
    half_width = RACK_SHELF_WIDTH_RAW*sx/2
    rack_rotation = _rotation_matrix(actor[71:77])
    tokens = actor[86:350].reshape(12,22)
    for logical in torch.where(tokens[:,0]>.5)[0].tolist():
        token = tokens[logical]
        name = ('small','medium')[int(token[3:5].argmax())]
        hx, hy = (v/2 for v in BOX_DIMENSIONS_M[name][:2])
        corners = actor.new_tensor([[-hx,-hy,0],[-hx,hy,0],[hx,-hy,0],[hx,hy,0]])
        local = (token[12:15]-actor[68:71]+corners@_rotation_matrix(token[15:21]).T)@rack_rotation
        right = int(token[8:12].argmax())%2==0
        lower, upper = (center-half_width,center) if right else (center,center+half_width)
        if (not torch.isfinite(local).all() or
            local[:,0].min()<lower+margin_m or local[:,0].max()>upper-margin_m or
            local[:,1].min()<RACK_RAW_BOUNDS_M[0][1]*sy+margin_m or
            local[:,1].max()>RACK_RAW_BOUNDS_M[1][1]*sy-margin_m):
            raise ValueError(f'Layout box{logical} footprint exceeds its assigned rack region: '
                             f'x=[{float(local[:,0].min()):.4f},{float(local[:,0].max()):.4f}], '
                             f'allowed=[{lower+margin_m:.4f},{upper-margin_m:.4f}]')


def layout_reset_observation(source, layout, spec, *, roller_clearance_m=0.0):
    """Retain the robot's initial state; move the box relative to the rack."""
    from ..scene.spawn import logical_cells
    from ....workcell.workcell_layout import scale
    layout.validate()
    actor = source.clone()
    tokens = actor[86:350].reshape(12, 22)
    active = torch.where(tokens[:, 0] > .5)[0]
    if active.tolist() != [4] or tokens[4, 3:5].argmax() != 0:
        raise ValueError('This distribution requires the measured shelf-2-left small-box seed')
    rack_rotation = _rotation_matrix(actor[71:77])
    delta = actor.new_tensor([layout.lateral_m, 0, 0])
    tokens[4, 12:15] += rack_rotation @ delta
    original = _rotation_matrix(tokens[4, 15:21])
    tokens[4, 15:21] = matrix6(rack_rotation @ yaw_matrix(layout.yaw_rad, actor) @ rack_rotation.T @ original)
    cells = logical_cells(spec)
    for logical in layout.distractors:
        cell = cells[logical]
        # Rear cell stays behind the target; upper cells are separate shelves.
        position, quaternion = cell.local_pose('small', scale('rack'), roller_clearance_m)
        from ..geometry.pose import quaternion_to_rotation_6d
        token = tokens[logical]
        token.zero_(); token[0] = 1; token[1] = 1; token[3] = 1
        token[5:8] = tokens[4, 5:8]
        token[8+cell.region_id] = 1
        token[12:15] = actor[68:71] + rack_rotation @ actor.new_tensor(position)
        token[15:21] = matrix6(rack_rotation @ _rotation_matrix(quaternion_to_rotation_6d(actor.new_tensor(quaternion))))
        token[21] = 1
    validate_layout_footprints(actor)
    return actor


def retarget_reference_rack(reference, source_actor, current_actor):
    """Move a whole measured robot path with the perceived target in rack space.

    The actual robot reset is unchanged. Base feedback must execute the offset.
    Only target geometry is consulted; simulator contact truth is unnecessary.
    """
    from .kinematic_exploration import target_token
    original, ok0 = target_token(source_actor[None])
    actual, ok1 = target_token(current_actor[None])
    if not bool(ok0.all() and ok1.all()):
        raise ValueError('Retargeting requires a valid selected target')
    r0 = _rotation_matrix(source_actor[71:77])
    r1 = _rotation_matrix(current_actor[71:77])
    old = r0.T @ (original[0, 12:15]-source_actor[68:71])
    new = r1.T @ (actual[0, 12:15]-current_actor[68:71])
    box0 = r0.T @ _rotation_matrix(original[0, 15:21])
    box1 = r1.T @ _rotation_matrix(actual[0, 15:21])
    rotation = box1 @ box0.T
    # Settling can change pitch slightly. Retarget planar yaw, not gravity tilt.
    angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    rotation = yaw_matrix(angle, reference)
    old = old.clone(); new = new.clone(); old[2] = new[2] = 0
    result = reference.clone()
    racks = _rotation_matrix(reference[:, 71:77])
    result[:, 68:71] += (racks @ (old-rotation.T @ new)[None, :, None]).squeeze(-1)
    result[:, 71:77] = matrix6(racks @ rotation.T)
    return result, dict(target_shift_rack_m=(new-old).tolist(), target_yaw_rad=angle)

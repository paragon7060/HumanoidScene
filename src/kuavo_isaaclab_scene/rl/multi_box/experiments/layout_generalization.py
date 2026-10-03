"""Reproducible layout splits and geometry retargeting of a measured path.

The default sampler covers the lower-shelf small-box tube. Explicit layouts
can move an already measured upper target without changing its shelf or type;
an explicit region can move it to the other half of the same shelf. All
active footprints must still fit. Generated observations describe reset
poses, never synthetic Q transitions or learned generalization evidence.
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
    depth_m: float = 0.0
    base_lateral_m: float = 0.0
    base_outward_m: float = 0.0
    base_yaw_rad: float = 0.0
    target_region: str | None = None
    align_initial_base_to_region: bool = False

    def validate(self):
        if self.split not in {'train', 'holdout', 'probe'}:
            raise ValueError('Unknown layout split')
        if not math.isfinite(self.lateral_m) or abs(self.lateral_m) > .08:
            raise ValueError('Measured-target layout lateral range is +/-8cm')
        if not math.isfinite(self.yaw_rad) or abs(self.yaw_rad) > math.radians(3):
            raise ValueError('Layout yaw range is +/-3degrees')
        if not math.isfinite(self.depth_m) or abs(self.depth_m) > .02:
            raise ValueError('Layout depth range is +/-2cm')
        if any(not math.isfinite(v) or abs(v)>.25 for v in (self.base_lateral_m,self.base_outward_m)):
            raise ValueError('Initial base translation must be within +/-25cm in rack coordinates')
        if not math.isfinite(self.base_yaw_rad) or abs(self.base_yaw_rad)>math.radians(15):
            raise ValueError('Initial base yaw must be within +/-15degrees')
        if len(set(self.distractors)) != len(self.distractors) or any(i not in (5, 6, 9) for i in self.distractors):
            raise ValueError('Distractors must use separate rear/upper cells 5,6,9')
        if self.target_region is not None and self.target_region not in (
                'shelf_2_right', 'shelf_2_left', 'shelf_3_right', 'shelf_3_left'):
            raise ValueError('Unknown target rack region')
        if type(self.align_initial_base_to_region) is not bool:
            raise ValueError('Initial base region alignment must be an explicit boolean')
        if self.align_initial_base_to_region and self.target_region is None:
            raise ValueError('Initial base region alignment requires an explicit target region')
        return self

    def record(self):
        record=asdict(self)
        if self.target_region is None:
            record.pop('target_region')
            record.pop('align_initial_base_to_region')
        return record


def sample_layout(seed, split, *, depth_limit_m=0.0):
    """Separate namespaces ensure training never consumes held-out seeds."""
    if split not in {'train', 'holdout'} or seed < 0:
        raise ValueError('Expected a nonnegative train/holdout seed')
    if not math.isfinite(depth_limit_m) or not 0<=depth_limit_m<=.02:
        raise ValueError('Depth sampling limit must be within0..2cm')
    generator = torch.Generator().manual_seed(seed + (0 if split == 'train' else 10_000_000))
    # A 26.6cm-wide box must stay wholly inside its assigned half-shelf.
    # The measured seed has about 4.7cm inward clearance before yaw. Sampling
    # 2..4cm leaves clearance rather than triggering an invalid-reset respawn.
    lateral = -float(.02+torch.rand((), generator=generator)*.02)
    yaw = float((torch.rand((), generator=generator)*2-1)*math.radians(1))
    count = int(torch.randint(0, 4, (), generator=generator))
    candidates = torch.tensor([5, 6, 9])[torch.randperm(3, generator=generator)]
    # Draw after the existing fields to preserve all prior zero-depth layouts.
    depth = float((torch.rand((),generator=generator)*2-1)*depth_limit_m) if depth_limit_m else 0.
    return GraspLayout(seed, split, lateral, yaw, tuple(sorted(candidates[:count].tolist())),depth)


def yaw_matrix(angle, like):
    c, s = math.cos(angle), math.sin(angle)
    return like.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def base_reset_observation(source, lateral_m=0., outward_m=0., yaw_rad=0.):
    """Move only the robot start; preserve world rack/box poses exactly.

    Translation uses rack X and outward +Y. Object perception must be
    re-expressed in the new robot frame, not translated with the robot.
    The inferred-scene restore reconstructs the new root from this rack pose.
    """
    GraspLayout(0,'probe',0.,base_lateral_m=lateral_m,
                base_outward_m=outward_m,base_yaw_rad=yaw_rad).validate()
    return _reexpress_initial_base(source,lateral_m,outward_m,yaw_rad)


def _reexpress_initial_base(source,lateral_m=0.,outward_m=0.,yaw_rad=0.):
    """Apply a validated random offset or an explicit nominal region translation."""
    actor=source.clone()
    if not (lateral_m or outward_m or yaw_rad):return actor
    rack_rotation=_rotation_matrix(source[71:77])
    displacement=rack_rotation@source.new_tensor([lateral_m,outward_m,0.])
    rotate=yaw_matrix(-yaw_rad,source)
    for start in (68,77):  # Rack and conveyor poses, each xyz + rotation6D.
        actor[start:start+3]=rotate@(source[start:start+3]-displacement)
        actor[start+3:start+9]=matrix6(rotate@_rotation_matrix(source[start+3:start+9]))
    original=source[86:350].reshape(12,22)
    tokens=actor[86:350].reshape(12,22)
    for logical in torch.where(original[:,0]>.5)[0].tolist():
        tokens[logical,12:15]=rotate@(original[logical,12:15]-displacement)
        tokens[logical,15:21]=matrix6(rotate@_rotation_matrix(original[logical,15:21]))
    return actor


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
    """Describe a reset relative to its source, with the requested background.

    A native success may already include several boxes. Its selected target
    is the explicit one-hot, not the count of active objects. This function
    describes only a new reset; measured source transitions remain untouched.
    """
    from ..scene.spawn import logical_cells
    from ....workcell.workcell_layout import scale, RACK_SHELF_CENTER_LOCAL_X_RAW
    from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
    layout.validate()
    if source.shape!=(464,) or not bool(torch.isfinite(source).all()):
        raise ValueError('A layout seed needs one finite464-D measured observation')
    actor = source.clone()
    tokens = actor[86:350].reshape(12, 22)
    selected=source[400:412]
    ids=torch.where(selected>.5)[0]
    if len(ids)!=1:
        raise ValueError('A layout seed must have exactly one selected measured target')
    target=int(ids[0])
    expected=selected.new_zeros(12);expected[target]=1
    if not torch.equal(selected,expected):
        raise ValueError('A layout seed requires an explicit one-hot selected target')
    if not bool(tokens[target,0]>.5 and actor[388+target]>.5):
        raise ValueError('The selected layout target must be an active measured box')
    cells = logical_cells(spec)
    rack_rotation = _rotation_matrix(actor[71:77])
    nominal_base_shift=0.
    if layout.target_region is not None:
        original_cell=cells[target]
        region=next(r for r in spec.rack_regions if r.name==layout.target_region)
        if region.shelf!=original_cell.shelf:
            raise ValueError('Target region remapping must preserve the measured shelf')
        original_token=tokens[target].clone()
        region_onehot=original_token.new_zeros(4);region_onehot[original_cell.region_id]=1
        if not torch.equal(original_token[8:12],region_onehot):
            raise ValueError('Measured target region must match its logical cell')
        kind=('small','medium')[int(original_token[3:5].argmax())]
        if kind not in region.allowed_box_types:
            raise ValueError('Measured box type is not allowed in the requested region')
        destination=next(cell for cell in cells if cell.region_name==region.name
                         and cell.depth_index==original_cell.depth_index)
        if destination.logical_id!=target:
            # Translate the physical box across the actual asymmetric rack
            # centre. Preserve orientation, depth, height and dimensions; this
            # creates a reset only, never reflected demonstration transitions.
            local=rack_rotation.T@(original_token[12:15]-actor[68:71])
            center=RACK_SHELF_CENTER_LOCAL_X_RAW*scale('rack')[0]
            nominal_base_shift=float(2*(center-local[0]))
            original_token[12:15]+=rack_rotation@actor.new_tensor([nominal_base_shift,0.,0.])
            original_token[8:12]=0;original_token[8+destination.region_id]=1
            tokens[target]=0
            target=destination.logical_id
            tokens[target]=original_token
            actor[400:412]=0;actor[400+target]=1
    if target in layout.distractors:
        raise ValueError('A distractor cannot replace the selected target')
    background=torch.arange(12,device=actor.device)!=target
    tokens[background]=0
    # Regional recipes use the same negative inward2..4cm sampler on both
    # halves. Legacy recipes retain the original signed rack-X displacement.
    direction=-1 if layout.target_region is not None and cells[target].side=='right' else 1
    delta = actor.new_tensor([direction*layout.lateral_m, layout.depth_m, 0])
    tokens[target, 12:15] += rack_rotation @ delta
    original = _rotation_matrix(tokens[target, 15:21])
    tokens[target, 15:21] = matrix6(rack_rotation @ yaw_matrix(layout.yaw_rad, actor) @ rack_rotation.T @ original)
    for logical in layout.distractors:
        cell = cells[logical]
        # Rear cell stays behind the target; upper cells are separate shelves.
        position, quaternion = cell.local_pose('small', scale('rack'), roller_clearance_m)
        from ..geometry.pose import quaternion_to_rotation_6d
        token = tokens[logical]
        token.zero_(); token[0] = 1; token[1] = 1; token[3] = 1
        token[5:8] = actor.new_tensor(BOX_DIMENSIONS_M['small'])
        token[8+cell.region_id] = 1
        token[12:15] = actor[68:71] + rack_rotation @ actor.new_tensor(position)
        token[15:21] = matrix6(rack_rotation @ _rotation_matrix(quaternion_to_rotation_6d(actor.new_tensor(quaternion))))
        token[21] = 1
    actor[388:400]=tokens[:,0]
    validate_layout_footprints(actor)
    if layout.align_initial_base_to_region:
        # A declared region-specific initial template, not a movement or a
        # successful contact state. Random XY/yaw still applies afterwards.
        actor=_reexpress_initial_base(actor,nominal_base_shift)
    return base_reset_observation(actor,layout.base_lateral_m,layout.base_outward_m,layout.base_yaw_rad)


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

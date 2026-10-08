"""S63 closed-TCP FK/Jacobians in the measured root frame; no Isaac imports."""
import xml.etree.ElementTree as ET

import numpy as np
import torch

from ....robots.robot_model import resolve_robot_model
from ....robots.end_effector import calibration_definition
from ....teleop.urdf_arm_ik import UrdfArm, axis_rotation
from ..state.schema import ACTUATED_BODY_JOINTS


def rotation(axis, angles):
    x, y, z = axis.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=angles.dtype, device=angles.device)
    return (eye + angles.sin()[:, None, None] * skew
            + (1 - angles.cos())[:, None, None] * (skew @ skew))


class TensorArmKinematics:
    """Known URDF joints, including measured torso motion and calibrated tool."""
    def __init__(self, *, device='cpu', dtype=torch.float32):
        model = resolve_robot_model('s63', 'leju-twofinger')
        definition = calibration_definition(model)
        self.device, self.dtype = torch.device(device), dtype
        self.arms = [UrdfArm(model.urdf_path, side) for side in ('left', 'right')]
        self.tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=dtype)
        for arm in self.arms:
            arm.set_tool_offset(definition['closed_offsets'][arm.side])
        tree = ET.parse(model.urdf_path).getroot()
        by_child = {j.find('child').get('link'): j for j in tree.findall('joint')}
        roots = {l.get('name') for l in tree.findall('link')} - set(by_child)
        if len(roots) != 1 or self.arms[0].parent != self.arms[1].parent:
            raise ValueError('Expected the packaged S63 shared arm-parent frame')
        self.root = roots.pop()
        chain, link = [], self.arms[0].parent
        while link != self.root:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.parent_chain = []
        for joint in reversed(chain):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            R = axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0])
            kind = joint.get('type')
            column = None if kind == 'fixed' else ACTUATED_BODY_JOINTS.index(joint.get('name'))
            axis = np.zeros(3) if kind == 'fixed' else np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            if kind not in ('fixed', 'revolute', 'prismatic'):
                raise ValueError('Unsupported torso joint')
            self.parent_chain.append((self.tensor(xyz), self.tensor(R), kind, column, self.tensor(axis)))
        self.columns = [[ACTUATED_BODY_JOINTS.index(name) for name in a.names] for a in self.arms]
        self.lower = torch.stack([self.tensor(a.lower) for a in self.arms])
        self.upper = torch.stack([self.tensor(a.upper) for a in self.arms])
        self.chains = [[(self.tensor(j.xyz), self.tensor(j.rotation),
                        None if j.axis is None else self.tensor(j.axis)) for j in a.joints] for a in self.arms]
        self.offsets = [self.tensor(a.tool_offset) for a in self.arms]

    def fk(self, q):
        if q.ndim != 2 or q.shape[1] != 20 or q.device != self.device or q.dtype != self.dtype or not torch.isfinite(q).all():
            raise ValueError('Expected finite measured20-D joints on the model device')
        n = len(q)
        p = q.new_zeros(n, 3)
        R = torch.eye(3, device=q.device, dtype=q.dtype).expand(n, 3, 3).clone()
        for xyz, origin_R, kind, column, axis in self.parent_chain:
            p = p + (R @ xyz[:, None]).squeeze(-1)
            R = R @ origin_R
            if kind == 'revolute':
                R = R @ rotation(axis, q[:, column])
            elif kind == 'prismatic':
                p = p + (R @ axis[:, None]).squeeze(-1) * q[:, column, None]
        positions, rotations, jacobians = [], [], []
        for columns, chain, offset in zip(self.columns, self.chains, self.offsets):
            tip, orientation = p.clone(), R.clone()
            axes, origins = [], []
            for xyz, origin_R, axis in chain:
                tip = tip + (orientation @ xyz[:, None]).squeeze(-1)
                orientation = orientation @ origin_R
                if axis is not None:
                    origins.append(tip)
                    axes.append((orientation @ axis[:, None]).squeeze(-1))
                    orientation = orientation @ rotation(axis, q[:, columns[len(axes) - 1]])
            tip = tip + (orientation @ offset[:, None]).squeeze(-1)
            axes, origins = torch.stack(axes, -2), torch.stack(origins, -2)
            J = torch.cat((torch.linalg.cross(axes, tip[:, None] - origins).transpose(-1, -2),
                           axes.transpose(-1, -2)), -2)
            positions.append(tip); rotations.append(orientation); jacobians.append(J)
        return torch.stack(positions, 1), torch.stack(rotations, 1), torch.stack(jacobians, 1)

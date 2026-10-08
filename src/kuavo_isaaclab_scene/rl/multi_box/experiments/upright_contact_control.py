"""Measured upright IK for TRAIN stage entry; closing holds the torso goal.

Only measured deployable state and known URDF geometry are used. This local
proposal is not contact/success evidence and never runs in greedy evaluation.
"""
import xml.etree.ElementTree as ET

import torch

from ....robots.robot_model import resolve_robot_model
from ..geometry.upright_torso import planar_position, upright_joint_step, torso_links_from_urdf


class UprightContactControl:
    def __init__(self, num_envs, raw):
        model = resolve_robot_model('s63', 'leju-twofinger')
        joints = {j.attrib['name']: j for j in ET.parse(model.urdf_path).findall('joint')}
        names = ('knee_joint', 'leg_joint', 'waist_pitch_joint')
        self.limits = raw.new_tensor([[float(joints[name].find('limit').attrib[k])
                                      for k in ('lower', 'upper')] for name in names])
        self.links = raw.new_tensor(torso_links_from_urdf(model.urdf_path))
        self.origin = raw.new_zeros(num_envs, 2)
        self.target = raw.new_zeros(num_envs, 2)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
        self.projected_proposal_rejections = 0

    def reset(self, ids):
        self.initialized[ids] = False

    def begin(self, ids, raw, pilot):
        current = planar_position(raw[:, :2], self.links)
        self.origin[ids] = current
        low = pilot.center[17:19] + pilot.scale[17:19] * (pilot.agent.upright_support_anchor - pilot.agent.upright_support_scale)
        high = pilot.center[17:19] + pilot.scale[17:19] * (pilot.agent.upright_support_anchor + pilot.agent.upright_support_scale)
        self.target[ids] = current.clamp(low, high)
        self.initialized[ids] = True

    def solve(self, explorer, pilot, raw, ids, guided, displacement, angular, J, p, projector):
        n = len(raw)
        stage = guided & (explorer.phase[ids] == 0)
        # Arms alone keep the previously tested closing feedback. Body changes
        # during stage entry only, then its last absolute goal is held.
        weight = torch.where(stage, .1, .04).to(raw)[:, None, None, None]
        arm_J = torch.cat((J[..., :3, :], weight * (projector @ J[..., 3:, :])), -2)
        error = torch.cat((displacement, weight.squeeze(-1) * angular), -1)
        arms = torch.linalg.solve(arm_J.transpose(-1, -2) @ arm_J +
            .0025 * torch.eye(7, device=raw.device, dtype=raw.dtype),
            (arm_J.transpose(-1, -2) @ error[..., None])).squeeze(-1)
        if stage.any():
            rows = stage.nonzero().flatten()
            q = raw[rows, :20]
            current = planar_position(q[:, :2], self.links)
            pitch = q[:, :3].sum(-1)
            limits = self.limits[None].expand(len(rows), -1, -1)
            task = raw.new_zeros(len(rows), 12, 16)
            task[:, :6, 2:9] = arm_J[rows, 0]
            task[:, 6:, 9:] = arm_J[rows, 1]
            for col in range(2):
                probe = current.clone(); probe[:, col] += .0001
                torso, _ = upright_joint_step(q[:, :3], probe, pitch, self.links, limits)
                perturbed = q.clone(); perturbed[:, :3] = torso
                perturbed_p, _, _ = explorer.kinematics.fk(perturbed)
                derivative = (perturbed_p - p[rows]) / .0001
                task[:, :3, col] = derivative[:, 0]
                task[:, 6:9, col] = derivative[:, 1]
            damping = torch.diag(raw.new_tensor([.2, .2] + [.0025] * 14))
            delta = torch.linalg.solve(task.transpose(-1, -2) @ task + damping,
                task.transpose(-1, -2) @ error[rows].reshape(len(rows), 12, 1)).squeeze(-1)
            support_anchor = pilot.agent.upright_support_anchor
            support_scale = pilot.agent.upright_support_scale
            low = pilot.center[17:19] + pilot.scale[17:19] * (support_anchor - support_scale)
            high = pilot.center[17:19] + pilot.scale[17:19] * (support_anchor + support_scale)
            lower = torch.maximum(self.origin[ids[rows]] - raw.new_tensor([.03, .05]), low)
            upper = torch.minimum(self.origin[ids[rows]] + raw.new_tensor([.03, .05]), high)
            if not (lower <= upper).all():
                raise ValueError('Measured contact handoff is outside attainable upright support')
            desired = torch.maximum(lower, torch.minimum(current + delta[:, :2].clamp(-.001, .001), upper))
            torso, achieved = upright_joint_step(q[:, :3], desired, pitch, self.links, limits)
            if not torch.isfinite(torso).all() or not torch.isfinite(achieved).all() \
                    or (torso.sum(-1) - pitch).abs().max() > 1e-5:
                raise ValueError('Upright IK violated finite fixed-pitch kinematics')
            # Measured physics can temporarily move outside the goal support.
            # The speed-limited IK then remains outside it even for a bounded
            # request. Keep the last valid torso goal so the existing controller
            # can recover, and use the arm-only solution for those rows. Never
            # store the out-of-support achieved point or cancel other envs.
            accepted = ((achieved >= lower) & (achieved <= upper)).all(-1)
            self.projected_proposal_rejections += int((~accepted).sum())
            self.target[ids[rows[accepted]]] = achieved[accepted]
            arms[rows[accepted]] = delta[accepted, 2:].reshape(-1, 2, 7)
        if not self.initialized[ids[guided]].all():
            raise ValueError('Upright contact goals lack the measured handoff origin')
        # Even a handoff already at the front stage holds measured torso X/Z,
        # rather than continuing the time-driven source rise while closing.
        normalized = (self.target[ids] - pilot.center[17:19]) / pilot.scale[17:19]
        return arms.clamp(-.02, .02), normalized

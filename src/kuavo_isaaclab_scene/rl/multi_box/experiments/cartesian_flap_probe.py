"""Frozen full TRAIN contact diagnosis; changed actions never count as SAC."""
from copy import deepcopy

import torch

from .tensor_arm_kinematics import TensorArmKinematics
from .staged_goal_sac import held_goal_coordinates
from ..demo_replay import _rotation_matrix
from ....robots.end_effector import closed_closing_axes
from ....workcell.workcell_layout import RACK_RAW_BOUNDS_M, scale as workcell_scale
from ..metrics.potentials import FRONT_STAGE_CLEARANCE_M


def cartesian_flap_probe_contract():
    return dict(name='frozen_actual_flap_cartesian_contact_diagnostic_v1',
        scope='original_full_TRAIN128_workplace_candidates8',
        handoff_both_midpoint_distance_m=.22, arm_joint_step_cap_rad=.02,
        orientation='calibrated_closing_axis_to_perceived_panel_normal',
        goal='perceived_assigned_panel_midpoint', body_correction_radius_unchanged=True,
        real_opposing_pinch_ticks_before_lift=8, lift_rack_z_m=.025,
        privileged_pinch_used_for_teacher_hold_and_lift=True,
        original_12cm_jaw_gate_controller_randomization_success_safety_preserved=True,
        standalone_SAC=False, Q_import_eligible=False)


class FrozenCartesianFlapProbe:
    def __init__(self, num_envs, raw):
        self.kinematics = TensorArmKinematics(device=raw.device, dtype=raw.dtype)
        axes = closed_closing_axes()
        self.axes = raw.new_tensor([axes[side] for side in ('left', 'right')])
        self.phase = torch.full((num_envs,), -1, dtype=torch.long, device=raw.device)
        self.assignment = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
        self.pinch_ticks = torch.zeros_like(self.phase)
        self.lost_pinch_ticks = torch.zeros_like(self.phase)
        self.lift_targets_rack = raw.new_zeros(num_envs, 2, 3)
        self.front_y = RACK_RAW_BOUNDS_M[1][1] * workcell_scale('rack')[1] + FRONT_STAGE_CLEARANCE_M
        self.statistics = dict(held_rows=0, handoff_episodes=0, guided_rows=0,
            guided_close_rows=0, confirmed_lift_episodes=0,
            affine_arm_goal_clamped_rows=0, measured_fk_max_position_error_m=0.)

    @torch.no_grad()
    def step(self, pilot, raw, critic, result, ids, supplemental):
        if supplemental is None or supplemental.shape != (len(raw), 38):
            raise ValueError('Actual38-D perceived flap relations required')
        physical, (actor, critic_features, original_goals) = result
        if ids is None or ids.shape != (len(raw),) or ids.dtype != torch.long or len(ids.unique()) != len(ids):
            raise ValueError('Preserve original global environment identities')
        p, R, J = self.kinematics.fk(raw[:, :20])
        tcp = raw[:, 50:68].reshape(-1, 2, 9)
        observed_R = _rotation_matrix(tcp[..., 3:])
        error = (p - tcp[..., :3]).norm(dim=-1)
        angle = torch.acos(((R.transpose(-1, -2) @ observed_R).diagonal(dim1=-2, dim2=-1).sum(-1) - 1).div(2).clamp(-1, 1))
        if error.max() > .01 or angle.max() > .03:
            raise ValueError('Measured S63 TCP does not match calibrated URDF; no guide action executed')
        self.statistics['held_rows'] += len(ids)
        self.statistics['measured_fk_max_position_error_m'] = max(self.statistics['measured_fk_max_position_error_m'], float(error.max()))
        relation = supplemental[:, :36].reshape(-1, 2, 2, 9)
        current_assignment = supplemental[:, 36:38].argmax(-1)
        rows = torch.arange(len(raw), device=raw.device)[:, None]
        hands = torch.arange(2, device=raw.device)[None]
        flaps = torch.stack((current_assignment, 1 - current_assignment), -1)
        distance = relation[rows, hands, flaps, :3].norm(dim=-1)
        handoff = (self.phase[ids] < 0) & (distance.amax(-1) <= .22) & (supplemental[:, 36:38].sum(-1) > .5)
        self.phase[ids[handoff]] = 0
        self.assignment[ids[handoff]] = current_assignment[handoff]
        self.statistics['handoff_episodes'] += int(handoff.sum())
        guided = self.phase[ids] >= 0
        if not guided.any():
            return result
        flaps = torch.stack((self.assignment[ids], 1 - self.assignment[ids]), -1)
        selected = relation[rows, hands, flaps]
        centers = tcp[..., :3] + (observed_R @ selected[..., :3, None]).squeeze(-1)
        panel_R = observed_R @ _rotation_matrix(selected[..., 3:])
        desired_axis = panel_R[..., 0]
        axis = (observed_R @ self.axes[None, ..., None]).squeeze(-1)
        dot = (axis * desired_axis).sum(-1)
        desired_axis = desired_axis * torch.where(dot < 0, -1., 1.)[..., None]
        cross = torch.linalg.cross(axis, desired_axis)
        sine = cross.norm(dim=-1)
        axis_error = torch.atan2(sine, dot.abs())
        rotation_error = cross * (axis_error / sine.clamp_min(1e-7))[..., None]
        rack = raw[:, 68:77]
        rack_R = _rotation_matrix(rack[:, 3:])
        outward = rack_R[..., 1]
        front = rack[:, :3] + outward * self.front_y
        offset = ((front[:, None] - centers) * outward[:, None]).sum(-1).clamp_min(0)
        stage = centers + offset[..., None] * outward[:, None]
        staged = (((stage - tcp[..., :3]).norm(dim=-1) <= .05) & (axis_error <= .25)).all(-1)
        self.phase[ids[(self.phase[ids] == 0) & staged]] = 1
        private = critic[:, raw.shape[1]:]
        if private.shape[1] < 41:
            raise ValueError('Declared teacher requires original physical pinch evidence')
        pinching = private[:, 35:37] > .5
        self.phase[ids[(self.phase[ids] == 0) & pinching.any(-1)]] = 1
        identity = private[:, 37:41].reshape(-1, 2, 2)
        opposing = (identity.sum(-1) > .5).all(-1) & (identity.argmax(-1)[:, 0] != identity.argmax(-1)[:, 1])
        confirmed = pinching.all(-1) & opposing & guided
        self.pinch_ticks[ids] = torch.where(confirmed, self.pinch_ticks[ids] + 1, 0)
        losing = (self.phase[ids] == 2) & ~confirmed
        self.lost_pinch_ticks[ids] = torch.where(losing, self.lost_pinch_ticks[ids] + 1, 0)
        self.phase[ids[self.lost_pinch_ticks[ids] >= 3]] = 1
        lift = (self.phase[ids] == 1) & (self.pinch_ticks[ids] >= 8)
        inverse = rack_R.transpose(-1, -2)
        rack_tcp = (inverse[:, None] @ (tcp[..., :3] - rack[:, None, :3])[..., None]).squeeze(-1)
        self.lift_targets_rack[ids[lift]] = rack_tcp[lift]
        self.lift_targets_rack[ids[lift], :, 2] += .025
        self.phase[ids[lift]] = 2
        self.statistics['confirmed_lift_episodes'] += int(lift.sum())
        target = torch.where((self.phase[ids] == 0)[:, None, None], stage, centers)
        target = torch.where((pinching & (self.phase[ids] < 2)[:, None])[..., None], tcp[..., :3], target)
        lifted = rack[:, None, :3] + (rack_R[:, None] @ self.lift_targets_rack[ids, ..., None]).squeeze(-1)
        target = torch.where((self.phase[ids] == 2)[:, None, None], lifted, target)
        displacement = (target - p) * (2. / 30.)
        norm = displacement.norm(dim=-1, keepdim=True)
        displacement = displacement * (.01 / norm.clamp_min(.01))
        angular = rotation_error * (1.5 / 30.)
        projector = torch.eye(3, device=raw.device, dtype=raw.dtype) - axis[..., :, None] * axis[..., None, :]
        weight = .04
        task_J = torch.cat((J[..., :3, :], weight * (projector @ J[..., 3:, :])), -2)
        task_error = torch.cat((displacement, weight * angular), -1)
        regularized = task_J.transpose(-1, -2) @ task_J + .0025 * torch.eye(7, device=raw.device, dtype=raw.dtype)
        dq = torch.linalg.solve(regularized, (task_J.transpose(-1, -2) @ task_error[..., None])).squeeze(-1).clamp(-.02, .02)
        measured_q = torch.stack([raw[:, cols] for cols in self.kinematics.columns], 1)
        requested_q = (measured_q + dq).clamp(self.kinematics.lower + .01, self.kinematics.upper - .01)
        requested = original_goals.clone()
        requested[:, 1:15] = (requested_q.reshape(-1, 14) - pilot.center[1:15]) / pilot.scale[1:15]
        anchor, radius = pilot.agent.anchor_and_scale(actor)
        bounded = torch.maximum(anchor - radius, torch.minimum(requested[:, :19], anchor + radius)).clamp(-1, 1)
        changed = (bounded[:, 1:15] - requested[:, 1:15]).abs().amax(-1) > 1e-6
        executed = original_goals.clone()
        executed[guided, 1:15] = bounded[guided, 1:15]
        ready = ((centers - tcp[..., :3]).norm(dim=-1) <= .018) & (axis_error <= .25) & (self.phase[ids] >= 1)[:, None]
        close = ready | pinching | (self.phase[ids] == 2)[:, None]
        near = pilot.agent.action_projector.entropy_mask(actor)[:, 19:21].bool()
        executed[guided, 19:21] = torch.where((close & near)[guided], 1., -1.)
        command = held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * executed, pilot.stage)
        untouched = [0, 1, 2, 3, 18, 19, 22, 23]
        if not torch.allclose(command[:, untouched], physical[:, untouched], atol=2e-5, rtol=0):
            raise ValueError('Guide changed base/torso/head or waist yaw')
        if not torch.isfinite(command).all() or not torch.isfinite(executed).all():
            raise ValueError('Nonfinite proposed contact command')
        self.statistics['guided_rows'] += int(guided.sum())
        self.statistics['guided_close_rows'] += int((guided & close.any(-1)).sum())
        self.statistics['affine_arm_goal_clamped_rows'] += int((guided & changed).sum())
        return command, (actor, critic_features, executed)


def install_frozen_cartesian_flap_probe(pilot_class, manifest_module):
    original_act, original_report = pilot_class.act, pilot_class.report
    original_manifest = manifest_module.checkpoint_manifest_fields
    contract, tracker = cartesian_flap_probe_contract(), [None]
    def act(self, raw, critic, index, **kwargs):
        if self.training or self.actor_updates or self.critic_updates or self.replay.size:
            raise ValueError('Cartesian diagnostic cannot train or consume learned Q/replay')
        result = original_act(self, raw, critic, index, **kwargs)
        if tracker[0] is None:
            tracker[0] = FrozenCartesianFlapProbe(128, raw)
        return tracker[0].step(self, raw, critic, result, kwargs.get('exploration_ids'), kwargs.get('supplemental'))
    def report(self):
        stats = {} if tracker[0] is None else tracker[0].statistics
        return original_report(self) | dict(frozen_cartesian_flap_probe=deepcopy(contract) | dict(actual_statistics=deepcopy(stats)))
    def manifest(*args, **kwargs):
        return original_manifest(*args, **kwargs) | dict(frozen_cartesian_flap_probe=deepcopy(contract))
    pilot_class.act, pilot_class.report = act, report
    manifest_module.checkpoint_manifest_fields = manifest
    def restore():
        pilot_class.act, pilot_class.report = original_act, original_report
        manifest_module.checkpoint_manifest_fields = original_manifest
    return restore

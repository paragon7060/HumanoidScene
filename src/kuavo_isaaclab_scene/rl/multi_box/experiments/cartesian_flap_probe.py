"""Frozen full TRAIN contact diagnosis; changed actions never count as SAC."""
from copy import deepcopy

import torch

from .tensor_arm_kinematics import TensorArmKinematics
from .staged_goal_sac import held_goal_coordinates
from ..demo_replay import _rotation_matrix
from ....robots.end_effector import closed_closing_axes
from ....workcell.workcell_layout import RACK_RAW_BOUNDS_M, scale as workcell_scale
from ..metrics.potentials import FRONT_STAGE_CLEARANCE_M
from ..geometry.grasp import nominal_flap_geometry
from .kinematic_exploration import target_token


def cartesian_flap_probe_contract(*, contact_region=False, arm_goal_bounds='source-affine', lift_drive='measured'):
    if arm_goal_bounds not in ('source-affine', 'urdf') or (arm_goal_bounds == 'urdf' and not contact_region):
        raise ValueError('URDF arm diagnostic requires the explicit contact-region method')
    if lift_drive not in ('measured', 'bounded-pending-target') or (
            lift_drive != 'measured' and arm_goal_bounds != 'urdf'):
        raise ValueError('Pending-target lift requires the explicit URDF diagnostic')
    contract = dict(name='frozen_actual_flap_cartesian_contact_diagnostic_v1',
        scope='original_full_TRAIN128_workplace_candidates8',
        handoff_both_midpoint_distance_m=.22, arm_joint_step_cap_rad=.02,
        orientation='calibrated_closing_axis_to_perceived_panel_normal',
        goal='perceived_assigned_panel_midpoint', body_correction_radius_unchanged=True,
        real_opposing_pinch_ticks_before_lift=8, lift_rack_z_m=.025,
        privileged_pinch_used_for_teacher_hold_and_lift=True,
        original_12cm_jaw_gate_controller_randomization_success_safety_preserved=True,
        standalone_SAC=False, Q_import_eligible=False)
    if contact_region:
        contract.update(name='frozen_actual_flap_region_contact_diagnostic_v2',
            goal='once_selected_accessible_point_inside_perceived_panel_region',
            tangent_margin_m=.005, supplemental_midpoint_observations_unchanged=True,
            original_neural_jaw_choices_preserved=True,
            orientation_held_after_real_pinch=True,
            extra_assisted_close_point_tolerance_m=.018,
            extra_assisted_close_axis_tolerance_rad=.25)
    if arm_goal_bounds == 'urdf':
        contract.update(name='frozen_actual_flap_URDF_contact_diagnostic_v3',
            body_correction_radius_unchanged=False,
            arm_goal_bounds='known_S63_URDF_margin0p01rad',
            source_affine_arm_goal_bounds_bypassed=True,
            non_arm_source_affine_goals_preserved=True,
            physical_joint_delta_and_controller_caps_unchanged=True,
            stored_teacher_goal_coordinates='source_affine_coordinates_may_exceed_normalized_bounds',
            stored_teacher_goals_NOT_SAC_actor_actions=True)
    if lift_drive == 'bounded-pending-target':
        contract.update(name='frozen_actual_flap_URDF_pending_lift_diagnostic_v4',
            lift_joint_target_rule='measured_pending_PD_target_plus_DLS_increment',
            lift_pending_target_increment_cap_rad=.02,
            lift_target_lead_over_measured_joint_limit_rad=.08,
            lift_integration_requires_current_real_opposing_bilateral_pinch=True,
            approach_and_insertion_target_rule_unchanged=True)
    return contract


def contact_region_offsets(relations, half_extents, *, margin=.005):
    """One stable panel-frame grasp point; midpoint perception stays unchanged."""
    if relations.ndim != 3 or relations.shape[1:] != (2, 9) or half_extents.shape != (*relations.shape[:2], 3):
        raise ValueError('Two assigned perceived panels and known half extents required')
    if not torch.isfinite(relations).all() or not torch.isfinite(half_extents).all() or (half_extents <= 0).any():
        raise ValueError('Finite positive panel geometry required')
    rotation = _rotation_matrix(relations[..., 3:])
    local_tcp = -(rotation.transpose(-1, -2) @ relations[..., :3, None]).squeeze(-1)
    inner = (half_extents - margin).clamp_min(0)
    point = torch.maximum(-inner, torch.minimum(local_tcp, inner))
    # The known stock panels have local-X thickness. Aim between the pads;
    # retaining the tangential contact location avoids forcing every grasp
    # to the panel midpoint (past actual successes are several cm away).
    point[..., 0] = 0
    return point


class FrozenCartesianFlapProbe:
    def __init__(self, num_envs, raw, *, contact_region=False, arm_goal_bounds='source-affine', lift_drive='measured'):
        if type(contact_region) is not bool:
            raise ValueError('Contact-region diagnostic must be explicit')
        self.contact_region = contact_region
        cartesian_flap_probe_contract(contact_region=contact_region, arm_goal_bounds=arm_goal_bounds, lift_drive=lift_drive)
        self.arm_goal_bounds = arm_goal_bounds
        self.lift_drive = lift_drive
        self.kinematics = TensorArmKinematics(device=raw.device, dtype=raw.dtype)
        axes = closed_closing_axes()
        self.axes = raw.new_tensor([axes[side] for side in ('left', 'right')])
        self.phase = torch.full((num_envs,), -1, dtype=torch.long, device=raw.device)
        self.assignment = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
        self.pinch_ticks = torch.zeros_like(self.phase)
        self.lost_pinch_ticks = torch.zeros_like(self.phase)
        self.lift_targets_rack = raw.new_zeros(num_envs, 2, 3)
        self.contact_offsets = raw.new_zeros(num_envs, 2, 3)
        self.front_y = RACK_RAW_BOUNDS_M[1][1] * workcell_scale('rack')[1] + FRONT_STAGE_CLEARANCE_M
        self.statistics = dict(held_rows=0, handoff_episodes=0, guided_rows=0,
            guided_close_rows=0, confirmed_lift_episodes=0,
            affine_arm_goal_clamped_rows=0, measured_fk_max_position_error_m=0.)
        if contact_region:
            self.statistics.update(original_neural_close_retained_rows=0,
                extra_assisted_close_rows=0)
        if arm_goal_bounds == 'urdf':
            self.statistics.update(source_affine_arm_range_bypassed_rows=0,
                URDF_arm_limit_clamped_rows=0)
        if lift_drive == 'bounded-pending-target':
            self.statistics.update(lift_pending_target_integrated_rows=0,
                lift_target_lead_clamped_rows=0)

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
        if self.contact_region and handoff.any():
            token, valid = target_token(raw)
            if not valid[handoff].all():
                raise ValueError('Contact point requires perceived selected box geometry')
            _, halves, normal_axes = nominal_flap_geometry(token[:, 5:8], token[:, 3:5].argmax(-1))
            if (normal_axes[handoff] != 0).any():
                raise ValueError('Contact-region diagnostic requires known local-X panels')
            offsets = contact_region_offsets(relation[rows, hands, flaps][handoff], halves[rows, flaps][handoff])
            self.contact_offsets[ids[handoff]] = offsets
        self.statistics['handoff_episodes'] += int(handoff.sum())
        guided = self.phase[ids] >= 0
        if not guided.any():
            return result
        flaps = torch.stack((self.assignment[ids], 1 - self.assignment[ids]), -1)
        selected = relation[rows, hands, flaps]
        centers = tcp[..., :3] + (observed_R @ selected[..., :3, None]).squeeze(-1)
        panel_R = observed_R @ _rotation_matrix(selected[..., 3:])
        goal_points = centers
        if self.contact_region:
            goal_points = centers + (panel_R @ self.contact_offsets[ids, ..., None]).squeeze(-1)
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
        offset = ((front[:, None] - goal_points) * outward[:, None]).sum(-1).clamp_min(0)
        stage = goal_points + offset[..., None] * outward[:, None]
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
        target = torch.where((self.phase[ids] == 0)[:, None, None], stage, goal_points)
        target = torch.where((pinching & (self.phase[ids] < 2)[:, None])[..., None], tcp[..., :3], target)
        lifted = rack[:, None, :3] + (rack_R[:, None] @ self.lift_targets_rack[ids, ..., None]).squeeze(-1)
        target = torch.where((self.phase[ids] == 2)[:, None, None], lifted, target)
        displacement = (target - p) * (2. / 30.)
        norm = displacement.norm(dim=-1, keepdim=True)
        displacement = displacement * (.01 / norm.clamp_min(.01))
        angular = rotation_error * (1.5 / 30.)
        if self.contact_region:
            angular = torch.where(pinching[..., None], 0., angular)
        projector = torch.eye(3, device=raw.device, dtype=raw.dtype) - axis[..., :, None] * axis[..., None, :]
        weight = .04
        task_J = torch.cat((J[..., :3, :], weight * (projector @ J[..., 3:, :])), -2)
        task_error = torch.cat((displacement, weight * angular), -1)
        regularized = task_J.transpose(-1, -2) @ task_J + .0025 * torch.eye(7, device=raw.device, dtype=raw.dtype)
        dq = torch.linalg.solve(regularized, (task_J.transpose(-1, -2) @ task_error[..., None])).squeeze(-1).clamp(-.02, .02)
        measured_q = torch.stack([raw[:, cols] for cols in self.kinematics.columns], 1)
        joint_proposal = measured_q + dq
        if self.lift_drive == 'bounded-pending-target':
            # A target reset close to measured q every tick can leave only a
            # few mrad of PD error under load. Preserve the measured pending
            # target during confirmed lift, with a bounded lead to prevent
            # unchecked accumulation when a hand is mechanically blocked.
            lift_integrating = (self.phase[ids] == 2) & confirmed
            pending = raw[:, :20] + raw[:, 416:436]
            pending_q = torch.stack([pending[:, cols] for cols in self.kinematics.columns], 1)
            integrated = pending_q + dq
            limited = torch.maximum(measured_q - .08, torch.minimum(integrated, measured_q + .08))
            self.statistics['lift_pending_target_integrated_rows'] += int(lift_integrating.sum())
            self.statistics['lift_target_lead_clamped_rows'] += int((lift_integrating &
                ((integrated - limited).abs().flatten(1).amax(-1) > 1e-6)).sum())
            joint_proposal = torch.where(lift_integrating[:, None, None], limited, joint_proposal)
        requested_q = joint_proposal.clamp(self.kinematics.lower + .01, self.kinematics.upper - .01)
        requested = original_goals.clone()
        requested[:, 1:15] = (requested_q.reshape(-1, 14) - pilot.center[1:15]) / pilot.scale[1:15]
        anchor, radius = pilot.agent.anchor_and_scale(actor)
        source_bounded = torch.maximum(anchor - radius, torch.minimum(requested[:, :19], anchor + radius)).clamp(-1, 1)
        changed = (source_bounded[:, 1:15] - requested[:, 1:15]).abs().amax(-1) > 1e-6
        bounded = source_bounded
        if self.arm_goal_bounds == 'urdf':
            # This explicit frozen teacher bypasses the learned affine arm
            # envelope only. Its stored goals still exactly decode the
            # physical command but are not bounded SAC actor actions.
            bounded = requested[:, :19]
            self.statistics['source_affine_arm_range_bypassed_rows'] += int((guided & changed).sum())
            at_limit = (joint_proposal - requested_q).abs().flatten(1).amax(-1) > 1e-6
            self.statistics['URDF_arm_limit_clamped_rows'] += int((guided & at_limit).sum())
        executed = original_goals.clone()
        executed[guided, 1:15] = bounded[guided, 1:15]
        ready = ((goal_points - tcp[..., :3]).norm(dim=-1) <= .018) & (axis_error <= .25) & (self.phase[ids] >= 1)[:, None]
        close = ready | pinching | (self.phase[ids] == 2)[:, None]
        near = pilot.agent.action_projector.entropy_mask(actor)[:, 19:21].bool()
        if self.contact_region:
            # Preserve the learned, production-projected jaws. The stricter
            # diagnostic geometry may add a close request but cannot force
            # an otherwise permitted neural close back open.
            executed[guided, 19:21] = torch.where((close & near)[guided], 1., original_goals[guided, 19:21])
            retained = (original_goals[:, 19:21] > 0) & near
            assisted = (close & near) & ~retained
            self.statistics['original_neural_close_retained_rows'] += int((guided & retained.any(-1)).sum())
            self.statistics['extra_assisted_close_rows'] += int((guided & assisted.any(-1)).sum())
        else:
            executed[guided, 19:21] = torch.where((close & near)[guided], 1., -1.)
        command = held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * executed, pilot.stage)
        untouched = [0, 1, 2, 3, 18, 19, 22, 23]
        if not torch.allclose(command[:, untouched], physical[:, untouched], atol=2e-5, rtol=0):
            raise ValueError('Guide changed base/torso/head or waist yaw')
        if not torch.isfinite(command).all() or not torch.isfinite(executed).all():
            raise ValueError('Nonfinite proposed contact command')
        self.statistics['guided_rows'] += int(guided.sum())
        requested_close = (executed[:, 19:21] > 0) & near if self.contact_region else close
        self.statistics['guided_close_rows'] += int((guided & requested_close.any(-1)).sum())
        if self.arm_goal_bounds == 'source-affine':
            self.statistics['affine_arm_goal_clamped_rows'] += int((guided & changed).sum())
        return command, (actor, critic_features, executed)


def install_frozen_cartesian_flap_probe(pilot_class, manifest_module, *, contact_region=False, arm_goal_bounds='source-affine', lift_drive='measured'):
    original_act, original_report = pilot_class.act, pilot_class.report
    original_manifest = manifest_module.checkpoint_manifest_fields
    contract, tracker = cartesian_flap_probe_contract(contact_region=contact_region, arm_goal_bounds=arm_goal_bounds, lift_drive=lift_drive), [None]
    def act(self, raw, critic, index, **kwargs):
        if self.training or self.actor_updates or self.critic_updates or self.replay.size:
            raise ValueError('Cartesian diagnostic cannot train or consume learned Q/replay')
        result = original_act(self, raw, critic, index, **kwargs)
        if tracker[0] is None:
            tracker[0] = FrozenCartesianFlapProbe(128, raw, contact_region=contact_region, arm_goal_bounds=arm_goal_bounds, lift_drive=lift_drive)
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

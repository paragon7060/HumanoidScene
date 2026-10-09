"""Perception-only coherent contact attempts in actual SAC TRAIN collection.

Selected episodes may try an accessible flap point, sustained closure and a
small lift. Neither pinch/success/contact labels nor critic inputs drive the
attempt. The unchanged environment decides whether it actually succeeds.
Greedy evaluation and actor/Q policy sampling never use this explorer.
"""
from copy import deepcopy

import torch

from .cartesian_flap_probe import contact_region_offsets
from .tensor_arm_kinematics import TensorArmKinematics
from .staged_goal_sac import held_goal_coordinates
from .kinematic_exploration import target_token
from ..demo_replay import _rotation_matrix
from ..geometry.grasp import nominal_flap_geometry
from ..metrics.potentials import FRONT_STAGE_CLEARANCE_M
from ....robots.end_effector import closed_closing_axes
from ....workcell.workcell_layout import RACK_RAW_BOUNDS_M, scale as workcell_scale


def perceived_contact_contract(*, settled_close=False, precise_feedback=False, motion_feedback=False, upright_feedback=False, interior_contact=False, predictive_feedback=False, whole_arm_clearance=False, independent_hand_close=False):
    if type(settled_close) is not bool:raise ValueError('Explicit settled-close variant required')
    if type(precise_feedback) is not bool or (precise_feedback and not settled_close):
        raise ValueError('Precise feedback requires the explicit settled-close variant')
    if type(motion_feedback) is not bool or (motion_feedback and not precise_feedback):
        raise ValueError('Motion feedback requires the explicit precise-feedback variant')
    if type(upright_feedback) is not bool or (upright_feedback and not motion_feedback):
        raise ValueError('Upright feedback requires the explicit motion-feedback variant')
    if type(interior_contact) is not bool or (interior_contact and not upright_feedback):
        raise ValueError('Interior contact requires the explicit upright-feedback variant')
    if type(predictive_feedback) is not bool or (predictive_feedback and (not upright_feedback or interior_contact)):
        raise ValueError('Predictive feedback requires upright control with the original contact target')
    if type(whole_arm_clearance) is not bool or (whole_arm_clearance and not predictive_feedback):
        raise ValueError('Whole-arm clearance requires the explicit predictive upright variant')
    if type(independent_hand_close) is not bool or (independent_hand_close and not whole_arm_clearance):
        raise ValueError('Independent hand closing requires explicit whole-arm clearance')
    result=dict(name='TRAIN_arm20_perceived_contact_attempt_v1',
        scope='selected_original20percent_actual_TRAIN_episodes_only',
        selection='existing_gentle_arm_episode_mask_no_second_draw',
        inputs='measured_joint_TCP_rack_box_geometry_perceived_flap_midpoint_and_pending_targets',
        privileged_pinch_contact_reward_or_success_inputs=False,
        handoff_both_midpoint_distance_m=.22,
        contact_point='once_selected_panel_tangent_point_margin5mm',
        front_stage_clearance_m=FRONT_STAGE_CLEARANCE_M,
        orientation='closing_line_to_perceived_panel_normal',
        joint_increment_cap_rad=.02, pending_target_lead_cap_rad=.08,
        close_point_tolerance_m=.018, close_axis_tolerance_rad=.25,
        sustained_projected_closed_command_ticks_before_attempt_lift=8,
        attempted_lift_rack_z_m=.025, max_guided_control_ticks=180,
        max_lift_attempt_control_ticks=40,
        commanded_close_NOT_confirmed_pinch_or_success=True,
        stored_actions='bounded_absolute_URDF_goals_exactly_decoding_actual_commands',
        actor_Q_targets_density_entropy_and_greedy_eval_unchanged=True,
        original_nominal_jaw_gate_PD_limits_DR_reward_success_safety_preserved=True,
        no_extra_RNG=True, curriculum=False)
    if settled_close:
        result.update(name='TRAIN_arm20_perceived_settled_contact_attempt_v2',
            close_target='once_captured_contact_point_in_measured_rack_frame',
            measured_jaw_settling_required=True,
            measured_minimum_closure_fraction=.85,
            measured_maximum_closure_change_per_tick=.005,
            consecutive_measured_settled_ticks_before_attempt_lift=6,
            sustained_projected_closed_command_ticks_before_attempt_lift=24,
            lift_point_tolerance_m=.006,
            settled_closure_NOT_confirmed_pinch_or_success=True,
            max_guided_control_ticks=360,max_lift_attempt_control_ticks=60)
    if precise_feedback:
        result.update(name='TRAIN_arm20_perceived_precise_feedback_attempt_v3',
            close_target='current_measured_flap_pose_with_once_selected_panel_tangent_point',
            close_point_tolerance_m=.006,
            closed_reacquire_maximum_point_distance_m=.012,
            closed_reacquire_maximum_axis_error_rad=.30,
            both_jaws_open_until_precise_closing_gate=True,
            orientation_feedback_until_lift=True,
            world_frame_close_latch=False)
    if motion_feedback:
        result.update(name='TRAIN_arm20_perceived_motion_feedback_attempt_v4',
            closed_position_feedback_gain_per_s=8.,
            open_and_lift_position_feedback_gain_per_s=2.,
            current_panel_motion_feedforward_fraction=.5,
            panel_motion_feedforward_cap_m_per_tick=.002,
            panel_motion_coordinates='measured_rack_frame_consecutive_held_ticks_same_phase',
            position_task_increment_cap_m_per_tick=.01,
            closing_reacquisition_and_measured_lift_gates_unchanged=True)
    if upright_feedback:
        result.update(name='TRAIN_arm20_upright_contact_attempt_v5',
            torso_goal='measured_handoff_bounded_actual_upright_IK_stage_adjustment_then_hold',
            torso_source_clock_rise_overridden_only_in_selected_guided_TRAIN=True,
            torso_stage_increment_cap_m_per_tick=.001,
            torso_local_handoff_radius_XZ_m=[.03, .05],
            torso_actual_software_and_policy_goal_support_intersected=True,
            torso_finite_difference_actual_IK_FK_Jacobian=True,
            torso_stage_regularization=.2, stage_orientation_weight_m=.1,
            closing_orientation_weight_m=.04, fixed_measured_reset_pitch=True,
            actor_Q_targets_density_entropy_and_greedy_eval_unchanged=False,
            SAC_upright_support_and_density_connected=True,
            evaluated_policy_never_uses_contact_explorer=True)
    if interior_contact:
        result.update(name='TRAIN_arm20_upright_interior_contact_attempt_v6',
            contact_point='once_selected_panel_tangent_point_margin20mm',
            contact_tangent_margin_m=.020,
            tangent_target_margin_NOT_physical_contact_or_success_relaxation=True,
            SAC_distribution_and_unassisted_eval_same_as_upright_v5=True)
    if predictive_feedback:
        result.update(name='TRAIN_arm20_upright_predictive_contact_attempt_v7',
            closed_panel_prediction_horizon_control_ticks=3,
            closed_panel_prediction_maximum_offset_m=.008,
            prediction_coordinates='consecutive_measured_rack_frame_points_same_phase',
            prediction_scope='selected_closed_surface_tracking_only',
            closing_and_lift_gates_use_current_unpredicted_perception=True,
            position_joint_and_pending_lead_limits_unchanged=True,
            no_privileged_contact_or_force_prediction=True,
            SAC_distribution_and_unassisted_eval_same_as_upright_v5=True)
    if whole_arm_clearance:
        from .rack_entry_clearance import rack_clearance_contract
        result.update(name='TRAIN_arm20_upright_whole_arm_rack_clearance_attempt_v8',
                      whole_arm_rack_clearance=rack_clearance_contract())
    if independent_hand_close:
        result.update(name='TRAIN_arm20_upright_independent_hand_close_attempt_v9',
            both_jaws_open_until_precise_closing_gate=False,
            each_jaw_open_until_its_own_precise_closing_gate=True,
            closed_reacquisition='same_12mm_and_0p30rad_checked_per_hand',
            closing_tracking_and_prediction='only_each_actually_projected_closed_hand',
            both_measured_settled_and_original_bilateral_geometry_required_for_lift=True,
            individual_closing_NOT_confirmed_pinch_or_success=True)
    return result


def bounded_panel_prediction(delta, *, horizon_ticks, maximum_offset_m):
    """Predict a short measured motion without changing actual close/lift gates."""
    if delta.ndim != 3 or delta.shape[1:] != (2, 3) or not torch.isfinite(delta).all() \
            or type(horizon_ticks) is not int or horizon_ticks < 1 \
            or type(maximum_offset_m) not in (float, int) or not 0 < maximum_offset_m <= .01:
        raise ValueError('Finite bilateral measured motion and bounded prediction required')
    offset = delta * horizon_ticks
    return offset * maximum_offset_m / offset.norm(dim=-1, keepdim=True).clamp_min(maximum_offset_m)


def contact_statistics(saved=None):
    empty=dict(handoff_episodes=0,guided_rows=0,closed_command_rows=0,
        attempted_lift_episodes=0,expired_attempts=0,
        pending_lead_clamped_rows=0,measured_fk_max_position_error_m=0.)
    if saved is None:return empty
    if not isinstance(saved,dict) or set(saved)!=set(empty):
        raise ValueError('Perceived contact collection statistics differ')
    if any(type(saved[k]) is not int or saved[k]<0 for k in empty if k!='measured_fk_max_position_error_m'):
        raise ValueError('Contact collection counts must be nonnegative integers')
    error=saved['measured_fk_max_position_error_m']
    if type(error) not in (float,int) or not 0<=error<=.01:
        raise ValueError('Contact collection FK error is invalid')
    return deepcopy(saved)


class PerceivedContactExploration:
    def __init__(self,num_envs,raw,statistics=None,*,settled_close=False,precise_feedback=False,motion_feedback=False,upright_feedback=False,interior_contact=False,predictive_feedback=False, whole_arm_clearance=False, independent_hand_close=False):
        if type(num_envs) is not int or num_envs<1:
            raise ValueError('Positive global environment count required')
        self.contract=perceived_contact_contract(settled_close=settled_close,precise_feedback=precise_feedback,motion_feedback=motion_feedback,upright_feedback=upright_feedback,interior_contact=interior_contact,predictive_feedback=predictive_feedback,whole_arm_clearance=whole_arm_clearance,independent_hand_close=independent_hand_close)
        self.settled_close=settled_close
        self.precise_feedback=precise_feedback
        self.motion_feedback=motion_feedback
        self.upright_feedback=upright_feedback
        self.predictive_feedback=predictive_feedback
        self.independent_hand_close=independent_hand_close
        self.upright_control=None
        if upright_feedback:
            from .upright_contact_control import UprightContactControl
            self.upright_control=UprightContactControl(num_envs,raw,whole_arm_clearance=whole_arm_clearance)
        self.kinematics=TensorArmKinematics(device=raw.device,dtype=raw.dtype)
        self.axes=raw.new_tensor([closed_closing_axes()[s] for s in ('left','right')])
        self.phase=torch.full((num_envs,),-1,dtype=torch.long,device=raw.device)
        self.assignment=torch.zeros_like(self.phase)
        self.closed_ticks=torch.zeros_like(self.phase)
        self.started=raw.new_full((num_envs,),-1)
        self.lift_started=raw.new_full((num_envs,),-1)
        self.last_clock=raw.new_full((num_envs,),-1)
        self.contact_offsets=raw.new_zeros(num_envs,2,3)
        self.lift_targets_rack=raw.new_zeros(num_envs,2,3)
        self.close_targets_rack=raw.new_zeros(num_envs,2,3)
        self.closing=torch.zeros(num_envs,dtype=torch.bool,device=raw.device)
        self.hand_closing=torch.zeros(num_envs,2,dtype=torch.bool,device=raw.device)
        self.individual_close_statistics=dict(projected_closed_hand_rows=[0,0],unilateral_closed_rows=0)
        self.settled_ticks=torch.zeros_like(self.phase)
        self.previous_closure=raw.new_zeros(num_envs,2)
        self.closure_initialized=torch.zeros_like(self.closing)
        self.previous_target_rack=raw.new_zeros(num_envs,2,3)
        self.previous_target_clock=raw.new_full((num_envs,),-1)
        self.previous_target_phase=torch.full_like(self.phase,-1)
        self.front_y=RACK_RAW_BOUNDS_M[1][1]*workcell_scale('rack')[1]+FRONT_STAGE_CLEARANCE_M
        self.statistics=contact_statistics(statistics)

    @torch.no_grad()
    def step(self,pilot,raw,result,ids,supplemental,chosen,clocks):
        physical,(actor,critic_features,original_goals)=result
        n=len(raw)
        if ids.shape!=(n,) or ids.dtype!=torch.long or ids.device!=raw.device or len(ids.unique())!=n \
                or (ids<0).any() or (ids>=len(self.phase)).any():
            raise ValueError('Distinct original global environment IDs required')
        if chosen.shape!=(n,) or chosen.dtype!=torch.bool or chosen.device!=raw.device:
            raise ValueError('Original episode selection mask required')
        if supplemental is None or supplemental.shape!=(n,38) or not torch.isfinite(supplemental).all():
            raise ValueError('Finite deployable flap midpoint perception required')
        clocks=torch.as_tensor(clocks,device=raw.device,dtype=raw.dtype).expand(n)
        if not torch.isfinite(clocks).all() or (clocks<0).any():
            raise ValueError('Measured nonnegative held clocks required')
        reset=clocks<self.last_clock[ids]
        if self.upright_control is not None:self.upright_control.reset(ids[reset])
        self.phase[ids[reset]]=-1;self.closed_ticks[ids[reset]]=0
        self.closing[ids[reset]]=False;self.settled_ticks[ids[reset]]=0
        self.hand_closing[ids[reset]]=False
        self.closure_initialized[ids[reset]]=False
        self.previous_target_clock[ids[reset]]=-1
        self.previous_target_phase[ids[reset]]=-1
        self.last_clock[ids]=clocks
        if not chosen.any():return result
        p,R,J=self.kinematics.fk(raw[:,:20])
        tcp=raw[:,50:68].reshape(n,2,9);observed_R=_rotation_matrix(tcp[...,3:])
        fk_error=(p-tcp[...,:3]).norm(dim=-1)
        angle=torch.acos(((R.transpose(-1,-2)@observed_R).diagonal(dim1=-2,dim2=-1).sum(-1)-1).div(2).clamp(-1,1))
        if fk_error[chosen].max()>.01 or angle[chosen].max()>.03:
            raise ValueError('Measured S63 TCP does not match calibrated URDF')
        self.statistics['measured_fk_max_position_error_m']=max(self.statistics['measured_fk_max_position_error_m'],float(fk_error[chosen].max()))
        relation=supplemental[:,:36].reshape(n,2,2,9)
        assignment=supplemental[:,36:].argmax(-1)
        rows=torch.arange(n,device=raw.device)[:,None];hands=torch.arange(2,device=raw.device)[None]
        flaps=torch.stack((assignment,1-assignment),-1)
        selected=relation[rows,hands,flaps]
        valid=supplemental[:,36:].sum(-1)>.5
        handoff=chosen&valid&(self.phase[ids]==-1)&(selected[...,:3].norm(dim=-1).amax(-1)<=.22)
        if handoff.any():
            token,token_valid=target_token(raw)
            if not token_valid[handoff].all():raise ValueError('Selected perceived box required')
            _,halves,normal_axes=nominal_flap_geometry(token[:,5:8],token[:,3:5].argmax(-1))
            if (normal_axes[handoff]!=0).any():raise ValueError('Known local-X flap panels required')
            assigned_half=halves[rows,flaps][handoff]
            margin=self.contract.get('contact_tangent_margin_m',.005)
            if self.contract.get('contact_tangent_margin_m') is not None and (assigned_half[...,1:]<=margin).any():
                raise ValueError('Flap tangents must contain the declared interior grasp margin')
            self.contact_offsets[ids[handoff]]=contact_region_offsets(selected[handoff],assigned_half,margin=margin)
            self.assignment[ids[handoff]]=assignment[handoff]
            self.started[ids[handoff]]=clocks[handoff];self.phase[ids[handoff]]=0
            if self.upright_control is not None:self.upright_control.begin(ids[handoff],raw[handoff],pilot)
            self.statistics['handoff_episodes']+=int(handoff.sum())
        expired=chosen&(self.phase[ids]>=0)&(self.phase[ids]<3)&(
            ((clocks-self.started[ids])>=self.contract['max_guided_control_ticks'])|
            ((self.phase[ids]==2)&((clocks-self.lift_started[ids])>=self.contract['max_lift_attempt_control_ticks'])))
        self.phase[ids[expired]]=3;self.statistics['expired_attempts']+=int(expired.sum())
        guided=chosen&valid&(self.phase[ids]>=0)&(self.phase[ids]<3)
        if not guided.any():return result
        flaps=torch.stack((self.assignment[ids],1-self.assignment[ids]),-1)
        selected=relation[rows,hands,flaps]
        centers=tcp[...,:3]+(observed_R@selected[...,:3,None]).squeeze(-1)
        panel_R=observed_R@_rotation_matrix(selected[...,3:])
        goal_points=centers+(panel_R@self.contact_offsets[ids,...,None]).squeeze(-1)
        axis=(observed_R@self.axes[None,...,None]).squeeze(-1)
        panel_normal=panel_R[...,0];dot=(axis*panel_normal).sum(-1)
        normal=panel_normal*torch.where(dot<0,-1.,1.)[...,None]
        cross=torch.linalg.cross(axis,normal);sine=cross.norm(dim=-1)
        axis_error=torch.atan2(sine,dot.abs())
        rotation_error=cross*(axis_error/sine.clamp_min(1e-7))[...,None]
        rack=raw[:,68:77];rack_R=_rotation_matrix(rack[:,3:]);outward=rack_R[...,1]
        front=rack[:,:3]+outward*self.front_y
        depth=((front[:,None]-goal_points)*outward[:,None]).sum(-1).clamp_min(0)
        stage=goal_points+depth[...,None]*outward[:,None]
        staged=(((stage-tcp[...,:3]).norm(dim=-1)<=.05)&(axis_error<=.25)).all(-1)
        self.phase[ids[guided&(self.phase[ids]==0)&staged]]=1
        point_distance=(goal_points-tcp[...,:3]).norm(dim=-1)
        ready=(point_distance<=self.contract['close_point_tolerance_m'])&(axis_error<=.25)
        latched=self.closing[ids] if self.settled_close else torch.zeros_like(guided)
        if self.precise_feedback:
            latched=latched&((point_distance<=.012)&(axis_error<=.30)).all(-1)
        close=guided&(((self.phase[ids]==1)&(ready.all(-1)|latched))|(self.phase[ids]==2))
        requested=original_goals.clone()
        if self.precise_feedback:requested[guided,19:21]=-1
        if self.independent_hand_close:
            per_hand_latch=self.hand_closing[ids]&(point_distance<=.012)&(axis_error<=.30)
            close_hands=guided[:,None]&(((self.phase[ids]==1)[:,None]&(ready|per_hand_latch))|
                                        (self.phase[ids]==2)[:,None])
            requested[:,19:21]=torch.where(close_hands,1.,requested[:,19:21])
        else:
            requested[close,19:21]=1
        requested=pilot.agent.action_projector(actor,requested)
        if self.independent_hand_close:
            actual_closed_hands=close_hands&(requested[:,19:21]>0)
            self.hand_closing[ids]=actual_closed_hands&(self.phase[ids]==1)[:,None]
            closed=actual_closed_hands.all(-1)
            counts=actual_closed_hands.sum(0).tolist()
            self.individual_close_statistics['projected_closed_hand_rows']=[
                a+b for a,b in zip(self.individual_close_statistics['projected_closed_hand_rows'],counts)]
            self.individual_close_statistics['unilateral_closed_rows']+=int(actual_closed_hands.sum(-1).eq(1).sum())
        else:
            closed=close&(requested[:,19:21]>0).all(-1)
        self.closed_ticks[ids]=torch.where(closed,self.closed_ticks[ids]+1,0)
        lift_ready=self.closed_ticks[ids]>=self.contract['sustained_projected_closed_command_ticks_before_attempt_lift']
        if self.settled_close:
            first_close=closed&(self.phase[ids]==1)&~self.closing[ids]
            self.close_targets_rack[ids[first_close]]=(rack_R.transpose(-1,-2)[:,None]@
                (goal_points-rack[:,None,:3])[...,None]).squeeze(-1)[first_close]
            self.closing[ids]=closed&(self.phase[ids]==1)
            held_close=rack[:,None,:3]+(rack_R[:,None]@self.close_targets_rack[ids,...,None]).squeeze(-1)
            closure=raw[:,46:48]
            settled=closed&self.closure_initialized[ids]&(closure>=.85).all(-1)&(
                (closure-self.previous_closure[ids]).abs().amax(-1)<=.005)
            self.settled_ticks[ids]=torch.where(settled,self.settled_ticks[ids]+1,0)
            self.previous_closure[ids]=closure;self.closure_initialized[ids]=True
            lift_ready&=(self.settled_ticks[ids]>=6)&(point_distance.amax(-1)<=.006)&(axis_error<=.25).all(-1)
            if not self.precise_feedback:
                lift_ready&=(held_close-tcp[...,:3]).norm(dim=-1).amax(-1)<=.006
        lift=guided&(self.phase[ids]==1)&lift_ready
        rack_tcp=(rack_R.transpose(-1,-2)[:,None]@(tcp[...,:3]-rack[:,None,:3])[...,None]).squeeze(-1)
        self.lift_targets_rack[ids[lift]]=rack_tcp[lift]
        self.lift_targets_rack[ids[lift],:,2]+=.025
        self.lift_started[ids[lift]]=clocks[lift];self.phase[ids[lift]]=2
        self.statistics['attempted_lift_episodes']+=int(lift.sum())
        target=torch.where((self.phase[ids]==0)[:,None,None],stage,goal_points)
        if self.settled_close and not self.precise_feedback:
            target=torch.where(self.closing[ids,None,None],held_close,target)
        lifted=rack[:,None,:3]+(rack_R[:,None]@self.lift_targets_rack[ids,...,None]).squeeze(-1)
        target=torch.where((self.phase[ids]==2)[:,None,None],lifted,target)
        displacement=(target-p)*(2./30)
        if self.motion_feedback:
            # Actual TCPs retreated 5--9 mm before the jaws closed. Test
            # stronger bounded position feedback only in
            # the closing interval. The task's drives and effort caps stay
            # unchanged. Prediction uses deployable panel poses, never force.
            closing_feedback=closed&(self.phase[ids]==1)
            feedback_mask=(actual_closed_hands&(self.phase[ids]==1)[:,None]) if self.independent_hand_close else closing_feedback[:,None]
            displacement=torch.where(feedback_mask[...,None],
                (target-p)*(self.contract['closed_position_feedback_gain_per_s']/30),displacement)
            target_rack=(rack_R.transpose(-1,-2)[:,None]@(target-rack[:,None,:3])[...,None]).squeeze(-1)
            consecutive=(clocks-self.previous_target_clock[ids]==1)&(self.previous_target_phase[ids]==self.phase[ids])
            delta=target_rack-self.previous_target_rack[ids]
            if self.predictive_feedback:
                predicted=bounded_panel_prediction(delta,
                    horizon_ticks=self.contract['closed_panel_prediction_horizon_control_ticks'],
                    maximum_offset_m=self.contract['closed_panel_prediction_maximum_offset_m'])
                predicted=(rack_R[:,None]@predicted[...,None]).squeeze(-1)
                # Compensate short tracking delay only for the hands actually
                # commanded closed in this variant. A skipped clock, phase change,
                # reopening or lift cannot carry an old panel velocity forward.
                displacement+=torch.where((feedback_mask&consecutive[:,None])[...,None],
                    predicted*(self.contract['closed_position_feedback_gain_per_s']/30),0.)
            motion=delta*self.contract['current_panel_motion_feedforward_fraction']
            motion*=self.contract['panel_motion_feedforward_cap_m_per_tick']/motion.norm(dim=-1,keepdim=True).clamp_min(self.contract['panel_motion_feedforward_cap_m_per_tick'])
            motion=(rack_R[:,None]@motion[...,None]).squeeze(-1)
            displacement+=torch.where((feedback_mask&consecutive[:,None])[...,None],motion,0.)
            self.previous_target_rack[ids[guided]]=target_rack[guided]
            self.previous_target_clock[ids[guided]]=clocks[guided]
            self.previous_target_phase[ids[guided]]=self.phase[ids[guided]]
        norm=displacement.norm(dim=-1,keepdim=True)
        displacement*=.01/norm.clamp_min(.01)
        angular=rotation_error*(1.5/30)
        hold_orientation=(self.phase[ids]==2) if self.precise_feedback else (self.closed_ticks[ids]>0)
        angular=torch.where(hold_orientation[:,None,None],0.,angular)
        projector=torch.eye(3,device=raw.device,dtype=raw.dtype)-axis[..., :,None]*axis[...,None,:]
        task_J=torch.cat((J[...,:3,:],.04*(projector@J[...,3:,:])),-2)
        task_error=torch.cat((displacement,.04*angular),-1)
        regularized=task_J.transpose(-1,-2)@task_J+.0025*torch.eye(7,device=raw.device,dtype=raw.dtype)
        dq=torch.linalg.solve(regularized,(task_J.transpose(-1,-2)@task_error[...,None])).squeeze(-1).clamp(-.02,.02)
        if self.upright_control is not None:
            dq,torso_goals=self.upright_control.solve(self,pilot,raw,ids,guided,displacement,angular,J,p,projector)
            requested[guided,17:19]=torso_goals[guided]
        measured_q=torch.stack([raw[:,cols] for cols in self.kinematics.columns],1)
        pending=raw[:,:20]+raw[:,416:436]
        pending_q=torch.stack([pending[:,cols] for cols in self.kinematics.columns],1)
        proposal=pending_q+dq
        limited=torch.maximum(measured_q-.08,torch.minimum(proposal,measured_q+.08))
        self.statistics['pending_lead_clamped_rows']+=int((guided&((proposal-limited).abs().flatten(1).amax(-1)>1e-6)).sum())
        target_q=limited.clamp(self.kinematics.lower+.01,self.kinematics.upper-.01)
        arms=(target_q.reshape(n,14)-pilot.center[1:15])/pilot.scale[1:15]
        if not torch.isfinite(arms[guided]).all() or (arms[guided].abs()>1.00001).any():
            raise ValueError('Contact exploration requires bounded full URDF arm coordinates')
        requested[guided,1:15]=arms[guided]
        requested=pilot.agent.action_projector(actor,requested)
        decoded=held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*requested,pilot.stage)
        command=torch.where(guided[:,None],decoded,physical)
        untouched=[0,1,2,3,22,23] if self.upright_feedback else [0,1,2,3,18,19,22,23]
        if not torch.allclose(command[:,untouched],physical[:,untouched],atol=2e-5,rtol=0):
            raise ValueError('Contact exploration changed a protected base/waist/head channel')
        if not torch.isfinite(command).all() or command.abs().max()>1.00001:
            raise ValueError('Invalid physically bounded contact exploration command')
        self.statistics['guided_rows']+=int(guided.sum())
        self.statistics['closed_command_rows']+=int(closed.sum())
        return command,(actor,critic_features,requested)

    def report(self):return deepcopy(self.statistics)

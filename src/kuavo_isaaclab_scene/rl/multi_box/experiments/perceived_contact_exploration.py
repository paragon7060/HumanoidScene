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


def perceived_contact_contract(*, settled_close=False):
    if type(settled_close) is not bool:raise ValueError('Explicit settled-close variant required')
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
    return result


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
    def __init__(self,num_envs,raw,statistics=None,*,settled_close=False):
        if type(num_envs) is not int or num_envs<1:
            raise ValueError('Positive global environment count required')
        self.contract=perceived_contact_contract(settled_close=settled_close)
        self.settled_close=settled_close
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
        self.settled_ticks=torch.zeros_like(self.phase)
        self.previous_closure=raw.new_zeros(num_envs,2)
        self.closure_initialized=torch.zeros_like(self.closing)
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
        self.phase[ids[reset]]=-1;self.closed_ticks[ids[reset]]=0
        self.closing[ids[reset]]=False;self.settled_ticks[ids[reset]]=0
        self.closure_initialized[ids[reset]]=False
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
            self.contact_offsets[ids[handoff]]=contact_region_offsets(selected[handoff],halves[rows,flaps][handoff])
            self.assignment[ids[handoff]]=assignment[handoff]
            self.started[ids[handoff]]=clocks[handoff];self.phase[ids[handoff]]=0
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
        ready=((goal_points-tcp[...,:3]).norm(dim=-1)<=.018)&(axis_error<=.25)
        latched=self.closing[ids] if self.settled_close else torch.zeros_like(guided)
        close=guided&(((self.phase[ids]==1)&(ready.all(-1)|latched))|(self.phase[ids]==2))
        requested=original_goals.clone()
        requested[close,19:21]=1
        requested=pilot.agent.action_projector(actor,requested)
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
            lift_ready&=(self.settled_ticks[ids]>=6)&(
                (held_close-tcp[...,:3]).norm(dim=-1).amax(-1)<=.006)&(
                (goal_points-tcp[...,:3]).norm(dim=-1).amax(-1)<=.006)&(axis_error<=.25).all(-1)
        lift=guided&(self.phase[ids]==1)&lift_ready
        rack_tcp=(rack_R.transpose(-1,-2)[:,None]@(tcp[...,:3]-rack[:,None,:3])[...,None]).squeeze(-1)
        self.lift_targets_rack[ids[lift]]=rack_tcp[lift]
        self.lift_targets_rack[ids[lift],:,2]+=.025
        self.lift_started[ids[lift]]=clocks[lift];self.phase[ids[lift]]=2
        self.statistics['attempted_lift_episodes']+=int(lift.sum())
        target=torch.where((self.phase[ids]==0)[:,None,None],stage,goal_points)
        if self.settled_close:
            target=torch.where(self.closing[ids,None,None],held_close,target)
        lifted=rack[:,None,:3]+(rack_R[:,None]@self.lift_targets_rack[ids,...,None]).squeeze(-1)
        target=torch.where((self.phase[ids]==2)[:,None,None],lifted,target)
        displacement=(target-p)*(2./30);norm=displacement.norm(dim=-1,keepdim=True)
        displacement*=.01/norm.clamp_min(.01)
        angular=rotation_error*(1.5/30)
        angular=torch.where((self.closed_ticks[ids]>0)[:,None,None],0.,angular)
        projector=torch.eye(3,device=raw.device,dtype=raw.dtype)-axis[..., :,None]*axis[...,None,:]
        task_J=torch.cat((J[...,:3,:],.04*(projector@J[...,3:,:])),-2)
        task_error=torch.cat((displacement,.04*angular),-1)
        regularized=task_J.transpose(-1,-2)@task_J+.0025*torch.eye(7,device=raw.device,dtype=raw.dtype)
        dq=torch.linalg.solve(regularized,(task_J.transpose(-1,-2)@task_error[...,None])).squeeze(-1).clamp(-.02,.02)
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
        untouched=[0,1,2,3,18,19,22,23]
        if not torch.allclose(command[:,untouched],physical[:,untouched],atol=2e-5,rtol=0):
            raise ValueError('Contact exploration changed base/waist/torso/head')
        if not torch.isfinite(command).all() or command.abs().max()>1.00001:
            raise ValueError('Invalid physically bounded contact exploration command')
        self.statistics['guided_rows']+=int(guided.sum())
        self.statistics['closed_command_rows']+=int(closed.sum())
        return command,(actor,critic_features,requested)

    def report(self):return deepcopy(self.statistics)

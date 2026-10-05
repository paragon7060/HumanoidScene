"""Read-only full-grasp measurements, separate from policy inputs and Q replay."""

import gzip
import json
import math

import torch

from .flap_geometry import compare_flap_centers
from ..experiments.pose_goal_sac import GoalGripperProjector
from ..geometry.grasp import (
    GRASP_ASSIGNMENT_SCALE_M, estimated_flap_center_poses, opposing_flap_reach_assignment,
    perceived_flap_center_poses,
)
from ..scene.spawn import physical_asset_types
from ..geometry import pose_to_position_rotation_6d
from ..geometry.pose import relative_pose
from ....workcell.rack_box_layout import BOX_DIMENSIONS_M


AUDIT_SOURCE='grasp_observation_audit_NOT_matching_Q_replay'


def validate_grasp_observation_audit(waves,*,enabled,training,steps,other_probe=False,
                                    full_distribution=False):
    if full_distribution and not enabled:
        raise ValueError('Full distribution capture requires the explicit frozen grasp audit')
    if not enabled:return
    if training or other_probe or not 31<=steps<=900:
        raise ValueError('Grasp observation audit requires frozen DEV,31..900steps and unchanged physics')
    if full_distribution:
        # Same complete DEV distribution, serialized in one physics scene.
        # This cannot silently turn a selected small diagnostic into a broad
        # result, or mix TRAIN/independent FINAL rows into a frozen audit.
        if len(waves)!=128 or any(w.get('split')!='validation'
                or len(w.get('layouts',[]))!=1
                or w.get('background_placement','original')!='original' for w in waves):
            raise ValueError('Full DEV audit requires128 frozen single-case waves with original backgrounds')
        rows=[w['layouts'][0].get('layout',{}) for w in waves]
        regions=('shelf_2_left','shelf_2_right','shelf_3_left','shelf_3_right')
        if len({r.get('seed') for r in rows})!=128 or any(
                sum(r.get('target_region')==region for r in rows)!=32 for region in regions):
            raise ValueError('Full DEV audit requires distinct seeds and32 original cases in each region')
        return
    if len(waves)!=1:
        raise ValueError('Selected grasp audit requires one frozen DEV wave')
    wave=waves[0]
    if wave.get('split')!='validation' or not 1<=len(wave.get('layouts',[]))<=16:
        raise ValueError('Grasp observation audit is limited to1..16 DEV layouts; TRAIN/FINAL excluded')
    if wave.get('background_placement','original')!='original':
        raise ValueError('Grasp observation audit preserves original background placement')


def compare_close_gates(raw,actual_relations,actual_assignment):
    """Use the production gate; separate pose error from reassignment error."""
    if raw.ndim!=2 or raw.shape[1]!=464 or actual_relations.shape!=(len(raw),2,2,9) \
            or actual_assignment.shape!=(len(raw),2):
        raise ValueError('Expected current464D policy, actual2x2 relations and opposing assignment')
    nominal=raw.new_zeros(len(raw),146)
    nominal[:,108:144]=raw[:,350:386]
    nominal[:,144:146]=raw[:,386:388]
    corrected=nominal.clone();corrected[:,108:144]=actual_relations.flatten(1)
    reassigned=corrected.clone()
    reassigned[:,144:146]=torch.nn.functional.one_hot(actual_assignment[:,0],2).to(raw) \
        * (raw[:,386:388].sum(-1)>.5)[:,None]
    gate=GoalGripperProjector()
    return dict(nominal=gate.near(nominal),actual_same_assignment=gate.near(corrected),
                actual_reassigned=gate.near(reassigned))


def finite_json(value):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def valid_measured_poses(*poses):
    valid=torch.ones(len(poses[0]),dtype=torch.bool,device=poses[0].device)
    for pose in poses:
        finite=torch.isfinite(pose).flatten(1).all(-1)
        unit=(pose[...,3:].norm(dim=-1)-1).abs()<=1e-3
        valid &= finite & unit.reshape(len(pose),-1).all(-1)
    return valid


class GraspObservationAudit:
    """Capture before-action and pre-reset truth without advancing physics."""

    def __init__(self,env,directory):
        from ..state.isaac_robot_proprio import IsaacRobotProprioAdapter
        self.env=env
        self.adapter=env._multi_box_privileged_grasp
        self.proprio=IsaacRobotProprioAdapter(env)
        kinds=physical_asset_types()
        self.types=torch.tensor([0 if k=='small' else 1 for k in kinds],device=env.device)
        self.sizes=torch.tensor([BOX_DIMENSIONS_M[k] for k in kinds],device=env.device)
        self.path=directory/'grasp_observation_audit.jsonl.gz'
        self.stream=gzip.open(self.path,'wt',encoding='utf8')
        self.pending=None;self.frames=self.rows=self.invalid_rows=0
        self.completed_step=None
        self.current_wave=None
        self.layout_metadata=None
        self.waves_started=0

    def begin_wave(self,index,layouts):
        # Local step numbers restart for every scene reset. Clear both the
        # old packet and its deduplication key before settling the new scene.
        self.pending=None
        self.completed_step=None
        self.current_wave=index
        self.layout_metadata=[r['layout'] for r in layouts]
        self.waves_started=getattr(self,'waves_started',0)+1

    @torch.no_grad()
    def measure(self,raw=None):
        env=self.env;adapter=self.adapter;g=env._multi_box_privileged_grasp_step
        pool=g.target_pool_id
        box=adapter._selected_box_pose(pool)
        panels=adapter._selected_flap_pose(pool)
        tcp=adapter.tcp.center_pose_w
        valid=valid_measured_poses(box,panels,tcp)
        ids=torch.where(valid)[0]
        rows=[dict(measurement_valid=bool(v)) for v in valid.cpu().tolist()]
        if len(ids):
            compare=compare_flap_centers(box[ids],panels[ids],adapter.flap_centers[pool[ids]],
                adapter.flap_normal_axes[pool[ids]],self.sizes[pool[ids]],self.types[pool[ids]],tcp[ids])
            actual=compare['actual_center_pose_world']
            poses,distances=perceived_flap_center_poses(actual,self.sizes[pool[ids]],self.types[pool[ids]],tcp[ids])
            _,assignment=opposing_flap_reach_assignment(distances,GRASP_ASSIGNMENT_SCALE_M)
            relations=pose_to_position_rotation_6d(relative_pose(tcp[ids,:,None],poses))
            nominal_relations=pose_to_position_rotation_6d(relative_pose(
                tcp[ids,:,None],compare['nominal_center_pose_world'][:,None]))
            if raw is None:
                _,nominal_distances=estimated_flap_center_poses(box[ids],self.sizes[pool[ids]],self.types[pool[ids]],tcp[ids])
                _,nominal_assignment=opposing_flap_reach_assignment(nominal_distances,GRASP_ASSIGNMENT_SCALE_M)
                gate_raw=box.new_zeros(len(ids),464)
                gate_raw[:,350:386]=nominal_relations.flatten(1)
                gate_raw[:,386:388]=torch.nn.functional.one_hot(nominal_assignment[:,0],2).to(box)
            else:gate_raw=raw[ids]
            gates=compare_close_gates(gate_raw,relations,assignment)
            values=finite_json(compare|{'corrected_assignment':assignment,'close_gates':gates})
            if raw is not None:
                values['actor_relation_disagreement_m']=finite_json((
                    raw[ids,350:386].reshape(-1,2,2,9)[...,:3]-nominal_relations[...,:3]).norm(dim=-1).amax((1,2)))
            for j,i in enumerate(ids.cpu().tolist()):
                rows[i].update({k:({a:b[j] for a,b in v.items()} if isinstance(v,dict) else v[j]) for k,v in values.items()})
        from ..state.robot_proprio import closure_fraction
        p=self.proprio
        closure=torch.stack([closure_fraction(p.robot.data.joint_pos[:,j],opened,closed)
            for j,opened,closed in zip(p.gripper_ids,p.open_commands,p.closed_commands,strict=True)],-1)
        command=torch.cat([env.action_manager.get_term(f'{side}_gripper').raw_actions for side in ('left','right')],-1)
        common=finite_json(dict(target_pool_id=pool,target_logical_id=g.target_logical_id,
            tcp_pose_world=tcp,box_pose_world=box,flap_link_pose_world=panels,
            reward_assignment=g.assigned_flap_index,surface_distance_by_hand_m=g.matched_flap_distance_m,
            pad_force_n=g.contacts.force_n,pad_in_region=g.contacts.in_region,
            opposed=g.contacts.opposed,pinching=g.pinch.hand_pinching,
            ambiguous_hands=g.pinch.ambiguous_hands,stable_hands=g.stable_hands,
            hold_time_s=g.success.hold_time_s,proof_lift=g.success.proof_lift,
            rack_clearance_m=g.rack_clearance_m,
            jaw_alignment_error_rad=g.raw.jaw_alignment_error_rad,capture_error_m=g.raw.capture_error_m,
            measured_closure_fraction=closure,controller_close_command=command))
        if raw is not None:
            common['actor_target_logical_id']=finite_json(raw[:,400:412].argmax(-1))
            common['actor_target_valid']=finite_json(raw[:,400:412].sum(-1)>.5)
        for i,row in enumerate(rows):row.update({k:v[i] for k,v in common.items()})
        return rows

    def prepare(self,step,raw,action,active,held_ids,previous):
        self.pending=dict(step=step,ids=torch.where(active)[0].cpu().tolist(),
            held=set(held_ids.cpu().tolist()),raw=raw.clone(),action=finite_json(action[:,20:22]),
            goals={} if previous is None else dict(zip(held_ids.cpu().tolist(),finite_json(previous[2][:,19:21]))),
            before=self.measure(raw))

    def finish(self,g,safety):
        packet=self.pending
        if packet is None or self.completed_step==packet['step']:return
        # Called inside termination compute, before ManagerBasedRLEnv resets
        # completed environments. Do not reconstruct this from post-reset obs.
        after=self.measure()
        causes=finite_json({k:getattr(safety,k) for k in ('invalid_box_pose','invalid_flap_pose',
            'robot_rack_collision','self_collision','obstacle_collision','workspace_limit',
            'box_drop','box_lift_limit','box_speed_limit')})
        success=finite_json(g.success.success)
        for i in packet['ids']:
            record=dict(step=packet['step']+1,environment=i,held_phase=i in packet['held'],
                physical_jaw_command=packet['action'][i],normalized_goal_jaws=packet['goals'].get(i),
                before=packet['before'][i],after=after[i],success=success[i],
                unsafe_causes={k:v[i] for k,v in causes.items()})
            if getattr(self,'current_wave',None) is not None:
                layout=self.layout_metadata[i]
                record.update(wave=self.current_wave,layout_seed=layout['seed'],
                              target_region=layout['target_region'])
            self.stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.rows+=1;self.invalid_rows+=int(not record['after']['measurement_valid'])
        self.frames+=1;self.completed_step=packet['step']

    def close(self):
        self.stream.close()
        (self.path.parent/'grasp_observation_audit_summary.json').write_text(json.dumps(dict(
            source=AUDIT_SOURCE,frames=self.frames,rows=self.rows,invalid_measurement_rows=self.invalid_rows,
            waves_started=getattr(self,'waves_started',0),
            collection_only=True,Q_import_eligible=False,policy_observations_unchanged=True,
            no_sensors_or_physics_steps_added=True,terminal_capture_before_reset=True),indent=2)+'\n')

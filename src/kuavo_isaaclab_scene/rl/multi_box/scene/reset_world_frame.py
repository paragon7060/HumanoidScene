"""Bounded frozen reset comparison with a measured original world frame."""

import math

import torch

from .reset_diagnostics import validate_reset_diagnostic_request

SUPPORT_NAMES=tuple(f'rack_roller_deck_{i:02d}' for i in (1,2,3))
PASSIVE_STATE_MODE='original_world_frame_and_passive_state'


def _validate_joint_state(state):
    if not isinstance(state,dict) or set(state)!={'joint_names','joint_positions_rad','joint_velocities_radps'}:
        raise ValueError('Measured support state requires joint names, positions and velocities')
    names=state['joint_names']
    if not isinstance(names,list) or not names or not all(isinstance(n,str) and n for n in names) \
            or len(set(names))!=len(names):
        raise ValueError('Measured support joint names must be nonempty and unique')
    for key in ('joint_positions_rad','joint_velocities_radps'):
        values=state[key]
        if not isinstance(values,list) or len(values)!=len(names) or not all(
                isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
            raise ValueError('Measured support joint state must have finite, matching widths')


def _finite_vector(value,size):
    if not isinstance(value,list) or len(value)!=size or not all(
            isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in value):
        raise ValueError('World frame requires finite numeric origin3 and wxyz pose7')
    if size==7 and abs(math.sqrt(sum(x*x for x in value[3:]))-1)>1e-4:
        raise ValueError('World-frame quaternion must already be unit length')


def validate_world_frame_rows(probe,num_envs):
    if not isinstance(probe,dict) or probe.get('probe_type') not in ('original_world_frame','current_world_frame',PASSIVE_STATE_MODE) \
            or not isinstance(probe.get('samples'),list) or len(probe['samples'])!=num_envs:
        raise ValueError('One explicit original world frame per environment is required')
    for row in probe['samples']:
        if probe['probe_type']=='current_world_frame':
            if set(row)!={'layout_seed'}:
                raise ValueError('Current-world control accepts seeds only, never supplied poses')
            continue
        _finite_vector(row.get('environment_origin_world_m'),3)
        _finite_vector(row.get('rack_pose_world_wxyz'),7)
        supports=row.get('fixed_support_root_poses_world_wxyz',{})
        if set(supports)!=set(SUPPORT_NAMES):
            raise ValueError('All three original fixed-support poses are required')
        for pose in supports.values():_finite_vector(pose,7)
        if probe['probe_type']==PASSIVE_STATE_MODE:
            states=row.get('fixed_support_joint_states',{})
            if not isinstance(states,dict) or set(states)!=set(SUPPORT_NAMES):
                raise ValueError('Measured passive-state comparison requires all three support states')
            for state in states.values():_validate_joint_state(state)
        elif 'fixed_support_joint_states' in row:
            raise ValueError('Passive state overrides require their explicit frozen probe type')
    return probe


def validate_reset_world_frame_request(waves,probe,*,reset_enabled,training,steps,other_probe=False):
    if not reset_enabled or other_probe:
        raise ValueError('World frame requires an otherwise unchanged frozen reset diagnostic')
    validate_reset_diagnostic_request(waves,enabled=True,training=training,steps=steps)
    if len(waves)!=1:
        raise ValueError('World-frame comparison requires exactly one frozen DEV wave')
    validate_world_frame_rows(probe,len(waves[0]['layouts']))
    if [s.get('layout_seed') for s in probe['samples']]!=[r['layout']['seed'] for r in waves[0]['layouts']]:
        raise ValueError('World frames must match the requested layout seeds in scene row order')
    return probe


def apply_startup_world_frame(env,probe):
    """Match measured world placement and, only when explicit, passive DOFs.

    Active boxes/robot are restored from the unchanged neutral layout next.
    Changing origins also relocates all inactive boxes when they are parked.
    This is a frozen comparison, never a default training reset or Q input.
    """
    from .reset_kinematics import refresh_teleported_articulations
    validate_world_frame_rows(probe,env.num_envs)
    for name in SUPPORT_NAMES:
        if not env.scene[name].is_fixed_base:
            raise ValueError('Original support frame requires all three fixed Bases')
    old_origins=env.scene.env_origins.clone()
    live=probe['probe_type']=='current_world_frame'
    origins=old_origins.clone() if live else old_origins.new_tensor([s['environment_origin_world_m'] for s in probe['samples']])
    delta=origins-old_origins
    assets=dict(env.scene.rigid_objects)|dict(env.scene.articulations)
    requested=({name:assets[name].data.root_pose_w.clone() for name in ('rack',*SUPPORT_NAMES)} if live else
        {'rack':old_origins.new_tensor([s['rack_pose_world_wxyz'] for s in probe['samples']])}|
        {name:old_origins.new_tensor([s['fixed_support_root_poses_world_wxyz'][name] for s in probe['samples']])
            for name in SUPPORT_NAMES})
    # Validate all required assets and tensor shapes before the first mutation.
    if not set(requested)<=set(assets):raise ValueError('World-frame assets are missing')
    moves={name:asset.data.root_pose_w.clone() for name,asset in assets.items()
        if name in requested or name.startswith('conveyor_')}
    for name,pose in moves.items():
        if pose.shape!=(env.num_envs,7):raise ValueError('World-frame root view shape mismatch')
        if name in requested:pose.copy_(requested[name])
        else:pose[:,:3]+=delta
    ids=torch.arange(env.num_envs,device=env.device)
    before={name:assets[name].data.root_pose_w.detach().cpu().tolist() for name in moves}
    placement_changed=bool((delta.norm(dim=-1)>1e-5).any()) or any(
        bool(((pose[:,:3]-pose.new_tensor(before[name])[:,:3]).norm(dim=-1)>1e-5).any()) or
        bool((torch.nn.functional.cosine_similarity(pose[:,3:],pose.new_tensor(before[name])[:,3:],dim=-1).abs()<1-1e-5).any())
        for name,pose in moves.items())
    passive={name:(assets[name].data.joint_pos.clone(),assets[name].data.joint_vel.clone()) for name in SUPPORT_NAMES}
    matched=probe['probe_type']==PASSIVE_STATE_MODE
    expected_passive=passive
    if matched:
        # Joint names, order and widths must match the actual runtime before
        # any origin, root or DOF is written. Never infer an index mapping.
        for name in SUPPORT_NAMES:
            for row in probe['samples']:
                if row['fixed_support_joint_states'][name]['joint_names']!=list(assets[name].joint_names):
                    raise ValueError('Measured support joint order differs from the actual runtime')
        expected_passive={name:tuple(passive[name][i].new_tensor([
            row['fixed_support_joint_states'][name][key] for row in probe['samples']])
            for i,key in enumerate(('joint_positions_rad','joint_velocities_radps'))) for name in SUPPORT_NAMES}
        if any(q.shape!=passive[name][0].shape or v.shape!=passive[name][1].shape
               for name,(q,v) in expected_passive.items()):
            raise ValueError('Measured support joint tensor shape differs from the actual runtime')
    link_before={name:assets[name].root_physx_view.get_link_transforms().clone() for name in SUPPORT_NAMES}
    env.scene.env_origins.copy_(origins)
    for name,pose in moves.items():assets[name].write_root_pose_to_sim(pose,env_ids=ids)
    if matched:
        for name,(q,v) in expected_passive.items():assets[name].write_joint_state_to_sim(q,v,env_ids=ids)
    refresh_teleported_articulations(env,[assets[name] for name in moves if name in env.scene.articulations],ids)
    env.sim.forward();env.scene.update(env.step_dt)
    after={name:assets[name].data.root_pose_w.detach().cpu().tolist() for name in moves}
    errors={name:float((assets[name].data.root_pose_w[:,:3]-pose[:,:3]).norm(dim=-1).max())
        for name,pose in moves.items()}
    if any(value>1e-5 for value in errors.values()):raise ValueError('Requested original root frame was not applied')
    quat_dots={name:float(torch.nn.functional.cosine_similarity(
        assets[name].data.root_pose_w[:,3:],pose[:,3:],dim=-1).abs().min()) for name,pose in moves.items()}
    if any(value<1-1e-5 for value in quat_dots.values()):raise ValueError('Requested original root rotation was not applied')
    for name,(q,v) in expected_passive.items():
        if not torch.equal(q,assets[name].data.joint_pos) or not torch.equal(v,assets[name].data.joint_vel):
            raise ValueError('World-frame comparison did not retain the requested passive joint state')
        if matched and (not torch.equal(q,assets[name].root_physx_view.get_dof_positions()) or
                        not torch.equal(v,assets[name].root_physx_view.get_dof_velocities())):
            raise ValueError('Actual backend did not apply the measured passive joint state')
    passive_retained=all(torch.equal(q,assets[name].data.joint_pos) and torch.equal(v,assets[name].data.joint_vel)
        for name,(q,v) in passive.items())
    state_changes={name:dict(maximum_position_change_rad=float((q-passive[name][0]).abs().max()),
        maximum_velocity_change_radps=float((v-passive[name][1]).abs().max()))
        for name,(q,v) in expected_passive.items()}
    link_shifts={name:(assets[name].root_physx_view.get_link_transforms()[...,:3]-pose[...,:3])
        .norm(dim=-1).max(dim=-1).values.cpu().tolist() for name,pose in link_before.items()}
    return dict(probe_type=probe['probe_type'],frozen_only=True,Q_import_eligible=False,
        requested=probe,original_origins_world_m=old_origins.cpu().tolist(),
        applied_origins_world_m=env.scene.env_origins.cpu().tolist(),roots_before=before,roots_after=after,
        maximum_root_position_errors_m=errors,minimum_root_absolute_quaternion_dots=quat_dots,
        passive_joint_positions_velocities_retained=passive_retained,
        passive_joint_states_matched_to_measured_reference=matched,
        initial_passive_joint_state_changed=not passive_retained,
        passive_joint_state_changes_from_current=state_changes,
        support_FK_refresh_without_physics_step=True,
        actual_backend_support_link_maximum_center_shift_m=link_shifts,
        world_root_placements_requested=True,world_root_placements_changed=placement_changed,
        initial_rack_relative_requested_layout_unchanged=True,
        physics_parameters_success_and_safety_unchanged=True,
        constructor_and_contact_solver_history_not_matched=True)

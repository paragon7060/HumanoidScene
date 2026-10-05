"""Opt-in firmer dynamic hinges; no fixed links or in-episode pose writes."""
from copy import deepcopy
import math

import torch


def firm_flap_dynamics_contract():
    # Nominal v2: zero stiffness, .05 damping and .45/.32 friction. A modest
    # spring/damper is also used by the earlier flap-pick scene (2/.2).
    return dict(name='firm_dynamic_flap_randomization_v1',
        joint_names=['joint_front','joint_back','joint_left','joint_right'],
        stiffness_nm_per_rad=[1.5,2.5],damping_nm_s_per_rad=[.15,.25],
        static_friction=[.45,.65],dynamic_friction=[.30,.40],
        initial_angle_degrees=[-1.,1.],spring_rest_angle_rad=0.,
        effort_limit_nm=5.,velocity_limit_rad_s=10.,
        distribution='independent_uniform_per_environment_panel_and_reset',
        joint_limits_masses_geometry_contact_materials_unchanged=True,
        dynamic_articulation=True,in_episode_state_overwrite=False,
        box_base_and_background_layout_randomization_unchanged=True)


def require_flap_dynamics(value):
    if value!=firm_flap_dynamics_contract():raise ValueError('Unknown flap dynamics/randomization contract')


def configure_flap_dynamics(cfg,contract):
    profile=contract.get('flap_dynamics')
    if profile is None:return
    require_flap_dynamics(profile)
    from .spawn import physical_asset_names
    for name in physical_asset_names():
        actuator=getattr(cfg.scene,name).actuators['flaps']
        actuator.stiffness=sum(profile['stiffness_nm_per_rad'])/2
        actuator.damping=sum(profile['damping_nm_s_per_rad'])/2
        actuator.friction=sum(profile['static_friction'])/2
        actuator.dynamic_friction=sum(profile['dynamic_friction'])/2
        if actuator.effort_limit_sim!=profile['effort_limit_nm'] or actuator.velocity_limit_sim!=profile['velocity_limit_rad_s']:
            raise ValueError('Flap profile must retain nominal physical torque and velocity limits')
    cfg.flap_dynamics=deepcopy(profile)


def randomize_flap_dynamics(env,ids):
    """Apply sampled parameters and small initial deflection at reset only."""
    profile=getattr(env.cfg,'flap_dynamics',None)
    if profile is None or not len(ids):return
    require_flap_dynamics(profile)
    from .spawn import physical_asset_names
    audit={}
    for name in physical_asset_names():
        asset=env.scene[name]
        joints,_=asset.find_joints(profile['joint_names'],preserve_order=True)
        if len(joints)!=4:raise ValueError('Firmer flap profile needs four physical hinge joints')
        shape=(len(ids),4)
        values={key:torch.empty(shape,device=env.device).uniform_(*profile[key]) for key in
            ('stiffness_nm_per_rad','damping_nm_s_per_rad','static_friction','dynamic_friction','initial_angle_degrees')}
        asset.write_joint_stiffness_to_sim(values['stiffness_nm_per_rad'],joint_ids=joints,env_ids=ids)
        asset.write_joint_damping_to_sim(values['damping_nm_s_per_rad'],joint_ids=joints,env_ids=ids)
        asset.write_joint_friction_coefficient_to_sim(values['static_friction'],joint_ids=joints,env_ids=ids)
        asset.write_joint_dynamic_friction_coefficient_to_sim(values['dynamic_friction'],joint_ids=joints,env_ids=ids)
        motor=asset.actuators['flaps']
        motor.stiffness[ids]=asset.data.joint_stiffness[ids][:,motor.joint_indices]
        motor.damping[ids]=asset.data.joint_damping[ids][:,motor.joint_indices]
        angle=values['initial_angle_degrees']*(math.pi/180)
        limits=asset.data.joint_pos_limits[ids[:,None],joints]
        angle=angle.clamp(min=limits[...,0],max=limits[...,1])
        zeros=torch.zeros_like(angle)
        asset.write_joint_state_to_sim(angle,zeros,joint_ids=joints,env_ids=ids)
        asset.set_joint_position_target(zeros,joint_ids=joints,env_ids=ids)
        asset.set_joint_velocity_target(zeros,joint_ids=joints,env_ids=ids)
        actual={key:getattr(asset.data,attr)[ids[:,None],joints] for key,attr in (
            ('stiffness_nm_per_rad','joint_stiffness'),('damping_nm_s_per_rad','joint_damping'),
            ('static_friction','joint_friction_coeff'),('dynamic_friction','joint_dynamic_friction_coeff'))}
        if any(not torch.allclose(actual[k],values[k],atol=1e-6,rtol=0) for k in actual):
            raise ValueError('Sampled flap dynamics failed initialized-buffer readback')
        sim_ids=ids.cpu()[:,None]
        stiffness=asset.root_physx_view.get_dof_stiffnesses()[sim_ids,joints]
        damping=asset.root_physx_view.get_dof_dampings()[sim_ids,joints]
        friction=asset.root_physx_view.get_dof_friction_properties()[sim_ids,joints]
        sim_actual=dict(stiffness_nm_per_rad=stiffness,damping_nm_s_per_rad=damping,
            static_friction=friction[...,0],dynamic_friction=friction[...,1])
        if any(not torch.allclose(sim_actual[k],values[k].cpu(),atol=1e-6,rtol=0) for k in sim_actual):
            raise ValueError('Sampled flap dynamics failed actual PhysX property readback')
        audit[name]={k:v.detach().cpu().tolist() for k,v in actual.items()}
        audit[name]['initial_angle_rad']=angle.detach().cpu().tolist()
        audit[name]['PhysX_properties_verified']=True
    env._flap_dynamics_last_reset_audit=dict(contract=profile,env_ids=ids.cpu().tolist(),assets=audit)
    # Auto-reset can overwrite the last-reset audit with a small subset. Keep
    # the verified current properties by global environment identity as well.
    # This is a CPU record, not an extra PhysX write or a simulation step.
    record=getattr(env,'_flap_dynamics_current_parameters',None)
    count=len(next(iter(audit.values()))['stiffness_nm_per_rad'])
    if count!=len(ids):raise ValueError('Flap reset audit identity count differs')
    num_envs=env.num_envs
    if record is None:
        record=dict(contract=deepcopy(profile),initialized=torch.zeros(num_envs,dtype=torch.bool),assets={})
        env._flap_dynamics_current_parameters=record
    if record['contract']!=profile or record['initialized'].shape!=(num_envs,):
        raise ValueError('Current flap property record belongs to another contract or environment count')
    selected=ids.detach().cpu()
    for name,properties in audit.items():
        if name not in record['assets']:
            record['assets'][name]={key:torch.zeros(num_envs,4) for key in
                ('stiffness_nm_per_rad','damping_nm_s_per_rad','static_friction','dynamic_friction')}
        cached=record['assets'][name]
        for key,value in cached.items():value[selected]=torch.tensor(properties[key])
    record['initialized'][selected]=True


def current_flap_dynamics_audit(env,*,original_layout_valid=None):
    """All initialized current profiles; replacements are labelled separately."""
    profile=getattr(env.cfg,'flap_dynamics',None)
    if profile is None:return None
    record=getattr(env,'_flap_dynamics_current_parameters',None)
    if record is None or record['contract']!=profile:
        raise ValueError('No verified current flap parameters were recorded')
    selected=torch.where(record['initialized'])[0]
    result=dict(contract=deepcopy(profile),env_ids=selected.tolist(),
        scope='current_verified_properties_for_all_initialized_environments',
        state_overwritten_or_randomization_resampled=False,
        assets={name:{**{key:value[selected].tolist() for key,value in properties.items()},
                      'PhysX_properties_verified':True} for name,properties in record['assets'].items()})
    if original_layout_valid is not None:
        valid=torch.as_tensor(original_layout_valid).detach().cpu()
        if valid.shape!=record['initialized'].shape or valid.dtype!=torch.bool:
            raise ValueError('Original-layout flags must match current flap environment identities')
        result.update(original_layout_valid=valid[selected].tolist(),
            invalid_original_cases_are_current_replacement_parameters=True)
    return result


def flap_initial_parameters(env,index):
    """Numeric, detached four-joint parameters for one pre-action snapshot."""
    profile=getattr(getattr(env,'cfg',None),'flap_dynamics',None)
    if profile is None:return None
    record=getattr(env,'_flap_dynamics_current_parameters',None)
    if record is None or record['contract']!=profile or type(index) is not int \
            or not 0<=index<len(record['initialized']) or not record['initialized'][index]:
        raise ValueError('Physical flap seed requires verified parameters for this environment')
    return {name:{key:value[index].numpy().copy() for key,value in properties.items()}
            for name,properties in record['assets'].items()}

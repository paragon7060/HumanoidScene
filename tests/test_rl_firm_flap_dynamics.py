"""The optional hinge profile modifies only requested reset environments."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import (
    firm_flap_dynamics_contract,require_flap_dynamics,configure_flap_dynamics,randomize_flap_dynamics,
    current_flap_dynamics_audit,flap_initial_parameters)
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names


def test_profile_is_opt_in_and_rejects_unknown_physics():
    cfg=SimpleNamespace(scene=SimpleNamespace())
    configure_flap_dynamics(cfg,{})
    assert not hasattr(cfg,'flap_dynamics')
    with pytest.raises(ValueError,match='Unknown'):
        require_flap_dynamics(firm_flap_dynamics_contract()|{'dynamic_articulation':False})


def test_hinge_randomization_is_independent_and_preserves_unselected_environments():
    class Asset:
        def __init__(self):
            self.data=SimpleNamespace(joint_stiffness=torch.zeros(4,4),joint_damping=torch.full((4,4),.05),
                joint_friction_coeff=torch.full((4,4),.45),joint_dynamic_friction_coeff=torch.full((4,4),.32),
                joint_pos_limits=torch.tensor([-3.14,3.14]).expand(4,4,2),
                joint_pos=torch.zeros(4,4),joint_pos_target=torch.zeros(4,4),joint_vel_target=torch.zeros(4,4))
            self.actuators={'flaps':SimpleNamespace(joint_indices=slice(None),stiffness=torch.zeros(4,4),damping=torch.full((4,4),.05))}
            self.root_physx_view=SimpleNamespace(get_dof_stiffnesses=lambda:self.data.joint_stiffness.clone(),
                get_dof_dampings=lambda:self.data.joint_damping.clone(),get_dof_friction_properties=lambda:torch.stack((self.data.joint_friction_coeff,self.data.joint_dynamic_friction_coeff,torch.zeros(4,4)),-1))
        def find_joints(self,names,preserve_order):return [0,1,2,3],names
        def put(self,name,value,joint_ids,env_ids):getattr(self.data,name)[env_ids[:,None],joint_ids]=value
        def write_joint_stiffness_to_sim(self,value,**kwargs):self.put('joint_stiffness',value,**kwargs)
        def write_joint_damping_to_sim(self,value,**kwargs):self.put('joint_damping',value,**kwargs)
        def write_joint_friction_coefficient_to_sim(self,value,**kwargs):self.put('joint_friction_coeff',value,**kwargs)
        def write_joint_dynamic_friction_coefficient_to_sim(self,value,**kwargs):self.put('joint_dynamic_friction_coeff',value,**kwargs)
        def write_joint_state_to_sim(self,pos,vel,**kwargs):self.put('joint_pos',pos,**kwargs)
        def set_joint_position_target(self,value,**kwargs):self.put('joint_pos_target',value,**kwargs)
        def set_joint_velocity_target(self,value,**kwargs):self.put('joint_vel_target',value,**kwargs)
    names=physical_asset_names();scene={name:Asset() for name in names}
    env=SimpleNamespace(cfg=SimpleNamespace(flap_dynamics=firm_flap_dynamics_contract()),scene=scene,device='cpu',num_envs=4)
    ids=torch.tensor([1,3]);torch.manual_seed(12);randomize_flap_dynamics(env,ids)
    for asset in scene.values():
        stiffness=asset.data.joint_stiffness
        assert stiffness[[0,2]].eq(0).all() and stiffness[ids].amin()>=1.5 and stiffness[ids].amax()<=2.5
        assert stiffness[ids].std()>0
        assert asset.data.joint_damping[ids].amin()>=.15 and asset.data.joint_damping[ids].amax()<=.25
        assert asset.data.joint_pos[ids].abs().max()<=torch.pi/180
        assert asset.data.joint_pos[[0,2]].eq(0).all() and asset.data.joint_pos_target.eq(0).all()
        assert (asset.data.joint_dynamic_friction_coeff<=asset.data.joint_friction_coeff).all()
    before={name:asset.data.joint_stiffness.clone() for name,asset in scene.items()}
    randomize_flap_dynamics(env,ids)
    assert all(not torch.equal(scene[name].data.joint_stiffness[ids],value[ids]) for name,value in before.items())
    assert env._flap_dynamics_last_reset_audit['env_ids']==[1,3]
    # The current-property record survives a later disjoint auto-reset and
    # snapshots never alias that mutable record.
    snapshot=flap_initial_parameters(env,1)
    randomize_flap_dynamics(env,torch.tensor([0]))
    full=current_flap_dynamics_audit(env,original_layout_valid=torch.tensor([False,True,False,True]))
    assert env._flap_dynamics_last_reset_audit['env_ids']==[0]
    assert full['env_ids']==[0,1,3] and full['original_layout_valid']==[False,True,True]
    assert full['invalid_original_cases_are_current_replacement_parameters']
    for name in names:
        torch.testing.assert_close(torch.tensor(full['assets'][name]['stiffness_nm_per_rad']),scene[name].data.joint_stiffness[[0,1,3]])
        torch.testing.assert_close(torch.tensor(snapshot[name]['stiffness_nm_per_rad']),scene[name].data.joint_stiffness[1])
    randomize_flap_dynamics(env,torch.tensor([1]))
    assert any(not torch.equal(torch.tensor(snapshot[name]['stiffness_nm_per_rad']),scene[name].data.joint_stiffness[1]) for name in names)
    with pytest.raises(ValueError,match='verified parameters'):flap_initial_parameters(env,2)
    with pytest.raises(ValueError,match='flags'):current_flap_dynamics_audit(env,original_layout_valid=torch.zeros(3,dtype=torch.bool))

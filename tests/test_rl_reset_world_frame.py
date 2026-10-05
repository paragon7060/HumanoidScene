from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_world_frame import (
    SUPPORT_NAMES,PASSIVE_STATE_MODE,validate_world_frame_rows,validate_reset_world_frame_request,apply_startup_world_frame,
)


def probe():
    pose=[10.,20.,0.,1.,0.,0.,0.]
    return dict(probe_type='original_world_frame',samples=[dict(layout_seed=123,
        environment_origin_world_m=[10.,20.,0.],rack_pose_world_wxyz=pose,
        fixed_support_root_poses_world_wxyz={name:[10.,20.,float(i),1.,0.,0.,0.]
            for i,name in enumerate(SUPPORT_NAMES)})])


@pytest.mark.parametrize('measured_passive_state',[False,True])
def test_world_frame_requires_frozen_DEV_one_step_and_exact_layout_identity(measured_passive_state):
    p=passive_probe() if measured_passive_state else probe()
    waves=[dict(split='validation',layouts=[dict(layout=dict(seed=123))])]
    validate_reset_world_frame_request(waves,p,reset_enabled=True,training=False,steps=1)
    for changed in [dict(training=True),dict(steps=2),dict(reset_enabled=False),dict(other_probe=True)]:
        args=dict(reset_enabled=True,training=False,steps=1);args.update(changed)
        with pytest.raises(ValueError):validate_reset_world_frame_request(waves,p,**args)
    for wrong in [[dict(split='holdout',layouts=waves[0]['layouts'])],waves*2,
                  [dict(split='validation',layouts=[dict(layout=dict(seed=124))])]]:
        with pytest.raises(ValueError):validate_reset_world_frame_request(wrong,p,reset_enabled=True,training=False,steps=1)


@pytest.mark.parametrize('change', ['missing_support','nan_origin','zero_quaternion','wrong_count'])
def test_invalid_frame_rejected_before_it_can_change_world_state(change):
    p=probe()
    if change=='missing_support':del p['samples'][0]['fixed_support_root_poses_world_wxyz'][SUPPORT_NAMES[0]]
    if change=='nan_origin':p['samples'][0]['environment_origin_world_m'][0]=float('nan')
    if change=='zero_quaternion':p['samples'][0]['rack_pose_world_wxyz'][3:]=[0.]*4
    if change=='wrong_count':p['samples']=[]
    with pytest.raises(ValueError):validate_world_frame_rows(p,1)


class Asset:
    def __init__(self,z=0.,fixed=True):
        self.is_fixed_base=fixed
        self.joint_names=['roller_joint']
        self.data=SimpleNamespace(root_pose_w=torch.tensor([[0.,0.,z,1.,0.,0.,0.]]),
            joint_pos=torch.tensor([[.2]]),joint_vel=torch.tensor([[.03]]))
        self.root_physx_view=SimpleNamespace(get_link_transforms=lambda:self.data.root_pose_w[:,None,[0,1,2,4,5,6,3]],
            get_dof_positions=lambda:self.data.joint_pos.clone(),get_dof_velocities=lambda:self.data.joint_vel.clone())
    def write_root_pose_to_sim(self,pose,env_ids):self.data.root_pose_w[env_ids]=pose
    def write_joint_state_to_sim(self,q,v,env_ids):
        self.data.joint_pos[env_ids]=q;self.data.joint_vel[env_ids]=v


class Scene(dict):
    def __init__(self):
        super().__init__(rack=Asset(),robot=Asset(),mb_s2_small_4=Asset(),conveyor_surface=Asset(),
            **{name:Asset(float(i)) for i,name in enumerate(SUPPORT_NAMES)})
        self.env_origins=torch.zeros(1,3)
        self.articulations={name:self[name] for name in SUPPORT_NAMES}
        self.rigid_objects={k:v for k,v in self.items() if k not in self.articulations}
    def update(self,dt):pass


def test_apply_keeps_passive_DOF_state_and_defers_robot_and_box_restoration(monkeypatch):
    monkeypatch.setattr('kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics.refresh_teleported_articulations',lambda *args:False)
    scene=Scene();env=SimpleNamespace(num_envs=1,device='cpu',scene=scene,step_dt=.03,sim=SimpleNamespace(forward=lambda:None))
    p=probe();before=deepcopy(p)
    result=apply_startup_world_frame(env,p)
    assert p==before and result['passive_joint_positions_velocities_retained']
    assert result['world_root_placements_changed']
    assert result['Q_import_eligible'] is False and result['constructor_and_contact_solver_history_not_matched']
    torch.testing.assert_close(scene.env_origins,torch.tensor([[10.,20.,0.]]))
    torch.testing.assert_close(scene['conveyor_surface'].data.root_pose_w[:,:3],torch.tensor([[10.,20.,0.]]))
    for name in ('robot','mb_s2_small_4'):assert scene[name].data.root_pose_w[:,:3].count_nonzero()==0
    for name in SUPPORT_NAMES:
        torch.testing.assert_close(scene[name].data.joint_pos,torch.tensor([[.2]]))
        torch.testing.assert_close(scene[name].data.joint_vel,torch.tensor([[.03]]))


def test_nonfixed_support_rejected_before_any_origin_or_pose_write():
    scene=Scene();scene[SUPPORT_NAMES[0]].is_fixed_base=False
    env=SimpleNamespace(num_envs=1,device='cpu',scene=scene)
    with pytest.raises(ValueError,match='fixed Bases'):apply_startup_world_frame(env,probe())
    assert scene.env_origins.count_nonzero()==0 and scene['rack'].data.root_pose_w[:,:3].count_nonzero()==0


def test_sham_root_setter_and_FK_control_reports_no_world_change(monkeypatch):
    monkeypatch.setattr('kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics.refresh_teleported_articulations',lambda *args:False)
    scene=Scene();env=SimpleNamespace(num_envs=1,device='cpu',scene=scene,step_dt=.03,sim=SimpleNamespace(forward=lambda:None))
    p=probe();s=p['samples'][0];s['environment_origin_world_m']=[0.,0.,0.]
    s['rack_pose_world_wxyz']=scene['rack'].data.root_pose_w[0].tolist()
    s['fixed_support_root_poses_world_wxyz']={name:scene[name].data.root_pose_w[0].tolist() for name in SUPPORT_NAMES}
    result=apply_startup_world_frame(env,p)
    assert result['world_root_placements_requested'] and not result['world_root_placements_changed']


def test_live_current_frame_control_accepts_seed_identity_but_rejects_any_pose_override():
    p=dict(probe_type='current_world_frame',samples=[dict(layout_seed=123)])
    waves=[dict(split='validation',layouts=[dict(layout=dict(seed=123))])]
    validate_reset_world_frame_request(waves,p,reset_enabled=True,training=False,steps=1)
    p['samples'][0]['rack_pose_world_wxyz']=[0.,0.,0.,1.,0.,0.,0.]
    with pytest.raises(ValueError,match='seeds only'):validate_world_frame_rows(p,1)


def test_live_current_frame_control_preserves_current_root_geometry(monkeypatch):
    monkeypatch.setattr('kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics.refresh_teleported_articulations',lambda *args:False)
    scene=Scene();scene.env_origins[:]=torch.tensor([[44.,24.,0.]])
    for asset in scene.values():asset.data.root_pose_w[:,:3]+=scene.env_origins
    before={name:asset.data.root_pose_w.clone() for name,asset in scene.items()}
    env=SimpleNamespace(num_envs=1,device='cpu',scene=scene,step_dt=.03,sim=SimpleNamespace(forward=lambda:None))
    result=apply_startup_world_frame(env,dict(probe_type='current_world_frame',samples=[dict(layout_seed=123)]))
    assert result['probe_type']=='current_world_frame' and not result['world_root_placements_changed']
    for name,asset in scene.items():torch.testing.assert_close(asset.data.root_pose_w,before[name])


def passive_probe():
    p=probe();p['probe_type']=PASSIVE_STATE_MODE
    p['samples'][0]['fixed_support_joint_states']={name:dict(joint_names=['roller_joint'],
        joint_positions_rad=[1.2],joint_velocities_radps=[-2.3]) for name in SUPPORT_NAMES}
    return p


@pytest.mark.parametrize('change',['missing_support','duplicate_names','nonfinite','wrong_width','implicit_override'])
def test_measured_passive_state_requires_explicit_complete_finite_state(change):
    p=passive_probe();states=p['samples'][0]['fixed_support_joint_states'];s=states[SUPPORT_NAMES[0]]
    if change=='missing_support':del states[SUPPORT_NAMES[0]]
    if change=='duplicate_names':s['joint_names']=['roller_joint','roller_joint']
    if change=='nonfinite':s['joint_velocities_radps']=[float('inf')]
    if change=='wrong_width':s['joint_positions_rad']=[]
    if change=='implicit_override':p['probe_type']='original_world_frame'
    with pytest.raises(ValueError):validate_world_frame_rows(p,1)


def test_measured_passive_state_joint_order_rejected_before_any_mutation():
    scene=Scene();p=passive_probe()
    p['samples'][0]['fixed_support_joint_states'][SUPPORT_NAMES[0]]['joint_names']=['other_joint']
    env=SimpleNamespace(num_envs=1,device='cpu',scene=scene)
    with pytest.raises(ValueError,match='joint order'):apply_startup_world_frame(env,p)
    assert scene.env_origins.count_nonzero()==0 and scene['rack'].data.root_pose_w[:,:3].count_nonzero()==0
    assert scene[SUPPORT_NAMES[0]].data.joint_pos.item()==pytest.approx(.2)


def test_measured_passive_state_written_without_physics_step_and_marked_changed(monkeypatch):
    monkeypatch.setattr('kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics.refresh_teleported_articulations',lambda *args:False)
    scene=Scene();env=SimpleNamespace(num_envs=1,device='cpu',scene=scene,step_dt=.03,sim=SimpleNamespace(forward=lambda:None))
    p=passive_probe();original=deepcopy(p)
    result=apply_startup_world_frame(env,p)
    assert p==original and result['passive_joint_states_matched_to_measured_reference']
    assert result['initial_passive_joint_state_changed'] and not result['passive_joint_positions_velocities_retained']
    assert result['Q_import_eligible'] is False and result['constructor_and_contact_solver_history_not_matched']
    for name in SUPPORT_NAMES:
        torch.testing.assert_close(scene[name].data.joint_pos,torch.tensor([[1.2]]))
        torch.testing.assert_close(scene[name].data.joint_vel,torch.tensor([[-2.3]]))
        assert result['passive_joint_state_changes_from_current'][name]['maximum_velocity_change_radps']==pytest.approx(2.33)
    for name in ('robot','mb_s2_small_4'):assert scene[name].data.root_pose_w[:,:3].count_nonzero()==0


def test_measured_passive_state_backend_mismatch_is_not_accepted(monkeypatch):
    monkeypatch.setattr('kuavo_isaaclab_scene.rl.multi_box.scene.reset_kinematics.refresh_teleported_articulations',lambda *args:False)
    scene=Scene();scene[SUPPORT_NAMES[0]].root_physx_view.get_dof_velocities=lambda:torch.zeros(1,1)
    env=SimpleNamespace(num_envs=1,device='cpu',scene=scene,step_dt=.03,sim=SimpleNamespace(forward=lambda:None))
    with pytest.raises(ValueError,match='Actual backend'):apply_startup_world_frame(env,passive_probe())

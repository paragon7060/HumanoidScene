"""Original reset evidence survives partial respawns without changing physics."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import (
    ResetFailureCapture, validate_reset_diagnostic_request, zero_passive_roller_velocities,
)
from kuavo_isaaclab_scene.rl.multi_box.scene.reset_settling import IsaacResetSettling
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells, physical_asset_names
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def environment():
    spec = MultiBoxSpec()
    p, q = logical_cells(spec)[0].local_pose('small')
    pose = torch.tensor([(*p, *q)]).repeat(4, 1)
    pose[1, 1] -= 1.
    pose[3, 3] = float('nan')
    velocity = torch.zeros(4, 6)
    velocity[2, 0] = .02
    rack = torch.zeros(4, 7); rack[:, 3] = 1
    robot = rack.clone(); robot[:, 0] = 10.
    box = SimpleNamespace(joint_names=['flap'], data=SimpleNamespace(
        root_pose_w=pose, root_vel_w=velocity, joint_pos=torch.ones(4, 1)))
    scene = dict(rack=SimpleNamespace(data=SimpleNamespace(root_pose_w=rack)),
                 robot=SimpleNamespace(data=SimpleNamespace(root_pose_w=robot)))
    scene.update({name:box for name in physical_asset_names()})
    active = torch.zeros(4, 12, dtype=torch.bool); active[:, 0] = True
    return SimpleNamespace(num_envs=4, device='cpu', common_step_counter=0,
        cfg=SimpleNamespace(multi_box=spec), scene=scene,
        _multi_box_active=active, _multi_box_pool_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_box_type_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_region_ids=torch.zeros(4,12,dtype=torch.long),
        _multi_box_rack_local_positions=torch.zeros(4,12,3))


def advance(env, tracker):
    env.common_step_counter += 1
    return tracker.measure(.1)


def test_first_failure_is_measured_before_respawn_and_survives_replacement_failures():
    env = environment(); tracker = IsaacResetSettling(env)
    tracker.failure_capture = ResetFailureCapture(env)
    original_pose = env.scene[tracker.names[0]].data.root_pose_w.clone()
    advance(env,tracker)
    assert tracker.failure_capture.records == []  # original first-step contact grace
    advance(env,tracker)
    captured = deepcopy(tracker.failure_capture.records)
    assert [r['environment'] for r in captured] == [1,3]
    assert captured[0]['box_pose_world'][1] == float(original_pose[1,1])
    assert captured[0]['left_region'] and not captured[0]['timed_out']
    assert captured[0]['robot_pose_world'][0] == 10.
    assert captured[1]['invalid_box_pose'] and captured[1]['box_pose_world'][3] is None
    assert captured[1]['rack_local_root_xyz_m'] is None
    json.dumps(captured, allow_nan=False)
    # This is the same tracker reset used by partial respawn. Park the old
    # asset and provoke a replacement failure; original evidence stays exact.
    tracker.reset(torch.tensor([1,3]))
    env.scene[tracker.names[0]].data.root_pose_w[1,1] = -100.
    advance(env,tracker); advance(env,tracker)
    assert tracker.failure_capture.records == captured
    assert tracker.invalid_count[1] == 2
    # Capture is bounded by the number of requested environments, not retries.
    assert len(tracker.failure_capture.records) <= env.num_envs


def test_timeout_is_distinct_from_geometry_and_capture_does_not_mutate_scene():
    env = environment(); tracker = IsaacResetSettling(env)
    tracker.failure_capture = ResetFailureCapture(env)
    box = env.scene[tracker.names[0]]
    original = [v.clone() for v in (box.data.root_pose_w,box.data.root_vel_w,box.data.joint_pos)]
    for _ in range(22): advance(env,tracker)
    cases = {r['environment']:r for r in tracker.failure_capture.records}
    assert 0 not in cases and tracker.ready[0]
    assert cases[2]['timed_out'] and not cases[2]['left_region']
    assert cases[2]['footprint_in_region'] and cases[2]['on_assigned_shelf']
    assert not cases[2]['stable']
    assert cases[2]['box_velocity_world'][0] == pytest.approx(.02)
    for before,after in zip(original,(box.data.root_pose_w,box.data.root_vel_w,box.data.joint_pos)):
        torch.testing.assert_close(before,after,equal_nan=True)


def test_disabled_capture_keeps_reset_outcomes_identical():
    plain,observed = environment(),environment()
    a,b = IsaacResetSettling(plain),IsaacResetSettling(observed)
    assert a.failure_capture is None
    b.failure_capture = ResetFailureCapture(observed)
    for _ in range(22):
        x,y = advance(plain,a),advance(observed,b)
        for name in x.__dataclass_fields__:
            torch.testing.assert_close(getattr(x,name),getattr(y,name))
    for name in ('invalid_count','region_invalid_count','shelf_invalid_count',
                 'footprint_invalid_count','timeout_invalid_count','nonfinite_invalid_count'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('waves,training,steps',[
    ([dict(split='train')],False,1),([dict(split='holdout')],False,1),
    ([dict(split='validation')],True,1),([dict(split='validation')],False,900),
    ([],False,1),
])
def test_diagnostic_mode_cannot_collect_training_or_inspect_independent_final(waves,training,steps):
    with pytest.raises(ValueError,match='frozen DEV'):
        validate_reset_diagnostic_request(waves,enabled=True,training=training,steps=steps)
    validate_reset_diagnostic_request(waves,enabled=False,training=training,steps=steps)


def test_frozen_dev_startup_capture_is_allowed():
    validate_reset_diagnostic_request([dict(split='validation')],enabled=True,training=False,steps=1)


def test_passive_roller_probe_removes_spin_only_in_selected_environment():
    class Asset:
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.tensor([[1.,2.],[3.,4.]]),
                joint_vel=torch.tensor([[5.,6.],[7.,8.]]))
            self.velocity_target=torch.full((2,2),9.)
            self.effort_target=torch.full((2,2),10.)
        def write_joint_velocity_to_sim(self,value,env_ids):self.data.joint_vel[env_ids]=value
        def set_joint_velocity_target(self,value,env_ids):self.velocity_target[env_ids]=value
        def set_joint_effort_target(self,value,env_ids):self.effort_target[env_ids]=value
    roller,box=Asset(),Asset()
    env=SimpleNamespace(scene=SimpleNamespace(articulations=dict(rack_roller_deck_02=roller,mb_s2_small_5=box)))
    assert zero_passive_roller_velocities(env,torch.tensor([1]))==['rack_roller_deck_02']
    assert roller.data.joint_pos.tolist()==[[1.,2.],[3.,4.]]
    assert roller.data.joint_vel.tolist()==[[5.,6.],[0.,0.]]
    assert roller.velocity_target.tolist()==[[9.,9.],[0.,0.]]
    assert roller.effort_target.tolist()==[[10.,10.],[0.,0.]]
    assert box.data.joint_vel.tolist()==[[5.,6.],[7.,8.]]


def test_passive_roller_probe_rejects_a_scene_without_roller_decks():
    with pytest.raises(ValueError,match='live rack roller'):
        zero_passive_roller_velocities(SimpleNamespace(scene=SimpleNamespace(articulations={})),torch.tensor([0]))


def test_startup_normal_contact_trace_uses_active_pool_mapping_and_preserves_physics():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import startup_normal_contact_snapshot
    class Scene(dict):
        pass
    scene=Scene()
    for pool in range(2):
        scene[f'box{pool}']=SimpleNamespace(data=SimpleNamespace(
            root_vel_w=torch.full((2,6),float(pool+1)),
            joint_vel=torch.full((2,4),float(pool+2)),
            body_link_vel_w=torch.full((2,5,6),float(pool+3))))
        scene[f'contact{pool}']=SimpleNamespace(body_names=['Body'],data=SimpleNamespace(
            net_forces_w=torch.tensor([[[1.,2.,3.]],[[4.,5.,6.]]])))
    scene['robot_contact']=SimpleNamespace(body_names=['arm','waist'],data=SimpleNamespace(
        net_forces_w=torch.arange(12,dtype=torch.float).reshape(2,2,3)))
    scene.articulations={'rack_roller_deck_02':SimpleNamespace(data=SimpleNamespace(
        joint_vel=torch.tensor([[5.,-7.],[0.,1.]])))}
    env=SimpleNamespace(device='cpu',scene=scene,
        _multi_box_active=torch.tensor([[True,True],[True,False]]),
        _multi_box_pool_ids=torch.tensor([[0,1],[1,-1]]))
    original={name:obj.data.root_vel_w.clone() for name,obj in scene.items() if name.startswith('box')}
    result=startup_normal_contact_snapshot(env,['box0','box1'],['contact0','contact1'],['robot_contact'])
    rows={(r['environment'],r['logical_id']):r for r in result['boxes']}
    assert set(rows)=={(0,0),(0,1),(1,0)}
    assert rows[(1,0)]['original_pool_id']==1
    assert rows[(1,0)]['box_velocity_world']==[2.]*6
    assert rows[(1,0)]['normal_contact_force_world_n']==[4.,5.,6.]
    assert rows[(1,0)]['maximum_absolute_flap_joint_speed_radps']==3.
    assert result['roller_maximum_absolute_speed_radps']['rack_roller_deck_02']==[7.,1.]
    assert result['robot']['robot_contact']['source_bodies']==['arm','waist']
    for name,value in original.items():torch.testing.assert_close(scene[name].data.root_vel_w,value)
    # Preserve nonfinite evidence as null; do not repair physics or claim a
    # vector-summed normal reporter identifies an individual rack collider.
    scene['box1'].data.joint_vel[1,0]=float('nan')
    result=startup_normal_contact_snapshot(env,['box0','box1'],['contact0','contact1'],[])
    rows={(r['environment'],r['logical_id']):r for r in result['boxes']}
    assert rows[(1,0)]['maximum_absolute_flap_joint_speed_radps'] is None
    assert 'collider identity' in result['measurement']
    json.dumps(result,allow_nan=False)
    assert torch.isnan(scene['box1'].data.joint_vel[1,0])


def test_startup_normal_contact_trace_rejects_ambiguous_box_sources():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import startup_normal_contact_snapshot
    sensor=SimpleNamespace(body_names=['Body','flap'],data=SimpleNamespace(net_forces_w=torch.zeros(1,2,3)))
    env=SimpleNamespace(device='cpu',scene={'box':None,'contact':sensor},
        _multi_box_active=torch.tensor([[True]]),_multi_box_pool_ids=torch.tensor([[0]]))
    with pytest.raises(ValueError,match='one box-body reporter'):
        startup_normal_contact_snapshot(env,['box'],['contact'],[])


@pytest.mark.parametrize('change',[
    {'waves':[dict(split='train')]},{'waves':[dict(split='holdout')]},
    {'training':True},{'steps':900},{'reset_enabled':False},{'other_probe':True},
    {'gap_m':float('nan')},{'gap_m':-.001},{'gap_m':.009},
])
def test_rear5_support_gap_cannot_mix_with_training_final_or_other_physics_probes(change):
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import validate_rear5_support_gap_diagnostic_request
    args=dict(waves=[dict(split='validation')],gap_m=0.,reset_enabled=True,
        training=False,steps=1,other_probe=False)
    args.update(change)
    with pytest.raises(ValueError):validate_rear5_support_gap_diagnostic_request(**args)
    args['gap_m']=None
    validate_rear5_support_gap_diagnostic_request(**args)  # existing training path stays opt-in


def test_pair_reporter_extension_keeps_previous_filters_and_collision_config():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import extend_startup_contact_pair_filters
    shared=['belt']
    a,b=(SimpleNamespace(filter_prim_paths_expr=shared,force_threshold=5.,history_length=1) for _ in range(2))
    scene=SimpleNamespace(a=a,b=b,collision_rules=['unchanged'])
    result=extend_startup_contact_pair_filters(scene,['a','b'],['rack','roller','rack','box'])
    assert shared==['belt']
    assert result['a']==result['b']==['belt','rack','roller','box']
    assert a.filter_prim_paths_expr is not b.filter_prim_paths_expr
    assert a.force_threshold==b.force_threshold==5.
    assert scene.collision_rules==['unchanged']


@pytest.mark.parametrize('change',[
    {'waves':[dict(split='train')]},{'waves':[dict(split='holdout')]},
    {'training':True},{'steps':900},{'reset_enabled':False},{'other_probe':True},
    {'solver':'unknown'},
])
def test_reset_solver_probe_rejects_training_final_and_combined_changes_before_mutation(change):
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import configure_reset_solver_probe
    cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/120,physx=SimpleNamespace(solver_type=0)),decimation=4)
    args=dict(waves=[dict(split='validation')],solver='TGS',reset_enabled=True,training=False,steps=1)
    args.update(change)
    with pytest.raises(ValueError):configure_reset_solver_probe(cfg,**args)
    assert cfg.sim.physx.solver_type==0
    args['solver']=None
    assert configure_reset_solver_probe(cfg,**args) is None
    assert cfg.sim.physx.solver_type==0


def test_reset_solver_probe_preserves_timestep_iterations_and_safety_and_marks_no_Q_import():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import configure_reset_solver_probe
    cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/120,physx=SimpleNamespace(solver_type=0,
        min_position_iteration_count=32,min_velocity_iteration_count=8)),decimation=4,
        rack_contact_force=10.,obstacle_contact_force=5.)
    before=json.dumps(vars(cfg.sim.physx),sort_keys=True)
    record=configure_reset_solver_probe(cfg,[dict(split='validation')],solver='TGS',
        reset_enabled=True,training=False,steps=1)
    assert cfg.sim.physx.solver_type==1
    assert cfg.sim.dt==1/120 and cfg.decimation==4
    assert cfg.sim.physx.min_position_iteration_count==32 and cfg.sim.physx.min_velocity_iteration_count==8
    assert cfg.rack_contact_force==10. and cfg.obstacle_contact_force==5.
    assert record['source_solver']=='PGS' and record['requested_solver']=='TGS'
    assert record['frozen_only'] is True and record['Q_import_eligible'] is False
    cfg.sim.physx.solver_type=0
    assert json.dumps(vars(cfg.sim.physx),sort_keys=True)==before


def test_pair_matrix_identity_is_not_the_vector_sum_and_nonfinite_pairs_are_preserved():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import strongest_normal_contact_pairs
    matrix=torch.tensor([[[9.,0.,0.],[-8.,0.,0.],[0.,0.,1.],[0.,0.,0.]],
                         [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]])
    rows=strongest_normal_contact_pairs(matrix,['rack','rear5','belt','idle'],count=2)
    assert [r['target_path'] for r in rows[0]['strongest_normal_pairs']]==['rack','rear5']
    assert [r['normal_force_magnitude_n'] for r in rows[0]['strongest_normal_pairs']]==[9.,8.]
    assert rows[1]['strongest_normal_pairs']==[]
    matrix[1,2,0]=float('nan')
    rows=strongest_normal_contact_pairs(matrix,['rack','rear5','belt','idle'])
    assert rows[1]['nonfinite_filter_count']==1
    assert rows[1]['strongest_normal_pairs'][0]['target_path']=='belt'
    assert rows[1]['strongest_normal_pairs'][0]['normal_force_magnitude_n'] is None
    json.dumps(rows,allow_nan=False)
    assert torch.isnan(matrix[1,2,0])
    with pytest.raises(ValueError,match='filter axis'):
        strongest_normal_contact_pairs(matrix,['ambiguous'])


def test_pair_target_pose_uses_measured_environment_identity_and_rejects_ambiguity():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import attach_contact_target_states
    state=dict(asset='deck2',body='Roller_7',pose=[[1.]*7,[2.]*7],velocity=[[3.]*6,[4.]*6])
    rows=[dict(strongest_normal_pairs=[dict(target_path='/scene/deck2/Roller_7')])]
    lookup={('/scene/deck2','Roller_7'):state}
    attach_contact_target_states(rows,[1],lookup)
    pair=rows[0]['strongest_normal_pairs'][0]
    assert pair['target_pose_world']==[2.]*7 and pair['target_velocity_world']==[4.]*6
    ambiguous={**lookup,('/scene','Roller_7'):state}
    rows=[dict(strongest_normal_pairs=[dict(target_path='/scene/deck2/Roller_7')])]
    attach_contact_target_states(rows,[1],ambiguous)
    assert rows[0]['strongest_normal_pairs'][0]['target_pose_resolution']=='unresolved_or_ambiguous'
    assert 'target_pose_world' not in rows[0]['strongest_normal_pairs'][0]


def test_support_snapshot_detects_fixed_base_or_roller_center_motion_without_physics_writes():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import support_root_snapshot
    pose=torch.zeros(2,3,7);pose[...,3]=1.;velocity=torch.zeros(2,3,6)
    asset=SimpleNamespace(body_names=['Base','r0','r1'],is_fixed_base=True,
        data=SimpleNamespace(body_link_pose_w=pose,body_link_vel_w=velocity))
    env=SimpleNamespace(scene=SimpleNamespace(articulations={'rack_roller_deck_02':asset}))
    initial=support_root_snapshot(env);assert initial['rack_roller_deck_02']['maximum_link_center_displacement_m']==[0.,0.]
    pose[1,2,0]=.2;velocity[1,0,2]=.3
    result=support_root_snapshot(env)['rack_roller_deck_02']
    assert result['is_fixed_base']
    assert result['maximum_link_center_displacement_m'][1]==pytest.approx(.2)
    assert result['base_velocity_world'][1][2]==pytest.approx(.3)
    assert pose[1,2,0]==pytest.approx(.2)


def test_mass_matrix_summary_distinguishes_independent_branches_from_coupling_without_writes():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import summarize_branch_mass_matrix
    matrix=torch.tensor([[[4.e-6,0.],[0.,9.e-6]],[[4.e-6,3.e-6],[3.e-6,9.e-6]]])
    before=matrix.clone()
    result=summarize_branch_mass_matrix(matrix)
    assert result['matrix_shape']==[2,2,2]
    assert result['maximum_normalized_off_diagonal']==pytest.approx([0.,.5])
    assert result['maximum_absolute_off_diagonal']==pytest.approx([0.,3.e-6])
    assert result['invalid_diagonal_count']==[0,0]
    assert torch.equal(matrix,before)


def test_mass_matrix_summary_retains_nonfinite_and_invalid_inertia_without_inventing_coupling():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import summarize_branch_mass_matrix
    matrix=torch.tensor([[[1.,float('nan')],[0.,1.]],[[0.,1.],[1.,2.]]])
    result=summarize_branch_mass_matrix(matrix)
    assert result['maximum_normalized_off_diagonal']==[None,None]
    assert result['nonfinite_entry_count']==[1,0]
    assert result['invalid_diagonal_count']==[0,1]
    json.dumps(result,allow_nan=False)
    assert torch.isnan(matrix[0,0,1]) and matrix[1,0,0]==0.
    with pytest.raises(ValueError):summarize_branch_mass_matrix(torch.zeros(2,3,4))


def test_support_dynamics_audit_reads_actual_view_parameters_and_keeps_other_assets_untouched():
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import startup_support_dynamics_snapshot
    view=SimpleNamespace(get_masses=lambda:torch.tensor([[1.,.05,.05]]),
        get_inertias=lambda:torch.eye(3).reshape(1,1,9).expand(1,3,9),
        get_dof_armatures=lambda:torch.zeros(1,2),get_dof_dampings=lambda:torch.full((1,2),.00002),
        get_dof_stiffnesses=lambda:torch.zeros(1,2),
        get_dof_max_forces=lambda:torch.full((1,2),.05),
        get_generalized_mass_matrices=lambda:torch.diag(torch.tensor([4.e-6,9.e-6]))[None])
    asset=SimpleNamespace(body_names=['Base','r0','r1'],joint_names=['r0_joint','r1_joint'],
        is_fixed_base=True,root_physx_view=view)
    env=SimpleNamespace(scene=SimpleNamespace(articulations={'rack_roller_deck_02':asset,'robot':None}))
    result=startup_support_dynamics_snapshot(env)['rack_roller_deck_02']
    assert result['environment0_body_mass_kg']==pytest.approx([1.,.05,.05])
    assert result['generalized_mass']['maximum_normalized_off_diagonal']==[0.]
    assert result['physics_parameters_written'] is False
    assert result['environment0_joint_stiffness']==[0.,0.]
    assert result['environment0_joint_max_force_nm']==pytest.approx([.05,.05])


@pytest.mark.parametrize('field',('damping','stiffness','max_force'))
def test_bearing_verifier_rejects_fallback_zero_damping_and_bad_gains_or_limits(field):
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import verify_passive_bearing_drives
    values=dict(damping=torch.full((2,3),.00002),stiffness=torch.zeros(2,3),max_force=torch.full((2,3),.05))
    view=SimpleNamespace(get_dof_dampings=lambda:values['damping'],
        get_dof_stiffnesses=lambda:values['stiffness'],get_dof_max_forces=lambda:values['max_force'])
    env=SimpleNamespace(scene=SimpleNamespace(articulations={'rack_roller_deck_02':SimpleNamespace(root_physx_view=view)}))
    assert verify_passive_bearing_drives(env,.00002)['rack_roller_deck_02']['instances']==2
    values[field][1,2]=0. if field=='damping' else float('inf')
    with pytest.raises(ValueError,match=field):verify_passive_bearing_drives(env,.00002)

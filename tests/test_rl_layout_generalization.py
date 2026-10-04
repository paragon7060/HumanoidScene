"""Retargeted goal context must specify the actual residual-action MDP."""
import torch
import pytest

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import (
    GraspLayout, matrix6, retarget_reference_rack, sample_layout, yaw_matrix,
    layout_reset_observation, validate_layout_footprints,
    base_reset_observation,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import RetargetedGoalResidual


def source():
    raw=torch.zeros(464); raw[439]=1
    raw[68:71]=torch.tensor([.6,.2,0.])
    raw[71:77]=matrix6(torch.eye(3))
    raw[86+4*22]=1;raw[86+4*22+3]=1
    raw[86+4*22+12:86+4*22+15]=torch.tensor([.8,.1,1.])
    raw[86+4*22+15:86+4*22+21]=matrix6(torch.eye(3))
    raw[388+4]=1;raw[400+4]=1  # mask + selected target
    return raw


def test_train_and_holdout_are_reproducible_separate_layouts():
    assert sample_layout(42,'train')==sample_layout(42,'train')
    assert sample_layout(42,'train')!=sample_layout(42,'holdout')
    for split in ('train','holdout'):
        for i in range(16):
            sample_layout(i,split).validate()
            assert -.04 <= sample_layout(i,split).lateral_m <= -.02


def rack_seed():
    actor=source()
    tokens=actor[86:350].reshape(12,22)
    tokens[4,8+1]=1
    tokens[4,12:15]=actor[68:71]+torch.tensor([-.24,-.21,1.])
    return actor


def test_sampler_fits_target_and_surrounding_box_footprints():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    for split in ('train','holdout'):
        for i in range(32):
            actor=layout_reset_observation(rack_seed(),sample_layout(i,split),MultiBoxSpec())
            validate_layout_footprints(actor)


def test_native_multi_box_seed_uses_selected_target_and_replaces_only_background():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    spec=MultiBoxSpec()
    measured=layout_reset_observation(rack_seed(),GraspLayout(1,'train',-.02,distractors=(5,6)),spec)
    saved=measured.clone()
    moved=layout_reset_observation(measured,GraspLayout(2,'train',0.,distractors=(9,)),spec)
    before=measured[86:350].reshape(12,22);after=moved[86:350].reshape(12,22)
    assert torch.equal(measured,saved)  # Actual training rows are never rewritten.
    assert torch.equal(after[4],before[4])
    assert torch.where(after[:,0]>.5)[0].tolist()==[4,9]
    assert torch.equal(moved[388:400],after[:,0])
    assert torch.equal(moved[400:412],measured[400:412])
    validate_layout_footprints(moved)


@pytest.mark.parametrize('bad', ['missing','ambiguous','weighted','inactive','unmasked'])
def test_layout_reset_rejects_invalid_selected_target_even_with_other_boxes(bad):
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    measured=rack_seed()
    if bad=='missing':measured[400:412]=0
    elif bad=='ambiguous':measured[405]=1
    elif bad=='weighted':measured[404]=.9;measured[405]=.1
    elif bad=='inactive':measured[86+4*22]=0
    else:measured[388+4]=0
    with pytest.raises(ValueError,match='selected'):
        layout_reset_observation(measured,GraspLayout(1,'train',0.),MultiBoxSpec())


def test_explicit_upper_layout_preserves_target_and_moves_actual_start_frame():
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
    from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    dataset=Path(__file__).resolve().parents[1]/'examples/demos/v2_grasp_quest_success.hdf5'
    batch,_=load_v2_grasp_demonstrations(dataset,self_collision_enabled=False)
    original=select_reference_episode(batch,1)['actor_obs'][0]
    layout=GraspLayout(1000,'train',-.025,.007,(5,),base_lateral_m=-.04,
                       base_outward_m=.06,base_yaw_rad=-.03)
    moved=layout_reset_observation(original,layout,MultiBoxSpec())
    tokens=moved[86:350].reshape(12,22)
    assert torch.where(tokens[:,0]>.5)[0].tolist()==[5,9]
    assert int(moved[400:412].argmax())==9
    assert torch.equal(tokens[9,3:12],original[86:350].reshape(12,22)[9,3:12])
    assert not torch.equal(moved[68:77],original[68:77])
    validate_layout_footprints(moved)
    with pytest.raises(ValueError,match='distractor cannot replace'):
        layout_reset_observation(original,GraspLayout(1000,'train',-.025,distractors=(9,)),MultiBoxSpec())


def test_half_shelf_crossing_is_rejected_before_physics():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    with pytest.raises(ValueError,match='footprint exceeds'):
        layout_reset_observation(rack_seed(),GraspLayout(1,'probe',-.05,-.0077),MultiBoxSpec())


def test_depth_distribution_moves_target_in_rack_frame_and_fits_footprints():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    seed=rack_seed()
    before=seed[86+4*22+12:86+4*22+15]
    for split in ('train','holdout'):
        for index in range(32):
            layout=sample_layout(index,split,depth_limit_m=.01)
            assert abs(layout.depth_m)<=.01
            actor=layout_reset_observation(seed,layout,MultiBoxSpec())
            delta=actor[86+4*22+12:86+4*22+15]-before
            assert torch.allclose(delta,torch.tensor([layout.lateral_m,layout.depth_m,0.]),atol=1e-6)
            validate_layout_footprints(actor)
    reference=torch.stack([seed,seed]);current=seed.clone()
    current[86+4*22+13]+=.01
    retargeted,report=retarget_reference_rack(reference,seed,current)
    assert torch.allclose(retargeted[:,69],reference[:,69]-.01,atol=1e-6)
    assert abs(report['target_shift_rack_m'][1]-.01)<1e-6


def test_reference_retarget_is_translation_equivariant_and_goal_visible():
    original=source();current=original.clone()
    current[86+4*22+12]+=.04
    reference=torch.stack([original,original])
    result,report=retarget_reference_rack(reference,original,current)
    assert torch.allclose(result[:,68],torch.full((2,),.56))
    assert abs(report['target_shift_rack_m'][0]-.04)<1e-6
    controller=RetargetedGoalResidual(dict(action=torch.zeros(1,24),
        actor_obs=original[None],next_actor_obs=original[None]))
    controller.reference_actor=result
    ao,co=controller.observations(current[None],torch.zeros(1,530),0)
    assert ao.shape==(1,492) and co.shape==(1,584)
    assert torch.equal(ao[:,-29:],co[:,-29:])
    physical=controller.physical_commands(torch.zeros(1,22),0,current[None])
    assert abs(float(physical[0,0])-.04*2/.15)<1e-5  # executes the move with bounded position feedback


def test_yaw_retarget_keeps_box_to_reference_robot_relation():
    original=source();current=original.clone();rotation=yaw_matrix(.03,original)
    current[86+4*22+15:86+4*22+21]=matrix6(rotation)
    result,_=retarget_reference_rack(original[None],original,current)
    rack_r=_rotation_matrix(result[0,71:77])
    assert torch.allclose(rack_r,rotation.T,atol=1e-6)
    target=current[86+4*22+12:86+4*22+15]
    box_in_rack=target-original[68:71]
    expected=result[0,68:71]+rack_r@box_in_rack
    assert torch.allclose(expected,target,atol=1e-6)


def test_retargeted_arm_residual_has_useful_goal_range_without_accumulation():
    raw=source()[None].repeat(2,1)
    controller=RetargetedGoalResidual(dict(action=torch.zeros(1,24),actor_obs=raw[:1],next_actor_obs=raw[:1]))
    residual=torch.zeros(1,22);residual[0,4]=.1
    first=controller.physical_commands(residual,0,raw[:1])
    assert abs(float(first[0,4])-.15)<1e-6  # .003rad / .02rad-per-command
    actual=raw[:1].clone();actual[:,420]=.003
    held=controller.physical_commands(residual,1,actual)
    assert held[0,4].abs()<1e-6


def test_actor_sees_changed_surrounding_box_with_identical_target_and_robot():
    raw=source()[None]
    controller=RetargetedGoalResidual(dict(action=torch.zeros(1,24),actor_obs=raw,next_actor_obs=raw))
    other=raw.clone();other[:,86+6*22]=1;other[:,86+6*22+12]=.4
    original,_=controller.observations(raw,torch.zeros(1,530),0)
    changed,_=controller.observations(other,torch.zeros(1,530),0)
    assert torch.equal(original[:,:174],changed[:,:174])
    assert not torch.equal(original,changed)


def test_layout_residual_cannot_enter_the_ordinary_physical_action_q_trainer(tmp_path):
    import json
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint
    (tmp_path/'manifest.json').write_text(json.dumps(dict(artifact_type='layout_reference_residual_sac')))
    with pytest.raises(ValueError,match='different contextual action space'):
        _compatible_checkpoint(tmp_path/'checkpoint.pt',{},data_only=True)


def test_base_initialization_preserves_all_box_poses_in_rack_frame():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    original=layout_reset_observation(rack_seed(),sample_layout(0,'train'),MultiBoxSpec())
    moved=base_reset_observation(original,.12,.18,.15)
    r0=_rotation_matrix(original[71:77]);r1=_rotation_matrix(moved[71:77])
    a=original[86:350].reshape(12,22);b=moved[86:350].reshape(12,22)
    for i in torch.where(a[:,0]>.5)[0]:
        assert torch.allclose(r0.T@(a[i,12:15]-original[68:71]),
                              r1.T@(b[i,12:15]-moved[68:71]),atol=1e-6)
        assert torch.allclose(r0.T@_rotation_matrix(a[i,15:21]),
                              r1.T@_rotation_matrix(b[i,15:21]),atol=1e-6)
    # Actual starting base in rack coordinates changed by the requested offset.
    delta=-(r1.T@moved[68:71])+(r0.T@original[68:71])
    assert torch.allclose(delta,torch.tensor([.12,.18,0.]),atol=1e-6)
    assert torch.equal(original[:20],moved[:20])
    validate_layout_footprints(moved)


def test_invalid_base_randomization_cannot_be_silently_clipped():
    with pytest.raises(ValueError,match='translation'):
        base_reset_observation(rack_seed(),.30,0.,0.)
    with pytest.raises(ValueError,match='yaw'):
        base_reset_observation(rack_seed(),0.,0.,.5)


@pytest.mark.parametrize('episode,region,target',[(0,'shelf_2_right',1),(1,'shelf_3_right',6)])
def test_region_reset_uses_real_right_pool_and_preserves_box_and_neutral_robot(episode,region,target):
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations, _rotation_matrix
    from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
    from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells, physical_pool_id
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    from kuavo_isaaclab_scene.workcell.workcell_layout import scale, RACK_SHELF_CENTER_LOCAL_X_RAW
    batch,_=load_v2_grasp_demonstrations(Path(__file__).resolve().parents[1]/
        'examples/demos/v2_grasp_quest_success.hdf5',self_collision_enabled=False)
    original=select_reference_episode(batch,episode)['actor_obs'][0]
    saved=original.clone();spec=MultiBoxSpec()
    layout=GraspLayout(5800+episode,'holdout',-.025,depth_m=.004,target_region=region)
    moved=layout_reset_observation(original,layout,spec)
    assert torch.equal(original,saved)
    assert torch.equal(original[:68],moved[:68])
    assert torch.equal(original[68:86],moved[68:86])  # Base did not silently move.
    old=original[86:350].reshape(12,22)[int(original[400:412].argmax())]
    tokens=moved[86:350].reshape(12,22);new=tokens[target]
    assert torch.where(tokens[:,0]>.5)[0].tolist()==[target]
    assert int(moved[400:412].argmax())==target
    assert int(new[8:12].argmax())==target//3
    assert physical_pool_id(logical_cells(spec)[target],0)==(1 if episode==0 else 12)
    assert torch.equal(old[3:8],new[3:8]) and torch.allclose(old[15:21],new[15:21],atol=1e-6)
    r=_rotation_matrix(original[71:77]);before=r.T@(old[12:15]-original[68:71])
    after=r.T@(new[12:15]-moved[68:71])
    expected=before.clone();expected[0]=2*RACK_SHELF_CENTER_LOCAL_X_RAW*scale('rack')[0]-before[0]+.025
    expected[1]+=.004
    assert torch.allclose(after,expected,atol=1e-6)
    validate_layout_footprints(moved)
    # Explicit nominal region alignment moves the real starting robot, then
    # applies independent random offsets; every world box stays on the rack.
    aligned=layout_reset_observation(original,GraspLayout(5900+episode,'train',-.025,
        depth_m=.004,target_region=region,align_initial_base_to_region=True,
        base_lateral_m=.12,base_outward_m=.16,base_yaw_rad=.10),spec)
    ra=_rotation_matrix(aligned[71:77]);ta=aligned[86:350].reshape(12,22)[target]
    assert torch.allclose(ra.T@(ta[12:15]-aligned[68:71]),after,atol=1e-6)
    nominal=float(2*(RACK_SHELF_CENTER_LOCAL_X_RAW*scale('rack')[0]-before[0]))
    root_delta=-(ra.T@aligned[68:71])+(r.T@original[68:71])
    assert torch.allclose(root_delta,original.new_tensor([nominal+.12,.16,0.]),atol=1e-6)
    assert torch.equal(original[:20],aligned[:20])
    validate_layout_footprints(aligned)
    if episode==1:
        with pytest.raises(ValueError,match='distractor cannot replace'):
            layout_reset_observation(original,GraspLayout(5901,'probe',-.025,
                target_region=region,distractors=(6,)),spec)


def test_region_reset_rejects_cross_shelf_and_target_distractor_alias():
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    with pytest.raises(ValueError,match='preserve the measured shelf'):
        layout_reset_observation(rack_seed(),GraspLayout(1,'probe',-.025,
            target_region='shelf_3_right'),MultiBoxSpec())
    with pytest.raises(ValueError,match='explicit target region'):
        GraspLayout(1,'probe',0.,align_initial_base_to_region=True).validate()
    # Legacy serialized recipes and transformations remain unchanged.
    assert 'target_region' not in sample_layout(1,'train').record()


@pytest.mark.parametrize('episode,region,distractors',[(0,'shelf_2_left',(5,6,9)),
    (0,'shelf_2_right',(5,6,9)),(1,'shelf_3_left',(5,6)),(1,'shelf_3_right',(5,9))])
def test_background_packing_probe_preserves_original_target_and_initial_robot(episode,region,distractors):
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
    from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
    from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import packed_background_reset_observation
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
    from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import DEPTH_GAP_M,logical_cells
    dataset=Path(__file__).resolve().parents[1]/'examples/demos/v2_grasp_quest_success.hdf5'
    batch,_=load_v2_grasp_demonstrations(dataset,self_collision_enabled=False)
    source=select_reference_episode(batch,episode)['actor_obs'][0]
    spec=MultiBoxSpec()
    original=layout_reset_observation(source,GraspLayout(1,'probe',-.025,.007,distractors,
        base_lateral_m=.04,base_outward_m=.07,base_yaw_rad=.05,
        target_region=region,align_initial_base_to_region=True),spec,roller_clearance_m=.01)
    unchanged=original.clone()
    packed=packed_background_reset_observation(original,spec,roller_clearance_m=.01)
    before=original[86:350].reshape(12,22);after=packed[86:350].reshape(12,22)
    target=int(original[400:412].argmax())
    assert torch.equal(original,unchanged)
    assert torch.equal(before[target],after[target])
    assert torch.equal(original[:86],packed[:86])
    assert torch.equal(original[350:],packed[350:])
    assert torch.equal(before[:,:12],after[:,:12])
    rotation=_rotation_matrix(packed[71:77])
    def depth(token):return -float((rotation.T@(token[12:15]-packed[68:71]))[1])
    assert depth(after[5])<depth(before[5])
    if logical_cells(spec)[target].region_id==logical_cells(spec)[5].region_id:
        assert depth(after[5])-depth(after[target])>=float(after[5,6]+after[target,6])/2+DEPTH_GAP_M-1e-6
    validate_layout_footprints(packed)

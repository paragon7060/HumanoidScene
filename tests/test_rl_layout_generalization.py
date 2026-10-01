"""Retargeted goal context must specify the actual residual-action MDP."""
import torch
import pytest

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import (
    GraspLayout, matrix6, retarget_reference_rack, sample_layout, yaw_matrix,
    layout_reset_observation, validate_layout_footprints,
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

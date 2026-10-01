"""Absolute goal labels must reproduce commands and stay separate from Q data."""
import json
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import matrix6,yaw_matrix


def raw_state():
    raw=torch.zeros(2,464);raw[:,439]=1
    raw[:,71:77]=matrix6(yaw_matrix(.3,raw)).expand(2,-1)
    raw[:,68:71]=torch.tensor([.6,.2,.7])
    raw[:,:20]=torch.linspace(-.2,.2,20)
    raw[:,416:436]=.003
    return raw


def test_absolute_goal_inverse_reproduces_executed_commands():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    generator=torch.Generator().manual_seed(5)
    physical=torch.rand(2,24,generator=generator)*2-1
    goals=coordinates.encode_physical(raw,physical)
    assert torch.allclose(coordinates.decode(raw,goals),physical,atol=1e-5)
    current=coordinates.encode_physical(raw,torch.zeros_like(physical))
    assert coordinates.decode(raw,current).abs().max()<1e-5


def test_student_input_uses_observations_and_clock_without_reference_goals():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    inputs=coordinates.observations(raw,0)
    assert inputs.shape==(2,439) and inputs[:,-1].eq(0).all()
    assert coordinates.observations(raw,900)[:,-1].eq(1).all()
    other=raw.clone();other[:,86+6*22]=1;other[:,86+6*22+12]=.4
    assert not torch.equal(inputs,coordinates.observations(other,0))


def test_time_encoding_adds_only_clock_features():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    features=coordinates.observations(raw,205,16)
    assert features.shape==(2,471)
    assert torch.allclose(features[:,:439],coordinates.observations(raw,205))
    assert features[:,439:455].abs().max()<1e-5
    assert torch.allclose(features[:,455:471],torch.tensor([(-1.)**i for i in range(1,17)]).expand(2,-1))


def test_goal_projection_matches_physical_close_gate_and_entropy_mask():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
    from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import GraspActionProjector
    coordinates=PoseGoalCoordinates();raw=raw_state()
    raw[:,386]=1 # left assigned flap0; other hand uses flap1
    raw[:,350]=.1;raw[:,377]=.3 # left near; right far
    physical=torch.zeros(2,24);physical[:,20:22]=1
    projected=GraspActionProjector([('body',20),('left_gripper',1),('right_gripper',1),('head',2)])(raw,physical)
    inputs=coordinates.observations(raw,0)
    z=physical.clone();z[:,22:24]=physical[:,20:22]
    goal_projector=GoalGripperProjector()
    assert torch.equal(goal_projector(inputs,z)[:,22:24],projected[:,20:22])
    assert torch.equal(goal_projector.entropy_mask(inputs)[:,22:24],torch.tensor([[1.,0.],[1.,0.]]))


def test_anchor_context_does_not_change_warm_start_predictions():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import widen_bc_actor,GoalGripperProjector
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    config=SACConfig(hidden=16)
    original=AsymmetricSAC(439,531,24,config)
    original.actor_normalizer.update(torch.randn(10,439)*.4+1.2)
    state=original.checkpoint()|dict(artifact_type=PoseStudent.artifact_type,
        action_coordinates=PoseGoalCoordinates.name,goal_center=torch.zeros(24),goal_scale=torch.ones(24))
    student=PoseStudent(state)
    new=AsymmetricSAC(441,533,24,config,action_projector=GoalGripperProjector())
    widen_bc_actor(student,new)
    raw=PoseGoalCoordinates().observations(raw_state(),51)
    expanded=torch.cat((raw,torch.randn(2,2)),-1)
    a=original.actor(original.actor_normalizer(raw),deterministic=True)[0]
    b=new.actor(new.actor_normalizer(expanded),deterministic=True)[0]
    assert torch.allclose(a,b,atol=1e-7)
    assert torch.equal(new.actor_normalizer.count,original.actor_normalizer.count)


@pytest.mark.parametrize('artifact',['pose_goal_student_BC_diagnostic_NOT_SAC','pose_goal_sac_no_live_reference'])
def test_ordinary_delta_trainer_rejects_pose_goal_artifacts(tmp_path,artifact):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint
    (tmp_path/'manifest.json').write_text(json.dumps({'artifact_type':artifact}))
    with pytest.raises(ValueError,match='separate absolute action coordinates'):
        _compatible_checkpoint(tmp_path/'checkpoint.pt',{},data_only=True)

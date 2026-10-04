"""Prevent cross-environment clock/waypoint leakage and reset-seam replay."""
from types import SimpleNamespace
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import pose_clock
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import staged_context,held_goal_coordinates
from test_rl_staged_base_hold import scene,Coordinates


def test_wave_reset_removes_old_wrench_and_effort_only_in_selected_environment():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import reset_wave_controller_state
    class Asset:
        num_joints=2
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.ones(2,2))
            self.velocity=torch.full((2,2),7.)
            self.effort=torch.full((2,2),9.)
            self.permanent_wrench_composer=SimpleNamespace(
                composed_force_as_torch=torch.full((2,1,3),5.),
                composed_torque_as_torch=torch.full((2,1,3),11.))
        def set_joint_velocity_target(self,x,env_ids):self.velocity[env_ids]=x
        def set_joint_effort_target(self,x,env_ids):self.effort[env_ids]=x
    robot,box=Asset(),Asset()
    class Scene(dict):
        def reset(self,ids):
            for a in self.values():
                a.permanent_wrench_composer.composed_force_as_torch[ids]=0
                a.permanent_wrench_composer.composed_torque_as_torch[ids]=0
    env=SimpleNamespace(scene=Scene(robot=robot,box=box))
    audit=reset_wave_controller_state(env,[robot,box],torch.tensor([1]))
    assert audit['before']['force_n'][0]>0 and audit['before']['torque_nm'][0]>0
    assert audit['after_scene_reset']==dict(force_n=[0.],torque_nm=[0.])
    for a in (robot,box):
        assert a.velocity.tolist()==[[7.,7.],[0.,0.]]
        assert a.effort.tolist()==[[9.,9.],[0.,0.]]
        assert a.data.joint_pos.eq(1).all()
        assert a.permanent_wrench_composer.composed_force_as_torch[0].eq(5).all()


def test_neutral_settling_uses_new_support_every_physics_step_without_replay():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import settle_neutral_wave_controllers
    # Old support was for a different body-frame gravity orientation. The
    # simulated new pose needs 2 N. Reusing 100 N, or stepping after clearing
    # support without applying the controller, moves this otherwise held root.
    state=SimpleNamespace(wrench=100.,position=0.,written=None,applied=0,processed=0)
    def process(action):
        assert action.eq(0).all();state.processed+=1
    def apply():state.wrench=2.;state.applied+=1
    def write():state.written=state.wrench
    def step(render):
        assert not render
        state.position+=state.written-2.
        state.wrench=100.  # each substep requires a fresh controller write
    env=SimpleNamespace(physics_dt=.01,
        action_manager=SimpleNamespace(action=torch.ones(2,24),process_action=process,apply_action=apply),
        scene=SimpleNamespace(write_data_to_sim=write,update=lambda dt:None),
        sim=SimpleNamespace(step=step))
    settle_neutral_wave_controllers(env,steps=5)
    assert state.position==0. and state.applied==5 and state.processed==1


def test_development_guard_detects_regional_loss_even_when_total_success_rises():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import DevelopmentSuccessGuard
    guard=DevelopmentSuccessGuard()
    layouts=[dict(episode_index=0,layout=dict(seed=i,target_region=r))
             for i,r in enumerate(('left','left','right','right'))]
    first=guard.evaluate(layouts,[dict(success=x) for x in (True,False,False,False)],0)
    assert not first['regression']
    check=guard.evaluate(layouts,[dict(success=x) for x in (False,False,True,True)],2)
    assert check['regression'] and check['best_wave']==0
    assert check['best_by_region']['left']['successes']==1
    assert not check['final_outcomes_used']
    with pytest.raises(ValueError,match='identical initial cases'):
        guard.evaluate([dict(episode_index=0,layout=dict(seed=99,target_region='left'))]*4,
                       [dict(success=True)]*4,4)
    initial_floor=DevelopmentSuccessGuard(.5).evaluate(layouts,
        [dict(success=x) for x in (True,True,False,False)],0)
    assert initial_floor['baseline_failed'] and not initial_floor['regression']


def test_feedback_rate_contract_checks_every_vector_environment():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
    base=torch.tensor([[.15,.15,.5]]).repeat(4,1)
    upper=torch.tensor([[.01]+[.02]*14]).repeat(4,1)
    validate_goal_feedback_rates(base,upper,torch.full((4,2),.01),.1,1/30)
    upper[3,4]=.03
    with pytest.raises(ValueError,match='rates differ'):
        validate_goal_feedback_rates(base,upper,.01,.1,1/30)


def test_different_physical_settling_times_have_independent_anchors_and_clocks():
    raw,templates=scene();raw=raw.repeat(3,1)
    raw[:,98]=torch.tensor([-.2,.1,.35])
    stages=BatchedBaseStages(Coordinates(),templates,raw)
    raw[:,20]=raw[:,98];raw[:,21]=.7
    zero=torch.zeros(3,3);active=torch.ones(3,dtype=torch.bool)
    for step in range(15):
        velocity=zero.clone();velocity[1,0]=.03
        held=stages.update(raw,velocity,zero,active,step)
    assert held.tolist()==[0,2]
    with pytest.raises(ValueError,match='unconfirmed'):stages.held_context(torch.tensor([1]))
    # The measured initial box anchor stays fixed even if its box later moves.
    raw[0,98]=9
    active[2]=False
    for step in range(15,30):held=stages.update(raw,zero,zero,active,step)
    assert held.tolist()==[0,1]
    torch.testing.assert_close(stages.anchors[:2,0],torch.tensor([-.2,.1]))
    assert stages.clocks(held,29).tolist()==[15,0]
    context=staged_context(raw[held],stages.held_context(held),.06)
    torch.testing.assert_close(context[:,1],torch.tensor([-.2,.1]))
    assert context[:,0].eq(1).all()


def test_per_environment_clock_and_yaw_match_individual_decoding():
    raw=torch.zeros(2,464);times=torch.tensor([0,700])
    torch.testing.assert_close(pose_clock(raw,times,594,593),torch.tensor([[0.],[593/594]]))
    for bad in (torch.tensor([-1,0]),torch.tensor([0]),torch.tensor([float('nan'),0])):
        with pytest.raises(ValueError,match='elapsed clock'):pose_clock(raw,bad,594)
    class Decoder:
        def decode(self,raw,goal):return goal
    stage=SimpleNamespace(phase='held_grasp',manipulation_start=True,
        target_xy=torch.tensor([[-.2,.7],[.1,.6]]),target_yaw=torch.tensor([.2,-.3]))
    requested=torch.randn(2,21)
    batched=held_goal_coordinates(Decoder(),raw,requested,stage)
    for i in range(2):
        single=SimpleNamespace(target_xy=stage.target_xy[i:i+1],target_yaw=float(stage.target_yaw[i]))
        torch.testing.assert_close(batched[i:i+1],held_goal_coordinates(Decoder(),raw[i:i+1],requested[i:i+1],single))
    context=staged_context(raw,stage,.05)
    torch.testing.assert_close(context[:,3],stage.target_yaw.sin())

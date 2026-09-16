"""Transfer must be physical, phase gated, and cannot renew progress on regrasp."""
from types import SimpleNamespace as NS
import pytest
import torch
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec
from kuavo_isaaclab_scene.rl.mdp.transfer import TransferProgress, rack_clearance
from test_rl_flap_rewards import modules


def test_pick_place_enables_collision_and_uses_tight_workspace():
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/rl_pick_place.py",
                 root / "src/kuavo_isaaclab_scene/configs/rl_pick_place.py"):
        configured = runpy.run_path(str(path))["configure_task"](task_spec("pick"))
        configured.validate()
        assert configured.collision_constraints_enabled
        assert configured.workspace_radius == 1.5


def test_whole_footprint_must_leave_rack():
    center=torch.tensor([[.6,0.,0.],[.73,0.,0.],[0.,.9,0.]])
    half=torch.full_like(center,.2)
    clear,left=rack_clearance(center,half,torch.zeros(3),torch.tensor([.5,.5,1.]),.025)
    assert clear.tolist()==[False,True,True]
    assert left[0].item()==pytest.approx(.125)
    # Bigger projection from rotating a box can put its corner back in the rack.
    half[1,0]=.3
    assert not rack_clearance(center,half,torch.zeros(3),torch.tensor([.5,.5,1.]),.025)[0][1]


def test_signed_progress_cancels_reversal_and_rebases_on_loss_phase_and_reset():
    p=TransferProgress(2,'cpu');phase=torch.tensor([2,2]);on=torch.ones(2,dtype=torch.bool)
    def step(x,d,enabled=on):
        p.advance(torch.tensor(x),torch.tensor(d),phase,enabled,on)
        return p.delta.clone()
    assert not step([.2,.2],[1.,1.]).any()
    torch.testing.assert_close(step([.1,.1],[.8,.8]),torch.tensor([[.1,.2],[.1,.2]]))
    torch.testing.assert_close(step([.2,.2],[1.,1.]),torch.tensor([[-.1,-.2],[-.1,-.2]]))
    assert not step([0.,0.],[0.,0.],~on).any()
    assert not step([0.,0.],[0.,0.]).any()  # reacquisition never credits the gap
    phase[0]=3
    assert not step([0.,0.],[0.,0.]).any()
    p.reset(torch.tensor([0]));assert p.ready.tolist()==[False,True]


def test_phase_chain_requires_extraction_support_release_and_stability(modules):
    commands,rewards=modules;n=2
    z=lambda:torch.zeros(n);f=lambda:torch.zeros(n,dtype=torch.bool)
    t=NS(spec=task_spec('pick_place',hold_seconds=.2,lift_height=.06,cargo_per_box=0,
                       collision_constraints_enabled=False),settling=None,transfer_progress=TransferProgress(n,'cpu'),
        ids=torch.arange(n),last_step=torch.full((n,),-1),phase=torch.ones(n,dtype=torch.long),
        reward_phase=torch.ones(n,dtype=torch.long),active_box=torch.zeros(n,dtype=torch.long),
        reward_box=torch.zeros(n,dtype=torch.long),transition=f(),
        centers=torch.tensor([[[0.,0.,.3]],[[0.,0.,.3]]]),initial_z=torch.full((n,1),.2),
        upright=torch.ones(n,1),nav_distance=torch.full((n,),100.),heading_error=z(),
        grasped=torch.tensor([True,False]),velocities=torch.zeros(n,1,6),flap_grasp=object(),
        reach_progress=None,unexpected_finger_force=torch.zeros(n,4),obstacle_forces=torch.zeros(n,2),
        supported=torch.zeros(n,1,dtype=torch.bool),released=f(),free_slots=torch.ones(n,1,dtype=torch.bool),
        button_pressed=f(),tools=torch.zeros(n,2,3),button_point=torch.zeros(n,3),
        cargo_ok=torch.ones(n,1,dtype=torch.bool),dwell=z(),success=f(),failure=f(),
        belt_running=f(),belt_time=z(),cfg=NS(collision_force=200),
        robot=NS(data=NS(root_pos_w=torch.zeros(n,3))),hand_grasp_flags=torch.zeros(n,2,dtype=torch.bool),
        hand_target_distance=torch.zeros(n,2),_measure=lambda:None,_goals=lambda:None,
        slot_goal=torch.tensor([[.5,0.,.4],[.5,0.,.4]]),belt_half=torch.full((n,1,3),.1),
        extracted=torch.zeros(n,1,dtype=torch.bool),extract_remaining=torch.full((n,1),.2),
        conveyor_support_force=torch.zeros(n,1),done_boxes=torch.zeros(n,1,dtype=torch.bool))
    t.poses=torch.cat((t.centers,torch.tensor([1.,0.,0.,0.]).expand(n,1,4)),-1)
    scene=type('Scene',(dict,),{})();scene.env_origins=torch.zeros(n,3)
    t._env=NS(common_step_counter=0,scene=scene,step_dt=.1,episode_length_buf=torch.full((n,),10))
    t.metrics={name:z() for name in ('success','boxes_placed','phase','cargo_retained','grasp_left',
        'grasp_right','lift_height','hold_fraction','left_target_distance','right_target_distance')}
    def step():
        t._env.common_step_counter+=1;commands.WorkcellCommand.refresh(t)
    step();step()
    assert t.phase.tolist()==[2,1] and not t.success.any()  # lift is a stage, not terminal
    t.centers[0,0]=torch.tensor([.5,0.,.5])
    step();assert t.dwell[0]==0  # above target is insufficient without rack clearance
    t.extracted[0]=True;t.extract_remaining[0]=0
    step();step();assert t.phase.tolist()==[3,1] and not t.success.any()
    t.supported[0]=True;t.grasped[0]=False
    t.released[0]=True;step();assert t.dwell[0]==0  # floating geometric support rejected
    t.conveyor_support_force[0]=1.;t.released[0]=False
    step();assert t.dwell[0]==0
    t.released[0]=True;t.velocities[0,0,0]=.2
    step();assert t.dwell[0]==0
    t.velocities[0]=0;step();step()
    assert t.success.tolist()==[True,False] and t.done_boxes.tolist()==[[True],[False]]
    assert not t.belt_running.any() and t.phase.tolist()==[3,1]
    before=t.dwell.clone();commands.WorkcellCommand.refresh(t)
    torch.testing.assert_close(t.dwell,before)


def test_transfer_reward_is_phase_gated_and_dt_independent(modules):
    _,rewards=modules
    t=NS(spec=task_spec('pick_place'),settling=None,refresh=lambda:None,
         reward_phase=torch.tensor([2,3]),transfer_progress=NS(delta=torch.tensor([[.1,.2],[.3,.4]])))
    env=NS(command_manager=NS(get_term=lambda _:t))
    for dt in (1/30,1/60):
        env.step_dt=dt
        torch.testing.assert_close(rewards.extraction_progress(env)*dt,torch.tensor([.1,0.]))
        torch.testing.assert_close(rewards.transfer_progress(env)*dt,torch.tensor([.2,0.]))
        torch.testing.assert_close(rewards.placement_progress(env)*dt,torch.tensor([0.,.4]))


def test_prepick_shelf_disturbance_does_not_return_on_lower_conveyor(modules, monkeypatch):
    _,rewards=modules
    t=NS(spec=task_spec('pick_place'),refresh=lambda:None,reward_phase=torch.tensor([1,2,3]))
    env=NS(command_manager=NS(get_term=lambda _:t))
    monkeypatch.setattr(rewards,'_grasp_stability',lambda _: (torch.ones(3),torch.zeros(3)))
    torch.testing.assert_close(rewards.prelift_disturbance(env),torch.tensor([1.,0.,0.]))


@pytest.mark.parametrize('overrides', [dict(control_mode='arms-only'),dict(grasp_mode='body'),
    dict(box_names=('small_box_0','medium_box_0')),dict(placement_contact_force=0.),
    dict(transfer_target_tolerance=float('nan')),dict(rack_extract_clearance=-.1)])
def test_invalid_transfer_contract_rejected(overrides):
    with pytest.raises(ValueError):
        task_spec('pick_place',**overrides).validate()

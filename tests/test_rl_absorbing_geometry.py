"""Absorbing boundaries and variable-proximity loops in the actual reward paths."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import __future__

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import (
    absorbing_geometry_contract,with_absorbing_geometry_profile,
    frozen_geometry_actor_contract,configured_geometry_shaping,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
    frozen_actor_reward_contract,configured_reward_weights,contact_reward_weights,learning_termination_mask,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.model import (
    MultiBoxRewardModel,GraspRewardInput,CommonRewardInput,RewardBreakdown,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import (
    frozen_capture_actor_contract,with_precision_capture_profile,configured_capture_geometry,
)
from kuavo_isaaclab_scene.rl.multi_box.metrics.potentials import grasp_gated_lift_inputs
from test_rl_precision_capture import source_contract


def model():
    return MultiBoxRewardModel(contact_reward_weights(),geometry_shaping=absorbing_geometry_contract())


def value(n=1):
    z=torch.zeros(n);no=torch.zeros(n,dtype=torch.bool);one=torch.ones(n)
    return GraspRewardInput(z,one,z,one,z,one,one,z,one,z,one,z,z,one,no,no,no,
        CommonRewardInput(no,no,no,no,no,z,z,z),previous_gated_alignment=z,geometry_terminated=no)


def test_profile_migration_is_actor_only_and_keeps_nonreward_fields_visible():
    old=with_precision_capture_profile(source_contract());saved=deepcopy(old)
    new=with_absorbing_geometry_profile(old)
    assert old==saved and frozen_geometry_actor_contract(new)==old
    assert frozen_capture_actor_contract(new)==frozen_capture_actor_contract(old)
    assert frozen_actor_reward_contract(new)==frozen_actor_reward_contract(old)
    assert configured_reward_weights(new['reward_profile'])==contact_reward_weights()
    assert configured_capture_geometry(new['reward_profile'])==dict(capture_scale_m=.025,capture_aggregation='weak-hand')
    assert configured_geometry_shaping(old['reward_profile']) is None
    assert configured_geometry_shaping(new['reward_profile'])==absorbing_geometry_contract()
    timeout=torch.tensor([True]);assert learning_termination_mask(new['reward_profile'],~timeout,timeout).all()
    new['terminal_contract']['rack_force']=99.
    assert frozen_geometry_actor_contract(new)!=old
    with pytest.raises(ValueError):with_absorbing_geometry_profile(saved|{'reward_profile':{'weights':{}}})


@pytest.mark.parametrize('key,bad',[('absorbing_geometry',{}),('capture_scale_m',.03),('contact_shaping',{}),('weights',{})])
def test_unknown_reward_cannot_use_actor_compatibility_bypass(key,bad):
    new=with_absorbing_geometry_profile(with_precision_capture_profile(source_contract()))
    new['reward_profile'][key]=bad
    for read in (frozen_actor_reward_contract,frozen_capture_actor_contract,frozen_geometry_actor_contract):
        with pytest.raises(ValueError):read(new)
    with pytest.raises(ValueError):configured_geometry_shaping(new['reward_profile'])


def test_ending_nearer_cannot_change_geometric_reward_at_an_absorbing_boundary():
    v=value(2);prev=torch.full((2,),.4)
    v=replace(v,previous_approach=prev,previous_front_staging=prev,previous_capture=prev,
        previous_jaw_gap=prev,previous_proof_lift=prev,previous_gated_alignment=prev,
        approach=torch.tensor([.1,1.]),front_staging=torch.tensor([.1,1.]),capture=torch.tensor([.1,1.]),
        proof_lift=torch.tensor([.1,1.]),alignment=torch.tensor([.1,1.]),geometry_terminated=torch.ones(2,dtype=torch.bool))
    result=model().grasp(v)
    for name,weight in [('approach',2.),('front_staging',1.),('capture',.5),('proof_lift',.4),('alignment',.5),('jaw_gap',0.)]:
        torch.testing.assert_close(result.terms[name+'_progress'],-weight*prev,atol=0,rtol=0)
    legacy=MultiBoxRewardModel(contact_reward_weights()).grasp(v)
    assert legacy.terms['approach_progress'][1]>legacy.terms['approach_progress'][0]
    # Actual distance costs still use the measured pose; no terminal fiction enters observations.
    assert result.terms['approach_distance_cost'][1]==0


def test_discounted_alignment_cycle_cannot_earn_reward_from_changing_proximity():
    # Align nearby, move away, then undo the angle far away. The old gate pays
    # the improvement fully and discounts the opposite move by the far gate.
    states=[(0.,.1),(1.,1.),(1.,.1),(0.,.1)]
    total=0.;oldtotal=0.;gamma=model().weights.discount
    for i,((pa,pp),(a,p)) in enumerate(zip(states,states[1:])):
        v=replace(value(),previous_alignment=torch.tensor([pa]),alignment=torch.tensor([a]),
            alignment_proximity=torch.tensor([p]),previous_gated_alignment=torch.tensor([pa*pp]))
        total+=gamma**i*model().grasp(v).terms['alignment_progress'].item()
        oldtotal+=gamma**i*MultiBoxRewardModel(contact_reward_weights()).grasp(v).terms['alignment_progress'].item()
    assert total==pytest.approx(0.,abs=5e-8)
    assert oldtotal>.4


def test_new_model_refuses_missing_or_nonboolean_boundary_inputs():
    for bad in (replace(value(),geometry_terminated=None),replace(value(),geometry_terminated=torch.ones(1)),
                replace(value(),previous_gated_alignment=None)):
        with pytest.raises(ValueError):model().grasp(bad)


def production_methods():
    path=Path(__file__).resolve().parents[1]/'src/kuavo_isaaclab_scene/rl/multi_box/managers/v2_grasp.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='V2GraspReward')
    functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__call__','reset')]
    namespace=dict(torch=torch,GraspRewardInput=GraspRewardInput,CommonRewardInput=CommonRewardInput,
        RewardBreakdown=RewardBreakdown,grasp_gated_lift_inputs=grasp_gated_lift_inputs,
        privileged_grasp_step=lambda env:env.grasp,grasp_safety_step=lambda env:env.safety,
        reset_settling_step=lambda env:env.settling,task_time_out=lambda env:env.timeout,
        _base_motion=lambda env:torch.zeros(2),mdp=SimpleNamespace(joint_pos_limits=lambda env:torch.zeros(2)))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace


def manager_fixture(terminal_kind):
    n=2;one=torch.ones(n);z=torch.zeros(n);no=torch.zeros(n,dtype=torch.bool);yes=~no
    potentials={k:one.clone() for k in ('approach','front_staging','alignment','capture','jaw_gap','proof_lift','alignment_proximity')}
    potentials['premature_close']=z
    success=SimpleNamespace(success=yes if terminal_kind=='success' else no,bilateral_pinch=yes,opposing_flaps=yes)
    grasp=SimpleNamespace(potentials=potentials,success=success,assigned_flap_index=torch.tensor([[0,1],[0,1]]),
        invalid_box_pose=no,invalid_flap_pose=no,one_hand_pinch_event=no,bilateral_pinch_event=no,success_event=success.success)
    safety=SimpleNamespace(unsafe=yes if terminal_kind=='unsafe' else no,robot_rack_collision=yes if terminal_kind=='unsafe' else no,
        self_collision=no,obstacle_collision=no,workspace_limit=no,box_drop=no,box_lift_limit=no,box_speed_limit=no)
    env=SimpleNamespace(grasp=grasp,safety=safety,timeout=yes if terminal_kind=='timeout' else no,
        settling=SimpleNamespace(ready=yes,just_ready=no),step_dt=1/30,
        action_manager=SimpleNamespace(action=torch.zeros(n,24),prev_action=torch.zeros(n,24)))
    previous={k:torch.full((n,),.4) for k in ('approach','front_staging','alignment','capture','jaw_gap','proof_lift')}
    reward=SimpleNamespace(model=model(),previous=previous,initialized=yes.clone(),
        previous_assignment=torch.tensor([[0,1],[0,1]]),lift_armed=yes.clone(),
        previous_gated_alignment=torch.full((n,),.4),contact_progress=None)
    return reward,env


@pytest.mark.parametrize('terminal_kind',['success','unsafe','timeout'])
def test_real_manager_supplies_all_terminal_kinds_and_partial_reset(terminal_kind):
    reward,env=manager_fixture(terminal_kind)
    methods=production_methods();methods['__call__'](reward,env)
    for term in ('approach_progress','front_staging_progress','capture_progress','proof_lift_progress','alignment_progress'):
        assert (env._multi_box_grasp_reward_breakdown.terms[term]<0).all()
    if terminal_kind=='unsafe':assert env._multi_box_grasp_reward_breakdown.terms['robot_rack_collision'].eq(-6).all()
    if terminal_kind=='success':assert env._multi_box_grasp_reward_breakdown.terms['success_event'].eq(8).all()
    methods['reset'](reward,torch.tensor([0]))
    assert reward.previous_gated_alignment.tolist()==[0.,1.]
    assert reward.initialized.tolist()==[False,True]


def test_real_manager_rebases_alignment_on_flap_assignment_switch():
    reward,env=manager_fixture('ongoing')
    reward.previous_assignment[0]=torch.tensor([1,0])
    reward.previous_gated_alignment[:]=.1
    env.grasp.potentials['alignment'][:]=.8
    env.grasp.potentials['alignment_proximity'][:]=.5
    production_methods()['__call__'](reward,env)
    term=env._multi_box_grasp_reward_breakdown.terms['alignment_progress']
    assert term[0].item()==pytest.approx(0.,abs=5e-8)
    assert term[1].item()>0


def test_real_terminal_assignment_or_contact_loss_does_not_rebase_to_terminal_pose():
    reward,env=manager_fixture('unsafe')
    reward.previous_assignment[:]=torch.tensor([[1,0],[1,0]])
    env.grasp.success.bilateral_pinch=torch.zeros(2,dtype=torch.bool)
    production_methods()['__call__'](reward,env)
    for term,weight in [('approach',2.),('front_staging',1.),('capture',.5),('proof_lift',.4),('alignment',.5)]:
        torch.testing.assert_close(env._multi_box_grasp_reward_breakdown.terms[term+'_progress'],
            -weight*torch.full((2,),.4),rtol=0,atol=0)


def initializer_inputs():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import servo_critic_contract
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank,KEYS,retention_config
    from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import measured_credit_config
    retention=retention_config('tail64-half');bank=TrainSuccessBank(518,577,retention).state()
    # Old labels deliberately need not be usable: none may survive migration.
    bank['episodes']['shelf_3_right']=[{'old_reward_to_discard':torch.tensor([123.])}]
    goal=dict(actor_dim=518,critic_dim=577,critic_action_encoding=servo_critic_contract(),
        train_success_retention=retention,physical_contract=with_precision_capture_profile(source_contract()))
    state=dict(artifact_type=ServoCriticReanchoredSACPilot.artifact_type,algorithm='hybrid_goal_sac',goal_contract=goal,
        actor_updates=0,critic_updates=0,optimizers=[{'state':{}} for _ in range(4)],
        model={'actor.weight':torch.tensor([.2]),'q1.weight':torch.tensor([.3]),'critic_normalizer.count':torch.tensor(0.)},
        config={'gamma':.999},successful_train_transitions=bank,measured_train_credit=measured_credit_config('measured-nstep16'))
    experience=dict(goal_contract=deepcopy(goal),executed_goal_transitions={k:torch.empty(0) for k in KEYS},
        successful_train_transitions=deepcopy(bank),measured_train_credit=deepcopy(state['measured_train_credit']),
        measured_train_credit_bank={'old_reward_to_discard':torch.tensor([456.])})
    return state,experience


def test_initializer_preserves_models_but_clears_every_old_reward_bank_without_mutating_sources():
    from prepare_absorbing_geometry_actor import prepare
    from prepare_actual_success_actor_tail import identical
    state,experience=initializer_inputs();old_state=deepcopy(state);old_experience=deepcopy(experience)
    new,replay=prepare(state,experience)
    assert identical(state,old_state) and identical(experience,old_experience)
    assert identical(new['model'],state['model']) and identical(new['optimizers'],state['optimizers'])
    assert replay['goal_contract']==new['goal_contract']!=state['goal_contract']
    for bank in (new['successful_train_transitions'],replay['successful_train_transitions'],replay['measured_train_credit_bank']):
        assert not any(bank['episodes'].values())
    assert new['measured_train_credit_bank_report']['rows']==0


@pytest.mark.parametrize('bad_source',['trained_Q','optimizer','critic_normalizer','nonfinite_Q','online_replay'])
def test_initializer_rejects_old_learning_or_nonempty_replay(bad_source):
    from prepare_absorbing_geometry_actor import prepare
    state,experience=initializer_inputs()
    if bad_source=='trained_Q':state['critic_updates']=1
    elif bad_source=='optimizer':state['optimizers'][0]['state']={0:{'step':1}}
    elif bad_source=='critic_normalizer':state['model']['critic_normalizer.count']=torch.tensor(1.)
    elif bad_source=='nonfinite_Q':state['model']['q1.weight']=torch.tensor([float('nan')])
    else:experience['executed_goal_transitions']['reward']=torch.ones(1)
    with pytest.raises(ValueError):prepare(state,experience)

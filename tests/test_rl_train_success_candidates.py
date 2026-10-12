"""Terminal model identity, bounded storage and TRAIN/DEV separation."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.train_success_candidates import TrainSuccessCandidates


def outcome(environment=0, region='shelf_2_left', size='small', step=7):
    # Synthetic protocol evidence, never a physical success measurement.
    return dict(wave=1,split='train',environment=environment,
        layout=dict(seed=100+environment,target_region=region,target_box_type=size),
        complete=True,initial_layout_valid=True,
        result=dict(steps=step,success=True,unsafe=False,pinching=[True,True],
            stable_hands=[True,True],opposing_flaps=True,proof_lift=True,
            hold_time_s=.2667,rack_clearance_m=.01,
            staged_base=dict(phase='held_grasp',manipulation_start=1)))


def pilot():
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.Adam(model.parameters())
    model(torch.ones(1,3)).sum().backward();optimizer.step();optimizer.zero_grad()
    return SimpleNamespace(training=True,actor_updates=4,critic_updates=16,
        checkpoint_state=lambda:dict(model=model.state_dict(),Adam=optimizer.state_dict()),
        model=model,optimizer=optimizer,
        body_behavior_sampler=SimpleNamespace(greedy_unselected_policy=True,
            selected=torch.tensor([False,True]),initialized=torch.tensor([True,True])))


def test_candidate_is_pre_update_model_and_does_not_mutate_model_Adam_or_RNG(tmp_path):
    p=pilot(); archive=TrainSuccessCandidates(tmp_path)
    before=deepcopy(p.checkpoint_state()); rng=torch.get_rng_state().clone()
    event=archive.capture(p,[outcome(),outcome(1)],control_step=7,action_counters=(4,16))
    assert torch.equal(rng,torch.get_rng_state())
    path=tmp_path/event['checkpoint']; saved=torch.load(path,weights_only=True)
    assert event['SHA256']==hashlib.sha256(path.read_bytes()).hexdigest()
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
    assert event['model_tensor_SHA256']==structure_sha256(saved['model'])
    assert json.loads(path.with_suffix('.json').read_text())==event
    for key,value in before['model'].items():
        assert torch.equal(saved['model'][key],value)
        assert torch.equal(p.model.state_dict()[key],value)
    for key,state in before['Adam']['state'].items():
        for term,value in state.items():
            assert torch.equal(saved['Adam']['state'][key][term],value)
            assert torch.equal(p.optimizer.state_dict()['state'][key][term],value)
    with torch.no_grad():p.model.weight.add_(1)
    assert not torch.equal(saved['model']['weight'],p.model.weight)
    assert [e['collection_policy_mode'] for e in event['evidence']]==[
        'greedy_current_policy','coherent_arm_exploration']
    assert saved['TRAIN_success_candidate']['entire_episode_used_one_frozen_model'] is False
    assert saved['TRAIN_success_candidate']['terminal_action_from_greedy_actor_only'] is False
    assert saved['TRAIN_success_candidate']['terminal_action_may_include_exploration_override'] is True
    assert saved['TRAIN_success_candidate']['direct_training_resume_allowed'] is False


def test_new_scope_bypasses_interval_but_hard_run_limit_never_deletes_unverified_files(tmp_path):
    p=pilot();p.body_behavior_sampler=None;now=[0.]
    a=TrainSuccessCandidates(tmp_path,maximum=3,clock=lambda:now[0])
    assert a.capture(p,[outcome()],control_step=7,action_counters=(4,16))
    assert a.capture(p,[outcome(1)],control_step=7,action_counters=(4,16)) is None
    assert a.capture(p,[outcome(2,size='medium')],control_step=7,action_counters=(4,16))
    now[0]=300
    assert a.capture(p,[outcome(3)],control_step=7,action_counters=(4,16))
    now[0]=600
    assert a.capture(p,[outcome(4,region='shelf_3_right')],control_step=7,action_counters=(4,16)) is None
    assert len(list(tmp_path.glob('*.pt')))==3
    with pytest.raises(ValueError,match='unique'):TrainSuccessCandidates(tmp_path)


@pytest.mark.parametrize('corrupt', ['unsafe','invalid','single_hand','low_hold','low_lift','wrong_split'])
def test_incomplete_or_unsafe_evidence_never_archived(tmp_path,corrupt):
    o=outcome()
    if corrupt=='unsafe':o['result']['unsafe']=True
    if corrupt=='invalid':o['initial_layout_valid']=False
    if corrupt=='single_hand':o['result']['pinching']=[True,False]
    if corrupt=='low_hold':o['result']['hold_time_s']=.24
    if corrupt=='low_lift':o['result']['rack_clearance_m']=.007
    if corrupt=='wrong_split':o['split']='validation'
    with pytest.raises(ValueError):
        TrainSuccessCandidates(tmp_path).capture(pilot(),[o],control_step=7,action_counters=(4,16))
    assert not list(tmp_path.iterdir())


def test_optimizer_already_updated_or_frozen_eval_cannot_capture(tmp_path):
    p=pilot();a=TrainSuccessCandidates(tmp_path)
    with pytest.raises(ValueError,match='precede'):
        a.capture(p,[outcome()],control_step=7,action_counters=(3,16))
    p.training=False
    with pytest.raises(ValueError,match='TRAIN'):
        a.capture(p,[outcome()],control_step=7,action_counters=(4,16))
    assert not list(tmp_path.iterdir())


def test_repeated_easy_scope_cannot_take_reserved_new_scope_slot(tmp_path):
    p=pilot();p.body_behavior_sampler=None;now=[0.]
    a=TrainSuccessCandidates(tmp_path,maximum=3,clock=lambda:now[0],
        expected_scopes={('shelf_2_left','small'),('shelf_3_right','small')})
    assert a.capture(p,[outcome()],control_step=7,action_counters=(4,16))
    now[0]=300
    assert a.capture(p,[outcome(1)],control_step=7,action_counters=(4,16))
    now[0]=600
    assert a.capture(p,[outcome(2)],control_step=7,action_counters=(4,16)) is None
    assert a.capture(p,[outcome(3,region='shelf_3_right')],control_step=7,action_counters=(4,16))
    assert a.report()['reserved_unseen_scopes']==[]

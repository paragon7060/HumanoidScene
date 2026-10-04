"""Synthetic fixtures verify archive provenance; they do not prove physical success."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.training_goal_collection import (
    FILENAME, TrainingGoalCollector, validate_training_collection,
)
from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import grasp_lift_terminal_contract


def fixture():
    layout = dict(split='train', seed=120352, target_region='shelf_3_right')
    contract = dict(name='staged_base_hold_remaining_hybrid_sac_v1', actor_dim=480, critic_dim=539,
                    physical_contract=dict(terminal_contract=grasp_lift_terminal_contract()))
    ao = torch.zeros(1,480); co = torch.zeros(1,539)
    ao[:,96] = 1; co[:,96] = 1
    ao[:,-6] = 1; co[:,-6] = 1
    goal = torch.linspace(-.7,.7,21)[None]; goal[:,19:] = torch.tensor([-1.,1.])
    following = (ao.clone(),co.clone())
    return layout,contract,(ao,co,goal),following


@pytest.mark.parametrize('bad', ['heldout','optimizer','teacher','not_staged'])
def test_collection_cannot_import_evaluation_or_live_teacher(bad):
    layout,_,_,_ = fixture()
    if bad=='heldout':layout['split']='holdout'
    with pytest.raises(ValueError,match='explicit TRAIN'):
        validate_training_collection(layout,staged_policy=bad!='not_staged',
                                     optimization=bad=='optimizer',live_teacher=bad=='teacher')


def test_success_archive_preserves_exact_chosen_goals_and_measured_next_state(tmp_path):
    layout,contract,previous,following = fixture()
    collector = TrainingGoalCollector(layout,contract)
    collector.append(previous,following,torch.tensor([5.]),torch.tensor([True]))
    expected=previous[2].clone();previous[2].zero_()  # The archive owns a stable copy.
    outcome=dict(wave=0,split='train',environment=0,layout=layout,initial_layout_valid=True,complete=True,
        result=dict(success=True,unsafe=False,unsafe_causes={},pinching=[True,True],
            stable_hands=[True,True],opposing_flaps=True,proof_lift=True,hold_time_s=.27,
            rack_clearance_m=.009,staged_base=dict(phase='held_grasp',manipulation_start=60)))
    report=collector.save(tmp_path,outcome,actor_updates=3389,critic_updates=15602,completed=True,interrupted=False)
    saved=torch.load(tmp_path/FILENAME,weights_only=True)
    torch.testing.assert_close(saved['goal_transitions']['action'],expected)
    torch.testing.assert_close(saved['goal_transitions']['next_critic_obs'],following[1])
    assert saved['source_actor_updates']==3389 and saved['optimizer_updates']==0
    assert not saved['evaluation_or_probe_imported'] and not saved['physical_delta_actions_inverted']
    assert report['success_bank']['rows']==1 and report['success_bank']['by_region']['shelf_3_right']['episodes']==1


def test_malformed_or_discontinuous_context_cannot_be_saved():
    layout,contract,previous,following = fixture();collector=TrainingGoalCollector(layout,contract)
    invalid=tuple(x.clone() for x in previous);invalid[2][:,19]=.5
    with pytest.raises(ValueError,match='binary jaws'):collector.append(invalid,following,torch.tensor([0.]),torch.tensor([False]))
    collector.append(previous,following,torch.tensor([0.]),torch.tensor([False]))
    invalid=tuple(x.clone() for x in previous);invalid[1][:,1]=1
    with pytest.raises(ValueError,match='continuous measured'):collector.append(invalid,following,torch.tensor([0.]),torch.tensor([False]))
    previous[0][:,-6]=0
    with pytest.raises(ValueError,match='held phase'):collector.append(previous,following,torch.tensor([0.]),torch.tensor([False]))


@pytest.mark.parametrize('mode,seed', [('teacher',0),('checkpoint-exploration',None),
                                     ('checkpoint-exploration',-1),('greedy',42)])
def test_behavior_provenance_is_required(mode,seed):
    layout,contract,_,_=fixture()
    with pytest.raises(ValueError,match='sampling seed'):
        TrainingGoalCollector(layout,contract,behavior=mode,behavior_seed=seed)


def test_exploration_failure_is_preserved_as_failure_without_success_rows(tmp_path):
    layout,contract,previous,following=fixture()
    collector=TrainingGoalCollector(layout,contract,behavior='checkpoint-exploration',behavior_seed=120352)
    collector.append(previous,following,torch.tensor([-.01]),torch.tensor([False]))
    outcome=dict(split='train',layout=layout,result=dict(success=False,time_out=True))
    collector.save(tmp_path,outcome,actor_updates=3389,critic_updates=15602,completed=True,interrupted=False)
    saved=torch.load(tmp_path/FILENAME,weights_only=True)
    assert saved['behavior']=='checkpoint-exploration' and saved['behavior_seed']==120352
    assert not saved['outcome']['result']['success']
    assert all(not episodes for episodes in saved['successful_train_transitions']['episodes'].values())


def test_unsafe_report_or_new_layout_cannot_supply_success_bank(tmp_path):
    layout,contract,previous,following = fixture();collector=TrainingGoalCollector(layout,contract)
    collector.append(previous,following,torch.tensor([-6.]),torch.tensor([True]))
    outcome=dict(wave=0,split='train',environment=0,layout=layout,initial_layout_valid=True,complete=True,
        result=dict(success=True,unsafe=True,unsafe_causes=dict(robot_rack_collision=True)))
    with pytest.raises(ValueError,match='Unsafe'):collector.save(tmp_path,outcome,actor_updates=1,critic_updates=1,completed=True,interrupted=False)
    changed=deepcopy(outcome);changed['layout']['seed']+=1
    with pytest.raises(ValueError,match='TRAIN identity'):collector.save(tmp_path,changed,actor_updates=1,critic_updates=1,completed=True,interrupted=False)
    assert not (tmp_path/FILENAME).exists()

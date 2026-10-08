"""A resumed/current policy must not be advertised as its nominal initializer."""
from copy import deepcopy
import json

import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.policy_manifest import checkpoint_manifest_fields


def fixture():
    physical=dict(robot_model='s63', gripper='leju-twofinger',
        terminal_contract=dict(rack_force_n=10., obstacle_force_n=5.))
    old=dict(name='staged_base_hold_remaining_hybrid_sac_v1', actor_dim=480,
        critic_dim=539, shelf_templates=dict(middle='old-small-template'))
    actual=dict(name='staged_actual_flap_regional_actor_memory_servo_retention_sac_v1',
        actor_dim=518, critic_dim=578,
        shelf_templates=dict(name='TRAIN_failed_supported_size_workplace_bootstrap_v1',
            requested_supported_combinations=6), physical_contract=deepcopy(physical))
    contract=physical|dict(goal_contract=old, initialized_not_trained=True)
    checkpoint=dict(artifact_type=actual['name'], goal_contract=actual,
        actor_updates=0, critic_updates=0, model={'not_metadata':object()})
    return contract, checkpoint


def test_actual_all_six_policy_replaces_nominal_description_before_manifest_upload():
    contract, checkpoint=fixture()
    original_contract=deepcopy(contract); original_goal=deepcopy(checkpoint['goal_contract'])
    fields=checkpoint_manifest_fields(contract, checkpoint, artifact_type=checkpoint['artifact_type'])
    manifest=contract|fields|dict(artifact_type=checkpoint['artifact_type'])
    decoded=json.loads(json.dumps(manifest, allow_nan=False))
    assert decoded['goal_contract']==original_goal
    assert decoded['goal_contract']['actor_dim']==518
    assert decoded['goal_contract']['shelf_templates']['requested_supported_combinations']==6
    assert decoded['initialization_source_goal_contract']==original_contract['goal_contract']
    assert decoded['authoritative_runtime_policy_contract_file']=='agent.yaml'
    assert contract==original_contract and checkpoint['goal_contract']==original_goal
    for key in ('robot_model','gripper','terminal_contract'):
        assert decoded[key]==contract[key]
    assert 'model' not in decoded
    # Both source and current contracts are separate metadata copies.
    fields['goal_contract']['shelf_templates']['requested_supported_combinations']=4
    fields['initialization_source_goal_contract']['actor_dim']=1
    assert contract==original_contract and checkpoint['goal_contract']==original_goal


def test_resumed_learning_is_not_marked_untrained_by_historical_input_flag():
    contract, checkpoint=fixture()
    checkpoint.update(actor_updates=3108, critic_updates=14478)
    fields=checkpoint_manifest_fields(contract, checkpoint, artifact_type=checkpoint['artifact_type'])
    assert fields['initialized_not_trained'] is False
    assert fields['policy_checkpoint_state']['actor_updates']==3108
    assert fields['policy_checkpoint_state']['critic_updates']==14478
    assert contract['initialized_not_trained'] is True


def test_existing_literal_physical_body_runner_keeps_its_contract_key():
    contract, checkpoint=fixture()
    physical=dict(name='staged_held_physical_body_hybrid_sac_v1',
        action_coordinates='literal_executed_physical_body21')
    checkpoint.update(artifact_type=physical['name'], physical_body_contract=physical)
    del checkpoint['goal_contract']
    fields=checkpoint_manifest_fields(contract, checkpoint, artifact_type=physical['name'])
    assert fields['physical_body_contract']==physical
    assert fields['policy_contract_key']=='physical_body_contract'
    assert fields['initialization_source_goal_contract']==contract['goal_contract']
    assert 'goal_contract' not in fields


def test_ambiguous_goal_and_physical_body_metadata_is_rejected():
    contract, checkpoint=fixture()
    checkpoint['physical_body_contract']=deepcopy(checkpoint['goal_contract'])
    with pytest.raises(ValueError):
        checkpoint_manifest_fields(contract, checkpoint, artifact_type=checkpoint['artifact_type'])


@pytest.mark.parametrize('wrong',['source_artifact','goal_name','missing_goal','missing_counter',
    'negative_counter','boolean_counter','nonfinite_goal'])
def test_corrupted_policy_metadata_is_rejected_before_publication(wrong):
    contract, checkpoint=fixture(); expected=checkpoint['artifact_type']
    if wrong=='source_artifact':checkpoint['artifact_type']='other-policy'
    elif wrong=='goal_name':checkpoint['goal_contract']['name']='other-policy'
    elif wrong=='missing_goal':del checkpoint['goal_contract']
    elif wrong=='missing_counter':del checkpoint['critic_updates']
    elif wrong=='negative_counter':checkpoint['actor_updates']=-1
    elif wrong=='boolean_counter':checkpoint['actor_updates']=True
    else:checkpoint['goal_contract']['scale']=float('nan')
    with pytest.raises(ValueError):
        checkpoint_manifest_fields(contract, checkpoint, artifact_type=expected)

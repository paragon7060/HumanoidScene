"""CPU comparison must preserve full requested coverage and frozen networks."""
import json
from types import SimpleNamespace

import pytest
import torch

from batched_staged_goal_with_drive import validate_managed_physics_device
from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import (
    REGIONS,validate_frozen_backend_policy_eval,frozen_network_snapshot,verify_frozen_network_snapshot,
)
from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import validate_reset_diagnostic_request
from kuavo_isaaclab_scene.rl.multi_box.scene.reset_world_frame import validate_reset_world_frame_request


def original_waves():
    return [dict(split='validation',layouts=[dict(episode_index=0,layout=dict(
        seed=121000+i,split='holdout',target_region=REGIONS[i%4])) for i in range(128)])]


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_full_original_frozen_backend_eval_is_explicit_and_Q_ineligible(tmp_path,device):
    waves=original_waves();path=tmp_path/'DEV.json';path.write_text(json.dumps(waves))
    result=validate_managed_physics_device(device,['--no-training','--frozen-physics-backend-eval',
        '--steps','900','--waves-json',str(path),'--reset-failure-diagnostics'])
    assert not result['training'] and not result['Q_import_eligible']
    assert result['original_DEV_requested']==128 and result['requested_per_region']==32
    validate_reset_diagnostic_request(waves,enabled=True,training=False,steps=900,
        frozen_backend_evaluation=True,physics_device=device)


@pytest.mark.parametrize('change',['TRAIN','FINAL','short','unbalanced','duplicate','packed','mixed_layout_role'])
def test_backend_eval_never_drops_requested_cases_or_imports_other_roles(change):
    waves=original_waves()
    if change=='TRAIN':waves[0]['split']='train'
    if change=='FINAL':waves[0]['split']='holdout'
    if change=='short':waves[0]['layouts'].pop()
    if change=='unbalanced':waves[0]['layouts'][0]['layout']['target_region']=REGIONS[1]
    if change=='duplicate':waves[0]['layouts'][1]['layout']['seed']=121000
    if change=='packed':waves[0]['background_placement']='packed'
    if change=='mixed_layout_role':waves[0]['layouts'][0]['layout']['split']='train'
    with pytest.raises(ValueError):
        validate_frozen_backend_policy_eval(waves,enabled=True,device='cpu',training=False,steps=900)


@pytest.mark.parametrize('flags',[
    ['--training'],[],['--training','--no-training'],['--no-training','--steps','899'],
    ['--no-training','--packed-background-probe'],['--no-training','--jaw-behavior=joint-epsilon30'],
])
def test_manager_rejects_implicit_mutating_or_partial_backend_eval_before_launch(tmp_path,flags):
    path=tmp_path/'DEV.json';path.write_text(json.dumps(original_waves()))
    with pytest.raises(ValueError):
        validate_managed_physics_device('cpu',['--frozen-physics-backend-eval','--waves-json',str(path),*flags])


def test_existing_world_frame_restriction_survives_and_explicit_full_eval_keeps_seed_identity():
    waves=original_waves()
    probe=dict(probe_type='current_world_frame',samples=[dict(layout_seed=r['layout']['seed']) for r in waves[0]['layouts']])
    with pytest.raises(ValueError):
        validate_reset_world_frame_request(waves,probe,reset_enabled=True,training=False,steps=900)
    validate_reset_world_frame_request(waves,probe,reset_enabled=True,training=False,steps=900,
        frozen_backend_evaluation=True,physics_device='cpu')
    probe['samples'][0]['layout_seed']=999
    with pytest.raises(ValueError,match='layout seeds'):
        validate_reset_world_frame_request(waves,probe,reset_enabled=True,training=False,steps=900,
            frozen_backend_evaluation=True,physics_device='cpu')


@pytest.mark.parametrize('module_name',['agent','warm_agent','body_anchor_actor','prior_normalizer'])
def test_frozen_eval_detects_learned_or_prior_or_normalizer_drift(module_name):
    agent=torch.nn.Linear(2,2)
    warm=torch.nn.Linear(2,2)
    prior_actor=torch.nn.Linear(2,2);normalizer=torch.nn.BatchNorm1d(2)
    anchor=torch.nn.Linear(2,2)
    pilot=SimpleNamespace(agent=agent,warm_start=SimpleNamespace(agent=warm),
        command_prior=SimpleNamespace(actor=prior_actor,normalizer=normalizer),body_anchor={'actor':anchor})
    before=frozen_network_snapshot(pilot)
    assert verify_frozen_network_snapshot(pilot,before)['all_model_and_normalizer_tensors_bit_identical']
    module={'agent':agent,'warm_agent':warm,'body_anchor_actor':anchor,'prior_normalizer':normalizer}[module_name]
    with torch.no_grad():next(iter(module.state_dict().values())).add_(1.)
    with pytest.raises(ValueError,match='model or normalizer tensors'):
        verify_frozen_network_snapshot(pilot,before)


def test_actual_flap_pilot_frozen_controller_snapshot_covers_all_real_prior_modules(tmp_path):
    from test_rl_actual_flap_residual_sac import pilots
    _,pilot,*_=pilots(tmp_path)
    pilot.training=False;pilot.agent.requires_grad_(False)
    before=frozen_network_snapshot(pilot)
    assert {'agent','warm_agent','goal_prior_agent','frozen_actor_prior',
        'body_anchor_actor','body_anchor_actor_normalizer'} <= before.keys()
    raw=torch.zeros(3,518);raw[:,144]=1.;raw[:,-1]=.15
    with torch.no_grad():action=pilot.agent.act(raw,deterministic=True)
    assert torch.isfinite(action).all() and action.shape==(3,21)
    assert verify_frozen_network_snapshot(pilot,before)['all_model_and_normalizer_tensors_bit_identical']

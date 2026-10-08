"""Coherent TRAIN exploration must retain episode identity and legal controls."""
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import (
    VARIANT, GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT,
    RampedArmBehaviorExploration, body_behavior_config, body_behavior_statistics)


def observation(n):
    raw = torch.zeros(n, 518)
    raw[:, -6] = 1
    return raw


def test_episode_mixture_ramp_and_global_ids_without_per_step_direction_jitter():
    torch.manual_seed(21)
    count = 20000
    sampler = RampedArmBehaviorExploration(count, 'cpu', body_behavior_config(VARIANT))
    ids = torch.arange(count)
    raw = observation(count)
    assert sampler.offset(raw, 0, ids).eq(0).all()
    bias = sampler.bias.clone()
    selected = sampler.selected.clone()
    rng = torch.random.get_rng_state().clone()
    half = sampler.offset(raw, 45, ids)
    full = sampler.offset(raw, 90, ids)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(half[:, 1:15], bias*.5)
    assert torch.equal(full[:, 1:15], bias)
    assert full[:, [0, 15, 16, 17, 18]].eq(0).all()
    assert full[~selected].eq(0).all() and full.abs().max() <= 1.6
    assert selected.float().mean() == pytest.approx(.2, abs=.01)
    active = torch.tensor([19001, 8, 800, 19])
    assert torch.equal(sampler.offset(raw[active], 180, active), full[active])
    assert sampler.report()['episodes_drawn'] == count
    sampler.reset(torch.tensor([8]))
    sampler.offset(raw[active], 180, active)
    assert sampler.report()['episodes_drawn'] == count+1
    assert torch.equal(sampler.bias[19001], bias[19001])
    assert torch.equal(sampler.bias[800], bias[800])
    sampler.offset(raw[active], torch.tensor([181., 181., 0., 181.]), active)
    assert sampler.report()['episodes_drawn'] == count+2


@pytest.mark.parametrize('change', ('width', 'nan', 'phase', 'duplicate', 'range', 'clock', 'negative'))
def test_rejects_invalid_identity_measured_state_or_clocks(change):
    sampler = RampedArmBehaviorExploration(4, 'cpu', body_behavior_config(VARIANT))
    raw = observation(2); ids = torch.tensor([3, 1]); clocks = torch.zeros(2)
    if change == 'width': raw = raw[:, :480]
    if change == 'nan': raw[0, 0] = float('nan')
    if change == 'phase': raw[0, -6] = 0
    if change == 'duplicate': ids = torch.tensor([1, 1])
    if change == 'range': ids = torch.tensor([1, 4])
    if change == 'clock': clocks = torch.zeros(3)
    if change == 'negative': clocks[0] = -1
    with pytest.raises(ValueError): sampler.offset(raw, clocks, ids)
    assert not sampler.initialized.any()


def test_configuration_and_persisted_statistics_fail_closed():
    assert body_behavior_config(None) is None and body_behavior_config('off') is None
    with pytest.raises(ValueError): body_behavior_config('wide')
    with pytest.raises(ValueError): RampedArmBehaviorExploration(4, 'cpu', {'variant': VARIANT})
    for key in body_behavior_statistics():
        bad = body_behavior_statistics(); bad[key] = -1
        with pytest.raises(ValueError): body_behavior_statistics(bad)
    bad = body_behavior_statistics(); bad['biased_episodes'] = 1
    with pytest.raises(ValueError): body_behavior_statistics(bad)


@pytest.mark.parametrize('variant',[VARIANT,GREEDY_REST_VARIANT,GENTLE_GREEDY_REST_VARIANT])
def test_managed_collection_option_is_allowed_in_TRAIN_and_rejected_in_frozen_probes(tmp_path,variant):
    import json
    from batched_staged_goal_with_drive import validate_managed_physics_device
    from test_rl_cpu_physics_training import inputs
    waves,contract=inputs();wp=tmp_path/'waves.json';mp=tmp_path/'manifest.json'
    wp.write_text(json.dumps(waves));mp.write_text(json.dumps(contract))
    common=['--waves-json',str(wp),'--training-manifest',str(mp),'--body-behavior',variant]
    assert validate_managed_physics_device('cpu',common+['--training','--cpu-physics-training'],
        learner_device='cuda:0')['training']
    wp.write_text(json.dumps(waves[:1]))
    with pytest.raises(ValueError,match='unchanged other physics'):
        validate_managed_physics_device('cpu',common+['--no-training','--frozen-physics-backend-eval'])
    with pytest.raises(ValueError,match='only named waypoints'):
        validate_managed_physics_device('cpu',common+['--no-training','--cpu-workplace-probe','--base-waypoint-probe'])


def test_greedy_rest_reuses_original20percent_selection_bias_and_wave_identity():
    count=20000;raw=observation(count);ids=torch.arange(count)
    original=RampedArmBehaviorExploration(count,'cpu',body_behavior_config(VARIANT))
    mixed=RampedArmBehaviorExploration(count,'cpu',body_behavior_config(GREEDY_REST_VARIANT))
    torch.manual_seed(93);before=original.offset(raw,90,ids)
    torch.manual_seed(93);after=mixed.offset(raw,90,ids)
    assert torch.equal(before,after) and torch.equal(original.selected,mixed.selected)
    assert mixed.selected.float().mean()==pytest.approx(.2,abs=.01)
    assert mixed.greedy_unselected_policy and not original.greedy_unselected_policy
    chosen=torch.tensor([10,1,300,5000]);selection=mixed.selected[chosen].clone()
    mixed.offset(raw[chosen],91,chosen)
    assert torch.equal(selection,mixed.selected[chosen])
    assert mixed.report()['episodes_drawn']==count


def test_gentle_behavior_preserves_selection_ramp_and_reduces_bias_without_extra_draws():
    count=2048;raw=observation(count);ids=torch.arange(count)
    wide=RampedArmBehaviorExploration(count,'cpu',body_behavior_config(GREEDY_REST_VARIANT))
    gentle=RampedArmBehaviorExploration(count,'cpu',body_behavior_config(GENTLE_GREEDY_REST_VARIANT))
    torch.manual_seed(18);old=wide.offset(raw,45,ids);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(18);new=gentle.offset(raw,45,ids)
    assert torch.equal(old_rng,torch.get_rng_state())
    assert torch.equal(gentle.selected,wide.selected)
    assert torch.allclose(new,old*.0125,atol=1e-9,rtol=1e-6)
    assert new.abs().max()<=.01
    assert gentle.greedy_unselected_policy
    assert new[:,[0,15,16,17,18]].eq(0).all()
    assert new[~gentle.selected].eq(0).all()
    assert torch.equal(new[:,1:15],gentle.bias*.5)
    selection=gentle.selected.clone()
    full=gentle.offset(raw,90,ids)
    assert torch.equal(gentle.selected,selection)
    assert torch.equal(full[:,1:15],gentle.bias) and full.abs().max()<=.02
    old_contract=body_behavior_config(GREEDY_REST_VARIANT)
    new_contract=body_behavior_config(GENTLE_GREEDY_REST_VARIANT)
    assert {k for k in old_contract if old_contract[k]!=new_contract[k]}=={
        'variant','latent_bias_std','max_abs_latent_bias'}

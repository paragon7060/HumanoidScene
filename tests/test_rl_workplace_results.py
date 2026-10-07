import pytest
from test_rl_cpu_workplace_probe import request
from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import SOURCE
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import CPU_PHYSICS_BACKEND,staged_solver_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results


def fixture():
    waves=request();manifest=dict(layout_waves=waves,sim_device='cpu',training=False,
        physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))
    outcomes=[]
    for i,row in enumerate(waves[0]['layouts']):
        success=(row['waypoint_probe']['name']=='candidate0' and row['layout']['seed']%100==0)
        outcomes.append(dict(wave=0,environment=i,split='train',layout=row['layout'],complete=True,
            initial_layout_valid=True,result=dict(success=success,unsafe=False,time_out=not success,
                pinching=[success,success],stable_hands=[success,success],opposing_flaps=success,
                proof_lift=success,hold_time_s=.267 if success else 0.,rack_clearance_m=.03,
                staged_base=dict(waypoint_probe=row['waypoint_probe']))))
    metrics=dict(outcomes=outcomes,CPU_workplace_probe=dict(source=SOURCE,Q_import_eligible=False,
        replay_rows_imported=0,frozen_network_integrity=dict(
        all_model_and_normalizer_tensors_bit_identical=True,compared_tensor_count=178),
        actor_critic_updates_and_replay_size_unchanged=True),learner=dict(
        training=False,actor_updates=0,critic_updates=0,replay_size=0,online_rows=0))
    return manifest,metrics


def test_repeated_candidates_do_not_become_independent_cases_and_invalid_requests_remain():
    manifest,metrics=fixture();row=metrics['outcomes'][8];row['initial_layout_valid']=False
    row['result']=dict(success=False,invalid_reset=True)
    result=summarize_workplace_results(manifest,metrics)
    assert result['unique_fresh_TRAIN_cases']==16 and result['physical_candidate_attempts']==128
    assert result['all_regions_have_a_measured_success_candidate'] and result['not128_independent_cases']
    best=result['promising_candidates_for_fresh_TRAIN_recheck']['shelf_2_left']
    assert best['requested']==4 and best['success']==1 and best['initial_invalid']==1


def test_zero_success_region_is_not_promoted_to_a_successful_workplace():
    manifest,metrics=fixture()
    for out in metrics['outcomes']:
        if out['layout']['target_region']=='shelf_3_right':out['result']['success']=False
    result=summarize_workplace_results(manifest,metrics)
    assert result['promising_candidates_for_fresh_TRAIN_recheck']['shelf_3_right'] is None
    assert not result['all_regions_have_a_measured_success_candidate']


@pytest.mark.parametrize('kind',['missing','partial','unsafe_success','no_pinch','different_candidate',
    'model_changed','actor_updated','nonfinite_hold','nonfinite_clearance','replay_imported','numerical_success'])
def test_incomplete_changed_or_unphysical_results_cannot_support_candidate_selection(kind):
    manifest,metrics=fixture();out=metrics['outcomes'][0]
    if kind=='missing':metrics['outcomes'].pop()
    elif kind=='partial':out['complete']=False
    elif kind=='unsafe_success':out['result']['unsafe']=True
    elif kind=='no_pinch':out['result']['pinching']=[True,False]
    elif kind=='different_candidate':out['result']['staged_base']['waypoint_probe']=dict(name='other',offset_xy_yaw=[0.,0.,0.])
    elif kind=='model_changed':metrics['CPU_workplace_probe']['frozen_network_integrity']['all_model_and_normalizer_tensors_bit_identical']=False
    elif kind=='actor_updated':metrics['learner']['actor_updates']=1
    elif kind=='nonfinite_hold':out['result']['hold_time_s']=float('nan')
    elif kind=='nonfinite_clearance':out['result']['rack_clearance_m']=float('nan')
    elif kind=='numerical_success':out['result']['numerical_failure']=True
    else:metrics['CPU_workplace_probe']['replay_rows_imported']=1
    with pytest.raises(ValueError):summarize_workplace_results(manifest,metrics)


def test_numerical_failures_remain_visible_in_the_original_denominator():
    manifest,metrics=fixture();out=metrics['outcomes'][8]
    out['result'].update(success=False,time_out=False,numerical_failure=True)
    summary=summarize_workplace_results(manifest,metrics)['regions']['shelf_2_left']
    candidate=next(x for x in summary if x['candidate']=='candidate0')
    assert candidate['requested']==4 and candidate['numerical_failure']==1
    assert candidate['success']==1 and candidate['other_terminal']==0


def test_learned_checkpoint_keeps_source_counters_without_training_or_replay_import():
    manifest,metrics=fixture()
    source=dict(actor_updates=689,critic_updates=4802,replay_size=0,online_rows=0)
    metrics['CPU_workplace_probe']['initial_learner_counters']=source
    metrics['learner'].update(source)
    result=summarize_workplace_results(manifest,metrics)
    assert result['frozen_source_actor_updates']==689 and result['frozen_source_critic_updates']==4802
    metrics['learner']['critic_updates']+=1
    with pytest.raises(ValueError):summarize_workplace_results(manifest,metrics)


@pytest.mark.parametrize('field',['replay_size','online_rows'])
def test_source_replay_is_forbidden_even_when_counter_does_not_change(field):
    manifest,metrics=fixture()
    source=dict(actor_updates=689,critic_updates=4802,replay_size=0,online_rows=0)
    source[field]=1
    metrics['CPU_workplace_probe']['initial_learner_counters']=source
    metrics['learner'].update(source)
    with pytest.raises(ValueError):summarize_workplace_results(manifest,metrics)

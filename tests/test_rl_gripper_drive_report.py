from copy import deepcopy

import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.gripper_drive_report import summarize_gripper_drive_probe


def fixture():
    layouts = [dict(episode_index=0, layout=dict(seed=i, target_region='shelf_2_left', base_lateral_m=i*.01)) for i in range(3)]
    waves = [dict(split='validation', gripper_drive_probe=name, layouts=deepcopy(layouts))
             for name in ('original', 'soft_2nm', 'original')]
    manifest = dict(training=False, contact_stability_probe=dict(name='four_claw_motor_drive_comparison',
        frozen_only=True, Q_import_eligible=False), layout_waves=waves)
    learner = dict(training=False, online_rows=0, replay_size=0, actor_updates=3389, critic_updates=15602)
    outcomes = [dict(wave=w, split='validation', layout=deepcopy(r['layout']), initial_layout_valid=True,
        result=dict(success=False, unsafe=False, pinching=[False, False], flap_distances=[.05, .08], unsafe_causes={}))
        for w in range(3) for r in layouts]
    audits = [dict(wave=w, name=name, frozen_only=True, Q_import_eligible=False,
        initialized_PhysX={k: dict(min=v, max=v) for k, v in
            (dict(stiffness=100., damping=5., effort_limit=2.) if w == 1
             else dict(stiffness=4000., damping=400., effort_limit=100.)).items()})
        for w, name in enumerate(('original', 'soft_2nm', 'original'))]
    return manifest, dict(learner=learner, outcomes=outcomes), audits, dict(completed_waves=3, interrupted=False)


def test_initial_failures_keep_full_denominator_and_exploded_distance_is_not_safe_approach():
    manifest, metrics, audits, status = fixture()
    out = metrics['outcomes']
    out[0]['initial_layout_valid'] = False
    out[0]['result'] = dict(success=False, invalid_reset=True)
    out[1]['result'].update(unsafe=True, flap_distances=[1e9, 1e9], unsafe_causes={'box_speed_limit': True})
    report = summarize_gripper_drive_probe(manifest, metrics, audits, status)
    first = report['waves'][0]
    assert first['attempts'] == 3 and first['valid'] == 2 and first['initial_invalid'] == 1
    assert first['safe_terminal_max_hand_distance_mean_m'] == .08
    pair = report['paired']['soft_vs_first_original']
    assert pair['same_initial_cases'] == 3 and pair['joint_valid_cases'] == 2
    assert pair['changed_initial_validity'] == 1
    assert pair['joint_valid_cause_changes']['box_speed_limit']['resolved'] == 1


def test_original_repeat_reports_variation_and_success_requires_full_physical_proof():
    manifest, metrics, audits, status = fixture()
    metrics['outcomes'][6]['result'].update(success=True, stable_hands=[True, True], pinching=[True, True],
        opposing_flaps=True, proof_lift=True, hold_time_s=.267, rack_clearance_m=.02)
    report = summarize_gripper_drive_probe(manifest, metrics, audits, status)
    assert report['paired']['original_repeat_vs_first_original']['full_denominator_success_gains'] == 1
    assert report['waves'][2]['success_rate'] == 1/3 and report['training_updates'] == 0
    metrics['outcomes'][6]['result']['unsafe'] = True
    with pytest.raises(ValueError, match='contradicts'):summarize_gripper_drive_probe(manifest, metrics, audits, status)


@pytest.mark.parametrize('change', ['case', 'neutral_source', 'missing_failure', 'trained', 'soft_audit', 'restore', 'partial'])
def test_report_rejects_case_drift_missing_denominator_training_bad_drive_and_partial_wave(change):
    manifest, metrics, audits, status = fixture()
    if change == 'case':metrics['outcomes'][3]['layout']['base_lateral_m'] += .01
    elif change == 'neutral_source':manifest['layout_waves'][1]['layouts'][0]['episode_index'] = 1
    elif change == 'missing_failure':metrics['outcomes'].pop(0)
    elif change == 'trained':metrics['learner']['online_rows'] = 1
    elif change == 'soft_audit':audits[1]['initialized_PhysX']['effort_limit']['max'] = 100
    elif change == 'restore':audits[2]['initialized_PhysX']['stiffness']['min'] = 100
    elif change == 'partial':status['completed_waves'] = 2
    with pytest.raises(ValueError):summarize_gripper_drive_probe(manifest, metrics, audits, status)

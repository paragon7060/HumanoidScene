"""Opt-in final-centimetre geometry; reward identity requires fresh Q/replay."""
from copy import deepcopy
from dataclasses import asdict


NAME = 'weak_hand_capture_25mm_v1'


def precision_capture_contract():
    return dict(name=NAME, scale_m=0.025, aggregation='weak-hand',
        score='0.25*(left+right)+0.5*min(left,right)',
        geometry='actual_calibrated_finger_straddle_and_flap_tangent_error',
        approach_front_contact_success_and_safety_unchanged=True,
        source_Q_replay_optimizer_import_allowed=False)


def frozen_capture_actor_contract(contract):
    """Strip only this known capture change when matching frozen actor inputs.

    Never use the returned contract for current Q, replay, or reward labels.
    Observation/action/terminal fields and the reviewed contact reward remain.
    """
    profile = contract.get('reward_profile', {})
    if 'precision_capture' not in profile:
        return contract
    from .contact_profile import contact_shaping_contract, contact_reward_weights
    if (profile['precision_capture'] != precision_capture_contract()
            or profile.get('capture_scale_m') != 0.025
            or profile.get('contact_shaping') != contact_shaping_contract()
            or profile.get('weights') != asdict(contact_reward_weights())):
        raise ValueError('Unknown precision capture cannot bypass actor compatibility')
    result = deepcopy(contract)
    result['reward_profile'].pop('precision_capture')
    result['reward_profile']['capture_scale_m'] = 0.10
    return result


def with_precision_capture_profile(contract):
    """Create a distinct physical reward contract; never mutate source data."""
    from .contact_profile import frozen_actor_reward_contract
    profile = contract['reward_profile']
    if ('precision_capture' in profile or 'contact_shaping' not in profile
            or profile.get('capture_scale_m') != 0.10):
        raise ValueError('Precision capture requires the reviewed 10cm contact reward')
    frozen_actor_reward_contract(contract)
    result = deepcopy(contract)
    result['reward_profile'].update(capture_scale_m=0.025,
        precision_capture=precision_capture_contract())
    return result


def configured_capture_geometry(profile):
    if not profile or 'precision_capture' not in profile:
        return dict(capture_scale_m=0.10, capture_aggregation='mean')
    frozen_capture_actor_contract(dict(reward_profile=profile))
    return dict(capture_scale_m=0.025, capture_aggregation='weak-hand')

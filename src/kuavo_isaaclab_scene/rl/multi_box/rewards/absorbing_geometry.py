"""Opt-in geometric shaping that shares Q's absorbing episode boundary."""
from copy import deepcopy
from dataclasses import asdict


def absorbing_geometry_contract():
    return dict(name='absorbing_geometric_potentials_v1',
        terminal_states=['success','unsafe','task_time_out'],
        zero_terminal_potentials=['approach','front_staging','alignment','capture','jaw_gap','proof_lift'],
        alignment_potential='alignment_mean_times_proximity_mean',
        alignment_progress='discount_times_next_potential_minus_previous_potential',
        reset_and_assignment_rebase=True,weights_geometry_success_safety_unchanged=True,
        source_Q_replay_optimizer_or_reward_labels_import_allowed=False)


def frozen_geometry_actor_contract(contract):
    """Strip only the recognized reward change when reading frozen actor inputs."""
    from .success_value import frozen_success_value_actor_contract
    contract=frozen_success_value_actor_contract(contract)
    profile=contract.get('reward_profile',{})
    if 'absorbing_geometry' not in profile:return contract
    from .contact_profile import contact_reward_weights,contact_shaping_contract
    from .precision_capture import precision_capture_contract
    if (profile['absorbing_geometry']!=absorbing_geometry_contract()
            or profile.get('weights')!=asdict(contact_reward_weights())
            or profile.get('contact_shaping')!=contact_shaping_contract()
            or profile.get('precision_capture')!=precision_capture_contract()
            or profile.get('capture_scale_m')!=.025):
        raise ValueError('Unknown absorbing geometry cannot bypass actor compatibility')
    result=deepcopy(contract);result['reward_profile'].pop('absorbing_geometry')
    return result


def with_absorbing_geometry_profile(contract):
    from .contact_profile import frozen_actor_reward_contract
    profile=contract['reward_profile']
    if 'absorbing_geometry' in profile or 'precision_capture' not in profile:
        raise ValueError('Absorbing geometry requires the reviewed precision contact reward')
    frozen_actor_reward_contract(contract)
    result=deepcopy(contract)
    result['reward_profile']['absorbing_geometry']=absorbing_geometry_contract()
    frozen_geometry_actor_contract(result)
    return result


def configured_geometry_shaping(profile):
    if not profile or 'absorbing_geometry' not in profile:return None
    frozen_geometry_actor_contract(dict(reward_profile=profile))
    return absorbing_geometry_contract()

"""An explicit fresh-Q comparison that values safe task completion more highly."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
from .weights import MultiBoxRewardWeights


def success_value_contract():
    return dict(name='safe_grasp_success64_v1', source_success_event_weight=8.,
        success_event_weight=64., only_safe_actual_task_success=True,
        all_other_reward_weights_geometry_and_terminal_rules_unchanged=True,
        source_Q_replay_optimizer_or_reward_labels_import_allowed=False)


class SafeSuccessValueWeights(MultiBoxRewardWeights):
    """Allow only the reviewed safe-success64 exception to the default ratio.

    Unsafe termination suppresses every grasp success event. Increasing its
    negative weight together with the success bonus would undo this comparison.
    The general multi-skill ratio and all other weights remain unchanged.
    """
    def validate(self):
        from .contact_profile import contact_reward_weights
        source=contact_reward_weights()
        expected=replace(source,grasp=replace(source.grasp,success_event=64.))
        if asdict(self)!=asdict(expected):
            raise ValueError('Safe success64 permits no other reward-weight changes')
        source.validate()


def success_value_weights():
    from .contact_profile import contact_reward_weights
    source=contact_reward_weights()
    expected=replace(source,grasp=replace(source.grasp,success_event=64.))
    result=SafeSuccessValueWeights(**{f.name:getattr(expected,f.name) for f in fields(expected)})
    result.validate()
    return result


def frozen_success_value_actor_contract(contract):
    """Strip this reward identity only for frozen actor compatibility checks."""
    profile=contract.get('reward_profile',{})
    if 'success_value' not in profile:return contract
    from .contact_profile import contact_reward_weights,contact_shaping_contract
    if (profile['success_value']!=success_value_contract()
            or profile.get('weights')!=asdict(success_value_weights())
            or profile.get('contact_shaping')!=contact_shaping_contract()):
        raise ValueError('Unknown success value cannot bypass actor compatibility')
    result=deepcopy(contract);result['reward_profile'].pop('success_value')
    result['reward_profile']['weights']=asdict(contact_reward_weights())
    return result


def with_success_value_profile(contract):
    from .contact_profile import frozen_actor_reward_contract
    profile=contract['reward_profile']
    if 'success_value' in profile or 'contact_shaping' not in profile:
        raise ValueError('Success64 requires the recognized source contact reward')
    frozen_actor_reward_contract(contract)
    result=deepcopy(contract)
    result['reward_profile'].update(weights=asdict(success_value_weights()),
        success_value=success_value_contract())
    frozen_actor_reward_contract(result)
    return result


def configured_success_value(profile):
    if 'success_value' not in profile:return None
    from .contact_profile import frozen_actor_reward_contract
    frozen_actor_reward_contract(dict(reward_profile=profile))
    return success_value_contract()

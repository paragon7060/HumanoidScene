"""A fixed smaller Gaussian; coherent wide TRAIN exploration remains separate."""
from dataclasses import replace
import math


BODY_STD_FACTOR = .25


def body_policy_spread_contract():
    return dict(name='quarter_std_body_policy_v1',body_latent_std_multiplier=BODY_STD_FACTOR,
        continuous_dimensions=19,min_policy_std=.0075,max_policy_std=.03,
        same_std_in_collection_actor_objective_and_all_target_Q_branches=True,
        log_std_bias_and_both_bounds_shifted_together=True,
        entropy_target_recomputed_by_squash_aware_base_class=True,
        greedy_body_and_binary_jaws_unchanged_at_initialization=True,
        AR_correlation_and_episode_body_bias_unchanged=True,
        goal_bounds_servo_controller_reward_randomization_success_safety_unchanged=True,
        trained_old_Q_optimizer_and_replay_import_allowed=False,curriculum=False)


def quarter_body_policy_config(config):
    if (config.min_policy_std,config.max_policy_std)!=(.03,.12) \
            or not math.isfinite(config.initial_policy_std) \
            or not .03<=config.initial_policy_std<=.12:
        raise ValueError('Quarter body std requires the reviewed original Gaussian bounds')
    return replace(config,initial_policy_std=config.initial_policy_std*BODY_STD_FACTOR,
        min_policy_std=.0075,max_policy_std=.03)


def validate_quarter_policy_state(state):
    profile=body_policy_spread_contract();goal=state.get('goal_contract',{});cfg=state.get('config',{})
    if (goal.get('body_policy_spread')!=profile
            or (cfg.get('min_policy_std'),cfg.get('max_policy_std'))!=(.0075,.03)
            or not isinstance(cfg.get('initial_policy_std'),(int,float))
            or not .0075<=cfg['initial_policy_std']<=.03
            or goal.get('exploration_std_initial')!=cfg['initial_policy_std']
            or goal.get('exploration_std_cap')!=cfg['max_policy_std']):
        raise ValueError('Saved quarter-Gaussian policy configuration or identity differs')
    target=min(-1.,math.log(cfg['max_policy_std'])+.5*math.log(2*math.pi*math.e)-.5)
    if state.get('entropy_contract')!=dict(name='squash_aware_active_dims_v2',target_per_dim=target):
        raise ValueError('Quarter Gaussian needs its own attainable saved entropy target')


def shift_body_log_std(model):
    """Copy the model, shifting only the19 continuous log-std output biases."""
    import torch
    result={k:v.clone() for k,v in model.items()}
    keys=[k for k,v in result.items() if k.startswith('actor.network.') and k.endswith('.bias')
        and tuple(v.shape)==(42,)]
    if len(keys)!=1 or not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('A finite21-goal hybrid actor output head is required')
    result[keys[0]][21:40]+=math.log(BODY_STD_FACTOR)
    return result,keys[0]

"""Optional TRAIN behavior: slowly enter one arm perturbation per episode.

Only collection changes. SAC's actor/target distributions and physical goal
coordinates remain unchanged; replay receives exactly the projected command.
"""
import torch
from copy import deepcopy


def episode_arm_exploration_contract():
    return dict(name='ramped_episode_arm_latent_bias_v1', arm_goal_columns=list(range(1,15)),
        middle_bias_std=.005, upper_bias_std=.02, max_standard_deviations=2., ramp_steps=90,
        phase='physically_confirmed_held_grasp', TRAIN_only=True,
        actor_and_target_distribution_unchanged=True, evaluation_bias=False,
        privileged_inputs=False, scene_curriculum=False)


def enable_episode_arm_exploration(state, experience):
    """Change only future collection, retaining matching actual Q data verbatim."""
    old=state['goal_contract']
    if (state.get('artifact_type')!='staged_base_hold_remaining_hybrid_sac_v1'
            or old!=experience.get('goal_contract') or 'episode_arm_exploration' in old
            or old.get('gripper_prior_bound') is not False or old.get('fixed_prior_radius')!=.05
            or old.get('collection_noise')!='independent_per_environment_AR1_pre_tanh_Gaussian'
            or not old.get('exploration_correlation')):
        raise ValueError('Require matching unmodified fixed-radius hybrid Q and actual replay')
    contract=deepcopy(old)
    contract.update(episode_arm_exploration=episode_arm_exploration_contract(),
        collection_noise='AR1_Gaussian_plus_ramped_episode_arm_latent_bias_v1')
    result=state.copy();result['goal_contract']=contract
    actual=experience.copy();actual['goal_contract']=contract
    # New resume compatibility does not relabel the sampler that made old data.
    actual['source_experience_collection_contract']=deepcopy(old)
    audit=dict(changed_fields=['episode_arm_exploration','collection_noise'],
        model_Q_actor_normalizers_optimizers_and_counters_unchanged=True,
        physical_observation_goal_waypoint_reward_and_safety_contract_unchanged=True,
        actual_replay_tensors_untransformed=True, new_training_updates=0,
        source_actor_updates=state['actor_updates'],source_critic_updates=state['critic_updates'],
        actual_replay_rows=len(actual['executed_goal_transitions']['reward']),
        matching_Q_and_actual_replay_reused=True, synthetic_or_evaluation_rows_added=0)
    result['episode_arm_behavior_initialization']=audit
    return result,actual,audit


class EpisodeArmExploration:
    def __init__(self, num_envs, device, contract):
        if contract != episode_arm_exploration_contract() or num_envs < 1:
            raise ValueError('Require the declared episode-arm behavior contract and positive environment count')
        self.contract=contract
        self.bias=torch.zeros(num_envs,14,device=device)
        self.initialized=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.latest={}

    def offset(self, observation, clocks, ids):
        if ids.ndim!=1 or ids.dtype!=torch.long or len(ids.unique())!=len(ids) \
                or (ids<0).any() or (ids>=len(self.bias)).any():
            raise ValueError('Distinct valid global environment IDs required')
        if observation.shape!=(len(ids),480) or not torch.isfinite(observation).all():
            raise ValueError('Episode-arm behavior requires measured480-D held observations')
        if not (observation[:,-6]==1).all():
            raise ValueError('Episode-arm behavior requires a physically confirmed held phase')
        if not isinstance(clocks,torch.Tensor):
            clocks=observation.new_full((len(ids),),clocks)
        if clocks.shape!=(len(ids),) or not torch.isfinite(clocks).all() or (clocks<0).any():
            raise ValueError('One measured nonnegative held clock per active environment required')
        regions=observation[:,94:98]
        if not ((regions==0)|(regions==1)).all() or not (regions.sum(-1)==1).all():
            raise ValueError('A selected rack region is required for arm exploration')
        new=ids[~self.initialized[ids]]
        if len(new):
            self.bias[new]=torch.randn(len(new),14,device=self.bias.device).clamp(-2,2)
            self.initialized[new]=True
        upper=regions[:,2:4].sum(-1)>.5
        std=torch.where(upper,.02,.005)
        u=(clocks.to(observation)/90).clamp(0,1)
        ramp=u.square()*(3-2*u)
        result=observation.new_zeros(len(ids),19)
        result[:,1:15]=self.bias[ids]*std[:,None]*ramp[:,None]
        self.latest=dict(active_environments=len(ids), initialized_environments=int(self.initialized.sum()),
            body_bias_rms=float(result[:,1:15].square().mean().sqrt()),
            body_bias_max=float(result.abs().max()), TRAIN_only=True, privileged_inputs=False)
        return result

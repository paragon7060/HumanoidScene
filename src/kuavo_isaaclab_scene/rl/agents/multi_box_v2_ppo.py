"""RSL-RL PPO configuration for the isolated multi-box v2 grasp skill."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MultiBoxV2GraspPPOCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "multi_box_v2_grasp"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

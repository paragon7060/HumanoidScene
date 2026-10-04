"""Held-base SAC with categorical jaw decisions and continuous arm/torso goals."""
from ...algorithms.hybrid_goal_sac import HybridGoalSAC
from .staged_goal_sac import StagedGoalSACPilot


class StagedHybridGoalSACPilot(StagedGoalSACPilot):
    artifact_type='staged_base_hold_remaining_hybrid_sac_v1'
    agent_class=HybridGoalSAC

    @property
    def contract(self):
        return super().contract|dict(gripper_policy='two_masked_Bernoulli_exact_four_action_expectation',
            continuous_goal_dimensions=19,critic_jaw_actions='actual_binary_minus_one_plus_one',
            discrete_entropy_target_per_jaw=.35,discrete_prior_weight=.05,
            continuous_only_Q_or_replay_imported=False)

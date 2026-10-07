"""State-dependent critic features for the existing bounded goal servo.

The actor and replay still carry literal absolute goals. Both critics see
exactly the body command those goals decode to at the measured drive state.
This shares clipping and pending-target subtraction with the real decoder;
it neither runs IK/physics nor changes the actor's goal-space entropy.
"""
import torch

from .physical_body_actions import goal_body_command,goal_body_delta


def servo_critic_contract():
    return dict(variant='actual_goal_to_body_servo_v1',format_version=1,
        actor_observation_dim=518,nominal_actor_observation_dim=480,
        replay_action_coordinates='original_absolute_projected_goals_and_binary_jaws',
        critic_action_coordinates='decoded_body_delta_servo_and_binary_jaws',
        critic_action_dim=21,controller_decoder='physical_body_actions.goal_body_command',
        measured_pending_joint_and_torso_targets=True,servo_step_clipping_preserved=True,
        scope='one_step_Q_measured_nstep_Q_actor_Q_and_all_target_Q_branches',
        actor_goal_sampler_entropy_and_actual_controller_unchanged=True,
        base_approach_hold_reward_success_safety_randomization_unchanged=True,
        trained_absolute_goal_Q_and_optimizers_compatible=False)


class BodyServoCriticEncoder:
    def __init__(self,coordinates,center,scale):
        self.coordinates=coordinates
        self.center=torch.as_tensor(center).detach().clone()
        self.scale=torch.as_tensor(scale).detach().clone()
        if self.center.shape!=(21,) or self.scale.shape!=(21,) \
                or not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all() \
                or (self.scale<=0).any():
            raise ValueError('Servo critic requires finite matching21-D goal center/scale')

    @property
    def contract(self):return servo_critic_contract()

    def physical_inputs(self,raw,goals):
        if raw is None or raw.ndim!=2 or raw.shape[1]!=518 \
                or goals.shape!=(len(raw),21) or not torch.isfinite(raw).all() \
                or not torch.isfinite(goals).all() or (goals.abs()>1.00001).any() \
                or not (goals[:,19:].abs()==1).all() \
                or not (raw[:,-6]==1).all():
            raise ValueError('Servo critic needs finite measured518-D held actor states')
        nominal=torch.cat((raw[:,:474],raw[:,-6:]),-1)
        return nominal,self.center.to(raw)+self.scale.to(raw)*goals

    def __call__(self,raw,goals):
        return goal_body_command(self.coordinates,*self.physical_inputs(raw,goals))

    def unclipped_body(self,raw,goals):
        return goal_body_delta(self.coordinates,*self.physical_inputs(raw,goals))

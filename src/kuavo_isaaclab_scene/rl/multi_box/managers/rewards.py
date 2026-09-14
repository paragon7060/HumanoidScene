"""Completion dominates; bounded potential shaping pays no renewable milestones."""
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg as Term
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from ..state import state
from ..kernels import potential_delta


class Progress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.initialized[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        t = state(env)
        height = ((t.centers[..., 2] - env.scene.env_origins[:, None, 2] - t.initial_centers[..., 2])
                  / t.spec.lift_height).clamp(0, 1)
        approach = (torch.exp(-8 * t.distances) * (.5 + .5 * t.alignment.square())).amax(1)
        transport = torch.exp(-t.destination_distance)
        t.conditions()
        extract = torch.exp(-4 * t.extraction_distance)
        features = torch.stack((approach, t.held.float() * height,
            t.held.float() * extract, t.held.float() * transport * t.free_slots.any(-1),
            t.valid_placement.float()), -1)
        weights = torch.tensor((1., 2., 2., 3., 4.), device=env.device)
        per_box = (features * weights).sum(-1)
        if t.spec.strategy == "staged":
            # Same tensor schema, target-conditioned skill objectives.
            per_box = per_box * torch.nn.functional.one_hot(t.target, 4)
        current = per_box.sum(-1)
        reward = potential_delta(self.previous, current, t.success | t.failure, t.spec.discount)
        reward = torch.where(self.initialized & t.ready, reward, 0.)
        self.previous[:] = current
        self.initialized[:] = t.ready
        return reward / env.step_dt


def placed(env):
    return state(env).credit.sum(-1) / env.step_dt


def success(env):
    return state(env).success.float() / env.step_dt


def failure(env):
    return state(env).failure.float() / env.step_dt


def time_cost(env):
    return state(env).ready.float()


def damage(env):
    t = state(env)
    return (t.paid & ~t.valid_placement).sum(-1).float()


@configclass
class RewardsCfg:
    progress = Term(func=Progress, weight=1.)
    placed = Term(func=placed, weight=25.)
    success = Term(func=success, weight=100.)
    failure = Term(func=failure, weight=-50.)
    time = Term(func=time_cost, weight=-.2)
    damage = Term(func=damage, weight=-2.)
    action_rate = Term(func=mdp.action_rate_l2, weight=-.01)
    joint_speed = Term(func=mdp.joint_vel_l2, weight=-.0001)

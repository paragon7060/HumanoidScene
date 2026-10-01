"""Fixed-scene residual SAC pilot over *measured* reference commands.

This is a different, contextual MDP from the ordinary 24-action task. Q sees
the 22 residual coordinates actually issued to this controller. Native replay
continues to record the resulting 24 physical commands. Neither checkpoint nor
replay may be imported into the ordinary SAC trainer as a full policy.
"""

import hashlib
import math
from pathlib import Path

import torch

from ...algorithms.asymmetric_sac import ActorFeatures, AsymmetricReplayBuffer, AsymmetricSAC
from ...algorithms.sac import SACConfig
from ...runners.storage import load_checkpoint, save_checkpoint


def validate_goal_feedback_rates(base,upper,head,torso_speed,dt):
    """Isaac stores uniform action scales as floats and per-joint scales as tensors."""
    for value,expected in [(base,[.15,.15,.5]),(upper,[.01]+[.02]*14),(head,[.01,.01])]:
        actual=torch.as_tensor(value,dtype=torch.float64).reshape(-1)
        if len(actual)==1:
            actual=actual.expand(len(expected))
        target=torch.tensor(expected,dtype=torch.float64,device=actual.device)
        if actual.shape!=target.shape or not torch.allclose(actual,target,atol=1e-8,rtol=1e-6):
            raise ValueError('Goal-feedback rates differ from the measured V2 controller')
    if abs(torso_speed-.1)>1e-9 or abs(dt-1/30)>1e-9:
        raise ValueError('Goal-feedback torso speed/control period differs')


class ReferenceResidual:
    name = 'fixed_measured_reference_residual_sac_v1'
    columns = tuple(i for i in range(24) if i not in (20, 21))

    def __init__(self, commands, scale=.05):
        if commands.ndim != 2 or commands.shape[1] != 24 or not len(commands):
            raise ValueError('Reference requires nonempty measured 24-action commands')
        if not torch.isfinite(commands).all() or (commands.abs() > 1.00001).any():
            raise ValueError('Reference commands must be finite and bounded')
        if not math.isfinite(scale) or not 0 < scale <= .2:
            raise ValueError('Residual scale must be in (0, .2]')
        self.commands, self.scale = commands, scale
        self.features = ActorFeatures(464, 'grasp_target_no_history')

    def context(self, index, count):
        if index < 0:
            raise ValueError('Reference index must be nonnegative')
        command = self.commands[min(index, len(self.commands)-1)].expand(count, -1).clone()
        if index >= len(self.commands):
            # Hold the accumulated PD targets after the measured path ends.
            command[:, self.columns] = 0
        progress = command.new_full((count, 1), min(index, len(self.commands))/len(self.commands))
        return torch.cat((command, progress), -1)

    def observations(self, raw_actor, raw_critic, index):
        if raw_actor.shape[-1] != 464 or raw_critic.shape[-1] != 530:
            raise ValueError('Residual pilot requires current 464/530 physical observations')
        context = self.context(index, len(raw_actor))
        return torch.cat((self.features(raw_actor), context), -1), torch.cat((raw_critic, context), -1)

    def physical_commands(self, residual, index):
        if residual.ndim != 2 or residual.shape[1] != 22 or not torch.isfinite(residual).all():
            raise ValueError('Residual must be a finite 22-action batch')
        if (residual.abs() > 1.00001).any():
            raise ValueError('Residual exceeds squashed policy bounds')
        command = self.context(index, len(residual))[:, :24].clone()
        command[:, self.columns] = (command[:, self.columns] + self.scale*residual).clamp(-1, 1)
        return command


class ReferenceGoalResidual(ReferenceResidual):
    """Anchor all continuous integrators to measured reference controller state."""
    name = 'fixed_measured_reference_goal_residual_sac_v2'

    def __init__(self, measured, scale=.05):
        super().__init__(measured['action'],scale)
        from .joint_offset import JointOffsetController
        from ..geometry.upright_torso import torso_links_from_urdf
        from ....robots.robot_model import resolve_robot_model
        self.joints = JointOffsetController()
        self.reference_actor = torch.cat((measured['actor_obs'],measured['next_actor_obs'][-1:]))
        self.links = self.commands.new_tensor(torso_links_from_urdf(
            resolve_robot_model('s63','leju-twofinger').urdf_path))

    def context(self,index,count):
        context=super().context(index,count)
        context[:,-1]=index/len(self.commands)  # keep elapsed index after the reference ends
        return context

    def physical_commands(self,residual,index,raw_actor):
        from ..demo_replay import _rotation_matrix
        from ..geometry.upright_torso import planar_position
        command=super().physical_commands(residual,index)
        if raw_actor.shape != (len(residual),464) or not (raw_actor[:,439]>.5).all():
            raise ValueError('Goal residual requires valid current controller telemetry')
        reference=self.reference_actor[min(index,len(self.commands))].expand(len(residual),-1)
        target=reference[:,:20]+reference[:,416:436]
        current=raw_actor[:,:20]+raw_actor[:,416:436]
        j=self.joints
        # Correction subtracts the current integrator offset every control step.
        # Residual is thus a bounded position-goal offset, not an accumulating delta.
        command[:,j.action_columns] += (target[:,j.joint_columns]-current[:,j.joint_columns])/command.new_tensor(j.scales)
        command[:,18:20] += (planar_position(target[:,:2],self.links)
                             -planar_position(current[:,:2],self.links))/(.1/30)
        rc=_rotation_matrix(raw_actor[:,71:77]);rr=_rotation_matrix(reference[:,71:77])
        # This difference form is exactly zero for identical source observations.
        delta_rotation=(rc-rr)@rr.transpose(-1,-2)
        translation=(raw_actor[:,68:71]-reference[:,68:71]) \
                    -(delta_rotation@reference[:,68:71,None]).squeeze(-1)
        heading=torch.atan2(delta_rotation[:,1,0],1+delta_rotation[:,0,0])
        pose_error=torch.cat((translation[:,:2],heading[:,None]),-1)
        command[:,:3] += pose_error/command.new_tensor([.15/30,.15/30,.5/30])
        return command.clamp(-1,1)


class ResidualSACPilot:
    """Small isolated pilot; reference dependency remains explicit at deployment."""

    def __init__(self, measured, dataset, directory, *, scale=.05, device='cpu',
                 checkpoint=None, training=True, updates_per_step=2,controller_mode='delta'):
        measured={key:value.to(device) for key,value in measured.items()}
        if controller_mode not in {'delta','goal'}:
            raise ValueError('Unknown residual controller')
        self.controller_mode=controller_mode
        self.controller = (ReferenceGoalResidual(measured,scale) if controller_mode=='goal'
                           else ReferenceResidual(measured['action'],scale))
        self.device, self.training = device, training
        self.directory = Path(directory)
        self.updates_per_step = updates_per_step
        self.actor_updates = self.critic_updates = self.online_rows = 0
        self.latest = {}
        self.online_history = []
        self.reference_sha256 = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
        cfg = SACConfig(hidden=128, actor_lr=3e-5, initial_policy_std=.02,
                        max_policy_std=.05, initial_alpha=1e-5, min_alpha=1e-7,
                        max_alpha=1e-3, entropy_backup=False, critic_layer_norm=True,
                        actor_q_normalize=True, freeze_actor_normalizer=True)
        self.agent = AsymmetricSAC(199, 555, 22, cfg, device=device)
        self.replay = AsymmetricReplayBuffer(10000, 199, 555, 22, device=device)
        seed = []
        for index in range(len(measured['action'])):
            zero=measured['action'].new_zeros((1,22))
            physical=(self.controller.physical_commands(zero,index,measured['actor_obs'][index:index+1])
                      if controller_mode=='goal' else self.controller.physical_commands(zero,index))
            if not torch.equal(physical,measured['action'][index:index+1]):
                raise ValueError('Zero residual does not reproduce actual seed commands exactly')
            ao, co = self.controller.observations(measured['actor_obs'][index:index+1].to(device),
                measured['critic_obs'][index:index+1].to(device), index)
            na, nc = self.controller.observations(measured['next_actor_obs'][index:index+1].to(device),
                measured['next_critic_obs'][index:index+1].to(device), index+1)
            seed.append(dict(actor_obs=ao, critic_obs=co, next_actor_obs=na, next_critic_obs=nc,
                action=ao.new_zeros((1, 22)), reward=measured['reward'][index:index+1].to(device),
                terminated=measured['terminated'][index:index+1].to(device)))
        self.seed = {key: torch.cat([row[key] for row in seed]) for key in seed[0]}
        self.replay.add(**self.seed)
        self.agent.actor_normalizer.update(self.seed['actor_obs'])
        self.agent.critic_normalizer.update(self.seed['critic_obs'])
        if checkpoint:
            saved = load_checkpoint(checkpoint, device=device)
            if saved.get('residual_contract') != self.contract:
                raise ValueError('Residual reference/scale/coordinates differ from checkpoint')
            self.agent.restore(saved, training=training)
            self.actor_updates = saved['actor_updates']
            self.critic_updates = saved['critic_updates']
            experience = Path(checkpoint).parent/'residual_experience.pt'
            if training and experience.exists():
                actual = torch.load(experience, map_location=device, weights_only=True)
                if actual.get('residual_contract') != self.contract:
                    raise ValueError('Previous executed residual experience contract differs')
                rows = actual['executed_transitions']
                for key, storage in self.replay.data.items():
                    if rows[key].shape[1:] != storage.shape[1:] or not torch.isfinite(rows[key]).all():
                        raise ValueError('Previous executed residual experience is invalid')
                if (rows['action'].abs() > 1.00001).any():
                    raise ValueError('Previous residual experience action exceeds bounds')
                self.replay.add(**rows)
                self.online_history.append({key: value.cpu() for key,value in rows.items()})
        else:
            # Zero mean reproduces measured physical commands exactly.
            with torch.no_grad():
                output = self.agent.actor.network[-1]
                output.weight[:22].zero_(); output.bias[:22].zero_()
            if training:
                for _ in range(500):
                    self.latest = self.agent.update(self.replay.sample(256, device), update_actor=False)
                    self.critic_updates += 1

    @property
    def contract(self):
        return dict(name=self.controller.name, reference_sha256=self.reference_sha256,
                    residual_scale=self.controller.scale, physical_action_dim=24,
                    residual_action_dim=22, reference_controls_grippers=True,
                    critic_action_coordinates='issued_residual', fixed_scene_only=True,
                    actor_observation_dim=199, critic_observation_dim=555)

    @torch.no_grad()
    def act(self, raw_actor, raw_critic, index):
        ao, co = self.controller.observations(raw_actor, raw_critic, index)
        residual = self.agent.act(ao, deterministic=not self.training)
        command=(self.controller.physical_commands(residual,index,raw_actor)
                 if self.controller_mode=='goal' else self.controller.physical_commands(residual,index))
        return command, (ao, co, residual)

    @torch.enable_grad()
    def observe(self, previous, next_actor, next_critic, reward, terminated, index):
        if not self.training:
            return
        ao, co, residual = previous
        na, nc = self.controller.observations(next_actor, next_critic, index+1)
        actual = dict(actor_obs=ao, critic_obs=co, action=residual, reward=reward,
                      next_actor_obs=na, next_critic_obs=nc, terminated=terminated)
        self.replay.add(**actual)
        self.online_history.append({key:value.detach().cpu().clone() for key,value in actual.items()})
        self.online_rows += len(reward)
        if self.online_rows < 64:
            return
        for _ in range(self.updates_per_step):
            # A short initialization prior fades; these are actor labels only.
            weight = 5*max(0., 1-self.actor_updates/512)
            ids = torch.randint(len(self.seed['action']), (256,), device=self.device)
            labels = {key: self.seed[key][ids] for key in ('actor_obs', 'action')}
            self.latest = self.agent.update(self.replay.sample(256, self.device),
                demonstration=labels, demonstration_weight=weight)
            self.actor_updates += 1; self.critic_updates += 1

    def save(self, final=False):
        if final and self.online_history:
            # Only actual issued actions/current rewards enter this resume file.
            # It is written once after the final step, never uploaded while open.
            temporary = self.directory/'residual_experience.tmp'
            torch.save(dict(residual_contract=self.contract,
                executed_transitions={key:torch.cat([row[key] for row in self.online_history])
                                      for key in self.online_history[0]},
                proposal_labels_imported=False), temporary)
            temporary.replace(self.directory/'residual_experience.pt')
        return save_checkpoint(self.directory, self.agent.checkpoint() | {
            'residual_contract': self.contract, 'actor_updates': self.actor_updates,
            'critic_updates': self.critic_updates, 'online_rows': self.online_rows,
            'artifact_type': 'fixed_scene_reference_residual_sac',
            'physical_config': 'unchanged_v2_grasp',
        }, iteration=self.actor_updates, keep=None)

    def report(self):
        return dict(residual_contract=self.contract, actor_updates=self.actor_updates,
                    critic_updates=self.critic_updates, online_rows=self.online_rows,
                    training=self.training, optimizer_metrics=self.latest,
                    normalizer_counts=dict(actor=float(self.agent.actor_normalizer.count),
                                           critic=float(self.agent.critic_normalizer.count)))

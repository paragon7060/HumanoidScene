"""Read frozen critics on the actual pre-step held state/action, without HDF."""
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256


class FrozenVideoQ:
    def __init__(self, pilot, indices):
        if pilot.training:
            raise ValueError('Video Q values require a frozen evaluation policy')
        self.pilot, self.indices = pilot, tuple(indices)
        self.counters = (pilot.actor_updates, pilot.critic_updates)
        self.model_sha = structure_sha256(pilot.agent.state_dict())
        self.pending, self.step = {}, None

    def check_counters(self):
        if self.pilot.training or self.counters != (self.pilot.actor_updates, self.pilot.critic_updates):
            raise ValueError('Policy or critic updates occurred during frozen Q video')

    @torch.no_grad()
    def prepare(self, step, ids, previous):
        self.check_counters()
        self.step = step
        self.pending = {i: dict(phase='base_approach', q1=None, q2=None, q_min=None,
                               reason='critic_trained_on_held_grasp_only') for i in self.indices}
        if previous is None:
            if len(ids):
                raise ValueError('Held IDs have no measured state/action tuple')
            return
        ao, co, goals = previous
        if len(set(ids.tolist())) != len(ids) or any(len(v) != len(ids) for v in previous):
            raise ValueError('Held IDs and actual critic state/action rows differ')
        selected = [(row, env_id) for row, env_id in enumerate(ids.tolist()) if env_id in self.indices]
        if not selected:
            return
        agent = self.pilot.agent
        device = next(agent.q1.parameters()).device
        rows = [row for row, _ in selected]
        actor, critic, action = [v[rows].to(device) for v in (ao, co, goals)]
        # Use the same critic normalizer and physical action encoder as SAC.
        qa = torch.cat((agent.critic_normalizer(critic),
                        agent.critic_action_features(actor, action)), -1)
        values = torch.stack((agent.q1(qa).flatten(), agent.q2(qa).flatten()), -1).cpu()
        if values.shape != (len(selected), 2):
            raise ValueError('Frozen critics must return one value per measured action')
        for (_, env_id), pair in zip(selected, values):
            finite = bool(torch.isfinite(pair).all())
            self.pending[env_id] = dict(phase='held_grasp',
                q1=float(pair[0]) if finite else None,
                q2=float(pair[1]) if finite else None,
                q_min=float(pair.min()) if finite else None,
                reason=None if finite else 'nonfinite_Q_estimate')

    def frame(self, step, env_id):
        self.check_counters()
        if step != self.step or env_id not in self.pending:
            raise ValueError('Q state/action and post-action video frame are out of step')
        return dict(control_step=step+1, **self.pending[env_id])

    def close_proof(self):
        self.check_counters()
        if structure_sha256(self.pilot.agent.state_dict()) != self.model_sha:
            raise ValueError('Frozen model changed without an update counter change')
        return dict(model_tensor_SHA256=self.model_sha,
            actor_updates=self.counters[0], critic_updates=self.counters[1],
            frozen_model_unchanged=True,
            semantics='Q(s_t,a_t) before the action, on its measured post-action frame; Q is an estimate, not success',
            approach_Q_unavailable=True, raw_observations_actions_stored=False,
            raw_replay_HDF_read=False, optimizer_updates=0)

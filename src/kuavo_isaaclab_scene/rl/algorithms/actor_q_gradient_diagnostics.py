"""Optional aggregate sensitivity of the existing body-Q actor objective.

Uses the actor update's real stochastic sample and enumerated critics. No new
samples, optimizer steps, simulator queries or transition rows are made.
Exact zero sensitivity is local, not proof of a broken learner.
"""
import torch


class ActorBodyQGradientDiagnostics:
    def __init__(self, interval):
        if type(interval) is not int or interval < 1:
            raise ValueError('A positive actor-proposal interval is required')
        self.interval, self.attempts, self.measurements = interval, 0, 0
        self.latest = None

    def report(self, *, actor_proposal=False):
        return dict(interval=self.interval, proposal_attempts=self.attempts,
            measurements=self.measurements,
            actor_proposal_in_this_update=actor_proposal,
            sampled_on_this_proposal=(actor_proposal and self.latest is not None
                and self.latest['proposal_attempt'] == self.attempts),
            latest_measurement=self.latest)

    def capture(self, agent, raw, body, branch_q, probability, q_scale):
        self.attempts += 1
        if (self.attempts - 1) % self.interval:
            return None
        from ..multi_box.experiments.independent_actor_heads import (
            head_labels, parameter_partitions, raw_head_indices,
        )
        labels = head_labels(agent.actor.network)
        partitions = parameter_partitions(agent.actor)
        scopes = raw_head_indices(agent, raw)
        n = len(raw)
        if not n or body.shape != (n, 19) or branch_q.shape != (n, 4) \
                or probability.shape != (n, 4) or not body.requires_grad:
            raise ValueError('Diagnostics require the actual actor sample and four jaw branches')
        parameters = list(agent.actor.parameters())
        # Hold jaw probabilities fixed: this measures only the body-Q path.
        # A sum gives per-row derivatives; parameter norms below divide by n
        # to match the mean actor objective.
        score = -(probability.detach() * q_scale * branch_q).sum(-1)
        if score.requires_grad:
            derivatives = torch.autograd.grad(score.sum(), (body, *parameters),
                retain_graph=True, allow_unused=True)
        else:
            derivatives = (None,) * (len(parameters) + 1)
        gradient = derivatives[0]
        gradient = torch.zeros_like(body) if gradient is None else gradient.detach()
        result = {}
        for index, label in enumerate(labels):
            rows = scopes == index
            fields = dict(rows=int(rows.sum()))
            for name, columns in (('left_arm', range(1, 8)), ('right_arm', range(8, 15)),
                                  ('body', range(19))):
                values = gradient[rows][:, list(columns)]
                finite = torch.isfinite(values).all(-1)
                fields[name] = dict(
                    finite_rows=int(finite.sum()),
                    exact_zero_rows=int((finite & (values == 0).all(-1)).sum()),
                    mean_row_gradient_L2=(float(torch.linalg.vector_norm(values[finite].double(), dim=-1).mean())
                        if bool(finite.any()) else None))
            terms = [derivatives[j+1].detach()
                     for j in partitions[label] if derivatives[j+1] is not None]
            nonfinite = sum(int((~torch.isfinite(term)).sum()) for term in terms)
            fields['nonfinite_parameter_gradient_elements'] = nonfinite
            fields['mean_body_Q_objective_parameter_gradient_L2'] = (
                None if nonfinite else
                float(torch.stack([term.double().square().sum() for term in terms]).sum().sqrt() / n)
                if terms else 0.)
            result[label] = fields
        self.measurements += 1
        self.latest = dict(name='existing_stochastic_body_Q_local_gradient_v1',
            proposal_attempt=self.attempts, measurement=self.measurements,
            interval=self.interval, rows=n, scopes=result,
            Q_scale=float(q_scale), goal_gradient_coordinates='normalized_requested_body_goals19',
            sampled_before_actor_guard=True, jaw_probabilities_held_fixed=True,
            entropy_and_imitation_and_discrete_probability_gradients_excluded=True,
            no_new_policy_sample_or_optimizer_or_raw_transition_output=True,
            actual_parameter_movement_or_task_success_proved=False)
        return self.latest

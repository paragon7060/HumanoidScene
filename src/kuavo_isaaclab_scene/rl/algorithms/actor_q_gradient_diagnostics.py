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
        self._pending = None

    def report(self, *, actor_proposal=False):
        return dict(interval=self.interval, proposal_attempts=self.attempts,
            measurements=self.measurements,
            actor_proposal_in_this_update=actor_proposal,
            sampled_on_this_proposal=(actor_proposal and self.latest is not None
                and self.latest['proposal_attempt'] == self.attempts),
            latest_measurement=self.latest)

    def capture(self, agent, raw, body, branch_q, probability, q_scale):
        self.attempts += 1
        self._pending = None
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
        self._pending = dict(parameter_ids=[id(p) for p in parameters],
            body_Q_gradients=[torch.zeros_like(p) if derivative is None else
                derivative.detach().clone()/n for p, derivative in zip(parameters, derivatives[1:])],
            partitions=partitions)
        return self.latest

    def complete_actor_objective(self, agent, actor_loss, success_retention_loss=None,
                                 *, successful_train_rows=0, sampled_regularization_loss=None):
        """Compare the sampled Q path with losses actually supplied to Adam.

        autograd.grad leaves parameter .grad, optimizers and the original
        backward graph untouched. Only aggregate scalars leave this method.
        A conflicting local gradient is not proof of an incorrect objective,
        failed guard acceptance, or failed physical rollout.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        parameters = list(agent.actor.parameters())
        if pending['parameter_ids'] != [id(p) for p in parameters]:
            raise ValueError('Actor parameters changed between Q and objective diagnostics')

        def gradients(loss):
            parts = (torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
                     if loss is not None and loss.requires_grad else [None]*len(parameters))
            return [torch.zeros_like(p) if derivative is None else derivative.detach()
                    for p, derivative in zip(parameters, parts)]

        total = gradients(actor_loss)
        retention = gradients(success_retention_loss)
        sampled_regularization = (gradients(sampled_regularization_loss)
                                  if sampled_regularization_loss is not None else None)
        scopes = {}
        regularization_scopes = {}
        for label, indices in pending['partitions'].items():
            vectors = [torch.cat([parts[index].double().flatten() for index in indices])
                       for parts in (pending['body_Q_gradients'], total, retention)]
            bad = sum(int((~torch.isfinite(vector)).sum()) for vector in vectors)
            fields = dict(nonfinite_gradient_elements=bad)
            if not bad:
                q, full, success = vectors
                qnorm, fullnorm, successnorm = [float(v.norm()) for v in vectors]
                qfull, qsuccess = float(q@full), float(q@success)
                fields.update(body_Q_parameter_gradient_L2=qnorm,
                    total_actor_parameter_gradient_L2=fullnorm,
                    weighted_success_retention_parameter_gradient_L2=successnorm,
                    body_Q_dot_total_actor_gradients=qfull,
                    body_Q_dot_weighted_success_retention_gradients=qsuccess,
                    cosine_body_Q_and_total_actor_gradients=(qfull/(qnorm*fullnorm)
                        if qnorm and fullnorm else None),
                    cosine_body_Q_and_success_retention_gradients=(qsuccess/(qnorm*successnorm)
                        if qnorm and successnorm else None))
            scopes[label] = fields
            if sampled_regularization is not None:
                cost = torch.cat([sampled_regularization[index].double().flatten()
                                  for index in indices])
                bad_cost = int((~torch.isfinite(cost)).sum())
                cost_fields = dict(nonfinite_gradient_elements=bad + bad_cost)
                if not bad and not bad_cost:
                    costnorm = float(cost.norm())
                    qcost, successcost = float(q@cost), float(success@cost)
                    cost_fields.update(weighted_sampled_regularization_parameter_gradient_L2=costnorm,
                        body_Q_parameter_gradient_L2=qnorm,
                        weighted_success_retention_parameter_gradient_L2=successnorm,
                        body_Q_dot_weighted_sampled_regularization_gradients=qcost,
                        weighted_success_retention_dot_sampled_regularization_gradients=successcost,
                        cosine_body_Q_and_sampled_regularization_gradients=(qcost/(qnorm*costnorm)
                            if qnorm and costnorm else None),
                        cosine_success_retention_and_sampled_regularization_gradients=(successcost/(successnorm*costnorm)
                            if successnorm and costnorm else None))
                regularization_scopes[label] = cost_fields
        result = dict(name='actual_actor_Q_retention_gradient_alignment_v1',
            proposal_attempt=self.attempts, successful_TRAIN_actor_rows=successful_train_rows,
            scopes=scopes, weighted_success_loss_excludes_teacher_and_entropy=True,
            total_loss_includes_all_actual_actor_objective_terms=True,
            before_Adam_projection_and_guard=True,
            scalar_loss_size_NOT_parameter_gradient_dominance=True,
            cosine_or_gradient_NOT_actual_Adam_direction_or_task_success=True,
            no_new_samples_optimizer_steps_raw_rows_or_parameter_grad_mutation=True)
        self.latest['actor_objective_gradient_alignment'] = result
        if sampled_regularization is not None:
            self.latest['sampled_actor_regularization_gradient_alignment'] = dict(
                name='actual_same_sample_actor_regularization_gradient_alignment_v1',
                proposal_attempt=self.attempts, scopes=regularization_scopes,
                weighted_loss_is_actual_actor_sample_regularization=True,
                before_Adam_projection_and_guard=True,
                not_actual_parameter_movement_or_force_or_task_success=True,
                no_new_samples_optimizer_steps_raw_rows_or_parameter_grad_mutation=True)
        return result

"""Register complete, physically verified TRAIN episodes without wave-end lag."""
from copy import deepcopy

from .staged_train_success import add_completed_training_wave, validate_success_outcome


class CompletedTrainSuccessRegistration:
    def __init__(self, bank, *, source_run, after_success_added=None, mode='completed-episode'):
        if mode not in ('completed-episode', 'wave'):
            raise ValueError('Unknown TRAIN success registration timing')
        self.bank, self.source_run = bank, source_run
        self.after_success_added, self.mode = after_success_added, mode
        self.registered = {}
        self.events = []

    def register(self, wave, outcomes, measured_batches, *, control_step,
                 actor_updates, critic_updates, at_wave_end=False):
        if wave['split'] != 'train':
            # Preserve the original evaluation/data separation check.
            return add_completed_training_wave(self.bank, wave, outcomes, measured_batches,
                source_run=self.source_run)
        if self.mode == 'wave' and not at_wave_end:
            return 0
        added = 0
        for outcome in outcomes:
            environment = outcome['environment']
            identity = (outcome['wave'], environment, outcome['layout']['seed'],
                outcome['layout']['target_region'], outcome['layout'].get('target_box_type'))
            if environment in self.registered:
                if self.registered[environment] != identity:
                    raise ValueError('Completed environment identity changed within one wave')
                validate_success_outcome('train', outcome)
                continue
            if not (outcome.get('result') or {}).get('success'):
                continue
            validate_success_outcome('train', outcome)

            def after_added():
                # Mark insertion before the cohort callback: accepted episodes
                # cannot be inserted twice, including after bank eviction.
                self.registered[environment] = identity
                self.events.append(dict(environment=environment, wave=outcome['wave'],
                    seed=outcome['layout']['seed'], region=outcome['layout']['target_region'],
                    box_type=outcome['layout'].get('target_box_type'),
                    terminal_control_step=outcome['result'].get('steps'),
                    registration_control_step=control_step,
                    actor_updates=actor_updates, critic_updates=critic_updates,
                    boundary='wave_end' if at_wave_end else 'completed_episode'))
                if self.after_success_added is not None:
                    self.after_success_added()

            added += add_completed_training_wave(self.bank, wave, [outcome], measured_batches,
                source_run=self.source_run, after_success_added=after_added)
        return added

    def report(self):
        return deepcopy(dict(mode=self.mode, registered_completed_TRAIN_episodes=len(self.events),
            events=self.events, registration_after_current_step_learner_observation=True,
            remaining_environment_resets_or_physics_changed=False,
            rewards_labels_success_safety_and_bank_capacity_unchanged=True,
            DEV_FINAL_import_allowed=False))

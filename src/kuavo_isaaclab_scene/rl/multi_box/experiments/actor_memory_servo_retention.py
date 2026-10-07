"""Conservative SAC with a separate, actor-only memory of past TRAIN successes."""
from copy import deepcopy

from .conservative_servo_retention import ConservativeServoRetentionSACPilot
from .actor_train_memory import (
    ActorTrainMemory, actor_memory_contract, compatibility_contract, structure_sha256,
)


class ActorMemoryServoRetentionSACPilot(ConservativeServoRetentionSACPilot):
    artifact_type = 'staged_actual_flap_actor_memory_conservative_servo_retention_sac_v1'

    def configure_controller(self, saved):
        super().configure_controller(saved)
        if saved is None or 'actor_training_memory' not in saved:
            raise ValueError('Verified actor-only successful TRAIN memory is required')
        self.actor_memory = ActorTrainMemory(saved['actor_training_memory'],
            compatibility=compatibility_contract(self.contract),
            frozen_anchor_SHA256=structure_sha256(self.body_anchor_state))
        self.actor_memory_initialization = deepcopy(saved.get('actor_memory_initialization'))
        if self.actor_memory_initialization is not None:
            origin = self.actor_memory_initialization
            if (not isinstance(origin, dict)
                    or origin.get('kind') != 'offline_actor_commands_from_own_successful_TRAIN_v1'
                    or origin.get('actor_memory_structure_SHA256') != structure_sha256(self.actor_memory.state())
                    or origin.get('frozen_body_anchor_SHA256') != structure_sha256(self.body_anchor_state)
                    or type(origin.get('offline_actor_steps')) is not int
                    or not 1 <= origin['offline_actor_steps'] <= 10000):
                raise ValueError('Actor initialization provenance differs from successful TRAIN memory')
            if saved['actor_updates'] == 0:
                actor = {k:v for k,v in saved['model'].items() if k.startswith('actor.')}
                if origin.get('fitted_actor_model_SHA256') != structure_sha256(actor):
                    raise ValueError('Unstarted actor differs from its offline initialization')

    @property
    def contract(self):
        return super().contract | dict(actor_training_memory=actor_memory_contract())

    def successful_actor_options(self, update_actor):
        if not update_actor:
            options, stats = super().successful_actor_options(False)
            return options, stats | dict(previous_success_actor_memory_rows=0,
                new_success_actor_memory_rows=0, previous_success_memory_rows_in_Q_batch=0)
        batch, stats = self.actor_memory.sample_actor(64, self.device, self.success_bank)
        return dict(successful_train=batch,
            success_goal_weight=self.success_bank.config['actor_goal_mse_weight']/self.radius**2,
            success_jaw_weight=self.success_bank.config['actor_jaw_nll_weight']), stats

    def checkpoint_extras(self):
        extras = super().checkpoint_extras() | dict(actor_training_memory=self.actor_memory.state())
        if self.actor_memory_initialization is not None:
            extras['actor_memory_initialization'] = deepcopy(self.actor_memory_initialization)
        return extras

    def experience_extras(self):
        extras = super().experience_extras() | dict(actor_training_memory=self.actor_memory.state())
        if self.actor_memory_initialization is not None:
            extras['actor_memory_initialization'] = deepcopy(self.actor_memory_initialization)
        return extras

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if 'actor_training_memory' not in state \
                or structure_sha256(state['actor_training_memory']) != structure_sha256(self.actor_memory.state()):
            raise ValueError('Checkpoint and replay actor-only memory provenance differ')
        if state.get('actor_memory_initialization') != self.actor_memory_initialization:
            raise ValueError('Checkpoint and replay offline actor initialization differ')

    def report(self):
        report = super().report() | dict(previous_successful_TRAIN_actor_memory=self.actor_memory.report())
        if self.actor_memory_initialization is not None:
            report['actor_memory_initialization'] = deepcopy(self.actor_memory_initialization)
        return report

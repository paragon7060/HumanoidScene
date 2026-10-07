"""Conservative SAC with a separate, actor-only memory of past TRAIN successes."""
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
        return super().checkpoint_extras() | dict(actor_training_memory=self.actor_memory.state())

    def experience_extras(self):
        return super().experience_extras() | dict(actor_training_memory=self.actor_memory.state())

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if 'actor_training_memory' not in state \
                or structure_sha256(state['actor_training_memory']) != structure_sha256(self.actor_memory.state()):
            raise ValueError('Checkpoint and replay actor-only memory provenance differ')

    def report(self):
        return super().report() | dict(previous_successful_TRAIN_actor_memory=self.actor_memory.report())

"""Reanchored actual-flap SAC with an opt-in physical servo critic encoder."""
from .actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from .servo_critic import BodyServoCriticEncoder


class ServoCriticReanchoredSACPilot(ReanchoredActualFlapSACPilot):
    artifact_type='staged_actual_flap_reanchored_servo_critic_hybrid_sac_v1'

    def configure_controller(self,saved):
        super().configure_controller(saved)
        self.agent.goal_servo_critic_encoder=BodyServoCriticEncoder(
            self.coordinates,self.center,self.scale)

    @property
    def contract(self):
        return super().contract|dict(critic_action_encoding=self.agent.goal_servo_critic_encoder.contract)

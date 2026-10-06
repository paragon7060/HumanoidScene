"""Same physical goals and servo Q with a fixed quarter-sized Gaussian."""
from .actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
from .body_policy_spread import (
    body_policy_spread_contract,quarter_body_policy_config,validate_quarter_policy_state,
)


class GentleServoCriticSACPilot(ServoCriticReanchoredSACPilot):
    artifact_type='staged_actual_flap_reanchored_gentle_servo_critic_hybrid_sac_v1'

    def __init__(self,*args,checkpoint=None,device='cpu',**kwargs):
        if checkpoint is not None:
            import torch
            state=torch.load(checkpoint,map_location=device,weights_only=True)
            if state.get('artifact_type')!=self.artifact_type:
                raise ValueError('Quarter Gaussian cannot resume another policy or its Q')
            validate_quarter_policy_state(state)
        super().__init__(*args,checkpoint=checkpoint,device=device,**kwargs)

    def learning_config(self,config):
        return quarter_body_policy_config(super().learning_config(config))

    @property
    def contract(self):
        return super().contract|dict(body_policy_spread=body_policy_spread_contract())

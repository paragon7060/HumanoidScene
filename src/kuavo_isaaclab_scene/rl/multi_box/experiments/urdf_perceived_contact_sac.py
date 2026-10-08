"""Full-arm SAC with deployable, TRAIN-only local contact exploration."""
from copy import deepcopy

from .urdf_strong_success_sac import URDFStrongSuccessSACPilot, STRONG_SERVO_COEFFICIENT
from .urdf_full_arm_sac import validate_full_arm_state
from .body_behavior_exploration import GENTLE_GREEDY_REST_VARIANT, body_behavior_config
from .perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract, contact_statistics,
)


class URDFPerceivedContactSACPilot(URDFStrongSuccessSACPilot):
    artifact_type='staged_actual_flap_URDF_full_arm_perceived_contact_exploration_sac_v1'
    settled_close=False
    precise_feedback=False
    motion_feedback=False
    upright_feedback=False
    interior_contact=False

    def __init__(self,*args,**kwargs):
        self.contact_explorer=None
        self._contact_statistics=contact_statistics()
        self._contact_num_envs=1
        kwargs.setdefault('body_behavior',GENTLE_GREEDY_REST_VARIANT)
        super().__init__(*args,**kwargs)
        if self.body_behavior!=body_behavior_config(GENTLE_GREEDY_REST_VARIANT):
            raise ValueError('Contact exploration requires the existing gentle20percent selection')

    def validate_saved_state(self,state):
        validate_perceived_contact_state(state,settled_close=self.settled_close,precise_feedback=self.precise_feedback,motion_feedback=self.motion_feedback)
        self._contact_statistics=contact_statistics(state['perceived_contact_statistics'])

    @property
    def contract(self):
        result=super().contract
        if self._URDF_configuring_source:return result
        return result|dict(TRAIN_perceived_contact_exploration=perceived_contact_contract(settled_close=self.settled_close,precise_feedback=self.precise_feedback,motion_feedback=self.motion_feedback,upright_feedback=self.upright_feedback,interior_contact=self.interior_contact))

    def reset_exploration(self,num_envs):
        if self.contact_explorer is not None:self._contact_statistics=self.contact_explorer.report()
        super().reset_exploration(num_envs)
        self.contact_explorer=None;self._contact_num_envs=num_envs

    def act(self,raw,critic,index,**kwargs):
        result=super().act(raw,critic,index,**kwargs)
        if not self.training or self.body_behavior_sampler is None:return result
        ids=kwargs.get('exploration_ids')
        if ids is None:
            import torch
            ids=torch.arange(len(raw),device=raw.device)
        chosen=self.body_behavior_sampler.selected[ids]
        if not chosen.any():return result
        if self.contact_explorer is None:
            self.contact_explorer=PerceivedContactExploration(self._contact_num_envs,raw,self._contact_statistics,
                settled_close=self.settled_close,precise_feedback=self.precise_feedback,motion_feedback=self.motion_feedback,upright_feedback=self.upright_feedback,interior_contact=self.interior_contact)
        return self.contact_explorer.step(self,raw,result,ids,kwargs.get('supplemental'),chosen,index)

    def contact_extras(self):
        return dict(perceived_contact_statistics=(self.contact_explorer.report() if self.contact_explorer
            is not None else deepcopy(self._contact_statistics)))

    def checkpoint_extras(self):return super().checkpoint_extras()|self.contact_extras()
    def experience_extras(self):return super().experience_extras()|self.contact_extras()

    def restore_experience_extras(self,state):
        super().restore_experience_extras(state)
        if contact_statistics(state.get('perceived_contact_statistics'))!=self._contact_statistics:
            raise ValueError('Contact collection checkpoint/replay statistics differ')

    def report(self):
        return super().report()|self.contact_extras()|dict(
            TRAIN_perceived_contact_exploration=perceived_contact_contract(settled_close=self.settled_close,precise_feedback=self.precise_feedback,motion_feedback=self.motion_feedback,upright_feedback=self.upright_feedback,interior_contact=self.interior_contact),
            evaluated_policy_never_uses_contact_explorer=True)


class URDFSettledContactSACPilot(URDFPerceivedContactSACPilot):
    artifact_type='staged_actual_flap_URDF_full_arm_perceived_settled_contact_exploration_sac_v2'
    settled_close=True


class URDFPreciseFeedbackSACPilot(URDFSettledContactSACPilot):
    artifact_type='staged_actual_flap_URDF_full_arm_perceived_precise_feedback_exploration_sac_v3'
    precise_feedback=True


class URDFMotionFeedbackSACPilot(URDFPreciseFeedbackSACPilot):
    artifact_type='staged_actual_flap_URDF_full_arm_perceived_motion_feedback_exploration_sac_v4'
    motion_feedback=True


def validate_perceived_contact_state(state,*,settled_close=False,precise_feedback=False,motion_feedback=False):
    expected=perceived_contact_contract(settled_close=settled_close,precise_feedback=precise_feedback,motion_feedback=motion_feedback)
    artifact=(URDFMotionFeedbackSACPilot if motion_feedback else URDFPreciseFeedbackSACPilot if precise_feedback else
        URDFSettledContactSACPilot if settled_close else URDFPerceivedContactSACPilot).artifact_type
    validate_full_arm_state(state,artifact_type=artifact,
        servo_coefficient=STRONG_SERVO_COEFFICIENT)
    if state['goal_contract'].get('TRAIN_perceived_contact_exploration')!=expected \
            or state.get('body_behavior')!=body_behavior_config(GENTLE_GREEDY_REST_VARIANT):
        raise ValueError('Saved contact exploration input/selection contract differs')
    if 'perceived_contact_statistics' not in state:
        raise ValueError('Saved contact collection statistics missing')
    contact_statistics(state['perceived_contact_statistics'])

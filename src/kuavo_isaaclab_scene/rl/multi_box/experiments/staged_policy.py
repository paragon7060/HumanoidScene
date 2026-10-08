"""Shared checkpoint dispatch for staged training and frozen video replay."""

from .staged_goal_sac import StagedGoalSACPilot
from .staged_hybrid_goal_sac import StagedHybridGoalSACPilot
from .physical_body_sac import PhysicalBodySACPilot
from .actual_flap_residual_sac import ActualFlapResidualSACPilot
from .actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from .actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
from .gentle_servo_critic_sac import GentleServoCriticSACPilot
from .servo_success_retention import ServoRetentionGentleSACPilot
from .conservative_servo_retention import ConservativeServoRetentionSACPilot
from .actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from .regional_actor_servo import RegionalActorMemorySACPilot
from .support_conservative_sac import SupportConservativeRegionalSACPilot
from .urdf_regional_goal_sac import URDFRegionalGoalSACPilot
from .urdf_servo_guard_sac import URDFServoGuardSACPilot
from .urdf_full_arm_sac import URDFFullArmSACPilot
from .urdf_strong_success_sac import URDFStrongSuccessSACPilot
from .urdf_perceived_contact_sac import URDFPerceivedContactSACPilot,URDFSettledContactSACPilot,URDFPreciseFeedbackSACPilot,URDFMotionFeedbackSACPilot
from .urdf_upright_contact_sac import URDFUprightContactSACPilot


def staged_policy_class(artifact_type):
    return {cls.artifact_type: cls for cls in (
        StagedGoalSACPilot, StagedHybridGoalSACPilot, PhysicalBodySACPilot,ActualFlapResidualSACPilot,
        ReanchoredActualFlapSACPilot,
        ServoCriticReanchoredSACPilot,
        GentleServoCriticSACPilot,
        ServoRetentionGentleSACPilot,
        ConservativeServoRetentionSACPilot,
        ActorMemoryServoRetentionSACPilot,
        RegionalActorMemorySACPilot,
        SupportConservativeRegionalSACPilot,
        URDFRegionalGoalSACPilot,
        URDFServoGuardSACPilot,
        URDFFullArmSACPilot,
        URDFStrongSuccessSACPilot,
        URDFPerceivedContactSACPilot,
        URDFSettledContactSACPilot,
        URDFPreciseFeedbackSACPilot,
        URDFMotionFeedbackSACPilot,
        URDFUprightContactSACPilot,
    )}.get(artifact_type)


def staged_policy_metadata(pilot):
    key = ('physical_body_contract' if isinstance(pilot, PhysicalBodySACPilot)
           else 'goal_contract')
    return {key: pilot.contract}

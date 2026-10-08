"""Explicit full-arm SAC comparison retaining actual successful servo commands.

Only the successful TRAIN interval coefficient changes. Policy sampling,
Q/returns, replay, physical commands, task and default profiles stay intact.
"""
from .urdf_full_arm_sac import (
    FullArmServoGuardSAC, URDFFullArmSACPilot, validate_full_arm_state,
)


STRONG_SERVO_COEFFICIENT = 1.


class StrongSuccessFullArmSAC(FullArmServoGuardSAC):
    success_servo_interval_coefficient = STRONG_SERVO_COEFFICIENT


class URDFStrongSuccessSACPilot(URDFFullArmSACPilot):
    artifact_type = 'staged_actual_flap_URDF_full_arm_strong_success_servo_hybrid_sac_v1'
    agent_class = StrongSuccessFullArmSAC

    def validate_saved_state(self, state):
        validate_strong_success_state(state)


def validate_strong_success_state(state):
    validate_full_arm_state(state, artifact_type=URDFStrongSuccessSACPilot.artifact_type,
                            servo_coefficient=STRONG_SERVO_COEFFICIENT)

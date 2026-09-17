"""V2 shadow diagnostics that never drive control."""

from .shadow_reward import (
    PoseShadowRewardEvaluator,
    ShadowRewardLogger,
    ShadowRewardStats,
    format_shadow_reward,
)

__all__ = (
    "PoseShadowRewardEvaluator",
    "ShadowRewardLogger",
    "ShadowRewardStats",
    "format_shadow_reward",
)

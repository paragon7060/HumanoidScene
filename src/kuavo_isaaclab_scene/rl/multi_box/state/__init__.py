"""Internal task-state boundary; concrete fields are intentionally pending."""

from .api import StateProvider
from .placement import PlacementSetState, PlacementSetTracker
from .schema import (
    DeployableBoxState,
    DeployableRobotState,
    DeployableTaskState,
    MultiBoxState,
    PrivilegedTaskState,
)

__all__ = (
    "DeployableBoxState",
    "DeployableRobotState",
    "DeployableTaskState",
    "MultiBoxState",
    "PlacementSetState",
    "PlacementSetTracker",
    "PrivilegedTaskState",
    "StateProvider",
)

"""Isaac Lab v2.3 raw OpenXR tracking adapter for the Kuavo collector.

Isaac Lab v2.3.2 feature-gates OpenXR queries from retargeter requirements.
The upstream device also documents a raw-data mode without retargeters, but an
empty retargeter list leaves the feature set empty.  Kuavo performs its own
safe bimanual mapping, so this adapter explicitly requests hand and head
tracking while preserving the upstream raw dictionary format.

Import this module only after :class:`isaaclab.app.AppLauncher` has started
Isaac Sim; the launcher adds Kit modules such as ``carb`` to Python's path.
"""

from __future__ import annotations

from typing import Any

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase


class RawQuestOpenXRDevice(OpenXRDevice):
    """Return raw Quest hand/head poses on Isaac Lab v2.3.2."""

    _KUAVO_REQUIRED_FEATURES = frozenset(
        {
            RetargeterBase.Requirement.HAND_TRACKING,
            RetargeterBase.Requirement.HEAD_TRACKING,
        }
    )

    def __init__(self, cfg: OpenXRDeviceCfg) -> None:
        super().__init__(cfg)
        required_features = getattr(self, "_required_features", None)
        if not isinstance(required_features, set):
            raise RuntimeError(
                "Unsupported Isaac Lab OpenXRDevice internals: expected a mutable "
                "_required_features set. Run ./doctor.sh and use Isaac Lab v2.3.2."
            )
        required_features.update(self._KUAVO_REQUIRED_FEATURES)

    def advance(self) -> dict[Any, Any]:
        """Poll the upstream device without applying an Isaac Lab retargeter."""
        raw = self._get_raw_data()
        if not isinstance(raw, dict):
            raise RuntimeError(f"OpenXR raw tracking returned {type(raw).__name__}, expected dict.")
        return raw

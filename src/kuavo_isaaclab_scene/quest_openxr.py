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

import os
from pathlib import Path
import time
from typing import Any, Callable

from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase


def start_quest_xr_session(simulation_app, *, enable_ui: bool = True, resolution_scale: float = 1.0,
                           render_quality: str = "performance", timeout_seconds: float = 90.0) -> None:
    """Start Kit's XR output, which loading the OpenXR experience does not start.

    Use the explicitly configured runtime instead of a previously saved Kit
    runtime choice. Pump Kit updates before creating head-locked UI layers.
    """
    import carb.settings
    from omni.kit.xr.core import XRCore

    runtime_path = os.environ.get("XR_RUNTIME_JSON")
    if not runtime_path or not Path(runtime_path).expanduser().is_file():
        raise RuntimeError("Set XR_RUNTIME_JSON to the installed CloudXR runtime manifest before starting Quest XR.")
    runtime_path = str(Path(runtime_path).expanduser().resolve())
    settings = carb.settings.get_settings()
    core = XRCore.get_singleton()
    if core is None:
        raise RuntimeError("Kit XRCore is unavailable; launch the Isaac Lab OpenXR experience.")
    settings.set_bool("/xr/ui/enabled", enable_ui)
    settings.set_float("/persistent/xr/profile/ar/render/resolutionMultiplier", resolution_scale)
    # XR has its own render preset, applied after SimulationCfg. Its default
    # balanced preset otherwise restores four lighting samples per pixel.
    settings.set_string("/persistent/xr/profile/ar/renderQuality", render_quality)
    print(f"[XR] Render resolution multiplier: {resolution_scale:.2f}; factory materials unchanged.", flush=True)
    if not core.is_xr_enabled():
        settings.set_string("/persistent/xr/system/openxr/activeRuntimeJSON", runtime_path)
        settings.set_string("/persistent/xr/system/openxr/runtime", "custom")
        settings.set_string("/persistent/xr/profile/ar/system/display", "OpenXR")
        print(f"[XR] Starting OpenXR output using {runtime_path}", flush=True)
        core.request_enable_profile("ar")

    deadline = time.monotonic() + timeout_seconds
    while simulation_app.is_running():
        simulation_app.update()
        if core.is_xr_enabled() and core.is_xr_display_enabled():
            print("[XR] OpenXR session and display are active.", flush=True)
            return
        if time.monotonic() >= deadline:
            break
    raise RuntimeError(
        "OpenXR output did not become active. Keep the Quest client connected and check the Kit/CloudXR logs; "
        "loading the scene alone does not confirm an XR connection."
    )


class RawQuestOpenXRDevice(OpenXRDevice):
    """Return raw Quest head and selected hand/controller input on Isaac Lab v2.3.2."""

    def __init__(self, cfg: OpenXRDeviceCfg, *, input_mode: str = "hands") -> None:
        if input_mode not in {"hands", "controllers"}:
            raise ValueError("input_mode must be hands or controllers.")
        super().__init__(cfg)
        required_features = getattr(self, "_required_features", None)
        if not isinstance(required_features, set):
            raise RuntimeError(
                "Unsupported Isaac Lab OpenXRDevice internals: expected a mutable "
                "_required_features set. Run ./doctor.sh and use Isaac Lab v2.3.2."
            )
        required_features.add(RetargeterBase.Requirement.HEAD_TRACKING)
        required_features.add(
            RetargeterBase.Requirement.MOTION_CONTROLLER if input_mode == "controllers"
            else RetargeterBase.Requirement.HAND_TRACKING
        )
        # Reserve A for the collector's explicit motion control instead of
        # Isaac Lab's optional anchor-rotation shortcut.
        self._unbind_all_buttons()
        self._physical_to_control = None

    def reset_head_reference(self):
        self._physical_to_control = None

    def reset(self):
        super().reset()
        self.reset_head_reference()

    def motion_head_pose(self, valid_world_pose):
        """Head motion without feedback from the moving robot-view anchor."""
        if valid_world_pose is None:
            return None
        head = self._xr_core.get_input_device("/user/head")
        physical = head.get_pose()
        if self._physical_to_control is None:
            self._physical_to_control = physical.GetInverse() * head.get_virtual_world_pose()
        pose = physical * self._physical_to_control
        q = pose.ExtractRotationQuat()
        import numpy as np
        return np.array([*pose.ExtractTranslation(), q.GetReal(), *q.GetImaginary()])

    def _query_controller(self, input_device):
        packet = super()._query_controller(input_device)
        if packet.size and input_device.has_pose("grip"):
            # Kit's unnamed default pose can be "aim". Arm tracking must
            # use the physical grip pose, and validate that same named pose.
            pose = input_device.get_virtual_world_pose("grip")
            position = pose.ExtractTranslation()
            quat = pose.ExtractRotationQuat()
            packet[0] = [*position, quat.GetReal(), *quat.GetImaginary()]
        return packet

    def bind_button(self, hand: str, button: str, callback: Callable[[], None]) -> None:
        """Bind a controller press, including devices that connect later."""
        if hand not in {"left", "right"}:
            raise ValueError("Controller hand must be left or right.")
        def on_press(_event) -> None:
            print(f"[BUTTON] {button.upper()} pressed ({hand} controller).", flush=True)
            callback()

        self._bind_button_press(f"/user/hand/{hand}", button, f"kuavo_{hand}_{button}", on_press)

    def recenter_view(self, camera_position_w, camera_quat_w_opengl) -> None:
        """Place the current physical viewpoint at the Kuavo head camera."""
        from omni.kit.xr.core import XRCore
        from pxr import Gf

        core = XRCore.get_singleton()
        if not core.is_xr_enabled():
            raise RuntimeError("Cannot recenter while the XR session is inactive.")
        # Sensor paths contain environment regular expressions; use the live
        # camera pose instead of treating a regex as a USD prim path.
        position = [float(value) for value in camera_position_w]
        w, x, y, z = (float(value) for value in camera_quat_w_opengl)
        view_pose = Gf.Matrix4d(1.0)
        view_pose.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        view_pose.SetTranslateOnly(Gf.Vec3d(*position))
        core.schedule_teleport_to_view(self._xr_anchor_headset_path, view_pose)

    def pin_view_position(self, camera_position_w, body_yaw_delta=0.0):
        """Attach the eye position without cancelling low-latency HMD rotation."""
        import math
        from pxr import Gf
        head = self._xr_core.get_input_device("/user/head")
        if head is None:
            return
        rotation = head.get_virtual_world_pose().ExtractRotationQuat()
        rotation = Gf.Quatd(math.cos(body_yaw_delta / 2), Gf.Vec3d(0, 0, math.sin(body_yaw_delta / 2))) * rotation
        self.recenter_view(camera_position_w, [rotation.GetReal(), *rotation.GetImaginary()])

    def advance(self) -> dict[Any, Any]:
        """Poll the upstream device without applying an Isaac Lab retargeter."""
        raw = self._get_raw_data()
        if not isinstance(raw, dict):
            raise RuntimeError(f"OpenXR raw tracking returned {type(raw).__name__}, expected dict.")
        from omni.kit.xr.core import XRPoseValidityFlags

        valid_flags = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        for target, path, pose_name in (
            (self.TrackingTarget.HAND_LEFT, "/user/hand/left", "wrist"),
            (self.TrackingTarget.HAND_RIGHT, "/user/hand/right", "wrist"),
            (self.TrackingTarget.HEAD, "/user/head", ""),
            (self.TrackingTarget.CONTROLLER_LEFT, "/user/hand/left", "grip"),
            (self.TrackingTarget.CONTROLLER_RIGHT, "/user/hand/right", "grip"),
        ):
            if target not in raw:
                continue
            device = self._xr_core.get_input_device(path)
            # Upstream reuses previous poses when tracking is lost. Do not
            # report those cached poses as live tracking to the safety mapper.
            if not device or not device.has_pose(pose_name):
                raw[target] = None
            elif device.get_virtual_world_pose_desc(pose_name).validity_flags & valid_flags != valid_flags:
                raw[target] = None
            elif target in {self.TrackingTarget.CONTROLLER_LEFT, self.TrackingTarget.CONTROLLER_RIGHT}:
                # A hand-only device can have a default pose too. Require the
                # controller trigger instead of silently switching input modes.
                if not device.has_input_gesture("trigger", "value"):
                    raw[target] = None
        return raw

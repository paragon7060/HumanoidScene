"""Small head-locked wrist-camera panels for Kuavo teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_rgba(image: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 RGBA image accepted by ByteImageProvider."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image, received {image.shape}.")
    if image.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(image)) <= 1.01 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    if image.shape[2] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=2)
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class QuestCameraOverlayCfg:
    """Two small panels leave the center of the native XR view unobstructed."""

    distance_m: float = 0.35
    plane_width_m: float = 0.14
    plane_height_m: float = 0.105
    horizontal_offset_m: float = 0.14
    vertical_offset_m: float = 0.12
    ui_resolution_width: int = 320
    forward_axis: str = "-z"

    def __post_init__(self) -> None:
        if self.distance_m <= 0.08:
            raise ValueError("XR camera overlay distance must be greater than the 0.08 m near plane.")
        if self.plane_width_m <= 0.0 or self.plane_height_m <= 0.0:
            raise ValueError("XR camera overlay plane dimensions must be positive.")
        if self.horizontal_offset_m <= self.plane_width_m / 2:
            raise ValueError("Wrist panels must leave a gap at the center of the view.")
        if self.forward_axis not in {"-z", "+z"}:
            raise ValueError("forward_axis must be '-z' or '+z'.")


class QuestCameraOverlay:
    """Show wrist RGB above each side of the view; never cover it with head RGB."""

    def __init__(self, head_resolution, wrist_resolution, cfg=None) -> None:
        self.cfg = cfg or QuestCameraOverlayCfg()
        self._enable_extensions()
        import carb.settings
        import omni.ui as ui
        from omni.kit.xr.core import XRCore, XRUsdLayerManager
        from omni.kit.xr.scene_view.utils import SceneViewAttachMode, UiContainer
        from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import UpdatePolicy, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf

        carb.settings.get_settings().set_bool("/xr/ui/enabled", True)
        layer = XRUsdLayerManager.get_singleton().get_usd_layer("kuavo_camera_overlay")
        if layer is None:
            layer = XRCore.get_singleton().create_xr_usd_layer(
                "/_xr/gui/kuavo_camera_overlay", meters_per_unit=0.01, up_axis="y"
            )
        if layer is None or not layer.is_valid():
            raise RuntimeError("Failed to create the Kuavo XR camera overlay USD layer.")
        self._xr_layer = layer
        layer.show()
        head_path = layer.ensure_device_prim_path("/user/head")
        self._providers = [ui.ByteImageProvider(), ui.ByteImageProvider()]
        self._gpu_frames = [None, None]
        self._containers = []
        self._components = []
        self._wanted_visible = True
        self._has_frames = [False, False]
        width_cm = self.cfg.plane_width_m * 100.0
        height_cm = self.cfg.plane_height_m * 100.0
        canvas_width = self.cfg.ui_resolution_width
        canvas_height = round(canvas_width * height_cm / width_cm)
        widget_type = self._make_widget_type(ui)
        forward = -1.0 if self.cfg.forward_axis == "-z" else 1.0
        wrist_width, wrist_height = wrist_resolution
        for index, (label, side) in enumerate((("LEFT WRIST", -1.0), ("RIGHT WRIST", 1.0))):
            self._set_provider(index, np.zeros((wrist_height, wrist_width, 3), dtype=np.uint8))
            component = WidgetComponent(
                widget_type, width=width_cm, height=height_cm,
                # resolution_scale changes logical UI scale too: using 24
                # there cropped a 480px widget to its top-left ~20px. Keep
                # logical pixels at scale 1 and convert cm with the unit scale.
                resolution_scale=1.0, unit_to_pixel_scale=canvas_width / width_cm,
                update_policy=UpdatePolicy.ALWAYS, color=[1.0, 1.0, 1.0, 1.0],
                widget_kwargs={"provider": self._providers[index], "label": label,
                               "width": canvas_width, "height": canvas_height},
            )
            container = UiContainer(
                component,
                space_stack=[
                    SpatialSource.new_prim_path_source(head_path),
                    SpatialSource.new_translation_source(Gf.Vec3d(
                        side * self.cfg.horizontal_offset_m * 100.0,
                        self.cfg.vertical_offset_m * 100.0,
                        forward * self.cfg.distance_m * 100.0,
                    )),
                ],
                scene_view_args={"custom_base_path": layer.get_top_level_prim_path()},
                attach_mode=SceneViewAttachMode.DO_NOT_ATTACH_TO_MAIN_VIEWPORT,
            )
            container.visible = False
            self._components.append(component)
            self._containers.append(container)

    def describe(self):
        print(f"[VIEW] XR layer visible={self._xr_layer.is_visible()} path={self._xr_layer.get_top_level_prim_path()}", flush=True)
        for index, container in enumerate(self._containers):
            print(f"[VIEW] panel {index}: visible={container.visible}, frames={self._has_frames[index]}, "
                  f"system={container.scene_view.system_path}", flush=True)

    @staticmethod
    def _enable_extensions() -> None:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        for name in ("omni.kit.xr.scene_view.core", "omni.kit.xr.scene_view.utils"):
            if not manager.is_extension_enabled(name):
                if not manager.set_extension_enabled_immediate(name, True):
                    raise RuntimeError(f"Required XR overlay extension could not be enabled: {name}")

    @staticmethod
    def _make_widget_type(ui):
        class WristCameraWidget(ui.Frame):
            def __init__(self, *, provider, label, width, height):
                super().__init__(width=width, height=height)
                # A Frame has one child: nest the image layout inside our Frame,
                # rather than making it a sibling of an empty ui.Widget.
                with self, ui.ZStack(width=width, height=height):
                    ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                                         width=width, height=height)
                    with ui.VStack():
                        with ui.ZStack(height=28):
                            ui.Rectangle(style={"background_color": 0xC0000000})
                            ui.Label(label, alignment=ui.Alignment.CENTER,
                                     style={"font_size": 18, "color": 0xFFFFFFFF})
                        ui.Spacer()
                        with ui.ZStack(height=28):
                            ui.Rectangle(style={"background_color": 0xC0000000})
                            self.status_label = ui.Label(
                                "FOLLOW OFF | REC OFF", alignment=ui.Alignment.CENTER,
                                style={"font_size": 16, "color": 0xFFFFFFFF},
                            )
        return WristCameraWidget

    def _set_provider(self, index, image: np.ndarray) -> None:
        import torch
        rgba = as_rgba(image)
        height, width = rgba.shape[:2]
        # The CPU ByteImageProvider overload stalls ~20ms per image on this
        # renderer. Upload through the native GPU-pointer API and retain the
        # tensor until the next rendered frame consumes it.
        frame = torch.as_tensor(rgba, device="cuda:0").contiguous()
        self._providers[index].set_bytes_data_from_gpu(frame.data_ptr(), [width, height])
        self._gpu_frames[index] = frame

    def update(self, head_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
        for index, rgb in enumerate((left_rgb, right_rgb)):
            self._set_provider(index, rgb)
            if not self._has_frames[index] and np.any(rgb):
                self._has_frames[index] = True
                print(f"[CAMERA] {'Left' if index == 0 else 'Right'} wrist RGB: "
                      f"min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.1f}", flush=True)
            self._containers[index].visible = self._wanted_visible and self._has_frames[index]

    def toggle_visible(self) -> bool:
        self._wanted_visible = not self._wanted_visible
        for index, container in enumerate(self._containers):
            container.visible = self._wanted_visible and self._has_frames[index]
        return self._wanted_visible

    def set_status(self, *, following: bool, recording: bool, hands_valid: bool, waiting: bool = False,
                   input_mode: str = "hands") -> None:
        recording_status = "ON" if recording else "WAIT" if waiting else "OFF"
        text = f"FOLLOW {'ON' if following else 'OFF'} | REC {recording_status}"
        if (following or waiting) and not hands_valid:
            text += " | CHECK CTRL" if input_mode == "controllers" else " | CHECK HANDS"
        for component in self._components:
            widget = component.widget
            if widget is not None:
                widget.status_label.text = text

    def close(self) -> None:
        for container in self._containers:
            container.hide()
            container.root.clear()
        self._containers.clear()
        self._components.clear()

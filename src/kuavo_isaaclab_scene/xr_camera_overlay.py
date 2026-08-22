"""Head-locked XR camera compositor for Kuavo teleoperation.

The compositor uses Isaac Sim 5.1 / Kit 107.3 XRSceneView and
ByteImageProvider APIs.  A
single opaque UI plane follows ``/user/head``: the Kuavo head camera fills the
plane and the two wrist cameras are overlaid at its left and right edges.
"""

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
    """Visual layout for the Quest head-locked camera screen."""

    distance_m: float = 0.65
    plane_width_m: float = 1.75
    plane_height_m: float = 1.05
    ui_resolution_width: int = 1280
    wrist_panel_width: int = 300
    wrist_panel_height: int = 225
    wrist_margin: int = 28
    forward_axis: str = "-z"

    def __post_init__(self) -> None:
        if self.distance_m <= 0.08:
            raise ValueError("XR camera overlay distance must be greater than the 0.08 m near plane.")
        if self.plane_width_m <= 0.0 or self.plane_height_m <= 0.0:
            raise ValueError("XR camera overlay plane dimensions must be positive.")
        if self.forward_axis not in {"-z", "+z"}:
            raise ValueError("forward_axis must be '-z' or '+z'.")


class QuestCameraOverlay:
    """Present head and wrist camera frames as an opaque head-locked XR UI."""

    def __init__(
        self,
        head_resolution: tuple[int, int],
        wrist_resolution: tuple[int, int],
        cfg: QuestCameraOverlayCfg | None = None,
    ) -> None:
        self.cfg = cfg or QuestCameraOverlayCfg()
        self._enable_extensions()

        import carb.settings
        import omni.ui as ui
        from omni.kit.xr.core import XRCore, XRUsdLayerManager
        from omni.kit.xr.scene_view.utils import SceneViewAttachMode, UiContainer
        from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import (
            UpdatePolicy,
            WidgetComponent,
        )
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf

        settings = carb.settings.get_settings()
        settings.set_bool("/xr/ui/enabled", True)

        head_width, head_height = head_resolution
        wrist_width, wrist_height = wrist_resolution
        self._head_provider = ui.ByteImageProvider()
        self._left_provider = ui.ByteImageProvider()
        self._right_provider = ui.ByteImageProvider()
        self._set_provider(self._head_provider, np.zeros((head_height, head_width, 4), dtype=np.uint8))
        self._set_provider(self._left_provider, np.zeros((wrist_height, wrist_width, 4), dtype=np.uint8))
        self._set_provider(self._right_provider, np.zeros((wrist_height, wrist_width, 4), dtype=np.uint8))

        # Obtain a managed XR-only stage layer so /user/head is updated during
        # late XR pose processing rather than following a one-frame-old Python
        # copy of the HMD pose.
        layer = XRUsdLayerManager.get_singleton().get_usd_layer("kuavo_camera_overlay")
        if layer is None:
            layer = XRCore.get_singleton().create_xr_usd_layer(
                "/_xr/gui/kuavo_camera_overlay",
                meters_per_unit=0.01,
                up_axis="y",
            )
        if layer is None or not layer.is_valid():
            raise RuntimeError("Failed to create the Kuavo XR camera overlay USD layer.")
        self._xr_layer = layer
        head_prim_path = layer.ensure_device_prim_path("/user/head")

        widget_type = self._make_widget_type(ui)
        plane_width_cm = self.cfg.plane_width_m * 100.0
        plane_height_cm = self.cfg.plane_height_m * 100.0
        resolution_scale = self.cfg.ui_resolution_width / plane_width_cm
        component = WidgetComponent(
            widget_type,
            width=plane_width_cm,
            height=plane_height_cm,
            resolution_scale=resolution_scale,
            unit_to_pixel_scale=1.0,
            update_policy=UpdatePolicy.ALWAYS,
            widget_kwargs={
                "head_provider": self._head_provider,
                "left_provider": self._left_provider,
                "right_provider": self._right_provider,
                "canvas_width": self.cfg.ui_resolution_width,
                "canvas_height": int(self.cfg.ui_resolution_width * plane_height_cm / plane_width_cm),
                "wrist_width": self.cfg.wrist_panel_width,
                "wrist_height": self.cfg.wrist_panel_height,
                "margin": self.cfg.wrist_margin,
            },
        )
        sign = -1.0 if self.cfg.forward_axis == "-z" else 1.0
        forward_offset_cm = sign * self.cfg.distance_m * 100.0
        self._container = UiContainer(
            component,
            space_stack=[
                SpatialSource.new_prim_path_source(head_prim_path),
                SpatialSource.new_translation_source(Gf.Vec3d(0.0, 0.0, forward_offset_cm)),
            ],
            scene_view_args={"custom_base_path": layer.get_top_level_prim_path()},
            attach_mode=SceneViewAttachMode.DO_NOT_ATTACH_TO_MAIN_VIEWPORT,
        )
        self._container.visible = True
        self._component = component

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
        class CameraCompositeWidget(ui.Widget):
            def __init__(
                self,
                *,
                head_provider,
                left_provider,
                right_provider,
                canvas_width: int,
                canvas_height: int,
                wrist_width: int,
                wrist_height: int,
                margin: int,
            ) -> None:
                super().__init__()
                with ui.ZStack(width=canvas_width, height=canvas_height):
                    ui.Rectangle(style={"background_color": 0xFF000000})
                    ui.ImageWithProvider(
                        head_provider,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                        width=canvas_width,
                        height=canvas_height,
                    )
                    with ui.VStack():
                        ui.Spacer(height=max(0, (canvas_height - wrist_height) // 2))
                        with ui.HStack(height=wrist_height):
                            ui.Spacer(width=margin)
                            CameraCompositeWidget._wrist_panel(
                                ui, left_provider, "LEFT WRIST", wrist_width, wrist_height
                            )
                            ui.Spacer()
                            CameraCompositeWidget._wrist_panel(
                                ui, right_provider, "RIGHT WRIST", wrist_width, wrist_height
                            )
                            ui.Spacer(width=margin)
                        ui.Spacer()

            @staticmethod
            def _wrist_panel(ui_module, provider, label: str, width: int, height: int) -> None:
                with ui.ZStack(width=width, height=height):
                    ui_module.Rectangle(
                        style={
                            "background_color": 0xFF000000,
                            "border_color": 0xFFE8E8E8,
                            "border_width": 3,
                            "border_radius": 6,
                        }
                    )
                    ui_module.ImageWithProvider(
                        provider,
                        fill_policy=ui_module.IwpFillPolicy.IWP_STRETCH,
                        width=width,
                        height=height,
                    )
                    with ui_module.VStack():
                        ui_module.Label(
                            label,
                            height=28,
                            alignment=ui_module.Alignment.CENTER,
                            style={
                                "font_size": 18,
                                "color": 0xFFFFFFFF,
                                "background_color": 0xA0000000,
                            },
                        )
                        ui_module.Spacer()

        return CameraCompositeWidget

    @staticmethod
    def _set_provider(provider, image: np.ndarray) -> None:
        rgba = as_rgba(image)
        height, width = rgba.shape[:2]
        provider.set_bytes_data(rgba.reshape(-1).data, [width, height])

    def update(self, head_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
        """Upload one synchronized set of camera images to the XR UI textures."""
        self._set_provider(self._head_provider, head_rgb)
        self._set_provider(self._left_provider, left_rgb)
        self._set_provider(self._right_provider, right_rgb)

    def close(self) -> None:
        if getattr(self, "_container", None) is not None:
            self._container.hide()
            self._container.root.clear()
            self._container = None
        self._component = None

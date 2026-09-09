"""Small text-only hand-mode status, independent of wrist cameras."""


class QuestControlStatus:
    def __init__(self, *, layer_name="kuavo_control_status", width=440, height=110,
                 font_size=18, position=(23., 18., -55.), pixels_per_cm=20., widget_type=None):
        from .xr_camera_overlay import QuestCameraOverlay
        QuestCameraOverlay._enable_extensions()
        import carb.settings
        import omni.ui as ui
        from omni.kit.xr.core import XRCore, XRUsdLayerManager
        from omni.kit.xr.scene_view.utils import SceneViewAttachMode, UiContainer
        from omni.kit.xr.scene_view.utils.manipulator_components.widget_component import UpdatePolicy, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf

        class StatusWidget(ui.Frame):
            def __init__(self):
                super().__init__(width=width, height=height)
                with self, ui.ZStack(width=width, height=height):
                    self.background = ui.Rectangle(style={"background_color": 0xB0202020})
                    self.label = ui.Label("HAND SWITCH READY", word_wrap=True, alignment=ui.Alignment.CENTER,
                                          style={"font_size": font_size, "color": 0xFFFFFFFF})

        carb.settings.get_settings().set_bool("/xr/ui/enabled", True)
        manager = XRUsdLayerManager.get_singleton()
        layer = manager.get_usd_layer(layer_name)
        if layer is None:
            layer = XRCore.get_singleton().create_xr_usd_layer(
                f"/_xr/gui/{layer_name}", meters_per_unit=.01, up_axis="y")
        if layer is None or not layer.is_valid():
            raise RuntimeError("Cannot create hand-switch status display")
        self.layer = layer
        self.layer_name, self.position = layer_name, position
        layer.show()
        self.font_size = font_size
        self.component = WidgetComponent(widget_type or StatusWidget, width=width / pixels_per_cm, height=height / pixels_per_cm,
                                         resolution_scale=1., unit_to_pixel_scale=pixels_per_cm,
                                         update_policy=UpdatePolicy.ALWAYS, color=[1., 1., 1., 1.])
        self.container = UiContainer(
            self.component,
            space_stack=[SpatialSource.new_prim_path_source(layer.ensure_device_prim_path("/user/head")),
                         SpatialSource.new_translation_source(Gf.Vec3d(*position))],
            scene_view_args={"custom_base_path": layer.get_top_level_prim_path()},
            attach_mode=SceneViewAttachMode.DO_NOT_ATTACH_TO_MAIN_VIEWPORT,
        )
        self.container.visible = True
        self.text = None
        self.alert = None
        self._widget = None

    def set_visible(self, visible):
        # Re-show the USD layer too; changing the scene-widget flag alone cannot
        # reveal a hidden XR layer (e.g. after an XR reconnect).
        if visible:
            self.layer.show()
            self.container.show()
        else:
            self.container.hide()

    def toggle_visible(self):
        visible = not self.container.visible
        self.set_visible(visible)
        return visible

    def describe(self):
        print(f"[XR PANEL] {self.layer_name}: layer={self.layer.is_visible()} "
              f"panel={self.container.visible} widget_ready={self.component.widget is not None} "
              f"position_cm={self.position}", flush=True)

    def update(self, text, *, alert=False):
        widget = self.component.widget
        if widget is None:
            return False
        if widget is not self._widget:
            self._widget = widget
            self.text = self.alert = None
        if self.text != text:
            widget.label.text = text
            self.text = text
        if self.alert != alert:
            widget.label.style = {
                "font_size": 30 if alert else self.font_size,
                "color": 0xFFFFFFFF,
            }
            widget.background.style = {
                "background_color": 0xD0202020 if alert else 0xB0202020,
            }
            self.alert = alert
        return True

    def close(self):
        self.container.hide()
        self.container.root.clear()

"""Small text-only hand-mode status, independent of wrist cameras."""


class QuestControlStatus:
    def __init__(self):
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
                super().__init__(width=440, height=110)
                with self, ui.ZStack(width=440, height=110):
                    ui.Rectangle(style={"background_color": 0xB0202020})
                    self.label = ui.Label("HAND SWITCH READY", word_wrap=True, alignment=ui.Alignment.CENTER,
                                          style={"font_size": 18, "color": 0xFFFFFFFF})

        carb.settings.get_settings().set_bool("/xr/ui/enabled", True)
        manager = XRUsdLayerManager.get_singleton()
        layer = manager.get_usd_layer("kuavo_control_status")
        if layer is None:
            layer = XRCore.get_singleton().create_xr_usd_layer(
                "/_xr/gui/kuavo_control_status", meters_per_unit=.01, up_axis="y")
        if layer is None or not layer.is_valid():
            raise RuntimeError("Cannot create hand-switch status display")
        self.layer = layer
        layer.show()
        self.component = WidgetComponent(StatusWidget, width=22., height=5.5,
                                         resolution_scale=1., unit_to_pixel_scale=20.,
                                         update_policy=UpdatePolicy.ALWAYS, color=[1., 1., 1., 1.])
        self.container = UiContainer(
            self.component,
            space_stack=[SpatialSource.new_prim_path_source(layer.ensure_device_prim_path("/user/head")),
                         SpatialSource.new_translation_source(Gf.Vec3d(23., 18., -55.))],
            scene_view_args={"custom_base_path": layer.get_top_level_prim_path()},
            attach_mode=SceneViewAttachMode.DO_NOT_ATTACH_TO_MAIN_VIEWPORT,
        )
        self.container.visible = True
        self.text = None

    def update(self, text):
        widget = self.component.widget
        if widget is not None and self.text != text:
            widget.label.text = text
            self.text = text

    def close(self):
        self.container.hide()
        self.container.root.clear()

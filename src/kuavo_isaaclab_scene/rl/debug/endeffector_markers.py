"""VR-visible calibrated tips, moving midpoint and nominal closed TCP (no physics)."""


class EndEffectorMarkers:
    def __init__(self, env):
        import isaaclab.sim as sim
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from ...robots.end_effector import get_end_effector_frames
        self.frames = get_end_effector_frames(env.scene["robot"])
        self.visible = True
        self.legend = "TIPS L: orange/pink R: cyan/blue; CENTER: red cube; LIVE midpoint: white sphere"
        prototypes = {}
        for i, color in enumerate(((1., .7, 0.), (1., .2, .6), (0., 1., 1.), (.15, .3, 1.),
                                    (1., 1., 1.), (1., .05, .05))):
            material = sim.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color)
            prototypes[f"point_{i}"] = (sim.CuboidCfg(size=(.008,)*3, visual_material=material) if i == 5
                                         else sim.SphereCfg(radius=.003, visual_material=material))
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/EndEffectorCenters", markers=prototypes))
        self.update()

    def update(self):
        import torch
        if not self.visible:
            return
        center = self.frames.center_pose_w[0, :, :3]
        if self.frames.definition:
            tips = self.frames.tips_w[0].reshape(4, 3)
            midpoint = self.frames.midpoint_w[0]
            positions = torch.cat((tips, midpoint, center))
            self.markers.visualize(translations=positions, marker_indices=[0, 1, 2, 3, 4, 4, 5, 5])
            delta = (midpoint - center).norm(dim=-1).mul(1000).tolist()
            gaps = (tips.reshape(2, 2, 3)[:, 0] - tips.reshape(2, 2, 3)[:, 1]).norm(dim=-1).mul(1000).tolist()
            self.info = (f"Live-center L/R: {delta[0]:.2f}/{delta[1]:.2f} mm\n"
                         f"Tip separation L/R: {gaps[0]:.2f}/{gaps[1]:.2f} mm; red = IK/reward TCP")
        else:
            self.markers.visualize(translations=center, marker_indices=[5, 5])
            self.info = "No finger calibration: original EEF shown"

    def toggle(self):
        self.visible = not self.visible
        self.markers.set_visibility(self.visible)

    def close(self):
        self.markers.set_visibility(False)

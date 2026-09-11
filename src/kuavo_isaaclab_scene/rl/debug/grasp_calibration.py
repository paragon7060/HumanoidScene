"""Editable, non-physical finger reference points for single-environment inspection."""

import json
import math
from pathlib import Path


class GraspCalibration:
    """Track finger frames without overwriting the editable child transforms."""

    def __init__(self, env, path, model_name, radius=.004):
        from pxr import Gf, UsdGeom, UsdShade
        import omni.usd
        if env.num_envs != 1:
            raise ValueError("Grasp calibration requires one environment.")
        if not math.isfinite(radius) or not .001 <= radius <= .03:
            raise ValueError("Marker radius must be between 0.001 and 0.03 m.")
        self.env = env
        self.model_name = model_name
        self.path = Path(path).expanduser().resolve()
        self.task = env.command_manager.get_term("workcell")
        self.names = tuple(self.task.spec.finger_bodies)
        self.ids = self.task.flap_grasp.finger_ids
        self.stage = omni.usd.get_context().get_stage()
        self.root = UsdGeom.Xform.Define(self.stage, "/Visuals/GraspCalibration")
        self.visible = True
        self.legend = "CALIBRATION: move each tip Xform; K saves all 4 points"
        self.frames, self.tips = [], []
        offsets = {name: [0., 0., 0.] for name in self.names}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if (saved.get("version") != 1 or saved.get("units") != "m"
                    or saved.get("frame") != "finger_link_local"
                    or saved.get("robot_model") != self.model_name):
                raise ValueError(f"Incompatible calibration file: {self.path}")
            offsets = saved["offsets"]
        for name in self.names:
            values = offsets[name]
            if len(values) != 3 or not all(math.isfinite(v) and abs(v) <= .2 for v in values):
                raise ValueError(f"Invalid local offset for {name}: {values}")
        colors = ((1., .7, 0.), (1., .2, .6), (0., 1., 1.), (.15, .3, 1.))
        for name, color in zip(self.names, colors):
            base = f"/Visuals/GraspCalibration/{name}"
            frame = UsdGeom.Xform.Define(self.stage, base)
            self.frames.append((frame.AddTranslateOp(), frame.AddOrientOp()))
            tip = UsdGeom.Xform.Define(self.stage, base + "/tip")
            tip.AddTranslateOp().Set(Gf.Vec3d(*offsets[name]))
            self.tips.append(tip)
            sphere = UsdGeom.Sphere.Define(self.stage, base + "/tip/marker")
            sphere.CreateRadiusAttr(radius)
            material = UsdShade.Material.Define(self.stage, base + "/material")
            shader = UsdShade.Shader.Define(self.stage, base + "/material/shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            from pxr import Sdf
            for field in ("diffuseColor", "emissiveColor"):
                shader.CreateInput(field, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        self.update()
        print(f"[GRASP CALIBRATION] K saves to {self.path}. Edit /Visuals/GraspCalibration/<finger>/tip. "
              "No saved offsets means LINK ORIGINS, not calibrated fingertips.", flush=True)

    def offsets(self):
        # Read the composed local transform, including edits made with the gizmo.
        return {name: list(tip.GetLocalTransformation().ExtractTranslation())
                for name, tip in zip(self.names, self.tips)}

    def update(self):
        from pxr import Gf
        data = self.task.robot.data
        positions = data.body_link_pos_w[0, self.ids].detach().cpu().tolist()
        quaternions = data.body_link_quat_w[0, self.ids].detach().cpu().tolist()
        for (translate, orient), p, q in zip(self.frames, positions, quaternions):
            translate.Set(Gf.Vec3d(*p))
            orient.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        self.info = "EDIT PREVIEW: save and restart consumers to apply\n" + "\n".join(
            f"{name} [m]: " + ", ".join(f"{v:.4f}" for v in xyz)
            for name, xyz in self.offsets().items())

    def save(self):
        offsets = self.offsets()
        if not all(math.isfinite(v) and abs(v) <= .2 for xyz in offsets.values() for v in xyz):
            print("[GRASP CALIBRATION] Not saved: offsets must be finite and within +/-0.2 m.", flush=True)
            return
        payload = dict(version=1, units="m", frame="finger_link_local",
                       robot_model=self.model_name, offsets=offsets)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2) + "\n")
        except OSError as exc:
            print(f"[GRASP CALIBRATION] Save failed: {exc}", flush=True)
            return
        print(f"[GRASP CALIBRATION] Saved {self.path}\n{json.dumps(offsets, indent=2)}", flush=True)

    def toggle(self):
        self.visible = not self.visible
        from pxr import UsdGeom
        self.root.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if self.visible else UsdGeom.Tokens.invisible)

    def close(self):
        from pxr import UsdGeom
        self.root.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

"""RTX world-space collider/contact inspection for one Quest RL environment.

Read the live stage's enabled colliders and PhysX cooking representation. Never
replace missing collision data with visual bounds and call it an actual collider.
"""

from collections import deque
import numpy as np

from .collision_geometry import hull_polygons, inward_face, polygon_edges, force_arrow


def rotate_points(points, pose):
    w, x, y, z = pose[3:]
    matrix = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return np.asarray(points) @ matrix.T + pose[:3]


class CollisionOverlay:
    def __init__(self, env):
        from pxr import UsdGeom, UsdUtils
        import omni.usd
        from omni.physx import get_physx_cooking_interface
        if env.num_envs != 1:
            raise ValueError("Collision overlay is for one inspection environment only.")
        self.env = env
        self.stage = omni.usd.get_context().get_stage()
        self.stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        self.cooking = get_physx_cooking_interface()
        self.root = UsdGeom.Xform.Define(self.stage, "/Visuals/RLCollisionInspection")
        self.visible, self.closed = True, False
        self.queue, self.requests, self.records = deque(), [], []
        self.pending, self.errors, self.ready = 0, 0, 0
        self.curves = {}
        colors = {"fingers": (0., .8, 1.), "flaps": (1., .55, .05),
                  "pad": (.1, 1., .25), "bounds": (.75, .25, 1.), "force": (1., .1, .05)}
        for name, color in colors.items():
            self.curves[name] = self._curve(name, color, .0012 if name != "pad" else .002)
        # Contact centers and candidate pad centroids use real non-physical geometry,
        # not desktop-only debug draw that may be omitted from XR composition.
        import isaaclab.sim as sim
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        self.points = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/RLCollisionContactPoints", markers={
                "contact": sim.SphereCfg(radius=.003, visual_material=sim.PreviewSurfaceCfg(
                    diffuse_color=(1., .1, .05), emissive_color=(1., .1, .05))),
                "pad": sim.SphereCfg(radius=.003, visual_material=sim.PreviewSurfaceCfg(
                    diffuse_color=(.1, 1., .25), emissive_color=(.1, 1., .25)))}))
        self.points.set_visibility(False)
        t = env.command_manager.get_term("workcell")
        for hand_id, body_id in enumerate(t.flap_grasp.finger_ids[2:4]):
            self._body(t.robot, body_id, "fingers", hand_id)
        for box_id, box in enumerate(t.boxes):
            for body_id in t.flap_grasp.body_ids[box_id]:
                self._body(box, body_id, "flaps", None)
        print("[COLLISION VIEW] cyan=enabled finger colliders; orange=flap colliders; "
              "green=inward face CANDIDATE; violet=RL acceptance bounds; red=mean contact/normal pair force (no friction). "
              "PC J toggles. Force arrow 1 cm/N, capped at 12 cm.", flush=True)
        self.info = "COLLIDERS: loading PhysX cooking data"

    def _curve(self, name, color, width):
        from pxr import UsdGeom, UsdShade, Sdf, Gf
        path = str(self.root.GetPath()) + "/" + name
        curve = UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation("constant")
        material = UsdShade.Material.Define(self.stage, path + "Material")
        shader = UsdShade.Shader.Define(self.stage, path + "Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        for channel in ("diffuseColor", "emissiveColor"):
            shader.CreateInput(channel, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
        return curve

    def _body(self, asset, body_id, kind, hand_id):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysicsSchemaTools
        root_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise ValueError(f"Missing inspection asset {root_path}")
        bodies = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.GetName() == asset.body_names[body_id] and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError(f"Expected one body {asset.body_names[body_id]} under {root_path}")
        body = bodies[0]
        if UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is False:
            return
        cache = UsdGeom.XformCache()
        body_world = cache.GetLocalToWorldTransform(body)
        rigid = Gf.Matrix4d().SetRotate(Gf.Transform(body_world).GetRotation())
        rigid.SetTranslateOnly(body_world.ExtractTranslation())
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            # Only shapes owned by this rigid body; nested rigid bodies are independent.
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if parent != body:
                continue
            transform = cache.GetLocalToWorldTransform(prim) * rigid.GetInverse()
            record = dict(asset=asset, body_id=body_id, kind=kind, hand_id=hand_id,
                          path=str(prim.GetPath()), polygons=[])
            self.records.append(record)
            def convert(vertices, faces, record=record, transform=transform):
                local = [tuple(transform.Transform(Gf.Vec3d(*map(float, v)))) for v in vertices]
                return hull_polygons(local, faces)
            if prim.IsA(UsdGeom.Cube):
                half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
                vertices = [(x*half, y*half, z*half) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                faces = [[0,1,3,2], [4,6,7,5], [0,4,5,1], [2,3,7,6], [0,2,6,4], [1,5,7,3]]
                record["polygons"] = convert(vertices, faces)
                self.ready += 1
            elif prim.IsA(UsdGeom.Mesh):
                self.pending += 1
                def result(status, convexes, record=record, convert=convert):
                    if self.closed:
                        return
                    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
                    polygons, error = [], None
                    try:
                        if status != PhysxCollisionRepresentationResult.RESULT_VALID:
                            raise ValueError(str(status))
                        for hull in convexes:
                            vertices = [(v.x, v.y, v.z) for v in hull.vertices]
                            faces = [list(hull.indices[p.index_base:p.index_base+p.num_vertices]) for p in hull.polygons]
                            polygons.extend(convert(vertices, faces))
                        if not polygons:
                            raise ValueError("empty cooked representation")
                    except Exception as exc:
                        error = str(exc)
                    self.queue.append((record, polygons, error))
                try:
                    handle = self.cooking.request_convex_collision_representation(
                        stage_id=self.stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                        run_asynchronously=True, on_result=result)
                    self.requests.append(handle)
                except Exception as exc:
                    self.queue.append((record, [], str(exc)))
            else:
                self.errors += 1
                print(f"[COLLISION VIEW] Unsupported collider {prim.GetTypeName()}: {prim.GetPath()}; NOT shown", flush=True)

    def toggle(self):
        self.visible = not self.visible
        from pxr import UsdGeom
        UsdGeom.Imageable(self.root.GetPrim()).CreateVisibilityAttr().Set("inherited" if self.visible else "invisible")
        if not self.visible:
            self.points.set_visibility(False)
        print(f"[COLLISION VIEW] {'ON' if self.visible else 'OFF'}", flush=True)

    def update(self, contact_sample_valid=True):
        while self.queue:
            record, polygons, error = self.queue.popleft()
            self.pending -= 1
            record["polygons"] = polygons
            self.errors += bool(error)
            self.ready += not bool(error)
            print(f"[COLLISION VIEW] {record['path']}: {error or 'PhysX cooked collider ready'}", flush=True)
        self.info = f"COLLIDERS ready={self.ready} pending={self.pending} unavailable={self.errors}; green=pad candidate"
        if not self.visible:
            return
        from pxr import Gf, UsdPhysics
        t = self.env.command_manager.get_term("workcell")
        pose_cache = {}
        def poses(asset):
            key = id(asset)
            if key not in pose_cache:
                pose_cache[key] = np.concatenate((asset.data.body_link_pos_w[0].detach().cpu().numpy(),
                    asset.data.body_link_quat_w[0].detach().cpu().numpy()), -1)
            return pose_cache[key]
        lines = {name: [] for name in self.curves}
        pad_polygons = [[], []]
        for record in self.records:
            if not record["polygons"]:
                continue
            prim = self.stage.GetPrimAtPath(record["path"])
            if not prim.IsValid() or not prim.IsActive() or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            pose = poses(record["asset"])[record["body_id"]]
            world = [(rotate_points(p, pose), rotate_points(normal, np.r_[np.zeros(3), pose[3:]]))
                     for p, normal in record["polygons"]]
            lines[record["kind"]].extend(polygon_edges(world))
            if record["hand_id"] is not None:
                pad_polygons[record["hand_id"]].extend(world)
        points, kinds = [], []
        if all(pad_polygons):
            centers = [np.concatenate([p for p, _ in polys]).mean(0) for polys in pad_polygons]
            for hand in (0, 1):
                face = inward_face(pad_polygons[hand], centers[1-hand]-centers[hand])
                if face is not None:
                    polygon, center, normal = face
                    # Half-mm display bias avoids z-fighting; center stays on the real face.
                    lines["pad"].extend(polygon_edges([(polygon + .0005*normal, normal)]))
                    points.append(center); kinds.append(1)
        # Show the CURRENT reward acceptance volume separately from cooked colliders.
        box_id = int(t.active_box[0].item())
        g = t.flap_grasp
        for candidate, body_id in enumerate(g.body_ids[box_id]):
            c = g.centers[box_id, candidate].detach().cpu().numpy()
            h = g.halves[box_id, candidate].detach().cpu().numpy()
            low, high = c-h-t.spec.flap_contact_margin, c+h+t.spec.flap_contact_margin
            if t.spec.flap_contact_region == "top_band":
                low[2] = max(low[2], c[2]+h[2]-t.spec.flap_top_band)
            vertices = np.array([(x,y,z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            pose = poses(t.boxes[box_id])[body_id]
            world = rotate_points(vertices, pose)
            lines["bounds"].extend([[world[i],world[j]] for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)])
        descriptions = []
        if contact_sample_valid:
            for finger in (2, 3):
                data = self.env.scene[f"grasp_contact_{finger}"].data
                if data.contact_pos_w is None or data.force_matrix_w is None:
                    continue
                for candidate, name in enumerate(t.spec.grasp_flaps):
                    index = box_id * 2 + candidate
                    point = data.contact_pos_w[0, 0, index].detach().cpu().numpy()
                    force = data.force_matrix_w[0, 0, index].detach().cpu().numpy()
                    arrow = force_arrow(point, force)
                    if not len(arrow):
                        continue
                    points.append(point); kinds.append(0)
                    lines["force"].extend(arrow)
                    descriptions.append(f"{'F' if finger == 2 else 'B'}/{name}={np.linalg.norm(force):.2f}N")
        self.info += "\n" + ("NORMAL FORCE: " + " ".join(descriptions) if descriptions else "No current mean contact/force")
        for name, segments in lines.items():
            curve = self.curves[name]
            curve.CreateCurveVertexCountsAttr([2]*len(segments))
            curve.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for segment in segments for p in segment])
        self.points.set_visibility(bool(points))
        if points:
            self.points.visualize(translations=np.asarray(points), marker_indices=kinds)

    def close(self):
        self.closed = True
        for handle in self.requests:
            if handle is not None:
                try:
                    self.cooking.cancel_collision_representation_task(handle)
                except Exception as exc:
                    print(f"[COLLISION VIEW] Cooking task cleanup: {exc}", flush=True)
        self.requests.clear()
        if self.visible:
            self.toggle()

"""CPU mesh preview of recorded PhysX poses; no graphics GPU/RTX dependency."""

import numpy as np


class SceneVideo:
    def __init__(self, env, caption="SAC policy | actual PhysX rollout | CPU mesh visualization"):
        self.caption = caption
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
        from kuavo_isaaclab_scene.teleop.urdf_arm_ik import quat_matrix
        self.quat_matrix = quat_matrix
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        owners = {}
        for asset in env.scene.articulations.values():
            for i, path in enumerate(asset.root_physx_view.link_paths[0]):
                owners[path] = (asset, i)
        for asset in env.scene.rigid_objects.values():
            owners[asset.root_physx_view.prim_paths[0]] = (asset, None)
        self.meshes = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            if mesh.ComputeVisibility() == 'invisible':
                continue
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            counts = mesh.GetFaceVertexCountsAttr().Get()
            if len(points) == 0 or counts is None:
                continue
            faces, start = [], 0
            for count in counts:
                for j in range(1,count-1):
                    faces.append((indices[start],indices[start+j],indices[start+j+1]))
                start += count
            if not faces:
                continue
            path = prim.GetPath().pathString
            body = prim
            while body and body.GetPath().pathString not in owners:
                body = body.GetParent()
            owner = owners.get(body.GetPath().pathString) if body else None
            transform = cache.GetLocalToWorldTransform(prim)
            if owner:
                body_world = cache.GetLocalToWorldTransform(body)
                rigid = Gf.Matrix4d().SetRotate(Gf.Transform(body_world).GetRotation())
                rigid.SetTranslateOnly(body_world.ExtractTranslation())
                transform = transform * rigid.GetInverse()
            matrix = np.asarray(transform)
            points = np.c_[points,np.ones(len(points))] @ matrix
            color = np.array((175,188,201))
            if '/Kuavo/' in path:
                color = np.array((211,219,227))
                if 'finger' in path or 'wheel' in path:
                    color = np.array((72,85,98))
            elif 'box' in path.lower():
                color = np.array((194,143,79))
            elif 'rack' in path.lower():
                color = np.array((99,135,166))
            points, faces = points[:,:3], np.asarray(faces)
            # Vertex clustering keeps connected surfaces while reducing CAD
            # tessellation. This is a mesh preview, not an RTX screenshot.
            if len(faces) > 2000:
                grid = .002 if 'finger' in path else .006
                _, inverse = np.unique(np.round(points/grid).astype(np.int64),axis=0,return_inverse=True)
                count = np.bincount(inverse)
                reduced = np.stack([np.bincount(inverse,weights=points[:,i])/count for i in range(3)],axis=1)
                faces = inverse[faces]
                faces = faces[(faces[:,0]!=faces[:,1]) & (faces[:,1]!=faces[:,2]) & (faces[:,2]!=faces[:,0])]
                _, unique = np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
                points,faces = reduced,faces[unique]
            self.meshes.append((points,faces,owner,color))
        print('[CPU VIDEO] meshes:',len(self.meshes),'triangles:',sum(len(x[1]) for x in self.meshes),flush=True)
        eye = np.array((-1.8,-2.2,1.7)); target = np.array((.12,.1,.82))
        forward = target-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward,(0,0,1)); right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        self.eye,self.basis = eye,np.array((right,up,forward)).T

    def frame(self, env, step, distance, grasp):
        import cv2
        all_triangles, all_colors = [], []
        for points,faces,owner,color in self.meshes:
            world = points
            if owner:
                asset,i = owner
                if i is None:
                    pos = asset.data.root_pos_w[0].cpu().numpy()
                    q = asset.data.root_quat_w[0].cpu().numpy()
                else:
                    pos = asset.data.body_link_pos_w[0,i].cpu().numpy()
                    q = asset.data.body_link_quat_w[0,i].cpu().numpy()
                world = points @ self.quat_matrix(q).T + pos
            triangles = world[faces]
            normal = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            normal /= np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-10)
            light = .45+.55*np.abs(normal @ np.array((.3,-.4,.866)))
            camera = (triangles-self.eye) @ self.basis
            valid = (camera[:,:,2]>.05).all(1)
            all_triangles.append(camera[valid]);all_colors.append((color[None]*light[:,None])[valid])
        triangles = np.concatenate(all_triangles); colors = np.concatenate(all_colors).clip(0,255).astype(np.uint8)
        depth = triangles[:,:,2].mean(1)
        projected = triangles[:,:,:2]/triangles[:,:,2,None]*720
        projected[:,:,0] += 480; projected[:,:,1] = 395-projected[:,:,1]
        frame = np.full((720,960,3),(235,238,241),dtype=np.uint8)
        inside = (projected[:,:,0].max(1)>=0) & (projected[:,:,0].min(1)<=960)
        inside &= (projected[:,:,1].max(1)>=0) & (projected[:,:,1].min(1)<=720)
        indices = np.flatnonzero(inside)
        for index in indices[np.argsort(-depth[indices])]:
            poly = projected[index]
            cv2.fillConvexPoly(frame,np.round(poly).astype(np.int32),tuple(map(int,colors[index])))
        cv2.rectangle(frame,(0,0),(960,66),(23,28,36),-1)
        cv2.putText(frame,self.caption,(15,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(frame,f't={step/30:.2f}s  |  right target: {distance*100:.1f} cm  |  grasp: {int(grasp)}',(15,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
        return frame

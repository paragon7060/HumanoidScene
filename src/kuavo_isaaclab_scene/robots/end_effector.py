"""Shared calibrated gripper frames. Positions are metres; quaternions are wxyz.

``center_pose_w`` is a rigid TCP at the nominal CLOSED jaw midpoint, retaining
the original EEF orientation. ``midpoint_w`` follows the actual two moving tips.
Neither quantity is a contact measurement or a physical rigid body.
"""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

FINGERS = ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")


def calibration_definition(model=None):
    from .robot_model import resolve_robot_model
    from .gripper_config import resolve_gripper_settings
    from ..core.paths import CONFIG_DIR, PACKAGE_CONFIG_DIR
    model = model or resolve_robot_model()
    # These measurements belong only to this model/hand, never silently reuse on S63/Allegro.
    if model.name != "s200062" or resolve_gripper_settings().name != "s200062_integrated":
        return None
    override = os.environ.get("KUAVO_GRASP_REFERENCE_POINTS")
    path = Path(override).expanduser() if override else CONFIG_DIR / "grasp_reference_points.json"
    if not path.exists() and not override:
        path = PACKAGE_CONFIG_DIR / "grasp_reference_points.json"
    data = json.loads(path.read_text())
    if (data.get("version") != 1 or data.get("units") != "m"
            or data.get("frame") != "finger_link_local" or data.get("robot_model") != model.name):
        raise ValueError(f"Incompatible gripper reference file: {path}")
    values = np.asarray([data["offsets"][n] for n in FINGERS], dtype=float)
    if values.shape != (4, 3) or not np.isfinite(values).all() or (np.abs(values) > .2).any():
        raise ValueError(f"Invalid gripper reference offsets: {path}")
    # Compute from the actual configured closed commands, not from a captured open pose.
    closed = closed_offsets(model.urdf_path, data["offsets"], resolve_gripper_settings())
    return dict(revision=1, robot_model=model.name, offsets=data["offsets"],
                center_frame="nominal_closed_midpoint_original_eef_orientation",
                closed_offsets=closed)


def closed_offsets(urdf_path, offsets, settings):
    """URDF FK with the existing closed four-bar solution (no simulator required)."""
    from .twofinger_linkage import initial_passive_positions
    from ..teleop.urdf_arm_ik import axis_rotation
    joints = {j.find("child").get("link"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    result = {}
    for side, letter in (("left", "l"), ("right", "r")):
        q = settings.command_for(side, settings.close_command)
        q.update(initial_passive_positions(q))
        root = f"zarm_{letter}7_link"

        def transform(link):
            chain = []
            while link != root:
                j = joints[link]
                chain.append(j)
                link = j.find("parent").get("link")
            p, r = np.zeros(3), np.eye(3)
            for j in reversed(chain):
                origin = j.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                p = p + r @ xyz
                r = r @ axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0])
                if j.get("type") != "fixed":
                    angle = q[j.get("name")]  # fail rather than assume a passive joint angle
                    r = r @ axis_rotation(np.fromstring(j.find("axis").get("xyz"), sep=" "), angle)
            return p, r

        tips = []
        for jaw in "fb":
            name = f"{letter}_{jaw}_finger"
            p, r = transform(name)
            tips.append(p + r @ np.asarray(offsets[name]))
        eef_p, eef_r = transform(f"zarm_{letter}7_end_effector")
        result[side] = (eef_r.T @ ((tips[0] + tips[1]) * .5 - eef_p)).tolist()
    return result


def center_offset(side):
    definition = calibration_definition()
    return tuple(definition["closed_offsets"][side]) if definition else (0., 0., 0.)


class EndEffectorFrames:
    def __init__(self, robot):
        import torch
        self.robot = robot
        self.definition = calibration_definition()
        self.tool_ids, _ = robot.find_bodies(["zarm_l7_end_effector", "zarm_r7_end_effector"], preserve_order=True)
        if len(self.tool_ids) != 2:
            raise ValueError("Expected two original EEF frames")
        self.closed = torch.tensor([self.definition["closed_offsets"][s] if self.definition else (0., 0., 0.)
                                    for s in ("left", "right")], device=robot.device)
        if self.definition:
            self.finger_ids, _ = robot.find_bodies(list(FINGERS), preserve_order=True)
            if len(self.finger_ids) != 4:
                raise ValueError("Calibrated frame requires four finger links")
            self.offsets = torch.tensor([self.definition["offsets"][n] for n in FINGERS], device=robot.device)

    @property
    def center_pose_w(self):
        import torch
        from isaaclab.utils.math import quat_apply
        d = self.robot.data
        q = d.body_link_quat_w[:, self.tool_ids]
        offset = self.closed[None].expand(q.shape[0], -1, -1)
        p = d.body_link_pos_w[:, self.tool_ids] + quat_apply(q.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
        return torch.cat((p, q), -1)

    @property
    def tips_w(self):
        from isaaclab.utils.math import quat_apply
        if not self.definition:
            raise ValueError("No calibrated finger points for this robot/gripper")
        d = self.robot.data
        q = d.body_link_quat_w[:, self.finger_ids]
        offset = self.offsets[None].expand(q.shape[0], -1, -1)
        p = d.body_link_pos_w[:, self.finger_ids] + quat_apply(q.reshape(-1, 4), offset.reshape(-1, 3)).reshape_as(offset)
        return p.reshape(-1, 2, 2, 3)

    @property
    def midpoint_w(self):
        return self.tips_w.mean(-2)

    @property
    def center_pose_b(self):
        """The same closed TCPs expressed in the robot root frame."""
        import torch
        from isaaclab.utils.math import subtract_frame_transforms
        pose = self.center_pose_w
        root = self.robot.data.root_pose_w
        p, q = subtract_frame_transforms(root[:, None, :3].expand(-1, 2, -1).reshape(-1, 3),
                                         root[:, None, 3:].expand(-1, 2, -1).reshape(-1, 4),
                                         pose[..., :3].reshape(-1, 3), pose[..., 3:].reshape(-1, 4))
        return torch.cat((p, q), -1).reshape(-1, 2, 7)


def get_end_effector_frames(robot):
    """Public API for any scene/eval/teleop: batched [env, left/right, ...] tensors."""
    if not hasattr(robot, "endeffector_center"):
        robot.endeffector_center = EndEffectorFrames(robot)
    return robot.endeffector_center


def spawn_center_prims(root):
    """Nonphysical child frames, included in normal scene spawning/cloning."""
    from pxr import Gf, Usd, UsdGeom
    definition = calibration_definition()
    if not definition:
        return
    for side, letter in (("left", "l"), ("right", "r")):
        bodies = [p for p in Usd.PrimRange(root) if p.GetName() == f"zarm_{letter}7_end_effector"]
        if len(bodies) != 1:
            raise ValueError(f"Missing unique {side} original EEF")
        xform = UsdGeom.Xform.Define(root.GetStage(), bodies[0].GetPath().AppendChild("endeffector_center"))
        xform.AddTranslateOp().Set(Gf.Vec3d(*definition["closed_offsets"][side]))
        xform.AddOrientOp().Set(Gf.Quatf(1.))
        xform.AddScaleOp().Set(Gf.Vec3f(1.))
        xform.GetPrim().SetCustomDataByKey("kuavo:centerDefinition", definition["center_frame"])

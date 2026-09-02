"""Closed four-bar geometry for the S200062/S56 integrated hand.

The donor URDF lumps bar_1+bar_2 and bar_3+finger into rigid subassemblies.
Keep that explicit approximation, but close each resulting four-bar physically:
base -> (1+2) -> (3+finger) -> 4 -> base. Only bar_1 is driven.
All coordinates are metres in the source CAD link frames, not MJCF helper
sites (which do not coincide with these meshes' pin holes).
"""

from __future__ import annotations

import math

TWO_FINGER_PRESETS = frozenset({"s200062_integrated", "s56_twofinger"})
LINKAGE_VERSION = 1
FINGER_PIN = (-0.0125, 0.0, -0.021)
# Sum of the zero-pose URDF joint origins plus FINGER_PIN minus bar_4 origin.
# The CAD hole is approximately (-0.00062, 0, -0.05); retaining the URDF's
# rounding here gives exactly coincident assembly anchors at reset.
FOLLOWER_PIN = (-0.0006261, 0.0, -0.0499953)


def pin_for(jaw: str, point: tuple[float, float, float]) -> tuple[float, float, float]:
    if jaw not in ("f", "b"):
        raise ValueError(f"Invalid jaw: {jaw}")
    return (point[0] if jaw == "f" else -point[0], point[1], point[2])


def _rotate(q: float, v: tuple[float, float]) -> tuple[float, float]:
    c, s = math.cos(q), math.sin(q)
    return (c * v[0] + s * v[1], -s * v[0] + c * v[1])


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(a[1] * b[0] - a[0] * b[1], a[0] * b[0] + a[1] * b[1])


def passive_joint_angles(driver: float, jaw: str = "f") -> tuple[float, float]:
    """Return passive (bar_3, bar_4) angles on the assembled branch.

    Used only for consistent reset poses and offline geometry tests. Runtime
    motion/contact is solved by PhysX, not by moving visual meshes or forcing
    passive target positions.
    """
    sign = 1.0 if jaw == "f" else -1.0
    if jaw not in ("f", "b") or not math.isfinite(driver):
        raise ValueError("Expected a finite driver angle and jaw f/b")
    q = sign * driver
    if not -0.25 - 1e-9 <= q <= 1e-9:
        raise ValueError("Validated two-finger driver range is f=[-0.25,0], b=[0,0.25]")
    a, b = (0.0125, -0.063137), (0.02, -0.09)
    crank = (0.011329 + 0.016204, 0.0063767 - 0.014934)
    coupler = (-0.0081591 + FINGER_PIN[0], -0.047301 + FINGER_PIN[2])
    follower = (FOLLOWER_PIN[0], FOLLOWER_PIN[2])
    rotated = _rotate(q, crank)
    c = (a[0] + rotated[0], a[1] + rotated[1])
    delta = (b[0] - c[0], b[1] - c[1])
    distance = math.hypot(*delta)
    r, s = math.hypot(*coupler), math.hypot(*follower)
    along = (r*r - s*s + distance*distance) / (2*distance)
    height_sq = r*r - along*along
    if height_sq < -1e-12:
        raise ValueError("Four-bar cannot close at this angle")
    ux, uz = delta[0]/distance, delta[1]/distance
    h = math.sqrt(max(0.0, height_sq))
    # The plus intersection is the source assembly, not the crossed branch.
    d = (c[0] + along*ux - h*uz, c[1] + along*uz + h*ux)
    theta = _angle(coupler, (d[0]-c[0], d[1]-c[1]))
    q4 = _angle(follower, (d[0]-b[0], d[1]-b[1]))
    return sign * (theta-q), sign * q4


def initial_passive_positions(commands: dict[str, float]) -> dict[str, float]:
    result = {}
    for side in "lr":
        for jaw in "fb":
            motor = f"{side}_{jaw}_bar_1_joint"
            if motor not in commands:
                continue
            q3, q4 = passive_joint_angles(commands[motor], jaw)
            result[f"{side}_{jaw}_bar_3_joint"] = q3
            result[f"{side}_{jaw}_bar_4_joint"] = q4
    return result


def validate_motor_commands(settings) -> None:
    expected = {"{side}_f_bar_1_joint", "{side}_b_bar_1_joint"}
    if set(settings.joint_names) != expected:
        raise ValueError("Closed two-finger presets must drive only bar_1; migrate the old bar_3 commands")
    for command in (settings.default_joint_pos, settings.open_command, settings.close_command):
        if set(command) != expected:
            raise ValueError("Use explicit f_bar_1/b_bar_1 commands for closed two-finger presets")
        for jaw in "fb":
            passive_joint_angles(command[f"{{side}}_{jaw}_bar_1_joint"], jaw)


def author_closed_linkages(stage) -> None:
    """Idempotently finalize a generated USD; geometry and source layers stay intact."""
    from pxr import Gf, UsdPhysics

    root = stage.GetDefaultPrim()
    root_path = str(root.GetPath())
    by_name = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    for side in "lr":
        for jaw in "fb":
            prefix = f"{side}_{jaw}"
            for index in (1, 3, 4):
                prim = by_name[f"{prefix}_bar_{index}_joint"]
                joint = UsdPhysics.RevoluteJoint.Define(stage, prim.GetPath())
                joint.CreateAxisAttr("Y")
                joint.CreateLowerLimitAttr(math.degrees(-0.698))
                joint.CreateUpperLimitAttr(math.degrees(0.698))
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                # USD angular gains are per degree; Isaac actuator gains are
                # per radian and overwrite these at runtime.
                drive.CreateStiffnessAttr(100.0 * math.pi/180 if index == 1 else 0.0)
                drive.CreateDampingAttr(10.0 * math.pi/180 if index == 1 else 0.0)
                drive.CreateMaxForceAttr(5.0)
                drive.CreateTargetPositionAttr(0.0)
                if index == 4:
                    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
                    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            closure = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/joints/{prefix}_loop_joint")
            closure.CreateBody0Rel().SetTargets([f"{root_path}/{prefix}_finger"])
            closure.CreateBody1Rel().SetTargets([f"{root_path}/{prefix}_bar_4"])
            closure.CreateAxisAttr("Y")
            closure.CreateLocalPos0Attr(Gf.Vec3f(*pin_for(jaw, FINGER_PIN)))
            closure.CreateLocalPos1Attr(Gf.Vec3f(*pin_for(jaw, FOLLOWER_PIN)))
            closure.CreateLocalRot0Attr(Gf.Quatf(1.0))
            closure.CreateLocalRot1Attr(Gf.Quatf(1.0))
            # A regular maximal-coordinate hinge closes the articulation tree.
            closure.CreateExcludeFromArticulationAttr(True)
            closure.CreateCollisionEnabledAttr(False)
    root.SetCustomDataByKey("kuavo:twofingerLinkageVersion", LINKAGE_VERSION)


def require_closed_linkages(root) -> None:
    from pxr import UsdPhysics

    if root.GetCustomDataByKey("kuavo:twofingerLinkageVersion") != LINKAGE_VERSION:
        raise RuntimeError("Outdated two-finger USD: run scripts/finalize_twofinger_usd.py")
    stage = root.GetStage()
    for side in "lr":
        for jaw in "fb":
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(
                f"{root.GetPath()}/joints/{side}_{jaw}_loop_joint"))
            if (not joint or not joint.GetExcludeFromArticulationAttr().Get()
                    or not joint.GetJointEnabledAttr().Get()
                    or not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets()):
                raise RuntimeError(f"Missing external loop-closing hinge: {side}_{jaw}")
            for attr, expected in ((joint.GetLocalPos0Attr(), pin_for(jaw, FINGER_PIN)),
                                   (joint.GetLocalPos1Attr(), pin_for(jaw, FOLLOWER_PIN))):
                if math.dist(tuple(attr.Get()), expected) > 1e-7:
                    raise RuntimeError(f"Incorrect CAD closure anchor: {side}_{jaw}")
            expected_bodies = (f"{root.GetPath()}/{side}_{jaw}_finger",
                               f"{root.GetPath()}/{side}_{jaw}_bar_4")
            if (str(joint.GetBody0Rel().GetTargets()[0]), str(joint.GetBody1Rel().GetTargets()[0])) != expected_bodies:
                raise RuntimeError(f"Closure attached to the wrong bodies: {side}_{jaw}")
            for index in (3, 4):
                passive = stage.GetPrimAtPath(f"{root.GetPath()}/joints/{side}_{jaw}_bar_{index}_joint")
                if not passive.IsA(UsdPhysics.RevoluteJoint):
                    raise RuntimeError(f"Passive bar_{index} is not revolute")
                drive = UsdPhysics.DriveAPI(passive, "angular")
                if drive and drive.GetStiffnessAttr().Get() != 0.0:
                    raise RuntimeError(f"Passive bar_{index} must not have a position servo")

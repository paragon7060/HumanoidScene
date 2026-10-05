"""Exercise real USD schema getters, not text spelling alone (CPU only)."""
import math
import pytest

Usd=pytest.importorskip('pxr.Usd')
UsdPhysics=pytest.importorskip('pxr.UsdPhysics')

from kuavo_isaaclab_scene.workcell.rack_rollers import (
    generate_roller_deck_usda,resolve_rack_roller_settings,write_passive_bearing_overlay,
)
from kuavo_isaaclab_scene.core.paths import RACK_ROLLER_RUNTIME_ASSET


def test_wrong_drive_namespace_does_not_set_the_schema_value():
    stage=Usd.Stage.CreateInMemory()
    prim=stage.DefinePrim('/Joint','PhysicsRevoluteJoint')
    drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
    from pxr import Sdf
    prim.CreateAttribute('physics:drive:angular:damping',Sdf.ValueTypeNames.Float).Set(.00002)
    assert drive.GetDampingAttr().Get()==0.
    assert math.isinf(drive.GetMaxForceAttr().Get())


def test_generated_roller_drives_are_readable_in_radian_units():
    settings=resolve_rack_roller_settings(enabled=True)
    stage=Usd.Stage.CreateInMemory()
    assert stage.GetRootLayer().ImportFromString(generate_roller_deck_usda(settings,math.radians(5.3)))
    count=0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive=UsdPhysics.DriveAPI(prim,'angular')
            assert drive.GetDampingAttr().Get()*180/math.pi==pytest.approx(settings.angular_damping,rel=1e-6)
            assert drive.GetStiffnessAttr().Get()==0.
            assert drive.GetMaxForceAttr().Get()==pytest.approx(.05)
            count+=1
    assert count==3*settings.rows*settings.columns


def test_overlay_restores_all_drives_without_changing_joint_or_shape_geometry(tmp_path):
    settings=resolve_rack_roller_settings(enabled=True)
    path=write_passive_bearing_overlay(tmp_path/'bearing.usda',settings)
    original=Usd.Stage.Open(str(RACK_ROLLER_RUNTIME_ASSET))
    corrected=Usd.Stage.Open(str(path))
    for tier in (1,2,3):
        for row in range(settings.rows):
            for col in range(settings.columns):
                root=f'/RackRollerRuntime/RollerDeck_{tier:02d}/Roller_r{row:02d}_c{col:02d}'
                old=UsdPhysics.RevoluteJoint(original.GetPrimAtPath(root+'_Joint'))
                new=UsdPhysics.RevoluteJoint(corrected.GetPrimAtPath(root+'_Joint'))
                assert old.GetBody0Rel().GetTargets()==new.GetBody0Rel().GetTargets()
                assert old.GetBody1Rel().GetTargets()==new.GetBody1Rel().GetTargets()
                for method in ('GetAxisAttr','GetLocalPos0Attr','GetLocalPos1Attr','GetLocalRot0Attr','GetLocalRot1Attr'):
                    assert getattr(old,method)().Get()==getattr(new,method)().Get()
                for attr in ('height','radius','axis'):
                    assert original.GetPrimAtPath(root+'/Geom').GetAttribute(attr).Get()==corrected.GetPrimAtPath(root+'/Geom').GetAttribute(attr).Get()
                drive=UsdPhysics.DriveAPI(new.GetPrim(),'angular')
                assert drive.GetDampingAttr().Get()*180/math.pi==pytest.approx(settings.angular_damping,rel=1e-6)
                assert drive.GetStiffnessAttr().Get()==0. and drive.GetMaxForceAttr().Get()==pytest.approx(.05)
    with pytest.raises(ValueError):write_passive_bearing_overlay(path,settings)

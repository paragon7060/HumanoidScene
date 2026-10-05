import json

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_view_identity import (
    environment_view_order,compare_contact_source_poses,
)


def test_backend_environment_order_must_match_scene_rows_not_lexical_sort():
    paths=[f'/World/envs/env_{i}/Box/Body' for i in range(12)]
    assert environment_view_order(paths,12)['matches_scene_row_order']
    wrong=environment_view_order(sorted(paths),12)
    assert not wrong['matches_scene_row_order'] and wrong['environment_ids'][2]==10
    assert not environment_view_order(paths[:2],12)['matches_scene_row_order']
    assert not environment_view_order(['/World/Shared/Body'],1)['matches_scene_row_order']


def test_contact_source_pose_check_uses_xyzw_and_accepts_quaternion_sign():
    sensor=torch.tensor([[10.,20.,30.,0.,0.,0.,1.],[40.,50.,60.,0.,0.,0.,-1.]])
    source=torch.tensor([[10.,20.,30.,1.,0.,0.,0.],[40.,50.,60.,1.,0.,0.,0.]])
    before=sensor.clone()
    result=compare_contact_source_poses(sensor,source)
    assert result['source_poses_agree'] and result['position_error_m']==[0.,0.]
    assert result['quaternion_absolute_unit_dot']==[1.,1.]
    torch.testing.assert_close(sensor,before)
    assert not compare_contact_source_poses(sensor.flip(0),source)['source_poses_agree']


def test_invalid_source_readback_is_recorded_without_repair_or_false_agreement():
    sensor=torch.tensor([[0.,0.,0.,0.,0.,0.,1.],[0.,0.,float('nan'),0.,0.,0.,0.]])
    source=torch.tensor([[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.]])
    result=compare_contact_source_poses(sensor,source)
    assert not result['source_poses_agree'] and result['invalid_pose_rows']==[False,True]
    assert result['position_error_m'][1] is None and result['quaternion_absolute_unit_dot'][1] is None
    json.dumps(result,allow_nan=False)
    assert torch.isnan(sensor[1,2])


def test_view_shape_mismatch_cannot_be_attributed():
    with pytest.raises(ValueError,match='equal environment'):
        compare_contact_source_poses(torch.zeros(2,7),torch.zeros(1,7))

"""DEV media must use the selected true body poses and retain original tensors."""
from types import SimpleNamespace
import pytest
import torch
from selected_scene_videos import measured_asset_view,validate_video_selection,video_scope,validate_video_layout_split


def test_measured_view_preserves_actual_flap_and_robot_pose_and_subtracts_only_origin():
    data=SimpleNamespace(root_pos_w=torch.arange(9.).reshape(3,3),root_quat_w=torch.arange(12.).reshape(3,4),
        body_link_pos_w=torch.arange(45.).reshape(3,5,3),body_link_quat_w=torch.arange(60.).reshape(3,5,4))
    before={k:v.clone() for k,v in vars(data).items()};origin=torch.tensor([[10.,20.,30.]])
    view=measured_asset_view(SimpleNamespace(data=data),2,origin).data
    torch.testing.assert_close(view.root_pos_w,data.root_pos_w[2:3]-origin)
    torch.testing.assert_close(view.body_link_pos_w,data.body_link_pos_w[2:3]-origin)
    assert torch.equal(view.body_link_quat_w,data.body_link_quat_w[2:3])
    assert all(torch.equal(v,getattr(data,k)) for k,v in before.items())
    rigid=SimpleNamespace(data=SimpleNamespace(root_pos_w=data.root_pos_w,root_quat_w=data.root_quat_w))
    assert not hasattr(measured_asset_view(rigid,1,origin).data,'body_link_pos_w')


@pytest.mark.parametrize('ids',[[0,0],[-1],[128],list(range(7))])
def test_invalid_video_selection_is_rejected(ids):
    with pytest.raises(ValueError):validate_video_selection(ids,128,900)


def test_video_selection_does_not_change_full_distribution_or_allow_short_policy_eval():
    assert validate_video_selection([0,5,42,3,2,4],128,900) is None
    assert validate_video_selection(None,128,1) is None
    with pytest.raises(ValueError):validate_video_selection([0],128,1)


def test_train_workplace_media_cannot_be_published_as_dev_or_final_score():
    scope=video_scope('train',workplace_search=True)
    assert scope['label']=='TRAIN workplace probe' and scope['workplace_search']
    assert video_scope('validation')['label']=='DEV'
    for split,workplace in [('train',False),('validation',True),('holdout',False),('holdout',True)]:
        with pytest.raises(ValueError):video_scope(split,workplace_search=workplace)


def test_canonical_dev_inner_holdout_label_is_preserved_without_mixing_train():
    heldout=[dict(layout=dict(split='holdout'))];training=[dict(layout=dict(split='train'))]
    validate_video_layout_split(heldout,'validation')
    validate_video_layout_split(training,'train')
    with pytest.raises(ValueError):validate_video_layout_split(training,'validation')
    with pytest.raises(ValueError):validate_video_layout_split(heldout,'train')

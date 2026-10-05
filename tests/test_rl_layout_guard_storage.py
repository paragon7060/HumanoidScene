import json
import pytest
from kuavo_isaaclab_scene.rl.multi_box.debug.layout_guard_storage import (
    load_layout_guard,store_layout_guard,
)


def test_full_evidence_is_stored_once_and_historical_inline_guards_still_read(tmp_path):
    guard=dict(ready=[True]*128,base_pose=[[1.,2.,3.]*20]*128,failures=[dict(environment=7)])
    references=store_layout_guard(tmp_path,3,128,guard)
    outcomes=[dict(wave=3,environment=i,initial_layout_guard=r) for i,r in enumerate(references)]
    assert len(list(tmp_path.glob('*.json')))==1
    assert all(load_layout_guard(row,tmp_path)==guard for row in outcomes)
    assert load_layout_guard(dict(initial_layout_guard=guard),tmp_path)==guard
    assert len(json.dumps(outcomes))<len(json.dumps([guard]*128))//10
    with pytest.raises(FileExistsError):store_layout_guard(tmp_path,3,128,guard)


@pytest.mark.parametrize('bad',['wave','environment','escape','symlink'])
def test_mismatched_or_external_guard_references_are_rejected(tmp_path,bad):
    ref=store_layout_guard(tmp_path,3,2,dict(ready=[True,False]))[0]
    outcome=dict(wave=3,environment=0,initial_layout_guard=ref)
    if bad=='wave':outcome['wave']=4
    elif bad=='environment':outcome['environment']=1
    elif bad=='escape':ref['file']='../outside.json'
    else:
        p=tmp_path/ref['file'];p.rename(tmp_path/'other.json');p.symlink_to(tmp_path/'other.json')
    with pytest.raises(ValueError):load_layout_guard(outcome,tmp_path)

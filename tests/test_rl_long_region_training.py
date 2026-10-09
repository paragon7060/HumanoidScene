"""Long TRAIN scheduling must preserve medium coverage and original DEV seeds."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from prepare_long_region_training import build
from prepare_region_grasp_layouts import prepare


def reference(tmp_path, mixed):
    recipe=prepare(tmp_path/'original',120_000_000,2,2,2,mixed_middle_box_types=mixed)
    def wave(group,split):
        return dict(split=split,layouts=[dict(episode_index=e['reference_episode_index'],layout=e['layout'])
            for e in recipe['records'] if e['group']==group])
    development=wave('development','validation')
    waves=[development,wave('train','train'),deepcopy(development)]
    path=tmp_path/'reference.json';path.write_text(json.dumps(waves))
    return path,json.loads(path.read_text())


@pytest.mark.parametrize('mixed',[False,True])
def test_long_generated_resets_keep_all_original_groups_and_disjoint_actual_seeds(tmp_path,mixed):
    source,original=reference(tmp_path,mixed)
    result=build(source,tmp_path/'long',seed_origin=121_000_000,train_waves=2,validate_every=1)
    waves=json.loads((tmp_path/'long/continuation_waves.json').read_text())
    expected=Counter((r['layout']['target_region'],r['layout'].get('target_box_type'))
        for r in original[0]['layouts'])
    forbidden={r['layout']['seed'] for wave in original for r in wave['layouts']}
    actual=[]
    for wave in waves:
        if wave['split']=='validation':assert wave==original[0]
        else:
            assert Counter((r['layout']['target_region'],r['layout'].get('target_box_type'))
                for r in wave['layouts'])==expected
            actual.extend(r['layout']['seed'] for r in wave['layouts'])
    assert len(actual)==len(set(actual))==16 and not forbidden.intersection(actual)
    assert result['mixed_middle_box_types']==mixed and result['independent_FINAL_not_in_plan']
    assert json.loads(source.read_text())==original


def test_unbalanced_medium_control_cannot_silently_lose_size_coverage(tmp_path):
    source,waves=reference(tmp_path,True)
    for wave in (waves[0],waves[-1]):
        next(e for e in wave['layouts'] if e['layout']['target_box_type']=='medium')['layout']['target_box_type']='small'
    source.write_text(json.dumps(waves))
    with pytest.raises(ValueError,match='six region/size groups'):
        build(source,tmp_path/'rejected',seed_origin=122_000_000,train_waves=2,validate_every=1)
    assert not (tmp_path/'rejected').exists()

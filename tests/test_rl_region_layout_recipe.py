"""Four-region recipes describe physical resets, with disjoint learning splits."""
import importlib.util
import json
from pathlib import Path

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import GraspLayout,layout_reset_observation
from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def test_balanced_recipe_preserves_actual_seed_and_checks_all_four_physical_regions(tmp_path):
    root=Path(__file__).resolve().parents[1]
    module_spec=importlib.util.spec_from_file_location('region_layout_recipe',root/'scripts/rl/prepare_region_grasp_layouts.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    output=tmp_path/'layouts';recipe=module.prepare(output,12000,2,1,2)
    batch,_=load_v2_grasp_demonstrations(root/'examples/demos/v2_grasp_quest_success.hdf5',self_collision_enabled=False)
    mapping=json.loads((output/'reference_episode_map.json').read_text())
    ids={group:set() for group in ('train','development','holdout')}
    regions={group:set() for group in ids}
    for row in recipe['records']:
        data=row['layout'];layout=GraspLayout(**data)
        source=select_reference_episode(batch,mapping[str(data['seed'])])['actor_obs'][0]
        saved=source.clone();reset=layout_reset_observation(source,layout,MultiBoxSpec())
        selected=int(reset[400:412].argmax())
        assert selected=={'shelf_2_left':4,'shelf_2_right':1,'shelf_3_left':9,'shelf_3_right':6}[data['target_region']]
        assert source.equal(saved) and source[:20].equal(reset[:20])
        assert data['align_initial_base_to_region'] and -.04<=data['lateral_m']<=-.02
        ids[row['group']].add(data['seed']);regions[row['group']].add(data['target_region'])
    assert all(value==set(module.REGIONS) for value in regions.values())
    assert not(ids['train']&ids['development'] or ids['train']&ids['holdout'] or ids['development']&ids['holdout'])
    assert recipe['box_fixed'] is False and recipe['synthetic_Q_transitions'] is False

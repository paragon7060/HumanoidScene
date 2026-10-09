#!/usr/bin/env python3
"""More real randomized TRAIN waves with unchanged four-region reset ranges.

This creates reset plans, not successful states, transitions or reward labels.
The original DEV resets repeat for model selection; independent FINAL stays
outside this training plan. Original invalid resets remain in the denominator.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from prepare_region_grasp_layouts import prepare, REGIONS


def build(reference_waves, output, *, seed_origin, train_waves=12, validate_every=3):
    if type(train_waves) is not int or not 1<=train_waves<=60 \
            or type(validate_every) is not int or not 1<=validate_every<=train_waves \
            or type(seed_origin) is not int or seed_origin<0:
        raise ValueError('Expected a nonnegative seed and1..60 TRAIN waves with valid DEV cadence')
    source=Path(reference_waves);reference=json.loads(source.read_text())
    development=deepcopy(reference[0])
    if development['split']!='validation' or development!=reference[-1]:
        raise ValueError('Original control must repeat the same whole DEV before and after TRAIN')
    quotas=Counter(e['layout']['target_region'] for e in development['layouts'])
    if set(quotas)!=set(REGIONS) or len(set(quotas.values()))!=1:
        raise ValueError('Whole DEV must contain equal counts for all four rack regions')
    per_region=next(iter(quotas.values()))
    size_quotas=Counter((e['layout']['target_region'], e['layout'].get('target_box_type'))
                        for e in development['layouts'])
    mixed_middle_box_types=any(size=='medium' for region,size in size_quotas)
    if mixed_middle_box_types:
        expected={(region,size):per_region//2 for region in REGIONS if region.startswith('shelf_2')
                  for size in ('small','medium')}
        expected.update({(region,'small'):per_region for region in REGIONS if region.startswith('shelf_3')})
        if per_region%2 or size_quotas!=expected:
            raise ValueError('Mixed-size control requires the original balanced six region/size groups')
    seen={e['layout']['seed'] for w in reference for e in w['layouts']}
    for entry in development['layouts']:
        layout=entry['layout']
        if layout['split']!='holdout' or entry['episode_index']!=int(layout['target_region'].startswith('shelf_3')):
            raise ValueError('DEV split or physical upper/lower reference differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    waves=[development];training_seeds=[];recipes=[]
    for index in range(train_waves):
        recipe_dir=output/'reset_recipes'/f'train_{index:03d}'
        recipe=prepare(recipe_dir,seed_origin+1000*index,per_region,0,2 if mixed_middle_box_types else 1,
                       mixed_middle_box_types=mixed_middle_box_types)
        records=[e for e in recipe['records'] if e['group']=='train']
        entries=[dict(episode_index=e['reference_episode_index'],layout=e['layout']) for e in records]
        seeds=[e['layout']['seed'] for e in entries]
        if len(set(seeds))!=len(seeds) or seen.intersection(seeds):
            raise ValueError('New TRAIN seeds overlap the control, DEV or another wave')
        assert all(e['layout']['split']=='train' for e in entries)
        assert Counter(e['layout']['target_region'] for e in entries)==quotas
        if Counter((e['layout']['target_region'], e['layout'].get('target_box_type')) for e in entries)!=size_quotas:
            raise ValueError('Long TRAIN plan changed the original region/size distribution')
        seen.update(seeds);training_seeds.extend(seeds);recipes.append(str(recipe_dir.relative_to(output)))
        waves.append(dict(split='train',layouts=entries))
        if (index+1)%validate_every==0 or index+1==train_waves:
            waves.append(deepcopy(development))
    plan=output/'continuation_waves.json';plan.write_text(json.dumps(waves,indent=2)+'\n')
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),
        artifact_type='long_randomized_actual_four_region_TRAIN_schedule_v1',
        reference_waves_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output_waves_SHA256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        regions=list(REGIONS),seed_origin=seed_origin,train_waves=train_waves,
        validation_waves=sum(w['split']=='validation' for w in waves),
        requested_per_wave=len(development['layouts']),requested_TRAIN_cases=len(training_seeds),
        train_per_region_per_wave=per_region,training_seeds=training_seeds,reset_recipe_directories=recipes,
        mixed_middle_box_types=mixed_middle_box_types,
        original_region_size_quotas={region+'/'+str(size):count for (region,size),count in size_quotas.items()},
        randomization_recipe='unchanged_prepare_region_grasp_layouts',
        lower_base_lateral_m=[-.20,.20],lower_base_outward_m=[.03,.25],lower_base_yaw_deg=[-15.,15.],
        upper_base_lateral_m=[-.08,.08],upper_base_outward_m=[.03,.10],upper_base_yaw_deg=[-5.,5.],
        upper_box_depth_m=[-.006,.006],box_lateral_m=[-.04,-.02],box_yaw_deg=[-1.,1.],
        box_and_background_randomization_unchanged=True,box_fixed=False,curriculum=False,
        generated_layouts_not_synthetic_transitions_or_success_labels=True,
        original_DEV_repeated_only_for_model_selection=True,independent_FINAL_not_in_plan=True,
        helper_reserved_holdout_files_not_imported=True,original_invalid_resets_stay_in_denominator=True,
        goal_not_complete=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-waves',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed-origin',type=int,required=True)
    parser.add_argument('--train-waves',type=int,default=12)
    parser.add_argument('--validate-every',type=int,default=3)
    args=parser.parse_args()
    result=build(args.reference_waves,args.output_dir,seed_origin=args.seed_origin,
        train_waves=args.train_waves,validate_every=args.validate_every)
    print(json.dumps({k:result[k] for k in ('train_waves','validation_waves','requested_per_wave',
        'requested_TRAIN_cases','independent_FINAL_not_in_plan')},ensure_ascii=False))


if __name__=='__main__':main()

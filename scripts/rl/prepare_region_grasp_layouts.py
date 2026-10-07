#!/usr/bin/env python3
"""Create disjoint train/development/final resets for all four rack regions.

This CPU-only tool describes initial conditions, not synthetic transitions.
The simulator must settle the dynamic boxes and measure every outcome.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import sample_layout


REGIONS=('shelf_2_left','shelf_2_right','shelf_3_left','shelf_3_right')


def prepare(output,seed_origin,train_per_region,development_per_region,eval_per_region,
            *, mixed_middle_box_types=False):
    counts=(train_per_region,development_per_region,eval_per_region)
    if seed_origin<0 or any(type(n) is not int or not 0<=n<=100 for n in counts) or eval_per_region<1:
        raise ValueError('Use a nonnegative seed origin and0..100 layouts per region, with at least one final layout')
    if type(mixed_middle_box_types) is not bool or mixed_middle_box_types and any(n%2 for n in counts):
        raise ValueError('Mixed middle sizes require even counts for balanced small/medium coverage')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    development=output/'development';development.mkdir()
    episodes={};records=[]
    for group_index,(group,count) in enumerate(zip(('train','development','holdout'),counts)):
        split='train' if group=='train' else 'holdout'
        for index in range(count):
            for region_index,region in enumerate(REGIONS):
                upper=region.startswith('shelf_3');right=region.endswith('right')
                seed=seed_origin+group_index*1000+region_index*100+index
                sampled=sample_layout(seed,split,depth_limit_m=.006 if upper else 0.)
                generator=torch.Generator().manual_seed(seed+310_000_000)
                lateral_bound=.08 if upper else .20
                outward_min,outward_max=(.03,.10) if upper else (.03,.25)
                yaw_bound=5 if upper else 15
                selected=(6 if right else 9) if upper else (1 if right else 4)
                layout=replace(sampled,target_region=region,align_initial_base_to_region=True,
                    target_box_type=('medium' if not upper and index%2 else 'small')
                        if mixed_middle_box_types else None,
                    distractors=tuple(i for i in sampled.distractors if i!=selected),
                    base_lateral_m=float((torch.rand((),generator=generator)*2-1)*lateral_bound),
                    base_outward_m=float(outward_min+torch.rand((),generator=generator)*(outward_max-outward_min)),
                    base_yaw_rad=float((torch.rand((),generator=generator)*2-1)*math.radians(yaw_bound))).validate()
                folder=development if group=='development' else output
                (folder/f'{split}_{index*4+region_index:03d}.json').write_text(json.dumps(layout.record(),indent=2)+'\n')
                episodes[str(seed)]=int(upper)
                records.append(dict(group=group,reference_episode_index=int(upper),layout=layout.record()))
    (output/'reference_episode_map.json').write_text(json.dumps(episodes,indent=2)+'\n')
    recipe=dict(artifact_type='four_region_dynamic_box_initial_base_layout_recipe',
        regions=list(REGIONS),physical_torso_extra_height_m=.06,
        region_remap='same shelf/type/depth cell; physical translation across actual rack centre',
        nominal_initial_base='translate source initial robot by the same region change, then randomize XY/yaw',
        starts_with_successful_arm_state=False,box_fixed=False,synthetic_Q_transitions=False,
        lower=dict(x_m=.20,outward_m=[.03,.25],yaw_deg=15),
        upper=dict(x_m=.08,outward_m=[.03,.10],yaw_deg=5,depth_m=.006),
        development_and_final_seeds_disjoint=True,records=records)
    if mixed_middle_box_types:
        recipe.update(box_type_sampling='middle small/medium balanced; upper small',
            size_reset_geometry_not_measured_success=True,
            measured_size_specific_waypoints_required_before_matching_SAC=True)
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    return recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed-origin',type=int,required=True)
    parser.add_argument('--train-per-region',type=int,default=4)
    parser.add_argument('--development-per-region',type=int,default=1)
    parser.add_argument('--eval-per-region',type=int,default=4)
    parser.add_argument('--mixed-middle-box-types',action='store_true',
        help='Prepare size-aware neutral resets; medium waypoints still require physical TRAIN calibration')
    args=parser.parse_args()
    recipe=prepare(args.output_dir,args.seed_origin,args.train_per_region,
                   args.development_per_region,args.eval_per_region,
                   mixed_middle_box_types=args.mixed_middle_box_types)
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),layouts=len(recipe['records']),
                         regions=recipe['regions'])))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Create a frozen supported-size TRAIN workplace matrix, without Q data."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import validate_cpu_workplace_probe


def prepare(recipe,physical,candidates=None):
    if candidates is None:
        candidates=[dict(name=name,offset_xy_yaw=offset) for name,offset in (
        ('original',[0.,0.,0.]),('closer4',[0.,-.04,0.]),('outward4',[0.,.04,0.]),
        ('left4',[-.04,0.,0.]),('right4',[.04,0.,0.]),('yaw_left5',[0.,0.,-math.radians(5)]),
            ('yaw_right5',[0.,0.,math.radians(5)]),('outward4_left4',[-.04,.04,0.]))]
    rows=[dict(episode_index=case['reference_episode_index'],layout=deepcopy(case['layout']),
        waypoint_probe=deepcopy(candidate)) for case in recipe['records'] if case['group']=='train'
        for candidate in candidates]
    waves=[dict(split='train',layouts=rows)]
    audit=validate_cpu_workplace_probe(waves,physical,enabled=True,device='cpu',training=False,
        steps=900,waypoint_enabled=True,explicit_frozen=True,unmeasured_size_probe=True)
    return waves,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-recipe',type=Path,required=True)
    parser.add_argument('--training-manifest',type=Path,required=True)
    parser.add_argument('--candidates-json',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    waves,audit=prepare(json.loads(args.layout_recipe.read_text()),json.loads(args.training_manifest.read_text()),
        json.loads(args.candidates_json.read_text()) if args.candidates_json else None)
    with args.output.open('x') as stream:json.dump(waves,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()),audit=audit)))


if __name__=='__main__':main()

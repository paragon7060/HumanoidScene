#!/usr/bin/env python3
"""Prepare measured size targets from two closed managed frozen TRAIN searches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import io
import torch

from compare_actor_train_memory import owned_stable_bytes
from summarize_batched_staged_run import read_snapshot
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplaces import build_size_workplaces


def closed_probe(run):
    run=run.resolve();parent=run.parent
    status=read_snapshot(parent/'status.json');managed=read_snapshot(parent/'launch.json')
    if (Path(managed['run']).resolve()!=run or status.get('training_exit_code')!=0
        or read_snapshot(run/'status.json').get('status')!='complete'):
        raise ValueError('Only a normally completed owned managed physical search is eligible')
    proc=Path('/proc',str(status['training_pid']))
    if proc.exists() and str(run).encode() in (proc/'cmdline').read_bytes():
        raise ValueError('The original physical writer must have stopped before preparation')
    if any(p.stat().st_uid!=os.getuid() for p in (run,parent)):
        raise ValueError('Use only our own completed searches')
    command=managed['command']
    checkpoint=Path(command[command.index('--checkpoint')+1])
    waypoints=Path(command[command.index('--waypoints')+1])
    manifest=read_snapshot(run/'manifest.json')
    result=summarize_workplace_results(manifest,read_snapshot(run/'metrics.json'))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import layout_generation_contract
    generator=manifest.get('layout_generation_contract')
    if generator!=layout_generation_contract():
        raise ValueError('Repeat discovery and confirmation with the corrected, recorded reset geometry')
    if not result.get('unmeasured_size_workplace_probe'):
        raise ValueError('Both searches must have measured supported sizes explicitly')
    hashes={name:hashlib.sha256(owned_stable_bytes(path)).hexdigest()
        for name,path in (('checkpoint',checkpoint),('waypoints',waypoints),('metrics',run/'metrics.json'))}
    hashes['layout_generation_contract']=generator
    state=torch.load(io.BytesIO(owned_stable_bytes(checkpoint)),map_location='cpu',weights_only=True)
    if (state['goal_contract']!=read_snapshot(run/'agent.yaml')
        or state['actor_updates']!=result['frozen_source_actor_updates']
        or state['critic_updates']!=result['frozen_source_critic_updates']):
        raise ValueError('Frozen outcomes must match the actual immutable policy and counters')
    return result,hashes,json.loads(owned_stable_bytes(waypoints))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--discovery-run',type=Path,required=True)
    parser.add_argument('--confirmation-run',type=Path,required=True)
    parser.add_argument('--selections-json',type=Path,required=True,
        help='REGION -> supported BOX_TYPE -> named candidate, all six required')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output file')
    first,h1,waypoints=closed_probe(args.discovery_run)
    second,h2,other=closed_probe(args.confirmation_run)
    if (h1['checkpoint']!=h2['checkpoint'] or h1['waypoints']!=h2['waypoints'] or waypoints!=other
        or h1['layout_generation_contract']!=h2['layout_generation_contract']):
        parser.error('Discovery and fresh confirmation must use the same immutable policy, waypoints and reset geometry')
    result=build_size_workplaces(waypoints,first,second,read_snapshot(args.selections_json),
        results_SHA256=dict(discovery=h1['metrics'],confirmation=h2['metrics']),checkpoint_SHA256=h1['checkpoint'])
    result['layout_generation_contract']=h1['layout_generation_contract']
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()),supported_region_size_targets=6,
        measured_TRAIN_only=True,fresh_Q_and_replay_still_required=True,
        physical_generalization_NOT_claimed=True)))


if __name__=='__main__':main()

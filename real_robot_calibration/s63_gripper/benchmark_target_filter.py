#!/usr/bin/env python3
"""Measure filter-only batch cost; no Isaac startup or robot access."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from kuavo_isaaclab_scene.robots.gripper_action import GripperTargetFilter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    synchronize=lambda: torch.cuda.synchronize() if args.device.startswith('cuda') else None
    rows=[]
    with torch.inference_mode():
        for count in [1,64,4096]:
            initial=torch.zeros(count,2,device=args.device)
            desired=torch.tensor([[1.,-1.]],device=args.device).expand(count,2)
            costs={}
            for stages in [1,2]:
                filt=GripperTargetFilter(initial,desired[0],
                    dict(closing_time_constant_s=.092,opening_time_constant_s=.076,stages=stages))
                for _ in range(200):filt.advance(desired,1/120)
                rounds=[]
                for _ in range(5):
                    synchronize();started=time.perf_counter()
                    for k in range(1000):
                        filt.advance(desired if k%100<50 else initial,1/120)
                    synchronize();rounds.append((time.perf_counter()-started)/1000)
                costs[stages]=median(rounds)
            rows.append(dict(environments=count,one_stage_us=costs[1]*1e6,two_stage_us=costs[2]*1e6,
                additional_both_hands_ms=(costs[2]-costs[1])*2*1000,
                additional_fraction_of_120hz_tick=(costs[2]-costs[1])*2*120))
    report=dict(device=args.device,method='Five 1000-advance rounds, synchronization only at round boundaries; two joints per hand; median amortized wall time',
                scope='Filter-only cost; does not establish full scene/RL throughput',rows=rows)
    output=Path(__file__).parent/'reports'/f'initial_response_filter_cost_{args.device.replace(":","_")}.json'
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

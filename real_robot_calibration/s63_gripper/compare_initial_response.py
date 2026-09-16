#!/usr/bin/env python3
"""Compare current Isaac candidate with all three recorded real repetitions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_dynamics import known_synthetic

ROOT=Path(__file__).resolve().parent


def normalized(trace, side, step, times):
    ss=[s for s in trace if s['side']==side and s['trial']==1 and s['step']==step]
    t=ss[0]['time_s']-1/120
    before=next(s for s in reversed(trace) if s['side']==side and s['time_s']<=t)
    start=before['driver_feedback_fraction_percent'][0]
    end=ss[-1]['driver_feedback_fraction_percent'][0]
    return np.interp(times,[0]+[s['time_s']-t for s in ss],
                     [0]+[(s['driver_feedback_fraction_percent'][0]-start)/(end-start) for s in ss])


def main():
    folder=ROOT/'reports'
    real=json.loads((folder/'wrist_gripper_motion_01_dynamics.json').read_text())
    optical=json.loads((folder/'wrist_gripper_motion_01_optical.json').read_text())
    candidate=json.loads((folder/'isaac_dynamics_initial_response_continuous.json').read_text())
    old=list(map(json.loads,(folder/'isaac_dynamics_calibrated_continuous.jsonl').open()))
    new=list(map(json.loads,(folder/'isaac_dynamics_initial_response_continuous.jsonl').open()))
    states=[r for r in map(json.loads,(ROOT/'data/wrist_gripper_motion_01/messages.jsonl').open())
            if r.get('topic')=='/leju_claw_state' and not known_synthetic(r)]
    times=np.arange(121)/120
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    rows=[]
    for ii,side in enumerate(['left','right']):
        for jj,(direction,step) in enumerate([('closing',3),('opening',4)]):
            ax=axes[jj,ii]
            observations=[o for o in real['observations'] if o['side']==side and o['direction']==direction]
            curves=[]
            for o in observations:
                t=o['command_receipt_unix_s']
                ss=[s for s in states if -.1<=s['receipt_unix_s']-t<=1.05]
                curve=np.interp(times,[s['receipt_unix_s']-t for s in ss],
                    [(s['position'][ii]-o['start_feedback_percent'])/o['travel_percent'] for s in ss])
                curves.append(curve)
                ax.plot(times,curve,color='black',alpha=.4,label='Real encoder (3 repeats)' if o['trial']==1 else None)
            for o in optical['observations']:
                if o['side']==side and o['direction']==direction and o['trial']==1:
                    for j in o['jaws']:
                        ax.plot(j['times_s'],j['projected_feature_fraction'],color='tab:blue',alpha=.4,
                                label='RGB feature progress (different observable)' if j['jaw']==1 else None)
            previous=normalized(old,side,step,times)
            proposed=normalized(new,side,step,times)
            old_rmse=float(np.sqrt(np.mean((np.asarray(curves)-previous)**2)))
            new_rmse=float(np.sqrt(np.mean((np.asarray(curves)-proposed)**2)))
            measured=float(np.median([o['travel_10_90_s'] for o in observations]))
            sim=next(s for s in candidate['comparison'] if s['side']==side and s['direction']==direction)
            travel=sim['isaac_primary_driver']['travel_10_90_s']
            row=dict(side=side,direction=direction,previous_rmse_fraction=old_rmse,
                candidate_rmse_fraction=new_rmse,rmse_reduction_fraction=1-new_rmse/old_rmse,
                real_feedback_10_90_s=measured,isaac_10_90_s=travel,relative_time_difference=travel/measured-1,
                shape_passed=new_rmse<old_rmse*.7,time_passed=abs(travel/measured-1)<=.10)
            rows.append(row)
            ax.plot(times,previous,'--',color='tab:gray',label='Previous one-stage Isaac')
            ax.plot(times,proposed,color='tab:green',linewidth=2,label='New two-stage Isaac')
            ax.set(title=f'{side.capitalize()} / {direction}',xlim=(0,.9),ylim=(-.03,1.04))
            ax.grid(alpha=.2)
            ax.set_xlabel('Seconds after command / simulator target update')
            ax.set_ylabel('Normalized progress')
            ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Initial response correction with a small two-stage target filter')
    fig.text(.5,.015,'Encoder fit is a feedback proxy. RGB feature progress is not millimeter tip gap; camera exposure time is unverified.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.96))
    fig.savefig(folder/'initial_response_comparison.png',dpi=150)
    cost=candidate['physics_loop_cost']
    performance_passed=cost is not None and cost['relative_difference']<=.05
    report=dict(status='validated' if all(r['shape_passed'] and r['time_passed'] for r in rows)
        and candidate['closure_passed'] and candidate['all_metrics_valid'] and performance_passed else 'needs_review',
        target_filter=candidate['target_filter'],pd_changed=False,comparison=rows,
        physics_loop_cost=cost,performance_passed=performance_passed,
        closure_passed=candidate['closure_passed'],all_metrics_valid=candidate['all_metrics_valid'],
        scope='Practical feedback-shape approximation, fitted to three repetitions of one motion amplitude; not force, contact, calibrated optical gap or full-scene throughput',
        acceptance='>=30% full-curve encoder RMSE reduction, <=10% 10-90 timing difference, <=5% stationary two-claw loop slowdown; no requirement for exact equality')
    validations = {}
    for mode in ['incremental']:
        p=folder/f'isaac_dynamics_initial_response_{mode}.json'
        if p.exists():
            r=json.loads(p.read_text())
            validations[mode]=dict(passed=r['closure_passed'] and r['all_metrics_valid'] and all(r['binary_semantics_checked'].values()),
                report=p.name,travel_10_90_s=[dict(side=o['side'],direction=o['direction'],time_s=o['isaac_primary_driver']['travel_10_90_s']) for o in r['comparison']])
    for side in ['left','right']:
        p=folder/f'isaac_mapping_initial_response_{side}.json'
        if p.exists():
            r=json.loads(p.read_text())
            validations[f'{side}_mapping']=dict(passed=r['passed'],report=p.name,
                max_full_cycle_fit_error_mm=r['max_full_cycle_fit_error_mm'],max_width_error_mm=r['max_width_error_mm'],
                max_closure_error_mm=r['max_closure_error_mm'],max_reset_zero_action_error_rad=r['max_reset_zero_action_error_rad'])
    report['additional_validation']=validations
    if any(not v['passed'] for v in validations.values()):report['status']='needs_review'
    report['all_additional_validation_complete']=len(validations)==3 and all(v['passed'] for v in validations.values())
    (folder/'initial_response_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

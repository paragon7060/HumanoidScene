#!/usr/bin/env python3
"""Offline two-stage filter candidate from recorded encoder and Isaac traces."""
import json
from pathlib import Path
import numpy as np
from analyze_dynamics import known_synthetic, timing_metrics

ROOT = Path(__file__).resolve().parent


def main():
    feedback = json.loads((ROOT/'reports/wrist_gripper_motion_01_dynamics.json').read_text())
    states = [r for r in map(json.loads, (ROOT/'data/wrist_gripper_motion_01/messages.jsonl').open())
              if r.get('topic') == '/leju_claw_state' and not known_synthetic(r)]
    baseline = list(map(json.loads, (ROOT/'reports/isaac_dynamics_continuous.jsonl').open()))
    current = list(map(json.loads, (ROOT/'reports/isaac_dynamics_calibrated_continuous.jsonl').open()))
    times = np.arange(121)/120
    rows = []
    for side, ii in [('left',0),('right',1)]:
        for direction, step in [('closing',3),('opening',4)]:
            real = [o for o in feedback['observations'] if o['side'] == side and o['direction'] == direction]
            curves = []
            for o in real:
                tt = o['command_receipt_unix_s']
                rr = [r for r in states if -.1 <= r['receipt_unix_s']-tt <= 1.05]
                curves.append(np.interp(times, [r['receipt_unix_s']-tt for r in rr],
                    [(r['position'][ii]-o['start_feedback_percent'])/o['travel_percent'] for r in rr]))
            curves = np.asarray(curves)
            def normalized(trace):
                ss = [r for r in trace if r['side'] == side and r['trial'] == 1 and r['step'] == step]
                start_t = ss[0]['time_s']-1/120
                start = next(r['driver_feedback_fraction_percent'][0] for r in reversed(trace)
                             if r['side'] == side and r['time_s'] <= start_t)
                end = ss[-1]['driver_feedback_fraction_percent'][0]
                return np.interp(times, [0]+[r['time_s']-start_t for r in ss],
                                 [0]+[(r['driver_feedback_fraction_percent'][0]-start)/(end-start) for r in ss])
            plant = normalized(baseline)
            previous = normalized(current)
            reference = float(np.median([o['travel_10_90_s'] for o in real]))
            candidates = []
            for tau in np.arange(.035,.141,.001):
                target = 1-np.exp(-times/tau)*(1+times/tau)
                predicted = np.convolve(np.diff(target, prepend=0), plant)[:len(times)]
                # Add settled tail only for the shared feedback timing audit.
                samples = [(i/120,[0.]) for i in range(-120,0)]
                samples += [(t,[100*p]) for t,p in zip(times,predicted)]
                samples += [(i/120,[100.]) for i in range(121,601)]
                timing = timing_metrics(samples,0,0)
                error = float(np.sqrt(np.mean((curves-predicted)**2)))
                deviation = abs(timing['travel_10_90_s']/reference-1)
                if deviation <= .08:
                    candidates.append((error,float(tau),timing['travel_10_90_s']))
            if not candidates:raise ValueError('No candidate within the 10% timing budget')
            error,tau,travel = min(candidates)
            rows.append(dict(side=side,direction=direction,candidate_tau_s=round(tau,3),stages=2,
                predicted_rmse_fraction=error,previous_rmse_fraction=float(np.sqrt(np.mean((curves-previous)**2))),
                real_feedback_10_90_s=reference,predicted_10_90_s=travel,
                predicted_relative_time_difference=travel/reference-1))
    report = dict(scope='Offline linear step-response approximation; requires actual Isaac verification',
        acceptance='Reduce full-curve encoder RMSE; offline timing within 8% leaves margin for actual Isaac <=10%; no millimeter or latency claim',
        candidates=rows)
    (ROOT/'reports/initial_response_candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':main()

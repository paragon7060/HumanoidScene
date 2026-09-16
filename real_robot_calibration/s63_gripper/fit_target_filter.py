#!/usr/bin/env python3
"""Generate preliminary target-filter taus from real/Isaac baseline reports.

Offline first-order approximation only. Does not edit configuration or use
Isaac/SSH; candidates require actual simulator validation before adoption.
"""
import argparse
import json
import math
from pathlib import Path
from analyze_dynamics import timing_metrics

FOLDER = Path(__file__).resolve().parent


def predicted_time(target_tau, plant_tau, dt=1/120):
    target = actual = 0.
    samples = [(i*dt,[0.]) for i in range(-120,1)]
    target_alpha = -math.expm1(-dt/target_tau)
    plant_alpha = -math.expm1(-dt/plant_tau)
    for i in range(1,601):
        target += target_alpha*(1-target)
        actual += plant_alpha*(target-actual)
        samples.append((i*dt,[100*actual]))
    return timing_metrics(samples,0,0)["travel_10_90_s"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=FOLDER/"reports/target_filter_candidates.json")
    args = parser.parse_args()
    real = json.loads((FOLDER/"reports/dynamics_speed_comparison.json").read_text())
    base = json.loads((FOLDER/"reports/isaac_dynamics_continuous.json").read_text())
    if base.get("target_filter") and any(base["target_filter"].values()):
        parser.error("Baseline must have no target filter")
    rows = []
    for reference in real["pooled"]:
        row = next(r for r in base["comparison"] if r["side"]==reference["side"] and r["direction"]==reference["direction"])
        plant_tau = row["isaac_primary_driver"]["travel_10_90_s"]/math.log(9)
        desired = reference["travel_10_90_s"]["median"]
        lo,hi = .001,.3
        if not predicted_time(lo,plant_tau)<=desired<=predicted_time(hi,plant_tau):
            parser.error("Reference outside modeled candidate range; collect/inspect data before expanding it")
        for _ in range(20):
            mid = (lo+hi)/2
            if predicted_time(mid,plant_tau)<desired:
                lo = mid
            else:
                hi = mid
        tau = round((lo+hi)/2,4)
        rows.append(dict(side=reference["side"],direction=reference["direction"],candidate_tau_s=tau,
                         real_feedback_10_90_s=desired,predicted_10_90_s=predicted_time(tau,plant_tau),plant_tau_s=plant_tau))
    report = dict(method="First-order approximation to measured Isaac baseline, preliminary candidate requiring actual CUDA validation",
                  real_reference="dynamics_speed_comparison.json",isaac_baseline="isaac_dynamics_continuous.json",candidates=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

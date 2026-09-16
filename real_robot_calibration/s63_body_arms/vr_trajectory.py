#!/usr/bin/env python3
"""Inspect VR sim responses and export bounded, NON-EXECUTABLE real replay candidates."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from calibrate import load_limits, DEFAULT_URDF
from trajectory import NAMES, finite_vector, sample


def read_log(path):
    with Path(path).open() as stream:
        rows=[json.loads(line) for line in stream]
    if not rows or rows[0].get("schema") != "kuavo_joint_response_v1":
        raise ValueError("Expected a joint response JSONL, not a dataset action array")
    meta=rows[0]
    names=meta.get("joint_names",[])
    if not names or len(set(names))!=len(names):
        raise ValueError("Missing/duplicate joint names")
    return meta,[r for r in rows if r.get("kind")=="step"],bool(rows[-1].get("kind")=="closed" and rows[-1].get("ok"))


def inspect(path):
    meta,rows,closed=read_log(path)
    segments={}
    for r in rows:
        entry=segments.setdefault(str(r["segment"]),{"steps":0,"exportable_steps":0,
            "start_s":r["sim_start_s"],"end_s":r["sim_end_s"]})
        entry["steps"]+=1;entry["exportable_steps"]+=bool(r["exportable"])
        entry["end_s"]=r["sim_end_s"]
    valid=[r for r in rows if r.get("exportable")]
    errors={}
    for i,n in enumerate(meta["joint_names"]):
        e=[math.degrees(r["q_after_rad"][i]-r["logical_joint_target_rad"][i]) for r in valid]
        errors[n]={"mean_deg":statistics.mean(e),"rms_deg":math.sqrt(statistics.mean(x*x for x in e)),
                   "max_abs_deg":max(abs(x) for x in e)} if e else None
    return {"source":str(Path(path).resolve()),"closed":closed,"metadata":meta,"segments":segments,
            "control_step_tracking":errors,"note":"Last logical command versus end-of-step sim q; not fitted motor delay or real torque"}


def candidate(path,capture,limits,segment,joint,scale=1.,time_scale=5.,waypoint_s=1.,start_s=None,end_s=None):
    meta,rows,closed=read_log(path)
    if not closed or not meta.get("logical_target_unbiased"):
        raise ValueError("Export requires a closed log and verified unbiased logical targets")
    if meta.get("robot_model")!="s63" or capture.get("robot_version")!=63 or capture.get("joint_names")!=NAMES:
        raise ValueError("Initial real replay candidate supports verified S63 only")
    if capture.get("baseline_source")!="live_quick_mode_targets":
        raise ValueError("Real capture must contain verified live quick targets")
    if joint not in NAMES[4:]:
        raise ValueError("Initial candidate selects one arm joint; body/gripper/base replay requires separate review")
    if not math.isfinite(scale) or not 0 < scale <= 1 or not math.isfinite(time_scale) or time_scale < 1 or not math.isfinite(waypoint_s) or waypoint_s < .1:
        raise ValueError("scale (0,1], time-scale >=1, waypoint interval >=0.1s required")
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in (start_s,end_s)):
        raise ValueError("Invalid source time range")
    rows=[r for r in rows if r["segment"]==segment and
          (start_s is None or r["sim_start_s"]>=start_s) and
          (end_s is None or r["sim_end_s"]<=end_s)]
    if len(rows)<2 or any(not r.get("exportable") or r.get("reset_after_step") or r.get("collision") or
                          r.get("context",{}).get("collision_modified") for r in rows):
        raise ValueError("Choose a continuous tracked, active, collision-free segment/time range")
    width=len(meta["joint_names"])
    for r in rows:
        finite_vector(r.get("logical_joint_target_rad"),width,"logical target")
        if not math.isfinite(r["sim_start_s"]) or not math.isfinite(r["sim_end_s"]) or r["sim_end_s"]<=r["sim_start_s"]:
            raise ValueError("Invalid control timestamps")
        if any(abs(v)>1e-6 for v in r.get("context",{}).get("base_action",[])):
            raise ValueError("Base input is outside this arm-only replay scope")
    for a,b in zip(rows,rows[1:]):
        if b["seq"]!=a["seq"]+1 or abs(b["sim_start_s"]-a["sim_end_s"])>1e-6 or b["sim_start_s"]-a["sim_start_s"]>.1:
            raise ValueError("Gapped/discontinuous source; do not bridge lost tracking or reset")
    reference=finite_vector(rows[0]["root_before_w"],7,"source root")
    translation,rotation=0.,0.
    for r in rows:
        for key in ("root_before_w","root_after_w"):
            p=finite_vector(r[key],7,"root pose")
            translation=max(translation,math.sqrt(sum((a-b)**2 for a,b in zip(p[:3],reference[:3]))))
            n=math.sqrt(sum(x*x for x in p[3:]));rn=math.sqrt(sum(x*x for x in reference[3:]))
            if not .9<=n<=1.1 or not .9<=rn<=1.1:
                raise ValueError("Invalid root quaternion")
            rotation=max(rotation,math.degrees(2*math.acos(min(1.,abs(sum(a*b for a,b in zip(p[3:],reference[3:]))/n/rn)))))
    if translation>.005 or rotation>.5:
        raise ValueError("Source base moved; initial arm replay requires stationary root")
    base=finite_vector(capture.get("baseline_rad"),18,"real baseline")
    low=[limits[n][0] for n in NAMES];high=[limits[n][1] for n in NAMES]
    if any(not lo-math.radians(.01)<=q<=hi+math.radians(.01) for q,lo,hi in zip(base,low,high)):
        raise ValueError("Real baseline outside URDF limits")
    src=meta["joint_names"].index(joint);dst=NAMES.index(joint)
    initial=rows[0]["logical_joint_target_rad"][src]
    offsets=[scale*(r["logical_joint_target_rad"][src]-initial) for r in rows]
    max_offset=max(abs(x) for x in offsets)
    if max_offset>math.radians(1):
        raise ValueError("Candidate exceeds +/-1deg real scope; trim source or explicitly reduce --scale")
    if max_offset<1e-5:
        raise ValueError("Selected joint has no useful source motion")
    margin=math.radians(.5)
    if any(not low[dst]+margin<=base[dst]+x<=high[dst]-margin for x in offsets):
        raise ValueError("Rebased targets violate real joint limit margin")
    selected=[rows[0]]
    for r in rows[1:]:
        if r["sim_start_s"]-selected[-1]["sim_start_s"]>=waypoint_s:
            selected.append(r)
    if selected[-1] is not rows[-1]:selected.append(rows[-1])
    phases=[{"label":"baseline_hold","duration_s":3.,"from_rad":base,"to_rad":base}]
    previous=base
    peak_v=peak_a=0.
    for index,r in enumerate(selected[1:],1):
        target=base.copy();target[dst]+=scale*(r["logical_joint_target_rad"][src]-initial)
        delta=abs(math.degrees(target[dst]-previous[dst]))
        requested=(r["sim_start_s"]-selected[index-1]["sim_start_s"])*time_scale
        duration=max(2.,requested,1.875*delta/.469,math.sqrt((10/math.sqrt(3))*delta/.361))
        peak_v=max(peak_v,1.875*delta/duration);peak_a=max(peak_a,(10/math.sqrt(3))*delta/duration**2)
        phases.append({"label":f"vr_waypoint_{index}","duration_s":duration,"from_rad":previous,"to_rad":target})
        previous=target
    delta=abs(math.degrees(previous[dst]-base[dst]));duration=max(4.,1.875*delta/.469,math.sqrt((10/math.sqrt(3))*delta/.361))
    phases += [{"label":"return","duration_s":duration,"from_rad":previous,"to_rad":base},
               {"label":"final_hold","duration_s":3.,"from_rad":base,"to_rad":base}]
    peak_v=max(peak_v,1.875*delta/duration);peak_a=max(peak_a,(10/math.sqrt(3))*delta/duration**2)
    total=sum(p["duration_s"] for p in phases)
    if total>900:raise ValueError("Candidate >900s; choose shorter source range")
    return {"schema":"kuavo_vr_replay_candidate_v1","send_supported":False,
            "review_required":["collision/clearance/balance at real baseline","matched sim replay of processed candidate","fresh real capture and command ownership"],
            "robot_version":63,"joint_names":NAMES,"joints":[joint],"baseline_rad":base,
            "lower_rad":low,"upper_rad":high,"capture":capture,"phases":phases,"duration_s":total,
            "max_offset_deg":math.degrees(max_offset),"max_velocity_deg_s":peak_v,"max_acceleration_deg_s2":peak_a,
            "source":str(Path(path).resolve()),"source_segment":segment,"source_start_s":rows[0]["sim_start_s"],
            "source_end_s":rows[-1]["sim_end_s"],"scale":scale,"time_scale":time_scale,"waypoint_s":waypoint_s,
            "target_semantics":"Only logical target offsets; solver/gravity targets NEVER exported",
            "processing":"single-joint extraction, rebase, sparse waypoints, quintic rest-to-rest retiming; compare with matched processed sim replay, not original live VR speed",
            "source_other_joint_max_motion_deg":{n:math.degrees(max(abs(r["logical_joint_target_rad"][i]-rows[0]["logical_joint_target_rad"][i]) for r in rows)) for i,n in enumerate(meta["joint_names"]) if n!=joint}}


def validate_candidate(plan, limits):
    """Rebuild from the source rather than trusting editable output waypoints."""
    if plan.get("schema") != "kuavo_vr_replay_candidate_v1" or plan.get("send_supported") is not False:
        raise ValueError("Expected a non-executable VR replay candidate")
    rebuilt = candidate(plan["source"], plan["capture"], limits, plan["source_segment"],
                        plan["joints"][0], plan["scale"], plan["time_scale"], plan["waypoint_s"],
                        plan["source_start_s"], plan["source_end_s"])
    if rebuilt != plan:
        raise ValueError("Candidate differs from its source and bounded generated sequence")
    return rebuilt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    view=commands.add_parser("inspect");view.add_argument("--log",type=Path,required=True)
    out=commands.add_parser("export");out.add_argument("--log",type=Path,required=True)
    out.add_argument("--capture",type=Path,required=True);out.add_argument("--segment",type=int,required=True)
    out.add_argument("--joint",choices=NAMES[4:],required=True);out.add_argument("--scale",type=float,default=1.)
    out.add_argument("--time-scale",type=float,default=5.);out.add_argument("--waypoint-s",type=float,default=1.)
    out.add_argument("--start-s",type=float);out.add_argument("--end-s",type=float)
    out.add_argument("--urdf",type=Path,default=DEFAULT_URDF);out.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="inspect":
            print(json.dumps(inspect(args.log),indent=2,allow_nan=False));return 0
        if args.output.suffix!=".json" or args.output.exists() or args.output.with_suffix(".csv").exists():
            raise ValueError("Choose new .json/.csv output paths")
        plan=candidate(args.log,json.loads(args.capture.read_text()),load_limits(args.urdf),args.segment,args.joint,
                       args.scale,args.time_scale,args.waypoint_s,args.start_s,args.end_s)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:stream.write(json.dumps(plan,indent=2,allow_nan=False)+"\n")
        with args.output.with_suffix(".csv").open("x",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["elapsed_s","phase"]+[n+"_rad" for n in NAMES]+[n+"_rad_s" for n in NAMES])
            for i in range(math.ceil(plan["duration_s"]*50)+1):
                t=min(i/50,plan["duration_s"]);q,v,label=sample(plan,t);writer.writerow([t,label]+q+v)
        print(json.dumps({"output":str(args.output.resolve()),"duration_s":plan["duration_s"],"send_supported":False,
                          "max_offset_deg":plan["max_offset_deg"],"max_velocity_deg_s":plan["max_velocity_deg_s"]}));return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        parser.exit(1,"ERROR: "+str(exc)+"\n")


if __name__=="__main__":raise SystemExit(main())

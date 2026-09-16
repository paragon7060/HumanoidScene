#!/usr/bin/env python3
"""Read-only named static-pose capture and offline quality checks; no actuation."""
import argparse
import bisect
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics

import calibrate
from trajectory import NAMES, finite_vector

TOPICS = {"sensors": "/sensors_data_raw", "motor": "/joint_cmd", "imu": "/imu"}


def analyze(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    summaries = [r for r in rows if r.get("kind") == "summary"]
    failures = ["record did not finish successfully"] if not summaries or not summaries[-1].get("ok") or any(r.get("kind") == "error" for r in rows) else []
    end = summaries[-1]["receipt_monotonic_s"] if summaries else max(r.get("receipt_monotonic_s", 0) for r in rows)
    channels = {}
    stats = {}
    enabled = [r for r in rows if r.get("topic") == "/enable_control_state"]
    if not enabled or enabled[-1].get("data") is not True:
        failures.append("WBC enable state missing/false")
    for key, topic in TOPICS.items():
        points = sorted((r for r in rows if r.get("kind") == "message" and r.get("topic") == topic
                         and end-3 <= r["receipt_monotonic_s"] <= end), key=lambda r:r["receipt_monotonic_s"])
        channels[key] = points
        stamps = [r["receipt_monotonic_s"] for r in points]
        info = {"samples":len(points), "coverage_s":stamps[-1]-stamps[0] if stamps else 0,
                "max_gap_s":max((b-a for a,b in zip(stamps,stamps[1:])),default=None),
                "last_age_s":end-stamps[-1] if stamps else None,
                "publishers":sorted({r.get("publisher", "") for r in points})}
        stats[key] = info
        if (len(points) < 60 or info["coverage_s"] < 2.9 or
                info["max_gap_s"] is None or info["max_gap_s"] > .1 or info["last_age_s"] > .1):
            failures.append(key+": sparse/stale/gapped 3s window")
        if len(info["publishers"]) != 1 or not info["publishers"][0]:
            failures.append(key+": ambiguous publisher")
        headers = [r.get("header_s",0) for r in points]
        if not headers or any(b <= a for a,b in zip(headers,headers[1:])):
            failures.append(key+": missing/frozen/nonmonotonic header")
    sensor, motor, imu = (channels[k] for k in ("sensors", "motor", "imu"))
    sensor_q = [finite_vector(r["joint_q"],20,"measured q") for r in sensor]
    sensor_v = [finite_vector(r["joint_v"],20,"measured velocity") for r in sensor]
    motor_q = [finite_vector(r["joint_q"],20,"motor q") for r in motor]
    for r in motor:
        finite_vector(r.get("joint_kp"),20,"motor kp")
        finite_vector(r.get("joint_kd"),20,"motor kd")
    if any(r.get("control_modes") != [2]*20 for r in motor):
        failures.append("motor mode differs from 2")
    motor_times = [r["receipt_monotonic_s"] for r in motor]
    pairs = []
    for r in sensor:
        index = bisect.bisect_right(motor_times,r["receipt_monotonic_s"])-1
        if index >= 0 and r["receipt_monotonic_s"]-motor_times[index] <= .05:
            pairs.append((r,motor[index]))
    if len(pairs) < 60:
        failures.append("fewer than 60 receipt pairs within 50ms")
    joints = {}
    for i,name in enumerate(NAMES):
        if not sensor_q or not motor_q:
            break
        span = math.degrees(max(q[i] for q in sensor_q)-min(q[i] for q in sensor_q))
        cmd_span = math.degrees(max(q[i] for q in motor_q)-min(q[i] for q in motor_q))
        median_v = math.degrees(statistics.median(v[i] for v in sensor_v))
        peak_v = math.degrees(max(abs(v[i]) for v in sensor_v))
        if span > .05 or abs(median_v) > .5 or peak_v > 10:
            failures.append(name+": measured posture not quiet")
        if cmd_span > .01:
            failures.append(name+": motor target changed")
        e = [math.degrees(s["joint_q"][i]-m["joint_q"][i]) for s,m in pairs]
        joint = {"measured_mean_deg":math.degrees(statistics.mean(q[i] for q in sensor_q)),
                 "measured_span_deg":span, "motor_span_deg":cmd_span,
                 "median_velocity_deg_s":median_v,"raw_peak_velocity_deg_s":peak_v,
                 "error_mean_deg":statistics.mean(e) if e else None,
                 "error_rms_deg":math.sqrt(statistics.mean(x*x for x in e)) if e else None}
        for field, points, output in (("joint_torque",sensor,"measured_effort_raw_mean"),
                                      ("tau",motor,"command_effort_raw_mean")):
            vectors = [r.get(field) for r in points]
            values = [v[i] if isinstance(v,list) and len(v)==20 else None for v in vectors]
            valid = all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
            joint[output] = statistics.mean(values) if values and valid else None
        joint["motor_gain_vectors"] = [list(v) for v in sorted({(r["joint_kp"][i],r["joint_kd"][i]) for r in motor})]
        if len(joint["motor_gain_vectors"]) != 1:
            failures.append(name+": motor gains changed")
        joints[name] = joint
    quats = [finite_vector(r["imu_quat_xyzw"],4,"IMU quaternion") for r in imu]
    frames = {r.get("frame_id", "") for r in imu}
    imu_change = None
    if quats:
        norms = [math.sqrt(sum(v*v for v in q)) for q in quats]
        if any(not .9 <= n <= 1.1 for n in norms) or any(r.get("orientation_covariance",[-1])[0]<0 for r in imu):
            failures.append("invalid IMU quaternion/covariance")
        if all(n > 0 for n in norms):
            ref = [v/norms[0] for v in quats[0]]
            imu_change = max(math.degrees(2*math.acos(min(1.,abs(sum(a*b/n for a,b in zip(ref,q)))))) for q,n in zip(quats,norms))
            if imu_change > .5:
                failures.append("IMU changed more than 0.5deg in static window")
    if len(frames) != 1 or not next(iter(frames), ""):
        failures.append("IMU frame changed/missing")
    return {"valid_for_static_comparison":not failures,"failures":failures,"window_s":3,
            "channels":stats,"paired_samples":len(pairs),"imu_change_deg":imu_change,
            "joints":joints,"effort_units":"UNVERIFIED; raw values, not measured output Nm",
            "source":str(Path(path).resolve()),"no_robot_command":True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",required=True,help="e.g. A_reference, B_bent, C_forward")
    parser.add_argument("--description",default="")
    parser.add_argument("--payload-kg",type=float,default=None,help="Only enter independently known attached mass")
    parser.add_argument("--approach",choices=("unspecified","positive","negative"),default="unspecified")
    parser.add_argument("--root",type=Path,default=Path(__file__).parent/"data/static_poses")
    args=parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",args.label):
        parser.error("label must contain only letters/numbers/_/-")
    if args.payload_kg is not None and (not math.isfinite(args.payload_kg) or args.payload_kg < 0):
        parser.error("payload mass must be finite and nonnegative")
    session=args.root/(args.label+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    session.mkdir(parents=True,exist_ok=False)
    metadata={"label":args.label,"description":args.description,"payload_kg":args.payload_kg,
              "approach":args.approach,"no_robot_command":True}
    (session/"condition.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    for command in (["capture","--output",str(session/"pose.json")],
                    ["record","--duration-s","12","--log",str(session/"record.jsonl")]):
        if calibrate.main(command):
            return 1
    try:
        result=analyze(session/"record.jsonl")
    except (ValueError,KeyError,IndexError,TypeError) as exc:
        result={"valid_for_static_comparison":False,"failures":["Invalid recorded data: "+str(exc)],
                "source":str((session/"record.jsonl").resolve()),"no_robot_command":True}
    result["condition"]=metadata
    (session/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"session":str(session.resolve()),"valid_for_static_comparison":result["valid_for_static_comparison"],
                      "failures":result["failures"]},ensure_ascii=False))
    return 0 if result["valid_for_static_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

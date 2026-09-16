#!/usr/bin/env python3
"""Analyze passive real-VR command/state logs; never imports ROS or contacts the robot."""
import argparse
import bisect
import json
import math
from pathlib import Path
import statistics

from trajectory import NAMES, finite_vector

ARM_NAMES = NAMES[4:]
BODY_NAMES = NAMES[:4]


def analyze(path, start_s=None, end_s=None):
    rows = [json.loads(line) for line in Path(path).open()]
    summary = [r for r in rows if r.get("kind") == "summary"]
    if not summary or not summary[-1].get("ok"):
        raise ValueError("VR record did not close with summary ok=true")
    metadata = next((r for r in rows if r.get("kind") == "metadata"), None)
    if metadata is None:
        raise ValueError("Missing remote metadata timestamp")
    origin = float(metadata["receipt_monotonic_s"])
    if start_s is not None or end_s is not None:
        lo = 0.0 if start_s is None else start_s
        hi = math.inf if end_s is None else end_s
        rows = [r for r in rows if r.get("kind") != "message" or
                lo <= float(r.get("receipt_monotonic_s", -math.inf))-origin <= hi]

    def series(topic, time_key, values, layer=None, length=14):
        result = []
        for row in rows:
            if row.get("topic") != topic or (layer is not None and row.get("layer") != layer):
                continue
            try:
                stamp, vector = float(row[time_key]), finite_vector(values(row), length, topic)
            except (KeyError, TypeError, ValueError):
                continue
            if stamp > 0:
                result.append((stamp, vector, row))
        return sorted(result)

    arm = series("/kuavo_arm_traj", "header_s",
                 lambda r: [math.radians(v) for v in r["payload"]["position"]], "vr_input")
    motor = series("/joint_cmd", "header_s", lambda r: r["joint_q"][4:18])
    measured = series("/sensors_data_raw", "header_s", lambda r: r["joint_q"][4:18])
    measured_velocity = series("/sensors_data_raw", "header_s", lambda r: r["joint_v"][4:18])
    motor_body = series("/joint_cmd", "header_s", lambda r: r["joint_q"][:4], length=4)
    measured_body = series("/sensors_data_raw", "header_s", lambda r: r["joint_q"][:4], length=4)
    for label, values in (("VR arm", arm), ("motor target", motor), ("measured state", measured)):
        if len(values) < 50:
            raise ValueError(f"Too few {label} samples: {len(values)}")

    def rate(values):
        receipt = sorted(float(r["receipt_monotonic_s"]) for _, _, r in values)
        gaps = [b-a for a, b in zip(receipt, receipt[1:]) if b > a]
        latency = [1000*(float(r["receipt_unix_s"])-t) for t, _, r in values]
        return {"count": len(values), "median_hz": 1/statistics.median(gaps),
                "max_gap_ms": 1000*max(gaps), "receipt_minus_header_ms": {
                    "median": statistics.median(latency), "min": min(latency), "max": max(latency)}}

    def spans(values):
        return {name: math.degrees(max(v[i] for _, v, _ in values)-min(v[i] for _, v, _ in values))
                for i, name in enumerate(ARM_NAMES)}

    def body_spans(values):
        return {name: math.degrees(max(v[i] for _, v, _ in values)-min(v[i] for _, v, _ in values))
                for i, name in enumerate(BODY_NAMES)}

    def prepare(values):
        return [t for t, _, _ in values], [v for _, v, _ in values]

    def interpolate(source, stamp):
        times, vectors = source
        index = bisect.bisect_left(times, stamp)
        if index == 0 or index == len(times):
            return None
        weight = (stamp-times[index-1])/(times[index]-times[index-1])
        return [a+weight*(b-a) for a, b in zip(vectors[index-1], vectors[index])]

    def score(source, target, delay, joint=None):
        errors = []
        for stamp, target_vector, _ in target:
            source_vector = interpolate(source, stamp-delay)
            if source_vector is None:
                continue
            ids = range(len(target_vector)) if joint is None else (joint,)
            errors.extend((target_vector[i]-source_vector[i])**2 for i in ids)
        return math.sqrt(statistics.mean(errors)) if len(errors) >= 100 else math.inf

    def best_delay(source_values, target_values, names):
        source = prepare(source_values)
        coarse = [i/1000 for i in range(0, 501, 5)]
        overall_coarse = min(coarse, key=lambda d: score(source, target_values, d))
        fine = [i/1000 for i in range(max(0, round(1000*overall_coarse)-10),
                                      min(500, round(1000*overall_coarse)+10)+1)]
        overall = min(fine, key=lambda d: score(source, target_values, d))
        joints = {}
        for index, name in enumerate(names):
            first = min(coarse, key=lambda d: score(source, target_values, d, index))
            candidates = [i/1000 for i in range(max(0, round(1000*first)-10),
                                                 min(500, round(1000*first)+10)+1)]
            delay = min(candidates, key=lambda d: score(source, target_values, d, index))
            joints[name] = {"delay_ms": round(1000*delay),
                            "rmse_deg": math.degrees(score(source, target_values, delay, index))}
        return {"overall_delay_ms": round(1000*overall),
                "overall_rmse_deg": math.degrees(score(source, target_values, overall)),
                "per_joint": joints}

    publishers = sorted({r.get("publisher") for _, _, r in arm})
    gains = {}
    motor_rows = [r for r in rows if r.get("topic") == "/joint_cmd"]
    for key in ("joint_kp", "joint_kd", "tau_max", "tau_ratio", "control_modes"):
        unique = {tuple(r[key][4:18]) for r in motor_rows}
        gains[key] = {"constant": len(unique) == 1, "values": list(next(iter(unique))) if len(unique) == 1 else None}
    return {"source": str(Path(path).resolve()), "status": "exploratory real-VR capture",
            "receipt_window_s": {"start": start_s, "end": end_s},
            "observed_arm_publishers": publishers,
            "streams": {"vr_arm_target": rate(arm), "motor_target": rate(motor), "measured": rate(measured)},
            "span_deg": {"vr_arm_target": spans(arm), "motor_target": spans(motor), "measured": spans(measured)},
            "body_span_deg": {"motor_target": body_spans(motor_body),
                              "measured": body_spans(measured_body)},
            "position_delay_fit": {"vr_target_to_motor_target": best_delay(arm, motor, ARM_NAMES),
                                   "motor_target_to_measured": best_delay(motor, measured, ARM_NAMES),
                                   "body_motor_target_to_measured": best_delay(
                                       motor_body, measured_body, BODY_NAMES)},
            "max_abs_measured_velocity_deg_s": max(abs(math.degrees(x)) for _, v, _ in measured_velocity for x in v),
            "motor_fields": gains,
            "limitations": ["Header-time trajectory fit is an estimate, not causal proof",
                            "No labeled controlled excitation or hold intervals",
                            "Command clipping/filtering can bias fitted delay",
                            "Demanded and measured torque units are not verified",
                            "No matched simulation replay yet"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-s", type=float, help="First receipt second relative to remote metadata")
    parser.add_argument("--end-s", type=float, help="Last receipt second relative to remote metadata")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    if ((args.start_s is not None and (not math.isfinite(args.start_s) or args.start_s < 0)) or
            (args.end_s is not None and not math.isfinite(args.end_s)) or
            (args.start_s is not None and args.end_s is not None and args.end_s <= args.start_s)):
        parser.error("Require finite 0 <= start-s < end-s")
    try:
        report = analyze(args.log, args.start_s, args.end_s)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"output": str(args.output.resolve()),
                          "observed_arm_publishers": report["observed_arm_publishers"],
                          "position_delay_fit": report["position_delay_fit"]}, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: "+str(exc)+"\n")


if __name__ == "__main__":
    main()

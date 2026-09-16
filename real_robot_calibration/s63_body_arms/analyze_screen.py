"""Analyze slow single-joint screening logs; no ROS, SSH or robot commands."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics

from trajectory import NAMES, finite_vector


def analyze_log(path):
    targets, sensors, motors, status = [], [], [], []
    expected = {}
    end = 0.
    with Path(path).open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = row.get("kind")
            if kind == "local_attempt":
                expected = {p["label"]: p["duration_s"] for p in
                            (row.get("plan") or {}).get("phases", [])}
            stamp = row.get("receipt_monotonic_s")
            if stamp is None:
                continue
            end = max(end, stamp)
            if kind == "target":
                targets.append((stamp, row["phase"], row["position_rad"]))
            elif row.get("topic") in ("/sensors_data_raw", "/joint_cmd"):
                q = finite_vector(row["joint_q"], 20, "logged q")[:18]
                (sensors if row["topic"] == "/sensors_data_raw" else motors).append((stamp, q))
            elif kind in ("summary", "error"):
                status.append(row)
    targets.sort(); sensors.sort(); motors.sort()
    motor_times = [s for s, _ in motors]
    intervals = []
    for stamp, label, q in targets:
        if not intervals or intervals[-1]["label"] != label:
            if intervals:
                intervals[-1]["end"] = stamp
            intervals.append(dict(start=stamp, end=end, label=label, q=q))
    starts = [p["start"] for p in intervals]
    samples = [[] for _ in intervals]
    excluded, bracketed = 0, 0
    for stamp, q in sensors:
        phase_index = bisect.bisect_right(starts, stamp)-1
        command_index = bisect.bisect_right(motor_times, stamp)-1
        if phase_index < 0:
            continue
        if command_index < 0:
            excluded += 1
            continue
        cmd = motors[command_index][1]
        if stamp-motor_times[command_index] > .05:
            phase = intervals[phase_index]
            next_index = command_index+1
            quiet = phase["label"] == "baseline_hold" or phase["label"].endswith(("/hold", "/baseline"))
            # A constant command bracket is valid for static error only. Never
            # widen timing pairs for ramps or use this to estimate actuator lag.
            if (not quiet or next_index >= len(motors) or
                    motor_times[next_index]-motor_times[command_index] > .3 or
                    motor_times[command_index] < phase["start"] or
                    motor_times[next_index] >= phase["end"] or
                    max(abs(a-b) for a,b in zip(cmd,motors[next_index][1])) > math.radians(.001)):
                excluded += 1
                continue
            bracketed += 1
        samples[phase_index].append((stamp, q, cmd))
    holds, completed = {}, set()
    for phase, points in zip(intervals, samples):
        label = phase["label"]
        if label not in expected or not (label == "baseline_hold" or
                                        label.endswith(("/hold", "/baseline"))):
            continue
        # Do not turn an aborted final plateau into a completed protocol.
        if phase["end"]-phase["start"] < expected[label]-.15:
            continue
        completed.add(label)
        points = [p for p in points if phase["end"]-.75 <= p[0] < phase["end"]]
        if len(points) >= 20:
            holds[label] = points
    initial = holds.get("baseline_hold", [])
    entries = {}
    if initial:
        for name in NAMES:
            i = NAMES.index(name)
            positives = [p for label, ps in holds.items() if
                         label.startswith(name+"/") and "/+1/hold" in label for p in ps]
            negatives = [p for label, ps in holds.items() if
                         label.startswith(name+"/") and "/-1/hold" in label for p in ps]
            returns = [ps for label, ps in holds.items() if
                       label.startswith(name+"/") and "/-1/baseline" in label]
            if not any(label.startswith(name+"/") for label in starts_label(intervals)):
                continue
            bias = statistics.mean(q[i]-cmd[i] for _, q, cmd in initial)
            deviations = [q[i]-cmd[i] for _, q, cmd in initial]
            center = statistics.median(deviations)
            noise = 1.4826*statistics.median(abs(x-center) for x in deviations)
            trigger = max(.1, math.degrees(3*noise))
            required = {label for label in expected if label.startswith(name+"/") and
                        label.endswith(("/hold", "/baseline"))}
            entry = dict(joint=name, protocol_complete=bool(required) and required <= completed,
                         measurement_valid=bool(required) and required <= holds.keys(),
                         source=str(Path(path).resolve()), baseline_bias_deg=math.degrees(bias),
                         trigger_deg=trigger, positive_samples=len(positives),
                         negative_samples=len(negatives))
            reasons = []
            for direction, points in (("positive", positives), ("negative", negatives)):
                if points:
                    error = statistics.mean(q[i]-cmd[i] for _, q, cmd in points)
                    corrected = math.degrees(error-bias)
                    span = math.degrees(max(q[i] for _, q, _ in points)-min(q[i] for _, q, _ in points))
                    entry[direction+"_error_deg"] = math.degrees(error)
                    entry[direction+"_corrected_deg"] = corrected
                    entry[direction+"_span_deg"] = span
                    if abs(corrected) > trigger:
                        reasons.append(direction+" corrected error")
                    if span > max(.05, math.degrees(3*noise)):
                        reasons.append(direction+" plateau not quiet")
            if returns:
                change = statistics.mean(q[i] for _, q, _ in returns[-1])-statistics.mean(q[i] for _, q, _ in initial)
                entry["return_change_deg"] = math.degrees(change)
                if abs(entry["return_change_deg"]) > trigger:
                    reasons.append("return changed")
            if not entry["protocol_complete"]:
                reasons.append("incomplete protocol")
            if not entry["measurement_valid"]:
                reasons.append("insufficient paired samples")
            entry["followup_reasons"] = reasons
            entries[name] = entry
    return dict(source=str(Path(path).resolve()), status=status, entries=entries,
                excluded_pairs=excluded, stable_bracket_pairs=bracketed,
                comparison="receipt-paired q versus motor q; age <=50ms, or constant same-plateau command bracket <=300ms",
                window="last 0.75s of completed plateau; minimum 20 samples")


def starts_label(intervals):
    return (phase["label"] for phase in intervals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [analyze_log(path) for path in args.log]
    merged = {}
    for report in reports:
        for name, entry in report["entries"].items():
            if entry["protocol_complete"] or name not in merged:
                merged[name] = entry
    result = dict(logs=reports, joints=merged,
                  missing_arm_joints=[name for name in NAMES[4:] if
                                      not merged.get(name, {}).get("protocol_complete")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    fields = ["joint", "protocol_complete", "measurement_valid", "baseline_bias_deg", "positive_error_deg",
              "negative_error_deg", "positive_corrected_deg", "negative_corrected_deg",
              "return_change_deg", "followup_reasons"]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged[name] for name in NAMES if name in merged)
    print(json.dumps(dict(output=str(args.output.resolve()),
                          missing_arm_joints=result["missing_arm_joints"],
                          followup={n:e["followup_reasons"] for n,e in merged.items()
                                    if e["followup_reasons"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()

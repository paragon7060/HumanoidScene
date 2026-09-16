#!/usr/bin/env python3
"""Convert real /joint_cmd and measured state to a validated S63 sim replay."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import statistics

from trajectory import NAMES, finite_vector


def _rows(path):
    path = Path(path).resolve()
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    metadata = next((r for r in rows if r.get("kind") == "metadata"), None)
    summary = [r for r in rows if r.get("kind") == "summary"]
    if metadata is None or metadata.get("robot_version") != 63:
        raise ValueError("Expected an S63 real-robot record")
    config = metadata.get("configuration", {})
    if config.get("NUM_JOINT") != 20 or config.get("NUM_ARM_JOINT") != 14:
        raise ValueError("Expected the verified S63 20-joint ordering")
    if not summary or summary[-1].get("ok") is not True:
        raise ValueError("Source record did not close with summary ok=true")
    return path, rows, metadata


def _series(rows, topic, origin, start_s, end_s):
    result = []
    for row in rows:
        if row.get("topic") != topic:
            continue
        receipt = float(row["receipt_monotonic_s"])-origin
        if not start_s <= receipt <= end_s:
            continue
        q = finite_vector(row["joint_q"][:18], 18, topic+" q")
        v = finite_vector(row["joint_v"][:18], 18, topic+" v")
        stamp = float(row["header_s"])
        if stamp > 0:
            result.append((stamp, q, v))
    result.sort()
    if len(result) < 50:
        raise ValueError(f"Too few {topic} samples in selected window")
    if any(b[0] <= a[0] or b[0]-a[0] > .1 for a, b in zip(result, result[1:])):
        raise ValueError(f"Non-monotonic or >100ms gap in {topic}")
    return result


def _command_series(rows, origin, start_s, end_s):
    result = []
    continuous = ("joint_q", "joint_v", "tau", "joint_kp", "joint_kd", "tau_max", "tau_ratio")
    for row in rows:
        if row.get("topic") != "/joint_cmd":
            continue
        receipt = float(row["receipt_monotonic_s"])-origin
        if not start_s <= receipt <= end_s:
            continue
        values = {key: finite_vector(row[key][:18], 18, "/joint_cmd "+key) for key in continuous}
        modes = list(row["control_modes"][:18])
        if len(modes) != 18 or any(isinstance(x, bool) or not isinstance(x, int) for x in modes):
            raise ValueError("Invalid /joint_cmd control_modes")
        stamp = float(row["header_s"])
        if stamp > 0:
            result.append((stamp, values, modes))
    result.sort()
    if len(result) < 50:
        raise ValueError("Too few /joint_cmd samples in selected window")
    if any(b[0] <= a[0] or b[0]-a[0] > .1 for a, b in zip(result, result[1:])):
        raise ValueError("Non-monotonic or >100ms gap in /joint_cmd")
    return result


def _interpolate(series, stamp):
    times = [r[0] for r in series]
    index = bisect.bisect_left(times, stamp)
    if index == 0:
        return series[0][1], series[0][2]
    if index == len(series):
        return series[-1][1], series[-1][2]
    before, after = series[index-1], series[index]
    weight = (stamp-before[0])/(after[0]-before[0])
    return ([a+weight*(b-a) for a, b in zip(before[1], after[1])],
            [a+weight*(b-a) for a, b in zip(before[2], after[2])])


def _interpolate_command(series, stamp):
    times = [r[0] for r in series]
    index = bisect.bisect_left(times, stamp)
    if index == 0:
        chosen = series[0]
        return dict(chosen[1]), list(chosen[2])
    if index == len(series):
        chosen = series[-1]
        return dict(chosen[1]), list(chosen[2])
    before, after = series[index-1], series[index]
    weight = (stamp-before[0])/(after[0]-before[0])
    values = {key: [a+weight*(b-a) for a, b in zip(before[1][key], after[1][key])]
              for key in before[1]}
    return values, list(before[2] if weight < .5 else after[2])


def export(log, start_s, end_s, rate_hz=30.0, hold_s=1.0):
    if not all(math.isfinite(x) for x in (start_s, end_s, rate_hz, hold_s)) or not 0 <= start_s < end_s:
        raise ValueError("Require finite 0 <= start < end")
    if not 10 <= rate_hz <= 100 or not .5 <= hold_s <= 5:
        raise ValueError("rate-hz must be [10,100] and hold-s [0.5,5]")
    path, rows, metadata = _rows(log)
    origin = float(metadata["receipt_monotonic_s"])
    command = _command_series(rows, origin, start_s, end_s)
    measured = _series(rows, "/sensors_data_raw", origin, start_s, end_s)
    first = max(command[0][0], measured[0][0])
    last = min(command[-1][0], measured[-1][0])
    if last-first < 3:
        raise ValueError("Selected command/state overlap is shorter than 3 seconds")
    count = math.floor((last-first)*rate_hz)+1
    samples = []
    for index in range(count):
        elapsed = index/rate_hz
        stamp = first+elapsed
        target, modes = _interpolate_command(command, stamp)
        real_q, real_v = _interpolate(measured, stamp)
        samples.append({"source_elapsed_s": elapsed,
                        "target_q_rad": target["joint_q"],
                        "target_v_rad_s": target["joint_v"],
                        "target_tau": target["tau"],
                        "target_kp": target["joint_kp"],
                        "target_kd": target["joint_kd"],
                        "target_tau_max_raw": target["tau_max"],
                        "target_tau_ratio": target["tau_ratio"],
                        "control_modes": modes,
                        "real_q_rad": real_q, "real_v_rad_s": real_v})
    limits = metadata["configuration"]
    low = [math.radians(x) for x in limits["min_joint_position_limits"][:18]]
    high = [math.radians(x) for x in limits["max_joint_position_limits"][:18]]
    for sample_row in samples:
        if any(not lo <= q <= hi for q, lo, hi in zip(sample_row["target_q_rad"], low, high)):
            raise ValueError("Selected motor target violates reported real joint limits")
    return {"schema": "kuavo_real_vr_replay_v2", "robot_model": "s63",
            "gripper": "leju-twofinger", "joint_names": NAMES,
            "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_receipt_window_s": {"start": start_s, "end": end_s},
            "source_header_window_s": {"start": first, "end": last},
            "sample_rate_hz": rate_hz, "hold_s": hold_s,
            "motion_duration_s": samples[-1]["source_elapsed_s"],
            "duration_s": 2*hold_s+samples[-1]["source_elapsed_s"],
            "units": "rad, rad/s, requested torque field, seconds", "samples": samples,
            "semantics": "Real /joint_cmd q/v/tau/gain/mode replay; real_q/v are time-aligned /sensors_data_raw references",
            "torque_note": "tau is the recorded WBC request; unit/driver summation hypotheses require diagnostic replay. tau_max_raw is hardware max_current, not an Nm cap.",
            "real_commands_sent": False}


def validate(plan):
    if plan.get("schema") not in ("kuavo_real_vr_replay_v1", "kuavo_real_vr_replay_v2") or plan.get("robot_model") != "s63":
        raise ValueError("Expected an S63 real-VR replay")
    if plan.get("gripper") != "leju-twofinger" or plan.get("joint_names") != NAMES:
        raise ValueError("Replay is pinned to S63 + leju-twofinger and verified joint ordering")
    source = Path(plan["source"])
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != plan.get("source_sha256"):
        raise ValueError("Replay source is missing or changed")
    rate = float(plan["sample_rate_hz"])
    samples = plan.get("samples", [])
    if len(samples) < 30 or not 10 <= rate <= 100:
        raise ValueError("Insufficient or invalid replay samples")
    previous = -math.inf
    for row in samples:
        elapsed = float(row["source_elapsed_s"])
        if elapsed <= previous:
            raise ValueError("Replay sample times must increase")
        previous = elapsed
        for key in ("target_q_rad", "target_v_rad_s", "real_q_rad", "real_v_rad_s"):
            finite_vector(row[key], 18, key)
        if plan["schema"] == "kuavo_real_vr_replay_v2":
            for key in ("target_tau", "target_kp", "target_kd", "target_tau_max_raw", "target_tau_ratio"):
                finite_vector(row[key], 18, key)
            modes = row.get("control_modes", [])
            if len(modes) != 18 or any(isinstance(x, bool) or not isinstance(x, int) for x in modes):
                raise ValueError("Invalid replay control_modes")
    expected = 2*float(plan["hold_s"])+samples[-1]["source_elapsed_s"]
    if abs(float(plan["duration_s"])-expected) > 1e-9:
        raise ValueError("Replay duration mismatch")
    return plan


def sample(plan, elapsed):
    plan = validate(plan) if plan.get("_validated") is not True else plan
    hold = float(plan["hold_s"])
    rows = plan["samples"]
    if elapsed < hold:
        row = rows[0]
        return row["target_q_rad"], [0.0]*18, row["real_q_rad"], row["real_v_rad_s"], 0.0, "initial_hold"
    source_t = elapsed-hold
    if source_t >= rows[-1]["source_elapsed_s"]:
        row = rows[-1]
        return row["target_q_rad"], [0.0]*18, row["real_q_rad"], row["real_v_rad_s"], rows[-1]["source_elapsed_s"], "final_hold"
    index = min(len(rows)-1, max(0, round(source_t*float(plan["sample_rate_hz"]))))
    row = rows[index]
    return row["target_q_rad"], row["target_v_rad_s"], row["real_q_rad"], row["real_v_rad_s"], row["source_elapsed_s"], "source"


def sample_command(plan, elapsed):
    """Return the full v2 command while preserving sample() compatibility."""
    plan = validate(plan) if plan.get("_validated") is not True else plan
    if plan.get("schema") != "kuavo_real_vr_replay_v2":
        q, v, real_q, real_v, source_t, phase = sample(plan, elapsed)
        return {"target_q_rad":q,"target_v_rad_s":v,"real_q_rad":real_q,
                "real_v_rad_s":real_v,"target_tau":[0.0]*18,
                "target_kp":[0.0]*18,"target_kd":[0.0]*18,
                "target_tau_max_raw":[0.0]*18,"target_tau_ratio":[1.0]*18,
                "control_modes":[2]*18,"source_elapsed_s":source_t,"phase":phase}
    hold = float(plan["hold_s"])
    rows = plan["samples"]
    if elapsed < hold:
        row, source_t, phase = rows[0], 0.0, "initial_hold"
    elif elapsed-hold >= rows[-1]["source_elapsed_s"]:
        row, source_t, phase = rows[-1], rows[-1]["source_elapsed_s"], "final_hold"
    else:
        source_t = elapsed-hold
        index = min(len(rows)-1, max(0, round(source_t*float(plan["sample_rate_hz"]))))
        row, phase = rows[index], "source"
    result = dict(row)
    if phase != "source":
        result["target_v_rad_s"] = [0.0]*18
    result.update(source_elapsed_s=source_t, phase=phase)
    return result


def compare(response):
    rows = [json.loads(line) for line in Path(response).open()]
    metadata = rows[0]
    sim_names = metadata.get("joint_names", [])
    if (metadata.get("schema") != "kuavo_joint_response_v1" or len(sim_names) != 18 or
            set(sim_names) != set(NAMES)):
        raise ValueError("Unexpected sim response joints")
    active = [r for r in rows if r.get("kind") == "step" and
              r.get("context", {}).get("phase") == "source" and not r.get("reset_after_step")]
    if len(active) < 30 or rows[-1].get("kind") != "closed" or rows[-1].get("ok") is not True:
        raise ValueError("Incomplete sim response")
    result = {}
    times = [float(r["context"]["source_elapsed_s"]) for r in active]

    def reference(real_index, stamp, field):
        index = bisect.bisect_left(times, stamp)
        if index == 0 or index == len(times):
            return None
        a, b = active[index-1], active[index]
        weight = (stamp-times[index-1])/(times[index]-times[index-1])
        av = a["context"][field][real_index]; bv = b["context"][field][real_index]
        return av+weight*(bv-av)

    def aligned_score(sim_index, real_index, delay, sim_field, real_field):
        errors = []
        for row, stamp in zip(active, times):
            expected = reference(real_index, stamp-delay, real_field)
            if expected is not None:
                errors.append((row[sim_field][sim_index]-expected)**2)
        return math.sqrt(statistics.mean(errors)) if len(errors) >= 30 else math.inf

    def best_alignment(sim_index, real_index):
        values = [r["context"]["real_q_rad"][real_index] for r in active]
        if math.degrees(max(values)-min(values)) < .5:
            return None
        coarse = [x/1000 for x in range(-500, 501, 5)]
        first = min(coarse, key=lambda d: aligned_score(sim_index, real_index, d, "q_after_rad", "real_q_rad"))
        fine = [x/1000 for x in range(round(first*1000)-5, round(first*1000)+6)]
        delay = min(fine, key=lambda d: aligned_score(sim_index, real_index, d, "q_after_rad", "real_q_rad"))
        return {"delay_ms": round(1000*delay),
                "q_rmse_deg": math.degrees(aligned_score(
                    sim_index, real_index, delay, "q_after_rad", "real_q_rad"))}

    for real_index, name in enumerate(NAMES):
        sim_index = sim_names.index(name)
        q_error = [math.degrees(r["q_after_rad"][sim_index]-r["context"]["real_q_rad"][real_index]) for r in active]
        v_error = [math.degrees(r["v_after_rad_s"][sim_index]-r["context"]["real_v_rad_s"][real_index]) for r in active]
        result[name] = {"q_rmse_deg": math.sqrt(statistics.mean(x*x for x in q_error)),
                        "q_mean_deg": statistics.mean(q_error),
                        "q_max_abs_deg": max(abs(x) for x in q_error),
                        "v_rmse_deg_s": math.sqrt(statistics.mean(x*x for x in v_error)),
                        "best_time_alignment": best_alignment(sim_index, real_index)}
    return {"source": str(Path(response).resolve()), "samples": len(active),
            "sim_minus_real": result,
            "delay_sign": "positive means sim response best matches an earlier real state (sim is slower)",
            "note": "Same real motor q/v target and source clock; position/velocity only, torque units excluded"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("export")
    out.add_argument("--log", type=Path, required=True); out.add_argument("--start-s", type=float, required=True)
    out.add_argument("--end-s", type=float, required=True); out.add_argument("--rate-hz", type=float, default=30.)
    out.add_argument("--hold-s", type=float, default=1.); out.add_argument("--output", type=Path, required=True)
    cmp = commands.add_parser("compare"); cmp.add_argument("--response", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output file")
    try:
        payload = (export(args.log, args.start_s, args.end_s, args.rate_hz, args.hold_s)
                   if args.command == "export" else compare(args.response))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"output": str(args.output.resolve()),
                          "duration_s": payload.get("duration_s"), "samples": len(payload.get("samples", []))
                          if isinstance(payload.get("samples"), list) else payload.get("samples")}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: "+str(exc)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded S63 calibration targets. Standard library only; no robot I/O."""
import math
import statistics

NAMES = ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"] + [
    f"zarm_{side}{i}_joint" for side in ("l", "r") for i in range(1, 8)]
BASELINE_LIMIT_TOLERANCE = math.radians(.01)  # encoder noise at a resting limit


def finite_vector(values, size, label):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{label}: expected {size} values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values):
        raise ValueError(f"{label}: finite numbers required")
    return list(values)


def stationarity_metrics(history, now):
    """Two seconds of fresh, quiet positions AND low signed median velocity.

    Raw motor velocity can have stationary quantization/noise spikes. Position
    range prevents opposite velocities from hiding an actual oscillation.
    """
    if not history:
        raise ValueError("stationary check needs recent position/velocity history")
    latest_stamp = history[-1][0]
    if not math.isfinite(now) or not math.isfinite(latest_stamp) or not 0 <= now-latest_stamp <= .1:
        raise ValueError("stationary check: stale/future sensor history")
    # A fresh callback may precede this check by tens of milliseconds. Anchor
    # the observed window to that callback, preserving the full 2 s history.
    recent = [(stamp, q, v) for stamp, q, v in history if latest_stamp-2. <= stamp <= latest_stamp]
    if len(recent) < 40 or recent[-1][0]-recent[0][0] < 1.95:
        raise ValueError("stationary check needs >=1.95 s of recent position/velocity history")
    times = [r[0] for r in recent]
    if any(not 0 < b-a <= .1 for a,b in zip(times,times[1:])):
        raise ValueError("stationary check: stale/gapped sensor history")
    positions = [finite_vector(q,18,"stationary position") for _,q,_ in recent]
    velocities = [finite_vector(v,18,"stationary velocity") for _,_,v in recent]
    spans = [math.degrees(max(q[i] for q in positions)-min(q[i] for q in positions)) for i in range(18)]
    medians = [math.degrees(statistics.median(v[i] for v in velocities)) for i in range(18)]
    peaks = [math.degrees(max(abs(v[i]) for v in velocities)) for i in range(18)]
    for i,name in enumerate(NAMES):
        if spans[i] > .05 or abs(medians[i]) > .5 or peaks[i] > 10.:
            raise ValueError(f"Robot not stationary: {name}, position_span={spans[i]:.4f} deg "
                             f"(limit 0.05), median_velocity={medians[i]:.3f} deg/s "
                             f"(limit 0.5), raw_peak={peaks[i]:.3f} deg/s (limit 10)")
    return {"window_s":times[-1]-times[0], "samples":len(recent),
            "position_span_deg":dict(zip(NAMES,spans)),
            "median_velocity_deg_s":dict(zip(NAMES,medians)),
            "raw_peak_velocity_deg_s":dict(zip(NAMES,peaks))}


def make_plan(capture, limits, kind="hold", joints=(), amplitude=.5, ramp=2., hold=5., cycles=3):
    """Targets are radians; velocities rad/s; one joint at a time, always return."""
    if capture.get("robot_version") != 63 or capture.get("joint_names") != NAMES:
        raise ValueError("Only the verified S63 4-body + 14-arm order is supported")
    baseline = finite_vector(capture.get("baseline_rad"), 18, "baseline_rad")
    if kind not in ("hold", "static", "motion"):
        raise ValueError("Unknown sequence")
    if any(not math.isfinite(v) for v in (amplitude, ramp, hold)):
        raise ValueError("Finite trajectory parameters required")
    if not 0 < amplitude <= 1 or not 2 <= ramp <= 10 or not 1 <= hold <= 30:
        raise ValueError("amplitude: (0,1] deg; ramp: [2,10] s; hold: [1,30] s")
    if not isinstance(cycles, int) or not 1 <= cycles <= 5:
        raise ValueError("cycles: integer [1,5]")
    if kind != "hold" and not joints:
        raise ValueError("Choose joints explicitly; there is no default moving joint")
    if kind == "hold" and joints:
        raise ValueError("hold keeps every joint at baseline; omit --joint")
    if len(set(joints)) != len(joints) or any(j not in NAMES for j in joints):
        raise ValueError("Unknown or duplicate joint")
    if any(j in NAMES[:4] for j in joints) and amplitude > .5:
        raise ValueError("Body joint amplitude is limited to +/-0.5 deg")
    low = finite_vector([limits[n][0] for n in NAMES], 18, "lower limits")
    high = finite_vector([limits[n][1] for n in NAMES], 18, "upper limits")
    if any(a >= b for a, b in zip(low, high)):
        raise ValueError("Invalid limits")
    if any(not a-BASELINE_LIMIT_TOLERANCE <= q <= b+BASELINE_LIMIT_TOLERANCE
           for q, a, b in zip(baseline, low, high)):
        raise ValueError("Baseline is outside URDF limits; no clipping or zero fallback")
    phases = [{"label": "baseline_hold", "duration_s": hold, "from_rad": baseline,
               "to_rad": baseline}]
    current = baseline
    for name in (() if kind == "hold" else joints):
        index = NAMES.index(name)
        for cycle in range(cycles if kind == "motion" else 1):
            for sign in (1, -1):
                target = baseline.copy()
                target[index] += math.radians(sign * amplitude)
                # Leave 0.5 deg margin for each displaced joint.
                margin = math.radians(.5)
                if not low[index] + margin <= target[index] <= high[index] - margin:
                    raise ValueError(f"{name}: +/- amplitude exceeds limits/margin; choose smaller amplitude")
                label = f"{name}/{cycle+1}/{sign:+d}"
                phases.append({"label": label + "/ramp", "duration_s": ramp,
                               "from_rad": current, "to_rad": target})
                phases.append({"label": label + "/hold", "duration_s": hold,
                               "from_rad": target, "to_rad": target})
                phases.append({"label": label + "/return", "duration_s": ramp,
                               "from_rad": target, "to_rad": baseline})
                phases.append({"label": label + "/baseline", "duration_s": hold,
                               "from_rad": baseline, "to_rad": baseline})
                current = baseline
    duration = sum(p["duration_s"] for p in phases)
    if duration > 900:
        raise ValueError("Sequence exceeds 900 s; split into separate joint tests")
    return {"schema": 1, "robot_version": 63, "joint_names": NAMES, "kind": kind,
            "baseline_rad": baseline, "lower_rad": low, "upper_rad": high,
            "amplitude_deg": amplitude, "ramp_s": ramp, "hold_s": hold, "cycles": cycles,
            "joints": list(joints), "duration_s": duration,
            "max_velocity_deg_s": 1.875 * amplitude / ramp if kind != "hold" else 0.,
            "max_acceleration_deg_s2": (10 / math.sqrt(3)) * amplitude / ramp**2 if kind != "hold" else 0.,
            "capture": capture, "phases": phases}


def sample(plan, elapsed):
    """Absolute elapsed time, never integrate a per-tick increment."""
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid elapsed time")
    for phase in plan["phases"]:
        duration = phase["duration_s"]
        if elapsed <= duration:
            s = max(0., min(1., elapsed / duration))
            blend = 10*s**3 - 15*s**4 + 6*s**5
            rate = (30*s**2 - 60*s**3 + 30*s**4) / duration
            q = [a + (b-a)*blend for a, b in zip(phase["from_rad"], phase["to_rad"])]
            v = [(b-a)*rate for a, b in zip(phase["from_rad"], phase["to_rad"])]
            return q, v, phase["label"]
        elapsed -= duration
    return plan["baseline_rad"].copy(), [0.]*18, "complete"


def validate_plan(plan):
    """Reconstruct on receipt so edited/untrusted JSON cannot bypass limits."""
    if plan.get("schema") != 1:
        raise ValueError("Unknown plan schema")
    limits = dict(zip(NAMES, zip(finite_vector(plan.get("lower_rad"), 18, "lower"),
                                finite_vector(plan.get("upper_rad"), 18, "upper"))))
    rebuilt = make_plan(plan["capture"], limits, plan["kind"], plan["joints"],
                        plan["amplitude_deg"], plan["ramp_s"], plan["hold_s"], plan["cycles"])
    if rebuilt != plan:
        raise ValueError("Plan differs from the bounded generated sequence")
    return rebuilt

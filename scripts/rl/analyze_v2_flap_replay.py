#!/usr/bin/env python3
"""Plot actual flap deflection from pre-reset physical replay telemetry.

This reads measurements; it neither evaluates a policy nor changes success.
Use the Isaac conda environment for NumPy/matplotlib. Telemetry must contain
history entries with synchronized actual_flap_poses/privileged_box_pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def analyze(path: Path, size_m, flap_length_m, hz):
    data = json.loads(path.read_text())
    neutral_z = 1.005 * size_m[2] + flap_length_m / 2
    neutral = np.array([[size_m[0] / 2.02, 0, neutral_z],
                        [-size_m[0] / 2.02, 0, neutral_z]])
    rows = []
    for sample in data["history"]:
        box = np.asarray(sample["privileged_box_pose"], dtype=float)
        box_rotation = rotation_wxyz(box[3:])
        flaps = np.asarray(sample["actual_flap_poses"], dtype=float)
        local = np.asarray(sample["flap_local_centers"], dtype=float)
        displacement, tilt = [], []
        for i in range(2):
            flap_rotation = rotation_wxyz(flaps[i, 3:])
            center_world = flaps[i, :3] + flap_rotation @ local[i]
            center_box = box_rotation.T @ (center_world - box[:3])
            displacement.append(float(np.linalg.norm(center_box - neutral[i])))
            # The panel normal is +/- X for these stock flap wrappers.
            normal_dot = abs(float((box_rotation.T @ flap_rotation)[0, 0]))
            tilt.append(float(np.degrees(np.arccos(np.clip(normal_dot, 0, 1)))))
        rows.append({"step": sample["step"], "seconds": sample["step"] / hz,
                     "center_displacement_m": displacement, "normal_tilt_deg": tilt,
                     "pinching": sample["pinching"],
                     "unsafe": sample["unsafe"], "success": sample["success"]})
    if not rows:
        raise ValueError("Replay contains no physical history")
    pinch_steps = [sum(row["pinching"][i] for row in rows) for i in range(2)]
    project = Path(__file__).resolve().parents[2]
    source = str(path.relative_to(project)) if path.is_relative_to(project) else path.name
    summary = {"source": source, "policy": data.get("policy"),
               "steps": data["steps"], "outcomes": data["outcomes"],
               "pinch_steps_left_right": pinch_steps,
               "bilateral_pinch_steps": sum(all(row["pinching"]) for row in rows),
               "max_displacement_m": np.max([r["center_displacement_m"] for r in rows], axis=0).tolist(),
               "max_normal_tilt_deg": np.max([r["normal_tilt_deg"] for r in rows], axis=0).tolist()}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--box-size-m", nargs=3, type=float, required=True)
    parser.add_argument("--flap-length-m", type=float, required=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if len(args.replay) != len(args.label):
        parser.error("Each replay requires one label")
    if min(*args.box_size_m, args.flap_length_m, args.control_hz) <= 0:
        parser.error("Geometry and control frequency must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    summaries = []
    for index, (path, label) in enumerate(zip(args.replay, args.label)):
        rows, summary = analyze(path.resolve(), args.box_size_m,
                                args.flap_length_m, args.control_hz)
        summary["label"] = label
        summaries.append(summary)
        time = [r["seconds"] for r in rows]
        color = f"C{index}"
        for flap, style in ((0, "-"), (1, "--")):
            axes[0].plot(time, [100*r["center_displacement_m"][flap] for r in rows],
                         style, color=color, label=f"{label}: flap {flap}")
            axes[1].plot(time, [r["normal_tilt_deg"][flap] for r in rows],
                         style, color=color)
        for hand, style in ((0, "-"), (1, "--")):
            axes[2].step(time, [int(r["pinching"][hand]) + index*1.3 for r in rows],
                         where="post", linestyle=style, color=color)
        if summary["outcomes"]["unsafe"]:
            axes[0].axvline(time[-1], color=color, alpha=.5)
            axes[0].annotate("rack failure", (time[-1], 0), xytext=(-80, 12),
                             textcoords="offset points", color=color)
    axes[0].set_ylabel("Actual vs nominal center (cm)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Actual vs nominal normal (deg)")
    axes[2].set_ylabel("Measured pinch (offset per run)")
    axes[2].set_xlabel("Control time (s)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Upper-shelf physical replay: flap deformation and contact\n"
                 "Diagnostic references, not learned SAC successes")
    fig.tight_layout()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=150)
    prefix.with_suffix(".json").write_text(json.dumps({
        "box_size_m": args.box_size_m, "flap_length_m": args.flap_length_m,
        "control_hz": args.control_hz, "summaries": summaries,
        "measurement": "Actual flap and box poses sampled together before reset; nominal is upright asset geometry",
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

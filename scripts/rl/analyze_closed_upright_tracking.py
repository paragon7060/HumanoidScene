"""Read-only torso goal/drive tracking diagnosis from a stopped SAC run.

Replay rows are actual held-phase transitions, not complete episode outcomes.
The privileged collision fields are used only for offline diagnosis. This tool
never imports observations into training or reads a writer's active payload.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/rl")]
from compare_actor_train_memory import owned_stable_bytes
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import (
    planar_position, torso_links_from_urdf,
)


def distribution(value):
    value = value.flatten().to(torch.float64)
    assert len(value) and torch.isfinite(value).all()
    levels = torch.tensor([0., .5, .9, .95, .99, 1.], dtype=value.dtype)
    return {name: float(x) for name, x in zip(
        ("min", "median", "p90", "p95", "p99", "max"),
        torch.quantile(value, levels))}


def torso_tracking_summary(rows, contract):
    """Compare saved executed goals, logical joint targets and measured physics."""
    n = len(rows["action"])
    assert n and rows["critic_obs"].shape == (n, 578)
    assert rows["next_critic_obs"].shape == (n, 578)
    assert rows["actor_obs"].shape == (n, 518)
    assert rows["action"].shape == (n, 21)
    assert all(torch.isfinite(x).all() for x in rows.values())
    raw = rows["critic_obs"][:, :464]
    nxt = rows["next_critic_obs"][:, :464]
    assert (raw[:, 439] == 1).all() and (nxt[:, 439] == 1).all()
    # robot_state_from_sensors stores logical target - measured joint angle;
    # the simulated gravity drive bias is intentionally absent from this field.
    pending = raw[:, :20] + raw[:, 416:436]
    next_pending = nxt[:, :20] + nxt[:, 416:436]
    links = raw.new_tensor(torso_links_from_urdf(
        resolve_robot_model("s63", "leju-twofinger").urdf_path))
    measured_xz = planar_position(raw[:, :2], links)
    next_measured_xz = planar_position(nxt[:, :2], links)
    pending_xz = planar_position(pending[:, :2], links)
    next_pending_xz = planar_position(next_pending[:, :2], links)
    goals = raw.new_tensor(contract["goal_center"]) + \
        raw.new_tensor(contract["goal_scale"]) * rows["action"]
    support = contract["URDF_upright_support"]
    assert support["upright_goal_columns"] == [17, 18]
    # Reconstruct the declared support directly from the saved model contract.
    from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import upright_support_affine
    center, scale = [raw.new_tensor(contract[x]) for x in ("goal_center", "goal_scale")]
    anchor, radius = upright_support_affine(center, scale, links)
    lower, upper = [center[17:19] + scale[17:19] * x for x in (anchor-radius, anchor+radius)]
    in_support = lambda x: ((x >= lower-1e-6) & (x <= upper+1e-6)).all(-1)
    assert in_support(goals[:, 17:19]).all(), "Executed goals left the declared support"
    degrees = 180 / torch.pi
    pitch_error = (raw[:, :3] - pending[:, :3]).sum(-1) * degrees
    next_pitch_error = (nxt[:, :3] - next_pending[:, :3]).sum(-1) * degrees
    # v2 privileged contract: last 7 fields are safety-reason booleans;
    # first of those is robot-rack collision. No motor effort is recorded here.
    rack = rows["critic_obs"][:, 464+59] > .5
    next_rack = rows["next_critic_obs"][:, 464+59] > .5
    assert ((rows["critic_obs"][:, 464+59:530] == 0) |
            (rows["critic_obs"][:, 464+59:530] == 1)).all()
    worst = int(next_pitch_error.abs().argmax())
    example = dict(row=worst, held_control_step=int(round(float(rows["critic_obs"][worst, 530])*900)),
        next_measured_joint_q_rad=nxt[worst, :3].tolist(),
        next_logical_joint_target_q_rad=next_pending[worst, :3].tolist(),
        executed_goal_XZ_m=goals[worst, 17:19].tolist(),
        next_measured_XZ_m=next_measured_xz[worst].tolist(),
        next_logical_target_XZ_m=next_pending_xz[worst].tolist(),
        measured_minus_commanded_pitch_deg=float(next_pitch_error[worst]),
        next_robot_rack_collision=bool(next_rack[worst]))
    proof = dict(recorded_at=datetime.now().astimezone().isoformat(),
        actual_held_transition_rows=n,
        partial_transitions_NOT_complete_episode_success_statistics=True,
        all_executed_torso_goals_inside_declared_support=True,
        unchanged_goal_support_XZ_m=dict(lower=lower.tolist(), upper=upper.tolist()),
        actual_measured_torso_XZ_range_m=dict(min=measured_xz.min(0).values.tolist(), max=measured_xz.max(0).values.tolist()),
        actual_logical_target_XZ_range_m=dict(min=pending_xz.min(0).values.tolist(), max=pending_xz.max(0).values.tolist()),
        pre_measured_outside_support_rows=int((~in_support(measured_xz)).sum()),
        next_measured_outside_support_rows=int((~in_support(next_measured_xz)).sum()),
        pre_logical_target_outside_support_rows=int((~in_support(pending_xz)).sum()),
        measured_minus_logical_target_XZ_absolute_error_mm={
            axis: distribution((measured_xz[:, i]-pending_xz[:, i]).abs()*1000)
            for i, axis in enumerate(("X", "Z"))},
        measured_minus_logical_target_pitch_absolute_error_deg=distribution(pitch_error.abs()),
        logical_pitch_target_deg=distribution(pending[:, :3].sum(-1)*degrees),
        measured_pitch_deg=distribution(raw[:, :3].sum(-1)*degrees),
        rows_above_3deg_pitch_tracking_error=int((pitch_error.abs()>3).sum()),
        rows_above_5deg_pitch_tracking_error=int((pitch_error.abs()>5).sum()),
        robot_rack_collision_rows=int(rack.sum()),
        above_3deg_pitch_error_and_robot_rack_collision_rows=int(((pitch_error.abs()>3)&rack).sum()),
        worst_next_pitch_tracking_example=example,
        applied_motor_torque_and_effort_saturation_NOT_available=True,
        actual_gravity_feedforward_NOT_directly_measured_by_this_replay=True,
        collision_correlation_NOT_causal_or_payload_load_proof=True,
        policy_goal_and_pending_target_are_distinct_from_measured_physics=True,
        no_training_data_import_or_controller_change=True, no_live_payload_reads=True,
        no_process_signals=True, goal_not_complete=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only diagnosis required"
    torch.set_num_threads(1)
    run = args.run_dir.resolve()
    managed = json.loads((run.parent / "launch.json").read_text())
    status = json.loads((run.parent / "status.json").read_text())
    assert Path(managed["run"]).resolve() == run
    for key in ("training_pid", "supervisor_pid"):
        assert not Path("/proc", str(status[key])).exists(), f"{key} is still alive"
    assert status.get("training_exit_code") is not None
    path = run / "staged_goal_experience.pt"
    before = path.stat()
    blob = owned_stable_bytes(path)
    saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    digest = hashlib.sha256(blob).hexdigest()
    del blob
    rows = saved["executed_goal_transitions"]
    contract = saved["goal_contract"]
    proof = torso_tracking_summary(rows, contract)
    proof.update(source_run_directory_name=run.name, source_replay_SHA256=digest,
        source_replay_bytes=before.st_size,
        actual_training_exit_code=status["training_exit_code"],
        source_writer_and_supervisor_stopped_verified=True)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    proof["source_replay_stat_unchanged"] = True
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    if args.output_plot:
        raw = rows["critic_obs"][:, :464]
        pending = raw[:, :20] + raw[:, 416:436]
        links = raw.new_tensor(torso_links_from_urdf(
            resolve_robot_model("s63", "leju-twofinger").urdf_path))
        measured_xz = planar_position(raw[:, :2], links)
        pending_xz = planar_position(pending[:, :2], links)
        n, degrees = len(raw), 180/torch.pi
        example = proof["worst_next_pitch_tracking_example"]
        lower = raw.new_tensor(proof["unchanged_goal_support_XZ_m"]["lower"])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        axes[0].hist((raw[:, :3].sum(-1)*degrees).numpy(), bins=60, alpha=.7,
                     color="#ee9977", label="Measured")
        axes[0].axvline(proof["logical_pitch_target_deg"]["median"],
                        color="#228833", linestyle="--", label="Logical target (median)")
        axes[0].set(xlabel="Torso pitch (degrees)", ylabel="Held replay rows", title="Pitch target versus physical state")
        axes[0].legend(fontsize=8)
        for i, label in enumerate(("X", "Z")):
            error = (measured_xz[:, i]-pending_xz[:, i]).abs()*1000
            error = error.sort().values
            axes[1].plot(error.numpy(), torch.linspace(0, 1, n).numpy(), label=label)
        axes[1].set(xlabel="Absolute position tracking error (mm)", ylabel="Cumulative row fraction", title="Torso tracking error")
        axes[1].legend()
        axes[2].bar(["Policy goal", "Logical target", "Measured"],
            [example["executed_goal_XZ_m"][0]*1000, example["next_logical_target_XZ_m"][0]*1000,
             example["next_measured_XZ_m"][0]*1000], color=["#4477aa", "#228833", "#ee6677"])
        axes[2].axhline(float(lower[0])*1000, color="grey", linestyle="--", label="Goal lower bound")
        axes[2].set(ylabel="Torso X (mm)", title="Largest observed pitch tracking error")
        axes[2].tick_params(axis="x", rotation=18)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"Stopped v5 run: {n:,} actual partial TRAIN rows; no success-rate inference", fontsize=11)
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_plot, dpi=160)
        plt.close(fig)
    print(json.dumps({k: proof[k] for k in (
        "actual_held_transition_rows", "measured_minus_logical_target_XZ_absolute_error_mm",
        "measured_minus_logical_target_pitch_absolute_error_deg", "logical_pitch_target_deg",
        "rows_above_3deg_pitch_tracking_error", "robot_rack_collision_rows",
        "worst_next_pitch_tracking_example")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

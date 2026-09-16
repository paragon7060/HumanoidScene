"""Benchmark PhysX dynamics terms used by lightweight real-like control."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

from kuavo_isaaclab_scene.robots.gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
)
from kuavo_isaaclab_scene.robots.robot_model import (
    add_robot_model_cli_args,
    export_robot_model_cli,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.repeats) < 1:
        parser.error("num-envs, iterations and repeats must be positive")
    if args.output.exists():
        parser.error("Choose a new output filename")
    export_robot_model_cli(args)
    export_gripper_cli(args)

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    sim = scene = None
    report = {
        "schema": "kuavo_dynamics_benchmark_v1",
        "device": args.device,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from kuavo_isaaclab_scene.rl.scenes.robot import build_robot_cfg

        @configclass
        class RobotOnlySceneCfg(InteractiveSceneCfg):
            robot = build_robot_cfg()

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=1.0 / 120.0, render_interval=4, device=args.device,
        ))
        scene = InteractiveScene(RobotOnlySceneCfg(
            num_envs=args.num_envs, env_spacing=3.0, replicate_physics=True,
        ))
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        for _ in range(12):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(1.0 / 120.0)
        torch.cuda.synchronize()

        view = robot.root_physx_view
        zero_qdd = torch.zeros_like(robot.data.joint_vel)

        def gravity():
            return view.get_gravity_compensation_forces()

        def inverse_dynamics():
            mass = view.get_generalized_mass_matrices()
            coriolis = view.get_coriolis_and_centrifugal_compensation_forces()
            gravity_force = view.get_gravity_compensation_forces()
            return torch.bmm(mass, zero_qdd.unsqueeze(-1)).squeeze(-1) + coriolis + gravity_force

        def measure(fn):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            values = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                for _ in range(args.iterations):
                    fn()
                torch.cuda.synchronize()
                values.append((time.perf_counter() - start) * 1000.0 / args.iterations)
            return {
                "median_ms_per_call": statistics.median(values),
                "min_ms_per_call": min(values),
                "max_ms_per_call": max(values),
                "all_ms_per_call": values,
            }

        report["num_dofs"] = robot.num_joints
        report["gravity_query"] = measure(gravity)
        report["mass_coriolis_gravity_and_matvec"] = measure(inverse_dynamics)
        report["control_rate_hz"] = 30.0
        for key in ("gravity_query", "mass_coriolis_gravity_and_matvec"):
            ms = report[key]["median_ms_per_call"]
            report[key]["single_gpu_wall_percent_at_30hz"] = ms * 30.0 / 10.0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("[DYNAMICS BENCHMARK] " + json.dumps(report), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if "error" in report else 0)


if __name__ == "__main__":
    main()

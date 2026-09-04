#!/usr/bin/env python3
"""Out-of-process LeRobot/GR00T inference worker for Isaac Lab evaluation."""

from __future__ import annotations

import argparse
import socket
import traceback

import torch

from .groot_lerobot_bridge import (
    LeRobotGrootRunner,
    receive_framed_pickle,
    send_framed_pickle,
)


def _feature_shapes(policy) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for key, feature in (getattr(policy.config, "input_features", {}) or {}).items():
        shape = getattr(feature, "shape", None)
        if shape is not None:
            result[str(key)] = tuple(int(size) for size in shape)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-fd", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--actions-per-inference", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    connection = socket.socket(fileno=args.ipc_fd)
    runner = None
    try:
        runner = LeRobotGrootRunner.from_pretrained(
            args.checkpoint,
            device=args.device,
            actions_per_inference=args.actions_per_inference,
            local_files_only=args.local_files_only,
            expected_action_dim=args.expected_action_dim,
            base_model_path=args.base_model_path,
            strict=not args.no_strict,
        )
        import lerobot

        send_framed_pickle(
            connection,
            {
                "status": "ready",
                "expected_input_keys": runner.expected_input_keys,
                "input_shapes": _feature_shapes(runner.policy),
                "output_action_dim": runner.output_action_dim,
                "lerobot_version": getattr(lerobot, "__version__", "unknown"),
            },
        )
        while True:
            request = receive_framed_pickle(connection)
            operation = request.get("op")
            try:
                if operation == "reset":
                    runner.reset()
                    send_framed_pickle(connection, {"status": "ok"})
                elif operation == "select_action":
                    sample = runner.select_action(request["observation"])
                    # A plain list keeps IPC compatible across the worker's
                    # NumPy 2.x and Isaac Sim's NumPy 1.x environments.
                    action = sample.action.detach().cpu().tolist()
                    send_framed_pickle(
                        connection,
                        {
                            "status": "ok",
                            "action": action,
                            "inferred_new_chunk": sample.inferred_new_chunk,
                            "inference_ms": sample.inference_ms,
                        },
                    )
                elif operation == "close":
                    send_framed_pickle(connection, {"status": "ok"})
                    break
                else:
                    raise ValueError(f"Unknown worker operation: {operation!r}")
            except Exception:
                send_framed_pickle(connection, {"status": "error", "error": traceback.format_exc()})
    except Exception:
        try:
            send_framed_pickle(connection, {"status": "error", "error": traceback.format_exc()})
        except OSError:
            pass
    finally:
        if runner is not None:
            runner.close()
        connection.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

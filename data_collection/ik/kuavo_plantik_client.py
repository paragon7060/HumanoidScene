"""Persistent client for the standalone Kuavo plantIK worker.

The worker is intentionally kept out of the Isaac process: Drake and Isaac can
have conflicting native dependencies, while this line protocol keeps the
collection code independent of ROS.
"""

from __future__ import annotations

import math
import os
import select
import subprocess
from typing import Iterable, Sequence


ARM_DOF = 14


def _finite_values(values: Iterable[float], expected: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


class KuavoPlantIkClient:
    """Call the Kuavo CoMIK solver through a long-lived native worker."""

    def __init__(
        self,
        executable: str | os.PathLike[str],
        urdf: str | os.PathLike[str],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.process = subprocess.Popen(
            [os.fspath(executable), os.fspath(urdf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def solve(
        self,
        q0: Sequence[float],
        left_position: Sequence[float],
        left_quat_xyzw: Sequence[float],
        right_position: Sequence[float],
        right_quat_xyzw: Sequence[float],
    ) -> list[float]:
        q0_values = _finite_values(q0, ARM_DOF, "q0")
        left_pos = _finite_values(left_position, 3, "left_position")
        left_quat = _finite_values(left_quat_xyzw, 4, "left_quat_xyzw")
        right_pos = _finite_values(right_position, 3, "right_position")
        right_quat = _finite_values(right_quat_xyzw, 4, "right_quat_xyzw")
        if self.process.poll() is not None:
            raise RuntimeError(f"plantIK worker exited with code {self.process.returncode}")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        values = q0_values + left_pos + left_quat + right_pos + right_quat
        self.process.stdin.write(" ".join(f"{value:.17g}" for value in values) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout_s)
        if not ready:
            self.close()
            raise TimeoutError(f"plantIK worker did not answer within {self.timeout_s:.3f}s")
        response = self.process.stdout.readline().strip().split()
        if not response or response[0] != "1":
            raise RuntimeError("Kuavo plantIK reported target as unreachable")
        return _finite_values(response[1:], ARM_DOF, "plantIK solution")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)

    def __enter__(self) -> "KuavoPlantIkClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ARM_DOF", "KuavoPlantIkClient"]


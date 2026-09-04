"""Tracking-loss safety primitives shared by Quest teleoperation paths."""

from __future__ import annotations

from dataclasses import dataclass


class GripperCommandLatch:
    """Keep the last gripper command when its corresponding hand disappears."""

    def __init__(self, active_sides: tuple[str, ...], initial_command: float = 1.0) -> None:
        self.active_sides = tuple(active_sides)
        self._commands = {side: float(initial_command) for side in self.active_sides}

    def reset(self, command: float = 1.0) -> None:
        for side in self.active_sides:
            self._commands[side] = float(command)

    def advance(
        self,
        desired_commands: tuple[float, ...],
        *,
        left_valid: bool,
        right_valid: bool,
    ) -> tuple[float, ...]:
        if len(desired_commands) != len(self.active_sides):
            raise ValueError("Desired gripper command count does not match active sides.")
        validity = {"left": bool(left_valid), "right": bool(right_valid)}
        for side, command in zip(self.active_sides, desired_commands):
            if validity.get(side, False):
                self._commands[side] = float(command)
        return tuple(self._commands[side] for side in self.active_sides)


@dataclass(frozen=True)
class TrackingSafetyState:
    control_allowed: bool
    recording_paused: bool
    abort_episode: bool
    recovered: bool
    loss_duration_s: float


class TrackingLossGuard:
    """Require stable recovery and abort episodes after prolonged tracking loss."""

    def __init__(self, *, recovery_frames: int = 5, abort_after_s: float = 1.0) -> None:
        if recovery_frames < 1:
            raise ValueError("recovery_frames must be at least one.")
        if abort_after_s <= 0.0:
            raise ValueError("abort_after_s must be positive.")
        self.recovery_frames = int(recovery_frames)
        self.abort_after_s = float(abort_after_s)
        self.reset()

    def reset(self) -> None:
        self._loss_started_at: float | None = None
        self._recovery_count = 0
        self._had_loss = False

    def advance(self, all_tracking_valid: bool, now_s: float) -> TrackingSafetyState:
        now_s = float(now_s)
        if not all_tracking_valid:
            if self._loss_started_at is None:
                self._loss_started_at = now_s
            self._had_loss = True
            self._recovery_count = 0
            duration = max(0.0, now_s - self._loss_started_at)
            return TrackingSafetyState(False, True, duration >= self.abort_after_s, False, duration)

        if not self._had_loss:
            return TrackingSafetyState(True, False, False, False, 0.0)

        self._recovery_count += 1
        if self._recovery_count < self.recovery_frames:
            started_at = now_s if self._loss_started_at is None else self._loss_started_at
            duration = max(0.0, now_s - started_at)
            return TrackingSafetyState(False, True, False, False, duration)

        self._loss_started_at = None
        self._recovery_count = 0
        self._had_loss = False
        return TrackingSafetyState(True, False, False, True, 0.0)

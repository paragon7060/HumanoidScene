"""Simulator-independent hand switching, command gestures and pinch hysteresis."""

import math
import numpy as np

from .teleop_mapping import _valid_pose, _pinch_distance


def hand_packet(hand):
    """Adapt fresh wrist tracking to the shared pose mapper, never to recording."""
    if not hand or not all(_valid_pose(hand.get(k)) for k in ("wrist", "thumb_tip", "index_tip")):
        return None
    return np.stack((np.asarray(hand["wrist"], dtype=float), np.zeros(7)))


def controller_squeeze(packet):
    if packet is None or np.shape(packet) != (2, 7) or not np.all(np.isfinite(packet)):
        return None
    return float(packet[1, 3])


class LongPress:
    """Fire once per deliberate hold; tracking loss is not a button release."""

    def __init__(self, seconds=1.2):
        self.seconds = seconds
        self.since = None
        self.latched = False
        self.last_time = None

    def update(self, now, value):
        if self.last_time is not None and now - self.last_time > .25:
            self.since = None
        self.last_time = now
        if value is None or not math.isfinite(value):
            self.since = None
            return False
        if value <= .2:
            self.since, self.latched = None, False
        elif value >= .8 and not self.latched:
            if self.since is None:
                self.since = now
            if now - self.since >= self.seconds:
                self.latched = True
                return True
        else:
            self.since = None
        return False


class HandModeSwitch:
    """Request -> countdown -> continuously valid destination input -> ready.

    The owner closes recording/holds motion on 'begin', captures references on
    'ready', and records a NEW episode only when switching to hands.
    """

    def __init__(self, mode="controllers"):
        self.mode = mode
        self.target = None
        self.deadline = None
        self.stable_since = None
        self.last_time = None
        self.press = LongPress()

    @property
    def pending(self):
        return self.target is not None

    def cancel(self):
        self.target = self.deadline = self.stable_since = None

    def update(self, now, squeeze, *, hands_ready, controllers_ready, head_ready):
        if self.press.update(now, squeeze):
            if self.pending:
                self.cancel()
                return "cancel"
            self.target = "hands" if self.mode == "controllers" else "controllers"
            self.deadline = now + 3.
            self.stable_since = None
            return "begin"
        if not self.pending:
            return None
        if self.last_time is not None and now - self.last_time > .25:
            self.stable_since = None
        self.last_time = now
        ready = hands_ready if self.target == "hands" else controllers_ready and squeeze is not None and squeeze <= .2
        if now < self.deadline or not head_ready or not ready:
            self.stable_since = None
            return None
        if self.stable_since is None:
            self.stable_since = now
        if now - self.stable_since >= .5:
            self.mode = self.target
            self.cancel()
            return "ready"
        return None

    def status(self, now):
        if not self.pending:
            return self.mode.upper()
        remaining = max(0, math.ceil(self.deadline - now))
        return (f"{self.target.upper()} IN {remaining}s - PUT CONTROLLERS DOWN" if remaining and self.target == "hands"
                else f"{self.target.upper()} IN {remaining}s - RELEASE GRIP" if remaining
                else f"WAIT: BOTH {self.target.upper()} + HEAD TRACKED")


class HandCommands:
    """Thumb-middle pinch for commands; thumb-index remains gripper control.

    A command needs a visibly extended index finger. The short candidate hold
    is exposed so the owner freezes the hand goal/gripper while gesturing.
    """

    def __init__(self):
        self.holds = {s: LongPress(1.0) for s in ("left", "right")}
        self.active = {s: False for s in self.holds}

    def update(self, now, hands):
        events = []
        for side, hold in self.holds.items():
            hand = hands[side]
            valid = hand and all(_valid_pose(hand.get(k)) for k in ("wrist", "thumb_tip", "middle_tip", "index_tip"))
            command = bool(valid and _pinch_distance(hand) >= .07 and
                           np.linalg.norm(hand["thumb_tip"][:3] - hand["middle_tip"][:3]) <= .025)
            self.active[side] = command
            if hold.update(now, float(command) if valid else None):
                events.append("preview" if side == "left" else "toggle")
        # A simultaneous command is ambiguous: STOP takes priority over REC.
        return ["preview"] if "preview" in events else events


class HandGripper:
    def __init__(self, close_distance=.055):
        self.close_distance = close_distance
        self.values = {"left": 1., "right": 1.}
        self.armed = {"left": True, "right": True}

    def sync(self, values):
        self.values.update(values)
        self.armed = {s: False for s in self.values}

    def update(self, side, hand, *, hold=False):
        distance = _pinch_distance(hand)
        if not hold and hand_packet(hand) is not None and math.isfinite(distance):
            if not self.armed[side]:
                self.armed[side] = (distance <= self.close_distance if self.values[side] < 0
                                    else distance >= self.close_distance + .015)
                return self.values[side]
            if distance <= self.close_distance:
                self.values[side] = -1.
            elif distance >= self.close_distance + .015:
                self.values[side] = 1.
        return self.values[side]


class HandTrackingGuard:
    """Do not turn a prolonged tracking outage into an automatic resumption."""

    def __init__(self):
        self.lost_since = None

    def update(self, now, *, following, valid):
        if not following or valid:
            self.lost_since = None
        elif self.lost_since is None:
            self.lost_since = now
        return self.lost_since is not None and now - self.lost_since >= 2.

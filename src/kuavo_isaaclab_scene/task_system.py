#!/usr/bin/env python3
"""Task logic for the Kuavo gravity-rack to conveyor workcell.

This module deliberately has no Isaac Sim imports.  The slot allocator and task
state machine can therefore be unit-tested without launching Kit and can also be
reused by a scripted controller, a learned policy, or a multi-worker scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import dist
from typing import Mapping, Sequence


Vec3 = tuple[float, float, float]


class PlacementMode(Enum):
    """Action needed before a box can be released at the conveyor infeed."""

    PLACE = auto()
    PUSH_THEN_PLACE = auto()
    WAIT = auto()
    BLOCKED = auto()


class TaskPhase(Enum):
    """High-level task lifecycle."""

    TRANSFERRING = auto()
    WAITING_FOR_BUTTON = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class PlacementPlan:
    """Collision-free plan for inserting one box into the conveyor queue."""

    mode: PlacementMode
    worker_id: str
    box_id: str
    drop_slot: int | None
    target: Vec3 | None
    empty_slot: int | None = None
    push_distance: float = 0.0
    # Ordered farthest-to-nearest so an executor can move boxes without overlap.
    push_box_ids: tuple[str, ...] = ()


class ConveyorSlotManager:
    """Allocate and reserve positions on a stopped, single-lane conveyor.

    Slot zero is the robot-reachable infeed.  A box is inserted directly when
    that slot is empty.  If it is occupied and a downstream gap exists, the
    contiguous queue is advanced by one slot before insertion.  Reservations
    prevent two workers from being sent to the same space.
    """

    def __init__(
        self,
        slot_centers: Sequence[Vec3],
        *,
        match_radius: float = 0.14,
    ) -> None:
        if len(slot_centers) < 2:
            raise ValueError("At least two conveyor slots are required.")
        self.slot_centers = tuple(slot_centers)
        self.match_radius = match_radius
        self._reservations: dict[int, tuple[str, str]] = {}

    @property
    def slot_pitch(self) -> float:
        return dist(self.slot_centers[0], self.slot_centers[1])

    def occupied_slots(self, box_positions: Mapping[str, Vec3]) -> dict[int, str]:
        """Map each detected slot to its closest box."""
        matches: list[tuple[float, int, str]] = []
        for box_id, position in box_positions.items():
            for slot_id, center in enumerate(self.slot_centers):
                distance = dist(position, center)
                if distance <= self.match_radius:
                    matches.append((distance, slot_id, box_id))

        occupied: dict[int, str] = {}
        used_boxes: set[str] = set()
        for _, slot_id, box_id in sorted(matches):
            if slot_id not in occupied and box_id not in used_boxes:
                occupied[slot_id] = box_id
                used_boxes.add(box_id)
        return occupied

    def reserve(
        self,
        worker_id: str,
        box_id: str,
        box_positions: Mapping[str, Vec3],
    ) -> PlacementPlan:
        """Reserve a safe infeed plan for one worker.

        A reservation at the infeed is treated as a short critical section.
        Other workers wait instead of trying to push a box that is currently
        being placed.
        """
        if any(owner == worker_id and reserved_box == box_id for owner, reserved_box in self._reservations.values()):
            raise ValueError(f"{worker_id}/{box_id} already owns a reservation.")

        if 0 in self._reservations:
            return PlacementPlan(PlacementMode.WAIT, worker_id, box_id, None, None)

        occupied = self.occupied_slots(box_positions)
        unavailable = set(occupied) | set(self._reservations)

        if 0 not in unavailable:
            self._reservations[0] = (worker_id, box_id)
            return PlacementPlan(
                PlacementMode.PLACE,
                worker_id,
                box_id,
                0,
                self.slot_centers[0],
                empty_slot=0,
            )

        empty_slot = next(
            (slot_id for slot_id in range(1, len(self.slot_centers)) if slot_id not in unavailable),
            None,
        )
        if empty_slot is None:
            return PlacementPlan(PlacementMode.BLOCKED, worker_id, box_id, None, None)

        # A non-contiguous gap cannot be filled by pushing only the infeed
        # queue.  In that case the robot must wait for the belt or another
        # worker to clear the obstruction.
        queue_end = 0
        while queue_end in occupied:
            queue_end += 1
        if queue_end != empty_slot:
            return PlacementPlan(PlacementMode.WAIT, worker_id, box_id, None, None)

        self._reservations[0] = (worker_id, box_id)
        push_ids = tuple(occupied[slot_id] for slot_id in range(empty_slot - 1, -1, -1))
        return PlacementPlan(
            PlacementMode.PUSH_THEN_PLACE,
            worker_id,
            box_id,
            0,
            self.slot_centers[0],
            empty_slot=empty_slot,
            push_distance=self.slot_pitch,
            push_box_ids=push_ids,
        )

    def commit(self, plan: PlacementPlan) -> None:
        """Release the reservation after the executor completes the plan."""
        owner = self._reservations.get(0)
        if owner != (plan.worker_id, plan.box_id):
            raise ValueError("Cannot commit a plan that does not own the infeed reservation.")
        del self._reservations[0]

    def cancel(self, plan: PlacementPlan) -> None:
        """Release an abandoned plan without changing conveyor occupancy."""
        if self._reservations.get(0) == (plan.worker_id, plan.box_id):
            del self._reservations[0]


class RackConveyorTask:
    """Gate conveyor motion behind box completion and a valid button press."""

    def __init__(self, rack_box_ids: Sequence[str]) -> None:
        if not rack_box_ids:
            raise ValueError("rack_box_ids must not be empty.")
        self.rack_box_ids = frozenset(rack_box_ids)
        self.transferred_box_ids: set[str] = set()
        self.phase = TaskPhase.TRANSFERRING
        self.early_button_presses = 0
        self.valid_button_presses = 0

    def update_transferred(self, box_ids_on_conveyor: Sequence[str]) -> TaskPhase:
        # Completion is latched.  Once the valid button press starts the belt,
        # totes are expected to leave the monitored loading slots.
        if self.phase is TaskPhase.COMPLETE:
            return self.phase
        self.transferred_box_ids = self.rack_box_ids.intersection(box_ids_on_conveyor)
        self.phase = (
            TaskPhase.WAITING_FOR_BUTTON
            if self.transferred_box_ids == self.rack_box_ids
            else TaskPhase.TRANSFERRING
        )
        return self.phase

    @property
    def remaining_box_ids(self) -> frozenset[str]:
        return self.rack_box_ids.difference(self.transferred_box_ids)

    @property
    def conveyor_enabled(self) -> bool:
        return self.phase is TaskPhase.COMPLETE

    def press_button(self) -> bool:
        """Accept the button only after every rack box is on the conveyor."""
        if self.phase is not TaskPhase.WAITING_FOR_BUTTON:
            self.early_button_presses += 1
            return False
        self.valid_button_presses += 1
        self.phase = TaskPhase.COMPLETE
        return True

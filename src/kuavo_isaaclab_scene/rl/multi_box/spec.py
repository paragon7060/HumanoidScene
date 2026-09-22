"""Versioned contract for the randomized rack-to-conveyor task.

This module is simulator-independent.  Scene construction, observations, and
rewards consume this contract, but must not add implicit task assumptions to it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..action_spaces import ACTION_SPACES


SCHEMA_VERSION = 2
MAX_BOXES = 12
BOX_TYPES = ("small", "medium")
SKILLS = ("grasp", "carry", "place")
PREDECESSOR = {"carry": "grasp", "place": "carry"}


@dataclass(frozen=True)
class RackRegionSpec:
    """One semantic rack area; it is not a single-box placement slot."""

    name: str
    shelf: int
    side: str
    allowed_box_types: tuple[str, ...]
    box_type_sampling: str = "uniform"

    def validate(self) -> None:
        expected_name = f"shelf_{self.shelf}_{self.side}"
        if self.name != expected_name:
            raise ValueError(f"Rack region {self.name!r} must be named {expected_name!r}.")
        if self.shelf not in (2, 3):
            raise ValueError("Multi-box rack regions are limited to shelves 2 and 3.")
        if self.side not in ("right", "left"):
            raise ValueError("Rack region side must be 'right' or 'left'.")
        if not self.allowed_box_types or len(set(self.allowed_box_types)) != len(self.allowed_box_types):
            raise ValueError(f"Rack region {self.name!r} needs distinct allowed box types.")
        unknown = set(self.allowed_box_types) - set(BOX_TYPES)
        if unknown:
            raise ValueError(f"Rack region {self.name!r} has unsupported box types: {sorted(unknown)}")
        if self.box_type_sampling != "uniform":
            raise ValueError("Rack box types currently use uniform random sampling.")


DEFAULT_RACK_REGIONS = (
    RackRegionSpec("shelf_2_right", 2, "right", ("small", "medium")),
    RackRegionSpec("shelf_2_left", 2, "left", ("small", "medium")),
    RackRegionSpec("shelf_3_right", 3, "right", ("small",)),
    RackRegionSpec("shelf_3_left", 3, "left", ("small",)),
)


@dataclass(frozen=True)
class MultiBoxSpec:
    """Task contract shared by all low-level skills and the high-level policy."""

    schema_version: int = SCHEMA_VERSION
    max_boxes: int = MAX_BOXES
    rack_regions: tuple[RackRegionSpec, ...] = DEFAULT_RACK_REGIONS
    region_sampling: str = "uniform"
    low_level_spawn_count: int = 1
    full_spawn_count_range: tuple[int, int] = (1, MAX_BOXES)
    rack_xy_jitter: tuple[float, float] = (0.10, 0.10)
    rack_yaw_jitter: float = math.radians(15.0)
    conveyor_xy_jitter: tuple[float, float] = (0.05, 0.05)
    conveyor_yaw_jitter: float = math.radians(3.0)

    strategy: str = "end-to-end"
    action_space: str = "all-joints"
    skill: str = "full"
    episode_seconds: float | None = None
    reset_bank: str | None = None
    snapshot_dir: str | None = None
    max_snapshots: int = 128

    # Simulator guards are not success criteria or policy observations.
    max_box_lift_height: float = 0.50
    max_box_linear_speed: float = 10.0
    max_box_angular_speed: float = 100.0
    workspace_radius: float = 1.5
    rack_contact_force: float = 10.0
    self_collision_clearance: float = 0.003
    self_collision_enabled: bool = True
    collision_constraints_enabled: bool = True

    # Reset physics is outside the task MDP.  A reset becomes trainable only
    # after its selected box remains on the assigned shelf/region and is still
    # for the configured hold time.
    reset_settle_min_seconds: float = 0.25
    reset_settle_hold_seconds: float = 0.25
    reset_settle_timeout_seconds: float = 2.0
    reset_settle_linear_speed: float = 0.01
    reset_settle_angular_speed: float = 0.05
    reset_shelf_clearance_range: tuple[float, float] = (-0.02, 0.05)

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.rack_regions)

    @property
    def spawn_shelves(self) -> tuple[int, ...]:
        return tuple(sorted({region.shelf for region in self.rack_regions}))

    def allowed_box_types(self, region_name: str) -> tuple[str, ...]:
        for region in self.rack_regions:
            if region.name == region_name:
                return region.allowed_box_types
        raise KeyError(f"Unknown rack region: {region_name}")

    @property
    def spawn_count_range(self) -> tuple[int, int]:
        """Inclusive active-box count range for the selected training scope."""
        if self.skill == "full":
            return self.full_spawn_count_range
        return (self.low_level_spawn_count, self.low_level_spawn_count)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Multi-box schema must be version {SCHEMA_VERSION}.")
        if self.max_boxes != MAX_BOXES:
            raise ValueError(f"This task contract requires N_max={MAX_BOXES}.")
        if self.region_sampling != "uniform":
            raise ValueError("Rack regions currently use uniform random sampling.")
        if self.low_level_spawn_count != 1:
            raise ValueError("Each low-level skill trains with exactly one active box.")
        if (len(self.full_spawn_count_range) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           for value in self.full_spawn_count_range)
                or not 1 <= self.full_spawn_count_range[0] <= self.full_spawn_count_range[1] <= self.max_boxes):
            raise ValueError(f"Full-task spawn count must be an inclusive range inside [1, {self.max_boxes}].")
        jitter = (*self.rack_xy_jitter, self.rack_yaw_jitter,
                  *self.conveyor_xy_jitter, self.conveyor_yaw_jitter)
        if (len(self.rack_xy_jitter) != 2 or len(self.conveyor_xy_jitter) != 2
                or any(not math.isfinite(value) or value < 0 for value in jitter)):
            raise ValueError("Rack/conveyor XY and yaw jitter limits must be finite and nonnegative.")

        for region in self.rack_regions:
            region.validate()
        if len({region.name for region in self.rack_regions}) != len(self.rack_regions):
            raise ValueError("Rack region names must be unique.")
        actual = {
            (region.shelf, region.side): frozenset(region.allowed_box_types)
            for region in self.rack_regions
        }
        expected = {
            (2, "right"): frozenset(("small", "medium")),
            (2, "left"): frozenset(("small", "medium")),
            (3, "right"): frozenset(("small",)),
            (3, "left"): frozenset(("small",)),
        }
        if actual != expected or len(self.rack_regions) != len(expected):
            raise ValueError(
                "Rack regions must be shelf 2 right/left with small+medium and "
                "shelf 3 right/left with small only."
            )

        if self.strategy not in ("staged", "end-to-end"):
            raise ValueError("Unknown multi-box training strategy.")
        if self.skill not in (*SKILLS, "full"):
            raise ValueError(f"Unknown skill {self.skill!r}; choose {(*SKILLS, 'full')}.")
        if self.action_space not in ACTION_SPACES:
            raise ValueError("Unknown action space.")
        if self.action_space != "all-joints":
            raise ValueError("The two-arm rack-to-conveyor task requires all-joints control.")
        if self.strategy == "end-to-end" and self.skill != "full":
            raise ValueError("End-to-end training uses the full task.")
        if self.skill in PREDECESSOR and not self.reset_bank:
            raise ValueError(f"{self.skill} needs a {PREDECESSOR[self.skill]} success reset bank.")
        if self.skill in ("grasp", "full") and self.reset_bank:
            raise ValueError(f"{self.skill} starts from a fresh randomized rack scene.")

        safety = (self.max_box_lift_height, self.max_box_linear_speed,
                  self.max_box_angular_speed, self.workspace_radius,
                  self.rack_contact_force, self.self_collision_clearance)
        if not all(math.isfinite(value) and value > 0 for value in safety):
            raise ValueError("Simulator safety limits must be finite and positive.")
        if self.self_collision_clearance >= 0.04:
            raise ValueError("Self-collision clearance must be below the 4 cm influence range.")

        settling = (
            self.reset_settle_min_seconds,
            self.reset_settle_hold_seconds,
            self.reset_settle_timeout_seconds,
            self.reset_settle_linear_speed,
            self.reset_settle_angular_speed,
        )
        if not all(math.isfinite(value) and value > 0 for value in settling):
            raise ValueError("Reset settling limits must be finite and positive.")
        if self.reset_settle_timeout_seconds < (
                self.reset_settle_min_seconds + self.reset_settle_hold_seconds):
            raise ValueError("Reset settling timeout must cover minimum and hold times.")
        low_clearance, high_clearance = self.reset_shelf_clearance_range
        if (not math.isfinite(low_clearance) or not math.isfinite(high_clearance)
                or low_clearance >= high_clearance):
            raise ValueError("Reset shelf-clearance range must be finite and ordered.")

        if self.episode_seconds is not None and (
                not math.isfinite(self.episode_seconds) or self.episode_seconds <= 0):
            raise ValueError("Optional episode timeout must be finite and positive.")
        if not self.collision_constraints_enabled:
            raise ValueError("Multi-box v2 requires collision constraints.")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be positive.")

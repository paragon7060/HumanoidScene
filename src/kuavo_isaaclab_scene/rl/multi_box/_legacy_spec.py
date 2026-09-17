"""Temporary v1 contract used only while the existing runner migrates to v2."""
from dataclasses import dataclass
import math
from ..action_spaces import ACTION_SPACES

SKILLS = ("pick", "extract", "carry", "place")
PREDECESSOR = dict(zip(SKILLS[1:], SKILLS[:-1]))


@dataclass(frozen=True)
class MultiBoxSpec:
    box_names: tuple = ("small_box_0", "small_box_1", "medium_box_0", "large_box_0")
    shelves: tuple = (1, 1, 2, 2)
    strategy: str = "end-to-end"
    action_space: str = "all-joints"
    skill: str = "full"
    episode_seconds: float = 120.0
    lift_height: float = .06
    max_box_lift_height: float = .50
    max_box_linear_speed: float = 10.
    max_box_angular_speed: float = 100.
    extraction_clearance: float = .06
    carry_distance: float = .25
    placement_hold: float = .5
    skill_hold: float = .2
    max_tilt: float = math.radians(40)
    placement_speed: float = .08
    placement_angular_speed: float = .35
    support_force: float = .2
    support_tolerance: float = .035
    clearance: float = .015
    release_distance: float = .025
    settle_seconds: float = .5
    failure_floor: float = .12
    workspace_radius: float = 4.0
    discount: float = .99
    reset_bank: str | None = None
    snapshot_dir: str | None = None
    max_snapshots: int = 128
    # Only moving toward the robot-side rack face counts as extraction.
    rack_outward_local: tuple = (0., 1., 0.)

    def validate(self):
        if (not all(math.isfinite(v) and v > 0 for v in (self.max_box_lift_height,
                self.max_box_linear_speed, self.max_box_angular_speed))
                or self.max_box_lift_height <= self.lift_height):
            raise ValueError("Box safety limits must be finite, positive and above the lift goal.")
        if self.action_space not in ACTION_SPACES:
            raise ValueError("Unknown action space.")
        if self.action_space == "right-arm" and self.skill in ("carry", "full"):
            raise ValueError("Carry/full requires base motion; select --action-space all-joints.")
        if len(self.box_names) != 4 or len(set(self.box_names)) != 4:
            raise ValueError("Exactly four distinct boxes are required.")
        if len(self.shelves) != 4 or sorted(self.shelves) != [1, 1, 2, 2]:
            raise ValueError("Require two boxes on shelf 1 and two on shelf 2.")
        if self.strategy not in ("staged", "end-to-end") or self.skill not in (*SKILLS, "full"):
            raise ValueError("Unknown strategy or skill.")
        if self.strategy == "end-to-end" and self.skill != "full":
            raise ValueError("End-to-end uses the full task.")
        if self.skill in PREDECESSOR and not self.reset_bank:
            raise ValueError(f"{self.skill} needs a {PREDECESSOR[self.skill]} success reset bank.")
        if self.skill in ("pick", "full") and self.reset_bank:
            raise ValueError("Pick/full evaluations must start from the rack, without a reset bank.")
        for key in ("episode_seconds", "lift_height", "placement_hold", "skill_hold",
                    "support_force", "workspace_radius", "carry_distance"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if not 0 < self.discount < 1 or self.max_snapshots < 1:
            raise ValueError("Invalid discount/snapshot limit.")
        if self.rack_outward_local not in ((0., 1., 0.), (0., -1., 0.), (1., 0., 0.), (-1., 0., 0.)):
            raise ValueError("Rack outward direction must be a signed local X/Y axis.")


def validate_shelves(plans, spec):
    by_key = {p.scene_key: p for p in plans.values()}
    for name, shelf in zip(spec.box_names, spec.shelves):
        if name not in by_key or not by_key[name].on_rack or by_key[name].shelf != shelf:
            raise ValueError(f"Capture {name} on shelf {shelf} in the selected rack-box-poses file.")

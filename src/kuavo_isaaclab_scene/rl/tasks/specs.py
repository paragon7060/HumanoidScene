"""Simulator-independent task presets; distances in meters and angles in radians."""

from dataclasses import dataclass, replace

PHASES = ("approach_rack", "pick", "carry", "place", "press_button")
TASKS = (*PHASES, "full")
PREDECESSOR = {"pick": "approach_rack", "carry": "pick", "place": "carry", "press_button": "place"}
REQUIRES_RESET_BANK = ("carry", "place", "press_button")


@dataclass(frozen=True)
class TaskSpec:
    name: str = "approach_rack"
    box_names: tuple[str, ...] = ("small_box_0",)
    episode_length_s: float = 15.0
    reset_bank: str | None = None
    snapshot_dir: str | None = None
    max_snapshots: int = 100
    approach_distance: float = 0.55
    navigation_tolerance: float = 0.10
    heading_tolerance: float = 0.18
    lift_height: float = 0.10
    grasp_distance: float = 0.16
    grasp_force: float = 0.20
    required_grasp_hands: int = 1
    max_tilt: float = 0.30
    settle_speed: float = 0.08
    settle_angular_speed: float = 0.35
    support_tolerance: float = 0.025
    clearance: float = 0.025
    hold_seconds: float = 0.30
    button_travel: float = 0.006
    button_hand_distance: float = 0.14
    conveyor_run_seconds: float = 0.5
    conveyor_speed: float = 0.12
    slot_count: int = 4
    slot_pitch: float = 0.52
    reset_xy_jitter: float = 0.20
    reset_yaw_jitter: float = 0.30
    prefill_count: int = 1
    cargo_per_box: int = 2
    cargo_radius: float = 0.012
    randomization: bool = True
    curriculum_steps: int = 300_000
    # Physical two-finger contacts; change these when onboarding another hand.
    finger_bodies: tuple[str, ...] = ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")
    tool_bodies: tuple[str, str] = ("zarm_l7_end_effector", "zarm_r7_end_effector")
    tool_offset: tuple[float, float, float] = (0.0, 0.0, -0.12)

    def validate(self) -> None:
        if self.name not in TASKS:
            raise ValueError(f"Unknown task {self.name!r}; choose {TASKS}")
        if not self.box_names or len(set(self.box_names)) != len(self.box_names):
            raise ValueError("box_names must be nonempty and unique.")
        if self.required_grasp_hands not in (1, 2):
            raise ValueError("required_grasp_hands must be 1 or 2.")
        if len(self.finger_bodies) != 4:
            raise ValueError("Provide two opposing finger bodies per hand (left then right).")
        if self.name in REQUIRES_RESET_BANK and not self.reset_bank:
            raise ValueError(f"{self.name} requires --reset-bank from a successful {PREDECESSOR[self.name]} rollout.")
        if self.reset_bank and self.name not in PREDECESSOR:
            raise ValueError(f"{self.name} uses a fresh rack reset, not a predecessor bank.")
        if self.slot_count < 1 or not 0 <= self.prefill_count < self.slot_count:
            raise ValueError("prefill_count must leave at least one free conveyor slot.")
        if self.slot_count < len(self.box_names) + self.prefill_count:
            raise ValueError("The stopped conveyor needs space for every selected box plus prefill. Reduce boxes/prefill.")
        if self.cargo_per_box not in (0, 1, 2):
            raise ValueError("cargo_per_box currently supports 0, 1 or 2.")
        if min(self.episode_length_s, self.hold_seconds, self.slot_pitch, self.cargo_radius) <= 0:
            raise ValueError("Durations, slot pitch and cargo radius must be positive.")


def task_spec(name: str, **overrides) -> TaskSpec:
    durations = dict(approach_rack=15.0, pick=15.0, carry=15.0, place=12.0, press_button=12.0, full=90.0)
    if name not in durations:
        raise ValueError(f"Unknown task {name!r}; choose {TASKS}")
    return replace(TaskSpec(name=name, episode_length_s=durations[name]), **overrides)

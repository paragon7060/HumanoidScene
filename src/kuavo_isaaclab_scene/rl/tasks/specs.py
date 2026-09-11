"""Simulator-independent task presets; distances in meters and angles in radians."""

from dataclasses import dataclass, replace
import math

PHASES = ("approach_rack", "pick", "carry", "place", "press_button")
TASKS = (*PHASES, "full")
PREDECESSOR = {"pick": "approach_rack", "carry": "pick", "place": "carry", "press_button": "place"}
REQUIRES_RESET_BANK = ("carry", "place", "press_button")


@dataclass(frozen=True)
class TaskSpec:
    name: str = "approach_rack"
    control_mode: str = "whole-body"
    active_arm: str = "both"  # arms-only: both/left/right; the other arm+hand is physically locked.
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
    grasp_mode: str = "body"
    grasp_hand: str = "right"
    # Shared candidate pool for BOTH hands, in sensor filter order.
    grasp_flaps: tuple[str, str] = ("flap_right", "flap_left")
    flap_contact_region: str = "surface"  # "top_band" restores upper-edge-only acquisition.
    flap_grasp_depth: float = 0.015
    flap_top_band: float = 0.030
    flap_contact_margin: float = 0.004
    grasp_contact_grace_s: float = 0.10
    grasp_hold_slip_m: float = 0.02
    grasp_open_tolerance_m: float = 0.008
    flap_lock_degrees: float = 0.5
    unexpected_contact_limit: float = 10.0
    obstacle_contact_force: float = 20.0
    reset_settle_seconds: float = 0.0
    reset_settle_hold_seconds: float = 0.2
    reset_settle_timeout: float = 0.0  # 0: fixed initial delay, no velocity gate or timeout failure
    prelift_position_scale: float = 0.05
    prelift_speed_scale: float = 0.20
    prelift_angular_scale: float = 1.0
    prelift_rotation_scale: float = math.radians(30)
    prelift_position_deadband: float = 0.005
    prelift_speed_deadband: float = 0.02
    prelift_angular_deadband: float = 0.10
    prelift_rotation_deadband: float = math.radians(5)
    prelift_grasp_scale: float = 0.25
    prelift_penalty_cap: float = 1.0
    grasp_lift_clearance: float = 0.01
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
    # Legacy fallback only. Calibrated S200062 uses robots/end_effector.py TCP instead.
    tool_offset: tuple[float, float, float] = (0.0, 0.0, -0.12)

    def validate(self) -> None:
        if self.name not in TASKS:
            raise ValueError(f"Unknown task {self.name!r}; choose {TASKS}")
        if self.control_mode not in ("whole-body", "arms-only"):
            raise ValueError("control_mode must be whole-body or arms-only.")
        if self.active_arm not in ("both", "left", "right"):
            raise ValueError("active_arm must be both, left or right.")
        if self.active_arm != "both" and (self.control_mode != "arms-only" or self.grasp_mode != "flap_top"
                or self.required_grasp_hands != 1 or self.grasp_hand != self.active_arm):
            raise ValueError("Single active_arm requires arms-only flap pick with the same single grasp_hand.")
        if self.control_mode == "arms-only" and self.name in ("approach_rack", "carry", "full"):
            raise ValueError(f"{self.name} requires base navigation; use whole-body or a stationary pick/place/press_button task.")
        if not self.box_names or len(set(self.box_names)) != len(self.box_names):
            raise ValueError("box_names must be nonempty and unique.")
        if self.required_grasp_hands not in (1, 2):
            raise ValueError("required_grasp_hands must be 1 or 2.")
        if self.grasp_mode not in ("body", "flap_top"):
            raise ValueError("grasp_mode must be body or flap_top.")
        if self.grasp_mode == "flap_top":
            if self.name != "pick" or self.control_mode != "arms-only":
                raise ValueError("flap_top currently requires a stationary arms-only pick.")
            if self.grasp_hand not in ("left", "right"):
                raise ValueError("grasp_hand must be left or right.")
            if (len(self.grasp_flaps) != 2 or len(set(self.grasp_flaps)) != 2
                    or any(n not in ("flap_front", "flap_back", "flap_left", "flap_right") for n in self.grasp_flaps)):
                raise ValueError("Choose two distinct flap body names shared by both hands.")
            if self.flap_contact_region not in ("surface", "top_band"):
                raise ValueError("flap_contact_region must be surface or top_band.")
            dimensions = (self.flap_grasp_depth, self.flap_top_band, self.flap_contact_margin,
                          self.flap_lock_degrees, self.unexpected_contact_limit, self.obstacle_contact_force,
                          self.prelift_position_scale, self.prelift_speed_scale, self.prelift_angular_scale,
                          self.prelift_rotation_scale, self.grasp_lift_clearance)
            if not all(math.isfinite(v) and v > 0 for v in dimensions):
                raise ValueError("Flap grasp/contact/lock settings must be finite and positive.")
            if not 0 <= self.grasp_contact_grace_s < self.hold_seconds:
                raise ValueError("Contact grace must be nonnegative and shorter than the success hold.")
            if not all(math.isfinite(v) and v > 0 for v in
                       (self.grasp_hold_slip_m, self.grasp_open_tolerance_m)):
                raise ValueError("Grasp hold slip/open tolerances must be finite and positive.")
            if self.flap_grasp_depth >= self.flap_top_band or self.flap_lock_degrees > 5:
                raise ValueError("Grasp depth must be inside top band; flap lock half-range must be <= 5 degrees.")
            if self.grasp_lift_clearance >= self.lift_height:
                raise ValueError("Grasp lift clearance must be smaller than the success lift height.")
            if not all(math.isfinite(v) and v >= 0 for v in (self.prelift_position_deadband,
                    self.prelift_speed_deadband, self.prelift_angular_deadband, self.prelift_rotation_deadband)):
                raise ValueError("Disturbance deadbands must be finite and nonnegative.")
            if not 0 <= self.prelift_grasp_scale <= 1 or not math.isfinite(self.prelift_penalty_cap) or self.prelift_penalty_cap <= 0:
                raise ValueError("Disturbance grasp scale must be in [0, 1] and cap positive and finite.")
        if len(self.finger_bodies) != 4:
            raise ValueError("Provide two opposing finger bodies per hand (left then right).")
        if not math.isfinite(self.reset_settle_seconds) or self.reset_settle_seconds < 0:
            raise ValueError("reset_settle_seconds must be finite and nonnegative.")
        if self.reset_settle_seconds:
            if self.grasp_mode != "flap_top":
                raise ValueError("Reset settling currently requires flap_top.")
            if (not math.isfinite(self.reset_settle_hold_seconds) or self.reset_settle_hold_seconds <= 0
                    or not math.isfinite(self.reset_settle_timeout) or self.reset_settle_timeout < 0
                    or self.reset_settle_seconds >= self.episode_length_s):
                raise ValueError("Settling needs a positive hold, nonnegative timeout and delay inside the episode.")
            if self.reset_settle_timeout and not (
                    self.reset_settle_seconds + self.reset_settle_hold_seconds < self.reset_settle_timeout < self.episode_length_s):
                raise ValueError("Positive settling timeout must follow delay+hold and lie inside the episode.")
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

    @property
    def grasp_hand_indices(self) -> tuple[int, ...]:
        return (0, 1) if self.required_grasp_hands == 2 else ((0,) if self.grasp_hand == "left" else (1,))


def task_spec(name: str, **overrides) -> TaskSpec:
    durations = dict(approach_rack=15.0, pick=15.0, carry=15.0, place=12.0, press_button=12.0, full=90.0)
    if name not in durations:
        raise ValueError(f"Unknown task {name!r}; choose {TASKS}")
    return replace(TaskSpec(name=name, episode_length_s=durations[name]), **overrides)

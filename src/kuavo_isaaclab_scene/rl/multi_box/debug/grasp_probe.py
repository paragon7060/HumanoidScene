"""Read-only grasp-success probe for Quest multi-box inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..success import (
    GraspSuccessInput,
    GraspSuccessResult,
    GraspSuccessTracker,
    PinchEvidence,
    RelativePoseStabilityTracker,
    classify_pinches,
)


# Initial value from the recorded success/failure Quest runs; recheck after
# fingertip-pad changes.  This is per jaw, not the commanded 50 N per gripper.
MIN_JAW_FORCE_N = 5.0


@dataclass(frozen=True)
class GraspProbeResult:
    pinch: PinchEvidence
    stable_hands: torch.Tensor
    success: GraspSuccessResult
    rack_clearance_m: torch.Tensor

    def report(self) -> str:
        hands = "/".join(str(int(v)) for v in self.pinch.hand_pinching[0].tolist())
        flaps = "/".join(str(int(v)) for v in self.pinch.hand_flap_index[0].tolist())
        stable = "/".join(str(int(v)) for v in self.stable_hands[0].tolist())
        return (
            f"GRASP PROBE | pinch L/R={hands} flap-index R=0 L=1: {flaps} "
            f"stable={stable} clearance={float(self.rack_clearance_m[0]):.3f}m "
            f"hold={float(self.success.hold_time_s[0]):.2f}s "
            f"success={int(self.success.success[0])}"
        )


class QuestGraspProbe:
    """Tracks one selected box without changing reward, control, or phase."""

    def __init__(self, device: str | torch.device, num_envs: int = 1):
        self.pose = RelativePoseStabilityTracker(num_envs, device)
        self.grasp = GraspSuccessTracker(num_envs, device)
        self.stability_armed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.target_logical_id = -1

    def reset(self) -> None:
        self.pose.reset()
        self.grasp.reset()
        self.stability_armed.zero_()
        self.target_logical_id = -1

    def update(self, snapshot, dt: float) -> GraspProbeResult:
        if snapshot.target_logical_id != self.target_logical_id:
            self.reset()
            self.target_logical_id = snapshot.target_logical_id
        pinch = classify_pinches(snapshot.contacts, min_jaw_force_n=MIN_JAW_FORCE_N)
        # Extraction from the sloped shelf naturally changes the hand-box pose
        # after first contact.  Start the stability reference only once the
        # approved physical grasp is complete and the box has cleared the rack;
        # stability then measures slip during the 0.25 s proof hold rather than
        # treating the extraction motion itself as slip.
        valid_flaps = ((pinch.hand_flap_index >= 0)
                       & (pinch.hand_flap_index < 2)).all(dim=-1)
        opposing = valid_flaps & (
            pinch.hand_flap_index[:, 0] != pinch.hand_flap_index[:, 1])
        ready_for_stability = (
            pinch.hand_pinching.all(dim=-1)
            & opposing
            & (snapshot.rack_clearance_m >= self.grasp.config.proof_lift_m)
        )
        # Proof lift only arms the hand-box stability reference.  Once armed,
        # keep measuring relative-pose drift while the physical opposing-flap
        # pinch remains valid.  Requiring rack clearance on every frame would
        # invalidate a correct carry as soon as the box is lowered toward the
        # conveyor, which is below the rack.
        self.stability_armed |= ready_for_stability
        track_stability = (
            pinch.hand_pinching.all(dim=-1)
            & opposing
            & self.stability_armed
        )
        stable = self.pose.update(
            snapshot.hand_to_box_pose,
            pinch.hand_pinching & track_stability[:, None],
        )
        result = self.grasp.update(GraspSuccessInput(
            hand_pinching=pinch.hand_pinching,
            hand_flap_index=pinch.hand_flap_index,
            relative_pose_stable=stable,
            rack_clearance_m=snapshot.rack_clearance_m,
        ), dt)
        return GraspProbeResult(pinch, stable, result, snapshot.rack_clearance_m)

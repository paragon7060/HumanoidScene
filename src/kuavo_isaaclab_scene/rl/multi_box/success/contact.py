"""Convert filtered finger-to-flap contacts into privileged pinch evidence.

The input order is [environment, hand (left/right), flap (right/left),
jaw (front/back)].  This module does not create policy observations or rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FingerFlapContacts:
    force_n: torch.Tensor
    in_region: torch.Tensor
    opposed: torch.Tensor
    available: torch.Tensor

    def validate(self) -> None:
        if self.force_n.ndim != 4 or self.force_n.shape[1:] != (2, 2, 2):
            raise ValueError("force_n must have shape [num_envs, 2 hands, 2 flaps, 2 jaws].")
        if self.in_region.shape != self.force_n.shape or self.in_region.dtype != torch.bool:
            raise ValueError("in_region must be boolean with the force shape.")
        if self.opposed.shape != self.force_n.shape[:3] or self.opposed.dtype != torch.bool:
            raise ValueError("opposed must be boolean [num_envs, 2 hands, 2 flaps].")
        if self.available.shape != self.force_n.shape[:1] or self.available.dtype != torch.bool:
            raise ValueError("available must be boolean [num_envs].")
        if not self.force_n.is_floating_point():
            raise TypeError("force_n must be floating point.")
        if len({x.device for x in (self.force_n, self.in_region, self.opposed, self.available)}) != 1:
            raise ValueError("All contact measurements must share one device.")


@dataclass(frozen=True)
class PinchEvidence:
    hand_pinching: torch.Tensor
    hand_flap_index: torch.Tensor
    qualified_flaps: torch.Tensor
    ambiguous_hands: torch.Tensor


def classify_pinches(contacts: FingerFlapContacts, *, min_jaw_force_n: float) -> PinchEvidence:
    """Accept exactly one flap per hand, with two valid and opposed jaw contacts.

    A hand touching both flaps is ambiguous and cannot establish a grasp of
    either.  Force is checked per jaw rather than as a sum, so one strong jaw
    cannot hide a missing opposing contact.
    """
    contacts.validate()
    if not math.isfinite(min_jaw_force_n) or min_jaw_force_n <= 0:
        raise ValueError("min_jaw_force_n must be finite and positive.")
    qualified = (
        contacts.available[:, None, None]
        & contacts.in_region.all(dim=-1)
        & contacts.opposed
        & torch.isfinite(contacts.force_n).all(dim=-1)
        & (contacts.force_n >= min_jaw_force_n).all(dim=-1)
    )
    count = qualified.sum(dim=-1)
    pinching = count == 1
    selected = torch.where(
        pinching, qualified.to(torch.long).argmax(dim=-1),
        torch.full_like(count, -1, dtype=torch.long),
    )
    return PinchEvidence(pinching, selected, qualified, count > 1)

"""Term-level sidecar logging for VR reward calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from ..rewards import (
    CarryRewardInput,
    CommonRewardInput,
    GraspRewardInput,
    MultiBoxRewardModel,
    PlaceRewardInput,
    RewardBreakdown,
)


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Shadow logger accepts one environment at a time.")
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Shadow reward logs must contain finite values.")
    return result


@dataclass
class ShadowRewardStats:
    steps: int = 0
    total_return: float = 0.0
    term_returns: dict[str, float] = field(default_factory=dict)
    term_abs_max: dict[str, float] = field(default_factory=dict)

    def update(self, breakdown: RewardBreakdown) -> None:
        self.steps += 1
        self.total_return += _scalar(breakdown.total)
        for name, value in breakdown.terms.items():
            scalar = _scalar(value)
            self.term_returns[name] = self.term_returns.get(name, 0.0) + scalar
            self.term_abs_max[name] = max(self.term_abs_max.get(name, 0.0), abs(scalar))

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "total_return": self.total_return,
            "term_returns": dict(self.term_returns),
            "term_abs_max": dict(self.term_abs_max),
        }


def format_shadow_reward(phase: str, raw: Mapping[str, object], breakdown: RewardBreakdown,
                         stats: ShadowRewardStats) -> str:
    weighted = sorted(
        ((name, _scalar(value)) for name, value in breakdown.terms.items()),
        key=lambda pair: abs(pair[1]), reverse=True)
    top = ", ".join(f"{name}={value:+.3f}" for name, value in weighted[:5])
    raw_text = ", ".join(f"{name}={_scalar(value):.3f}" for name, value in raw.items())
    return (f"V2 SHADOW | {phase.upper()} | step={stats.steps}\n"
            f"step_reward={_scalar(breakdown.total):+.3f} return={stats.total_return:+.3f}\n"
            f"terms: {top}\nraw: {raw_text}")


class ShadowRewardLogger:
    """Append-only JSONL sidecar; dataset observations/actions stay untouched."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self.stats = ShadowRewardStats()

    def record(self, *, step: int, sim_time_s: float, phase: str,
               raw: Mapping[str, object], potentials: Mapping[str, object],
               breakdown: RewardBreakdown, events: Mapping[str, object]) -> None:
        self.stats.update(breakdown)
        row = {
            "schema": "multi_box_v2_shadow_reward_v1",
            "step": int(step),
            "sim_time_s": _scalar(sim_time_s),
            "phase": phase,
            "raw": {name: _scalar(value) for name, value in raw.items()},
            "potentials": {name: _scalar(value) for name, value in potentials.items()},
            "events": {name: bool(_scalar(value)) for name, value in events.items()},
            "weighted_terms": {name: _scalar(value) for name, value in breakdown.terms.items()},
            "step_reward": _scalar(breakdown.total),
            "return": self.stats.total_return,
        }
        self._stream.write(json.dumps(row, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PoseShadowRewardEvaluator:
    """Evaluate dense v2 terms without inventing contact or collision events.

    The selected phase is operator-controlled during VR inspection.  Event
    rewards and common penalties stay zero until their real signal adapters
    exist, which keeps the resulting sidecar honest and easy to audit.
    """

    PHASES = ("grasp", "carry", "place")

    def __init__(self, phase: str = "grasp", model=None):
        self.model = model or MultiBoxRewardModel()
        self.phase = ""
        self.previous: dict[str, torch.Tensor] | None = None
        self.set_phase(phase)

    def set_phase(self, phase: str) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"Shadow phase must be one of {self.PHASES}.")
        if phase != self.phase:
            self.phase = phase
            self.previous = None

    def reset(self) -> None:
        self.previous = None

    @staticmethod
    def _common(reference: torch.Tensor) -> CommonRewardInput:
        zero = torch.zeros_like(reference)
        event = torch.zeros_like(reference, dtype=torch.bool)
        return CommonRewardInput(
            robot_rack_collision_event=event,
            self_collision_event=event.clone(),
            box_drop_event=event.clone(),
            obstacle_collision_event=event.clone(),
            normalized_base_motion=zero,
            normalized_action_rate=zero.clone(),
            normalized_joint_limit=zero.clone(),
        )

    def evaluate(self, potentials: Mapping[str, torch.Tensor]) -> RewardBreakdown:
        current = {name: value for name, value in potentials.items()}
        if not current:
            raise ValueError("Shadow evaluation requires normalized potentials.")
        if self.previous is None:
            self.previous = {name: value.detach().clone() for name, value in current.items()}
        previous = self.previous
        reference = next(iter(current.values()))
        event = torch.zeros_like(reference, dtype=torch.bool)
        common = self._common(reference)
        if self.phase == "grasp":
            result = self.model.grasp(GraspRewardInput(
                previous_approach=previous["approach"], approach=current["approach"],
                previous_alignment=previous["alignment"], alignment=current["alignment"],
                previous_capture=previous["capture"], capture=current["capture"],
                previous_proof_lift=previous["proof_lift"], proof_lift=current["proof_lift"],
                bilateral_pinch_event=event, success_event=event.clone(), common=common,
            ))
        elif self.phase == "carry":
            result = self.model.carry(CarryRewardInput(
                previous_extraction=previous["extraction"], extraction=current["extraction"],
                previous_belt=previous["belt"], belt=current["belt"],
                previous_free_space=previous["free_space"], free_space=current["free_space"],
                previous_pre_place_height=previous["pre_place_height"],
                pre_place_height=current["pre_place_height"],
                grasp_loss_event=event, placed_box_disturbance_event=event.clone(),
                success_event=event.clone(), common=common,
            ))
        else:
            result = self.model.place(PlaceRewardInput(
                previous_footprint=previous["footprint"], footprint=current["footprint"],
                previous_alignment=previous["alignment"], alignment=current["alignment"],
                previous_free_space=previous["free_space"], free_space=current["free_space"],
                previous_descent=previous["descent"], descent=current["descent"],
                previous_stability=previous["stability"], stability=current["stability"],
                support_event=event, correct_release_event=event.clone(),
                premature_release_event=event.clone(), success_event=event.clone(), common=common,
            ))
        self.previous = {name: value.detach().clone() for name, value in current.items()}
        return result

"""Independent schedules for recorded VR demos and live teacher corrections."""

from __future__ import annotations

import math


TEACHER_SCHEDULE_ARGUMENTS = (
    "teacher_batch_fraction", "teacher_min_batch_fraction", "teacher_bc_strength",
    "teacher_decay_updates", "online_ik_min_episode_fraction", "online_ik_decay_updates",
)


def add_teacher_schedule_arguments(parser):
    parser.add_argument("--teacher-batch-fraction", type=float, default=0.0,
                        help="Independent actor-only live teacher batch fraction; zero preserves legacy mixed BC.")
    parser.add_argument("--teacher-min-batch-fraction", type=float, default=0.0,
                        help="Retain teacher correction after recorded VR imitation retires.")
    parser.add_argument("--teacher-bc-strength", type=float, default=10.0)
    parser.add_argument("--teacher-decay-updates", type=int, default=128_000,
                        help="Explicit actor-update horizon, preserved across pilot and full-run resumes.")
    parser.add_argument("--online-ik-min-episode-fraction", type=float, default=0.0)
    parser.add_argument("--online-ik-decay-updates", type=int, default=0,
                        help="Independent expert horizon; zero uses the recorded-demo horizon.")


def validate_teacher_schedule(args):
    teacher_fraction(args, 0)
    if not math.isfinite(args.teacher_bc_strength) or not 0 <= args.teacher_bc_strength <= 1000:
        raise ValueError("Teacher BC strength must be finite and in [0, 1000]")
    if args.teacher_batch_fraction and not (args.online_teacher_labels and args.demo_dataset):
        raise ValueError("Independent teacher imitation requires online teacher labels and IK demos")
    if args.online_ik_decay_updates < 0:
        raise ValueError("Online IK update horizon must be nonnegative")
    imitation_fraction(args.online_ik_episode_fraction, args.online_ik_min_episode_fraction,
                       0, max(1, args.online_ik_decay_updates))


def imitation_fraction(initial: float, minimum: float, updates: int, horizon: int) -> float:
    """Linear decay with a floor, measured in actual actor updates across resumes."""
    if (not math.isfinite(initial) or not math.isfinite(minimum)
            or not 0 <= minimum <= initial < 1 or horizon < 1 or updates < 0):
        raise ValueError("Invalid imitation schedule")
    return max(minimum, initial * max(0.0, 1.0 - updates / horizon))


def teacher_fraction(args, actor_updates: int) -> float:
    return imitation_fraction(
        getattr(args, "teacher_batch_fraction", 0.0),
        getattr(args, "teacher_min_batch_fraction", 0.0), actor_updates,
        getattr(args, "teacher_decay_updates", 128_000))

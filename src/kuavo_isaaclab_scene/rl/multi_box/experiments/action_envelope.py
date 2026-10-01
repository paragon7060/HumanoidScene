"""Explicit body-command ablation for measured v2 replay, not a SAC fix.

The envelope contains every body command in the410-step measured success.
It leaves arm/gripper/head commands unchanged. Clipping a deployed actor is
only a diagnostic: it must not silently alter the training distribution.
"""

import torch


def diagnostic_body_limits(action_terms, *, device="cpu", dtype=torch.float32):
    expected = {"base": 3, "upper_body": 15, "height": 2,
                "left_gripper": 1, "right_gripper": 1, "head": 2}
    terms = list(action_terms)
    if len(terms) != len(expected) or dict(terms) != expected:
        raise ValueError("Body-envelope diagnostic requires the24-D upright v2 action contract")
    by_term = {"base": [.2, .2, .4], "upper_body": [.35] + [1.] * 14,
               "height": [.2, .2], "left_gripper": [1.],
               "right_gripper": [1.], "head": [1., 1.]}
    return torch.tensor([value for name, _ in terms for value in by_term[name]],
                        device=device, dtype=dtype)

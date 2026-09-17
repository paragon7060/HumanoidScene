# Initial reward ratios

All dense potentials are normalized to `[0, 1]` and use
`gamma * Phi(next) - Phi(current)`.  This prevents a policy from earning the
same positive shaping reward by holding a static pose.  Event inputs must be
one-step pulses; success state itself must not be passed every step.

The initial ratio is:

| Scope | Dense total | Pinch/support/release events | Success | Major failure |
|---|---:|---:|---:|---:|
| Grasp | 1.50 | 1.00 | 3.00 | drop 8.00 |
| Carry | 2.50 | — | 4.00 | grasp loss 3.00, drop 8.00 |
| Place | 3.00 | 1.50 | 5.00 | premature release 2.50, drop 8.00 |
| High level | — | first placement 5.00 | full task 12.00 | failure 8.00 |

Common per-control-step regularization uses `0.002 * normalized_base_motion`,
`0.001 * normalized_action_rate`, and `0.002 * normalized_joint_limit`.
Collision events cost 4-6 and are also expected to drive termination when the
separate failure predicate says so.  These are starting ratios to tune from
term distributions after the first short training run, not hidden task logic.

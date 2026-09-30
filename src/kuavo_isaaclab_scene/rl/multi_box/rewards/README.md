# V2 reward ratios

Dense grasp geometry scores are normalized to `[0, 1]`. Front staging, approach, capture,
jaw gap and lift use `gamma * Phi(next) - Phi(current)`; alignment uses the
signed change in closing-axis score multiplied by near-flap proximity. Holding
a static pose earns no positive progress. Event inputs must be one-step pulses;
success state itself must not be passed every step.

The initial ratio is:

| Scope | Dense total | Pinch/support/release events | Success | Major failure |
|---|---:|---:|---:|---:|
| Grasp | 4.40 | first one-hand pinch 2.00, opposing pair 1.00 | 5.00 | drop/workspace 8.00 |
| Carry | 2.50 | — | 4.00 | grasp loss 3.00, drop 8.00 |
| Place | 3.00 | 1.50 | 5.00 | premature release 2.50, drop 8.00 |
| High level | — | first placement 5.00 | full task 12.00 | failure 8.00 |

Grasp overrides base/action-rate regularization to `0.0002`/`0.0001` and adds
a per-step premature-close cost capped at `0.002`. Other skills use common
`0.002 * normalized_base_motion`, `0.001 * normalized_action_rate`, and
`0.002 * normalized_joint_limit`.
Grasp also pays `0.002 * (1 - front_staging)` and
`0.002 * (1 - approach)` per task step. Each score remains in `[0, 1]`, so
these costs together cannot exceed 3.6 over a 30-second episode at 30 Hz;
one rack collision costs 6. They create a persistent incentive to approach
before a first success without making a deliberate collision cheaper than
waiting for the episode limit.
Collision events cost 4-6 and are also expected to drive termination when the
separate failure predicate says so.  These are starting ratios to tune from
term distributions after the first short training run, not hidden task logic.

For grasp, each hand is assigned a different flap. The front-staging score
(weight `1.0`, scale `0.80 m`) measures lateral/height distance to its assigned
flap lane (with `0.10 m` X/Z tolerance) and distance to a plane `0.08 m` outside
the rack front. Depth stops
contributing once the hand reaches that plane, so moving inward to pinch is
not penalized by this term. The flap-surface approach score (weight `2.0`)
uses a `0.22 m` distance scale; capture and alignment still focus the final
few centimeters. The original `0.083 m` scale still selects the flap pairing,
preserving the deployable observation's assignment rule. Rack collisions
remain terminal failures. SAC records
per-hand flap/front distances, the fraction under `0.10 m`, and positive and
absolute progress so signed averages do not hide movement.

Existing Quest replay files contain rewards from the earlier geometry and
three-axis torso dynamics. SAC now uses them only for actor imitation: 20% of
the actor batch and a `10 * fraction` imitation weight at the start of learning, declining
linearly to zero over the first 30% of planned updates. The critic and online
replay use only current-environment rewards. The gripper gap shaping weight is
zero; closing earns a positive grasp reward only through verified pinch events.
An optional demo-guided warmup begins with actor-only behavior cloning and
keeps the grippers open outside 0.12 m from their assigned flap centers.
The recovery profile freezes actor normalization after fitting demo and initial
reset observations, routes the selected box into one actor feature slot,
starts Gaussian exploration at standard deviation 0.15 (cap 0.3) and alpha 0.001
(floor 0.00001), and scales rewards by 10 inside SAC. Its critic target omits
the future entropy bonus. A protected online buffer retains actual transitions
within 0.25 m of either flap or with a pinch; 25% of critic samples come from
that buffer when available. No legacy recorded reward is inserted into it.
Optional IK warmup collects current-environment entry actions, fits the actor
once to this broader data before unguided rollout, and then fixes its revised
normalizer. These real transitions may train Q. Logs separate IK warmup from
SAC rollout and report pinch, proof lift, stability and continuous hold time.
A critic-only warmup (500 updates) protects the initial actor from an untrained
Q gradient. Online actor learning uses 0.00003 rather than the critic's 0.0003.
Held-success terminal transitions have a separate checkpointed CPU replay
(10,000 rows, up to 5% of the critic batch); no legacy reward enters it.
Imitation decay counts actor updates, preserving the initial 20% fraction
through critic warmup. The evaluated conservative profile uses BC strength 100
and Gaussian std 0.02/cap 0.05; these explicit overrides are in the run record.

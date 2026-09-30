# Multi-box v2 SAC: safe rack approach correction

The GPU 3 SAC run `sac_mbv2_guided_gpu3_20260928_233436` reached 860
iterations without a bilateral pinch or success. The median hand-to-flap
distance stayed near 1.2 m. In its first 100 iterations there were 3,345
unsafe terminations and 3,356 timeouts; in the last 100 observed during this
review there were about 200 unsafe terminations and 5,214 timeouts. The policy
learned to avoid contact while remaining far from both flaps.

At a typical 1.1 m hand-to-front-stage distance, the previous 0.25 m
exponential scale yielded a potential of about 0.012. The new 0.80 m scale
yields about 0.253. Its target remains the assigned flap's lane **in front of
the rack**; the depth term saturates at the safe front plane, so entering the
rack gives no extra front-stage reward. The near-flap approach, pinch and
success predicates and collision thresholds are unchanged. This is a reward
correction, not a curriculum.

Each SAC iteration now records eligible rack and obstacle failures by the
nearest hand's front-stage distance bands (up to 0.25, 0.5, 1, and 2 m) and
how often both hands reach 0.25 m. Finite hand/flap distances beyond 5 m are
excluded from replay and reward/distance averages and counted as
`implausible_distance_transitions`. This prevents rare finite simulator pose
explosions from contaminating replay. The earlier run had one iteration with
an 8.5-million-metre reported hand/flap distance; its median distance remained
near 1.2 m.

Focused CPU tests passed (34 tests). A four-environment GPU 0 Isaac smoke run
completed one SAC update, produced the new front-stage bins and reward
breakdown, and exited successfully. These checks establish runtime wiring,
not learning success. The next full run should compare front-stage occupancy,
distance-conditioned rack collision rates, hand-to-flap distance, pinch and
success against the prior run before claiming improvement.

The full follow-up run was started on GPU 3 at 2026-09-29 23:09 KST with
2,048 environments in `artifacts/rl/drive_runs/sac_mbv2_safe_front_gpu3_20260929_230917`.
It uses the two available successful demonstrations, 20% initial demo sampling
decaying over the first 30% of training, a 2.5-million-transition CPU replay
buffer, and 20-iteration checkpoints. The existing authenticated connection backs
up the unique run directory to `HumanoidScene-RL` every five minutes and keeps
the latest two verified checkpoints locally. The preceding GPU 3 run was
stopped and its final Drive upload was verified before this run started. At
the time of this report, Isaac initialization was still underway; this launch
does not establish that the policy has improved.

# Multi-box v2 SAC recovery experiments — 2026-09-30

## Latest measured result and next run

**2026-10-01 03:37 KST: the critical-retention run was stopped deliberately at
iteration 133 because SAC learning diverged, not because grasp was solved.**
It collected 2,583,590 valid transitions during this resume and 24 additional
online IK expert successes, with **zero SAC-from-reset/handoff successes** and
zero numerical-recovery episodes. The existing 16 plus 24 new expert successes
produce 2,560 protected 64-step success-tail transitions. It stopped cleanly
with exit code 0; final checkpoints/logs are Drive checksum-verified.

![Measured critic divergence and success attribution](assets/rl_v2_q_divergence_20261001.png)

The mean rollout reward stayed negative, while mean actor Q grew to 539,442,
Q loss to 3.672 billion and temperature alpha to 8.756. On the **same stored
executed success transitions**, checkpoint 50 predicted mean Q 655.8,
checkpoint 100 predicted 215,894.9, and checkpoint 133 predicted 1,744,840.4;
the latter valued its own unexecuted actions still higher (1,813,365.3).
Actor action MSE against these measured actions also grew from 0.414 to 0.828.
This is evidence of critic/policy instability and imitation being overwhelmed,
not genuine grasp improvement. The critical-label retention fix worked, but
did not address this separate SAC failure.

The follow-up separates three corrections:

- Add LayerNorm to both critic hidden layers for the fresh v2 learner.
- Normalize the actor's Q term by detached `max(mean(abs(Q)), 1)`, so growing
  Q scale cannot silently erase the configured imitation penalty. This adapts
  the Q-scale idea in the [TD3+BC authors' implementation](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)
  to this SAC actor; it is not a conversion to TD3 or a claim of success.
- Make the capped-policy entropy target depend on the tanh Jacobian at the
  current detached mean, including saturated teacher actions. Gaussian std
  feasibility alone was insufficient. For `std <= 1`, the per-dimension
  target includes `log(1-tanh(mean)^2) - max_std^2`; the existing 0.5-nat
  margin remains. Add an explicit `--max-alpha` (v2 default 0.001).

`--experience-checkpoint` starts a **fresh model/Q/optimizer** and imports only
the protected genuine executed success-tail transitions from a compatible
checkpoint. It cannot be combined with `--checkpoint`. Physical action,
observation, reward, contacts and safety contracts must match; learning settings
can change for data-only imports. Actor-only teacher labels remain a separate
`--teacher-label-checkpoint` input. Full policy resume still requires matching
learning/architecture contracts. New diagnostics include `actor_q_scale`,
`target_entropy_mean` and `imported_success_rows`.

Regression checks: 69 SAC/collection/alternative CPU tests passed, covering saturated-mean
entropy, temperature bounds, Q normalization and genuine experience import.
A fixed-success-replay GPU 3 comparison completed 5,000 updates per case:

| Metric at update 5,000 | Previous critic/raw-Q actor | LayerNorm/scaled-Q actor |
| --- | ---: | ---: |
| Q loss | 148,002.34 | 0.9986 |
| Mean policy Q | 12,788.84 | 89.56 |
| Teacher action MSE | 0.05223 | 0.001144 |
| Alpha | 0.000004418 | 0.000002636 |

![Fixed executed-success-replay comparison](assets/rl_v2_q_stability_probe_20261001.png)

Both cases use a fresh critic, the same 2,560 real success-tail transitions,
same actor initialization fit (1,000 updates, MSE 0.000683), revised squash-aware
entropy target, and alpha bound. The previous case uses raw Q and BC loss weight
200; the new combination uses normalized Q and BC weight 2 (20% times strength
10). Thus this is a combined stabilization comparison, not an ablation proving
LayerNorm alone. Alpha stays small in both, yet the previous critic still grows;
temperature alone cannot explain Q divergence. Only success-tail states are
covered and the probe omits live simulation and the gripper action projection.
It checks numerical behavior, **not SAC grasp success**.

Before another long run, a distinct bounded live GPU 3 execution will check
128 environments, 120 iterations, batch 1,024, four updates/vector step, and
300,000 CUDA replay transitions. It imports real success tails from stopped
checkpoint 133 and critical teacher labels from the immutable label seed,
without restoring their Q/model/optimizer. Teacher fit is 20,000; initial
expert/imitation fractions remain 20%. For the bounded run only, decay spans
its full planned update horizon rather than the default first 30%, so it does
not remove imitation during this short verification. Safety and reset/success
conditions remain unchanged. Long-run scheduling will be recorded separately
after measuring this live check.

### Previous launch (01:42 KST)

Latest 2026-10-01 01:42 KST: source `bf63054` resumed on GPU 3 / 1,024
environments in `artifacts/rl/drive_runs/sac_mbv2_critical_retention_gpu3_20261001_0141/`,
child `sac_20261001_014208_f5427a`. It restores policy checkpoint 51 and imports
actor-only labels exported from checkpoint 29, with persistent critical-label
storage described below. Additional iterations: 949, targeting iteration 1,000.
The fitted actor, Q/optimizers and imitation-decay counter are preserved. Uniform
Q replay refills until 100,000 valid transitions; another teacher fit is not run.

The previous run stopped cleanly at iteration 51, with 10,785 critic / 6,785
actor updates, **nine new online IK held successes**, zero SAC-from-reset or
SAC-handoff successes, and zero numerical failures across this resume. Its
protected replay now contains 1,024 actual success-tail transitions (16 successes
including the seven restored). Checkpoints and final logs are Drive verified;
local checkpoints 50/51 remain. Source 29's labels were recovered from its
verified Drive checkpoint into a distinct immutable actor-only seed artifact,
without policy, optimizer, reward or Q fields. All unrelated user processes
remained untouched. Before/after fit videos and photos below are also attached
natively in [the Notion experiment record](https://app.notion.com/p/3eb63918d42a81f694a9e060aeffe7cc).

At 01:55 KST, startup/reset settling completed (87 control steps), model and
optimizers restored, and the actor-only import log confirmed 100,000 imported
labels and 100,000 retained critical rows. Label-row counts can include repeated
snapshot samples; they are not counts of independent grasps. Uniform replay
refill is active. Learned SAC grasp success remains unverified.

The first fresh 1,024-env GPU 3 run
`artifacts/rl/drive_runs/sac_mbv2_success_tail_gpu3_20260930_2238/`
ended at IK warmup iteration 18 with `ValueError: Non-finite robot gravity
compensation or stiffness`. It had collected 589,117 valid transitions,
46,555 priority teacher labels, zero held successes and 48 unsafe terminations.
Its last mean flap distances were 0.415/0.404 m. **SAC optimization had not
started.** This was not an OOM: own GPU use was about 55,513 MiB. Final logs
were Drive checksum-verified. No checkpoint existed yet because regular saves
were every 50 iterations; its in-memory replay/labels could not be recovered.

The recovery now includes std-cap-compatible entropy targets and opt-in
per-environment dynamics recovery, described below. A real two-env GPU 3
fault-injection probe passed: env 0 alone was reset after injected NaN gravity;
env 1 remained ready, next-step observations/feedforward were finite, and the
failed transition was terminated and excluded. CPU coverage: 82 relevant
SAC/demo/gravity/terminal checks passed.

The new distinct fresh GPU 3 run started from source `2b627a8` in
`artifacts/rl/drive_runs/sac_mbv2_dynamics_recovery_gpu3_20260930_2341/`,
with child `sac_20260930_234237_dac410`. It retains 1,024 environments, 900 IK warmup
vector steps, 3,000,000-transition CUDA replay (22.50 GiB), batch 4,096 and 16
updates/vector step, teacher fit 20,000, 1,000 iterations, initial 20% expert/
imitation fractions decaying over 153,600 actor updates, and save interval 50.
One additional checkpoint is saved when warmup completes; an environment-step
exception also attempts an atomic recovery checkpoint. These protect initial
collection without increasing ordinary checkpoint frequency. Safety, reset,
bilateral grasp/proof-lift success and the no-curriculum contract remain unchanged.

Measured 2026-10-01: this run finished iteration 20 with 652,258 valid
transitions and **five IK warmup held successes**. It then failed during
iteration 21: env 1007 had non-finite root state and Coriolis compensation,
although gravity, mass matrices and joint state remained finite. Partial
reset ran, but strict grasp geometry still encountered an invalid TCP
quaternion. The atomic emergency checkpoint succeeded: checkpoint 21 contains
100,000 teacher labels and **448 genuine success-tail transitions**, representing
seven successes including two in the incomplete iteration. Optimizer/actor
updates are both zero; these are expert collection successes, not SAC success.
The checkpoint and final logs are Drive checksum-verified.

The follow-up refreshes PhysX articulation kinematics after reset and
invalidates cached robot root/link views without advancing simulation time.
A pre-grasp guard also checks all robot link poses, root and joint state after
the last physics substep, covering errors with no subsequent actuator write.
An actual GPU 3 two-environment probe passed both NaN-gravity and final-substep
zero-link-quaternion/non-finite-root-cache injections: only env 0 reset,
env 1 remained ready, observations/feedforward were finite, and the failed
transition was excluded. Corrupt read caches were injected; NaNs were never
written to native physics. Relevant CPU checks: 41 passed. This verifies the
recovery path, not the underlying source of rare native root-state divergence.

Resumed at 2026-10-01 00:39 KST from source `b4a5306` in
`artifacts/rl/drive_runs/sac_mbv2_pose_recovery_gpu3_20261001_0039/`,
child `sac_20261001_003940_9e46f4`. The previous failed training PID is gone
and GPU 3 was empty before launch. CUDA visibility is restricted to physical
GPU 3; 979 additional iterations target original iteration 1,000.

This restores checkpoint 21 with 1,024 environments and 256 IK vector
steps to refill non-serialized uniform Q replay, then the pending 20,000-update
teacher fit and SAC optimization. Successful tails, labels and the original
153,600 actor-update imitation decay horizon are restored. Safety, rewards,
success criteria and reset distribution are unchanged.
The supervisor uses the existing authenticated Drive connection, checks every
300 seconds, saves every 50 iterations plus warmup/exception safeguards, and
keeps the latest two checksum-verified checkpoints. Local free space was
109 GiB at launch; verified Drive free capacity was 4.984 TiB.

Measured after refill: iteration 29 completed teacher fit (20,000 updates),
reducing action MSE from 0.13151 to 0.00080953. This checkpoint has 16 critic
updates and zero SAC actor updates. At iteration 32, critic updates reached
1,552; numerical failures remain zero, successful tails remain 448, and
SAC hand-to-flap distances were 0.603/0.522 m. The critic-only warmup deliberately
holds the fitted actor for the first 4,000 optimizer updates. Action MSE and
short-range progress are not held grasp-success evidence.

### Current-contract video before teacher fit

[![Checkpoint 21 terminal before reset: distance 105.7 cm, pinch 0, unsafe 1](assets/rl_v2_teacher_before_terminal_20261001.png)](assets/rl_v2_teacher_before_20261001.mp4)

[H.264 video: current-contract checkpoint 21 before teacher fit](assets/rl_v2_teacher_before_20261001.mp4).
Same seed 42, one environment, S63/Leju, rack-rollers and current 464/530/24
observation/action contract. Deterministic policy ended at control step 84
(2.8 seconds), zero grasp success and one unsafe termination. The checkpoint
contains Quest BC initialization but **no online teacher fit or SAC updates**.
Photos and video are captured after termination computation but **before reset**;
the terminal image is not the respawned state. They render CPU USD meshes at
actual PhysX poses. This view alone does not identify the specific unsafe cause.
The terminal mean flap distance was 105.7 cm.

### Same-contract video after teacher fit

[![Checkpoint 29 at 9.5 seconds: distance 73.7 cm, pinch 0, unsafe 0](assets/rl_v2_teacher_after_late_20261001.png)](assets/rl_v2_teacher_after_20261001.mp4)

[H.264 video after teacher fit](assets/rl_v2_teacher_after_20261001.mp4).
Same seed 42 and physical/control configuration. The deterministic rollout
completed 300 control steps / 10 seconds without unsafe, invalid-reset or timeout
termination. Held success remained zero. The photo is the last sampled frame
at 9.5 seconds; its mean flap distance is 73.7 cm, with no bilateral pinch.
Checkpoint 29 contains the completed teacher fit and 16 critic updates, but
zero SAC actor updates. This comparison establishes longer safe motion in one
case, not grasp success. Endpoint distances are at different times (2.8 versus
9.5 seconds), so they are not a paired distance-improvement estimate.

### Protect critical teacher labels from FIFO churn

The resumed run reached iteration 38 with 4,493 critic / 493 actor updates,
zero numerical failures and zero SAC-from-reset successes. A separate data
issue became visible: critical rows in its ordinary 500,000-label teacher FIFO
fell from 56,011 to 13,109 as newer off-target labels overwrote them. At 1,024
environments and 30 Hz this FIFO covers only about 16.3 simulated seconds,
shorter than a 30-second episode. Priority **sampling** did not protect
priority **storage**. Episode ages and resets also affect mean distance, so
the label drop alone does not establish the cause of a distance increase.

The fix adds a separate CPU FIFO of 100,000 critical actor-only labels. New
off-target labels never overwrite that stratum; new critical labels replace
older critical labels when it fills. Half each teacher batch still comes from
critical labels, and the existing total 20% initial imitation and decay schedule
are unchanged. Extra CPU storage is about 186 MiB with this observation/action
contract, with no additional CUDA replay allocation. Checkpoint snapshots remain
bounded to 100,000 labels, so ordinary checkpoint size does not increase.

`--teacher-label-checkpoint` optionally restores actor-only labels from an older
compatible v2 checkpoint while policy/optimizer/Q restoration uses `--checkpoint`.
The source manifest must match the environment/observation/action contract;
labels must have matching dimensions and finite normalized actions. This path
never imports the source policy, optimizer, reward or Q transitions. The next
resume uses checkpoint 29 as its label source, retaining the more recent policy
checkpoint. CPU tests cover retention through FIFO overwrite, bounded snapshots,
restore, and rejection of incompatible/non-finite labels; 66 relevant checks pass.

![IK successes, distances and outcomes in the corrected run](assets/rl_v2_ik_success_progress_20261001.png)

The earlier 64-env episodic-guidance run stopped cleanly at iteration 193.
Online IK episodes produced **two new held successes**, while SAC from reset
produced **zero**. Its final checkpoint/log upload is checksum-verified after
the user reauthenticated the existing Drive connection. Expert successes
establish physical feasibility, not learned SAC success.

The data corrections carried into the new run:

- Preserve the last 64 executed, valid transitions of successful episodes
  (about 2.13 s), ending history at each reset or excluded transition.
- Sample half each teacher imitation batch from bilateral flap distance <=
  0.25 m or close labels, half uniformly; overall imitation still starts at 20%.
- Bound teacher servo acceleration using measured joint velocity when SAC
  queries an IK action it does not execute. Joint position remains measured-state based.

![SAC distances and controller-attributed successes](assets/rl_v2_recovery_20260930.png)

The previous GPU 3 follow-up is
`artifacts/rl/drive_runs/sac_mbv2_episodic_guidance_gpu3_20260930_2120/`,
64 environments and 180 additional iterations from policy checkpoint 60.
It uses `--online-ik-episode-fraction 0.2`, online correction labels, CPU
replay capacity 300,000, batch 1,024 and four updates/vector step. This keeps
64 sampled critic rows per new transition, matching the preceding 32-env
batch-512 pilot. The stored actor-update decay horizon/counter are preserved,
so the resumed expected IK fraction starts below 20%. Warmup is zero; real
online transitions refill uniform Q replay before updates. Existing imitation
and protected success data are restored. This is a measured recovery attempt,
not evidence of learned-policy success. Safety and task/reset contracts remain
unchanged.

The acquisition/pull pilot recorded one genuine held success at iteration 20:
bilateral opposing-flap pinch and proof lift remained valid for 0.267 s, and
the success reward and terminal transition were both present. This was an
**IK collection success**, not a learned SAC success. Its checkpoint includes
20,172 current-environment imitation labels and one protected success row.
The pilot was stopped after iteration 21; its final Drive upload was verified.

GPU 3 continued from that checkpoint in the distinct directory
`artifacts/rl/drive_runs/sac_mbv2_acquire_pull_transfer_gpu3_20260930_2035/`.
It retains 32 environments, the same acquisition/pull/body-assistance settings,
20,000 teacher-fit updates and online correction labels. Uniform Q replay is
not serialized, so 256 IK vector steps refill it before fitting the actor and
resuming SAC. The original 15,360-actor-update imitation decay horizon is
preserved by checkpoint restore; the new invocation requests 150 additional
iterations. Checkpoints are saved every 20 iterations, with five-minute Drive
verification and retention of the newest two verified checkpoints. It stopped
after iteration 67 with zero SAC successes and thirteen unsafe terminations;
the latest policy checkpoint is iteration 60. Large-scale training is still
conditional on SAC sustaining entry and grasp from reset. The follow-up uses
continued expert episodes, described below, while retaining the learned policy.

## Recorded policy failure and visual evidence

[![Checkpoint 580 at t=2 s: distance 81.2 cm, pinch=0, unsafe=1](assets/rl_v2_policy_580_late_20260928.png)](assets/rl_v2_policy_580_20260928.mp4)

[Play/download the original 2026-09-28 policy video](assets/rl_v2_policy_580_20260928.mp4).
This is deterministic checkpoint 580, evaluated for 90 control steps / three
seconds: zero successes and one unsafe termination. It renders CPU USD meshes
at actual live PhysX poses, rather than RTX screenshots. The late pose shows
failure to approach the flap; the image alone does not identify a gravity
feedforward malfunction. Upright torso X/Z control, actual target-error
telemetry and measured velocity diagnostics address the control uncertainty.
The recovery run logs gravity compensation ON and torso pitch/tracking errors.

The same video, scene image and controller-attributed metric plot are attached
natively in the [Notion experiment record](https://app.notion.com/p/3eb63918d42a81f694a9e060aeffe7cc),
with the problem, correction, method and measured limitation beside each.
The video is an earlier failure case, **not footage of the new policy**.
New success footage must be attributed to the controller that actually drove
it; an IK collection success is not SAC-from-reset success.

## Entropy target feasibility correction

![Observed alpha growth and the infeasible entropy target](assets/rl_v2_entropy_cap_20260930.png)

A further exploration defect was identified while the fresh run collected IK
warmup data: asymmetric SAC kept target entropy `-1` per active action despite
its latent Gaussian standard-deviation cap `0.02`. Even before tanh, the largest
available differential entropy is `log(0.02) + 0.5*log(2*pi*e) = -2.493` nats
per dimension. Tanh can only lower that entropy. The target was unattainable,
so increasing alpha could never satisfy it once variance reached the cap.
The preceding run's logged alpha rose from 0.0000288 at iteration 61 to
0.003408 at iteration 193 while std remained near 0.02. This demonstrates the
mismatch, but does not isolate it as the sole cause of policy failure.

Asymmetric SAC now targets the smaller of `-1` and the Gaussian upper bound
minus 0.5 nats per active dimension: `-2.993` at cap 0.02. Broad default caps
retain `-1`. Blocked grippers still contribute neither log-probability nor
entropy target. Target, active dimensions and signed entropy error are logged
and the entropy contract is recorded in checkpoints/manifests. This changes
training optimization, not physical action or task compatibility.

Validation: 42 SAC/demo CPU checks passed. The new regression places a zero-mean
actor at the 0.02 cap and confirms that alpha can decrease with the attainable
target, whereas the previous target would keep increasing it. The running
777e922 process does **not** hot-reload this change; it will be checkpointed
and relaunched before prolonged SAC optimization. Successful data and teacher
fit are to be preserved through that transition.

## Non-finite dynamics: isolation and data preservation

The observed exception arose during `scene.write_data_to_sim()` before reward,
termination and replay filtering could run. `gravity_drive_bias` checked the
whole batch and raised when any robot row contained a non-finite feedforward
or stiffness. Thus a single numerical problem could terminate every parallel
environment. The old logs do not identify whether gravity, inverse-dynamics
mass/Coriolis output or joint/root state first became invalid.

Only v2 SAC opts in to recovery. On a non-finite feedforward row, the common
writer calls the environment's ordinary partial-reset managers before the
next physics write, restores that row's drive target to its repaired measured
position and zeroes that failed write's feedforward/velocity/effort buffers.
Other rows retain their original targets and torque. Bad stiffness/configuration
and unsuccessful state repair still raise; the default writer remains strict.
The numerical mask persists across the control step and forces invalid-reset
termination. The SAC collector separately counts it as a numerical failure and
excludes its entire transition from Q, teacher imitation and success history.
Invalid/reset observations are not fitted into the normalizers. It is never reported as a grasp success.

`numerical_failure_episodes` and `numerical_failure_cause/*` record gravity,
feedforward, mass, Coriolis, joint-state and root-state causes where available.
The failing env IDs and cause counts also appear in the console. Persistent
recovery counts require investigation rather than being evidence of successful
learning. The recovery probe injects a force row, not a claim that all forms
of PhysX corruption are repairable.

Reproduce the small GPU-isolated probe:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src \
  /path/to/isaac-python scripts/rl/probe_v2_numerical_recovery.py \
  --device cuda:0 --output /tmp/v2_numerical_recovery.json
```

Measured result: `passed=true`, failed env `[0]`, unaffected env `[1]`, only
row 0 reset, finite observations/feedforward, next-step failure flag cleared.
A completed-warmup checkpoint and an environment-step exception recovery
checkpoint now preserve policy, optimizers, teacher labels and successful tails.
Uniform replay remains memory-only and refills on restart.

## Reusable progress figures

Generate controller-attributed distance, held-success and outcome plots directly
from a run's completed `metrics.jsonl` rows. The command uses CPU only and emits
an image plus a JSON summary; it tolerates an unfinished trailing live-log row.

```bash
python scripts/rl/plot_v2_run.py --run-dir /absolute/path/to/sac_run \
  --output /absolute/path/to/progress.png
```

The success panel separates IK warmup, online IK and SAC from reset. Numerical
failures are separate from ordinary unsafe and timeout outcomes. Early warmup
plots do not establish that the learned SAC policy has improved.

## Baseline and scope

The preceding GPU 3 run `sac_mbv2_safe_front_gpu3_20260929_230917` stopped
at iteration 573 with zero bilateral pinches and successes. Its final Drive
upload was checksum-verified. The observed recent 100-iteration window at
iteration 465 had mean left/right flap distances of 1.165/1.202 m and zero
samples with both hands within 0.25 m of the front lanes. The run learned to
avoid failure without approaching the grasp.

The physical task remains bilateral opposing-flap grasp with proof lift,
S63/Leju two-finger grippers, rack rollers, upright torso X/Z control, rack
contact threshold 10 N, obstacle threshold 5 N, and no self-collision check.
No curriculum, easier reset distribution, or relaxed success predicate is used.

## Diagnosed causes and corrections

1. SAC used a floor alpha of 0.01 with about -14 summed action log probability
   and gamma 0.999. A continuing entropy contribution of roughly 0.14 per step
   was much larger than the signed distance progress, often around 0.0003.
   This can inflate the value of avoiding terminal contacts. The recovery
   profile uses alpha 0.001 initially, floor 0.00001, and no entropy term in
   the critic backup. Entropy remains in actor optimization. Actual Q, target,
   entropy contribution, and Gaussian standard deviation are now logged.
2. After demo behavior cloning, actor running normalization previously changed
   on every online vector step. This transformed the pretrained actor's input
   coordinates during rollout. Demo and initial reset observations now fit a
   fixed actor normalizer; the critic normalizer still updates online.
3. The flat actor received all 12 box slots and a target one-hot index. A new
   target-centric encoder gathers the selected token and retains robot state,
   rack/conveyor poses, hand–flap relations, skill state and previous actions.
   It initially had 174 features for the 440-D deployable contract (now 198
   features / 464-D after the controller-state correction below), so swapping
   logical box slots does not alter policy input. The critic keeps its full
   privileged observation.
4. The actor's untrained variance branch produced large random joint increments
   despite a pretrained mean. The recovery profile initializes that branch at
   standard deviation 0.15 and caps it at 0.3 by default. The second pilot uses
   0.08/0.15 and alpha 0.0001 with a 0.000001 floor. BC warmup has correlated
   noise; the new IK warmup uses temporally coherent servo actions.
5. Potential progress alone provided weak sustained learning signal before a
   success. Two bounded distance costs, 0.002 times each missing front-stage
   and reach score, distinguish remaining far from approaching. Their worst
   30-second total is 3.6, below the 6-point rack failure. SAC multiplies all
   rewards by 10 internally; logged task rewards remain in their original units.
6. Uniform replay overwrote the few useful near-flap transitions. A protected
   100,000-transition online buffer retains real transitions within 0.25 m of
   either flap or with pinch contact, and supplies 25% of critic batches when
   nonempty. It uses measured current-environment rewards, never legacy rewards.
7. Existing legacy demonstrations describe three independent torso-joint
   increments. Their converted rewards and next states cannot safely be treated
   as new-dynamics critic experience. They remain actor-only imitation data,
   with sampling starting at 20% and decaying over the first 30% of planned
   updates. The BC loss scale is now independent of that sample fraction.
   The loader also accepts newly recorded native 24-action upright-torso demos;
   old 25-action files still pass the explicit conversion path.
8. A batched IK collector uses current deployable poses and successful-demo
   wrist orientations to stage both hands outside the rack, insert, close,
   and propose a small lift. It uses the existing bounded teleop servo and
   writes through the shared RL action manager. Physical pinch and success
   are still measured by the unchanged environment. Its real transitions
   train Q and a separate initial actor imitation pass before IK is disabled.
   The teacher buffer retains the whole warmup up to online replay capacity,
   rather than just its last, possibly stationary, frames. Afterwards only
   SAC produces rollout actions; the initial 20% imitation batch mixes legacy
   and current-environment teacher samples and decays as configured.
9. Quest collection now converts its requested body angles into the upright
   torso's two X/Z action increments. The body mapper reset still receives
   three joint angles, preserving ordinary collection and legacy control.

## Execution and validation

`artifacts/rl/probes/bc_baseline_gpu3_20260930/sac_20260930_170035_1b5a43`
ran 8 environments for 30 iterations without SAC updates, using 5,000 BC
updates and zero warmup noise. Its best iteration-average left/right flap
distances were about 0.397/0.446 m. It had 16 unsafe terminations and no
bilateral pinch or success. Its offline MSE of 0.0079 did not establish a
successful deployed policy.

`artifacts/rl/probes/recovery_gpu3_20260930/sac_20260930_171227_d9628c`
was a 64-environment, 60-iteration recovery pilot, with 5,000 BC updates and
current-environment SAC updates after 8,192 valid warmup transitions. Early
actor-normalizer count stays at 974, and real near-flap samples populate the
protected buffer. It was stopped at iteration 26 after 343 unsafe terminations
and no bilateral pinch/success. Gaussian std reached about 0.65 before the new
cap was added. The mean hand distances had returned to about 0.9 m.

`artifacts/rl/probes/ik_entry_gpu3_20260930/sac_20260930_172823_a6a358`
completed 16 environments and 30 iterations with IK actions and no SAC updates.
At iteration 10, left/right mean flap distances were 0.074/0.077 m and 11.3%
of valid steps had bilateral pinch. The following iteration had 0.44% of steps
meeting instantaneous proof-lift/stability, but no successful held episode.
There were five unsafe and eleven timeout terminations. This establishes
useful physical exploration, not learned-policy success.

The subsequent 64-environment transfer pilot is
`artifacts/rl/drive_runs/sac_mbv2_ik_transfer_pilot_gpu3_20260930_1746/`.
It uses 384 vector steps of IK collection, 5,000 teacher imitation updates,
then SAC with 4 updates/vector step. Its 40 iterations test unguided SAC
rollout after transfer. Checkpoints are saved every ten iterations and
checksum-verified by the existing Drive supervisor every five minutes.

This first transfer pilot stopped just after initial imitation with
`A success terminal transition is missing its success reward`.
Partial resets invalidated a global grasp cache and recomputed all environments;
this added a second hold-time tick for environments that had not reset and
could consume their success-event latch before the next physical reward step.
The success tracker now accepts the physical step ID and advances each row at
most once per step; resetting one row clears only that row's timing. A CPU
regression reproduces the former early-success sequence. The fixed rerun is
`artifacts/rl/drive_runs/sac_mbv2_ik_transfer_fix_gpu3_20260930_1800/`.
Previously reported instantaneous/held IK milestones may include this timing
error; physical contact distances and bilateral pinch remain useful, but
successful held-grasp episodes must be checked again with the fix.

Checkpoints also retain the optimizer-update counter, imitation decay horizon
and completed teacher-fit flag. Resuming a trained actor does not reset its
demo schedule or refit it to the initial guide. Online replay still starts empty.

The fixed transfer pilot completed 30 iterations without the success-reward
error: three held-grasp successes were recorded during warmup/the mixed
handoff iteration. Pure SAC iterations 14-30 had no pinch/success, and final
left/right distances were 0.825/0.942 m. Total unsafe terminations were 409.
Teacher imitation MSE started near 0.0025 but rose to about 0.046 within the
first 300 SAC updates. Transferring the mean alone therefore did not preserve
entry behavior during online learning.

The next profile adds 500 critic-only updates before actor/temperature changes,
an online actor rate of 0.00003 (critic/imitation initialization 0.0003), a
stronger initial BC multiplier of 100 and Gaussian std 0.02/cap 0.05. Initial
alpha is 0.00001 with a 0.0000001 floor. The imitation sample fraction remains
20% and its decay counts actual actor updates, so critic warmup does not consume
the imitation schedule. Of this imitation batch, 80% comes from current IK
collection and 20% from the legacy demos when teacher samples are available.

A separate 10,000-row CPU buffer retains verified held-success terminal
transitions even after large uniform/near-flap buffers overwrite them. It
supplies at most 5% of a critic batch, capped by the number of available real
success rows, and is included in checkpoints for restart. It contains no
converted legacy reward. Logs distinguish `successful_warmup_episodes` and
`successful_sac_episodes`, with an explicit mixed handoff label.

This conservative pilot is
`artifacts/rl/drive_runs/sac_mbv2_conservative_pilot_gpu3_20260930_1822/`,
64 environments and 30 iterations. Add the following overrides to the command
below, using its new unique directory and `--max-iterations 30`:

```bash
--actor-lr 0.00003 --critic-warmup-updates 500 --demo-bc-strength 100 \
--initial-policy-std 0.02 --max-policy-std 0.05 \
--initial-alpha 0.00001 --min-alpha 0.0000001
```

## Controller-state observation correction

The conservative pilot completed without runtime errors but did not preserve
entry. One success was recorded after the actor took over an already advanced
IK episode; this does not establish success from reset. Even the actor-frozen
critic warmup moved away from the flaps, so untrained Q gradients are not the
only possible cause.

Joint increments accumulate into pending PD targets, while observations had
only measured position and velocity. Equal measured states can therefore have
different future motion under the same action. The deployable contract now
adds 20 ordered logical target-minus-measured-joint errors, three local base
commands, and one availability flag immediately before the previous action.
Actor/critic dimensions are 464/530; the target-centric actor has 198 inputs.
The dynamic base now reports measured `root_vel_w` instead of command velocity;
the kinematic base retains the command-derived twist. All policies using the
shared v2 observation builder receive this correction. Old controller targets
cannot be reconstructed reliably from demo actions alone; old 403/469 and
440/506 records use zeros with availability flag zero. New Quest collection
records the actual controller telemetry. Old checkpoints cannot be resumed.

The next GPU 3 pilot is
`artifacts/rl/drive_runs/sac_mbv2_controller_state_pilot_gpu3_20260930_1842/`,
64 environments, 60 iterations, 900 IK warmup vector steps (57,600 valid
transitions), 5,000 teacher-fit updates, std 0.01/cap 0.02, actor rate 0.00003,
500 critic-only updates, BC strength 100 and alpha 0.00001/floor 0.0000001.
The longer collection covers repeat episodes and gives unguided SAC more time
to run from new resets. Checkpoints save every 20 iterations, upload on the
existing five-minute Drive cycle, and retain only verified older files.
Success during collection is reported separately from SAC success.

## Labels on learner-visited states

The controller-observation pilot was stopped after iteration 40. Its frozen
actor initially reduced left-hand distance, but the right hand lagged and the
policy again drifted: final mean distances were 0.927/0.948 m, with no SAC
held success. Controller state alone is therefore insufficient. Offline
teacher MSE was 0.00213, illustrating that a small average joint-action error
can still accumulate into a different closed-loop path.

`--online-teacher-labels` enables current-pose IK correction labels on states
visited by SAC, without replacing the executed policy action. This is a
DAgger-style data collection option: teacher labels enter a separate
actor-only buffer containing observation/action, with no reward, next state,
or Bellman terminal flag. Critic replay always uses the action actually
executed and its measured reward. The teacher restages a hand pair when it
has drifted over 30 cm from the box. Label collection ends with the existing
20%-to-zero imitation schedule. It does not change the task reset distribution
or safety predicates. The option requires IK guidance and a successful demo
source for wrist orientations; the default remains disabled.

Checkpoints retain at most 100,000 teacher labels (about 186 MiB at 464/24
float32 dimensions) and restore them for resumed actor imitation. Actual
success transitions remain in their independent critic buffer. Initial
collection labels no longer store duplicate critic/reward fields, reducing
CPU memory and preventing counterfactual Q data.

The next test uses 16 environments, 60 iterations, 900 IK vector steps,
5,000 teacher-fit updates, 500 critic-only updates, std 0.01/cap 0.02 and
BC strength 1000. The stronger multiplier constrains early Q-driven changes;
the sample fraction still starts at 20% and decreases. It is a pilot setting,
not a measured successful profile.

## Obstacle guard did not match the documented task contract

A subsequent source audit found that the obstacle guard still used net robot
contact minus rack vectors. Contrary to `RL_MULTI_BOX_V2_PILOT.md`, it therefore
included task boxes and floor, and could also sum or cancel multiple contacts
on one body. Fingers were omitted but gripper bases were not. This is a real
common-environment discrepancy; it can penalize box handling, but the existing
aggregate logs cannot prove which object caused each historical termination.

The guard now uses explicit robot-body-to-workcell pair filters. In the
current v2 scene the obstacle list is the conveyor surface, two rails and
four legs. Task boxes and ground are absent by construction. Rack/rollers
retain their separate 10 N filter; the conveyor guard remains 5 N, with the
maximum individual pair magnitude rather than a net resultant. Seven extra
filtered vectors per collision-relevant robot body require about 1.64 MiB
of output tensors at 2,048 environments, apart from sensor/PhysX overhead.
No additional box-pool pair filters are created. Quest contact markers and
reward calibration use the same filtered forces and separate thresholds.
Future workcell obstacles must be added through `eligible_obstacle_targets`.
The actor never receives these privileged contact filters. SAC logs
`unsafe_obstacle/<Surface|RailLeft|RailRight|Leg0..3>` and each object's
maximum eligible force, using pre-reset snapshots.

The already started teacher-label pilot is stopped before completion so the
next test uses this corrected safety contract. Its partial run is not used as
a success-rate comparison. New manifests include an explicit contact contract,
and old-contract SAC checkpoints are rejected on resume.

## Pilot result and VR grasp retargeting

The corrected-contact/online-label pilot completed all 60 iterations and its
final checkpoint/logs passed Drive verification. SAC distances reached
0.302/0.295 m at iteration 51, versus about 0.93/0.95 m in the earlier pilot,
but physical SAC pinch and held success remained zero. Imitation reached zero
at iteration 52 and the behavior deteriorated; final distances were
0.544/0.869 m. This is improved entry, not a successful grasp policy. The
16- versus 64-environment runs also differ in several settings, so the result
is not an isolated causal ablation.

The IK guide supplied bilateral contact but no held-success episode in this
pilot. Its grasp location was the neutral flap center and its wrist lift only
25 mm; a flexible flap can move without lifting the box. The existing VR
successes grasp about 7.4/8.1 cm along the flap and 1.8 cm above the neutral
center, rather than exactly at its center. These are measured demonstration
offsets, not a claim that all center grasps are physically impossible.

`--ik-grasp-goal demo` retargets a physically annotated successful grasp offset
onto the new perceived neutral flap center. Live inference still uses only
pose/proprioception; contact annotations choose the reference offline. Missing
physical pinch annotations cause an error instead of accepting closed-jaw
commands as success. `--ik-lift-distance-m` controls the proposed wrist motion
(8-150 mm) separately from the unchanged 8 mm box clearance / 0.25 s success
predicate. The default center/25 mm guide is retained as an option. Observation
relations remain neutral flap centers as requested.

The follow-up GPU 3 run is
`artifacts/rl/drive_runs/sac_mbv2_vr_goal_pilot_gpu3_20260930_1958/`, 32 environments,
120 iterations, 900 IK vector steps, 20,000 teacher imitation updates,
demo grasp offsets and an 80 mm wrist lift. It keeps actor rate 0.00003,
critic warmup 500, std 0.01/cap 0.02 and BC strength 1000. For this diagnostic
run imitation decays across the full planned run (`--demo-decay-fraction 1`)
rather than vanishing before the policy's first complete approach episode.
It still starts at 20% and decreases; no curriculum or terminal relaxation is
introduced. A large run must scale minibatch/update count with the number of
environments: the 16-environment pilot sampled 128 critic rows per new
transition (4 updates * 512 / 16). Keeping four updates with 2,048 environments
would reduce that ratio to 2 with a 1,024 batch, which is not the same learning
profile. Monitor real held success and unconstrained SAC performance before
calling any of these settings solved.

## Acquisition pose and body reach assistance

The final-VR-pose-first guide produced no physical pinch in its early rollout,
although several environments reached its close/lift command phase. A closing
command is therefore not evidence of grasp. The pilot was stopped and its
partial data is not considered a successful teacher dataset.

`--ik-grasp-goal center-to-demo` now acquires at the neutral flap center and
only then proposes the demonstration-offset pull plus wrist lift. This keeps
the reference's final pose from replacing the initial acquisition pose.
`--ik-base-clearance-m` and `--ik-torso-forward-m` expose the previously fixed
0.65 m base clearance and zero forward-torso assistance. The next pilot uses
0.55 m clearance and up to 0.10 m upright forward translation while a hand
is still more than 5 cm from the goal. The standard torso solver retains its
15 cm forward and pitch limits; after lift begins, body commands stop.
All rack/obstacle contacts remain subject to the same 10 N / 5 N checks.
These settings affect exploration labels, not task reset difficulty or actor
observation geometry. Defaults preserve the earlier center/0.65 m/no-forward
profile for comparisons.

The acquisition/pull pilot directory is
`artifacts/rl/drive_runs/sac_mbv2_acquire_pull_pilot_gpu3_20260930_2014/`,
32 environments with 120 iterations originally planned. It recorded its first
held IK success at iteration 20 and stopped after iteration 21 with eleven
unsafe terminations. Its checkpoint was used for the transfer continuation
described above, rather than discarding the first new successful transition.
This also shows why an early zero-pinch window is not enough to declare the
collector incapable; the continuation still needs a learned-policy check.

## Continued expert collection and deployable closing evidence

The 20,000-update transfer fit reached MSE 0.000732, but after the first
actor-frozen SAC rollouts the hands again moved roughly 0.7 m away. No held
SAC success was observed through iteration 54. Increasing offline fit alone
therefore did not establish a usable entry controller.

`--online-ik-episode-fraction 0.2` adds episode-stable expert collection while
SAC continues to optimize. Each new episode is assigned to IK with the
configured initial probability, declining with the stored actor-update
imitation schedule. An active episode keeps its assignment until termination;
the default zero retains pure SAC behavior. This is an additional collection
option, separate from the 20% actor-imitation minibatch. Q replay receives the
actual executed action and measured reward for both controllers, with no
invented next state. Reset distributions, success criteria, action dimensions
and collision thresholds are unchanged.

Logs separate online IK/SAC distances, bilateral pinch and held successes.
`successful_sac_from_reset_episodes` excludes episodes that first used warmup
or expert actions, so a handoff cannot appear as end-to-end SAC success.

The correction guide also previously advanced its close counter whenever it
*proposed* closed jaws, even when SAC had actually left them open. Lift readiness
now requires both actual close commands and measured closure over 50%, for the
same 15 ticks. This controller telemetry is deployable and does not substitute
for the unchanged physical contact/success checks. CPU regressions cover
hypothetical closing, episode-stable sampling, decay-at-reset and warmup
handoff attribution. Related SAC/demo checks pass: 37 tests.

To enable continued collection on either the direct SAC entrypoint or Drive
supervisor, add:

```bash
--guided-warmup-mode ik --online-teacher-labels --online-ik-episode-fraction 0.2
```

## Graceful stop and final-checkpoint preservation

Several requested pilot stops raised `KeyboardInterrupt` inside Isaac's foreign
callbacks, which could swallow it and allow more Python iterations before the
supervisor's 120-second owned-child timeout. Only the last periodic checkpoint
survived in those cases. V2 SAC now installs deferred stop handlers, checks the
request after a complete transition/update, saves the current partial iteration
atomically and exits with explicit `stopped` status. The supervisor accepts that
status only when it initiated the stop; an unexpected zero-exit `stopped` run
remains an error. Existing immediate handlers for other runners are preserved.
Checkpoint retention/Drive finalization rules are unchanged, and stopped runs
are not labelled naturally completed training.
Related checks pass: 38 SAC/demo tests and 12 supervisor lifecycle tests.

The first episodic continuation initialized correctly but the checkpoint's
strict exploration-dictionary comparison rejected the new collection fraction.
Compatibility now permits changing only `online_ik_episode_fraction`; actor
features, observations, actions, reward and contact contracts remain checked.
The failed initialization's final logs were Drive-verified before retry.

## Reproduce the transfer pilot

Discover the existing host-local remote with `bash scripts/rl/gdrive.sh listremotes`
and set `RL_DRIVE_REMOTE_ROOT='<remote>:HumanoidScene-RL'` privately. Use a new
`--experiment-dir`; the supervisor creates it and refuses an existing folder.

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/rl/sac_with_drive.py \
  --experiment multi-box-v2-grasp --robot-model s63 --gripper leju-twofinger \
  --experiment-dir artifacts/rl/drive_runs/<unique-experiment-name> \
  --gpu 3 --num-envs 64 --max-iterations 40 --rollout-steps 32 \
  --batch-size 512 --updates-per-step 4 \
  --replay-capacity 150000 --replay-device cpu \
  --learning-starts 0 --warmup-vector-steps 384 \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 \
  --demo-pretrain-steps 1000 --guided-warmup-mode ik --teacher-pretrain-steps 5000 \
  --initial-alpha 0.0001 --min-alpha 0.000001 \
  --initial-policy-std 0.08 --max-policy-std 0.15 --demo-bc-strength 20 \
  --save-interval 10 --no-self-collision \
  --remote-root "$RL_DRIVE_REMOTE_ROOT" \
  --gpu-limit-mib 12000 --gpu-reserve-mib 0 --max-seconds 7200
```

With `--guided-warmup-mode bc`, the original actor-guided noisy collector is
retained. `--no-demo-guided-warmup` selects bounded random exploration.
The same options work on `scripts/rl/multi_box.sh grasp-v2-sac` directly,
without the Drive/resource-supervisor flags.

Compare `rollout_policy`, per-hand distances, bilateral pinch, stable/opposing
flap/proof-lift fractions, maximum continuous hold time and `termination/success`.
Only the last establishes completed held-grasp episodes. New source and
observation contracts require a fresh run; an incompatible checkpoint is rejected.

Focused tests cover target-slot invariance, fixed normalization under online
distribution shift, the absence of an idle entropy bonus in critic targets,
distance-cost bounds, native/legacy demo loading, terminal observations,
reward composition, contact attribution and upright-torso kinematics.

Notion record: https://app.notion.com/p/3eb63918d42a81f694a9e060aeffe7cc

Related primary-source approach:
[Efficient Online Reinforcement Learning with Offline Data / RLPD](https://github.com/ikostrikov/rlpd).
The recovery changes do not claim to implement the full RLPD algorithm.

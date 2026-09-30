# Multi-box v2 SAC recovery experiments — 2026-09-30

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

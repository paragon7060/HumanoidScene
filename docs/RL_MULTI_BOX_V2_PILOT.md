# Multi-box v2 grasp SAC pilot

This is the bounded learning check before a long multi-box run. It trains only
the first low-level `grasp` skill. Carry and place training assemblies remain
separate follow-up work.

## On the other PC

Update the same branch and run the four-step wiring check first:

```bash
git pull --ff-only
bash scripts/rl/multi_box.sh grasp-v2-sac \
  --smoke-test --num-envs 4 --device cuda:0 --headless
```

Then run the bounded pilot:

```bash
bash scripts/rl/multi_box.sh grasp-v2-sac-pilot \
  --device cuda:0 --headless
```

The pilot is capped in code at 64 environments, 20 iterations, 32 vector
steps per iteration, 50,000 CPU replay transitions, one optimizer update per
vector step and checkpoints every five iterations. At the defaults it collects
40,960 transitions. A process lock rejects a duplicate v2 grasp SAC run.

Outputs are written under:

```text
artifacts/rl/multi_box_v2/grasp_sac/sac_<timestamp>_<id>/
```

The run directory contains `manifest.json`, `env.yaml`, `agent.yaml`,
`metrics.jsonl`, checkpoints and `status.json`. Run artifacts are ignored by
Git. Google Drive upload is a separate process; follow `docs/RL_GOOGLE_DRIVE.md`
only when backup is needed.

## Terminal contract

Grasp success terminates without bootstrap. Robot-to-rack structural or roller contact
above 10 N is recorded as `robot_rack_collision`; eligible robot contact with
other obstacles above 5 N is recorded separately. These use maximum eligible
body/pair force, not a sum over all links. Task boxes and floor are excluded
from the obstacle guard. Either event, together with a
workspace radius above 1.5 m, box drop, excessive lift, or excessive linear or
angular box speed terminates as unsafe. If success and unsafe occur on the same
step, unsafe wins and no success bonus is paid. Timeouts are truncated and may
bootstrap from the captured pre-reset observation.

Every SAC step reconstructs the manager terminal masks and fails immediately
if they differ from the environment outputs. A success terminal must contain a
positive `success_event`. Metrics include:

- `termination/success`
- `termination/unsafe`
- `termination/time_out`
- `terminated_episodes` and `timeout_episodes`
- every `reward_term/*` mean and `reward_term_nonzero/*` rate
- `reward_breakdown_max_abs_error`
- non-finite transition and optimizer counts
- per-hand flap/front distance, bilateral pinch, proof lift, stability and maximum hold time
- `rollout_policy` and `teacher_pretrain_*` to separate IK collection from SAC performance

For successful Quest demos, target-centric observations and IK-assisted initial
collection, see [the recovery experiments and command](RL_V2_RECOVERY_20260930.md).
The standard 20-iteration pilot does not run the full recovery warmup; use its
explicit command when checking the transfer from IK to the SAC actor.

Teacher imitation retains up to 100,000 critical actor-only labels in a
separate CPU FIFO so off-target collection cannot evict all approach/closure
examples. Half each teacher batch uses this stratum; the overall imitation
fraction follows its configured teacher schedule. Metrics distinguish
`teacher_critical_rows` in ordinary FIFO from `teacher_persistent_critical_rows`
in protected storage. Checkpoint label snapshots remain capped at 100,000 rows.

When resuming a policy, `--teacher-label-checkpoint /absolute/path/to/older.pt`
can recover labels from a compatible earlier v2 checkpoint independently of
`--checkpoint`. `sac_with_drive.py` forwards this option for v2 runs. Source
manifest and label dimensions/actions are checked. Only `actor_obs` and `action`
are imported through this path; policy, optimizer, reward and Q data are not.

The pilot passes the wiring and numerical check when `status.json` is complete,
`nonfinite_transitions` stays zero, reward breakdown error stays at or below
floating-point tolerance, optimizer updates occur, and terminal counts match
the episode totals. Policy quality still requires examining whether success
frequency rises and unsafe terminations fall in a longer controlled run.

## SAC stability and data-only restart

V2 now enables `--critic-layer-norm` and `--actor-q-normalize` by default.
The actor Q term uses a detached reciprocal mean absolute Q scale, keeping
its magnitude comparable to imitation. `--max-alpha` defaults to 0.001 and
must be at least `--initial-alpha`; tight std caps also use a squash-aware,
active-channel entropy target. These changes target critic/temperature
divergence observed in the recovery experiment; they do not establish grasp
success. `--no-critic-layer-norm` and `--no-actor-q-normalize` allow comparison.

To retain executed success experience while discarding an unstable policy:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/rl/multi_box.sh grasp-v2-sac \
  --experience-checkpoint /absolute/path/to/old/checkpoint_00000133.pt \
  --teacher-label-checkpoint /absolute/path/to/teacher/checkpoint_00000029.pt \
  --demo-dataset examples/demos/v2_grasp_quest_success.hdf5 \
  --guided-warmup-mode ik --demo-guided-warmup --online-teacher-labels \
  --initial-alpha 0.00001 --max-alpha 0.001 --min-alpha 0.0000001 \
  --initial-policy-std 0.01 --max-policy-std 0.02 \
  --demo-batch-fraction 0.2 --demo-bc-strength 10 --no-self-collision
```

Do not combine experience import with full `--checkpoint` resume. Experience
import checks the physical observation/action/reward/contact/safety contracts
and transition dimensions/finiteness, but imports no source model or optimizer.
Only previously executed protected success tails train Q; hypothetical teacher
labels still train the actor alone. `sac_with_drive.py` forwards these options
for v2 and retains the existing five-minute verified upload/retention workflow.
Monitor `actor_q_scale`, `target_entropy_mean`, Q/target values, imitation error,
and `successful_sac_from_reset_episodes` separately from IK successes.

Full policy resume may adjust the episode-guidance fraction and critic-only
warmup update threshold: these control collection/update timing without changing
network parameters or the physical task. Architecture, actor features and other
saved learning contracts still must match. The optimizer update counter is
preserved, so a resumed critic that already exceeded the threshold does not
repeat that warmup. The stored recorded-demo decay horizon is preserved on resume.

## Separate live corrections from recorded VR imitation

The recovery run showed a short pilot's VR decay horizon could also retire all
online teacher labels and expert success collection during the first 12% of a
long continuation. The unassisted actor then drifted far from the teacher and
produced no held grasp successes. Enable independent actor-only corrections:

```bash
--online-teacher-labels \
--teacher-batch-fraction 0.2 --teacher-min-batch-fraction 0.1 \
--teacher-bc-strength 100 --teacher-decay-updates 128000 \
--online-ik-episode-fraction 0.2 --online-ik-min-episode-fraction 0.1 \
--online-ik-decay-updates 128000
```

These options work in `multi_box.sh grasp-v2-sac` and `sac_with_drive.py`.
VR imitation still starts at `--demo-batch-fraction 0.2` and decays to zero.
Its loss and minibatch are separate from current teacher corrections. Teacher
labels contain only deployable observations and proposed actions: they never
enter Q replay as imagined rewards/transitions, and do not replace SAC actions.
Only the separately selected expert episodes execute IK actions. These remain
IK successes in the metrics; `successful_sac_from_reset_episodes` measures the
actor without an expert action override. Expert assignment changes only at
episode boundaries. The explicit horizons count actual actor updates across
resumes, independent of a short pilot's iteration count. A nonzero floor keeps
correction/expert collection available; reduce or disable it explicitly when
measured unassisted success supports doing so. This changes the training method,
not resets, physics, reward, success conditions, or curriculum.

The teacher loss weight is fraction times strength (initial 20, floor 10 in this
example). This is an experiment setting, not proof it will learn the task. The
default teacher fraction/floor is zero, preserving the legacy mixed VR/teacher
path. Independent schedules are recorded in the manifest; changing actor
imitation settings requires a fresh run/data-only import rather than silently
changing a full policy resume. Watch `teacher_bc_loss`, `teacher_bc_weight`,
`teacher_bc_fraction`, `teacher_bc_samples_this_iteration` and
`online_teacher_labels_this_iteration` separately from `demo_bc_*`.

Safety diagnostics also log `unsafe_rack_peak_body/<body>` and
`contact_force/rack_body_<body>_max_n` from the same already-filtered matrices
used by the rack guard. The peak counter attributes one body per eligible
rack failure, before reset; it does not sum contact forces or add sensors.

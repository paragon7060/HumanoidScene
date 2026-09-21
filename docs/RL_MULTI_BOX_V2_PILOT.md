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
above 20 N is recorded as `robot_rack_collision`; other aggregate obstacle
contact above 20 N is recorded separately. Either event, together with a
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

The pilot passes the wiring and numerical check when `status.json` is complete,
`nonfinite_transitions` stays zero, reward breakdown error stays at or below
floating-point tolerance, optimizer updates occur, and terminal counts match
the episode totals. Policy quality still requires examining whether success
frequency rises and unsafe terminations fall in a longer controlled run.

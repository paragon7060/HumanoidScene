# Multi-box v2 skeleton

This package is isolated from the existing four-box runner.  The implemented
boundary currently covers:

- randomized rack and conveyor anchors;
- up to 12 logical boxes in shelf 2 and shelf 3 rack regions;
- one random box for each low-level skill environment;
- high-level selection of one logical box, locked until placement;
- explicit deployable `current_skill` state with fixed `grasp -> carry -> place`
  transitions driven by real-sensor/perception evidence;
- an injectable low-level policy registry;
- a versioned, format-neutral reset snapshot container.
- an injectable `state -> success -> observation -> reward` step pipeline.
- the approved grasp-success predicate: both hands pinch opposing right/left
  flaps, maintain stable hand-box poses, lift 8 mm clear of the rack, and hold
  continuously for 0.25 s.
- the approved carry-success predicate: the opposing-flap physical grasp
  remains valid without a hand-box relative-pose constraint, box tilt from
  gravity stays within 20 degrees, all four footprint corners lie inside the
  belt, no placed box overlaps it, and its bottom is 5-15 cm above the belt.
- the approved place-success predicate: belt support, both physical grasps
  released, full footprint containment, no overlap, long-axis alignment
  within 10 degrees, motion below 5 cm/s and 0.2 rad/s, held for 0.5 s.
- live per-box placement state: placed boxes are unavailable to the high-level
  selector, but any later disturbance revokes placement and makes them
  selectable again; full success requires every active box to be placed now.
- separate deployable and privileged tensor-state schemas.  Exact box velocity,
  contact force, collision force, support, overlap, and success measurements
  remain privileged and do not implicitly enter policy observations.
- a replaceable `PerceptionFrame` interface for up to 12 logical box poses and
  rack/conveyor anchors. Initial simulation training may gather exact Isaac
  poses through `IsaacScenePerceptionAdapter`; pose noise, missed detections,
  and a real perception producer come later. Poses are world-frame xyz+wxyz:
  each box articulation root, the rack root, and the conveyor belt cuboid
  center. The belt top lies 15 mm above its cuboid center.
- `IsaacRobotProprioAdapter` reads only the 20 actuated torso/arm/head joints in
  fixed policy order, base pose/twist, calibrated TCP poses, and measured and
  commanded left/right gripper closure. Passive claw linkage joints are excluded.
- a separate actor-side placement **estimate** from perceived box and belt
  poses plus measured gripper opening. It requires full box footprint within
  the belt, box bottom within 2 cm of the belt surface, long-axis error within
  10 degrees, an observed bilateral opening (closure fraction at most 0.2),
  and 0.5 s of height stability within 5 mm. No TCP separation is required.
  Release is latched per box, so grasping the next box does not revoke an
  undisturbed prior placement. The exact contact/motion-based place success
  remains separate and cannot be passed to the actor state schema.
- a deployable skill-transition estimator: grasp needs both grippers measured
  at least 80% closed, both TCP-to-box relative poses stable within 10 mm and
  10 degrees, and the selected box lifted 8 mm above its selection pose for
  0.25 s. Carry then needs continued measured closure, box tilt within
  20 degrees, the full footprint over free belt space, and the box bottom 5-15 cm above
  the belt. Place uses the separate pose-based placement estimate. The
  per-skill progress vector is diagnostic progress, not a calibrated success
  probability. This transition source never reads contact force or box speed.
- an opt-in single-step runtime (`MultiBoxDeployableRuntime`, with an Isaac
  source wrapper) that reads each sensor source once, updates the placement
  estimate, calls a replaceable high-level box selector only when a target is
  needed, advances at most one skill, and returns the resulting actor state.
  `FirstSelectableBoxSelector` is a deterministic integration check, not a
  trained policy. When every box is estimated placed, it leaves the controller
  waiting without selecting a phantom target. Call `reset(env_ids)` after each
  scene reset; this runtime does not reset the simulator itself.
- an isolated headless inspection entrypoint at
  `python -m kuavo_isaaclab_scene.rl.multi_box.experiments.inspect_runtime`
  (defaults: S63/Leju, two boxes, ten zero-action steps). It acquires a
  single-process lock, owns and closes exactly one Isaac app/environment, and
  never starts OpenXR. It is for integration inspection, not reward validation.
- base-relative actor observations with 12 masked box tokens, rotation-6D poses,
  robot proprioception/TCP poses, target and current skill, plus a separate
  asymmetric critic view containing privileged physics measurements.
- a bootable staged-grasp manager assembly with 25 unified S63/Leju actions and
  a fixed 403-value deployable actor observation.  Run the isolated GPU smoke
  check with `python -m kuavo_isaaclab_scene.rl.multi_box.experiments.inspect_grasp_assembly
  --steps 2 --num-envs 1 --headless --device cuda:0`.
- a vectorized privileged grasp adapter that maps each environment's active
  logical box to its physical pool asset, reads four jaw-to-flap contact
  matrices, checks the 5 N per-jaw contact/region/opposition rule, measures
  rack-relative proof lift and 10 mm/10 degree hand-box stability, and emits
  one-step bilateral-pinch and success events.  It remains outside the actor
  observation contract.
- the staged grasp reward manager now consumes those privileged tensors using
  the approved potential/event weights, applies base-motion, action-rate and
  joint-limit costs, and terminates on exact grasp success. Arm/torso obstacle
  contact above 20 N, a dropped/unstable box, excessive box lift/speed, and a
  base position beyond the 1.5 m workspace are hard failures.
- v2 potential shaping and the v2 SAC/PPO learners use the same 0.999 discount.
  At 30 Hz this retains credit across multi-second skills; the generic SAC
  implementation keeps its independent 0.99 default.
- an asymmetric PPO contract with a 403-value deployable actor group and a
  separate 62-value simulator-only critic group. RSL-RL maps the critic to
  `policy + critic` while the actor receives only `policy`.
- an isolated v2 training entrypoint at `bash scripts/rl/multi_box.sh grasp-v2`.
  Add `--smoke-test --num-envs 4` for one short wiring update. It writes a
  manifest required by the existing verified Drive backup. Run artifacts stay
  outside Git and may be uploaded to the configured private Drive destination.
- an isolated asymmetric SAC entrypoint at
  `bash scripts/rl/multi_box.sh grasp-v2-sac`. The actor receives only the
  403-value deployable observation; twin Q critics receive that observation
  plus the 62 privileged values. Replay stores both current/next views so
  terminal transitions bootstrap from their pre-reset observations. The
  default 250k CPU replay is about 1.7 GiB; use `--smoke-test --num-envs 4`
  for a four-step, one-update wiring check. Every SAC iteration also checks
  that the Isaac reward equals the sum of the v2 breakdown and records each
  weighted term plus its nonzero rate in `metrics.jsonl`.
- a bounded learning pilot at
  `bash scripts/rl/multi_box.sh grasp-v2-sac-pilot --device cuda:0`. It caps
  the run at 64 environments, 20 iterations and 50k replay transitions, starts
  updates after 4096 warmup transitions, saves every five iterations, and
  records per-cause `success`/`unsafe`/`time_out` terminal counts. Smoke and
  pilot profiles are mutually exclusive.
- terminal transitions retain their pre-reset actor and critic observations.
  SAC bootstraps through timeouts but never through success or unsafe terminal
  states, verifies the manager masks every step, and rejects any success
  terminal that is missing its success reward. If success and unsafe occur on
  the same step, unsafe wins and the success bonus is suppressed.
- independent grasp/carry/place/high-level reward composition with normalized
  potential differences, one-shot events, explicit initial weights, and no
  dependency on the legacy reward managers.
- initial SI-unit metric normalization plus append-only JSONL/HUD shadow
  diagnostics for measuring raw, normalized, weighted, and cumulative values
  during one-environment VR teleoperation without changing the dataset. The
  v2 JSONL records grasp/carry/place breakdowns together on every step and
  applies the read-only probe event pulses plus the same active common costs
  used by grasp training. Event bonuses are latched once per episode so a
  policy cannot farm reward by repeatedly losing and reacquiring a pinch.
- a read-only Isaac adapter for mode-2 VR inspection.  It locks the nearest
  active box at reset, measures live TCP/flap geometry, rack extraction,
  conveyor footprint/alignment/clearance, bottom height, and privileged box
  speed, then feeds the normalized grasp/carry/place dense terms.  J/L changes
  target and 1/2/3 changes the explicitly inspected phase.
- mode-2-only filtered reports for all four finger links against the selected
  physical box's right/left flaps.  HUD/JSONL include per-jaw contact force,
  actual-flap-region membership, and jaw opposition.
- a read-only mode-2 grasp probe: each jaw must press its assigned flap with at
  least 5 N inside the contact region, both jaws must oppose, the two hands
  must hold different flaps, hand-to-box drift must stay within 10 mm/10 degrees,
  and an 8 mm proof lift above the reset height must also clear the sloped rack
  shelf for 0.25 s.
- read-only carry/place probes: carry requires a prior grasp success, continued
  bilateral opposing-flap pinch, box tilt within 20 degrees, the whole box
  footprint over the belt, free space, and a 5-15 cm pre-place height.  Place
  requires a prior carry success, at least
  0.2 N of actual box-body/belt support, both physical grasps released, full
  footprint containment, free space, long-axis alignment within 10 degrees,
  and box motion below 5 cm/s and 0.2 rad/s for 0.5 s.
  All probe flags are logged but do not trigger reward, control changes, or
  skill transitions.  Until placed-box tracking is wired, overlap with any
  other active box near the belt is conservatively rejected.

`MultiBoxTaskSkeleton.validate_scene()` permits scene work.  Training assembly
uses `validate_training()` and fails until all four deliberate providers are
injected: success, internal state, policy observation, and reward.  This keeps
the legacy four-box predicates from silently becoming v2 behavior.

The following are intentionally absent until their design is approved:

- trained high-level selection and carry/place policies;
- full multi-box episode success/failure and carry/place manager assemblies;
- a real perception/robot telemetry producer and full privileged-state adapter;
- observation normalization and policy-network architecture;
- remaining collision/event adapters and empirical weight tuning.

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
- the approved carry-success predicate: the grasp remains valid, all four box
  footprint corners lie inside the belt, no placed box overlaps it, and its
  bottom is 5-15 cm above the belt.
- the approved place-success predicate: belt support, both grippers released
  and 2 cm clear, full footprint containment, no overlap, long-axis alignment
  within 10 degrees, motion below 5 cm/s and 0.2 rad/s, held for 0.5 s.
- live per-box placement state: placed boxes are unavailable to the high-level
  selector, but any later disturbance revokes placement and makes them
  selectable again; full success requires every active box to be placed now.
- separate deployable and privileged tensor-state schemas.  Exact box velocity,
  contact force, collision force, support, overlap, and success measurements
  remain privileged and do not implicitly enter policy observations.
- base-relative actor observations with 12 masked box tokens, rotation-6D poses,
  robot proprioception/TCP poses, target and current skill, plus a separate
  asymmetric critic view containing privileged physics measurements.
- independent grasp/carry/place/high-level reward composition with normalized
  potential differences, one-shot events, explicit initial weights, and no
  dependency on the legacy reward managers.
- initial SI-unit metric normalization plus append-only JSONL/HUD shadow
  diagnostics for measuring raw, normalized, weighted, and cumulative values
  during one-environment VR teleoperation without changing the dataset.
- a read-only Isaac adapter for mode-2 VR inspection.  It locks the nearest
  active box at reset, measures live TCP/flap geometry, rack extraction,
  conveyor footprint/alignment/clearance, bottom height, and privileged box
  speed, then feeds the normalized grasp/carry/place dense terms.  J/L changes
  target and 1/2/3 changes the explicitly inspected phase.
- mode-2-only filtered reports for all four finger links against the selected
  physical box's right/left flaps.  HUD/JSONL include per-jaw contact force,
  actual-flap-region membership, and jaw opposition; no numerical threshold,
  success event, or phase transition consumes these values yet.

`MultiBoxTaskSkeleton.validate_scene()` permits scene work.  Training assembly
uses `validate_training()` and fails until all four deliberate providers are
injected: success, internal state, policy observation, and reward.  This keeps
the legacy four-box predicates from silently becoming v2 behavior.

The following are intentionally absent until their design is approved:

- phase transitions and target reselection;
- full-task success and failure predicates;
- simulator/real adapters that populate the approved state schemas;
- observation normalization and policy-network architecture;
- approved contact thresholds, collision/event adapters, and empirical weight tuning;
- episode termination and timeout policy.

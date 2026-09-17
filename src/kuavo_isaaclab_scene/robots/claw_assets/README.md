# Leju two-finger package

This directory is the single Python implementation for the Leju two-finger
claw used by S63, S200062 and S56. The authoritative numeric configuration is:

```text
src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json
```

Edit that file for actuator gains, contact material, the soft distal pads and
their compliance, the default squeeze force, and each host robot's calibration.
The three entries in `configs/grippers.json` are aliases that select
`runtime_presets`; they must not copy the package values.

| File | Responsibility |
|---|---|
| `package.py` | Config and asset paths, default force, URDF composition |
| `linkage.py` | Four-bar kinematics and loop-closing USD joints |
| `geometry.py` | Jaw gap/contact geometry |
| `force.py` | Force feedforward and feedback servo |
| `vr.py` | Quest/RL debug contact sensors and force-close action setup |
| `usd.py` | The contact model: inertials, colliders, materials, soft pads, marker disable |
| `isaaclab.py` | Isaac Lab spawn; names the host's links and mesh scope only |

The contact model exists once, in `usd.py`. `author_claw_jaw_contact()` authors
the jaw/housing materials and colliders and `author_claw_distal_pads()` authors
the soft fingertip pads and their compliant material. A standalone claw, the S63
composition and the S200062/S56 integration all call those two functions, so the
grasping surface can never be tuned in one path and missed in the other.

Hosts differ only in what they hand those functions. The packaged claw and S63
carry real `collisions` meshes; the S200062/S56 donor USDs ship an empty
`collisions` scope, so their `visuals` meshes are passed as the collider source
and their wrist links are listed in `replace_colliders` to drop the crude URDF
cylinder the mesh hull replaces.

The old modules `robots/twofinger_linkage.py`,
`robots/twofinger_geometry.py`, and `robots/gripper_force.py` only preserve
existing imports. Add new code here.

Runtime-only config changes take effect after restarting the process. Rebuild
USD-baked geometry or physics with:

```bash
bash scripts/build_leju_claw.sh
bash scripts/build_s63_twofinger.sh
```

The extractor preserves the package-owned action, actuator, contact, runtime,
and force-control sections when it refreshes donor metadata.

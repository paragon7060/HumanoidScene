# Leju two-finger package

This directory is the single Python implementation for the Leju two-finger
claw used by S63, S200062 and S56. The authoritative numeric configuration is:

```text
src/kuavo_isaaclab_scene/assets/leju_claw_two_finger/config.json
```

Edit that file for actuator gains, contact material, flat distal pads, the
default squeeze force, and each host robot's calibration. The three entries in
`configs/grippers.json` are aliases that select `runtime_presets`; they must not
copy the package values.

| File | Responsibility |
|---|---|
| `package.py` | Config and asset paths, default force, URDF composition |
| `linkage.py` | Four-bar kinematics and loop-closing USD joints |
| `geometry.py` | Jaw gap/contact geometry |
| `force.py` | Force feedforward and feedback servo |
| `vr.py` | Quest/RL debug contact sensors and force-close action setup |
| `usd.py` | USD inertials, colliders, contact pads and marker disable |
| `isaaclab.py` | Isaac Lab spawn and integrated-host contact setup |

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

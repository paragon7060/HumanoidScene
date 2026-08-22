# Kuavo visual materials

The supplied Kuavo URDF is not missing all material bindings: most visual
links are explicitly authored as pure white, and STL meshes do not carry
textures. Re-running the URDF converter therefore still produces an almost
white robot. This repository adds a link-level rendering override after the
instanceable robot USD is spawned. It does not alter collision geometry,
inertial properties, joints, actuators, or policy observations/actions.

## Run with a palette

`industrial_blue` is the default and needs no extra option:

```bash
./run_scene.sh --robot-material industrial_blue
./run_manager_env.sh --robot-material industrial_blue --num-envs 1
```

The included presets are:

- `industrial_blue`: silver shell, graphite joints, blue accents, metallic
  camera mounts, and dark sensor faces;
- `high_visibility`: the same material separation with safety-orange accents;
- `original`: disables the override and shows the source URDF/USD white-grey
  palette.

For example:

```bash
./run_scene.sh --robot-material high_visibility
./preview_quest_local.sh --robot-material industrial_blue
./eval_groot.sh --robot-material industrial_blue --mock-policy \
  --headless --episodes 1 --max-steps 5
```

The same options are accepted by the standalone scene, manager-based runner,
Quest preview/collection launchers, and GR00T evaluator. The environment
variables `KUAVO_ROBOT_MATERIAL` and `KUAVO_ROBOT_MATERIAL_CONFIG` provide the
same selection for custom launch code.

## Customize colors and link groups

Edit [`configs/robot_materials.json`](../configs/robot_materials.json), or keep
an independent deployment file:

```bash
./run_scene.sh --robot-material my_palette \
  --robot-material-config /data/workcell/robot_materials.json
```

Colors are linear RGB values in `[0, 1]`. `roughness=0` is mirror-smooth and
`roughness=1` is matte; `metallic=0` is dielectric/plastic and `metallic=1` is
metal. Every `link_rules[].pattern` is a Python full-match regular expression
against a direct child link below the Kuavo articulation prim. The first
matching rule is used.

The checked-in deployment JSON and
`src/kuavo_isaaclab_scene/configs/robot_materials.json` wheel fallback are
kept byte-identical. After changing the deployment file for distribution,
copy it to the package fallback or run the wheel build workflow that
synchronizes deployment configs.

## Visual check in Isaac Sim

1. Run `./run_scene.sh --robot-material industrial_blue`.
2. In Stage, expand `/World/envs/env_0/KuavoVisualMaterials/Looks` and confirm
   the five PreviewSurface materials exist.
3. Select `/World/envs/env_0/Kuavo/base_link`, then inspect **Material on
   selected models**. Its binding should target the palette material and use
   `strongerThanDescendants`.
4. Switch to **RTX - Real-Time** if the viewport is using a non-material debug
   display mode.

Use `--robot-material original` as an A/B comparison. No saved source USD is
modified by this runtime override.

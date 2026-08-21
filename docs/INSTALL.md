# Installation and first run

This repository is self-contained for the Kuavo workcell code, JSON layouts,
URDF, meshes, and USD assets. Isaac Sim, Isaac Lab, NVIDIA CloudXR, policy
weights, and LeRobot are external runtime dependencies and are not vendored.

## 1. Tested stack

- Linux workstation with an NVIDIA RTX GPU and a driver supported by Isaac Sim
- Python 3.11.13
- Isaac Sim 5.0.0
- Isaac Lab 0.45.7
- PyTorch 2.7.0 + CUDA 12.8
- optional LeRobot 0.6.0 / Dataset format v3.0 in a separate Python 3.12 environment
- optional Meta Quest 3/3S plus NVIDIA CloudXR Runtime and Quest client

Other compatible Isaac Lab releases may work, but the versions above are the
reproducible reference. The launch scripts never assume a particular username,
home directory, conda installation, or checkout path.

## 2. Clone and configure Isaac Lab

```bash
git clone git@github.com:paragon7060/HumanoidScene.git
cd HumanoidScene

conda activate env_isaaclab
export ISAACLAB_PYTHON="$(command -v python)"
./setup.sh
```

If Isaac Lab is already importable from the active conda environment,
`ISAACLAB_PYTHON` can be omitted. The launcher searches the active environment,
`python3`, `python`, and common conda locations. To validate without installing
the package again:

```bash
./setup.sh --check-only
```

For URDF-to-USD regeneration only, also point to the Isaac Lab source checkout:

```bash
export ISAACLAB_DIR=/absolute/path/to/IsaacLab
./convert_kuavo.sh
```

The generated Kuavo USD is already committed, so conversion is not required for
a normal clone.

## 3. First scene and manager environment

Open the editable standalone scene:

```bash
./run_scene.sh --prefill 2
```

Run the manager-based environment:

```bash
./run_manager_env.sh --num-envs 1 --steps 100000
```

Choose different rack contents per launch:

```bash
./run_scene.sh --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'
```

Shelf numbers are `1=bottom`, `2=middle`, and `3=top`. Each box size has two
instances. See [the workcell guide](ISAACSIM_WORKCELL_GUIDE.md) for editing,
rotation/scale capture, and respawn.

## 4. Meta Quest hand tracking and data collection

Install NVIDIA CloudXR Runtime on the Isaac workstation and the matching CloudXR
client on the Quest. Locate the runtime's `openxr_cloudxr.json`, then launch the
collector from the workstation:

```bash
export XR_RUNTIME_JSON=/absolute/path/to/openxr_cloudxr.json
./collect_quest_teleop.sh \
  --dataset-format hdf5 \
  --dataset datasets/kuavo_quest_teleop.hdf5 \
  --rack-boxes '1:small*2;2:medium,large;3:xlarge*2'
```

The collector enables Isaac Lab OpenXR, maps both tracked hands to Kuavo arm
targets, records head/wrist cameras, and shows the head camera with wrist-camera
overlays in the headset. The desktop Isaac Sim UI can also display the camera
feeds in small viewport windows.

CloudXR Runtime, the Quest client, and the proprietary CloudXR npm package are
not committed to this repository. The reproducible local browser integration is
included as a pinned patch and setup script:

```bash
export CLOUDXR_NPM_TGZ=/absolute/path/to/nvidia-cloudxr-6.2.0.tgz
./setup_quest_browser.sh
npm --prefix .external/cloudxr-js-samples/simple run dev-server
```

Network pairing, controller bindings, browser/IWER preview, recentering, and
troubleshooting are documented in the
[Quest teleoperation guide](QUEST3_KUAVO_TELEOP_GUIDE.md).

## 5. LeRobot Dataset v3 and GR00T

Keep LeRobot in a separate environment and export its interpreter:

```bash
conda activate lerobot060_groot
export LEROBOT_PYTHON="$(command -v python)"
"${LEROBOT_PYTHON}" -c \
  'from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == "v3.0"'

conda activate env_isaaclab
export ISAACLAB_PYTHON="$(command -v python)"
./collect_quest_teleop.sh --dataset-format both
```

Datasets, videos, runs, checkpoints, and WandB outputs are intentionally ignored
by Git. Evaluate a manager-compatible GR00T output with:

```bash
./eval_groot.sh --mock-policy --headless --episodes 1 --max-steps 5
```

Then follow the [GR00T N1.7 evaluation guide](GROOT_N1_7_EVAL_GUIDE.md) to load
real policy weights and select the action convention.

## 6. Layout portability

Runtime layout files live in `configs/`. A clone can use them directly. To keep
site-specific layouts outside Git, set:

```bash
export KUAVO_CONFIG_DIR=/absolute/path/to/writable/configs
export KUAVO_WORKCELL_LAYOUT=/absolute/path/to/workcell_layout.json
export KUAVO_RACK_BOX_POSES=/absolute/path/to/rack_box_poses.json
```

Copy `.env.example` as a reference for all supported environment variables. The
scripts read exported shell variables; they do not automatically execute `.env`.

## 7. Verification and packaging

Simulator-independent tests:

```bash
"${ISAACLAB_PYTHON}" -m pytest -q
bash -n ./*.sh scripts/*.sh
```

Build a wheel containing every committed USD/URDF/mesh and fallback config:

```bash
./scripts/build_wheel.sh
```

The largest committed file is below GitHub's 100 MB per-file limit, so Git LFS
is not required. Runtime outputs are excluded; the source assets remain regular
Git files and are available immediately after cloning.

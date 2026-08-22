# Installation and first run

This repository is self-contained for the Kuavo workcell code, JSON layouts,
URDF, meshes, and USD assets. Isaac Sim, Isaac Lab, NVIDIA CloudXR, policy
weights, and LeRobot are external runtime dependencies and are not vendored.

## 1. Supported stable stack

This repository pins the current NVIDIA GA stack rather than following a moving
branch:

- Ubuntu 22.04 Linux x86_64, GLIBC 2.35 or newer;
- NVIDIA RTX GPU, 32 GB or more system RAM, and 16 GB or more VRAM recommended;
- current NVIDIA production driver (580.65.06 or newer is NVIDIA's Linux
  recommendation for the 5.1 generation);
- Python 3.11;
- Isaac Sim 5.1.0 from NVIDIA's Python package index;
- Isaac Lab tag `v2.3.2` / package `isaaclab==0.54.2`;
- PyTorch 2.7.0 and torchvision 0.22.0 from the CUDA 12.8 wheel index;
- Gymnasium 1.2.1 and NumPy 1.x;
- optional LeRobot 0.6.0 / Dataset format v3.0 in a separate Python 3.12 environment;
- optional Meta Quest 3/3S plus NVIDIA CloudXR Runtime and Quest client.

NVIDIA currently publishes Isaac Lab 3.0 as a beta tied to Isaac Sim 6.0.x.
It is intentionally not the default for a stable deployment. The exact pins
used by scripts live in `versions/stable.env`; future upgrades should change
that file and the compatibility tests together.

The Quest/OpenXR path targets x86_64. NVIDIA documents OpenXR as unsupported on
DGX Spark/aarch64, so the automated installer rejects that architecture instead
of producing a partially working Quest setup.

## 2. One-command conda, Isaac Sim, and Isaac Lab installation

Install Miniconda or Anaconda first, then clone this repository. The installer
creates a new `env_isaaclab_232` environment and leaves every existing conda
environment untouched:

```bash
git clone git@github.com:paragon7060/HumanoidScene.git
cd HumanoidScene

./install_isaaclab_stable.sh --dry-run
./install_isaaclab_stable.sh
```

The script performs these pinned operations:

1. checks Linux x86_64 and GLIBC 2.35+;
2. creates a Python 3.11 conda environment;
3. installs `isaacsim[all,extscache]==5.1.0` from NVIDIA's package index;
4. installs the official PyTorch 2.7.0 CUDA 12.8 wheels;
5. clones the exact Isaac Lab `v2.3.2` tag under `.external/`;
6. installs the `isaaclab` core extension from the exact source tag;
7. installs this repository editable and runs the runtime doctor.

The first Isaac Sim invocation displays NVIDIA's EULA prompt and may spend more
than ten minutes downloading/caching extensions. Review and answer that prompt
yourself. The installer does not silently set `OMNI_KIT_ACCEPT_EULA`.

To choose another non-existing environment name or source checkout:

```bash
./install_isaaclab_stable.sh \
  --env-name kuavo_sim_51 \
  --isaaclab-dir /data/tools/IsaacLab-v2.3.2
```

If a named environment already exists, the installer stops without modifying
it. `--reuse-env` must be given explicitly to reuse it; its final versions must
still pass the runtime doctor.

Activate and validate the finished installation:

```bash
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
export ISAACLAB_DIR="$PWD/.external/IsaacLab-v2.3.2"

./doctor.sh
./setup.sh --check-only
```

`doctor.sh` reports the interpreter, GLIBC, GPU/driver, and package versions
without starting the Isaac Sim GUI. Every simulator launcher also fails early
on an old/mixed runtime. `KUAVO_ALLOW_UNSUPPORTED_RUNTIME=1` is available only
as a temporary diagnostic escape hatch; it is not a supported deployment.

## 3. Manual installation (equivalent commands)

Use this path when you want to inspect each NVIDIA installation step:

```bash
conda create -y -n env_isaaclab_232 python=3.11 pip
conda activate env_isaaclab_232
python -m pip install --upgrade pip setuptools "wheel==0.45.1"

python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

mkdir -p .external
git clone --depth 1 --branch v2.3.2 \
  https://github.com/isaac-sim/IsaacLab.git .external/IsaacLab-v2.3.2
cd .external/IsaacLab-v2.3.2
python -m pip install --editable source/isaaclab
cd ../..

python -m pip install onnx==1.18.0 typing_extensions==4.12.2 psutil==5.9.8
python -m pip install --no-build-isolation -e .
export ISAACLAB_PYTHON="$(command -v python)"
export ISAACLAB_DIR="$PWD/.external/IsaacLab-v2.3.2"
./doctor.sh
```

Verify Isaac Lab itself before launching this workcell:

```bash
"${ISAACLAB_DIR}/isaaclab.sh" -p \
  "${ISAACLAB_DIR}/scripts/tutorials/00_sim/create_empty.py" --headless
```

For URDF-to-USD regeneration only, keep `ISAACLAB_DIR` exported and run
`./convert_kuavo.sh`. The generated Kuavo USD is already committed, so
conversion is not required for a normal clone.

The installer intentionally installs Isaac Lab core rather than executing the
upstream `--install none` wrapper. In v2.3.2 that wrapper also installs
Mimic/RL/Jupyter packages, then launches VS Code settings bootstrap and an EULA
prompt. This workcell only imports Isaac Lab core (including manager-based
environments, cameras, OpenXR devices, and retargeters), so the smaller install
avoids unrelated dependency drift. NVIDIA's EULA is still shown on the first
actual Isaac Sim run and must be accepted by the operator.

`pip check` may report one upstream metadata conflict: Isaac Lab v2.3.2 pins
`starlette==0.49.1`, while the Isaac Sim 5.1 FastAPI wheel declares
`starlette<0.46`. The workcell does not use Isaac Sim's FastAPI service; retain
Isaac Lab's pin for its device/livestream code rather than manually changing it.

## 4. First scene and manager environment

Open the editable standalone scene:

```bash
./run_scene.sh --prefill 2
```

Run the manager-based environment:

```bash
./run_manager_env.sh --num-envs 1 --steps 100000
```

The official Allegro hands are enabled by default and are fetched from the
Isaac asset library on first use. Use `--gripper none` for an offline/legacy
15-D smoke test, or see [Gripper configuration](GRIPPER_CONFIGURATION.md) for
custom local USDs and mount calibration.

Choose different rack contents per launch:

```bash
./run_scene.sh --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'
```

Shelf numbers are `1=bottom`, `2=middle`, and `3=top`. Each box size has two
instances. See [the workcell guide](ISAACSIM_WORKCELL_GUIDE.md) for editing,
rotation/scale capture, and respawn.

## 5. Meta Quest hand tracking and data collection

Meta Quest 코드도 위에서 설치한 동일한 `env_isaaclab_232` 환경을 사용한다.
Quest 전용으로 Isaac Sim 또는 Isaac Lab을 다시 설치하거나, 구버전
`env_isaaclab` 환경을 활성화하면 안 된다. 먼저 OpenXR experience와 extension
cache가 Isaac Sim 5.1.0 / Isaac Lab v2.3.2 구성에 포함됐는지 확인한다:

```bash
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./quest_doctor.sh
```

이 명령은 GUI를 열지 않는다. CloudXR Runtime 없이도 PC browser/IWER preview에
필요한 repository/Isaac 구성은 검사할 수 있다.

Install NVIDIA CloudXR Runtime on the Isaac workstation and the matching CloudXR
client on the Quest. Locate the runtime's `openxr_cloudxr.json`, then launch the
collector from the workstation:

```bash
export XR_RUNTIME_JSON=/absolute/path/to/openxr_cloudxr.json
./quest_doctor.sh --require-runtime
./collect_quest_teleop.sh \
  --dataset-format hdf5 \
  --dataset datasets/kuavo_quest_teleop.hdf5 \
  --rack-boxes '1:small*2;2:medium,large;3:xlarge*2'
```

The collector enables Isaac Lab OpenXR, maps both tracked hands to Kuavo arm
targets, records head/wrist cameras, and shows the head camera with wrist-camera
overlays in the headset. The desktop Isaac Sim UI can also display the camera
feeds in small viewport windows.

`RawQuestOpenXRDevice` compatibility adapter explicitly enables raw hand/head
tracking in Isaac Lab v2.3.2 while leaving Kuavo's existing calibration,
smoothing, safety clamp, and differential-IK mapper in control of the action.

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

## 6. LeRobot Dataset v3 and GR00T

Keep LeRobot in a separate environment and export its interpreter:

```bash
conda activate lerobot060_groot
export LEROBOT_PYTHON="$(command -v python)"
"${LEROBOT_PYTHON}" -c \
  'from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == "v3.0"'

conda activate env_isaaclab_232
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

## 7. Layout portability

Runtime layout files live in `configs/`. A clone can use them directly. To keep
site-specific layouts outside Git, set:

```bash
export KUAVO_CONFIG_DIR=/absolute/path/to/writable/configs
export KUAVO_WORKCELL_LAYOUT=/absolute/path/to/workcell_layout.json
export KUAVO_RACK_BOX_POSES=/absolute/path/to/rack_box_poses.json
```

Copy `.env.example` as a reference for all supported environment variables. The
scripts read exported shell variables; they do not automatically execute `.env`.

## 8. Verification and packaging

Simulator-independent tests:

```bash
"${ISAACLAB_PYTHON}" -m pytest -q
bash -n ./*.sh scripts/*.sh
./install_isaaclab_stable.sh --dry-run
```

Build a wheel containing every committed USD/URDF/mesh and fallback config:

```bash
./scripts/build_wheel.sh
```

The largest committed file is below GitHub's 100 MB per-file limit, so Git LFS
is not required. Runtime outputs are excluded; the source assets remain regular
Git files and are available immediately after cloning.

## Official version references

- [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)
- [Isaac Lab v2.3 pip installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- [Isaac Sim 5.1 Python installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)

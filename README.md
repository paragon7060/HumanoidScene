# HumanoidScene: Kuavo Isaac Lab workcell

Kuavo 5.5 humanoid가 경사진 랙의 열린 박스를 빈 컨베이어 공간으로 옮기고,
모든 박스를 처리한 뒤 실제 물리 버튼을 누르는 Isaac Lab 환경이다. 편집 가능한
standalone scene과 manager-based 환경, Meta Quest 양손 teleoperation, head/wrist
camera, LeRobot Dataset v3 수집, GR00T N1.7 action 평가 코드를 함께 제공한다.

다른 컴퓨터에서 시작할 때는 [설치 및 첫 실행 가이드](docs/INSTALL.md)를 먼저
따르면 된다. 저장소에는 실행에 필요한 Kuavo/Rack/Box USD, URDF, mesh와 기본
layout JSON이 포함된다. Isaac Sim/Isaac Lab, CloudXR, LeRobot, policy weight는
각 라이선스와 GPU 환경에 맞게 별도로 설치한다.

```bash
git clone git@github.com:paragon7060/HumanoidScene.git
cd HumanoidScene
./install_isaaclab_stable.sh
conda activate env_isaaclab_232
export ISAACLAB_PYTHON="$(command -v python)"
./setup.sh
./run_scene.sh --prefill 2
```

주요 문서:

- [설치, 의존성, 첫 실행](docs/INSTALL.md)
- [Isaac Sim 배치 편집·위치/회전/크기 캡처](docs/ISAACSIM_WORKCELL_GUIDE.md)
- [Meta Quest 3/3S teleoperation과 LeRobot 수집](docs/QUEST3_KUAVO_TELEOP_GUIDE.md)
- [Allegro hand와 교체 가능한 gripper 설정](docs/GRIPPER_CONFIGURATION.md)
- [GR00T N1.7 evaluation](docs/GROOT_N1_7_EVAL_GUIDE.md)
- [외부 asset과 runtime 안내](THIRD_PARTY_ASSETS.md)

Meta Quest 없이 PC Chrome/IWER 연결 경로를 먼저 확인하려면
`./quest_doctor.sh`로 OpenXR 구성요소를 점검한 뒤
`./preview_quest_browser.sh`를 실행한다. 실제 Quest/OpenXR 수집은
`./quest_doctor.sh --require-runtime`과 `./collect_quest_teleop.sh`를 사용한다.

Isaac Lab workcell based on the supplied factory reference image. With the
default legacy six-tote layout, the complete task is:

1. remove all six open totes from one gravity-fed, three-tier rack;
2. place them only in unoccupied/reserved conveyor space;
3. advance the stopped conveyor queue by one slot when the infeed is occupied;
4. press the green fence button after all six task totes are confirmed;
5. start the conveyor only after the valid button press.

The supported GA stack is Isaac Lab v2.3.2 (Python package 0.54.2) on Isaac
Sim 5.1.0 and Python 3.11. Isaac Lab 3.0 is not selected while NVIDIA labels
it beta. The scene uses the user-supplied rack/box USDs and NVIDIA's warehouse
and Digital Twin conveyor assets. The authoritative execution, Isaac Sim
editing, pose/rotation/scale capture, and respawn workflow is maintained in
[`docs/ISAACSIM_WORKCELL_GUIDE.md`](docs/ISAACSIM_WORKCELL_GUIDE.md).

## Complete operating guide

Use [`docs/ISAACSIM_WORKCELL_GUIDE.md`](docs/ISAACSIM_WORKCELL_GUIDE.md) for the full
workflow. In particular, it documents how to pause Isaac Sim, move the eight
standalone box root prims onto the measured rack, save the stage, capture their
Rack.usd-relative position/rotation/scale, and reproduce the arrangement in
both `scene.py` and the manager-based environment.

## Repository layout and installation

```text
HumanoidScene/
├── src/kuavo_isaaclab_scene/  # installable Python package
│   ├── assets/                # USD, URDF, meshes and textures
│   └── configs/               # immutable wheel fallback layouts
├── configs/                   # mutable deployment/captured layouts
├── scripts/                   # canonical launch/capture utilities
├── integrations/cloudxr/      # reproducible Quest browser bridge patch
├── tests/                     # simulator-independent unit tests
├── docs/                      # operating documentation
├── artifacts/                 # local previews and saved stages (ignored)
├── pyproject.toml             # setuptools package metadata
└── *.sh                       # backward-compatible launch wrappers
```

GR00T N1.7 평가 진입점은 `./eval_groot.sh`이며, 실제 모델을 로드하지 않는
짧은 wiring test는 다음과 같다:

```bash
./eval_groot.sh --mock-policy --headless --episodes 1 --max-steps 5
```

The root shell wrappers add `src` to `PYTHONPATH`, so installation is optional
for normal use. For development or deployment into an Isaac Lab environment:

```bash
export ISAACLAB_PYTHON=/absolute/path/to/isaaclab-environment/bin/python
./setup.sh
./doctor.sh
```

Build a distributable wheel with all USD/URDF/mesh assets and a synchronized
fallback copy of the active `configs/`:

```bash
./scripts/build_wheel.sh
```

The root wrappers export `KUAVO_CONFIG_DIR=<project>/configs`. An installed
wheel can use another writable deployment config directory by setting the same
environment variable; without it, the wheel uses its packaged fallback JSON.

## Implemented workcell

- fixed-base Kuavo 5.5 at the center of the rack/conveyor work area. Its
  checked-in USD is converted from Leju Robotics' official `biped_s55` URDF
  and STL meshes and preserves the source-authored colors without a runtime
  visual-material override;
- one packaged `src/kuavo_isaaclab_scene/assets/Rack.usd` steel rack bay
  (replaces the earlier official
  Nucleus `RackLongEmpty_A2`), already authored in real meters at
  105.1 cm (width) x 88.1 cm (depth) x 216.5 cm (height), spawned at
  identity scale (no per-axis unit conversion needed);
- three authored shelf tiers with a measured ~5.1-degree ramp tilt. No
  synthetic RollerDeck, roller cylinders, or front-stop colliders are added;
  the provided `Rack.usd` supplies the complete rack geometry and collision;
- eight standalone local boxes
  (`src/kuavo_isaaclab_scene/assets/{Small,Medium,Large,XLarge}Box.usd`,
  two of each), spawned as PhysX articulations. Their open-top bodies have
  four free-swinging flap lids; a shared CLI/JSON/Python dictionary chooses
  which instances start on each shelf, while unused instances remain in
  floor staging;
- black safety fence and one articulated button-station asset containing the
  yellow post, bezel, and illuminated spring-loaded green plunger;
  (mounted on the fence post opposite the original side);
- official `ConveyorBelt_A08` with a stopped PhysX surface velocity;
- nine visible conveyor slots and a yellow robot-reachable infeed;
- zero to three foreign-worker totes selected with `--prefill`;
- slot occupancy, reservation, queue-push, and full-conveyor handling;
- button gating and conveyor startup state machine;
- head-mounted and chest/waist-mounted cameras matching real Kuavo5
  hardware, plus two wrist-mounted cameras (not present on the real robot)
  added for close-range manipulation visibility.
- configurable left/right wrist grippers. The default `allegro` preset uses
  two independently controlled official Allegro Hand articulations; use
  `--gripper none` for the legacy handless schema or a custom
  `--gripper-config` JSON for another end effector.

The button is not a wrist-distance proxy. The packaged `button_station.usda` contains
a fixed post link and an 18 mm prismatic plunger with a return spring. A press
is accepted after at least 6 mm of measured joint travel, and only while all
active task boxes are on the conveyor (six totes in the default layout, or the
boxes selected by a non-empty custom rack layout).

The open tote is a real hollow rigid body made from a bottom and four wall
colliders, not a solid cube:

```text
src/kuavo_isaaclab_scene/assets/open_tote.usda
```

## Run interactively

```bash
cd HumanoidScene
./run_scene.sh --prefill 2
```

`--prefill 2` places two foreign-worker boxes at the conveyor infeed. The first
Kuavo placement plan therefore requests a 0.26 m queue push before release.

## Choose rack boxes for each run

Shelf numbers are `1=bottom`, `2=middle`, and `3=top`. Box names are
case-insensitive: `small`, `medium`, `large`, and `xlarge`. For example:

```bash
./run_scene.sh --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'
```

The same option is accepted by the manager-based launcher:

```bash
./run_manager_env.sh --num-envs 1 --steps 100000 --rack-boxes \
  '1:small*2,medium;2:large,xlarge;3:medium,large,xlarge'
```

Each shelf supports up to four boxes. The first two entries occupy the front
row and the next two the rear row. Only two instances of each type exist, so a
type may appear at most twice across all shelves. Every unused instance stays
under `/World/envs/env_0/Workcell/StagingBoxes` instead of being nested inside
the rack prim.

For a persistent code-level dictionary, edit only
`DEFAULT_RACK_BOX_LAYOUT` in `src/kuavo_isaaclab_scene/rack_box_layout.py`:

```python
DEFAULT_RACK_BOX_LAYOUT = {
    1: {"small": 2, "medium": 1},
    2: {"large": 1, "xlarge": 1},
    3: ["medium", "large", "xlarge"],
}
```

You can also keep multiple layouts as JSON and select one per run:

```json
{
  "shelves": {
    "1": {"small": 2, "medium": 1},
    "2": ["large", "xlarge"],
    "3": ["medium", "large", "xlarge"]
  }
}
```

```bash
./run_scene.sh --rack-box-layout /absolute/path/rack_layout.json
./run_manager_env.sh --num-envs 1 --rack-box-layout /absolute/path/rack_layout.json
```

All four source box USDs currently have the same authored bounding box.
`BOX_DIMENSIONS_M` in `src/kuavo_isaaclab_scene/rack_box_layout.py` provides editable physical target
sizes for the named variants; spawn scales are computed automatically.

When a non-empty custom layout is selected, `scene.py` treats those local box
instances as the button-gated task boxes and parks the legacy six oracle totes
away from the rack. `--auto-demo` and `--verify-button` remain legacy-tote
checks, so run them without a custom layout. In the manager-based environment,
the option also changes task progress, success, and termination tracking to use
the selected standalone box instances.

## Box flap joint friction

All four flap revolute joints use Isaac Lab/PhysX joint-axis friction. Set a
fixed value in the standalone scene with:

```bash
./run_scene.sh --rack-boxes '1:small*2;2:medium,large;3:xlarge' \
  --flap-static-friction 0.40 --flap-dynamic-friction 0.25
```

Randomize every box and every flap independently once at standalone-scene
startup with:

```bash
./run_scene.sh --rack-boxes '1:small*2;2:medium,large;3:xlarge' \
  --randomize-flap-friction \
  --flap-static-friction-range 0.15 0.65 \
  --flap-dynamic-friction-range 0.08 0.45
```

The manager-based robustness environment enables this randomization by
default and samples again for every environment reset. Its ranges can be
changed with the same arguments:

```bash
./run_manager_env.sh --num-envs 8 --steps 100000 \
  --flap-static-friction-range 0.15 0.65 \
  --flap-dynamic-friction-range 0.08 0.45
```

For a deterministic manager-based run:

```bash
./run_manager_env.sh --num-envs 1 --no-randomize-flap-friction \
  --flap-static-friction 0.40 --flap-dynamic-friction 0.25
```

PhysX requires dynamic friction to be no greater than static friction. Random
samples are therefore clamped per flap to satisfy that constraint. Persistent
code defaults and ranges are in `src/kuavo_isaaclab_scene/box_flap_friction.py`.

## Task-system demonstration

```bash
./run_scene.sh --auto-demo --prefill 2 --ignore-captured-box-poses
```

The oracle demonstration validates the non-overlap and completion logic. It
script-moves a pair of rack totes at 3.0, 9.5, and 16.0 seconds, presses the
button at 20 seconds, and starts the conveyor. This corresponds to roughly
10 seconds per three totes.

For a short headless smoke test:

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode performance \
  --auto-demo --demo-speed 4 --prefill 2 --steps 720 \
  --ignore-captured-box-poses
```

Expected terminal events include:

```text
[PLAN] Push ... queued tote(s) by 0.26 m, then place ...
[TASK] All six rack totes are on the conveyor. Green button is now armed.
[BUTTON] Valid green-button press accepted ...
[SUCCESS] Task complete ... Conveyor started at 0.22 m/s.
```

Verify the physical press and spring return without opening a window:

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode performance \
  --verify-button --steps 300 --ignore-captured-box-poses
```

The check scripted-places all six totes, then passes only when a kinematic
contact probe depresses the actual plunger beyond 6 mm, completes the gated
task/starts the conveyor, and returns below 2 mm after release.

## Changing the layout

Isaac coordinates use metres in `(x, y, z)` order. For a temporary visual
adjustment, select
`/World/envs/env_0/Workcell/SafetySystem/ButtonStation` in the Stage tree and
use the Move tool. This edit lasts only for the current composed stage.

For an actual-site calibration, both runtime variants now load the same
`configs/workcell_layout.json`. Use this workflow:

1. Start `./run_scene.sh --prefill 2` and pause the timeline.
2. Position, orient, and (for the rack only) resize the root groups listed
   below with Isaac Sim's Move, Rotate, and Scale tools.
3. Use **File > Save As** and save, for example,
   `artifacts/output/measured_workcell.usda`.
4. Close Isaac Sim and capture the edited poses (and, for the rack, scale):

   ```bash
   ./capture_layout.sh artifacts/output/measured_workcell.usda
   ```

5. Relaunch either `run_scene.sh` or `run_manager_env.sh`. Both read the newly
   captured layout automatically.

| Layout anchor | Prim/group to move in Isaac Sim |
|---|---|
| Kuavo | `/World/envs/env_0/Kuavo` |
| Rack | `/World/envs/env_0/Workcell/Racks/Rack` |
| Conveyor | `/World/envs/env_0/Workcell/ConveyorSystem` |
| Fence + button | `/World/envs/env_0/Workcell/SafetySystem` |
| Fence only | `/World/envs/env_0/Workcell/SafetySystem/Fence` |
| Button post/station only | `/World/envs/env_0/Workcell/SafetySystem/ButtonStation` |

The Stage hierarchy is organized for editing:

```text
Workcell
├── Racks
│   └── Rack  (captured pose/scale anchor; identity-local Rack.usd Visual)
├── StagingBoxes  (all local box prims, including shelf-positioned instances)
├── LegacyTask  (legacy oracle Totes and Cargo, separate from the rack anchor)
├── SafetySystem  (Fence, ButtonStation, verification probe)
├── ConveyorSystem  (Visual, Surface, slots, foreign totes)
├── DynamicObstacles  (worker, AMR)
└── Cameras
```

The capture stores position and world quaternion `(w, x, y, z)` for every
anchor, and the rack anchor's world scale as well. The Rack root owns this
transform while its `Visual` child stays at an identity local transform.
Rack shelf-tier points,
rack box positions, cargo, conveyor slots/surface, belt direction, conveyor
occupancy checks, fence wires,
button press axis, and the Kuavo base are all regenerated in the captured
coordinate frames and at the captured rack scale. Floor-mounted equipment
will normally need yaw rotation only, but pitch and roll are also
propagated.

Only the rack anchor's scale is functionally consumed; resizing any other
anchor's `Visual` prim in Isaac Sim has no effect on the derived geometry.
Configured box poses are authored as measured local points in the Rack anchor and
transformed by the captured Rack anchor Xform through
`workcell_layout.local_point_to_world(...)`. The supplied rack asset is thus
the position, orientation, scale, visual, and collision reference.

To try a separate calibration without replacing the default file:

```bash
KUAVO_WORKCELL_LAYOUT=/absolute/path/layout.json ./run_scene.sh
```

For a persistent button-station pose without using the capture tool, edit its
`pos` and `rot` in `configs/workcell_layout.json`.
Both environments share this value;
duplicate edits in Python are unnecessary.

The complete post, bezel, collision geometry, and plunger move together. The
button faces `-Y`.

Other principal geometry controls are:

- rack real-world footprint: edit `"rack"."scale"` in
  `configs/workcell_layout.json`
  directly, or use the Isaac Sim Scale tool and recapture. The current local
  asset is 88.1 cm deep x 105.1 cm wide x 216.5 cm high at identity scale;
- local box target sizes: edit `BOX_DIMENSIONS_M` in
  `src/kuavo_isaaclab_scene/rack_box_layout.py`;
- rack box shelf coordinates and seating clearance: the `RACK_*_RAW`
  constants in `src/kuavo_isaaclab_scene/rack_box_layout.py`;
- conveyor slot spacing: `CONVEYOR_SLOT_PITCH`.

Anchor translation and rotation should be changed through
the packaged `configs/workcell_layout.json`; the derived physical geometry is kept aligned
automatically.

## Render a preview

```bash
./run_scene.sh --headless --device cuda:0 --rendering_mode quality --steps 1 \
  --prefill 2 --screenshot artifacts/output/gravity_rack_workcell.png
```

Please inspect the result yourself:

```text
artifacts/output/gravity_rack_workcell.png
```

Check these visual points:

- Kuavo stands between the single rack and the conveyor;
- the rack contents match `--rack-boxes` (or the legacy two totes per shelf
  when no custom layout is supplied);
- boxes sit flush on the shelf surfaces, not embedded in the frame;
- the provided Rack.usd shelf meshes/colliders are used without synthetic
  roller-deck or front-stop geometry;
- the black fence is alongside the rack, with the yellow button post on the
  opposite end of the fence from the plain far post;
- the green button faces the robot;
- two foreign boxes occupy the stopped conveyor when `--prefill 2` is used;
- green slot markers and the yellow infeed marker align with the belt.

## Save the composed stage

```bash
./run_scene.sh --headless --device cuda:0 --steps 1 --prefill 2 \
  --save-stage artifacts/output/kuavo_gravity_rack_task.usda
```

## Controller integration

The pure task logic is in `task_system.py`. It is independent of Isaac Sim and
provides:

- `ConveyorSlotManager.reserve(...)`
- `ConveyorSlotManager.commit(...)`
- `PlacementPlan` with `PLACE`, `PUSH_THEN_PLACE`, `WAIT`, and `BLOCKED`
- `RackConveyorTask.update_transferred(...)`
- button gating through `RackConveyorTask.press_button()`

A robot controller should reserve a plan, execute its IK/grasp/push/release
motion, verify the final tote pose, and then commit the reservation. Two workers
cannot reserve the infeed simultaneously.

The base Kuavo USD has no finger joints, so the default configuration attaches
two independently actuated Allegro articulations at runtime. `--auto-demo`
still does not claim physical robot manipulation: it validates task logic with
scripted box motion, while learned/teleoperated control uses the configured
hands.

## Manager-based robustness environment

There are now two runtime layers:

- `scene.py`: one visual `InteractiveScene` plus the custom conveyor/task
  scheduler used by the oracle demonstration;
- `manager_env.py`: a real Isaac Lab `ManagerBasedRLEnvCfg`, registered as
  `Isaac-Kuavo-RobustWorkcell-v0`.

The manager-based environment contains:

- a 17-dimensional default manager action: 15 waist/dual-arm targets plus two
  binary Allegro commands (`--gripper none` restores the previous 15-D schema);
- a dynamically sized state/policy observation including both 16-joint hands
  and physical button travel;
- head-mounted 120x160 RGB and depth observations (`robustness_camera`,
  the ``policy``-group vision term);
- chest/waist-mounted, left-wrist, and right-wrist 120x160 RGB observations
  (`waist_camera`, `left_wrist_camera`, `right_wrist_camera`);
- observation noise for robot state, object state, RGB, and depth;
- randomized tote/cargo mass, friction, restitution, robot arm mass, actuator
  gains, box-flap joint friction, gravity, lighting, rack-box poses, cargo
  poses, and conveyor prefill;
- randomized moving human and AMR paths, speeds, phases, and offsets;
- task progress, cargo retention, tote stability, obstacle clearance, action
  smoothness, and success rewards;
- timeout, cargo spill, tote drop, moving-obstacle contact, and task success
  terminations;
- a curriculum that gradually increases pose noise, mover speed, path
  variation, and cargo disturbances over 1.5 million environment steps.

### Cameras

| Camera | Mount point | Real Kuavo5 hardware? |
|---|---|---|
| `robustness_camera` / `head_camera` | `head_camera_base` | Yes |
| `waist_camera` | `waist_camera_base` | Yes |
| `left_wrist_camera` | `zarm_l7_end_effector` | No (added for manipulation) |
| `right_wrist_camera` | `zarm_r7_end_effector` | No (added for manipulation) |

The head and waist mounts match the physical sensor locations already
authored in the Kuavo5 USD (`head_camera_base`, `waist_camera_base`). The
wrist cameras are not part of the real hardware; they were added so both
hands stay observable at close range while manipulating totes, looking
outward along the gripper's reach direction (local -Z at the end-effector
link, since the arm hangs down from the wrist joint in Kuavo's default
standing pose).

`scene.py` spawns the same four cameras (as `head_camera`, `waist_camera`,
`left_wrist_camera`, `right_wrist_camera`) for visual inspection, even
though the oracle demo does not read from them. Because camera sensor prims
are now always part of both scene configs, `--enable_cameras` is forced on
internally in `scene.py`; omitting it previously left the camera-adjacent
joints without valid bodies and crashed articulation initialization.

When running with a GUI (i.e. without `--headless`), each of these cameras
also opens as its own small tiled viewport window (320x240 by default, 
stacked from the main viewport's top-left corner) via `camera_viewports.py`.
This is a pure visual convenience for interactive inspection and has no
effect on headless runs or on the `Camera` sensor observations themselves.

Run two randomized environments:

```bash
./run_manager_env.sh --headless --device cuda:0 --rendering_mode performance \
  --num-envs 2 --steps 240 --seed 17
```

Open one environment for your own visual inspection:

```bash
./run_manager_env.sh --num-envs 1 --steps 100000
```

The moving worker and AMR assets are packaged under
`src/kuavo_isaaclab_scene/assets/`.

## Loose cargo and spill safety

Each of the six task totes contains two loose rigid objects, for 12 cargo
objects total. They are not welded to the tote. The open tote uses five compound
colliders, high solver iterations, low restitution, and damped cargo dynamics.

Cargo retention is evaluated in the coordinate frame of its assigned tote. An
item outside the inner wall bounds or above the rim triggers `cargo_spill`.
Policies are rewarded for keeping totes upright and angular velocity low, so
fast but jerky transport is not treated as successful.

## Tests

```bash
"${ISAACLAB_PYTHON}" -m pytest -q
```

The tests cover direct placement, queue pushing, full conveyor blocking,
multi-worker reservation, early-button rejection, and a complete six-tote task
with pre-filled foreign boxes.

## Verified locally

- task-system unit tests: 6 passed;
- physical button contact check: 18.00 mm maximum travel, press detected,
  gated task/conveyor completion detected, and spring return detected;
- original PhysX task flow: 720 steps, six totes loaded, button accepted, belt
  started;
- manager-based initialization: action/observation/event/reward/termination and
  curriculum managers all parsed successfully;
- two parallel randomized environments: 240 manager steps, no idle cargo spill,
  no mover collision, mean cargo retention 1.0;
- all four local USD assets parsed with their expected collision/articulation
  schemas.

The remaining Kuavo warnings concern the pre-existing `head_camera_base` and
`head_radar` inertia values; the robustness workcell assets did not add new
invalid-mass warnings.

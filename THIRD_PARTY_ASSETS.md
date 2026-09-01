# Third-party assets and runtime dependencies

The default robot is Leju Robotics' official
[`biped_s200062`](https://github.com/LejuRobotics/kuavo-ros-opensource/tree/main/src/kuavo_assets/models/biped_s200062)
asset at revision `5d60846b092b425a7a3c06479bdfdbc2b100e890` (subtree
`1836a5aa13ab879ca34da7040efd9a31a60ebbe2`). The complete upstream model
directory is stored under `assets/kuavo_s200062/`, including its URDF, STL,
config, RViz, and MuJoCo XML files. `biped_s200062.source.urdf` preserves the
upstream URDF; the runtime copy changes only local mesh paths and repairs the
upstream head-camera joint's missing `camera_base` link reference to the
existing `head_camera_base` link. The generated fixed-base USD is also local.

For comparison this repository also includes the Kuavo URDF/STL files from
Leju Robotics' official
[`biped_s63`](https://github.com/LejuRobotics/kuavo-ros-opensource/tree/main/src/kuavo_assets/models/biped_s63)
asset and the USD generated from that URDF for Isaac Lab. The pinned source
revision is `ff063125fe9bd070ab199f83a349439e04a8fd16`. The checked-in URDF
changes the ROS `package://` mesh prefix to repository-relative paths so the
converter works without a ROS workspace. Although the official S63 URDF
defines no separate dexterous-finger links, its `*_hand_pitch_noHand.STL`
files contain baked dexterous-hand geometry. This port therefore selects the
finger-free `l_hand_pitch.STL` and `r_hand_pitch.STL` alternatives shipped in
the same official S63 directory. The packaged source is deliberately limited
to the 29 meshes referenced by the adapted URDF; unused hand/tool meshes are
not copied.

This repository also packages the official
[`biped_s56`](https://gitee.com/leju-robot/kuavo-ros-opensource/tree/master/src/kuavo_assets/models/biped_s56)
URDF and the 35 STL meshes used by the runtime model, pinned to Gitee revision
`4d7b0ee8441ed6588ac0134daae05e8a5c78a4e7`. The upstream package manifest
declares the asset package as BSD. The runtime URDF changes ROS
`package://kuavo_assets/models/biped_s56/meshes/` references to local relative
paths. As with S63, the upstream `*_hand_pitch_nohand.STL` visuals contain a
baked hand without finger joints, so this port selects the supplied
finger-free `*_hand_pitch.STL` files and mounts the packaged Robotiq assets at
the empty wrist frames. Those end-effector frames are rotated by Ry(pi) so the
external grippers point outward. The fixed torso root is spawned at the
upstream MuJoCo home height of 0.98 m. The generated fixed-base USD is packaged
under `assets/kuavo_s56/usd/`.

No S62 geometry, mass, inertia, joint origin, or wheel placement is copied into
the robot. The S63 URDF already carries the requested S62 appearance profile:
opaque white for 28 rendered robot meshes and gray
`(0.611765, 0.658824, 0.670588, 1)` for the radar. The scene therefore applies
no runtime visual-material override. Packaged-resource tests lock both the S63
physical values and this material mapping so a future conversion cannot
silently replace them.
The provided rack and box USD files
are also included to reproduce the workcell. These files remain subject to the
rights and terms of their original owners; this repository does not grant
additional rights to them.

The default scene references NVIDIA's original warehouse and Digital Twin
conveyor USDs from the Isaac Sim/Nucleus asset root at runtime. Isaac Sim,
Isaac Lab, that environment content, NVIDIA CloudXR Runtime, the CloudXR Quest
client, and the CloudXR JavaScript package are **not** redistributed here.
Install and use them under their respective NVIDIA terms. `integrations/cloudxr/`
contains only the local IsaacLab bridge source and a patch for the upstream
Apache-2.0 CloudXR JavaScript sample; its setup script obtains the upstream
source and user-provided package separately.

The default `robotiq_2f85` preset uses the Robotiq 2F-85-style `leju_claw` in OpenLET's
[`leju-kuavo-challenge-cup-2026`](https://gitcode.com/OpenLET/leju-kuavo-challenge-cup-2026/tree/master/src/challenge_cup_simulator/models/biped_s52)
at revision `51b3defaf8c032957647c7aa193d1fa20daef1f3`. Its MJCF mounts one gripper below each `zarm_*7_link`;
the S63 port uses the corresponding `zarm_*7_end_effector` frames so the claw
starts after the wrist housing instead of overlapping it. The source claw
uses eight revolute linkage joints per hand, couples the two drivers with a
tendon/equality constraints, and exposes one 0-255 actuator per hand. The
packaged Isaac/PhysX port preserves the source meshes, link frames, joint
limits, materials, and pad contact boxes. Because URDF/PhysX cannot directly
represent the source MJCF closed loop, its eight tree joints are sent
synchronized targets by one binary action per hand. The original MJCF is kept
beside the port for provenance. The previous S200049 gripper assets and preset
have been removed so the Leju claw is the only packaged hand; `none` remains
available for the handless schema.

Before making a public fork or redistributing a wheel, verify that you have the
right to redistribute every USD, URDF, mesh, texture, robot model, and NVIDIA
runtime component. Generated datasets, videos, checkpoints, and local captured
stages are excluded by `.gitignore`.

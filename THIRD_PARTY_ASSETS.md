# Third-party assets and runtime dependencies

This repository intentionally includes the Kuavo 5 URDF/USD/mesh files and the
provided rack and box USD files required to reproduce the workcell. Those files
remain subject to the rights and terms of their original owners; this repository
does not grant additional rights to them.

The scene can also reference NVIDIA Isaac Sim/Nucleus assets at runtime. Isaac
Sim, Isaac Lab, Omniverse content, NVIDIA CloudXR Runtime, the CloudXR Quest
client, and the CloudXR JavaScript package are **not** redistributed here. Install
and use them under their respective NVIDIA terms. `integrations/cloudxr/`
contains only the local IsaacLab bridge source and a patch for the upstream
Apache-2.0 CloudXR JavaScript sample; its setup script obtains the upstream
source and user-provided package separately.

Before making a public fork or redistributing a wheel, verify that you have the
right to redistribute every USD, URDF, mesh, texture, robot model, and NVIDIA
runtime component. Generated datasets, videos, checkpoints, and local captured
stages are excluded by `.gitignore`.

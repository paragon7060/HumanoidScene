# CloudXR.js local IsaacLab integration

This directory contains the small source addition and patch that add the
`Local Kuavo IsaacLab (IWER/Quest)` backend to NVIDIA's CloudXR JavaScript
sample. It forwards WebXR head/hand/controller tracking to
`preview_quest_browser.sh` on WebSocket port 8765 and renders the returned Kuavo
camera composite.

Run from the HumanoidScene repository:

```bash
export CLOUDXR_NPM_TGZ=/absolute/path/to/nvidia-cloudxr-6.2.0.tgz
./setup_quest_browser.sh
npm --prefix .external/cloudxr-js-samples/simple run dev-server
```

The setup script clones the upstream Apache-2.0 sample at pinned commit
`29941936e90234a06847ba1c209d70f60b6b59bd`, applies the patch, copies
`LocalIsaacLabBridge.ts`, and installs the user-provided NVIDIA package. The
upstream source, `node_modules`, build output, and NVIDIA package are kept under
`.external/` or outside this repository and are ignored by Git.

`--patch-only` performs no npm installation. `CLOUDXR_JS_SAMPLES_DIR` can point
to a separate clean checkout at the pinned commit.

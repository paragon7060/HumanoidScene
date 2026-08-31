# CloudXR.js local IsaacLab integration

This directory contains the small source addition and patch that add the
`Local Kuavo IsaacLab (IWER/Quest)` backend to NVIDIA's CloudXR JavaScript
sample. It forwards WebXR head/hand/controller tracking to
`preview_quest_browser.sh` on WebSocket port 8765 and renders the returned Kuavo
camera composite.

This backend is a preview and does not record datasets. For NVIDIA downloads,
the OpenXR runtime manifest, Linux service prerequisites, and actual Quest
collection, start with the [main README walkthrough](../../README.md#quest-collection).
Actual OpenXR collection uses the sample's `Manual Input IP:Port` backend and
the CloudXR Runtime signaling endpoint, not the local preview port 8765.

`runtime_service.cpp` hosts the separately installed NVIDIA Runtime SDK through
its C API. `run_cloudxr_runtime.sh` builds and launches it. The advertised endpoint
defaults to loopback, but Runtime 6.2.1 can still bind signaling on all interfaces;
use only on a trusted LAN. It supports native WSS using PEM files; Runtime 6.2.1 expects the
PEM contents in its property API, so the host reads the files without logging
the private key. See [runtime and HTTPS operation](../../docs/QUEST_RUNTIME_SERVICE.md).

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

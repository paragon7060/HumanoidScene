# CloudXR.js local IsaacLab integration

This directory contains the small source addition and patch that add the
`Local Kuavo IsaacLab (IWER/Quest)` backend to NVIDIA's CloudXR JavaScript
sample. It forwards WebXR head/hand/controller tracking to
`preview_quest_browser.sh` on WebSocket port 8765 and renders the returned Kuavo
camera composite.

The browser also sends tracked controller grips, thumbstick axes and trigger
values. The Python preview adapts WebXR's down-positive Y axis to the shared
OpenXR body mapper, enabling the same base translation/rotation and torso lift
as collection. Missing/stale controller input stops the base and holds torso
height. Protocol v2 remains compatible with older clients, but those clients
cannot move the base until rebuilt. Thumbstick indices 2/3 and trigger index 0
follow the [WebXR Gamepads mapping](https://immersive-web.github.io/webxr-gamepads-module/#xr-standard-gamepad-mapping).

After updating this integration, run `./setup_quest_browser.sh --patch-only`
and `npm --prefix .external/cloudxr-js-samples/simple run build`, then reload
the Quest page and restart the preview. No simulator is needed for the input
extraction check: `node tests/test_browser_controller_packet.cjs`.

This backend is a preview and does not record datasets. For NVIDIA downloads,
the OpenXR runtime manifest, Linux service prerequisites, and actual Quest
collection, start with the [Quest quick-start guide](../../docs/QUEST3_QUICKSTART.md).
Actual OpenXR collection uses the sample's `Manual Input IP:Port` backend and
the CloudXR Runtime signaling endpoint, not the local preview port 8765.

The Quest certificate patch opens the certificate link in the same tab, avoiding
the sample's new-tab navigation when the headset browser blocks it. Certificate
review remains manual; use Back to return to the client after reviewing it.

The workcell defaults patch selects `Manual Input IP:Port`, H.264, 72 FPS and
80 Mbps. It also migrates the browser's saved upstream AV1/90 FPS settings once.
The RTX 3060 is not on CloudXR Runtime 6.2.1's GPU allowlist and its encoder
cannot serve the upstream AV1 default. Per-eye resolution is left unchanged.

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

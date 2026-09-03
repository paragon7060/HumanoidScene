// Run with: node tests/test_browser_controller_packet.cjs
// Uses the already installed sample TypeScript compiler; no browser or GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../.external/cloudxr-js-samples/simple/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname, '../integrations/cloudxr/LocalIsaacLabBridge.ts'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const context = { exports: {}, WebSocket: { OPEN: 1 } };
vm.runInNewContext(compiled, context);
const { trackedController, LocalIsaacLabBridge } = context.exports;
const transform = { position: { x: 0, y: 1, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 } };
const controller = {
  handedness: 'left', gripSpace: {},
  gamepad: { mapping: 'xr-standard', axes: [.9, .8, -.5, -1], buttons: [{ value: .75 }] },
};
const frame = { getPose: () => ({ transform }) };
const parsed = trackedController(frame, controller, {});
assert.deepEqual(Array.from(parsed.thumbstick), [-.5, -1]); // not touchpad axes
assert.equal(parsed.trigger, .75);
assert.equal(trackedController(frame, { ...controller, hand: {} }, {}), null);
assert.equal(trackedController({ getPose: () => null }, controller, {}), null);
assert.equal(trackedController(frame, { ...controller, gamepad: null }, {}), null);
assert.equal(trackedController(frame, { ...controller, gamepad: { mapping: '' } }, {}), null);
const empty = trackedController(frame, { ...controller, gamepad: { mapping: 'xr-standard', axes: [], buttons: [] } }, {});
assert.deepEqual(Array.from(empty.thumbstick), [0, 0]);
assert.equal(empty.trigger, 0);

let message;
const bridge = Object.create(LocalIsaacLabBridge.prototype);
bridge.socket = { readyState: 1, send: data => { message = JSON.parse(data); } };
bridge.referenceSpace = {};
bridge.trackingSequence = 0;
const trackingFrame = {
  ...frame,
  getViewerPose: () => ({ transform, views: [] }),
  session: { inputSources: [controller, { ...controller, handedness: 'right' }] },
};
bridge.sendTracking(10, trackingFrame);
assert.deepEqual(message.left_controller.thumbstick, [-.5, -1]);
assert.equal(message.right_controller.trigger, .75);
assert.ok(message.left_hand.wrist);
bridge.sendTracking(11, { ...trackingFrame, session: { inputSources: [] } });
assert.equal(message.left_controller, null);
assert.equal(message.right_controller, null);
console.log('Browser controller extraction and tracking serialization passed.');

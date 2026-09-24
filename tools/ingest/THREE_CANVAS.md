# Live Three.js scenes on an owned WebGPU canvas

`createGpuThreeCanvas` connects the existing source-owned scene bridge to a real
canvas. It negotiates a WebGPU device when one is not supplied, configures an sRGB
presentation view, creates matching depth/MSAA attachments, and owns their
resize and disposal lifecycle. It adds no animation loop or retained renderer.

```js
import * as THREE from './pinned-three/build/three.core.js';
import {createGpuThreeCanvas} from './three_canvas.mjs';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202030);
const mesh = new THREE.Mesh(new THREE.BoxGeometry(),
  new THREE.MeshPhongMaterial({color: 0x3876cc}));
scene.add(mesh, new THREE.AmbientLight(0xffffff, Math.PI));
const camera = new THREE.PerspectiveCamera(60, 2, 0.1, 100);
camera.position.z = 4;
const renderer = await createGpuThreeCanvas(canvas, scene, {
  three: THREE,
  powerPreference: 'high-performance',
  target: {sampleCount: 4, maxBytes: 128 * 1024 * 1024},
  scene: {renderer: {maxDraws: 1024, instancing: true, renderBundles: true}},
});
renderer.setSize(800, 400, devicePixelRatio);

// Inside the application's existing frame callback:
mesh.rotation.y += 0.01;
renderer.render(camera); // Immediately submits with current canvas/depth views.

// Structural changes still have an explicit asynchronous preparation boundary.
mesh.material.flatShading = true;
await renderer.prepare();
renderer.render(camera);
await renderer.whenIdle();
renderer.dispose();
```

The caller supplies the pinned r186 module and retains ownership of its Scene,
camera, materials, geometries and borrowed textures. All source admission and
material restrictions in `THREE_SCENE.md` continue to apply. This is an explicit
asynchronous factory, not a replacement for a Three.js renderer constructor,
not automatic application specialization and not complete Three.js parity.

## Device and pipeline selection

Pass `device` to borrow an existing device; the session never destroys it.
Otherwise the session requests an adapter and owns its requested device, including
cleanup after initialization failure or disposal. `gpu` can supply a browser
GPU interface explicitly, for example in a worker. It defaults to `navigator.gpu`.
No adapter means `GPU_CANVAS_ADAPTER`; no GPU interface means
`GPU_CANVAS_UNAVAILABLE`. This explicit path does not silently create a WebGL
renderer or change the application's backend choice.

`requiredFeatures` is an array of feature names and `requiredLimits` is a map of
numeric limits. Both are checked before device creation and again on the device.
Minimum alignment limits use the opposite comparison from maximum capacities.
Caller-supplied devices must already satisfy the requested capabilities. A
preferred power profile is a request, not proof of a particular GPU or speed.

`target.format` is `rgba8unorm` or `bgra8unorm`; omission uses the GPU interface's
preferred canvas format, or `bgra8unorm` for a borrowed device without that
interface. The actual renderer format is the compatible `-srgb` view format, so
linear material output receives the display transfer function. Depth is
`depth24plus` by default, with `depth32float` and `null` also available.
`sampleCount` is 1 or 4. A four-sample target resolves into the current canvas
texture. Source pipeline options may not contradict those attachment settings.

The direct path uses **opaque, sRGB canvas compositing**. Premultiplied/transparent
canvas composition, HDR output, tone mapping and Display-P3 are not implied.
Transparent *scene materials* retain the existing renderer's blending behavior.

## Sizing, frames and lifecycle

`resize(width, height)` sets drawing-buffer dimensions in pixels.
`setSize(width, height, pixelRatio=1)` floors logical dimensions times the supplied
ratio. Neither method changes CSS, camera aspect/projection, nor observes layout.
The application owns those choices. A zero width or height suspends submission;
`lastFrameRendered` is false after a suspended render. Resume with a nonzero size.
Changing canvas dimensions directly requires a matching `resize()` before drawing.

A fresh presentation texture/view is acquired on each render. Views are never
cached across browser turns. Same-size resizing allocates nothing. Used old
attachments survive resizing until a queue completion fence retires them; ordinary
frames add no queue-completion fence in this owner. `target.maxBytes` bounds
old-plus-new attachment payload, including pending retirees. Depth24plus uses a
conservative four-byte texel charge. Browser swapchain and driver-private storage
are outside that owned payload estimate. Budget errors occur before allocation;
retrying after `whenIdle()` can release retired capacity.

`render(camera, frameOptions)` forwards explicit clear/load and other existing
scene-frame options, but rejects attachment overrides. Its work is synchronous.
`prepare()` excludes concurrent rendering/resizing and preserves the existing
scene bridge's retryable source-admission errors. Native validation, resource
failure and device loss are terminal. `whenIdle()` observes cumulative validation
and submitted work. Disposal ends pending waits, unconfigures the canvas, retires
owned resources and destroys only an owned device. It is idempotent.

An optional `signal` controls initialization **and the resulting session lifetime**.
Aborting pending adapter/device/pipeline work ends the caller's wait; it is not a
claim to cancel a native GPU promise. Devices/renderers that arrive late are
released without publication. A borrowed device stays alive. Synchronous abort
delivery during a frame releases ownership after that submission boundary exits.
A new renderer must be created explicitly after terminal loss; no automatic
backend migration or replay is performed.

## Deployment

Direct users import `three_canvas.mjs`. The existing `buildAnimation` option
`{webgpu:true, threeScene:true}` now includes all three canvas modules and exports
`createGpuThreeCanvas`, `createGpuCanvasTarget`, `createGpuCanvasRenderer` and
`GpuCanvasError` from `gpu_playback.mjs`. The existing CLI flags are
`--animation-webgpu --animation-three-scene`. CPU-only and ordinary GPU builds
do not acquire these optional modules. Importing a generated entry requests no
device and configures no canvas.

`tests/e2e/three_canvas/index.html` exercises actual source objects, sRGB pixels,
single/four-sample rendering, resize/suspension/resumption, structural preparation
and owner replacement. Serve the repository root over localhost or HTTPS with
the pinned Three.js build present. Its result is exposed as
`window.__f3dCanvasResult`. Missing WebGPU is blocked, not passed. The synchronous
pixel captures belong only to this correctness probe, not a production frame loop.

## Lower-level composition

`createGpuCanvasTarget(device, canvas, options)` from `gpu_canvas.mjs` owns only
the canvas attachments. Its frozen `rendererOptions` contains the required format,
depth format and sample count. `withFrame(attachments => ...)` lends current views
for **synchronous, immediately submitted** work and returns false when suspended.
Do not retain these views, yield, or return a Promise. Detecting a thenable does
not cancel work already started by the caller. Do not configure the canvas from
another owner while this target is live.

`createGpuCanvasRenderer(canvas, factory, options)` from `gpu_canvas_renderer.mjs`
adds device negotiation and renderer lifetime ownership. The factory receives
`(device, rendererOptions)` and returns an owner with `render(input, frame)`,
`dispose()`, `whenIdle()` and optional `prepare()`. This can adapt other existing
explicit GPU passes without another device owner or animation scheduler.

The Node tests run the actual canvas/session modules against native API recorders.
The source-factory wiring tests substitute the existing scene factory explicitly;
they do not certify source-scene rendering or execute shaders. Native pixels,
browser behavior and speed remain separate validation requirements. No measured
acceleration or full Three.js renderer compatibility is claimed.

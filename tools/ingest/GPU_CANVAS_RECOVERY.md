# Explicit canvas device-loss recovery

`createRecoverableGpuCanvasRenderer` reconstructs the existing device/canvas/
renderer ownership stack after a real device loss. It is an opt-in application
lifetime owner, not a replacement renderer, automatic retry loop, retained-backend
fallback, or source-state checkpoint system.

```js
import {createRecoverableGpuCanvasRenderer} from './gpu_canvas_recovery.mjs';

// Keep the application scene, camera, animation controller and asset sources
// outside this factory. Recreate only device-local resources inside it.
const canvasRenderer = await createRecoverableGpuCanvasRenderer(
  canvas,
  async (device, attachments, {signal, generation, recoveryAttempt}) => {
    return createYourRenderer(device, attachments, {signal});
  },
  {
    requiredFeatures: [],
    requiredLimits: {},
    target: {sampleCount: 4, depthFormat: 'depth32float'},
    maxRecoveryAttempts: 3,
  },
);

canvasRenderer.render(camera); // Same synchronous frame contract as before.

// In the application's existing lifecycle handler, not a second frame loop:
await canvasRenderer.whenLost();
await canvasRenderer.recover();
canvasRenderer.render(camera); // The application explicitly submits a new frame.
```

Loss alone never requests another adapter, invokes the reconstruction factory,
replays the last frame, or advances an animation clock. After loss the old
renderer, canvas attachments and owned device are retired, and the old generation's
pending operations reject. `recover()` performs one explicit attempt. Successful
recovery publishes the new fully initialized session, then rendering may resume.

## Reconstruction contract

The renderer factory has the same contract as `createGpuCanvasRenderer`: it
receives a borrowed device, immutable attachment configuration and a lifetime
signal, and returns `render`, `dispose` and `whenIdle` methods with optional
`prepare`. It must reconstruct all device-local pipelines, buffers, bindings,
textures and renderer state from retained application data. Resources from the
lost device cannot be lent to the replacement. The factory context additionally
contains the candidate `generation` and `recoveryAttempt`. A generation's signal
is aborted when it is retired, including unsuccessful reconstruction.

Initialization waits for the existing canvas and renderer validation boundaries.
Each successful publication increments `generation`, starting at one. Failed
attempts do not increment it, so several factory invocations can receive the same
candidate generation; `recoveryAttempt` distinguishes attempts. The initial attempt
is zero. The configured capability requirements, target settings and factory remain
fixed across retries, with independent snapshots of option arrays and records.
Native devices, GPU providers and source data remain borrowed objects, not clones.

Recreating GPU resources does not restore arbitrary user JavaScript effects,
external subscriptions, network responses or application state hidden inside a
factory. Keep the source scene, animation clock and reloadable asset data outside
the factory. A lost GPU-only render target or simulation buffer needs an
application-specific checkpoint/reinitialization policy; this owner does not
invent one or copy lost bytes back from the device.

## Device ownership and capabilities

For an initially negotiated device, `recover()` requests a **fresh adapter and
device**, revalidating every required feature and limit. A weaker replacement is
refused instead of silently lowering requirements, changing the backend, or
claiming a restored frame. The device is destroyed on retirement only when the
underlying canvas session owns it.

For a borrowed device, provide `recover({device: replacement})`. The replacement
must be a different device that has not already participated in this owner's
renderer construction. Neither borrowed device is destroyed. Calling `recover()`
without a replacement fails before consuming an attempt or requesting capabilities.
Explicit `recover({device: null})` opts into fresh negotiation and ownership, even
when the previous generation borrowed its device. Conversely, an owned session
can recover onto a fresh borrowed device. The default for subsequent recoveries
follows the **last successfully published** ownership mode.

A device must expose a loss notification. Only an observed loss authorizes
recovery. Ordinary invalid frame arguments remain correctable frame errors.
Terminal renderer, shader and queue failures remain terminal: destroying an owned
device during that failure's cleanup must not turn the original error into a
recoverable loss. External lifetime abort and explicit disposal are permanent,
not recoverable interruptions.

## State and application coordination

`state` is `ready`, `lost`, `recovering`, `failed` or `disposed` after the initial
factory returns. `generation`, `recoveryAttempts`, `maxRecoveryAttempts`,
`recoverable`, `lastLoss` and `lastError` describe the lifecycle. `lastLoss` is an
immutable `{generation, recoveryAttempt, reason, message}` record for the most
recent published generation's loss. A later failed reconstruction updates
`lastError` but does not rewrite that original loss record.

`whenLost()` returns the last published generation's loss promise. It resolves
on native loss and rejects on terminal failure, abort or disposal. While a
replacement is still being initialized, it continues to describe the previous
lost generation. Only successful publication installs the new generation's
pending loss promise. There is no background polling or notification callback.

The usual `render`, `resize`, `setSize`, `prepare`, `whenIdle`, `dispose`, `canvas`,
`device`, `ownsDevice`, `rendererOptions`, `width`, `height`, `suspended` and
`lastFrameRendered` surface remains available. Native handles are generation-local;
do not cache the `device` getter across a recovery. `diagnostics.session` contains
the current underlying owner's diagnostics, or null when there is no live owner.
`lastFrameRendered` becomes false on loss/reconstruction and only a subsequent
successful render can make it true.

Render/prepare/idle operations cannot use a lost or initializing generation.
Concurrent recovery is refused. Old asynchronous preparation and idle completions
cannot succeed as operations on the new generation, clear its preparation lock,
stop it, or unconfigure its canvas. Cancellation rejects promptly even when an
adapter, device or renderer promise has not settled; a device/renderer that arrives
late is retired by its original owner. Cancellation does not promise that a
browser's underlying request or arbitrary user promise is cancellable.

The wrapper retains a logical lease on its canvas while lost or recovering.
Another recoverable owner must wait for disposal. A separately constructed plain
canvas target that legitimately claims the otherwise unconfigured canvas causes
recovery to fail ownership admission; it is never stolen or unconfigured. The
existing canvas target's identity token also protects replacements against late
cleanup from older generations.

## Dimensions, limits and failed attempts

The most recent successful drawing-buffer size is preserved, not the original
constructor size. While lost, `resize`/`setSize` may queue new dimensions without
allocating GPU resources or changing the canvas element yet. The replacement
validates those dimensions against its own limits and budgets; no downscaling is
performed. A zero-sized canvas remains suspended across recovery. Resizing during
active reconstruction is refused, so its initialization snapshot stays coherent.

`maxRecoveryAttempts` defaults to three and must be an integer from zero through
64. It counts explicit retries, successful or not, and excludes initial
construction. Malformed options, missing borrowed replacements and concurrent
calls fail before consuming an attempt. A failed actual reconstruction consumes
one attempt, retires its partial ownership, leaves `generation` unchanged and
returns to `lost` for another explicit attempt within the budget. Exceeding the
budget does not create requests or resources. There are no timers, backoff or
hidden retry loops.

Resource budgets remain those of the existing canvas/renderer owners. Driver
retirement after device destruction, user-retained snapshots and external source
assets are not estimates of this wrapper's owned GPU memory. Repeated attempts
can temporarily retain unresolved host promises, bounded here by the attempt
limit rather than a guarantee of host cancellation.

## HDR and source-owned Three.js canvases

`createRecoverableGpuHdrCanvasRenderer(canvas, createRenderer, options)` applies
this same lifecycle to the existing opaque whole-image HDR canvas stack. Its
factory receives the linear `rgba16float` scene attachment profile, not the final
canvas format. `renderTarget` still controls scene depth/MSAA and texture budget;
`output` still controls the default tone-mapping operator and exposure. Recovery
recreates the linear color/depth/MSAA targets, output shader pipeline, uniform
buffer and bind groups on the replacement device. It does not preserve lost HDR
history, replay previous frame output overrides, or introduce per-material
`toneMapped` semantics. The HDR module is loaded only when its factory is called.

For the admitted source-owned Three.js scene path, the convenience factories
`createRecoverableGpuThreeCanvas` and `createRecoverableGpuThreeHdrCanvas` retain
the exact caller Scene, camera and pinned Three module while rebuilding the
existing `createGpuThreeScene` owner. They accept the existing source-canvas
options plus `maxRecoveryAttempts`:

```js
import {createRecoverableGpuThreeHdrCanvas} from './three_canvas_recovery.mjs';

const view = await createRecoverableGpuThreeHdrCanvas(canvas, scene, {
  three: THREE, // the pinned r186 source module remains caller-owned
  scene: {
    renderer: {maxDraws: 2048, renderBundles: true},
    texture: {maxTextureBytes: 128 * 1024 * 1024},
    deformation: {maxJoints: 256},
  },
  renderTarget: {sampleCount: 4, depthFormat: 'depth32float'},
  output: {toneMapping: 'agx', exposure: 1},
  maxRecoveryAttempts: 3,
});

// The application retains its existing animation/frame loop and camera.
view.render(camera);
await view.whenLost();
await view.recover(); // pass {device: freshDevice} for a borrowed-device owner
view.render(camera);
```

Only source-backed automatic texture ownership is admitted by these convenience
factories. A supplied `scene.textures` binding map (even an empty map), or
`autoTextures:false`, is refused before device acquisition. Opaque native views
and samplers cannot be carried across devices. Applications with custom borrowed
texture uploads should instead use the generic reconstruction factory, create
new native resources for its new device, and lend those new bindings to their
renderer. This is not permission to reuse handles from the lost generation.

Source reconstruction reads the **current** source geometry, attributes, texture
pixels, poses and scene structure. It does not advance a retained AnimationMixer
or controller, reload URLs, decode images, clone identities or dispose the
application's scene. Data must remain available in the existing admitted source
profiles. CPU-side changes that were never uploaded before loss can consequently
appear in the reconstructed generation; recovery is not an exact restoration of
the previous GPU-visible snapshot. GPU-only data, discarded image sources and
external input effects still require an application-specific reconstruction policy.

Source preparation budgets and feature options are copied before the first
asynchronous construction; native source objects remain live. Attachment format,
depth and sample-count conflicts are errors, not silently repaired settings. The
separate original `three_canvas.mjs` factories retain their existing behavior;
recovery is explicitly selected, not enabled by a global renderer patch.

## Generated-package deployment

The build API's `canvasRecovery:true` switch requires `webgpu:true`. It exports
both generic recovery factories from `gpu_playback.mjs` and includes their direct
and HDR dependencies. With `threeScene:true`, it also exports both source recovery
factories. The new switch does not itself enable IBL, background rendering,
external HDR decoding, retained Three components, or native services at import:

```js
buildAnimation('actor.gltf', 'dist/actor', {
  webgpu: true,
  threeScene: true,
  canvasRecovery: true,
});
// import {createRecoverableGpuThreeHdrCanvas} from './dist/actor/gpu_playback.mjs';
```

Already included source/HDR modules are deduplicated. Every emitted dependency
participates in manifest hashes and exact byte-budget admission before the output
directory is created. CPU-only/default GPU output and explicitly disabled recovery
packages gain no recovery modules or exports. This flag changes available package
entry points, not how the application starts a canvas or chooses to recover it.

## Validation

```sh
node --test tools/ingest/gpu_canvas_recovery.test.mjs \
  tools/ingest/gpu_hdr_canvas_recovery.test.mjs \
  tools/ingest/three_canvas_recovery.test.mjs \
  tools/ingest/gpu_canvas_recovery.package.test.mjs
```

Tests execute the production recovery manager, device negotiator and canvas
attachment owner. A recorded GPU/renderer boundary checks resource identity,
submission, capabilities, retirement, cancellation, ownership and failure ordering.
It is not native browser/GPU execution, a pixel-equivalence test, a performance
measurement, or a guarantee that an arbitrary source factory can reconstruct all
of its lost data.

HDR tests execute the production intermediate-target and output-pass modules as
well as the native owners. Source adapter tests substitute the source-scene factory
boundary to isolate retained identity, current data and option forwarding. Package
tests execute the actual builder, move its output away from the original toolkit,
and import its emitted entry to render, lose, reconstruct and resume. Their model
decoder/pose metadata and unrelated mesh/source factory remain explicit fixtures.
Artifact hashing, exact/one-byte-short budget admission and feature isolation are
checked. These tests do not claim native GPU pixels or full Three.js scene coverage.

# Depth-only render-target transfers

Depth copies now run through the same target ownership and renderer submission
FIFO as color copies, draws and readbacks. They use a depth-only GPU render pass:
no CPU readback, intermediate color image, filtering, scaling or implicit resolve.
Color and stencil are not written. Pixels outside the requested rectangle retain
their destination depth.

## Renderer API

```js
// Source targets need attached DepthTextures for this ordinary texture API.
// Both depth formats must match, and both targets must be single-sample.
await renderer.copyTextureToTextureAsync(
  sourceTarget.depthTexture,
  destinationTarget.depthTexture,
  sourceBox2,          // null means the full source depth attachment
  destinationVector2 // null means (0, 0)
);

// F3D extension: a rectangular depth-only copy with explicit completion.
await renderer.copyRenderTargetDepthPixelsAsync(sourceTarget, destinationTarget, {
  sourceX: 0, sourceY: 0,
  x: 0, y: 0,
  width: 128, height: 128,
  signal: abortController.signal,
});
```

`copyTextureToTexture()` is the void source counterpart. It validates source
arguments synchronously; failures discovered after enqueue/submission surface at
the next renderer error boundary. Its Async counterpart returns the completion
promise. No source texture flags, versions or data are rewritten to route a copy.
In particular, DepthTexture ownership is checked through `renderTarget.depthTexture`,
not the color-only `isRenderTargetTexture` flag.

The extension requires an attached source DepthTexture; its destination may have
private, non-sampled depth storage. Different depth formats are rejected unless
`allowFormatConversion: true` is explicitly supplied to the extension. Conversion
uses the destination depth format's representable precision. The source-style
texture API does not implicitly enable conversion.

Coordinates follow the selected renderer surface: WebGPU top-left or WebGL
bottom-left. `flipY` explicitly overrides this convention for the extension. It
changes both region origins, not the order of copied rows. Coordinates and target
generations are captured when called; GPU contents are read at the operation's
ordered submission. Later Box2/Vector2/options edits cannot retarget a pending copy.
Target resize, replacement or disposal rejects a queued use of its old generation.

The existing source RenderTarget profile remains single-sample when a DepthTexture
is attached. This change does not silently broaden that profile to MSAA, cube or
array depth textures, comparison sampling, or nonzero depth mips.

## Native target API

```js
const source = createGpuRenderTarget(device, {
  width: 128, height: 128, depthFormat: 'depth32float', sampleDepth: true,
});
const destination = createGpuRenderTarget(device, {
  width: 128, height: 128, depthFormat: 'depth32float', copyDestination: true,
});

destination.copyDepthFrom(source); // submits synchronously; returns true
await destination.whenIdle();      // includes source validation and copy completion
```

Native depth formats: `depth16unorm`, `depth24plus`, `depth32float`,
`depth24plus-stencil8`, and feature-enabled `depth32float-stencil8`. Copies require
native handles on the same GPUDevice and distinct source/destination targets.
The source must have `sampleDepth: true`; the destination must opt into
`copyDestination: true`. Neither depth attachment needs COPY_SRC/COPY_DST usage:
the source is sampled with `textureLoad`, and the destination is a depth attachment.

Native sample profiles are explicit:

- One sample to one sample copies that sample.
- Four samples to four samples preserves all four independently.
- Four samples to one sample requires `sampleIndex: 0`, `1`, `2`, or `3`.
- One sample to four samples is rejected; samples are not implicitly replicated.

No sample averaging, depth reconstruction, camera-space conversion or depth-range
remapping is inferred. `allowFormatConversion` is required for different formats.
Zero-width/height copies return false without GPU allocation or submission.

## Ownership and completion

The source target owner checks the combined allocation cost before creating either
missing target. Depth copies invalidate only the destination's depth sampling
replica, not its color replica or color mipmaps. Subsequent sampling refreshes the
existing depth replica on the GPU. Native resize keeps used old attachments alive
until their retirement fence completes.

`maxBytes` bounds the depth footprint of a copy (default 64 MiB). Per-device pending
depth-copy uniforms are limited by `maxInFlightBytes` (default 64 KiB) and 64 pending
copies. Each copy charges 32 bytes. The strictest unfinished caller's byte ceiling
continues to apply to subsequent calls. Pipeline caches are bounded by admitted
depth-format/sample profiles and shared by device, not duplicated per target.
Renderer depth requests also share the existing renderer transfer request limit.

Cancelling before submission prevents the write. Cancelling afterward rejects the
caller's wait but cannot undo submitted depth. The native uniform stays reserved
until the submitted-work fence and validation settle, even after caller cancellation
or an early validation error. Device loss terminates pending native copies. Invalid
pipelines are evicted. Error scopes never remain open across an await.

## Validation boundary

Run the focused production-owner/controller tests with:

```sh
node --unhandled-rejections=strict --test \
  tools/ingest/tests/gpu_depth_transfer.test.mjs \
  tools/ingest/tests/three_depth_transfer.test.mjs \
  tools/ingest/tests/three_source_texture_transfers.test.mjs
```

The tests exercise real transfer planners, source/native target owners, renderer
transfer controller and native readback code, with explicitly documented GPU,
source-descriptor and renderer-queue boundary doubles. They test allocation,
ordering, region capture, sample selection, cancellation, replica invalidation,
stencil-preserving pass descriptors and injected failures. They do not compile WGSL
or establish browser rendering, exact Three.js pixel equivalence, or a speedup.

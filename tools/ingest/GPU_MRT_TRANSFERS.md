# Selected MRT color transfers

The core native target API can upload, copy and read a selected color attachment
without separating it from its shared target owner or allocating a second image.
The attachment's format determines the input/output type and component count:
RGBA/BGRA8 returns Uint8 RGBA, RGBA16F preserves Uint16 half-float bits, and
R/RG/RGBA32F preserves one/two/four Float32 components. No numerical conversion,
blending, scaling or implicit color-space conversion is introduced.

```js
const source = createGpuRenderTarget(device, {
  width: 256, height: 128,
  colorFormats: ['rgba8unorm', 'rgba16float', 'r32float'],
  copyDestination: true,
});
const destination = createGpuRenderTarget(device, {
  width: 256, height: 128,
  colorFormats: ['r32float', 'rgba8unorm', 'rgba16float'],
  copyDestination: true,
});
source.writePixels(halfFloatBits, {colorIndex: 1});
destination.copyFrom(source, {sourceColorIndex: 1, colorIndex: 2});
const halfFloatResult = await destination.readPixels({colorIndex: 2});
await destination.whenIdle();
```

Both selectors default to zero, including for ordinary single-color targets.
Indices must be non-negative safe integers within the attachment vector. Missing
attachments fail before GPU work or staging allocation, even for empty regions.
Depth is shared and has no color selector. Same-owner copies remain rejected.

Copies validate the selected formats, not the primaries. They directly address
GPU textures and add no staging buffer or CPU readback. Copy-compatible sRGB
pairs retain raw bits. A multisampled source supplies the selected attachment's
resolved single-sample texture. Upload/copy destinations still require one
sample. A selected source mip can populate an MRT attachment; existing MRT
storage itself remains 2D and mip-zero. Cube/2D transfers retain explicit face
selection at each cube endpoint, independently of the color attachment selector.

Rectangles, byte offsets, row strides, flipY and optional caller output retain
the existing transfer contracts. Uploads consume their bytes during the call.
Reads and copies submit at the call and capture selection before returning;
later writes, option edits or resize cannot change an accepted read's contents,
format or dimensions. Resizing retains all used attachments until the submitted
work fence. Cancellation/loss/disposal releases readback staging and leaves
caller output untouched when completion has not committed it.

All attachments share the existing owner allocation budget and per-device
readback budget. Charges use the selected format, including 256-byte padded
readback rows. A later read cannot relax an earlier outstanding read's promised
budget. Selecting a different attachment is not a way around request, ownership,
nonreentrancy, feedback or device-identity checks.

## Renderer integration

The explicit renderer APIs use the same attachment selectors and shared FIFO:

```js
// Source RenderTargets retain their ordinary textures[] identities.
// The selected source/destination attachments must have copy-compatible formats.
await renderer.writeRenderTargetPixelsAsync(source, halfFloatBits, {
  colorIndex: 1,
});
await renderer.copyRenderTargetPixelsAsync(source, destination, {
  sourceColorIndex: 1, colorIndex: 2,
});
const pixels = await renderer.readRenderTargetMipPixelsAsync(destination, 0, {
  colorIndex: 2,
});

// Source-shaped copies derive selection from the actual texture objects.
await renderer.copyTextureToTextureAsync(
  source.textures[1], destination.textures[2]
);
// The void copyTextureToTexture form and the storage facade's delegated form
// use the same owner and queue; no compute/storage path is substituted.
```

Indices, rectangles, orientation and upload bytes are captured at the call,
including before initialization or while preparation is pending. Texture copies
capture the source/destination attachment identities and ordering, not merely
an index into a later texture vector. Replacement, reordering, format changes
or disposal invalidate the captured target generation instead of retargeting
accepted work. Source automatic-resize leases retain their original attachment
set until the existing retirement conditions permit release.

Nested write protection checks the selected texture. A child write cannot
replace a secondary attachment already sampled by an earlier parent draw, but
a write to an independent, unsampled sibling does not falsely collide with the
primary attachment. Ordinary same-target copy rejection remains unchanged.

Only writes to attachment zero invalidate the existing primary WebGL sampling
replica. Native MRT inputs borrow the actual per-attachment images and need no
replica or CPU round trip. The source owner's readback session forwards the
complete format vector and color selector to the shared native readback helper.

Renderer upload-byte/request limits, target storage budgets, and native staging
budgets remain shared across attachments. GPU completion and mapping do not
block later queued submissions. Caller/renderer/texture cancellation stops pending
reads without committing caller output; already-submitted writes are not undone.

This extends the explicit `readRenderTargetMipPixelsAsync` readback method.
It does not change the source-shaped `readRenderTargetPixelsAsync` method's
primary-attachment restriction, admit MRT cubes/mip chains, or add additional
WebGL material sampling paths. Depth remains a single shared attachment without
a color selector. Source texture-shaped copies remain 2D; cube endpoints use
explicit target/face APIs. The source facade retains its existing admitted color
formats, which are narrower than the native target's format set.

## Verification

`gpu_mrt_transfer.test.mjs` runs the production target, transfer and readback
modules against a recording native-call boundary with independent byte arrays.
It covers every admitted format in each of three attachment slots, mixed-format
cross-index copies, raw sRGB copies, subregions, strides, output guards, cube
faces, authored mips, resolved MSAA sources, snapshots, shared budgets, rejection,
cancellation, resize, validation errors and device loss. It does not execute
WGSL, rasterize MSAA, establish browser pixel parity or measure performance.

`three_renderer_mrt_transfers.test.mjs` executes the production controller, source
target owner and native modules with structural source-class descriptors and a
controlled queue driver. It exercises selected typed reads, texture-shaped
copies, snapshots, exact attachment hazards, generation invalidation, shared
budgets, cancellation, and native sampling residency. It does not construct the
full renderer class or substitute for pinned-source browser comparison.

```sh
node --test tools/ingest/gpu_mrt_transfer.test.mjs \
  tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/three_render_targets.test.mjs \
  tools/ingest/three_renderer_mrt_transfers.test.mjs
```

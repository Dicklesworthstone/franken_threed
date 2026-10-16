# Render-target inputs for TSL compute

The explicit r186 WebGPU renderer can consume its own rendered **color and depth
attachments** in `compute()` / `computeAsync()`. This connects render-to-compute
workflows without downloading pixels, re-uploading them, or executing an upstream
renderer. It is general WebGPU execution, not a Rust/Wasm or speedup claim.

## Source-facing use

Use an ordinary source `RenderTarget` and, for depth, its owned `DepthTexture`.
A TSL kernel can use `textureLoad(target.texture, coordinates)` and
`textureLoad(target.depthTexture, coordinates)` together. The latter has the native
scalar depth type. Kernels may write their results to the existing storage-buffer
or `StorageTexture` outputs; ordinary target attachments are read-only inputs.

The application-level ordering is:

```js
// renderer is the project's initialized r186 WebGPU renderer surface.
// depthKernel is a source TSL ComputeNode reading target.depthTexture.
renderer.setRenderTarget(target);
renderer.render(scene, camera);
renderer.compute(depthKernel);
renderer.setRenderTarget(null);

// This waits for queued work and its validation, including target producers.
await renderer.waitForGPU();
```

`compileComputeAsync(kernel)` prepares the native input bindings and compute
pipeline without dispatching the kernel. Existing `getArrayBufferAsync()` and
`copyStorageBufferToRenderTargetAsync()` observe input-producer validation failures
as well as compute failures. A compute result can feed a subsequent render using
the existing storage/compute geometry or buffer-to-target routes.

## Supported attachments

Color inputs use the shared target owner's existing single-color 2D profile:
RGBA8 linear/sRGB, linear RGBA16F, and linear R/RG/RGBA32F. Source sample counts zero
and one use one sample. Four-sample color targets supply their resolved primary
texture, not a multisampled texture binding. Generated color mipmaps are refreshed
before a color consumer. Float32 nearest sampling uses explicit unfilterable-float
layouts; linear float32 color sampling requires `float32-filterable`.

Depth inputs use the target-owned single-sample source DepthTexture profile:
UnsignedIntType/DepthFormat (`depth24plus`), FloatType/DepthFormat
(`depth32float`), or UnsignedInt248Type/DepthStencilFormat
(`depth24plus-stencil8`). Bindings use `sampleType: 'depth'` and a `depth-only` view
of mip zero. Depth is not copied through RGBA, half floats or the WebGL bridge's
R32F replica. Ordinary nearest/linear native depth sampling does not require the
float32 **color** filtering feature. Stencil remains owned by the rendering path
and is not exposed as a compute input.

Both attachments of one target can be bound by a single kernel. The native owner
borrows them together once, while multiple targets remain borrowed through the
same dispatch. A depth-only consumer neither generates nor dirties color mips.

The scope remains explicit: comparison depth samplers, multisampled source depth,
array/cube/multiple-color targets, writable ordinary target attachments, decoded
image/video compute inputs, and partial CPU texture snapshots are not enabled by
this feature. The independent native target API can expose multisampled depth;
this does not expand the source target inspector's single-sample depth contract.

## Queue and lifetime semantics

On an initialized renderer, calls capture source target generation and sampler
state before entering the renderer FIFO. They do **not** snapshot GPU pixels.
Render A -> compute A -> render B -> compute B therefore reads A then B, even
when scene preparation blocks submission. Later sampler mutations do not change
already captured calls. Resizing, disposing or replacing a captured attachment
invalidates the queued use instead of silently recapturing another generation.

Before allocating input storage, the complete dispatch is checked against the
existing shared target budget, counting each target once. No duplicate GPU images
are charged or created for compute. A bounded adapter cache holds native samplers;
views are weakly associated with the owned native textures. Source target and
GPU-device ownership stay with the renderer.

The new `withSampledTextures(callback)` native-owner operation borrows resolved
color and sampled depth together under one synchronous lifetime guard. Existing
`withTexture` / `withDepthTexture` remain unchanged in meaning. No borrow may
cross an `await`, and reentrant render, resize, sample and dispose operations are
rejected. Source disposal requested while an upload callback is running releases
attachments only after all native borrows unwind.

Producer allocation/mipmap errors, native binding scopes, device loss and renderer
lifetime cancellation propagate through compute completion. Error scopes are
never left installed across an asynchronous wait. Loss/disposal interrupts pending
completion even when a driver scope or fence never settles.

## Validation and provenance

```sh
node --experimental-vm-modules --test \
  tools/ingest/tests_three_compute_sampled.mjs \
  tools/ingest/three_compute_targets.test.mjs \
  tools/ingest/three_compute_depth_targets.test.mjs \
  tools/ingest/three_compute_targets_native.test.mjs
```

These 46 tests comprise eight existing sampled-input integration cases, 19 color
native-target cases, ten source depth/joint-input cases, and nine cases executing
the real native target owner. Test fixtures use recording GPU objects and a
shader-builder double, not a physical device or the actual pinned WGSL compiler.
Numbers recorded in FIFO tests establish producer selection, not GPU pixel math.
Native tests execute the real allocation/mipmap/borrow implementation; unused
readback and transfer seams throw. All changed JavaScript modules are syntax
checked. Full repository builds and browser pixel comparisons are not claimed.

Source behavior was checked against frozen Three.js commit `148ef33`, specifically
`TextureNode.setupUV`, `WGSLNodeBuilder.getUniforms`, and
`WebGPUUtils.getTextureSampleData`. The native route does not use the WebGL
sampler2D path's vertically flipped replicas. No upstream pin or source file is
changed. Static imports connect the new adapter into the existing runtime module
graph; no separate hand-maintained package file list is introduced.

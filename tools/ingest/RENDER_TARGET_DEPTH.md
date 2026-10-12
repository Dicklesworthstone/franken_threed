# Sampled depth attachments

Part of `ft-173` (generalized render targets). This is new WebGPU execution with
retained JavaScript resource ownership, not Rust specialization or a speedup claim.

## Native offscreen owner

```js
import { createGpuRenderTarget } from './gpu_render_target.mjs';

const target = createGpuRenderTarget(device, {
  width: 512, height: 256,
  format: 'rgba16float', depthFormat: 'depth32float',
  sampleCount: 1, sampleDepth: true,
});
target.withFrame((attachments) => {
  // Render into attachments and submit before returning.
});
target.withDepthTexture((depth) => {
  // This is the SAME depth texture used by attachments.depthView.
  // Create its depth-only view, encode consumers, and submit before returning.
});
await target.whenIdle();
target.dispose();
```

`sampleDepth` defaults to false. Enabling it adds TEXTURE_BINDING usage to the
existing depth attachment; it allocates no second texture, creates no shaders,
and issues no copies or resolves. The callback has the same synchronous,
submitted-use lifetime and reentry restrictions as `withTexture`. Use with no
depth attachment is rejected, including for suspended zero-sized targets.

Native depth formats are `depth16unorm`, `depth24plus`, and `depth32float`.
One- and four-sample attachments are supported. Four-sample depth remains a
multisampled texture: use `texture_depth_multisampled_2d` and an explicit sample
index in WGSL. This API does not choose a depth-resolve policy for the application.
The source renderer's ordinary single-sample depth route is a separate contract.

Depth borrowing neither generates nor acknowledges pending color mipmaps.
Color rendering, mip generation, resolved-color readback, and their existing
APIs remain independent. Peak resize budgets include retired generations; a
depth-only consumer counts as use and keeps its old attachment alive until the
queue fence completes. Disposal, partial allocation failures, validation errors,
and device loss follow the same terminal-state/resource-release rules as color.
Depth16 is charged at two bytes per texel; Depth24plus and Depth32 at four.
These are logical storage budgets, not measurements of driver allocation overhead.

## Source DepthTexture and renderer bindings

```js
const target = new THREE.RenderTarget(512, 256, {
  depthBuffer: true,
  depthTexture: new THREE.DepthTexture(512, 256, THREE.FloatType),
  samples: 0,
});
// On the existing program-capable renderer path, an ordinary ShaderMaterial
// sampler2D uniform can read: texture2D(tDepth, vUv).r
const uniforms = { tDepth: { value: target.depthTexture } };
```

The existing renderer's target-binding Map now recognizes source DepthTexture
objects through their `renderTarget` association, without requiring or mutating
`isRenderTargetTexture` (which r186 sets on color attachments, not depth). The
renderer still requires its existing initialization and scene-preparation steps.
This does not add node/TSL material support or change renderer route selection.

For source targets, `DepthFormat` with `UnsignedIntType` uses `depth24plus` and
`FloatType` uses `depth32float`. The latter is useful when a precise native depth
format is required; Depth24plus's backing representation is implementation-defined.
The target's dimensions define attachment storage, including after `setSize`.
Source depth image metadata remains borrowed and is not rewritten to simulate an
upload. Shared depth attachments, source UnsignedShortType, cube/array/volume
attachments, stencil, comparison samplers, generated/manual depth mipmaps, and
multisampled source depth textures remain explicit refusals. The native API above
can expose multisample depth directly; this source sampler2D path cannot invent a
resolve or sample-index policy for it.

The first read creates a GPU-only, single-level `r32float` sampling image. A
fullscreen textureLoad pass reads the actual rendered depth attachment, flips
its rows to the established source texture UV convention, and stores the loaded
f32 depth value without RGBA8/half-float quantization, color transfer, packing or
CPU readback. The resulting ordinary sampler2D binding returns depth in R,
zero in G/B, and one in A. Use the R component for depth reconstruction. The
native color attachment and its readback session keep their original orientation.
This is an explicit rendering profile, not complete WebGL depth-texture parity.

Nearest filtering needs no optional GPU feature. Either linear minification or
linear magnification requires `float32-filterable` enabled on the supplied device;
there is no silent nearest or lower-precision fallback. Without that feature,
the binding reports `unfilterable-float` and all three WebGPU sampler filters,
including the mip filter, are nearest. The existing program renderer consumes
that binding type. Core material paths which require filterable textures still
need their program route. Depth anisotropy must be one; nearest/linear filters
and the three ordinary wrapping modes are admitted, with LOD clamped to zero.

Depth and color sampling images have independent dirty flags. Another target
render invalidates both, but a depth read neither generates nor acknowledges
color mipmaps, and a color read cannot acknowledge a pending depth refresh.
Repeated depth reads reuse the image, view, bind group and sampler; sampler-only
edits replace only that sampler and acknowledge the current source versions.
`entry.depthCopies` counts submitted depth-orientation passes, not render calls.

The shared source-target budget includes native attachments, complete color
pyramids/replicas when present, and four bytes per depth-sampling-image texel.
A replica-budget refusal preserves existing rendered storage and can be retried.
Attaching, replacing or removing a DepthTexture invalidates queued generations
even when the native format stays depth24plus. Source target, color texture or
depth texture disposal releases the generation and all its sampling resources;
listeners are removed. Reentrant depth sampling is refused. Color and depth
feedback are both rejected before allocation when the source is the current
destination target. GPU errors stop publication, and synchronous sampling errors
are recorded immediately so `whenIdle()` cannot falsely report success after a
failed target has already been retired.

## Validation

```sh
node --test tools/ingest/gpu_render_target.test.mjs \
  tools/ingest/gpu_render_target_mipmaps.test.mjs \
  tools/ingest/gpu_render_target_depth.test.mjs \
  tools/ingest/three_render_targets.test.mjs \
  tools/ingest/three_render_target_mipmaps.test.mjs \
  tools/ingest/three_render_target_depth.test.mjs
```

These tests execute the production owner against native-call recorders. They
check attachment identity, all three depth formats, multisample descriptors,
color/depth independence, budgets, retirement, cancellation and failures. They
are not native shader execution, pixel-parity evidence or performance results.

The source-depth suite additionally checks the r186 association rule, native-depth
input to the R32F pass, independent refreshes, feature-dependent filtering, source
replacement/resizing/disposal, feedback, shared budgets, and injected allocation,
pipeline, bind-group, sampler, submit, validation and device-loss failures. The
implementation session passed 147 tests across these six suites (53 new depth
tests and 94 existing regressions). A Chromium attempt was blocked before GPU
initialization; native shader execution and pixel equivalence remain unverified.

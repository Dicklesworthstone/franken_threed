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

Native depth formats are `depth16unorm`, `depth24plus`, `depth32float`,
`depth24plus-stencil8` and feature-gated `depth32float-stencil8`.
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
Depth16 is charged at two bytes per texel; Depth24plus and Depth32 at four;
combined depth/stencil at eight.
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

For source targets and detached depth inputs, `DepthFormat` with
`UnsignedShortType`, `UnsignedIntType` or `FloatType` uses `depth16unorm`,
`depth24plus` or `depth32float` respectively. `DepthStencilFormat` with
`UnsignedInt248Type` uses `depth24plus-stencil8`; with `FloatType` it uses
`depth32float-stencil8`, requiring that feature on the enabled device.
Depth24plus's backing representation remains implementation-defined.
The target's dimensions define attachment storage, including after `setSize`.
The owner initializes depth image dimensions lazily from the selected target,
retaining the source texture identity. The native renderer retains depth comparison and multisample bindings
under the target owner's dimension rules; shared/layered attachment ownership is
described in `THREE_RENDER_TARGETS.md`. Generated/manual depth mipmaps remain
unsupported. The WebGL sampler2D replica described below cannot invent a resolve
or sample-index policy for multisampled depth.

The first ordinary read creates a GPU-only, single-level `r32float` sampling image. A
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
Repeated depth reads reuse the image, view, bind group and sampler. The WebGL
sampler is captured when the source target first allocates depth. Public edits
without `needsUpdate` retain those installed parameters. An acknowledged
compare-function change updates only the comparison sampler; setting the public
function to `null` keeps an installed comparison mode, as pinned WebGLTextures
does. Changing initialized filters, wrapping, unpack controls or Source identity
with `needsUpdate` is explicitly refused until the target is disposed/recreated:
the source replaces its GL image in that case, so preserving the old attachment
contents would be incorrect.
`entry.depthCopies` counts submitted depth-orientation passes, not render calls.

### Custom comparison shaders

An attached single-sample 2D `DepthTexture` can feed a `sampler2DShadow` in either
`ShaderMaterial` or GLSL3 `RawShaderMaterial`. Set its comparison function and
nearest/linear filters before initializing the target:

```js
target.depthTexture.compareFunction = THREE.LessEqualCompare;
target.depthTexture.minFilter = target.depthTexture.magFilter = THREE.LinearFilter;
// GLSL3 fragment expression, with shadowCoord a vec4:
// float visibility = textureProj(tDepth, shadowCoord);
```

The owner creates a separate UV-oriented `depth32float` image and writes the
loaded attachment value through `frag_depth`. Its native comparison sampler
retains the installed comparison function, filters and wrapping. Linear depth
comparison does not require `float32-filterable`; it uses depth filtering, not
an R32F color sampler. Fixed-point source formats carry a per-binding reference
clamp into the compiler; float depth keeps its unclamped comparison reference.
This applies to ordinary and projected comparisons, including sampler helpers,
arrays and vertex-stage base-level comparisons.

Ordinary and comparison images have independent caches and dirty state. A
versioned transition into comparison mode leaves already issued ordinary views
alive; a new depth write invalidates both copies. Comparison-only reads do not
allocate an R32F image or regenerate color mipmaps. Each allocated replica costs
four bytes per texel in the shared budget and follows the existing generation
leases, resize, disposal, feedback and validation rules. `depthComparison.copies`
counts the comparison orientation submissions. Cube/array comparison replicas
and multisample depth resolves remain outside this WebGL profile; native
node/compute depth bindings retain their existing dimension/sample support.

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
check attachment identity, supported depth formats, multisample descriptors,
color/depth independence, budgets, retirement, cancellation and failures. They
are not native shader execution, pixel-parity evidence or performance results.

`three_depth_sampler_source.test.mjs` executes the pinned WebGLTextures code to
check its sampler acknowledgement and image-replacement behavior.
`three_program_depth_sampling.test.mjs` exercises actual converted and raw
ShaderMaterial consumers, including submitted fixed/float comparison flags,
queued coordinates, copy ordering, warm reuse and source comparison updates.
The projected-shadow browser checkpoints in `program_texture_sampling.mjs`
were added but could not be run because Chromium could not launch in this
environment; these new paths have no fresh pixel or performance result.

The source-depth suite additionally checks the r186 association rule, native-depth
input to the R32F pass, independent refreshes, feature-dependent filtering, source
replacement/resizing/disposal, feedback, shared budgets, and injected allocation,
pipeline, bind-group, sampler, submit, validation and device-loss failures. The
initial implementation passed 147 tests across these six suites (53 new depth
tests and 94 existing regressions).

The later depth-format slice adds sixteen source-backed owner checks in
`three_depth_formats.test.mjs`. `native_depth_formats.mjs` matches fourteen real
Chromium 153/SwiftShader checkpoints with pinned r186: 14,336 color bytes and
24 depth floats, including depth16 compute reads, both new formats at 1x/4x,
framebuffer copies and stencil retention. GPU/browser errors are zero. Pinned
float-depth/stencil compute sampling creates an invalid all-aspects view, so it
is excluded from the claimed browser parity; candidate depth-only views have
recording-device checks. These results establish correctness, not performance.

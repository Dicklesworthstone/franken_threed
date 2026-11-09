# Source render targets on the new backend

The renderer facade connects `setRenderTarget()` to native offscreen attachments,
feeds `target.texture` into later material bindings, and exposes queue-ordered
`readRenderTargetPixelsAsync()`. The WebGL-compatible surface also executes
`ShaderMaterial`/`RawShaderMaterial` and admitted ShaderLib programs into these
targets, including fullscreen Mesh roots used by postprocessing passes. No
retained renderer draws these passes. Ownership is retained-JavaScript
preparation plus new WebGPU execution; this is not Rust specialization or
measured acceleration.

```js
const target = new THREE.RenderTarget(512, 512, {
  type: THREE.HalfFloatType,
  samples: 4,
});
renderer.setRenderTarget(target);
await renderer.compileAsync(sourceScene, sourceCamera);
renderer.render(sourceScene, sourceCamera);

screenMesh.material.map = target.texture;
renderer.setRenderTarget(null);
await renderer.compileAsync(screenScene, screenCamera);
renderer.render(screenScene, screenCamera);

// WebGPU surface: Uint8Array for RGBA8; raw Uint16 half-float bits for RGBA16F.
const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, 16, 16);
```

The WebGL-compatible surface takes a caller-provided typed array after `height`
and uses bottom-left readback coordinates/row order. Its synchronous readback
remains an exact-backend operation, not a disguised Promise.

Both public asynchronous signatures select MRT color attachments. The native
WebGPU signature also selects any of the six faces of a source `CubeRenderTarget`:

```js
// Native WebGPU: textureIndex defaults to 0; faceIndex defaults to +X (0).
const selected = await renderer.readRenderTargetPixelsAsync(
  target, x, y, width, height, textureIndex, faceIndex,
);

// WebGL-compatible surface: returns the same caller-provided output view.
// Leave the cube-face argument undefined for an ordinary 2D MRT target.
await renderer.readRenderTargetPixelsAsync(
  target, x, y, width, height, output, undefined, textureIndex,
);
```

The selected attachment determines the result's precision: RGBA8 uses
`Uint8Array`, RGBA16F preserves raw half-float bits in `Uint16Array`, and
R/RG/RGBA32F uses one/two/four `Float32Array` components per pixel. R8 returns
one `Uint8Array` value per texel; packed R11G11B10 returns one original
`Uint32Array` word per texel. Native R/RG/RGBA32 integer attachments return
one/two/four `Int32Array` or `Uint32Array` components without a float conversion.
Reads use base mip zero regardless of the active
render mip or cube face. The native cube profile remains single-sample; source
`WebGLCubeRenderTarget` retains its explicit admission limit. Array layers and
3D slices use the same explicit `faceIndex` argument; they do not infer a layer
from mutable renderer state.

Source-shaped reads and the explicit mip/depth readback extensions share the
renderer transfer request limit, target generation and submission FIFO. A later
write can submit while an earlier read waits for mapping, without changing the
earlier copied bytes. Target disposal/reconfiguration and renderer disposal
cancel unfinished reads before publishing caller-buffer contents. These paths
reuse the existing bounded per-device staging budget.

## Implemented profile

Single-color 2D RGBA8 (linear or sRGB) and linear RGBA16F targets; optional private
depth storage; zero/one or four samples with color resolve; physical-pixel target
viewport/scissor; nearest/linear filtering and source wrap/anisotropy settings.
Canvas pixel ratio does not change offscreen attachment dimensions. Ordinary
explicit render targets retain working-linear output; a native target selected
with `setOutputRenderTarget()` instead receives the renderer's final output
transform through the pinned source TSL graph. This output-target route accepts
one admitted 2D color attachment, including its existing precision, depth and
MSAA profiles. It shares the render FIFO, captured sampling leases, transfer
ordering and disposal boundaries with other native target operations. The
renderer caches private linear framebuffers separately for each output target;
automatic private resizing preserves accepted generations. Public target
disposal or replacement invalidates pending operations as before.

Native targets also admit linear single-channel R8 (`RedFormat` with
`UnsignedByteType`) and packed HDR RGB (`RGBFormat` with
`UnsignedInt101111Type`). These are the two source attachments used by SSGI.
Packed HDR rendering requires `rg11b10ufloat-renderable` on the supplied device.
Sampling, mip generation, four-sample resolves, uploads, copies and readback
preserve each format. Storage costs one/four bytes per texel respectively;
render-attachment limits use the separate WebGPU costs and component alignment.
The AO-then-GI MRT therefore costs twelve attachment bytes per sample while its
single-sample storage uses five bytes per pixel. A Chromium 151/SwiftShader
differential matched all 340 texels/packed words across both formats, four mip
levels and single/four-sample rendering, with no GPU errors.

An explicit `DepthTexture` remains allocated when `depthBuffer: false` disables
its use as a render attachment. Copies and later sampling can therefore retain
TRAA's depth history while color rendering leaves it unchanged. Clear calls,
pipeline depth/stencil state and render-pass attachments all follow the enabled
attachment flags; storage budgets include the history depth. Sampling that same
target's depth while rendering its color still encounters the existing feedback
restriction; TRAA uses separate resolve and history targets.

### Integer render targets

The native WebGPU surface supports R/RG/RGBA32 signed and unsigned targets for
object IDs, counters and other integer outputs. Use `IntType` or
`UnsignedIntType` with `RedIntegerFormat`, `RGIntegerFormat` or
`RGBAIntegerFormat`; the corresponding ordinary R/RG/RGBA source-format aliases
also retain the pinned backend's integer storage mapping. These targets require
`NoColorSpace`, unnormalized storage and one sample (`samples: 0` or `1`).

The retained r186 TSL builder emits the destination's integer fragment type.
The pipeline owner validates scalar type and component count before creation, preserves
typed `outputStruct()` members in mixed float/integer MRT, and rejects blending
on integer attachments. Ordinary and scissored clears preserve all 32 bits,
including IDs above float32's exact integer range and negative signed values.

Later native graphs read the same physical image with `TSL.textureLoad()`;
array targets select a layer with `.depth(TSL.int(layer))`, and 3D targets use
`TSL.texture3DLoad()`. Their bindings use `uint`/`sint` without a sampler or an
orientation replica. Authored filter properties remain unchanged. Integer cube
targets support rendering, copies and typed readback; integer cube sampling is
not admitted because WGSL has no cube `textureLoad` overload.

Uploads, raw texture copies, authored mip levels and asynchronous readbacks use
the existing target owner, immutable queue captures and byte budgets. Signed
uploads retain `Int32Array` snapshots, and both result types preserve caller
output tails and cancellation boundaries. Automatic integer mip generation is
not admitted; selected float attachments in mixed MRT can still generate mips.
The WebGL-compatible target path retains its explicit integer-format admission
limit. Source-WGSL comparisons and recording-device tests cover these paths;
they do not establish browser pixel parity, hardware validation or speedup.

### Copies within one target

An ordinary 2D MRT target can copy between distinct, compatible color images
through the existing public source or explicit transfer APIs:

```js
renderer.copyTextureToTexture(target.textures[0], target.textures[1]);
// The explicit extension also selects cropped regions and authored mip levels.
await renderer.copyRenderTargetPixelsAsync(target, target, {
  sourceColorIndex: 0,
  colorIndex: 1,
});
```

The copy runs on the GPU under one native target lock. Its selected images,
regions and mip levels are captured before queueing; later writes, history
swaps and recognized automatic resizing retain the accepted generation. Current
and previous PassNode images can participate independently, including different
physical images at the same color index. Allocation preflight counts shared
history storage once, and source-shaped copies preserve the destination's
captured mip-generation policy.

Copies can also reuse one physical color image when the selected subresources
are disjoint. Different mip levels are independent; at one mip, cube faces and
2D-array layer ranges must not overlap. Different XY rectangles in one layer
remain feedback, and every depth slice of a 3D mip belongs to one subresource.
These checks apply to actual native images, including aliases through history
slots. Formats must be copy-compatible and the destination must be single-sample;
compatible linear/sRGB copies preserve stored bits without color conversion.

The initialized native source API accepts `Box3`/`Vector3` ranges within the same
cube, array or volume texture. Explicit `copyRenderTargetPixelsAsync` accepts
`depthOrArrayLayers` (default one) in addition to `sourceFace`/`face`; the complete
range shares one native command and its byte budget. Nonzero destinations require
authored mip levels with `generateMipmaps: false`.

The MRT transfer tests compare exact copied bytes across six formats and both
coordinate origins. They also exercise independent mip rectangles, queued
history swaps, resize leases, cancellation, selected mip-generation commands
and native error propagation. The 17 subresource-copy tests also construct the
actual pinned renderer and targets, compare copy descriptors with the pinned
WebGPU backend, and check exact bytes across 2D, cube, array and volume copies.
These are production-owner/recording-fixture checks; they do not execute mip
shaders or establish browser pixel parity.

### Shared depth between 2D passes

Distinct 2D color targets can attach the same source `DepthTexture`, including
one/four-sample and MRT targets. The first source target remains its owner;
borrowers use the same native image, preserving depth writes in both directions.
Borrower disposal or resize preserves the original image. Owner disposal,
replacement and direct configuration changes invalidate its borrowers, while
active program leases keep their storage and byte charges until release.
Recognized automatic pass resizing retains matching older color/depth generations
until accepted consumers finish. Native allocation validation also gates every
borrower's completion and readback publication.

Attached depth must match the color extent and sample count. A borrower with
`depthBuffer: false` can sample shared depth while rendering differently sized
color, without attaching or resizing the shared image. Feedback checks follow
the physical depth identity, including owners with PassNode color history.
Writes through any borrower invalidate the WebGL depth sampling replica.

The native renderer preserves r186's first-use clear per destination target:
the first borrower render clears shared depth even with `autoClearDepth: false`;
later renders load it. Actual `PassNode`/`RenderPipeline` dependencies initialize
the owner before its borrower. A standalone borrower used before its owner has
native storage is explicitly rejected rather than eagerly changing source
initialization. Shared cube/array/volume depth remains outside this profile.

`three_render_targets_shared_depth.test.mjs` covers resource identity, history
feedback, disposal, byte budgets, source cloning and queued resize generations.
`native_shared_depth.mjs` compares 18 browser checkpoints with pinned r186:
111,616 RGBA8 bytes match exactly in Chromium 151/SwiftShader, including cold and
warm deferred-style RenderPipeline composition and 1x/4x depth. Both lanes have
zero GPU/browser errors and assert the actual shared GPUTexture identity. This
is software-GPU correctness evidence, not a hardware performance measurement.

Ordinary detached 2D `DepthTexture` inputs also have native, zero-initialized
storage at source version zero. Attached and detached depth share the pinned
format mapping:

| Source format | Source type | Native storage |
| --- | --- | --- |
| `DepthFormat` | `UnsignedShortType` | `depth16unorm` |
| `DepthFormat` | `UnsignedIntType` | `depth24plus` |
| `DepthFormat` | `FloatType` | `depth32float` |
| `DepthStencilFormat` | `UnsignedInt248Type` | `depth24plus-stencil8` |
| `DepthStencilFormat` | `FloatType` | `depth32float-stencil8` |

Float depth/stencil requires the enabled `depth32float-stencil8` device feature.
Depth16 is charged at two bytes per texel; combined depth/stencil at eight.
Existing dimension and single/four-sample rules remain: sampling selects the depth aspect,
while a detached framebuffer snapshot preserves all aspects and samples.
Graphics and compute share this owner, including
captured sampler state, versioned generations, resource limits and cancellation.
No CPU color upload or invented source render target supplies the initial depth.
`native_depth_formats.mjs` matches fourteen pinned-source browser checkpoints:
14,336 RGBA bytes and 24 depth floats are identical in Chromium 153/SwiftShader,
with zero GPU/browser errors. Both new formats cover 1x/4x rendering, framebuffer
snapshots and stencil preservation; depth16 also covers compute reads. Pinned
r186 emits an invalid all-aspects view for float-depth/stencil compute sampling,
so that operation has only candidate layout/ownership checks here, not claimed
source pixel parity.
Recognized automatic target resizing preserves accepted transfer generations
until their validation, GPU completion and readback publication finish.

A paired Chromium 151.0.7922.34 / SwiftShader browser check matched the pinned
renderer exactly for six cube-face readbacks, six directional samples, the
source-supported cube compute result, and output pixels for linear, sRGB, ACES,
linear tone mapping, sRGB target textures, four-sample MSAA and a converted clear.
Both runs reported no WebGPU validation errors. This browser predates r186's
identity-string texture-view swizzle descriptor; both runs used the same adapter
that removes only `swizzle: 'rgba'`. These are checked-pixel correctness results,
not physical-device performance or complete rendering equivalence.

The converted-clear browser case uses a depthless output target. The pinned
source recursively calls `clearDepth()` / `clear()` / `_renderOutput()` when its
first converted clear reaches a fresh depth-bearing default output target; the
observed source result is a stack overflow. This source failure is not evidence
of candidate pixel equivalence for that condition.
A further browser probe of the first canvas-depth clear with automatic clearing
disabled failed in the reference with a lost WebGPU instance during readback on
this software-GPU configuration. Its selector, FIFO and depth-state behavior is
covered by the recording-device tests; this probe is not counted as a pixel pass.

Native `WebGPURenderer` accepts `setRenderTarget(target, 0, mipLevel)` for
allocated single-sample color levels. Authored `texture.mipmaps` descriptors
allocate levels independently of automatic generation; `generateMipmaps: false`
preserves their separately rendered contents. Each level has matching private
depth storage, allocated lazily within the shared budget. Queued draws and clears
retain their selected level. See
[THREE_RENDER_TARGET_MIP_SELECTION.md](./THREE_RENDER_TARGET_MIP_SELECTION.md)
for viewport/scissor behavior, read/copy ordering and the explicit restrictions
on MSAA and sampled depth. The WebGL-compatible facade also renders and clears
complete authored 2D and WebGL cube pyramids under that document's color-only,
single-sample profile. It preserves physical WebGL viewport/scissor dimensions;
applications must set a viewport that fits the selected level.

On the WebGL-compatible surface, each target owns independent shader-program
state. Intermediate passes disable renderer tone mapping and emit linear values;
an sRGB attachment applies its transfer function natively rather than encoding
twice in the shader. Fragment coordinates and point-sprite quad expansion use
target and viewport dimensions. The built-in PointsMaterial size/scale uniforms
retain the renderer's canvas pixel ratio and logical height, as r186 does. Target programs remain available when the canvas uses its
whole-image HDR fallback. Existing shader/compiler, material, shadow, environment
and geometry restrictions still apply; this does not admit arbitrary GLSL or TSL
on the WebGPU public surface.

`render(root, camera)` and `compileAsync(root, camera)` accept source Object3D
roots as well as Scenes. Ordinary roots are traversed and updated in place,
without reparenting. A non-Scene root uses empty scene-level effects while its
object callbacks receive the original root. Callback-free, small fullscreen
shader roots use the per-call capture described below instead of borrowing
mutable shader inputs through asynchronous preparation.

Sampling uses a persistent GPU-only replica of resolved color, refreshed lazily
after writes. Native WebGPU preserves attachment row order; the WebGL-compatible
surface flips rows to match its texture convention. This costs an additional
color image and one fullscreen GPU pass per written-and-sampled version, not a
CPU download or upload. An unchanged version needs no extra pass. Native views stay stable across
content writes; resizing/disposal invalidates them and triggers consumer
re-preparation. Aggregate target storage, including replicas, is bounded.

Two different targets can be alternated for ping-pong rendering. Reading the same
target being drawn is rejected as feedback. Deferred target/canvas calls retain
FIFO order and captured renderer frame settings. `draw -> readback -> draw`
submits the copy before the second draw without waiting for buffer mapping.
Target reconfiguration/disposal invalidates queued uses; renderer disposal also
cancels pending readbacks.

## Native cube color inputs

The native `CubeRenderTarget` profile supports RGBA8 (linear or sRGB), linear
RGBA16F and linear R/RG/RGBA32F color with one sample. `CubeCamera.update()` renders
all six faces through the new backend; its color can then feed
`TSL.cubeTexture()` graphs or ordinary Standard/Physical material environments.
HDR captures retain their floating-point attachment precision. The source PMREM
generator filters rendered cube faces into its HDR atlas before the material's
consumer draw. Float32 sampling retains the existing `float32-filterable`
capability requirement for linear filtering; nearest filtering remains available.

```js
const cubeTarget = new THREE.CubeRenderTarget(256, {
  type: THREE.HalfFloatType,
  generateMipmaps: false,
});
const captureCamera = new THREE.CubeCamera(0.1, 100, cubeTarget);
captureCamera.update(renderer, environmentScene);
scene.environment = cubeTarget.texture;
renderer.render(scene, camera);
```

Selecting or rendering the target registers its exact texture privately. The
pinned constructor's null `texture.renderTarget` pointer remains unchanged.
Native graphics and compute bindings borrow the same six-layer GPU color image;
they neither upload the empty source face descriptors nor create a 2D sampling
replica. Samplers and resource generations are captured at the call boundary,
while contents are observed at each ordered submission. Repeated cube captures
and consumers can therefore share the image without losing intervening draws.
Sampling the active destination, disposal and replaced target textures retain
the existing feedback and stale-generation checks.

Authored cube mip levels can be rendered and sampled independently with automatic
generation disabled. In pinned r186, an uncompressed cube's `texture.mipmaps`
array describes levels **after** the base level, so three entries allocate four
levels. Render-target descriptors allocate storage; they do not upload images.

```js
const cubeTarget = new THREE.CubeRenderTarget(8, {
  type: THREE.HalfFloatType,
  generateMipmaps: false,
  minFilter: THREE.LinearMipmapLinearFilter,
});
cubeTarget.texture.mipmaps = [{}, {}, {}]; // 8x8, 4x4, 2x2 and 1x1 per face.
renderer.setRenderTarget(cubeTarget, 5, 2); // -Z face, 2x2 mip.
renderer.render(scene, camera);
const halfFloatBits = await renderer.readRenderTargetMipPixelsAsync(
  cubeTarget, 2, {face: 5},
);
```

`CubeCamera.activeMipmapLevel` can also select each authored level before its six
renders. A selected nonzero mip acquires a matching private six-face depth image,
charged once to the shared budget and reused on subsequent face draws. Targets
with an attached sampled `CubeDepthTexture` retain the existing base-level render
restriction. Public source-shaped pixel reads still select mip zero; the explicit
mip readback extension above selects higher levels.

Cube resizing preserves the pinned source's distinction between logical and
physical dimensions. `RenderTarget.setSize()` updates the target and viewport,
but leaves all six face descriptors unchanged. Native allocation and mip counts
continue to follow those descriptors, as pinned `Textures.getSize()` does; a
smaller viewport renders into part of the face and a larger one clips to the
attachment. Construct a new cube target when a different physical resolution is
required. Dynamic point-shadow `mapSize` updates follow the same source behavior.
Their recognized source resize preserves earlier queued six-face generations and
charges each until its final consumer completes. Public `setSize()` and
`dispose()` still invalidate pending uses of their replaced residency.

### Automatic cube mip generation

Native cube rendering now applies the pinned cube filtering shader to every face
of each successive mip level, retaining its face orientation, implicit
derivatives and sampler behavior. RGBA8/sRGB and RGBA16F are supported; Float32
cube filtering requires the enabled `float32-filterable` device feature and keeps
its original attachment precision. The generation flag is captured with each
accepted source render, so queued CubeCamera updates generate only after the
sixth face. Manual clears and subsequent sampled borrows do not infer another
generation from a later public flag change.

Initialize the target while generation is enabled when a full pyramid is needed:

```js
await renderer.init();
const cubeTarget = new THREE.CubeRenderTarget(256, {
  type: THREE.HalfFloatType,
  generateMipmaps: true,
  minFilter: THREE.LinearMipmapLinearFilter,
});
renderer.initRenderTarget(cubeTarget);
new THREE.CubeCamera(0.1, 100, cubeTarget).update(renderer, environmentScene);
```

This explicit initialization matters in pinned r186: a cold CubeCamera disables
generation before the first face initializes storage, so that source path has
only one level. `initRenderTarget()` first allocates the complete pyramid. The
native implementation preserves both behaviors, including the unchanged
`CubeRenderTarget.fromEquirectangularTexture()` helper and its temporary resource
disposal. Selection and layout queries alone do not initialize the allocation.
Once initialized, changing the generation flag or authored mip descriptors does
not replace the image until its source texture version changes. Native cube
samplers retain explicit LOD access with ordinary `LinearFilter`/`NearestFilter`,
using nearest mip selection as the pinned backend does.

`three_renderer_cube_mipmaps.test.mjs` covers source initialization, metadata-only
selection, ordered captures, mutable flags, clears, budgets, disposal and authored
levels. Chromium 151/SwiftShader differentials matched 11,232 stored face/mip
channel values and 2,640 sampled values exactly across six cold/initialized
CubeCamera and equirectangular cases, including HDR, sRGB and Float32. The 660
samples cover axes, corners and near-seam directions over five LODs. Both paths
used identical identity-swizzle and declared cube-view metadata adapters for
that browser; source readback row padding was unpacked. No numeric tolerance or
GPU validation errors occurred.

Plain source `TSL.cubeTexture()` color nodes require a scene/material for their
environment rotation and cannot run in the pinned source's context-free compute
updates. Compute applications can use the source's own `PlainCubeTextureNode`
extension pattern from `examples/jsm/lighting/vxgi/VXGIVolume.js`: extend
`CubeTextureNode` and define `setupUV()` without material environment rotation.
That application-authored node retains the correct cube binding type and can be
passed to a `wgslFn` cube texture argument unchanged on both renderers. Passing a
cube image through a 2D `TSL.texture()` node is not a valid substitute.

`node tools/ingest/three_node_cube_targets.test.mjs` checks source CubeCamera
execution, exact source-generated WGSL, shared graphics/compute color bindings
with that explicit source node extension, the plain color node's compute-context
limitation, PMREM composition, queue order, budgets and invalidation with a
recording device.

`node tools/ingest/three_render_target_cube_hdr_mips.test.mjs` checks source HDR
formats and authored mip counts, all face/level attachments, private depth budgets,
native sampled views, typed readbacks, generation invalidation and CubeCamera
LOD rendering. Its device fixtures record submissions and copy independent byte
arrays; they do not execute shaders or establish browser pixel parity.

Chromium 151/SwiftShader checks, with the same identity-swizzle adapter on both
renderers, matched r186 exactly for depthless RGBA16F/RGBA32F cubes: 12 base-face
reads and 36 face/LOD samples across three levels, preserving negative values
(-0.75 to -0.125) and HDR values (2.5 to 3.75), with zero GPU validation errors.
Candidate-only private-depth checks also passed 36 direct mip reads plus rendering
and sampling with zero errors. Those depth checks are not source parity evidence:
pinned r186 attaches a six-layer depth view at nonzero cube mips, causing GPU
validation errors.

## Reused fullscreen shader passes

On the WebGL-compatible surface, callback-free `Mesh` roots with a source
`ShaderMaterial` or `RawShaderMaterial`, no children/instancing/deformation, and
at most six vertices capture their call-time inputs. This includes the ordinary
fullscreen triangle shape used by `FullScreenQuad`. Reusing the same mesh or
changing its material/uniforms in the same turn no longer replaces earlier
queued pass values. `renderAsync()` enqueues before its first await, so mixed
synchronous draw, async draw and readback calls retain their submission order
even during initialization.

Captures copy uniform scalars, arrays, typed-array slices, plain nested structs
and source math values; shader text/defines and material raster fields; camera
and world matrices; geometry upload bytes and draw ranges. They install into
stable private execution views at serialized preparation/submission boundaries,
never by temporarily rewriting application-owned objects across an await.
`compileAsync(root, camera)` prepares that same execution view; uniform-only
steady-state draws do not need a new preparation or material identity. A queued
pass is prepared by itself, not together with future passes whose inputs have
not yet been installed.

Texture objects remain borrowed GPU-resource identities, not pixel snapshots.
Disposal, uploaded source-version changes, sampler/transform edits or a changed
sampled-target generation fail as `THREE_PASS_STALE` before that pass submits.
Normal GPU writes to a ping-pong target do not change its identity and remain
queue ordered. Shared-memory inputs, accessors, custom callable uniform values
and unsupported mutable classes fail explicitly rather than executing hidden
copy hooks. This capture path uses additional CPU copying and memory; it is not
a zero-copy or performance claim. The default capture limits are 2 MiB per call,
32 MiB total pending plus installed state, and 64 retained pass roots. Captures
are released on queue failure/disposal; the last installed state is charged until
replaced or the renderer is disposed.

General scenes, large meshes and callback-bearing roots keep the existing
live-source path. This is **not** a complete per-call scene-state snapshot. Shader
compiler and renderer admission restrictions also remain in force.

## WebGL MRT generated mipmaps

WebGL-authored 2D and array MRT targets can now generate independent mip
pyramids for each color attachment. Mixed formats and base-only siblings keep
their own allocation sizes; every array mip retains all layers. Rendering
captures each attachment's generation flag, resolves any four-sample color,
then generates only the requested chains. Manual clears preserve the existing
resolved and generated images. Sampling borrows the existing GPU orientation
replicas with the full selected color pyramid, original precision and setup
sampler; warm sampling reuses those images.

For this profile, configure generation before an attachment is first initialized. An initialized
chain can be disabled and re-enabled without replacing its rendered contents,
including across `needsUpdate` and sampler edits. Enabling generation on an
attachment initialized with only a base level currently requires disposing and
recreating the target in F3D; the pinned WebGL source can grow that chain in place.
Authored WebGL MRT framebuffer arrays remain unsupported.
The existing shared budget charges every color level, multisample base and
sampling replica, and queued leases retain old generations through retirement.

The focused MRT, facade and mip-owner suites pass 101 Node tests, including nine
new cases using pinned r186 source classes and the existing recording devices.
They verify per-color/layer descriptors, generation and resolve order, captured
flags, immutable setup samplers, exact byte charges, budget failures and
disposal. They do not execute the mip shaders or establish fresh rasterized
parity or hardware performance; browser execution was unavailable in this host.

## Native array MRT mipmaps

Native 2D-array targets now support multiple color attachments with independent
formats, mip counts and generation choices. Every color level retains the full
array layer count. Rendering selects one layer and a mip present in every
attachment; sampling, uploads and copies can select a longer chain on one color
independently. Invalid layer or common-mip selections fail before allocation.

Generated pyramids reduce each layer separately without mixing attachments.
RGBA8, half-float and float32 colors keep their original precision; float32
reduction uses the existing unfilterable texture-load path. Mixed integer and
floating-point targets retain authored integer levels while generating only
eligible floating-point colors, including after resize.

Four-sample targets resolve each attachment's selected layer into level zero
before generating mips. Nonzero render mips still require single-sample color
and private depth. PassNode history colors each retain their own full layered
pyramid, generation state and MSAA images. Queued generation leases, disposal
and resize retirement continue to charge every live image to the shared budget.

The focused array-MRT owner tests check descriptors, command order, sampling
leases, exact allocation charges, independent history images and retirement.
They use the existing recording-device fixtures and do not execute mip shaders
or establish pinned-source pixel equivalence. Cube MRT retains its existing
admission limit; authored volume MRT levels follow the profile below.

## Native volume targets and layered transfers

`RenderTarget3D` and registered `WebGL3DRenderTarget` instances now own actual
WebGPU 3D color textures. `setRenderTarget(volume, slice, mipLevel)` selects the
color attachment's `depthSlice`; clear, scissored-clear, geometry and background
passes all preserve it, including slice zero. Multiple color attachments share
the selected slice. Source `WebGL3DRenderTarget.texture.renderTarget` remains
null: the existing target owner records its association privately.

```js
const volume = new THREE.RenderTarget3D(32, 32, 16, {depthBuffer: false});
renderer.setRenderTarget(volume, 7);
renderer.render(sourceScene, sourceCamera);

const material = new THREE.MeshBasicNodeMaterial();
material.colorNode = THREE.TSL.texture3D(volume.texture,
  THREE.TSL.vec3(0.25, 0.5, 0.75)).rgb;
```

Graphics and compute borrow this same native image. Sampling retains the source
3D node shader, filters, wrapR and authored LOD. An ordinary depth attachment is
independently 2D; `useArrayDepthTexture` or an attached array depth texture selects
per-slice 2D depth views. Color depth shrinks at each authored mip, while depth
array layer counts follow the source's separate depth allocation. XYZ limits,
volumetric byte budgets and mip-local slice bounds are checked before native
work. Resizing depth replaces storage generations and retains submitted images
until their queue work completes.

Explicit target upload, copy and readback operations use `face`/`sourceFace` as
the selected volume slice and validate it against the requested mip. Typed
readbacks retain their captured bytes across later writes. The existing
`copyTextureToTexture` bridge also supports multi-slice copies among admitted
3D textures and 2D arrays, including target color to compute storage and the
reverse direction. Box3/Vector3 coordinates address mip-local voxels or layers;
formats remain identical and no sampler or conversion shader is introduced.
An explicit one-slice box can copy between a volume and an ordinary 2D image.

Public `copyFramebufferToTexture` retains the pinned r186 behavior: it copies
base mip zero, slice zero, even when a different render slice/mip is active.
The facade regression compares its encoded command to the pinned renderer and
backend. Use an explicit texture copy when a different source slice is needed.

Native volume MRT also retains independent authored mip chains and formats for
every color. Rendering requires a mip present in all attachments and selects a
slice within that mip's reduced Z extent. Sampling, typed uploads, copies and
readback select one color's allocated levels independently, including levels
beyond a sibling's shorter chain. Graphics and compute borrow the original 3D
images; no sampling replica or mip-generation pass is introduced. Private
nonzero-mip depth retains its independent 2D or array allocation and byte charge.
Accepted sampling leases and submitted work retain every old color chain across
resize until their consumers and queue fences finish.

Volume color is single-sample. Automatic 3D mip generation, volume history
rotation and storage-to-target writes above mip zero remain explicit boundaries.
Automatic generation is rejected for every volume attachment, including a
secondary color with only one allocated level. Copies between distinct MRT
images or distinct mips remain valid when their formats and regions match;
different Z slices of the same image and mip still share one subresource and
cannot be copied onto themselves. These additions do not close the full
render-target compatibility bead.

The focused volume owner, source target, facade, storage-copy and framebuffer
tests record native commands and independently compare transferred bytes.
They do not execute GPU shaders or establish browser pixel parity or speedups.
The added volume-MRT tests exercise the production GPU/source owners and compute
sampling bridge with recording fixtures, including secondary-color readback,
mip-local slice ranges, typed transfers, allocation limits and resize retirement.

## Remaining gaps and verification

Public MRT, native cube, array and 3D readback do not close all target rendering
and sampling contracts. Source `WebGLCubeRenderTarget` authored mip rendering
retains the bounded WebGL profile above. Nonzero mip rendering retains the other
restrictions above. Shader programs retain their
explicit unsupported-feature errors. Mesh-root, offscreen shader and small-pass
capture support remove postprocessing blockers but do not establish complete
EffectComposer compatibility, effect parity or arbitrary scene snapshots.

`node --test tools/ingest/three_pass_snapshot.test.mjs tools/ingest/three_renderer_pass_snapshot.test.mjs`
checks per-call data capture and the actual facade/target/readback orchestration
with source-data, scene/compiler and GPU boundary doubles. These tests do not
execute WGSL or certify the pinned Three.js implementation or rasterized pixels.

`node --test tools/ingest/three_render_targets.test.mjs tools/ingest/three_renderer_targets.test.mjs`
checks residency and public renderer ordering using recording devices and
scene/canvas factory doubles. `node tools/ingest/three_renderer_public_readback.test.mjs`
checks the public native/WebGL signatures with pinned Three.js target classes,
six cube faces, mixed MRT precision, queue ordering, request limits and target
cancellation. Native copies use a byte-array fixture, not a GPU rasterizer. The additional
`three_renderer_program_targets.test.mjs`, `three_scene_program_targets.test.mjs`
and `three_scene_roots.test.mjs` exercise destination-specific program state,
the actual scene bridge, borrowed shader textures, uniforms, root identity,
callbacks and geometry cleanup with the compiler/GPU boundaries replaced.
`tools/ingest/three_renderer.test.mjs` additionally exercises the actual bridge
and pinned source objects when the upstream checkout is available. GPU pixels,
full upstream conformance and hardware performance require separate validation;
the recording tests do not establish them.

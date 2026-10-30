# Explicit render-target mip selection

The renderer facades render and clear into allocated color mips selected by
`setRenderTarget(target, face, mipLevel)`, with separate native and WebGL source
profiles below. This removes a blocker
for authored texture pyramids and multipass effects that render each resolution
separately. The implementation uses the existing JavaScript preparation, target
ownership and new WebGPU submission path; it adds no retained renderer, Rust
specialization or measured acceleration.

## Native source surface

```js
const target = new THREE.RenderTarget(128, 64, {
  generateMipmaps: false,
  minFilter: THREE.LinearMipmapLinearFilter,
});
// As in the pinned native renderer, the array length allocates the pyramid.
// These are render destinations; the entries do not upload CPU image data.
target.texture.mipmaps = Array.from({ length: 4 }, () => ({}));

renderer.setRenderTarget(target, 0, 2);
renderer.render(scene, camera); // 32 x 16 color, with matching private depth.
renderer.setRenderTarget(null);
```

The native 2D profile is single-color targets with zero/one
source samples and optional private depth or depth/stencil storage. It keeps the
existing attachment-format and draw-pipeline admission rules. Nonzero mip levels
reject multisampled targets, explicitly attached source `DepthTexture` objects
and the canvas. Native cube, MRT and array profiles are described separately in
[THREE_RENDER_TARGETS.md](./THREE_RENDER_TARGETS.md). A rejected selector preserves
the previous binding.

Mip count is `texture.mipmaps.length` when nonzero; otherwise it is the full
pyramid for `generateMipmaps: true`, or one level. Counts must fit the target
extent. Allocation and automatic generation are independent: disabling
generation preserves separately authored levels. With generation enabled, a
later sampling/read/copy boundary can regenerate lower levels from level zero.
Use `generateMipmaps: false` when authored lower levels must survive.

The pinned native r186 renderer scales viewport and scissor width/height by the
selected mip while retaining their x/y origins. The facade follows that rule,
uses the selected attachment dimensions for rendering and ArrayCamera clipping,
and captures the selector and frame settings for each queued call. It does not
rewrite source viewport/scissor vectors or temporarily mutate source target
dimensions. General scene data retains the existing live-source contract.

## WebGL-authored mip destinations

The WebGL-compatible facade accepts ordinary 2D targets and
`WebGLCubeRenderTarget` with a complete authored pyramid, one sample, no depth or
stencil attachment, and `generateMipmaps: false` when a draw or clear is accepted.
Its viewport must fit inside the selected level in physical pixels. These
restrictions preserve the pinned WebGL scratch framebuffer's depthless behavior
without inventing native attachment or viewport semantics.

```js
const target = new THREE.WebGLRenderTarget(64, 32, {
  depthBuffer: false,
  generateMipmaps: false,
  minFilter: THREE.LinearMipmapLinearFilter,
});
target.texture.mipmaps = Array.from({ length: 7 }, () => ({}));
target.viewport.set(0, 0, 16, 8); // Physical dimensions of level two.
renderer.setRenderTarget(target, 0, 2);
renderer.render(scene, camera);
renderer.setRenderTarget(null);
```

WebGL keeps viewport and scissor extents in physical pixels; it does not divide
them again by the selected level. The facade converts their bottom-left origins
using that level's height and clips scissors to the attachment. Queued calls
retain their selected face, level and frame settings without changing the source
vectors. Higher-level writes preserve the other authored levels and perform no
automatic regeneration. Sampling refreshes the existing GPU-only source-oriented
replica across the complete pyramid, while explicit mip readbacks select the
requested level through the existing transfer queue.

Authored WebGL cube descriptors include the base level: an 8-by-8 cube needs four
entries. Native `CubeRenderTarget` descriptors omit that base entry. Source
`CubeCamera` updates retain each queued face and level; later generation-flag
changes do not replace the accepted cube storage. Partial pyramids, attached
depth textures, MSAA, array/volume targets and oversized higher-mip viewports
remain outside this WebGL profile.

## Ownership, readback and copies

Each selected mip uses a one-level color view. Native private depth storage is allocated
lazily at that mip's extent, reused on subsequent draws, charged to the shared
target budget and retired with the target generation. Budget refusal occurs
before the rendering callback. Resize, disposal and mip-count changes invalidate
queued uses of the old generation. Existing target feedback refusal also applies
when reading and writing different mips of the same target.

Material sampling uses the surface's source orientation. Native WebGPU preserves
attachment row order, matching r186 `WGSLNodeBuilder.isFlipY() === false` and
`TextureNode.setupUV()`. WebGL programs retain their vertically flipped replica.
The orientation is fixed per renderer owner; both paths keep the existing lazy,
budgeted sampling image and lifetime checks. Raw copies and readback do not
change their existing orientation conventions. This also corrects native
level-zero target-texture sampling.

Ordinary `readRenderTargetPixelsAsync` retains the source API's level-zero
behavior, regardless of the active selector. The facade extension
`readRenderTargetMipPixelsAsync(target, level, options)` reads an explicit level;
options include `x`, `y`, `width`, `height` and a typed-array `output`. Default
bounds cover that level. Drawing, clearing, copying and readback stay in the
renderer's common FIFO.

Raw target copies independently select `sourceMipLevel` and
`destinationMipLevel`. The destination must have that level allocated, use one
sample and disable automatic mip generation. Bounds and bottom-left coordinate
conversion use each endpoint's selected mip dimensions; volume slices shrink
with the mip while array layers and cube faces retain their indices. Copies
preserve the other authored levels, invalidate any existing sampling replica and
use the original GPU allocation without CPU readback or temporary color images.

```js
await renderer.copyRenderTargetPixelsAsync(source, destination, {
  sourceMipLevel: 1,
  destinationMipLevel: 2,
  sourceX: 1, sourceY: 1, x: 2, y: 0, width: 4, height: 2,
});
const pixels = await renderer.readRenderTargetMipPixelsAsync(destination, 2);
```

The WebGL source-shaped `copyTextureToTexture` and its asynchronous extension
also accept their `dstLevel` argument for authored 2D color targets. A level-zero
source supports cropped regions and offset destinations. When both source and
destination levels are nonzero, this path currently requires zero source and
destination origins: pinned r186 passes extents as `blitFramebuffer` endpoints,
so offset rectangles can scale or reverse rather than perform a raw copy. Such
offsets remain an explicit admission gap; the raw target-copy extension above
does not promise those blit semantics. WebGL's null source region floors the
base dimensions by the source level, retaining zero-area one-dimensional tails.
Depth copies still require level zero at both endpoints.

For native source-compatible mip copies, supply a `Box2` region in the selected
level's coordinates. Pinned native r186 defaults a null copy region to the base
image dimensions even when `sourceLevel` is nonzero; an explicit mip-sized region
avoids an out-of-bounds reference copy. Queued operations capture levels and
rectangle scalars when accepted, so later edits do not redirect their writes.

The native owner also exposes
`createGpuRenderTarget(device, { ..., mipLevelCount, generateMipmaps: false })`
and `target.withFrame(callback, { mipLevel })`. These bounded borrows retain the
existing synchronous submission and lifetime rules.

## Source basis and verification

Source semantics were checked against pinned Three.js r186 commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`: native renderer target selectors and
viewport/scissor handling, `Textures` mip allocation/private depth behavior, and
WebGPUBackend single-level color attachment views. The WebGL profile follows
`WebGLRenderer.setRenderTarget`, `WebGLTextures.setupRenderTarget` and
`CubeCamera.update` for its distinct viewport, attachment and descriptor rules.

The authored WebGL extension passed five focused facade/selector cases, all eight
actual-r186 WebGL cube cases and all 24 existing target-mip owner cases. Coverage
includes queued levels and readbacks, NPOT clears, physical viewport/scissor
capture, source-oriented sampling replicas, all cube faces and generation-flag
snapshots. These 37 checks use recording GPU boundaries; no WebGL mip pixel or
performance result is claimed. Unrelated stale fixtures in the full renderer
target suite were also observed on the unchanged baseline.

The destination-mip copy extension is covered by eight target-owner cases and
six actual-r186 facade cases in `gpu_target_transfer.test.mjs` and
`three_renderer_public_readback.test.mjs`. They compare independent byte arrays
for RGBA8, half-float bit patterns and float32 exceptional values; preserve
untouched levels; exercise NPOT bounds, both coordinate origins, queued option
snapshots, zero-area tails and invalid generation/level requests. These recording
device checks establish transfer and scheduling behavior, not rasterized pixel
parity or hardware performance.

The focused tests run directly with:

```sh
node tools/ingest/gpu_render_target_mipmaps.test.mjs
node tools/ingest/three_render_target_mipmaps.test.mjs
node tools/ingest/three_renderer_targets.test.mjs
```

These native-profile cases cover authored versus generated pyramids, selector
admission, matching color/depth extents, lazy storage budgets, FIFO capture,
viewport/scissor immutability, clear/read/copy interactions, callback failures,
resize, retirement and per-surface sampling orientation. They use recording GPU
boundaries; they do not establish
rasterized pixel parity or hardware performance.

`tests/e2e/webgpu_renderer/scenarios_render_mips.js` supplies source-compatible
browser scenarios for actual rendering, cross-mip copies, texture sampling and
updates. Both `native_render_mips` and `native_render_mips_updates` passed the
existing pixel-comparison thresholds against the pinned renderer, with no
reported mismatches or mean pixel difference. Browser validation is separate
from the host tests.

A Chromium 151 WebGPU run using SwiftShader additionally verified an authored
8x8 pyramid with red, green, blue and yellow levels. Explicit readbacks returned
the exact expected RGBA bytes and lengths (256, 64, 16 and 4). Reads queued before
a later white write to level one retained the earlier green values; a subsequent
read observed white. The ordinary source readback still returned red level zero,
the selector reset to zero, and the source viewport remained unchanged. No GPU
validation errors occurred. This is software GPU execution evidence, not a
hardware performance measurement or complete upstream conformance claim.
